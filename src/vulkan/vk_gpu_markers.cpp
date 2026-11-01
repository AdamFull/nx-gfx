#include "src/vulkan/vk_gpu_markers.h"

#include "core/foundation/platform/os.h"

#include <algorithm>
#include <cstring>

namespace nx::gfx::detail {
namespace {

constexpr u32 NOT_TIMED = ~u32{0};
constexpr u64 RECALIBRATE_NS = 1'000'000'000;

} // namespace

void VulkanGpuMarkers::init(const VkDevice device, const u32 frames,
                            const f32 period_ns, const u32 valid_bits,
                            const u32 copy_valid_bits,
                            const PFN_vkGetCalibratedTimestampsEXT calibrated,
                            const bool inside_passes) {
  m_device = device;
  m_period_ns = period_ns;
  m_valid_bits = nx::clamp<u32>(valid_bits, 1, 64);
  m_copy_bits = nx::min<u32>(copy_valid_bits, 64);
  m_calibrated = calibrated;
  m_inside_passes = inside_passes;
  m_slots.reserve(frames);
  for (u32 i = 0; i < frames; ++i)
    m_slots.push_back(nx::make_unique<Slot>());
}

void VulkanGpuMarkers::shutdown() noexcept {
  for (const nx::unique_ptr<Slot> &slot : m_slots)
    if (slot->pool != VK_NULL_HANDLE)
      vkDestroyQueryPool(m_device, slot->pool, nullptr);
  m_slots.clear();
  m_open.store(nullptr, std::memory_order_release);
  m_ready.clear();
  m_ready_first = 0;
  m_ready_count = 0;
  if (m_copy_pool != VK_NULL_HANDLE)
    vkDestroyQueryPool(m_device, m_copy_pool, nullptr);
  m_copy_pool = VK_NULL_HANDLE;
  m_copies_count = 0;
}

// The clocks are read a moment apart, so the reading kept is the tightest of a
// few. It is one device clock, so this needs no host time domain to match.
void VulkanGpuMarkers::calibrate() noexcept {
  if (m_calibrated == nullptr)
    return;
  VkCalibratedTimestampInfoEXT info{
      VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT};
  info.timeDomain = VK_TIME_DOMAIN_DEVICE_EXT;
  u64 tightest = ~u64{0};
  for (int attempt = 0; attempt < 4; ++attempt) {
    u64 device_ticks = 0;
    u64 deviation = 0;
    const u64 before = nx::os::ticks_ns();
    if (m_calibrated(m_device, 1, &info, &device_ticks, &deviation) !=
        VK_SUCCESS)
      return;
    const u64 after = nx::os::ticks_ns();
    if (after - before >= tightest)
      continue;
    tightest = after - before;
    m_anchor_ticks = device_ticks;
    m_anchor_ns = before + (after - before) / 2;
    m_have_anchor = true;
    m_calibrated_at_ns = after;
  }
}

VulkanGpuMarkers::List *
VulkanGpuMarkers::open_frame(const u32 frame, const VkCommandBuffer cmd) {
  if (frame >= m_slots.size())
    return nullptr;
  Slot &slot = *m_slots[frame];
  const u32 used = nx::min(slot.next_query.load(std::memory_order_relaxed),
                           MOST_A_FRAME * 2);
  if (used != 0) {
    if (slot.submitted) {
      collect(slot);
      slot.to_reset = used;
    } else {
      // Nothing it recorded ran, its reset included.
      slot.ever_reset = false;
    }
  }
  for (u32 i = 0; i < slot.lists_used.load(std::memory_order_relaxed); ++i) {
    slot.lists[i].records.clear();
    slot.lists[i].depth = 0;
  }
  slot.lists_used.store(0, std::memory_order_relaxed);
  slot.next_query.store(0, std::memory_order_relaxed);
  slot.submitted = false;
  collect_copies();

  if (!m_wanted.load(std::memory_order_acquire))
    return nullptr;
  if (slot.pool == VK_NULL_HANDLE) {
    VkQueryPoolCreateInfo create{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    create.queryType = VK_QUERY_TYPE_TIMESTAMP;
    create.queryCount = MOST_A_FRAME * 2;
    if (vkCreateQueryPool(m_device, &create, nullptr, &slot.pool) != VK_SUCCESS)
      return nullptr;
    slot.lists.resize(MOST_LISTS);
    for (List &list : slot.lists)
      list.slot = &slot;
    if (m_ready.empty())
      m_ready.resize(READY_MARKERS);
  }
  if (!slot.ever_reset) {
    vkCmdResetQueryPool(cmd, slot.pool, 0, MOST_A_FRAME * 2);
    slot.ever_reset = true;
  } else if (slot.to_reset != 0) {
    vkCmdResetQueryPool(cmd, slot.pool, 0, slot.to_reset);
  }
  slot.to_reset = 0;
  if (m_calibrated != nullptr &&
      (!m_have_anchor ||
       nx::os::ticks_ns() - m_calibrated_at_ns > RECALIBRATE_NS))
    calibrate();
  m_open.store(&slot, std::memory_order_release);
  return add_list(0);
}

void VulkanGpuMarkers::frame_submitted(const u32 frame) noexcept {
  m_open.store(nullptr, std::memory_order_release);
  if (frame >= m_slots.size())
    return;
  m_slots[frame]->submitted = true;
  m_slots[frame]->submitted_ns = nx::os::ticks_ns();
}

VulkanGpuMarkers::List *
VulkanGpuMarkers::add_list(const u32 position) noexcept {
  Slot *const slot = m_open.load(std::memory_order_acquire);
  if (slot == nullptr)
    return nullptr;
  const u32 index = slot->lists_used.fetch_add(1, std::memory_order_acq_rel);
  if (index >= MOST_LISTS) {
    slot->lists_used.store(MOST_LISTS, std::memory_order_release);
    m_groups_dropped.fetch_add(1, std::memory_order_relaxed);
    return nullptr;
  }
  List &list = slot->lists[index];
  list.position = position;
  return &list;
}

void VulkanGpuMarkers::push(List &list, const VkCommandBuffer cmd,
                            const nx::string_view name,
                            const bool in_render_pass) {
  const u32 depth = list.depth++;
  if (depth >= MAX_DEPTH) {
    m_groups_dropped.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  list.open[depth] = NOT_TIMED;
  if (in_render_pass && !m_inside_passes)
    return;
  Slot &slot = *list.slot;
  const u32 query = slot.next_query.fetch_add(2, std::memory_order_relaxed);
  if (query >= MOST_A_FRAME * 2) {
    m_groups_dropped.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  List::Record record;
  record.query = query;
  record.depth = nx::cast<u16>(depth);
  record.name_size = nx::cast<u8>(nx::min(name.size(), GPU_MARKER_NAME_BYTES));
  std::memcpy(record.name, name.data(), record.name_size);
  list.open[depth] = nx::cast<u32>(list.records.size());
  list.records.push_back(record);
  // Not the top of the pipe: a begin there may pass the group before it, and a
  // lane needs groups that do not overlap.
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, slot.pool,
                      query);
}

void VulkanGpuMarkers::pop(List &list, const VkCommandBuffer cmd) noexcept {
  if (list.depth == 0)
    return;
  const u32 depth = --list.depth;
  if (depth >= MAX_DEPTH || list.open[depth] == NOT_TIMED)
    return;
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                      list.slot->pool,
                      list.records[list.open[depth]].query + 1);
}

void VulkanGpuMarkers::end_open(List &list,
                                const VkCommandBuffer cmd) noexcept {
  while (list.depth > 0)
    pop(list, cmd);
}

u32 VulkanGpuMarkers::copy_begin(const VkCommandBuffer cmd,
                                 const nx::string_view name) noexcept {
  if (cmd == VK_NULL_HANDLE || m_copy_bits == 0 ||
      !m_wanted.load(std::memory_order_acquire))
    return NO_COPY;
  const nx::scoped_lock<nx::mutex> held(m_copy_lock);
  if (m_copy_pool == VK_NULL_HANDLE) {
    VkQueryPoolCreateInfo create{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    create.queryType = VK_QUERY_TYPE_TIMESTAMP;
    create.queryCount = COPIES_WAITING * 2;
    if (vkCreateQueryPool(m_device, &create, nullptr, &m_copy_pool) !=
        VK_SUCCESS)
      return NO_COPY;
    // A copy queue may not be able to reset a query, so this resets them here.
    vkResetQueryPool(m_device, m_copy_pool, 0, COPIES_WAITING * 2);
    m_copies.resize(COPIES_WAITING);
  }
  if (m_copies_count == COPIES_WAITING) {
    m_groups_dropped.fetch_add(1, std::memory_order_relaxed);
    return NO_COPY;
  }
  const u32 copy = (m_copies_first + m_copies_count) % COPIES_WAITING;
  ++m_copies_count;
  Copy &entry = m_copies[copy];
  entry.name_size = nx::cast<u8>(nx::min(name.size(), GPU_MARKER_NAME_BYTES));
  std::memcpy(entry.name, name.data(), entry.name_size);
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_copy_pool,
                      copy * 2);
  return copy;
}

void VulkanGpuMarkers::copy_end(const VkCommandBuffer cmd,
                                const u32 copy) noexcept {
  if (copy != NO_COPY)
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_copy_pool,
                        copy * 2 + 1);
}

void VulkanGpuMarkers::collect_copies() noexcept {
  const nx::scoped_lock<nx::mutex> held(m_copy_lock);
  while (m_copies_count != 0 && m_have_anchor && !m_ready.empty()) {
    struct Result {
      u64 ticks;
      u64 available;
    } results[2]{};
    const VkResult result = vkGetQueryPoolResults(
        m_device, m_copy_pool, m_copies_first * 2, 2, sizeof(results), results,
        sizeof(Result),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if ((result != VK_SUCCESS && result != VK_NOT_READY) ||
        results[0].available == 0 || results[1].available == 0)
      return;
    const Copy &entry = m_copies[m_copies_first];
    gpu_marker &marker = next_ready();
    marker = {};
    marker.queue = gpu_marker_queue::Copy;
    marker.name_size = entry.name_size;
    std::memcpy(marker.name, entry.name, entry.name_size);
    place(marker, results[0].ticks, results[1].ticks, m_copy_bits);
    vkResetQueryPool(m_device, m_copy_pool, m_copies_first * 2, 2);
    m_copies_first = (m_copies_first + 1) % COPIES_WAITING;
    --m_copies_count;
  }
}

gpu_marker &VulkanGpuMarkers::next_ready() noexcept {
  if (m_ready_count == READY_MARKERS) {
    m_ready_first = (m_ready_first + 1) % READY_MARKERS;
    --m_ready_count;
    ++m_markers_dropped;
  }
  gpu_marker &marker = m_ready[(m_ready_first + m_ready_count) % READY_MARKERS];
  ++m_ready_count;
  return marker;
}

void VulkanGpuMarkers::place(gpu_marker &marker, const u64 begin_ticks,
                             const u64 end_ticks,
                             const u32 valid_bits) const noexcept {
  // A timestamp's high bits are not valid, and it may have wrapped.
  const u32 shift = 64 - valid_bits;
  const auto host_ns = [&](const u64 ticks) {
    const i64 since =
        static_cast<i64>((ticks - m_anchor_ticks) << shift) >> shift;
    return m_anchor_ns + static_cast<u64>(static_cast<i64>(
                             static_cast<f64>(since) * m_period_ns));
  };
  marker.begin_ns = host_ns(begin_ticks);
  marker.end_ns = nx::max(host_ns(end_ticks), marker.begin_ns);
}

void VulkanGpuMarkers::collect(Slot &slot) {
  const u32 used = nx::min(slot.next_query.load(std::memory_order_relaxed),
                           MOST_A_FRAME * 2);
  m_ticks.resize(used);
  // No wait: its frame has finished, and a lost device reports that instead. A
  // recording that never ended leaves queries unwritten, and the frame goes.
  const VkResult result = vkGetQueryPoolResults(
      m_device, slot.pool, 0, used, m_ticks.size() * sizeof(u64),
      m_ticks.data(), sizeof(u64), VK_QUERY_RESULT_64_BIT);
  if (result != VK_SUCCESS)
    return;
  ++m_frames_read;
  const u32 number = ++m_frame_number;

  nx::small_vector<const List *, MOST_LISTS> lists;
  const u32 count =
      nx::min(slot.lists_used.load(std::memory_order_relaxed), MOST_LISTS);
  for (u32 i = 0; i < count; ++i)
    if (!slot.lists[i].records.empty())
      lists.push_back(&slot.lists[i]);
  std::sort(lists.begin(), lists.end(), [](const List *a, const List *b) {
    return a->position < b->position;
  });

  if (m_calibrated == nullptr && !lists.empty()) {
    m_anchor_ticks = m_ticks[lists.front()->records.front().query];
    m_anchor_ns = slot.submitted_ns;
    m_have_anchor = true;
  }
  for (const List *const list : lists)
    for (const List::Record &record : list->records) {
      gpu_marker &marker = next_ready();
      marker = {};
      marker.frame = number;
      marker.depth = record.depth;
      marker.name_size = record.name_size;
      std::memcpy(marker.name, record.name, record.name_size);
      place(marker, m_ticks[record.query], m_ticks[record.query + 1],
            m_valid_bits);
    }
}

usize VulkanGpuMarkers::read(const std::span<gpu_marker> out) noexcept {
  const usize count = nx::min(out.size(), m_ready_count);
  for (usize i = 0; i < count; ++i)
    out[i] = m_ready[(m_ready_first + i) % READY_MARKERS];
  if (count != 0) {
    m_ready_first = (m_ready_first + count) % READY_MARKERS;
    m_ready_count -= count;
  }
  return count;
}

gpu_marker_stats VulkanGpuMarkers::stats() const noexcept {
  return {.frames = m_frames_read,
          .groups_dropped = m_groups_dropped.load(std::memory_order_relaxed),
          .markers_dropped = m_markers_dropped,
          .calibrated = m_calibrated != nullptr};
}

} // namespace nx::gfx::detail
