#include "src/vulkan/vk_gpu_markers.h"

#include "core/foundation/platform/os.h"

#include <cstring>

namespace nx::gfx::detail {
namespace {

constexpr u32 NOT_TIMED = ~u32{0};
constexpr u64 RECALIBRATE_NS = 1'000'000'000;

} // namespace

void VulkanGpuMarkers::init(const VkDevice device, const u32 frames,
                            const f32 period_ns, const u32 valid_bits,
                            const PFN_vkGetCalibratedTimestampsEXT calibrated,
                            const bool inside_passes) {
  m_device = device;
  m_period_ns = period_ns;
  m_valid_mask = valid_bits >= 64 ? ~u64{0} : (u64{1} << valid_bits) - 1;
  m_calibrated = calibrated;
  m_inside_passes = inside_passes;
  m_stats.calibrated = calibrated != nullptr;
  m_slots.resize(frames);
}

void VulkanGpuMarkers::shutdown() noexcept {
  for (Slot &slot : m_slots)
    if (slot.pool != VK_NULL_HANDLE)
      vkDestroyQueryPool(m_device, slot.pool, nullptr);
  m_slots.clear();
  m_ready.clear();
  m_ready_first = 0;
  m_ready_count = 0;
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
    const u64 host = before + (after - before) / 2;
    const f64 device_ns =
        static_cast<f64>(device_ticks & m_valid_mask) * m_period_ns;
    m_offset_ns = static_cast<i64>(host) - static_cast<i64>(device_ns);
    m_have_offset = true;
    m_calibrated_at_ns = after;
  }
}

VulkanGpuMarkers::Slot *
VulkanGpuMarkers::open_frame(const u32 frame, const VkCommandBuffer cmd) {
  if (frame >= m_slots.size())
    return nullptr;
  Slot &slot = m_slots[frame];
  if (slot.written != 0) {
    if (slot.submitted) {
      collect(slot);
      slot.to_reset = slot.written;
    } else {
      // Nothing it recorded ran, its reset included.
      slot.ever_reset = false;
    }
    slot.written = 0;
  }
  slot.submitted = false;
  slot.depth = 0;
  slot.records.clear();

  if (!m_wanted.load(std::memory_order_acquire))
    return nullptr;
  if (slot.pool == VK_NULL_HANDLE) {
    VkQueryPoolCreateInfo create{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    create.queryType = VK_QUERY_TYPE_TIMESTAMP;
    create.queryCount = MOST_A_FRAME * 2;
    if (vkCreateQueryPool(m_device, &create, nullptr, &slot.pool) != VK_SUCCESS)
      return nullptr;
    slot.records.reserve(MOST_A_FRAME);
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
      (!m_have_offset ||
       nx::os::ticks_ns() - m_calibrated_at_ns > RECALIBRATE_NS))
    calibrate();
  return &slot;
}

void VulkanGpuMarkers::frame_submitted(const u32 frame) noexcept {
  if (frame >= m_slots.size())
    return;
  m_slots[frame].submitted = true;
  m_slots[frame].submitted_ns = nx::os::ticks_ns();
}

void VulkanGpuMarkers::push(Slot &slot, const VkCommandBuffer cmd,
                            const nx::string_view name,
                            const bool in_render_pass) {
  const u32 depth = slot.depth++;
  if (depth >= MAX_DEPTH) {
    ++m_stats.groups_dropped;
    return;
  }
  if (in_render_pass && !m_inside_passes) {
    slot.open[depth] = NOT_TIMED;
    return;
  }
  if (slot.records.size() == MOST_A_FRAME) {
    slot.open[depth] = NOT_TIMED;
    ++m_stats.groups_dropped;
    return;
  }
  const u32 index = nx::cast<u32>(slot.records.size());
  Slot::Record record;
  record.depth = nx::cast<u16>(depth);
  record.name_size = nx::cast<u8>(nx::min(name.size(), GPU_MARKER_NAME_BYTES));
  std::memcpy(record.name, name.data(), record.name_size);
  slot.records.push_back(record);
  slot.open[depth] = index;
  slot.written = (index + 1) * 2;
  // Not the top of the pipe: a begin there may pass the group before it, and a
  // lane needs groups that do not overlap.
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, slot.pool,
                      index * 2);
}

void VulkanGpuMarkers::pop(Slot &slot, const VkCommandBuffer cmd) noexcept {
  if (slot.depth == 0)
    return;
  const u32 depth = --slot.depth;
  if (depth >= MAX_DEPTH || slot.open[depth] == NOT_TIMED)
    return;
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, slot.pool,
                      slot.open[depth] * 2 + 1);
}

void VulkanGpuMarkers::end_open(Slot &slot,
                                const VkCommandBuffer cmd) noexcept {
  while (slot.depth > 0)
    pop(slot, cmd);
}

void VulkanGpuMarkers::collect(Slot &slot) {
  m_ticks.resize(slot.written);
  // No wait: its frame has finished, and a lost device reports that instead.
  const VkResult result = vkGetQueryPoolResults(
      m_device, slot.pool, 0, slot.written, m_ticks.size() * sizeof(u64),
      m_ticks.data(), sizeof(u64), VK_QUERY_RESULT_64_BIT);
  if (result != VK_SUCCESS)
    return;
  ++m_stats.frames;

  const auto device_ns = [this](const u64 ticks) {
    return static_cast<f64>(ticks & m_valid_mask) * m_period_ns;
  };
  i64 offset = m_offset_ns;
  if (!m_have_offset && !slot.records.empty())
    offset = static_cast<i64>(slot.submitted_ns) -
             static_cast<i64>(device_ns(m_ticks[0]));

  for (usize i = 0; i < slot.records.size(); ++i) {
    const Slot::Record &record = slot.records[i];
    if (m_ready_count == READY_MARKERS) {
      m_ready_first = (m_ready_first + 1) % READY_MARKERS;
      --m_ready_count;
      ++m_stats.markers_dropped;
    }
    gpu_marker &marker =
        m_ready[(m_ready_first + m_ready_count) % READY_MARKERS];
    ++m_ready_count;
    const i64 begin = static_cast<i64>(device_ns(m_ticks[i * 2])) + offset;
    const i64 end = static_cast<i64>(device_ns(m_ticks[i * 2 + 1])) + offset;
    marker.begin_ns = static_cast<u64>(nx::max<i64>(begin, 0));
    marker.end_ns =
        nx::max(static_cast<u64>(nx::max<i64>(end, 0)), marker.begin_ns);
    marker.depth = record.depth;
    marker.name_size = record.name_size;
    std::memcpy(marker.name, record.name, record.name_size);
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

gpu_marker_stats VulkanGpuMarkers::stats() const noexcept { return m_stats; }

} // namespace nx::gfx::detail
