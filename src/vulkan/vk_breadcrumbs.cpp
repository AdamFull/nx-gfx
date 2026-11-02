#include "src/vulkan/vk_breadcrumbs.h"

#include <algorithm>
#include <cstring>

namespace nx::gfx::detail {

namespace {

// A word is a group's number over its label, so a slot's word says whether
// it is that group's or an older one's, which wrapped round to it.
constexpr u32 LABEL_BITS = 12;
constexpr u32 LABEL_MASK = (1u << LABEL_BITS) - 1u;
constexpr u32 NUMBER_MASK = (1u << (32u - LABEL_BITS)) - 1u;
// Past the first few, a group not reached says only what was recorded.
constexpr usize NOT_STARTED_SHOWN = 3;

[[nodiscard]] constexpr u32 number_of(const u32 word) noexcept {
  return word >> LABEL_BITS;
}

} // namespace

bool VulkanBreadcrumbs::init(const VmaAllocator allocator) {
  if (vkCmdWriteBufferMarkerAMD == nullptr)
    return false;
  VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  info.size = u64{ENTRIES} * 2u * sizeof(u32);
  info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VmaAllocationCreateInfo alloc{};
  alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
  alloc.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
  VmaAllocationInfo allocation{};
  if (vmaCreateBuffer(allocator, &info, &alloc, &m_buffer, &m_allocation,
                      &allocation) != VK_SUCCESS)
    return false;
  vmaSetAllocationName(allocator, m_allocation, "gpu breadcrumbs");
  std::memset(allocation.pMappedData, 0, info.size);
  m_words = static_cast<const u32 *>(allocation.pMappedData);
  m_allocator = allocator;
  return true;
}

void VulkanBreadcrumbs::shutdown() noexcept {
  if (m_allocator != VK_NULL_HANDLE && m_buffer != VK_NULL_HANDLE)
    vmaDestroyBuffer(m_allocator, m_buffer, m_allocation);
  m_buffer = VK_NULL_HANDLE;
  m_allocation = VK_NULL_HANDLE;
  m_words = nullptr;
  m_allocator = VK_NULL_HANDLE;
}

u32 VulkanBreadcrumbs::begin(const VkCommandBuffer cmd,
                             const u64 label) noexcept {
  if (m_buffer == VK_NULL_HANDLE)
    return 0;
  u32 number = m_next.fetch_add(1, std::memory_order_relaxed) & NUMBER_MASK;
  if (number == 0)
    number = m_next.fetch_add(1, std::memory_order_relaxed) & NUMBER_MASK;
  const u32 mark = number << LABEL_BITS |
                   (label <= LABEL_MASK ? static_cast<u32>(label) : 0u);
  vkCmdWriteBufferMarkerAMD(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_buffer,
                            u64{number % ENTRIES} * 2u * sizeof(u32), mark);
  return mark;
}

void VulkanBreadcrumbs::end(const VkCommandBuffer cmd,
                            const u32 mark) noexcept {
  if (m_buffer == VK_NULL_HANDLE || mark == 0)
    return;
  vkCmdWriteBufferMarkerAMD(
      cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_buffer,
      u64{number_of(mark) % ENTRIES} * 2u * sizeof(u32) + sizeof(u32), mark);
}

nx::vector<VulkanBreadcrumbs::Entry>
VulkanBreadcrumbs::recent(const usize most) const {
  if (m_words == nullptr)
    return {};
  (void)vmaInvalidateAllocation(m_allocator, m_allocation, 0, VK_WHOLE_SIZE);
  return read({m_words, usize{ENTRIES} * 2u},
              m_next.load(std::memory_order_relaxed) & NUMBER_MASK, most);
}

nx::vector<VulkanBreadcrumbs::Entry>
VulkanBreadcrumbs::read(const std::span<const u32> words, const u32 next,
                        const usize most) {
  nx::vector<Entry> all;
  for (u32 age = ENTRIES - 1u; age > 0u; --age) {
    const u32 number = (next - age) & NUMBER_MASK;
    if (number == 0)
      continue;
    const usize slot = usize{number % ENTRIES} * 2u;
    const u32 started = words[slot];
    const u32 done = words[slot + 1u];
    if (number_of(done) == number)
      all.push_back({done & LABEL_MASK, State::Done});
    else if (number_of(started) == number)
      all.push_back({started & LABEL_MASK, State::Running});
    else
      all.push_back({0, State::NotStarted});
  }
  // Slots never written come first and say nothing.
  usize first = 0;
  while (first < all.size() && all[first].state == State::NotStarted)
    ++first;
  usize last = all.size();
  usize not_started = 0;
  while (last > first && all[last - 1u].state == State::NotStarted) {
    ++not_started;
    --last;
  }
  last = std::min(all.size(), last + std::min(not_started, NOT_STARTED_SHOWN));
  const usize begin = last - first > most ? last - most : first;
  nx::vector<Entry> out;
  for (usize i = begin; i < last; ++i)
    out.push_back(all[i]);
  return out;
}

} // namespace nx::gfx::detail
