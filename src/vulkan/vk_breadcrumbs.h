#pragma once

#include "core/foundation/core/foundation.h"
#include "src/vulkan/vk_common.h"
#include "src/vulkan/vk_mem_alloc_include.h"

#include <atomic>
#include <span>

namespace nx::gfx::detail {

/// How far the GPU got, in host memory it writes as it goes, which outlives a
/// lost device: each debug group begun writes a word when the queue reaches it
/// and another once everything before its end has finished. A driver's own dump
/// of a hang may name nothing, as it keeps only the next frame's first marker.
class VulkanBreadcrumbs {
public:
  /// Debug groups remembered.
  static constexpr u32 ENTRIES = 1024;

  enum class State : u8 { NotStarted, Running, Done };
  struct Entry {
    u64 label = 0;
    State state = State::NotStarted;
  };

  /// False where it could not get its memory; marks then do nothing.
  [[nodiscard]] bool init(VmaAllocator allocator);
  void shutdown() noexcept;

  /// Marks a debug group named @p label beginning on @p cmd, and returns what
  /// end() takes for it.
  [[nodiscard]] u32 begin(VkCommandBuffer cmd, u64 label) noexcept;
  void end(VkCommandBuffer cmd, u32 mark) noexcept;

  /// The last @p most groups marked, oldest first.
  [[nodiscard]] nx::vector<Entry> recent(usize most) const;

  /// What @p words, two for each of ENTRIES, say of the last @p most groups
  /// marked before @p next, oldest first. Those never reached past the first
  /// few are left out: they only say what was recorded.
  [[nodiscard]] static nx::vector<Entry> read(std::span<const u32> words,
                                              u32 next, usize most);

private:
  VmaAllocator m_allocator = VK_NULL_HANDLE;
  VkBuffer m_buffer = VK_NULL_HANDLE;
  VmaAllocation m_allocation = VK_NULL_HANDLE;
  const u32 *m_words = nullptr;
  std::atomic<u32> m_next{1};
};

} // namespace nx::gfx::detail
