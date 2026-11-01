#pragma once

#include "core/foundation/threading/sync.h"
#include "nx/gfx/memory.h"

#include <atomic>
#include <span>

namespace nx::gfx::detail {

/// What is alive, by kind, and when it came and went, for whoever reads it.
/// Any thread.
class VulkanResourceLedger {
public:
  void created(gpu_resource_kind kind, u64 id, u64 bytes,
               nx::string_view name) noexcept;
  void destroyed(gpu_resource_kind kind, u64 id, u64 bytes,
                 nx::string_view name) noexcept;

  void set_events(bool on);
  usize read(std::span<gpu_resource_event> out) noexcept;
  [[nodiscard]] u64 dropped() const noexcept {
    return m_dropped.load(std::memory_order_relaxed);
  }
  void totals(gpu_memory_stats &out) const noexcept;

private:
  struct Live {
    std::atomic<u64> bytes{0};
    std::atomic<u32> count{0};
  };

  void note(gpu_resource_kind kind, u64 id, u64 bytes, nx::string_view name,
            bool created) noexcept;

  Live m_live[3];
  std::atomic<bool> m_on{false};
  std::atomic<u64> m_dropped{0};
  nx::mutex m_lock;
  nx::vector<gpu_resource_event> m_events;
  usize m_first = 0;
  usize m_count = 0;
};

} // namespace nx::gfx::detail
