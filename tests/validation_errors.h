#pragma once

#include "core/foundation/diagnostics/log.h"

#include <atomic>

namespace nx::gfx::test {

/// Counts the validation layer's errors: a VUID, or a hazard that
/// synchronization validation found (its reports name no VUID).
class ValidationErrors final : public nx::log::Sink {
public:
  void write(const nx::log::Record &r) override {
    const nx::string_view message(r.message, r.length);
    if (message.find("VUID") != nx::string_view::npos ||
        message.find("hazard detected") != nx::string_view::npos)
      m_count.fetch_add(1, std::memory_order_relaxed);
  }
  [[nodiscard]] u32 count() const noexcept {
    return m_count.load(std::memory_order_relaxed);
  }

private:
  std::atomic<u32> m_count{0};
};

} // namespace nx::gfx::test
