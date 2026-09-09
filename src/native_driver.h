#pragma once

#include "core/foundation/strings/utf8_string_view.h"

namespace nxe::rhi {

using NativeDriverResolve = void *(*)(void *context,
                                      const char *symbol) noexcept;

/// Non-owning access to a process-local graphics driver library. The provider
/// that created this view must outlive the RHI device and release the library
/// only after every native object has been destroyed.
struct NativeDriverView {
  void *context = nullptr;
  NativeDriverResolve resolve = nullptr;
  nx::string_view identity = "system";

  [[nodiscard]] bool custom() const noexcept { return resolve != nullptr; }

  [[nodiscard]] void *find(const char *symbol) const noexcept {
    return resolve != nullptr ? resolve(context, symbol) : nullptr;
  }
};

} // namespace nxe::rhi
