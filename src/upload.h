#pragma once

#include "core/foundation/core/foundation.h"

namespace nxe::rhi {

struct UploadTicket {
  u64 value = 0;
  bool failed = false;

  [[nodiscard]] bool valid() const noexcept { return value != 0; }
  [[nodiscard]] bool ok() const noexcept { return !failed; }

  [[nodiscard]] static UploadTicket failure() noexcept { return {0, true}; }
};

}
