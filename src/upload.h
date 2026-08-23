#pragma once

#include "core/foundation/core/foundation.h"
#include "core/foundation/strings/utf8_string_view.h"

namespace nxe::rhi {

struct UploadTicket {
  u64 value = 0;
  bool failed = false;

  [[nodiscard]] bool valid() const noexcept { return value != 0; }
  [[nodiscard]] bool ok() const noexcept { return !failed; }

  [[nodiscard]] static UploadTicket failure() noexcept { return {0, true}; }
};

enum class LoadStatus : u8 {
  None,
  Pending,
  Decoding,
  Uploading,
  Ready,
  Failed,    ///< Missing file, malformed data, or an unsupported format.
};

[[nodiscard]] nx::string_view to_string(LoadStatus status) noexcept;

struct LoadRequest {
  u32 id = 0;
  u32 generation = 0;
  [[nodiscard]] bool valid() const noexcept { return id != 0; }
};

}
