// nx_gfx_spirv normalize <in.spv> <out.spv>: nx::gfx::normalize_spirv on a
// compiled module, as nx_gfx_compile_slang runs it after slangc. <out> may be
// <in>.
#include "core/foundation/platform/filesystem.h"
#include "nx/gfx/spirv.h"

#include <cstdio>
#include <cstring>

int main(const int argc, char **const argv) {
  if (argc != 4 || std::strcmp(argv[1], "normalize") != 0) {
    std::fprintf(stderr, "usage: nx_gfx_spirv normalize <in.spv> <out.spv>\n");
    return 2;
  }
  auto read = nx::fs::file_read(nx::fs::path_view(argv[2]));
  if (!read.has_value()) {
    std::fprintf(stderr, "nx_gfx_spirv: cannot read %s\n", argv[2]);
    return 1;
  }
  nx::blob<u8> &module = read.value();
  const auto size = nx::gfx::normalize_spirv({module.data(), module.size()});
  if (!size.has_value()) {
    const nx::string_view why = nx::gfx::spirv_error_name(size.error());
    std::fprintf(stderr, "nx_gfx_spirv: %s: %.*s\n", argv[2], int(why.size()),
                 why.data());
    return 1;
  }
  if (!nx::fs::file_write(nx::fs::path_view(argv[3]),
                          {module.data(), size.value()})
           .has_value()) {
    std::fprintf(stderr, "nx_gfx_spirv: cannot write %s\n", argv[3]);
    return 1;
  }
  return 0;
}
