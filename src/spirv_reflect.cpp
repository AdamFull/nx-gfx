#include "rendering/rhi/spirv_reflect.h"

#include "foundation/containers/small_vector.h"

#include <cstring>

namespace nxe::rhi {
namespace {

// SPIR-V 1.x physical layout: a five-word header, then a stream of
// instructions whose first word packs the word count and the opcode.
constexpr u32 SPIRV_MAGIC = 0x07230203u;
constexpr usize SPIRV_HEADER_WORDS = 5;

constexpr u16 OP_ENTRY_POINT = 15;
constexpr u16 OP_EXECUTION_MODE = 16;
constexpr u16 OP_CONSTANT = 43;
constexpr u16 OP_EXECUTION_MODE_ID = 331;

constexpr u32 EXECUTION_MODEL_GL_COMPUTE = 5;
constexpr u32 EXEC_MODE_LOCAL_SIZE = 17;
constexpr u32 EXEC_MODE_LOCAL_SIZE_ID = 38;

/// A SPIR-V literal string: NUL-terminated and padded out to a word boundary.
/// Returns the view and, through @p out_words, how many words it occupies.
nx::string_view literal_string(const std::span<const u32> words,
                               const usize start, usize &out_words) noexcept {
  out_words = 0;
  for (usize i = start; i < words.size(); ++i) {
    const u32 word = words[i];
    if ((word & 0x000000FFu) == 0 || (word & 0x0000FF00u) == 0 ||
        (word & 0x00FF0000u) == 0 || (word & 0xFF000000u) == 0) {
      out_words = i - start + 1;
      break;
    }
  }
  if (out_words == 0)
    return {}; // Unterminated: malformed.

  const auto *chars = reinterpret_cast<const char *>(words.data() + start);
  const usize capacity = out_words * sizeof(u32);
  usize length = 0;
  while (length < capacity && chars[length] != '\0')
    ++length;
  return nx::string_view(chars, length);
}

/// Validates the header and returns the blob as words, or an empty span.
///
/// The words must be read through the original allocation: SpirvComputeEntry
/// names point into it, so copying to an aligned scratch buffer would leave
/// them dangling. In practice every producer of these blobs (a file read into
/// nx::blob, an embedded array) is at least 4-byte aligned; anything else is
/// declined rather than mis-parsed.
std::span<const u32> as_words(const std::span<const u8> code) noexcept {
  if (code.size() < SPIRV_HEADER_WORDS * sizeof(u32) ||
      code.size() % sizeof(u32) != 0)
    return {};
  if (reinterpret_cast<uintptr_t>(code.data()) % alignof(u32) != 0)
    return {};

  u32 magic = 0;
  std::memcpy(&magic, code.data(), sizeof(magic));
  // A byte-reversed magic is valid SPIR-V of the opposite endianness. Nothing
  // this engine targets produces it, and silently mis-parsing would be worse
  // than declining.
  if (magic != SPIRV_MAGIC)
    return {};

  return {reinterpret_cast<const u32 *>(code.data()),
          code.size() / sizeof(u32)};
}

} // namespace

usize spirv_compute_entries(const std::span<const u8> code,
                            const std::span<SpirvComputeEntry> out) noexcept {
  const std::span<const u32> words = as_words(code);
  if (words.empty() || out.empty())
    return 0;

  // Pass one: the compute entry points, and any 32-bit constants in case a
  // size is stated through OpExecutionModeId.
  nx::small_vector<u32, 8> entry_ids;
  nx::small_vector<u32, 16> constant_ids;
  nx::small_vector<u32, 16> constant_values;
  usize count = 0;

  usize cursor = SPIRV_HEADER_WORDS;
  while (cursor < words.size()) {
    const u32 instruction = words[cursor];
    const u16 opcode = nx::cast<u16>(instruction & 0xFFFFu);
    const usize length = instruction >> 16;
    // A zero length would loop forever; anything overrunning the blob is
    // truncated. Either way the stream is unusable from here on.
    if (length == 0 || cursor + length > words.size())
      return count;

    if (opcode == OP_ENTRY_POINT && length >= 4) {
      // [1] execution model, [2] <id>, [3..] name, then interfaces.
      if (words[cursor + 1] == EXECUTION_MODEL_GL_COMPUTE &&
          count < out.size()) {
        usize name_words = 0;
        const nx::string_view name =
            literal_string(words, cursor + 3, name_words);
        if (name_words != 0) {
          out[count].name = name;
          out[count].local_size[0] = 0;
          out[count].local_size[1] = 0;
          out[count].local_size[2] = 0;
          entry_ids.push_back(words[cursor + 2]);
          ++count;
        }
      }
    } else if (opcode == OP_CONSTANT && length == 4) {
      // [1] result type, [2] result <id>, [3] value. Only a single-word
      // (32-bit) constant can be a workgroup dimension.
      constant_ids.push_back(words[cursor + 2]);
      constant_values.push_back(words[cursor + 3]);
    }

    cursor += length;
  }

  if (count == 0)
    return 0;

  const auto resolve_constant = [&](const u32 id, u32 &value) noexcept {
    for (usize i = 0; i < constant_ids.size(); ++i) {
      if (constant_ids[i] == id) {
        value = constant_values[i];
        return true;
      }
    }
    return false;
  };

  // Pass two: match execution modes back to the entry points found above.
  cursor = SPIRV_HEADER_WORDS;
  while (cursor < words.size()) {
    const u32 instruction = words[cursor];
    const u16 opcode = nx::cast<u16>(instruction & 0xFFFFu);
    const usize length = instruction >> 16;
    if (length == 0 || cursor + length > words.size())
      return count;

    const bool is_mode =
        opcode == OP_EXECUTION_MODE || opcode == OP_EXECUTION_MODE_ID;
    if (is_mode && length >= 6) {
      usize index = entry_ids.size();
      for (usize i = 0; i < entry_ids.size(); ++i) {
        if (entry_ids[i] == words[cursor + 1]) {
          index = i;
          break;
        }
      }

      if (index < count) {
        const u32 mode = words[cursor + 2];
        if (opcode == OP_EXECUTION_MODE && mode == EXEC_MODE_LOCAL_SIZE) {
          out[index].local_size[0] = words[cursor + 3];
          out[index].local_size[1] = words[cursor + 4];
          out[index].local_size[2] = words[cursor + 5];
        } else if (opcode == OP_EXECUTION_MODE_ID &&
                   mode == EXEC_MODE_LOCAL_SIZE_ID) {
          u32 resolved[3] = {};
          // Specialisation constants will not resolve; the entry keeps its
          // zero size, which callers read as "not knowable here".
          if (resolve_constant(words[cursor + 3], resolved[0]) &&
              resolve_constant(words[cursor + 4], resolved[1]) &&
              resolve_constant(words[cursor + 5], resolved[2])) {
            out[index].local_size[0] = resolved[0];
            out[index].local_size[1] = resolved[1];
            out[index].local_size[2] = resolved[2];
          }
        }
      }
    }

    cursor += length;
  }

  return count;
}

bool spirv_local_size(const std::span<const u8> code,
                      const nx::string_view entry_point,
                      u32 out_size[3]) noexcept {
  SpirvComputeEntry entries[16];
  const usize count = spirv_compute_entries(code, entries);

  for (usize i = 0; i < count; ++i) {
    if (entries[i].name != entry_point)
      continue;
    if (entries[i].local_size[0] == 0)
      return false; // Present, but the size is not stated as literals.
    out_size[0] = entries[i].local_size[0];
    out_size[1] = entries[i].local_size[1];
    out_size[2] = entries[i].local_size[2];
    return true;
  }
  return false;
}

} // namespace nxe::rhi
