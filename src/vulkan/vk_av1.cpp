#include "core/rendering/rhi/vulkan/vk_av1.h"

namespace nxe::rhi {
namespace {

class BitReader {
public:
  BitReader(const u8 *const data, const u32 size) noexcept
      : m_data(data), m_bits(nx::cast<u64>(size) * 8) {}

  [[nodiscard]] u32 f(const u32 n) noexcept {
    u32 value = 0;
    for (u32 i = 0; i < n; ++i)
      value = (value << 1) | bit();
    return value;
  }

  // uvlc() (spec 4.10.3): a run of zeros, a one, then that many value bits.
  [[nodiscard]] u32 uvlc() noexcept {
    u32 leading = 0;
    while (leading < 32 && bit() == 0)
      ++leading;
    if (leading >= 32)
      return 0xFFFFFFFFu;
    return f(leading) + (1u << leading) - 1;
  }

  [[nodiscard]] bool overrun() const noexcept { return m_overrun; }

private:
  [[nodiscard]] u32 bit() noexcept {
    if (m_pos >= m_bits) {
      m_overrun = true;
      return 0;
    }
    const u32 byte = m_data[m_pos >> 3];
    const u32 b = (byte >> (7 - (m_pos & 7))) & 1u;
    ++m_pos;
    return b;
  }

  const u8 *m_data;
  u64 m_bits;
  u64 m_pos = 0;
  bool m_overrun = false;
};

// leb128 (spec 4.10.5): little-endian base-128, up to eight bytes. Advances @p p.
[[nodiscard]] u64 leb128(const u8 *&p, const u8 *const end, bool &ok) noexcept {
  u64 value = 0;
  for (u32 i = 0; i < 8; ++i) {
    if (p >= end) {
      ok = false;
      return 0;
    }
    const u8 b = *p++;
    value |= nx::cast<u64>(b & 0x7f) << (i * 7);
    if ((b & 0x80) == 0)
      return value;
  }
  ok = false;
  return 0;
}

constexpr u32 OBU_SEQUENCE_HEADER = 1;
constexpr u8 SELECT_SCREEN_CONTENT_TOOLS = 2;
constexpr u8 SELECT_INTEGER_MV = 2;

// Unspecified colour code points (spec 6.4.2), the defaults when a stream leaves
// its colour description out.
constexpr u32 CP_BT_709 = 1;
constexpr u32 CP_UNSPECIFIED = 2;
constexpr u32 TC_SRGB = 13;
constexpr u32 TC_UNSPECIFIED = 2;
constexpr u32 MC_IDENTITY = 0;
constexpr u32 MC_UNSPECIFIED = 2;

void parse_color_config(BitReader &r, const u32 seq_profile,
                        StdVideoAV1ColorConfig &cc) {
  cc.flags = {};
  const u32 high_bitdepth = r.f(1);
  u32 bit_depth = 8;
  if (seq_profile == 2 && high_bitdepth != 0)
    bit_depth = r.f(1) != 0 ? 12 : 10;
  else
    bit_depth = high_bitdepth != 0 ? 10 : 8;
  cc.BitDepth = nx::cast<u8>(bit_depth);

  const u32 mono_chrome = seq_profile == 1 ? 0 : r.f(1);
  cc.flags.mono_chrome = mono_chrome;

  const u32 color_description_present = r.f(1);
  cc.flags.color_description_present_flag = color_description_present;
  u32 cp = CP_UNSPECIFIED;
  u32 tc = TC_UNSPECIFIED;
  u32 mc = MC_UNSPECIFIED;
  if (color_description_present != 0) {
    cp = r.f(8);
    tc = r.f(8);
    mc = r.f(8);
  }
  cc.color_primaries = nx::cast<StdVideoAV1ColorPrimaries>(cp);
  cc.transfer_characteristics =
      nx::cast<StdVideoAV1TransferCharacteristics>(tc);
  cc.matrix_coefficients = nx::cast<StdVideoAV1MatrixCoefficients>(mc);

  if (mono_chrome != 0) {
    cc.flags.color_range = r.f(1);
    cc.subsampling_x = 1;
    cc.subsampling_y = 1;
    cc.chroma_sample_position =
        nx::cast<StdVideoAV1ChromaSamplePosition>(0); // UNKNOWN
    cc.flags.separate_uv_delta_q = 0;
    return;
  }

  u32 subsampling_x = 1;
  u32 subsampling_y = 1;
  if (cp == CP_BT_709 && tc == TC_SRGB && mc == MC_IDENTITY) {
    cc.flags.color_range = 1;
    subsampling_x = 0;
    subsampling_y = 0;
  } else {
    cc.flags.color_range = r.f(1);
    if (seq_profile == 0) {
      subsampling_x = 1;
      subsampling_y = 1;
    } else if (seq_profile == 1) {
      subsampling_x = 0;
      subsampling_y = 0;
    } else if (bit_depth == 12) {
      subsampling_x = r.f(1);
      subsampling_y = subsampling_x != 0 ? r.f(1) : 0;
    } else {
      subsampling_x = 1;
      subsampling_y = 0;
    }
    if (subsampling_x != 0 && subsampling_y != 0)
      cc.chroma_sample_position =
          nx::cast<StdVideoAV1ChromaSamplePosition>(r.f(2));
  }
  cc.subsampling_x = nx::cast<u8>(subsampling_x);
  cc.subsampling_y = nx::cast<u8>(subsampling_y);
  cc.flags.separate_uv_delta_q = r.f(1);
}

[[nodiscard]] bool parse_seq_header_obu(const u8 *const data, const u32 size,
                                        Av1SequenceHeader &out) {
  BitReader r(data, size);
  StdVideoAV1SequenceHeader &sh = out.header;
  StdVideoAV1SequenceHeaderFlags &f = sh.flags;
  f = {};

  const u32 seq_profile = r.f(3);
  sh.seq_profile = nx::cast<StdVideoAV1Profile>(seq_profile);
  f.still_picture = r.f(1);
  f.reduced_still_picture_header = r.f(1);

  bool decoder_model = false;
  u32 buffer_delay_length = 0;
  if (f.reduced_still_picture_header != 0) {
    (void)r.f(5); // seq_level_idx[0]
  } else {
    f.timing_info_present_flag = r.f(1);
    if (f.timing_info_present_flag != 0) {
      out.timing.num_units_in_display_tick = r.f(32);
      out.timing.time_scale = r.f(32);
      out.timing.flags = {};
      const u32 equal = r.f(1);
      out.timing.flags.equal_picture_interval = equal;
      if (equal != 0)
        out.timing.num_ticks_per_picture_minus_1 = r.uvlc();
      sh.pTimingInfo = &out.timing;

      decoder_model = r.f(1) != 0;
      if (decoder_model) {
        buffer_delay_length = r.f(5) + 1;
        (void)r.f(32); // num_units_in_decoding_tick
        (void)r.f(5);  // buffer_removal_time_length_minus_1
        (void)r.f(5);  // frame_presentation_time_length_minus_1
      }
    }
    f.initial_display_delay_present_flag = r.f(1);
    const u32 op_count = r.f(5); // operating_points_cnt_minus_1
    for (u32 i = 0; i <= op_count; ++i) {
      (void)r.f(12); // operating_point_idc[i]
      const u32 level = r.f(5);
      if (level > 7)
        (void)r.f(1); // seq_tier[i]
      if (decoder_model && r.f(1) != 0) {
        (void)r.f(buffer_delay_length); // decoder_buffer_delay
        (void)r.f(buffer_delay_length); // encoder_buffer_delay
        (void)r.f(1);                    // low_delay_mode_flag
      }
      if (f.initial_display_delay_present_flag != 0 && r.f(1) != 0)
        (void)r.f(4); // initial_display_delay_minus_1[i]
    }
  }

  sh.frame_width_bits_minus_1 = nx::cast<u8>(r.f(4));
  sh.frame_height_bits_minus_1 = nx::cast<u8>(r.f(4));
  sh.max_frame_width_minus_1 =
      nx::cast<u16>(r.f(nx::cast<u32>(sh.frame_width_bits_minus_1) + 1));
  sh.max_frame_height_minus_1 =
      nx::cast<u16>(r.f(nx::cast<u32>(sh.frame_height_bits_minus_1) + 1));

  f.frame_id_numbers_present_flag =
      f.reduced_still_picture_header != 0 ? 0 : r.f(1);
  if (f.frame_id_numbers_present_flag != 0) {
    sh.delta_frame_id_length_minus_2 = nx::cast<u8>(r.f(4));
    sh.additional_frame_id_length_minus_1 = nx::cast<u8>(r.f(3));
  }

  f.use_128x128_superblock = r.f(1);
  f.enable_filter_intra = r.f(1);
  f.enable_intra_edge_filter = r.f(1);

  if (f.reduced_still_picture_header != 0) {
    sh.seq_force_screen_content_tools = SELECT_SCREEN_CONTENT_TOOLS;
    sh.seq_force_integer_mv = SELECT_INTEGER_MV;
    sh.order_hint_bits_minus_1 = 0;
  } else {
    f.enable_interintra_compound = r.f(1);
    f.enable_masked_compound = r.f(1);
    f.enable_warped_motion = r.f(1);
    f.enable_dual_filter = r.f(1);
    f.enable_order_hint = r.f(1);
    if (f.enable_order_hint != 0) {
      f.enable_jnt_comp = r.f(1);
      f.enable_ref_frame_mvs = r.f(1);
    }
    const u32 choose_sct = r.f(1);
    const u32 force_sct = choose_sct != 0 ? SELECT_SCREEN_CONTENT_TOOLS : r.f(1);
    sh.seq_force_screen_content_tools = nx::cast<u8>(force_sct);
    if (force_sct > 0) {
      const u32 choose_imv = r.f(1);
      sh.seq_force_integer_mv =
          choose_imv != 0 ? SELECT_INTEGER_MV : nx::cast<u8>(r.f(1));
    } else {
      sh.seq_force_integer_mv = SELECT_INTEGER_MV;
    }
    sh.order_hint_bits_minus_1 =
        f.enable_order_hint != 0 ? nx::cast<u8>(r.f(3)) : 0;
  }

  f.enable_superres = r.f(1);
  f.enable_cdef = r.f(1);
  f.enable_restoration = r.f(1);

  parse_color_config(r, seq_profile, out.color);
  sh.pColorConfig = &out.color;

  f.film_grain_params_present = r.f(1);

  if (r.overrun())
    return false;
  out.valid = true;
  return true;
}

} // namespace

bool parse_av1_sequence_header(const u8 *const data, const u32 size,
                               Av1SequenceHeader &out) {
  const u8 *p = data;
  const u8 *const end = data + size;
  while (p < end) {
    const u8 header = *p++;
    const u32 obu_type = (header >> 3) & 0xfu;
    const bool extension = ((header >> 2) & 1u) != 0;
    const bool has_size = ((header >> 1) & 1u) != 0;
    if (extension) {
      if (p >= end)
        return false;
      ++p; // extension_header: temporal_id/spatial_id
    }
    u64 payload_size = 0;
    if (has_size) {
      bool ok = true;
      payload_size = leb128(p, end, ok);
      if (!ok)
        return false;
    } else {
      payload_size = nx::cast<u64>(end - p);
    }
    if (nx::cast<u64>(end - p) < payload_size)
      return false;
    if (obu_type == OBU_SEQUENCE_HEADER)
      return parse_seq_header_obu(p, nx::cast<u32>(payload_size), out);
    p += payload_size;
  }
  return false;
}

bool parse_av1_codec_private(const u8 *const data, const u32 size,
                             Av1SequenceHeader &out) {
  // av1C (AV1CodecConfigurationRecord): a 4-byte fixed header, then configOBUs -
  // where the sequence header lives.
  if (size < 4)
    return false;
  return parse_av1_sequence_header(data + 4, size - 4, out);
}

} // namespace nxe::rhi
