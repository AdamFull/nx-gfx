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

  // su(n) (spec 4.10.6): n bits, top bit is the sign.
  [[nodiscard]] i32 su(const u32 n) noexcept {
    const u32 value = f(n);
    const u32 sign_mask = 1u << (n - 1);
    if ((value & sign_mask) != 0)
      return nx::cast<i32>(value) - nx::cast<i32>(2u * sign_mask);
    return nx::cast<i32>(value);
  }

  // ns(n) (spec 4.10.7): a non-symmetric encoding of [0, n) that spends the
  // fewest bits on the smaller values.
  [[nodiscard]] u32 ns(const u32 n) noexcept {
    const u32 w = floor_log2(n) + 1;
    const u32 m = (1u << w) - n;
    const u32 v = f(w - 1);
    if (v < m)
      return v;
    return (v << 1) - m + f(1);
  }

  [[nodiscard]] u32 byte_pos_ceil() const noexcept {
    return nx::cast<u32>((m_pos + 7) >> 3);
  }
  [[nodiscard]] bool overrun() const noexcept { return m_overrun; }

  [[nodiscard]] static u32 floor_log2(u32 x) noexcept {
    u32 s = 0;
    while (x != 0) {
      x >>= 1;
      ++s;
    }
    return s - 1;
  }

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
      out.equal_picture_interval = equal != 0;
      if (equal != 0)
        out.timing.num_ticks_per_picture_minus_1 = r.uvlc();
      sh.pTimingInfo = &out.timing;

      decoder_model = r.f(1) != 0;
      out.decoder_model_info_present = decoder_model;
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

  out.order_hint_bits =
      f.enable_order_hint != 0 ? nx::cast<u8>(sh.order_hint_bits_minus_1 + 1) : 0;

  if (r.overrun())
    return false;
  out.valid = true;
  return true;
}

// --- Frame header (spec 5.9) -------------------------------------------------

constexpr u32 NUM_REF_FRAMES = 8;
constexpr u32 REFS_PER_FRAME = 7;
constexpr u32 TOTAL_REFS_PER_FRAME = 8;
constexpr u32 PRIMARY_REF_NONE = 7;
constexpr u32 REFRESH_ALL = 0xFF;

constexpr u32 KEY_FRAME = 0;
constexpr u32 INTER_FRAME = 1;
constexpr u32 INTRA_ONLY_FRAME = 2;
constexpr u32 SWITCH_FRAME = 3;

constexpr u32 LAST_FRAME = 1;
constexpr u32 LAST2_FRAME = 2;
constexpr u32 LAST3_FRAME = 3;
constexpr u32 GOLDEN_FRAME = 4;
constexpr u32 BWDREF_FRAME = 5;
constexpr u32 ALTREF2_FRAME = 6;
constexpr u32 ALTREF_FRAME = 7;

constexpr u32 SUPERRES_NUM = 8;
constexpr u32 SUPERRES_DENOM_MIN = 9;
constexpr u32 SUPERRES_DENOM_BITS = 3;

constexpr u32 MAX_TILE_WIDTH = 4096;
constexpr u32 MAX_TILE_AREA = 4096 * 2304;
constexpr u32 MAX_TILE_COLS = 64;
constexpr u32 MAX_TILE_ROWS = 64;

constexpr u32 MAX_SEGMENTS = 8;
constexpr u32 SEG_LVL_ALT_Q = 0;
constexpr u32 SEG_LVL_MAX = 8;
constexpr u32 MAX_LOOP_FILTER = 63;
constexpr u32 RESTORATION_TILESIZE_MAX = 256;

constexpr u32 GM_IDENTITY = 0;
constexpr u32 GM_TRANSLATION = 1;
constexpr u32 GM_ROTZOOM = 2;
constexpr u32 GM_AFFINE = 3;
constexpr u32 WARPEDMODEL_PREC_BITS = 16;
constexpr u32 GM_ABS_TRANS_BITS = 12;
constexpr u32 GM_ABS_TRANS_ONLY_BITS = 9;
constexpr u32 GM_ABS_ALPHA_BITS = 12;
constexpr u32 GM_ALPHA_PREC_BITS = 15;
constexpr u32 GM_TRANS_PREC_BITS = 6;
constexpr u32 GM_TRANS_ONLY_PREC_BITS = 3;

[[nodiscard]] i32 clip3(const i32 lo, const i32 hi, const i32 v) noexcept {
  return v < lo ? lo : (v > hi ? hi : v);
}

class FrameHeaderParser {
public:
  FrameHeaderParser(BitReader &r, const Av1SequenceHeader &seq,
                    const Av1RefState &refs, Av1FrameHeader &out) noexcept
      : m_r(r), m_seq(seq), m_refs(refs), m_out(out) {}

  [[nodiscard]] bool parse() {
    const StdVideoAV1SequenceHeader &sh = m_seq.header;
    const StdVideoAV1SequenceHeaderFlags &sf = sh.flags;
    m_num_planes = m_seq.color.flags.mono_chrome != 0 ? 1 : 3;
    for (u32 i = 0; i < NUM_REF_FRAMES; ++i) {
      m_ref_valid[i] = m_refs.valid[i];
      m_ref_order_hint[i] = m_refs.order_hint[i];
    }

    StdVideoDecodeAV1PictureInfo &pic = m_out.picture;
    StdVideoDecodeAV1PictureInfoFlags &pf = pic.flags;

    const u32 id_len =
        sf.frame_id_numbers_present_flag != 0
            ? nx::cast<u32>(sh.additional_frame_id_length_minus_1) +
                  nx::cast<u32>(sh.delta_frame_id_length_minus_2) + 3
            : 0;

    u32 frame_type = KEY_FRAME;
    m_show_frame = 1;
    m_error_resilient = 0;
    m_out.showable_frame = false;

    if (sf.reduced_still_picture_header != 0) {
      frame_type = KEY_FRAME;
      m_frame_is_intra = true;
    } else {
      if (m_r.f(1) != 0) { // show_existing_frame
        m_out.show_existing_frame = true;
        m_out.frame_to_show_map_idx = nx::cast<u8>(m_r.f(3));
        if (sf.frame_id_numbers_present_flag != 0)
          (void)m_r.f(id_len); // display_frame_id
        m_out.valid = !m_r.overrun();
        return m_out.valid;
      }
      frame_type = m_r.f(2);
      m_frame_is_intra =
          frame_type == INTRA_ONLY_FRAME || frame_type == KEY_FRAME;
      m_show_frame = m_r.f(1);
      if (m_show_frame != 0)
        m_out.showable_frame = frame_type != KEY_FRAME;
      else
        m_out.showable_frame = m_r.f(1) != 0;
      if (frame_type == SWITCH_FRAME ||
          (frame_type == KEY_FRAME && m_show_frame != 0))
        m_error_resilient = 1;
      else
        m_error_resilient = m_r.f(1);
    }
    pic.frame_type = nx::cast<StdVideoAV1FrameType>(frame_type);
    pf.error_resilient_mode = m_error_resilient;
    m_out.frame_is_intra = m_frame_is_intra;

    if (frame_type == KEY_FRAME && m_show_frame != 0) {
      for (u32 i = 0; i < NUM_REF_FRAMES; ++i) {
        m_ref_valid[i] = false;
        m_ref_order_hint[i] = 0;
      }
    }

    pf.disable_cdf_update = m_r.f(1);
    if (sh.seq_force_screen_content_tools == STD_VIDEO_AV1_SELECT_SCREEN_CONTENT_TOOLS)
      m_allow_screen_content_tools = m_r.f(1);
    else
      m_allow_screen_content_tools = sh.seq_force_screen_content_tools;
    pf.allow_screen_content_tools = m_allow_screen_content_tools;

    if (m_allow_screen_content_tools != 0) {
      if (sh.seq_force_integer_mv == STD_VIDEO_AV1_SELECT_INTEGER_MV)
        m_force_integer_mv = m_r.f(1);
      else
        m_force_integer_mv = sh.seq_force_integer_mv;
    } else {
      m_force_integer_mv = 0;
    }
    if (m_frame_is_intra)
      m_force_integer_mv = 1;
    pf.force_integer_mv = m_force_integer_mv;

    u32 current_frame_id = 0;
    if (sf.frame_id_numbers_present_flag != 0) {
      current_frame_id = m_r.f(id_len);
      mark_ref_frames(id_len, sh);
    }
    pic.current_frame_id = current_frame_id;

    if (frame_type == SWITCH_FRAME)
      m_frame_size_override = 1;
    else if (sf.reduced_still_picture_header != 0)
      m_frame_size_override = 0;
    else
      m_frame_size_override = m_r.f(1);
    pf.frame_size_override_flag = m_frame_size_override;

    m_order_hint = m_r.f(m_seq.order_hint_bits);
    pic.OrderHint = nx::cast<u8>(m_order_hint);

    if (m_frame_is_intra || m_error_resilient != 0)
      m_primary_ref_frame = PRIMARY_REF_NONE;
    else
      m_primary_ref_frame = m_r.f(3);
    pic.primary_ref_frame = nx::cast<u8>(m_primary_ref_frame);

    if (m_seq.decoder_model_info_present)
      return false; // buffer removal times unsupported

    u32 refresh_frame_flags = 0;
    if (frame_type == SWITCH_FRAME ||
        (frame_type == KEY_FRAME && m_show_frame != 0))
      refresh_frame_flags = REFRESH_ALL;
    else
      refresh_frame_flags = m_r.f(8);
    pic.refresh_frame_flags = nx::cast<u8>(refresh_frame_flags);

    if ((!m_frame_is_intra || refresh_frame_flags != REFRESH_ALL) &&
        m_error_resilient != 0 && sf.enable_order_hint != 0) {
      for (u32 i = 0; i < NUM_REF_FRAMES; ++i) {
        const u32 ref_order_hint = m_r.f(m_seq.order_hint_bits);
        if (ref_order_hint != m_ref_order_hint[i])
          m_ref_valid[i] = false;
      }
    }

    for (u32 i = 0; i < NUM_REF_FRAMES; ++i)
      pic.OrderHints[i] = 0;

    if (m_frame_is_intra) {
      frame_size(sh);
      render_size();
      if (m_allow_screen_content_tools != 0 &&
          m_upscaled_width == m_frame_width)
        m_allow_intrabc = m_r.f(1);
    } else {
      u32 frame_refs_short = 0;
      if (sf.enable_order_hint != 0)
        frame_refs_short = m_r.f(1);
      pf.frame_refs_short_signaling = frame_refs_short;
      u32 last_frame_idx = 0;
      u32 gold_frame_idx = 0;
      if (frame_refs_short != 0) {
        last_frame_idx = m_r.f(3);
        gold_frame_idx = m_r.f(3);
        set_frame_refs(last_frame_idx, gold_frame_idx);
      }
      for (u32 i = 0; i < REFS_PER_FRAME; ++i) {
        if (frame_refs_short == 0)
          m_out.ref_frame_idx[i] = nx::cast<u8>(m_r.f(3));
        if (sf.frame_id_numbers_present_flag != 0) {
          const u32 n = nx::cast<u32>(sh.delta_frame_id_length_minus_2) + 2;
          const u32 delta_frame_id = m_r.f(n) + 1;
          pic.expectedFrameId[i] =
              (current_frame_id + (1u << id_len) - delta_frame_id) %
              (1u << id_len);
        }
      }
      for (u32 i = 0; i < REFS_PER_FRAME; ++i) {
        const u32 slot = m_out.ref_frame_idx[i];
        pic.OrderHints[LAST_FRAME + i] = m_ref_order_hint[slot];
      }

      if (m_frame_size_override != 0 && m_error_resilient == 0)
        frame_size_with_refs(sh);
      else {
        frame_size(sh);
        render_size();
      }

      if (m_force_integer_mv != 0)
        m_allow_high_precision_mv = 0;
      else
        m_allow_high_precision_mv = m_r.f(1);
      pf.allow_high_precision_mv = m_allow_high_precision_mv;

      read_interpolation_filter();
      pf.is_motion_mode_switchable = m_r.f(1);
      if (m_error_resilient != 0 || sf.enable_ref_frame_mvs == 0)
        m_use_ref_frame_mvs = 0;
      else
        m_use_ref_frame_mvs = m_r.f(1);
      pf.use_ref_frame_mvs = m_use_ref_frame_mvs;
    }
    pf.allow_intrabc = m_allow_intrabc;

    if (sf.reduced_still_picture_header != 0 || pf.disable_cdf_update != 0)
      pf.disable_frame_end_update_cdf = 1;
    else
      pf.disable_frame_end_update_cdf = m_r.f(1);

    if (m_primary_ref_frame == PRIMARY_REF_NONE)
      setup_past_independence();
    else
      load_previous();

    tile_info(sh);
    quantization_params(sh);
    segmentation_params();
    delta_q_params();
    delta_lf_params();
    compute_lossless();
    loop_filter_params();
    cdef_params(sf);
    lr_params(sf);
    read_tx_mode();
    frame_reference_mode();
    skip_mode_params(sf);

    if (m_frame_is_intra || m_error_resilient != 0 ||
        sf.enable_warped_motion == 0)
      pf.allow_warped_motion = 0;
    else
      pf.allow_warped_motion = m_r.f(1);
    pf.reduced_tx_set = m_r.f(1);

    global_motion_params();
    film_grain_params(sf, frame_type);

    m_out.header_size = m_r.byte_pos_ceil();
    if (m_r.overrun())
      return false;

    finalize_pointers();
    m_out.valid = true;
    return true;
  }

private:
  void frame_size(const StdVideoAV1SequenceHeader &sh) {
    if (m_frame_size_override != 0) {
      m_frame_width =
          m_r.f(nx::cast<u32>(sh.frame_width_bits_minus_1) + 1) + 1;
      m_frame_height =
          m_r.f(nx::cast<u32>(sh.frame_height_bits_minus_1) + 1) + 1;
    } else {
      m_frame_width = nx::cast<u32>(sh.max_frame_width_minus_1) + 1;
      m_frame_height = nx::cast<u32>(sh.max_frame_height_minus_1) + 1;
    }
    superres_params(sh);
    compute_image_size();
  }

  void superres_params(const StdVideoAV1SequenceHeader &sh) {
    u32 use_superres = 0;
    if (sh.flags.enable_superres != 0)
      use_superres = m_r.f(1);
    u32 denom = SUPERRES_NUM;
    if (use_superres != 0) {
      const u32 coded_denom = m_r.f(SUPERRES_DENOM_BITS);
      denom = coded_denom + SUPERRES_DENOM_MIN;
      m_out.picture.coded_denom = nx::cast<u8>(coded_denom);
    }
    m_out.picture.flags.use_superres = use_superres;
    m_upscaled_width = m_frame_width;
    m_frame_width =
        (m_upscaled_width * SUPERRES_NUM + (denom / 2)) / denom;
  }

  void compute_image_size() {
    m_out.mi_cols = 2 * ((m_frame_width + 7) >> 3);
    m_out.mi_rows = 2 * ((m_frame_height + 7) >> 3);
  }

  void render_size() {
    if (m_r.f(1) != 0) { // render_and_frame_size_different
      m_render_width = m_r.f(16) + 1;
      m_render_height = m_r.f(16) + 1;
      m_out.picture.flags.render_and_frame_size_different = 1;
    } else {
      m_render_width = m_upscaled_width;
      m_render_height = m_frame_height;
    }
  }

  void frame_size_with_refs(const StdVideoAV1SequenceHeader &sh) {
    bool found_ref = false;
    for (u32 i = 0; i < REFS_PER_FRAME; ++i) {
      if (m_r.f(1) != 0) {
        const u32 slot = m_out.ref_frame_idx[i];
        m_upscaled_width = m_refs.upscaled_width[slot];
        m_frame_width = m_upscaled_width;
        m_frame_height = m_refs.frame_height[slot];
        m_render_width = m_refs.render_width[slot];
        m_render_height = m_refs.render_height[slot];
        found_ref = true;
        break;
      }
    }
    if (!found_ref) {
      frame_size(sh);
      render_size();
    } else {
      superres_params(sh);
      compute_image_size();
    }
  }

  void read_interpolation_filter() {
    StdVideoDecodeAV1PictureInfo &pic = m_out.picture;
    if (m_r.f(1) != 0) { // is_filter_switchable
      pic.interpolation_filter = STD_VIDEO_AV1_INTERPOLATION_FILTER_SWITCHABLE;
      pic.flags.is_filter_switchable = 1;
    } else {
      pic.interpolation_filter =
          nx::cast<StdVideoAV1InterpolationFilter>(m_r.f(2));
    }
  }

  static u32 tile_log2(const u32 blk_size, const u32 target) {
    u32 k = 0;
    while ((blk_size << k) < target)
      ++k;
    return k;
  }

  void tile_info(const StdVideoAV1SequenceHeader &sh) {
    StdVideoAV1TileInfo &ti = m_out.tile_info;
    const bool sb128 = sh.flags.use_128x128_superblock != 0;
    const u32 sb_shift = sb128 ? 5u : 4u;
    const u32 sb_size = sb_shift + 2;
    const u32 sb_cols =
        sb128 ? ((m_out.mi_cols + 31) >> 5) : ((m_out.mi_cols + 15) >> 4);
    const u32 sb_rows =
        sb128 ? ((m_out.mi_rows + 31) >> 5) : ((m_out.mi_rows + 15) >> 4);
    const u32 max_tile_width_sb = MAX_TILE_WIDTH >> sb_size;
    const u32 max_tile_area_sb = MAX_TILE_AREA >> (2 * sb_size);
    const u32 min_log2_tile_cols = tile_log2(max_tile_width_sb, sb_cols);
    const u32 max_log2_tile_cols =
        tile_log2(1, nx::min(sb_cols, MAX_TILE_COLS));
    const u32 max_log2_tile_rows =
        tile_log2(1, nx::min(sb_rows, MAX_TILE_ROWS));
    const u32 min_log2_tiles = nx::max(
        min_log2_tile_cols, tile_log2(max_tile_area_sb, sb_rows * sb_cols));

    u32 tile_cols_log2 = 0;
    u32 tile_rows_log2 = 0;
    const bool uniform = m_r.f(1) != 0;
    ti.flags.uniform_tile_spacing_flag = uniform ? 1u : 0u;

    u32 tile_cols = 0;
    u32 tile_rows = 0;
    if (uniform) {
      tile_cols_log2 = min_log2_tile_cols;
      while (tile_cols_log2 < max_log2_tile_cols) {
        if (m_r.f(1) != 0)
          ++tile_cols_log2;
        else
          break;
      }
      const u32 tile_width_sb =
          (sb_cols + (1u << tile_cols_log2) - 1) >> tile_cols_log2;
      u32 i = 0;
      for (u32 start = 0; start < sb_cols; start += tile_width_sb, ++i)
        m_out.mi_col_starts[i] = nx::cast<u16>(start << sb_shift);
      m_out.mi_col_starts[i] = nx::cast<u16>(m_out.mi_cols);
      tile_cols = i;

      const u32 min_log2_tile_rows =
          min_log2_tiles > tile_cols_log2 ? min_log2_tiles - tile_cols_log2 : 0;
      tile_rows_log2 = min_log2_tile_rows;
      while (tile_rows_log2 < max_log2_tile_rows) {
        if (m_r.f(1) != 0)
          ++tile_rows_log2;
        else
          break;
      }
      const u32 tile_height_sb =
          (sb_rows + (1u << tile_rows_log2) - 1) >> tile_rows_log2;
      u32 j = 0;
      for (u32 start = 0; start < sb_rows; start += tile_height_sb, ++j)
        m_out.mi_row_starts[j] = nx::cast<u16>(start << sb_shift);
      m_out.mi_row_starts[j] = nx::cast<u16>(m_out.mi_rows);
      tile_rows = j;
    } else {
      u32 widest_tile_sb = 0;
      u32 start = 0;
      u32 i = 0;
      for (; start < sb_cols; ++i) {
        m_out.mi_col_starts[i] = nx::cast<u16>(start << sb_shift);
        const u32 max_width = nx::min(sb_cols - start, max_tile_width_sb);
        const u32 size_sb = m_r.ns(max_width) + 1;
        widest_tile_sb = nx::max(size_sb, widest_tile_sb);
        start += size_sb;
      }
      m_out.mi_col_starts[i] = nx::cast<u16>(m_out.mi_cols);
      tile_cols = i;
      tile_cols_log2 = tile_log2(1, tile_cols);

      u32 max_tile_area_sb2 =
          min_log2_tiles > 0 ? (sb_rows * sb_cols) >> (min_log2_tiles + 1)
                             : sb_rows * sb_cols;
      const u32 max_tile_height_sb = nx::max(max_tile_area_sb2 / widest_tile_sb, 1u);
      start = 0;
      u32 j = 0;
      for (; start < sb_rows; ++j) {
        m_out.mi_row_starts[j] = nx::cast<u16>(start << sb_shift);
        const u32 max_height = nx::min(sb_rows - start, max_tile_height_sb);
        const u32 size_sb = m_r.ns(max_height) + 1;
        start += size_sb;
      }
      m_out.mi_row_starts[j] = nx::cast<u16>(m_out.mi_rows);
      tile_rows = j;
      tile_rows_log2 = tile_log2(1, tile_rows);
    }

    for (u32 c = 0; c < tile_cols; ++c)
      m_out.width_in_sbs_minus_1[c] = nx::cast<u16>(
          ((m_out.mi_col_starts[c + 1] - m_out.mi_col_starts[c]) >> sb_shift) -
          1);
    for (u32 rr = 0; rr < tile_rows; ++rr)
      m_out.height_in_sbs_minus_1[rr] = nx::cast<u16>(
          ((m_out.mi_row_starts[rr + 1] - m_out.mi_row_starts[rr]) >> sb_shift) -
          1);

    if (tile_cols_log2 > 0 || tile_rows_log2 > 0) {
      ti.context_update_tile_id =
          nx::cast<u16>(m_r.f(tile_rows_log2 + tile_cols_log2));
      m_out.tile_size_bytes = nx::cast<u8>(m_r.f(2) + 1);
      ti.tile_size_bytes_minus_1 = nx::cast<u8>(m_out.tile_size_bytes - 1);
    } else {
      ti.context_update_tile_id = 0;
      m_out.tile_size_bytes = 1;
      ti.tile_size_bytes_minus_1 = 0;
    }
    ti.TileCols = nx::cast<u8>(tile_cols);
    ti.TileRows = nx::cast<u8>(tile_rows);
    m_tile_cols_log2 = tile_cols_log2;
    m_tile_rows_log2 = tile_rows_log2;
  }

  i8 read_delta_q() {
    if (m_r.f(1) != 0)
      return nx::cast<i8>(m_r.su(7));
    return 0;
  }

  void quantization_params(const StdVideoAV1SequenceHeader &sh) {
    (void)sh;
    StdVideoAV1Quantization &q = m_out.quantization;
    q.base_q_idx = nx::cast<u8>(m_r.f(8));
    q.DeltaQYDc = read_delta_q();
    u32 diff_uv_delta = 0;
    if (m_num_planes > 1) {
      if (m_seq.color.flags.separate_uv_delta_q != 0)
        diff_uv_delta = m_r.f(1);
      q.DeltaQUDc = read_delta_q();
      q.DeltaQUAc = read_delta_q();
      if (diff_uv_delta != 0) {
        q.DeltaQVDc = read_delta_q();
        q.DeltaQVAc = read_delta_q();
      } else {
        q.DeltaQVDc = q.DeltaQUDc;
        q.DeltaQVAc = q.DeltaQUAc;
      }
    }
    q.flags.diff_uv_delta = diff_uv_delta;
    const u32 using_qmatrix = m_r.f(1);
    q.flags.using_qmatrix = using_qmatrix;
    if (using_qmatrix != 0) {
      q.qm_y = nx::cast<u8>(m_r.f(4));
      q.qm_u = nx::cast<u8>(m_r.f(4));
      q.qm_v = m_seq.color.flags.separate_uv_delta_q != 0
                   ? nx::cast<u8>(m_r.f(4))
                   : q.qm_u;
    }
  }

  void segmentation_params() {
    static constexpr u8 feature_bits[SEG_LVL_MAX] = {8, 6, 6, 6, 6, 3, 0, 0};
    static constexpr bool feature_signed[SEG_LVL_MAX] = {
        true, true, true, true, true, false, false, false};
    static constexpr i32 feature_max[SEG_LVL_MAX] = {
        255, MAX_LOOP_FILTER, MAX_LOOP_FILTER, MAX_LOOP_FILTER,
        MAX_LOOP_FILTER, 7, 0, 0};

    StdVideoDecodeAV1PictureInfoFlags &pf = m_out.picture.flags;
    StdVideoAV1Segmentation &seg = m_out.segmentation;
    const u32 enabled = m_r.f(1);
    pf.segmentation_enabled = enabled;
    if (enabled == 0) {
      for (u32 i = 0; i < MAX_SEGMENTS; ++i) {
        seg.FeatureEnabled[i] = 0;
        for (u32 j = 0; j < SEG_LVL_MAX; ++j)
          seg.FeatureData[i][j] = 0;
      }
      return;
    }

    u32 update_map = 1;
    u32 temporal_update = 0;
    u32 update_data = 1;
    if (m_primary_ref_frame != PRIMARY_REF_NONE) {
      update_map = m_r.f(1);
      if (update_map != 0)
        temporal_update = m_r.f(1);
      update_data = m_r.f(1);
    }
    pf.segmentation_update_map = update_map;
    pf.segmentation_temporal_update = temporal_update;
    pf.segmentation_update_data = update_data;

    if (update_data != 0) {
      for (u32 i = 0; i < MAX_SEGMENTS; ++i) {
        seg.FeatureEnabled[i] = 0;
        for (u32 j = 0; j < SEG_LVL_MAX; ++j) {
          i32 clipped = 0;
          if (m_r.f(1) != 0) {
            seg.FeatureEnabled[i] |= nx::cast<u8>(1u << j);
            if (feature_signed[j]) {
              const i32 value = m_r.su(1 + feature_bits[j]);
              clipped = clip3(-feature_max[j], feature_max[j], value);
            } else {
              const i32 value = nx::cast<i32>(m_r.f(feature_bits[j]));
              clipped = clip3(0, feature_max[j], value);
            }
          }
          seg.FeatureData[i][j] = nx::cast<i16>(clipped);
        }
      }
    }
  }

  void delta_q_params() {
    StdVideoDecodeAV1PictureInfo &pic = m_out.picture;
    u32 delta_q_present = 0;
    if (m_out.quantization.base_q_idx > 0)
      delta_q_present = m_r.f(1);
    pic.flags.delta_q_present = delta_q_present;
    if (delta_q_present != 0)
      pic.delta_q_res = nx::cast<u8>(m_r.f(2));
    m_delta_q_present = delta_q_present;
  }

  void delta_lf_params() {
    StdVideoDecodeAV1PictureInfo &pic = m_out.picture;
    if (m_delta_q_present != 0) {
      if (m_allow_intrabc == 0)
        pic.flags.delta_lf_present = m_r.f(1);
      if (pic.flags.delta_lf_present != 0) {
        pic.delta_lf_res = nx::cast<u8>(m_r.f(2));
        pic.flags.delta_lf_multi = m_r.f(1);
      }
    }
  }

  [[nodiscard]] bool seg_feature_active(const u32 seg_id,
                                        const u32 feature) const {
    return m_out.picture.flags.segmentation_enabled != 0 &&
           (m_out.segmentation.FeatureEnabled[seg_id] & (1u << feature)) != 0;
  }

  [[nodiscard]] i32 get_qindex(const u32 seg_id) const {
    const u32 base = m_out.quantization.base_q_idx;
    if (seg_feature_active(seg_id, SEG_LVL_ALT_Q)) {
      const i32 data = m_out.segmentation.FeatureData[seg_id][SEG_LVL_ALT_Q];
      return clip3(0, 255, nx::cast<i32>(base) + data);
    }
    return nx::cast<i32>(base);
  }

  void compute_lossless() {
    const StdVideoAV1Quantization &q = m_out.quantization;
    bool coded_lossless = true;
    for (u32 seg = 0; seg < MAX_SEGMENTS; ++seg) {
      const i32 qindex = get_qindex(seg);
      const bool lossless = qindex == 0 && q.DeltaQYDc == 0 &&
                            q.DeltaQUAc == 0 && q.DeltaQUDc == 0 &&
                            q.DeltaQVAc == 0 && q.DeltaQVDc == 0;
      if (!lossless)
        coded_lossless = false;
    }
    m_coded_lossless = coded_lossless;
    m_all_lossless = coded_lossless && m_frame_width == m_upscaled_width;
  }

  void set_default_ref_deltas() {
    static constexpr i8 defaults[TOTAL_REFS_PER_FRAME] = {1, 0, 0,  0,
                                                          -1, 0, -1, -1};
    for (u32 i = 0; i < TOTAL_REFS_PER_FRAME; ++i)
      m_lf_ref_deltas[i] = defaults[i];
    m_lf_mode_deltas[0] = 0;
    m_lf_mode_deltas[1] = 0;
  }

  void loop_filter_params() {
    StdVideoAV1LoopFilter &lf = m_out.loop_filter;
    if (m_coded_lossless || m_allow_intrabc != 0) {
      lf.loop_filter_level[0] = 0;
      lf.loop_filter_level[1] = 0;
      set_default_ref_deltas();
      store_ref_deltas();
      return;
    }
    lf.loop_filter_level[0] = nx::cast<u8>(m_r.f(6));
    lf.loop_filter_level[1] = nx::cast<u8>(m_r.f(6));
    if (m_num_planes > 1 &&
        (lf.loop_filter_level[0] != 0 || lf.loop_filter_level[1] != 0)) {
      lf.loop_filter_level[2] = nx::cast<u8>(m_r.f(6));
      lf.loop_filter_level[3] = nx::cast<u8>(m_r.f(6));
    }
    lf.loop_filter_sharpness = nx::cast<u8>(m_r.f(3));
    const u32 delta_enabled = m_r.f(1);
    lf.flags.loop_filter_delta_enabled = delta_enabled;
    if (delta_enabled != 0) {
      const u32 delta_update = m_r.f(1);
      lf.flags.loop_filter_delta_update = delta_update;
      if (delta_update != 0) {
        for (u32 i = 0; i < TOTAL_REFS_PER_FRAME; ++i)
          if (m_r.f(1) != 0)
            m_lf_ref_deltas[i] = nx::cast<i8>(m_r.su(7));
        for (u32 i = 0; i < 2; ++i)
          if (m_r.f(1) != 0)
            m_lf_mode_deltas[i] = nx::cast<i8>(m_r.su(7));
      }
    }
    store_ref_deltas();
  }

  void store_ref_deltas() {
    StdVideoAV1LoopFilter &lf = m_out.loop_filter;
    for (u32 i = 0; i < TOTAL_REFS_PER_FRAME; ++i)
      lf.loop_filter_ref_deltas[i] = m_lf_ref_deltas[i];
    lf.loop_filter_mode_deltas[0] = m_lf_mode_deltas[0];
    lf.loop_filter_mode_deltas[1] = m_lf_mode_deltas[1];
  }

  void cdef_params(const StdVideoAV1SequenceHeaderFlags &sf) {
    StdVideoAV1CDEF &cdef = m_out.cdef;
    if (m_coded_lossless || m_allow_intrabc != 0 || sf.enable_cdef == 0) {
      cdef.cdef_bits = 0;
      cdef.cdef_y_pri_strength[0] = 0;
      cdef.cdef_y_sec_strength[0] = 0;
      cdef.cdef_uv_pri_strength[0] = 0;
      cdef.cdef_uv_sec_strength[0] = 0;
      cdef.cdef_damping_minus_3 = 0;
      return;
    }
    cdef.cdef_damping_minus_3 = nx::cast<u8>(m_r.f(2));
    cdef.cdef_bits = nx::cast<u8>(m_r.f(2));
    const u32 count = 1u << cdef.cdef_bits;
    for (u32 i = 0; i < count; ++i) {
      cdef.cdef_y_pri_strength[i] = nx::cast<u8>(m_r.f(4));
      u32 sec = m_r.f(2);
      if (sec == 3)
        sec += 1;
      cdef.cdef_y_sec_strength[i] = nx::cast<u8>(sec);
      if (m_num_planes > 1) {
        cdef.cdef_uv_pri_strength[i] = nx::cast<u8>(m_r.f(4));
        u32 usec = m_r.f(2);
        if (usec == 3)
          usec += 1;
        cdef.cdef_uv_sec_strength[i] = nx::cast<u8>(usec);
      }
    }
  }

  void lr_params(const StdVideoAV1SequenceHeaderFlags &sf) {
    static constexpr StdVideoAV1FrameRestorationType remap[4] = {
        STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_NONE,
        STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_SWITCHABLE,
        STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_WIENER,
        STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_SGRPROJ};
    StdVideoAV1LoopRestoration &lr = m_out.loop_restoration;
    for (u32 i = 0; i < STD_VIDEO_AV1_MAX_NUM_PLANES; ++i)
      lr.FrameRestorationType[i] = STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_NONE;
    if (m_all_lossless || m_allow_intrabc != 0 || sf.enable_restoration == 0)
      return;

    bool uses_lr = false;
    bool uses_chroma_lr = false;
    for (u32 i = 0; i < m_num_planes; ++i) {
      const u32 lr_type = m_r.f(2);
      lr.FrameRestorationType[i] = remap[lr_type];
      if (lr.FrameRestorationType[i] !=
          STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_NONE) {
        uses_lr = true;
        if (i > 0)
          uses_chroma_lr = true;
      }
    }
    m_out.picture.flags.UsesLr = uses_lr ? 1u : 0u;
    m_out.picture.flags.usesChromaLr = uses_chroma_lr ? 1u : 0u;
    if (!uses_lr)
      return;

    u32 lr_unit_shift = 0;
    if (m_seq.header.flags.use_128x128_superblock != 0) {
      lr_unit_shift = m_r.f(1) + 1;
    } else {
      lr_unit_shift = m_r.f(1);
      if (lr_unit_shift != 0)
        lr_unit_shift += m_r.f(1);
    }
    lr.LoopRestorationSize[0] =
        nx::cast<u16>(RESTORATION_TILESIZE_MAX >> (2 - lr_unit_shift));
    u32 lr_uv_shift = 0;
    if (m_seq.color.subsampling_x != 0 && m_seq.color.subsampling_y != 0 &&
        uses_chroma_lr)
      lr_uv_shift = m_r.f(1);
    lr.LoopRestorationSize[1] =
        nx::cast<u16>(lr.LoopRestorationSize[0] >> lr_uv_shift);
    lr.LoopRestorationSize[2] = lr.LoopRestorationSize[1];
  }

  void read_tx_mode() {
    if (m_coded_lossless)
      m_out.picture.TxMode = STD_VIDEO_AV1_TX_MODE_ONLY_4X4;
    else
      m_out.picture.TxMode = m_r.f(1) != 0 ? STD_VIDEO_AV1_TX_MODE_SELECT
                                           : STD_VIDEO_AV1_TX_MODE_LARGEST;
  }

  void frame_reference_mode() {
    if (m_frame_is_intra)
      m_reference_select = 0;
    else
      m_reference_select = m_r.f(1);
    m_out.picture.flags.reference_select = m_reference_select;
  }

  [[nodiscard]] i32 get_relative_dist(const u32 a, const u32 b) const {
    if (m_seq.header.flags.enable_order_hint == 0)
      return 0;
    i32 diff = nx::cast<i32>(a) - nx::cast<i32>(b);
    const i32 m = 1 << (m_seq.order_hint_bits - 1);
    diff = (diff & (m - 1)) - (diff & m);
    return diff;
  }

  void skip_mode_params(const StdVideoAV1SequenceHeaderFlags &sf) {
    bool skip_mode_allowed = false;
    if (!m_frame_is_intra && m_reference_select != 0 &&
        sf.enable_order_hint != 0) {
      i32 forward_idx = -1;
      i32 backward_idx = -1;
      u32 forward_hint = 0;
      u32 backward_hint = 0;
      for (u32 i = 0; i < REFS_PER_FRAME; ++i) {
        const u32 ref_hint = m_ref_order_hint[m_out.ref_frame_idx[i]];
        if (get_relative_dist(ref_hint, m_order_hint) < 0) {
          if (forward_idx < 0 ||
              get_relative_dist(ref_hint, forward_hint) > 0) {
            forward_idx = nx::cast<i32>(i);
            forward_hint = ref_hint;
          }
        } else if (get_relative_dist(ref_hint, m_order_hint) > 0) {
          if (backward_idx < 0 ||
              get_relative_dist(ref_hint, backward_hint) < 0) {
            backward_idx = nx::cast<i32>(i);
            backward_hint = ref_hint;
          }
        }
      }
      if (forward_idx < 0) {
        skip_mode_allowed = false;
      } else if (backward_idx >= 0) {
        skip_mode_allowed = true;
        const u32 a = nx::cast<u32>(nx::min(forward_idx, backward_idx));
        const u32 b = nx::cast<u32>(nx::max(forward_idx, backward_idx));
        m_out.picture.SkipModeFrame[0] = nx::cast<u8>(LAST_FRAME + a);
        m_out.picture.SkipModeFrame[1] = nx::cast<u8>(LAST_FRAME + b);
      } else {
        i32 second_forward_idx = -1;
        u32 second_forward_hint = 0;
        for (u32 i = 0; i < REFS_PER_FRAME; ++i) {
          const u32 ref_hint = m_ref_order_hint[m_out.ref_frame_idx[i]];
          if (get_relative_dist(ref_hint, forward_hint) < 0) {
            if (second_forward_idx < 0 ||
                get_relative_dist(ref_hint, second_forward_hint) > 0) {
              second_forward_idx = nx::cast<i32>(i);
              second_forward_hint = ref_hint;
            }
          }
        }
        if (second_forward_idx >= 0) {
          skip_mode_allowed = true;
          const u32 a = nx::cast<u32>(nx::min(forward_idx, second_forward_idx));
          const u32 b = nx::cast<u32>(nx::max(forward_idx, second_forward_idx));
          m_out.picture.SkipModeFrame[0] = nx::cast<u8>(LAST_FRAME + a);
          m_out.picture.SkipModeFrame[1] = nx::cast<u8>(LAST_FRAME + b);
        }
      }
    }
    m_out.picture.flags.skip_mode_present =
        skip_mode_allowed ? m_r.f(1) : 0u;
  }

  [[nodiscard]] static u32 inverse_recenter(const u32 r, const u32 v) {
    if (v > 2 * r)
      return v;
    if ((v & 1) != 0)
      return r - ((v + 1) >> 1);
    return r + (v >> 1);
  }

  [[nodiscard]] u32 decode_subexp(const u32 num_syms) {
    u32 i = 0;
    u32 mk = 0;
    const u32 k = 3;
    while (true) {
      const u32 b2 = i != 0 ? k + i - 1 : k;
      const u32 a = 1u << b2;
      if (num_syms <= mk + 3 * a)
        return m_r.ns(num_syms - mk) + mk;
      if (m_r.f(1) != 0) {
        ++i;
        mk += a;
      } else {
        return m_r.f(b2) + mk;
      }
    }
  }

  [[nodiscard]] u32 decode_unsigned_subexp_with_ref(const u32 mx, const u32 r) {
    const u32 v = decode_subexp(mx);
    if ((r << 1) <= mx)
      return inverse_recenter(r, v);
    return mx - 1 - inverse_recenter(mx - 1 - r, v);
  }

  [[nodiscard]] i32 decode_signed_subexp_with_ref(const i32 low, const i32 high,
                                                  const i32 r) {
    const u32 x = decode_unsigned_subexp_with_ref(
        nx::cast<u32>(high - low), nx::cast<u32>(r - low));
    return nx::cast<i32>(x) + low;
  }

  void read_global_param(const u32 type, const u32 ref, const u32 idx) {
    u32 abs_bits = GM_ABS_ALPHA_BITS;
    u32 prec_bits = GM_ALPHA_PREC_BITS;
    if (idx < 2) {
      if (type == GM_TRANSLATION) {
        abs_bits = GM_ABS_TRANS_ONLY_BITS - (m_allow_high_precision_mv == 0);
        prec_bits = GM_TRANS_ONLY_PREC_BITS - (m_allow_high_precision_mv == 0);
      } else {
        abs_bits = GM_ABS_TRANS_BITS;
        prec_bits = GM_TRANS_PREC_BITS;
      }
    }
    const u32 prec_diff = WARPEDMODEL_PREC_BITS - prec_bits;
    const i32 round = (idx % 3) == 2 ? (1 << WARPEDMODEL_PREC_BITS) : 0;
    const i32 sub = (idx % 3) == 2 ? (1 << prec_bits) : 0;
    const i32 mx = 1 << abs_bits;
    const i32 r = (m_prev_gm[ref][idx] >> prec_diff) - sub;
    m_gm_params[ref][idx] =
        (decode_signed_subexp_with_ref(-mx, mx + 1, r) << prec_diff) + round;
  }

  void global_motion_params() {
    StdVideoAV1GlobalMotion &gm = m_out.global_motion;
    for (u32 ref = 0; ref < NUM_REF_FRAMES; ++ref) {
      gm.GmType[ref] = GM_IDENTITY;
      for (u32 i = 0; i < STD_VIDEO_AV1_GLOBAL_MOTION_PARAMS; ++i)
        m_gm_params[ref][i] = (i % 3) == 2 ? (1 << WARPEDMODEL_PREC_BITS) : 0;
    }
    if (!m_frame_is_intra) {
      for (u32 ref = LAST_FRAME; ref <= ALTREF_FRAME; ++ref) {
        u32 type = GM_IDENTITY;
        if (m_r.f(1) != 0) { // is_global
          if (m_r.f(1) != 0)
            type = GM_ROTZOOM;
          else
            type = m_r.f(1) != 0 ? GM_TRANSLATION : GM_AFFINE;
        }
        gm.GmType[ref] = nx::cast<u8>(type);
        if (type >= GM_ROTZOOM) {
          read_global_param(type, ref, 2);
          read_global_param(type, ref, 3);
          if (type == GM_AFFINE) {
            read_global_param(type, ref, 4);
            read_global_param(type, ref, 5);
          } else {
            m_gm_params[ref][4] = -m_gm_params[ref][3];
            m_gm_params[ref][5] = m_gm_params[ref][2];
          }
        }
        if (type >= GM_TRANSLATION) {
          read_global_param(type, ref, 0);
          read_global_param(type, ref, 1);
        }
      }
    }
    for (u32 ref = 0; ref < NUM_REF_FRAMES; ++ref)
      for (u32 i = 0; i < STD_VIDEO_AV1_GLOBAL_MOTION_PARAMS; ++i)
        gm.gm_params[ref][i] = m_gm_params[ref][i];
  }

  void film_grain_params(const StdVideoAV1SequenceHeaderFlags &sf,
                         const u32 frame_type) {
    if (sf.film_grain_params_present == 0 ||
        (m_show_frame == 0 && !m_out.showable_frame))
      return;
    StdVideoAV1FilmGrain &fg = m_out.film_grain;
    const u32 apply_grain = m_r.f(1);
    m_out.picture.flags.apply_grain = apply_grain;
    if (apply_grain == 0)
      return;
    m_out.has_film_grain = true;
    fg.grain_seed = nx::cast<u16>(m_r.f(16));
    u32 update_grain = 1;
    if (frame_type == INTER_FRAME)
      update_grain = m_r.f(1);
    fg.flags.update_grain = update_grain;
    if (update_grain == 0) {
      fg.film_grain_params_ref_idx = nx::cast<u8>(m_r.f(3));
      return;
    }
    fg.num_y_points = nx::cast<u8>(m_r.f(4));
    for (u32 i = 0; i < fg.num_y_points; ++i) {
      fg.point_y_value[i] = nx::cast<u8>(m_r.f(8));
      fg.point_y_scaling[i] = nx::cast<u8>(m_r.f(8));
    }
    u32 chroma_scaling_from_luma = 0;
    if (m_seq.color.flags.mono_chrome == 0)
      chroma_scaling_from_luma = m_r.f(1);
    fg.flags.chroma_scaling_from_luma = chroma_scaling_from_luma;

    if (m_seq.color.flags.mono_chrome != 0 || chroma_scaling_from_luma != 0 ||
        (m_seq.color.subsampling_x == 1 && m_seq.color.subsampling_y == 1 &&
         fg.num_y_points == 0)) {
      fg.num_cb_points = 0;
      fg.num_cr_points = 0;
    } else {
      fg.num_cb_points = nx::cast<u8>(m_r.f(4));
      for (u32 i = 0; i < fg.num_cb_points; ++i) {
        fg.point_cb_value[i] = nx::cast<u8>(m_r.f(8));
        fg.point_cb_scaling[i] = nx::cast<u8>(m_r.f(8));
      }
      fg.num_cr_points = nx::cast<u8>(m_r.f(4));
      for (u32 i = 0; i < fg.num_cr_points; ++i) {
        fg.point_cr_value[i] = nx::cast<u8>(m_r.f(8));
        fg.point_cr_scaling[i] = nx::cast<u8>(m_r.f(8));
      }
    }
    fg.grain_scaling_minus_8 = nx::cast<u8>(m_r.f(2));
    fg.ar_coeff_lag = nx::cast<u8>(m_r.f(2));
    const u32 num_pos_luma = 2 * fg.ar_coeff_lag * (fg.ar_coeff_lag + 1);
    u32 num_pos_chroma = num_pos_luma;
    if (fg.num_y_points > 0) {
      ++num_pos_chroma;
      for (u32 i = 0; i < num_pos_luma; ++i)
        fg.ar_coeffs_y_plus_128[i] = nx::cast<i8>(m_r.f(8) - 128);
    }
    if (chroma_scaling_from_luma != 0 || fg.num_cb_points > 0)
      for (u32 i = 0; i < num_pos_chroma; ++i)
        fg.ar_coeffs_cb_plus_128[i] = nx::cast<i8>(m_r.f(8) - 128);
    if (chroma_scaling_from_luma != 0 || fg.num_cr_points > 0)
      for (u32 i = 0; i < num_pos_chroma; ++i)
        fg.ar_coeffs_cr_plus_128[i] = nx::cast<i8>(m_r.f(8) - 128);
    fg.ar_coeff_shift_minus_6 = nx::cast<u8>(m_r.f(2));
    fg.grain_scale_shift = nx::cast<u8>(m_r.f(2));
    if (fg.num_cb_points > 0) {
      fg.cb_mult = nx::cast<u8>(m_r.f(8));
      fg.cb_luma_mult = nx::cast<u8>(m_r.f(8));
      fg.cb_offset = nx::cast<u16>(m_r.f(9));
    }
    if (fg.num_cr_points > 0) {
      fg.cr_mult = nx::cast<u8>(m_r.f(8));
      fg.cr_luma_mult = nx::cast<u8>(m_r.f(8));
      fg.cr_offset = nx::cast<u16>(m_r.f(9));
    }
    fg.flags.overlap_flag = m_r.f(1);
    fg.flags.clip_to_restricted_range = m_r.f(1);
  }

  void mark_ref_frames(const u32 id_len,
                       const StdVideoAV1SequenceHeader &sh) {
    const u32 diff_len = nx::cast<u32>(sh.delta_frame_id_length_minus_2) + 2;
    const u32 cur = m_out.picture.current_frame_id;
    for (u32 i = 0; i < NUM_REF_FRAMES; ++i) {
      const u32 ref_id = m_refs.frame_id[i];
      if (cur > (1u << diff_len)) {
        if (ref_id > cur || ref_id < cur - (1u << diff_len))
          m_ref_valid[i] = false;
      } else {
        if (ref_id > cur && ref_id < (1u << id_len) + cur - (1u << diff_len))
          m_ref_valid[i] = false;
      }
    }
  }

  void set_frame_refs(const u32 last_frame_idx, const u32 gold_frame_idx) {
    static constexpr u32 ref_frame_list[REFS_PER_FRAME - 2] = {
        LAST2_FRAME, LAST3_FRAME, BWDREF_FRAME, ALTREF2_FRAME, ALTREF_FRAME};
    i32 ref_idx[REFS_PER_FRAME];
    for (u32 i = 0; i < REFS_PER_FRAME; ++i)
      ref_idx[i] = -1;
    ref_idx[LAST_FRAME - LAST_FRAME] = nx::cast<i32>(last_frame_idx);
    ref_idx[GOLDEN_FRAME - LAST_FRAME] = nx::cast<i32>(gold_frame_idx);

    bool used[NUM_REF_FRAMES] = {};
    used[last_frame_idx] = true;
    used[gold_frame_idx] = true;

    const u32 cur_hint = 1u << (m_seq.order_hint_bits - 1);
    i32 shifted[NUM_REF_FRAMES];
    for (u32 i = 0; i < NUM_REF_FRAMES; ++i)
      shifted[i] = nx::cast<i32>(cur_hint) +
                   get_relative_dist(m_ref_order_hint[i], m_order_hint);

    const auto find_latest_backward = [&]() -> i32 {
      i32 r = -1;
      i32 latest = -1;
      for (u32 i = 0; i < NUM_REF_FRAMES; ++i) {
        const i32 h = shifted[i];
        if (!used[i] && h >= nx::cast<i32>(cur_hint) && (r < 0 || h > latest)) {
          r = nx::cast<i32>(i);
          latest = h;
        }
      }
      return r;
    };
    const auto find_earliest_backward = [&]() -> i32 {
      i32 r = -1;
      i32 earliest = -1;
      for (u32 i = 0; i < NUM_REF_FRAMES; ++i) {
        const i32 h = shifted[i];
        if (!used[i] && h >= nx::cast<i32>(cur_hint) &&
            (r < 0 || h < earliest)) {
          r = nx::cast<i32>(i);
          earliest = h;
        }
      }
      return r;
    };
    const auto find_latest_forward = [&]() -> i32 {
      i32 r = -1;
      i32 latest = -1;
      for (u32 i = 0; i < NUM_REF_FRAMES; ++i) {
        const i32 h = shifted[i];
        if (!used[i] && h < nx::cast<i32>(cur_hint) && (r < 0 || h > latest)) {
          r = nx::cast<i32>(i);
          latest = h;
        }
      }
      return r;
    };

    i32 ref = find_latest_backward();
    if (ref >= 0) {
      ref_idx[ALTREF_FRAME - LAST_FRAME] = ref;
      used[ref] = true;
    }
    ref = find_earliest_backward();
    if (ref >= 0) {
      ref_idx[BWDREF_FRAME - LAST_FRAME] = ref;
      used[ref] = true;
    }
    ref = find_earliest_backward();
    if (ref >= 0) {
      ref_idx[ALTREF2_FRAME - LAST_FRAME] = ref;
      used[ref] = true;
    }
    for (u32 i = 0; i < REFS_PER_FRAME - 2; ++i) {
      const u32 rf = ref_frame_list[i];
      if (ref_idx[rf - LAST_FRAME] < 0) {
        const i32 f = find_latest_forward();
        if (f >= 0) {
          ref_idx[rf - LAST_FRAME] = f;
          used[f] = true;
        }
      }
    }
    i32 earliest_ref = -1;
    i32 earliest_hint = -1;
    for (u32 i = 0; i < NUM_REF_FRAMES; ++i) {
      const i32 h = shifted[i];
      if (earliest_ref < 0 || h < earliest_hint) {
        earliest_ref = nx::cast<i32>(i);
        earliest_hint = h;
      }
    }
    for (u32 i = 0; i < REFS_PER_FRAME; ++i) {
      if (ref_idx[i] < 0)
        ref_idx[i] = earliest_ref;
      m_out.ref_frame_idx[i] = nx::cast<u8>(ref_idx[i]);
    }
  }

  void setup_past_independence() {
    for (u32 i = 0; i < MAX_SEGMENTS; ++i) {
      m_out.segmentation.FeatureEnabled[i] = 0;
      for (u32 j = 0; j < SEG_LVL_MAX; ++j)
        m_out.segmentation.FeatureData[i][j] = 0;
    }
    for (u32 ref = 0; ref < NUM_REF_FRAMES; ++ref)
      for (u32 i = 0; i < STD_VIDEO_AV1_GLOBAL_MOTION_PARAMS; ++i)
        m_prev_gm[ref][i] = (i % 3) == 2 ? (1 << WARPEDMODEL_PREC_BITS) : 0;
    set_default_ref_deltas();
  }

  void load_previous() {
    const u32 prev = m_out.ref_frame_idx[m_primary_ref_frame];
    for (u32 ref = 0; ref < NUM_REF_FRAMES; ++ref)
      for (u32 i = 0; i < STD_VIDEO_AV1_GLOBAL_MOTION_PARAMS; ++i)
        m_prev_gm[ref][i] = m_refs.gm_params[prev][ref][i];
    const StdVideoAV1LoopFilter &plf = m_refs.loop_filter[prev];
    for (u32 i = 0; i < TOTAL_REFS_PER_FRAME; ++i)
      m_lf_ref_deltas[i] = plf.loop_filter_ref_deltas[i];
    m_lf_mode_deltas[0] = plf.loop_filter_mode_deltas[0];
    m_lf_mode_deltas[1] = plf.loop_filter_mode_deltas[1];
    m_out.segmentation = m_refs.segmentation[prev];
  }

  void finalize_pointers() {
    StdVideoDecodeAV1PictureInfo &pic = m_out.picture;
    m_out.tile_info.pMiColStarts = m_out.mi_col_starts;
    m_out.tile_info.pMiRowStarts = m_out.mi_row_starts;
    m_out.tile_info.pWidthInSbsMinus1 = m_out.width_in_sbs_minus_1;
    m_out.tile_info.pHeightInSbsMinus1 = m_out.height_in_sbs_minus_1;
    pic.pTileInfo = &m_out.tile_info;
    pic.pQuantization = &m_out.quantization;
    pic.pSegmentation = &m_out.segmentation;
    pic.pLoopFilter = &m_out.loop_filter;
    pic.pCDEF = &m_out.cdef;
    pic.pLoopRestoration = &m_out.loop_restoration;
    pic.pGlobalMotion = &m_out.global_motion;
    pic.pFilmGrain = m_out.has_film_grain ? &m_out.film_grain : nullptr;

    m_out.frame_width = m_frame_width;
    m_out.frame_height = m_frame_height;
    m_out.upscaled_width = m_upscaled_width;
    m_out.render_width = m_render_width;
    m_out.render_height = m_render_height;
  }

  BitReader &m_r;
  const Av1SequenceHeader &m_seq;
  const Av1RefState &m_refs;
  Av1FrameHeader &m_out;

  u8 m_num_planes = 3;
  bool m_frame_is_intra = false;
  u32 m_show_frame = 1;
  u32 m_error_resilient = 0;
  u32 m_allow_screen_content_tools = 0;
  u32 m_force_integer_mv = 0;
  u32 m_frame_size_override = 0;
  u32 m_primary_ref_frame = PRIMARY_REF_NONE;
  u32 m_order_hint = 0;
  u32 m_allow_intrabc = 0;
  u32 m_allow_high_precision_mv = 0;
  u32 m_use_ref_frame_mvs = 0;
  u32 m_reference_select = 0;
  u32 m_delta_q_present = 0;
  u32 m_tile_cols_log2 = 0;
  u32 m_tile_rows_log2 = 0;

  u32 m_frame_width = 0;
  u32 m_frame_height = 0;
  u32 m_upscaled_width = 0;
  u32 m_render_width = 0;
  u32 m_render_height = 0;

  bool m_coded_lossless = false;
  bool m_all_lossless = false;

  bool m_ref_valid[NUM_REF_FRAMES] = {};
  u8 m_ref_order_hint[NUM_REF_FRAMES] = {};
  i8 m_lf_ref_deltas[TOTAL_REFS_PER_FRAME] = {};
  i8 m_lf_mode_deltas[2] = {};
  i32 m_prev_gm[NUM_REF_FRAMES][STD_VIDEO_AV1_GLOBAL_MOTION_PARAMS] = {};
  i32 m_gm_params[NUM_REF_FRAMES][STD_VIDEO_AV1_GLOBAL_MOTION_PARAMS] = {};
};

constexpr u32 OBU_FRAME_HEADER = 3;
constexpr u32 OBU_FRAME = 6;

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

bool parse_av1_frame_header(const u8 *const data, const u32 size,
                            const Av1SequenceHeader &seq,
                            const Av1RefState &refs, Av1FrameHeader &out) {
  out = Av1FrameHeader{};
  if (!seq.valid || data == nullptr || size == 0)
    return false;

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
      ++p;
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
    if (obu_type == OBU_FRAME_HEADER || obu_type == OBU_FRAME) {
      BitReader r(p, nx::cast<u32>(payload_size));
      FrameHeaderParser parser(r, seq, refs, out);
      return parser.parse();
    }
    p += payload_size;
  }
  return false;
}

} // namespace nxe::rhi
