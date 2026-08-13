#pragma once

/**
 * @file vk_av1.h
 * @brief Parsing an AV1 sequence header into the Std structs a Vulkan Video AV1
 * decode session's parameters need (namespace nxe::rhi).
 */

#include "core/rendering/rhi/vulkan/vk_common.h"

namespace nxe::rhi {

struct Av1SequenceHeader {
  bool valid = false;
  StdVideoAV1SequenceHeader header{};
  StdVideoAV1ColorConfig color{};
  StdVideoAV1TimingInfo timing{};

  // Derived values the frame header needs but the Std header does not carry.
  u8 order_hint_bits = 0; // 0 when enable_order_hint is false
  bool decoder_model_info_present = false;
  bool equal_picture_interval = false;
};

[[nodiscard]] bool parse_av1_sequence_header(const u8 *data, u32 size,
                                             Av1SequenceHeader &out);

[[nodiscard]] bool parse_av1_codec_private(const u8 *data, u32 size,
                                           Av1SequenceHeader &out);

struct Av1RefState {
  bool valid[STD_VIDEO_AV1_NUM_REF_FRAMES] = {};
  u8 frame_type[STD_VIDEO_AV1_NUM_REF_FRAMES] = {};
  u32 frame_id[STD_VIDEO_AV1_NUM_REF_FRAMES] = {};
  u32 upscaled_width[STD_VIDEO_AV1_NUM_REF_FRAMES] = {};
  u32 frame_width[STD_VIDEO_AV1_NUM_REF_FRAMES] = {};
  u32 frame_height[STD_VIDEO_AV1_NUM_REF_FRAMES] = {};
  u32 render_width[STD_VIDEO_AV1_NUM_REF_FRAMES] = {};
  u32 render_height[STD_VIDEO_AV1_NUM_REF_FRAMES] = {};
  u8 order_hint[STD_VIDEO_AV1_NUM_REF_FRAMES] = {};
  u8 saved_order_hints[STD_VIDEO_AV1_NUM_REF_FRAMES]
                      [STD_VIDEO_AV1_NUM_REF_FRAMES] = {};
  StdVideoAV1LoopFilter loop_filter[STD_VIDEO_AV1_NUM_REF_FRAMES] = {};
  StdVideoAV1Segmentation segmentation[STD_VIDEO_AV1_NUM_REF_FRAMES] = {};
  i32 gm_params[STD_VIDEO_AV1_NUM_REF_FRAMES][STD_VIDEO_AV1_NUM_REF_FRAMES]
              [STD_VIDEO_AV1_GLOBAL_MOTION_PARAMS] = {};
};

struct Av1FrameHeader {
  bool valid = false;

  // A frame that only re-shows a decoded reference: no header past the index.
  bool show_existing_frame = false;
  u8 frame_to_show_map_idx = 0;

  StdVideoDecodeAV1PictureInfo picture{};
  StdVideoAV1TileInfo tile_info{};
  StdVideoAV1Quantization quantization{};
  StdVideoAV1Segmentation segmentation{};
  StdVideoAV1LoopFilter loop_filter{};
  StdVideoAV1CDEF cdef{};
  StdVideoAV1LoopRestoration loop_restoration{};
  StdVideoAV1GlobalMotion global_motion{};
  StdVideoAV1FilmGrain film_grain{};
  bool has_film_grain = false;

  u16 mi_col_starts[STD_VIDEO_AV1_MAX_TILE_COLS + 1] = {};
  u16 mi_row_starts[STD_VIDEO_AV1_MAX_TILE_ROWS + 1] = {};
  u16 width_in_sbs_minus_1[STD_VIDEO_AV1_MAX_TILE_COLS] = {};
  u16 height_in_sbs_minus_1[STD_VIDEO_AV1_MAX_TILE_ROWS] = {};

  u32 frame_width = 0;
  u32 frame_height = 0;
  u32 upscaled_width = 0;
  u32 render_width = 0;
  u32 render_height = 0;
  u32 mi_cols = 0;
  u32 mi_rows = 0;

  u8 ref_frame_idx[STD_VIDEO_AV1_REFS_PER_FRAME] = {};
  bool frame_is_intra = false;
  bool show_frame = false;
  bool showable_frame = false;
  u8 tile_size_bytes = 1;
  u8 tile_cols_log2 = 0;
  u8 tile_rows_log2 = 0;

  u32 header_size = 0; // bytes of frame header consumed, byte-rounded up

  // Byte offsets into the OBU run parse_av1_frame_header was given, for the
  // decode command: where the frame header OBU starts, and each tile's data.
  static constexpr u32 kMaxTiles = 64;
  u32 frame_header_offset = 0;
  u32 tile_count = 0;
  u32 tile_offsets[kMaxTiles] = {};
  u32 tile_sizes[kMaxTiles] = {};
};

[[nodiscard]] bool parse_av1_frame_header(const u8 *data, u32 size,
                                          const Av1SequenceHeader &seq,
                                          const Av1RefState &refs,
                                          Av1FrameHeader &out);

} // namespace nxe::rhi
