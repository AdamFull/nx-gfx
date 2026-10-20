#include "nx/gfx/commands.h"
#include "nx/gfx/device.h"
#include "nx/gfx/frame.h"
#include "nx/gfx/native.h"
#include "nx/gfx/pipelines.h"
#include "nx/gfx/resources.h"
#include "nx/gfx/upload.h"
#include "rendering/rhi/device.h"

namespace nx::gfx {

nx::result<device *, device_init_error> create_device(const device_desc &desc) {
  nx::unique_ptr<device> dev = nx::make_unique<device>();
  if (!dev->init(desc))
    return nx::err(device_init_error{.report = dev->init_report(),
                                     .message = nx::string(dev->init_error())});
  return dev.release();
}

void destroy_device(device *dev) noexcept {
  const nx::unique_ptr<device> owned = nx::adopt_unique(dev);
}

const device_caps &caps(const device &dev) noexcept { return dev.caps(); }

gpu_device_status device_status(const device &dev) noexcept {
  return dev.device_status();
}

u32 frames_in_flight(const device &dev) noexcept {
  return dev.frames_in_flight();
}

bool supports(const device &dev, const format fmt,
              const texture_usage_flags usage) {
  return dev.supports(fmt, usage);
}

void flush_uploads(device &dev) noexcept { dev.flush_uploads(); }
void wait_idle(device &dev) noexcept { dev.wait_idle(); }
void wait_submitted(device &dev) noexcept { dev.wait_submitted(); }
bool save_pipeline_cache(device &dev) noexcept {
  return dev.save_pipeline_cache();
}

void hold_retirements(device &dev, const u64 serial) noexcept {
  dev.hold_retirements(serial);
}

void release_retirements(device &dev, const u64 serial) noexcept {
  dev.release_retirements(serial);
}

// Resources.

texture_handle create_texture(device &dev, const texture_desc &desc) {
  return dev.create_texture(desc);
}

void destroy_texture(device &dev, const texture_handle texture) noexcept {
  dev.destroy_texture(texture);
}

bool replace_texture(device &dev, const texture_handle target,
                     const texture_handle replacement) {
  return dev.replace_texture(target, replacement);
}

bool set_texture_view_base_mip(device &dev, const texture_handle texture,
                               const u32 base_mip) {
  return dev.set_texture_view_base_mip(texture, base_mip);
}

bool supports_mip_streaming(const device &dev) noexcept {
  return dev.supports_mip_streaming();
}

texture_usage_flags usage(const device &dev,
                          const texture_handle texture) noexcept {
  return dev.texture_usage(texture);
}

resource_state texture_state(const device &dev,
                             const texture_handle texture) noexcept {
  return dev.texture_state(texture);
}

void set_texture_state(device &dev, const texture_handle texture,
                       const resource_state state) noexcept {
  dev.set_texture_state(texture, state);
}

buffer_handle create_buffer(device &dev, const buffer_desc &desc) {
  return dev.create_buffer(desc);
}

void destroy_buffer(device &dev, const buffer_handle buffer) noexcept {
  dev.destroy_buffer(buffer);
}

sampler_handle create_sampler(device &dev, const sampler_desc &desc) {
  return dev.create_sampler(desc);
}

void destroy_sampler(device &dev, const sampler_handle sampler) noexcept {
  dev.destroy_sampler(sampler);
}

memory_requirements texture_requirements(const device &dev,
                                         const texture_desc &desc) {
  return dev.texture_requirements(desc);
}

heap_handle create_heap(device &dev, const heap_desc &desc) {
  return dev.create_heap(desc);
}

void destroy_heap(device &dev, const heap_handle heap) noexcept {
  dev.destroy_heap(heap);
}

texture_handle create_texture_in_heap(device &dev, const texture_desc &desc,
                                      const heap_handle heap,
                                      const u64 offset) {
  return dev.create_texture_in_heap(desc, heap, offset);
}

query_pool_handle create_timestamp_pool(device &dev, const u32 count) {
  return dev.create_timestamp_pool(count);
}

void destroy_query_pool(device &dev, const query_pool_handle pool) noexcept {
  dev.destroy_query_pool(pool);
}

bool read_timestamps(const device &dev, const query_pool_handle pool,
                     const std::span<u64> out) {
  return dev.read_timestamps(pool, out);
}

u32 texture_index(const device &dev, const texture_handle texture) noexcept {
  return dev.texture_index(texture);
}

u32 rw_texture_index(const device &dev, const texture_handle texture,
                     const u32 mip) noexcept {
  return dev.rw_texture_index(texture, mip);
}

u32 sampler_index(const device &dev, const sampler_handle sampler) noexcept {
  return dev.sampler_index(sampler);
}

u64 buffer_address(const device &dev, const buffer_handle buffer) noexcept {
  return dev.buffer_address(buffer);
}

void *buffer_mapped(const device &dev, const buffer_handle buffer) noexcept {
  return dev.buffer_mapped(buffer);
}

void flush_buffer(device &dev, const buffer_handle buffer, const u64 offset,
                  const u64 size) noexcept {
  dev.flush_buffer(buffer, offset, size);
}

void invalidate_buffer(device &dev, const buffer_handle buffer,
                       const u64 offset, const u64 size) noexcept {
  dev.invalidate_buffer(buffer, offset, size);
}

// Shaders and pipelines.

shader_handle create_shader(device &dev, const shader_desc &desc) {
  return dev.create_shader(desc);
}

void destroy_shader(device &dev, const shader_handle shader) noexcept {
  dev.destroy_shader(shader);
}

const shader_reflection &reflection(const device &dev,
                                    const shader_handle shader) noexcept {
  return dev.reflection(shader);
}

pipeline_job prepare_pipeline(device &dev, const graphics_pipeline_desc &desc) {
  return dev.prepare_pipeline(desc);
}

pipeline_job prepare_pipeline(device &dev, const compute_pipeline_desc &desc) {
  return dev.prepare_pipeline(desc);
}

pipeline_handle publish_pipeline(device &dev, pipeline_job &&job) {
  return dev.publish_pipeline(std::move(job));
}

void destroy_pipeline(device &dev, const pipeline_handle pipeline) noexcept {
  dev.destroy_pipeline(pipeline);
}

nx::workload_statistics
pipeline_compile_statistics(const device &dev) noexcept {
  return dev.pipeline_compile_statistics();
}

// Swapchains and frames.

swapchain_handle create_swapchain(device &dev, const swapchain_desc &desc) {
  return dev.create_swapchain(desc);
}

swapchain_handle create_offscreen_swapchain(device &dev, const u32 width,
                                            const u32 height,
                                            const format fmt) {
  return dev.create_offscreen_swapchain(width, height, fmt);
}

bool resize_swapchain(device &dev, const swapchain_handle swapchain,
                      const u32 width, const u32 height) {
  return dev.resize_swapchain(swapchain, width, height);
}

void destroy_swapchain(device &dev, const swapchain_handle swapchain) noexcept {
  dev.destroy_swapchain(swapchain);
}

bool offscreen_swapchain(const device &dev, const swapchain_handle swapchain) {
  return dev.offscreen_swapchain(swapchain);
}

format swapchain_format(const device &dev) noexcept {
  return dev.swapchain_format();
}

extent2d swapchain_extent(const device &dev, const swapchain_handle swapchain) {
  return dev.swapchain_extent(swapchain);
}

surface_alpha swapchain_alpha(const device &dev,
                              const swapchain_handle swapchain) {
  return dev.swapchain_alpha(swapchain);
}

bool swapchain_stale(const device &dev, const swapchain_handle swapchain) {
  return dev.swapchain_stale(swapchain);
}

command_list *begin_frame(device &dev) {
  return dev.begin_frame(dev.m_frame) ? &dev.m_frame : nullptr;
}

acquire_result acquire(device &dev, const swapchain_handle swapchain) {
  return dev.acquire(swapchain);
}

texture_handle current_backbuffer(const device &dev,
                                  const swapchain_handle swapchain) noexcept {
  return dev.current_backbuffer(swapchain);
}

bool end_frame(device &dev) { return dev.end_frame(); }

command_list *begin_headless_frame(device &dev) {
  return dev.begin_headless_frame(dev.m_frame) ? &dev.m_frame : nullptr;
}

bool end_headless_frame(device &dev) { return dev.end_headless_frame(); }

recording_split split_frame(device &dev, command_list &frame, const u32 count) {
  const recording_split split = dev.split_frame(frame, count);
  // On the owner thread, before any of the split's recordings begins: the
  // lists of an earlier split have all ended by now.
  const usize needed = nx::cast<usize>(split.first) + split.count;
  while (dev.m_recordings.size() < needed)
    dev.m_recordings.push_back(nx::make_unique<command_list>());
  return split;
}

command_list *begin_recording(device &dev, const recording_split split,
                              const u32 index) {
  const usize at = nx::cast<usize>(split.first) + index;
  if (index >= split.count || at >= dev.m_recordings.size())
    return nullptr;
  command_list &list = *dev.m_recordings[at];
  return dev.begin_recording(split, index, list) ? &list : nullptr;
}

void end_recording(device &dev, command_list &list) { dev.end_recording(list); }

// Commands.

void begin_render_pass(command_list &cmd, const render_pass_desc &desc) {
  cmd.begin_render_pass(desc);
}

void end_render_pass(command_list &cmd) { cmd.end_render_pass(); }

void begin_compute_pass(command_list &cmd, const nx::string_view name) {
  cmd.begin_compute_pass(name);
}

void end_compute_pass(command_list &cmd) { cmd.end_compute_pass(); }

void set_viewport(command_list &cmd, const viewport &vp) {
  cmd.set_viewport(vp);
}

void set_scissor(command_list &cmd, const rect2d &rect) {
  cmd.set_scissor(rect);
}

void set_cull_mode(command_list &cmd, const cull_mode cull) {
  cmd.set_cull_mode(cull);
}

void set_front_face(command_list &cmd, const front_face face) {
  cmd.set_front_face(face);
}

void reset_queries(command_list &cmd, const query_pool_handle pool,
                   const u32 first, const u32 count) {
  cmd.reset_queries(pool, first, count);
}

void write_timestamp(command_list &cmd, const query_pool_handle pool,
                     const u32 index) {
  cmd.write_timestamp(pool, index);
}

void bind_pipeline(command_list &cmd, const pipeline_handle pipeline) {
  cmd.bind_pipeline(pipeline);
}

void push_constants(command_list &cmd, const void *data, const u32 size,
                    const u32 offset) {
  cmd.push_constants(data, size, offset);
}

void push_constants(command_list &cmd, const std::span<const std::byte> bytes,
                    const u32 offset) {
  cmd.push_constants(bytes.data(), nx::cast<u32>(bytes.size()), offset);
}

void bind_index_buffer(command_list &cmd, const buffer_handle buffer,
                       const index_type type, const u64 offset) {
  cmd.bind_index_buffer(buffer, type, offset);
}

void draw(command_list &cmd, const u32 vertex_count, const u32 instance_count,
          const u32 first_vertex, const u32 first_instance) {
  cmd.draw(vertex_count, instance_count, first_vertex, first_instance);
}

void draw_indexed(command_list &cmd, const u32 index_count,
                  const u32 instance_count, const u32 first_index,
                  const i32 vertex_offset, const u32 first_instance) {
  cmd.draw_indexed(index_count, instance_count, first_index, vertex_offset,
                   first_instance);
}

void dispatch(command_list &cmd, const u32 groups_x, const u32 groups_y,
              const u32 groups_z) {
  cmd.dispatch(groups_x, groups_y, groups_z);
}

void draw_indirect(command_list &cmd, const buffer_handle buffer,
                   const u64 offset, const u32 draw_count, const u32 stride) {
  cmd.draw_indirect(buffer, offset, draw_count, stride);
}

void draw_indexed_indirect(command_list &cmd, const buffer_handle buffer,
                           const u64 offset, const u32 draw_count,
                           const u32 stride) {
  cmd.draw_indexed_indirect(buffer, offset, draw_count, stride);
}

void draw_mesh_tasks_indirect(command_list &cmd, const buffer_handle buffer,
                              const u64 offset, const u32 draw_count,
                              const u32 stride) {
  cmd.draw_mesh_tasks_indirect(buffer, offset, draw_count, stride);
}

void dispatch_indirect(command_list &cmd, const buffer_handle buffer,
                       const u64 offset) {
  cmd.dispatch_indirect(buffer, offset);
}

void barrier(command_list &cmd, const texture_barrier *textures,
             const u32 texture_count, const buffer_barrier *buffers,
             const u32 buffer_count, const state_tracking tracking) {
  cmd.barrier(textures, texture_count, buffers, buffer_count, tracking);
}

void barrier(command_list &cmd, const texture_barrier &texture) {
  cmd.barrier(texture);
}

bool copy_texture_to_buffer(command_list &cmd, const texture_handle texture,
                            const buffer_handle buffer, const u64 row_pitch,
                            const u32 mip, const u32 layer) {
  return cmd.copy_texture_to_buffer(texture, buffer, row_pitch, mip, layer);
}

bool copy_buffer(command_list &cmd, const buffer_handle source,
                 const u64 source_offset, const buffer_handle destination,
                 const u64 destination_offset, const u64 size) {
  return cmd.copy_buffer(source, source_offset, destination, destination_offset,
                         size);
}

void push_debug_group(command_list &cmd, const nx::string_view name,
                      const f32 r, const f32 g, const f32 b) {
  cmd.push_debug_group(name, r, g, b);
}

void pop_debug_group(command_list &cmd) { cmd.pop_debug_group(); }

// Uploads.

upload_ticket upload_texture(device &dev, const texture_handle texture,
                             const std::span<const u8> src,
                             const std::span<const image_subresource> layout,
                             const resource_state final_state) {
  return dev.uploader().upload_texture(texture, src, layout, final_state);
}

upload_ticket upload_texture(device &dev, const texture_handle texture,
                             const texture_data &data,
                             const resource_state final_state) {
  return dev.uploader().upload_texture(texture, data, final_state);
}

upload_ticket upload_buffer(device &dev, const buffer_handle buffer,
                            const u64 dst_offset, const std::span<const u8> src,
                            const resource_state final_state) {
  return dev.uploader().upload_buffer(buffer, dst_offset, src, final_state);
}

upload_ticket clear_buffer(device &dev, const buffer_handle buffer,
                           const resource_state final_state) {
  return dev.uploader().clear_buffer(buffer, final_state);
}

readback_result read_buffer(device &dev, const buffer_handle buffer,
                            const u64 offset, const u64 size) {
  return dev.uploader().read_buffer(buffer, offset, size);
}

readback_result read_texture(device &dev, const texture_handle texture,
                             const u32 mip, const u32 layer) {
  return dev.uploader().read_texture(texture, mip, layer);
}

bool upload_complete(device &dev, const upload_ticket ticket) noexcept {
  return dev.uploader().is_complete(ticket);
}

void wait_upload(device &dev, const upload_ticket ticket) noexcept {
  dev.uploader().wait(ticket);
}

void wait_uploads(device &dev) noexcept { dev.uploader().flush(); }

uploader_stats upload_stats(device &dev) noexcept {
  return dev.uploader().stats();
}

// Native access. Metal has none of it yet (nx-gfx.md, section 7).

#if defined(NX_RHI_VULKAN)

native_device native(const device &dev) noexcept { return dev.native(); }

native_texture native(const device &dev,
                      const texture_handle texture) noexcept {
  return dev.native_texture(texture);
}

u32 native_layout(const device &dev, const resource_state state) noexcept {
  return dev.native_layout(state);
}

bool native_extension(const device &dev, const nx::string_view name) noexcept {
  return dev.native_extension(name);
}

void *begin_native(command_list &cmd) noexcept { return cmd.begin_native(); }
void end_native(command_list &cmd) noexcept { cmd.end_native(); }

#else

native_device native(const device &) noexcept {
  return {.api = native_api::Metal};
}

native_texture native(const device &, texture_handle) noexcept { return {}; }
u32 native_layout(const device &, resource_state) noexcept { return 0; }
bool native_extension(const device &, nx::string_view) noexcept {
  return false;
}
void *begin_native(command_list &) noexcept { return nullptr; }
void end_native(command_list &) noexcept {}

#endif

native_queue native(const device &dev,
                    const native_device_client &client) noexcept {
  return dev.native_queue(client);
}

} // namespace nx::gfx
