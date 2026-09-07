
#include "core/rendering/rhi/metal/mtl_internal.h"

#include "core/foundation/platform/filesystem.h"

namespace nxe::rhi {
namespace {

NSURL *archive_url(const nx::string_view path) noexcept {
  if (path.empty())
    return nil;
  return [NSURL fileURLWithPath:to_ns_string(path)];
}

MTLPrimitiveType to_mtl_topology(const PrimitiveTopology topology) noexcept {
  switch (topology) {
  case PrimitiveTopology::TriangleStrip:
    return MTLPrimitiveTypeTriangleStrip;
  case PrimitiveTopology::LineList:
    return MTLPrimitiveTypeLine;
  case PrimitiveTopology::LineStrip:
    return MTLPrimitiveTypeLineStrip;
  case PrimitiveTopology::PointList:
    return MTLPrimitiveTypePoint;
  case PrimitiveTopology::TriangleList:
  default:
    return MTLPrimitiveTypeTriangle;
  }
}

MTLPrimitiveTopologyClass
to_mtl_topology_class(const PrimitiveTopology topology) noexcept {
  switch (topology) {
  case PrimitiveTopology::PointList:
    return MTLPrimitiveTopologyClassPoint;
  case PrimitiveTopology::LineList:
  case PrimitiveTopology::LineStrip:
    return MTLPrimitiveTopologyClassLine;
  default:
    return MTLPrimitiveTopologyClassTriangle;
  }
}

MTLCullMode to_mtl_cull(const CullMode cull) noexcept {
  switch (cull) {
  case CullMode::Front:
    return MTLCullModeFront;
  case CullMode::Back:
    return MTLCullModeBack;
  case CullMode::None:
  default:
    return MTLCullModeNone;
  }
}

MTLWinding to_mtl_winding(const FrontFace face) noexcept {
  return face == FrontFace::Clockwise ? MTLWindingClockwise
                                      : MTLWindingCounterClockwise;
}

/// Translates a BlendMode preset into Metal's factor/op pairs. The RHI names
/// intent rather than factors, so this is the one place the presets are
/// spelled out.
void apply_blend(MTLRenderPipelineColorAttachmentDescriptor *attachment,
                 const BlendState &blend) noexcept {
  attachment.writeMask =
      (blend.write_mask & 0x1 ? MTLColorWriteMaskRed : MTLColorWriteMaskNone) |
      (blend.write_mask & 0x2 ? MTLColorWriteMaskGreen : MTLColorWriteMaskNone) |
      (blend.write_mask & 0x4 ? MTLColorWriteMaskBlue : MTLColorWriteMaskNone) |
      (blend.write_mask & 0x8 ? MTLColorWriteMaskAlpha : MTLColorWriteMaskNone);

  if (!blend.enabled || blend.mode == BlendMode::Opaque) {
    attachment.blendingEnabled = NO;
    return;
  }

  attachment.blendingEnabled = YES;
  attachment.rgbBlendOperation = MTLBlendOperationAdd;
  attachment.alphaBlendOperation = MTLBlendOperationAdd;

  switch (blend.mode) {
  case BlendMode::AlphaBlend:
    attachment.sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    attachment.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    attachment.sourceAlphaBlendFactor = MTLBlendFactorOne;
    attachment.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    break;
  case BlendMode::Premultiplied:
    attachment.sourceRGBBlendFactor = MTLBlendFactorOne;
    attachment.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    attachment.sourceAlphaBlendFactor = MTLBlendFactorOne;
    attachment.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    break;
  case BlendMode::Additive:
    attachment.sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    attachment.destinationRGBBlendFactor = MTLBlendFactorOne;
    attachment.sourceAlphaBlendFactor = MTLBlendFactorZero;
    attachment.destinationAlphaBlendFactor = MTLBlendFactorOne;
    break;
  case BlendMode::PremultipliedAdditive:
    attachment.sourceRGBBlendFactor = MTLBlendFactorOne;
    attachment.destinationRGBBlendFactor = MTLBlendFactorOne;
    attachment.sourceAlphaBlendFactor = MTLBlendFactorOne;
    attachment.destinationAlphaBlendFactor = MTLBlendFactorOne;
    break;
  case BlendMode::Screen:
    attachment.sourceRGBBlendFactor = MTLBlendFactorOne;
    attachment.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceColor;
    attachment.sourceAlphaBlendFactor = MTLBlendFactorOne;
    attachment.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    break;
  case BlendMode::Multiply:
    attachment.sourceRGBBlendFactor = MTLBlendFactorDestinationColor;
    attachment.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    attachment.sourceAlphaBlendFactor = MTLBlendFactorOne;
    attachment.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    break;
  case BlendMode::Opaque:
  default:
    attachment.blendingEnabled = NO;
    break;
  }
}

id<MTLFunction> lookup_function(const MetalDeviceState &state,
                                const ShaderEntry &entry,
                                const nx::string_view pipeline_name,
                                const char *stage) noexcept {
  const MetalShader *shader = state.shaders.try_get(entry.shader);
  if (shader == nullptr || shader->library == nil) {
    nx::loge(log_rhi, "pipeline('{}'): {} shader handle is not live",
             pipeline_name, stage);
    return nil;
  }

  NSString *name = to_ns_string(entry.entry_point);
  id<MTLFunction> function = [shader->library newFunctionWithName:name];
  if (function == nil) {
    nx::loge(log_rhi, "pipeline('{}'): no {} entry point '{}' in the library",
             pipeline_name, stage, entry.entry_point);
  }
  return function;
}

struct MetalGraphicsPipelineJob {
  nx::string name;
  id<MTLDevice> device = nil;
  id<MTLFunction> vertex = nil;
  id<MTLFunction> fragment = nil;
  id<MTLBinaryArchive> archive = nil;
  nx::mutex *compile_lock = nullptr;
  MTLPixelFormat color_formats[MAX_COLOR_ATTACHMENTS] = {};
  BlendState blend[MAX_COLOR_ATTACHMENTS] = {};
  u32 color_count = 0;
  MTLPixelFormat depth_format = MTLPixelFormatInvalid;
  bool depth_has_stencil = false;
  PrimitiveTopology topology = PrimitiveTopology::TriangleList;
  CullMode cull = CullMode::None;
  FrontFace front_face = FrontFace::CounterClockwise;
  bool depth_test = false;
  bool depth_write = false;
  CompareOp depth_compare = CompareOp::LessOrEqual;
  u32 sample_count = 1;
};

struct MetalComputePipelineJob {
  nx::string name;
  id<MTLDevice> device = nil;
  id<MTLFunction> function = nil;
  id<MTLBinaryArchive> archive = nil;
  nx::mutex *compile_lock = nullptr;
  u32 group_size[3] = {1, 1, 1};
};

[[nodiscard]] bool prepare_graphics_pipeline(
    MetalDeviceState &state, const GraphicsPipelineDesc &desc,
    MetalGraphicsPipelineJob &out) {
  MetalGraphicsPipelineJob job;
  job.name = nx::string(desc.name);
  job.device = state.device;
  job.vertex = lookup_function(state, desc.vertex, desc.name, "vertex");
  if (job.vertex == nil)
    return false;
  if (desc.fragment.shader.valid()) {
    job.fragment =
        lookup_function(state, desc.fragment, desc.name, "fragment");
    if (job.fragment == nil)
      return false;
  }
  job.archive = state.pipeline_archive;
  job.compile_lock = &state.pipeline_archive_lock;
  job.color_count = nx::min(desc.color_count, MAX_COLOR_ATTACHMENTS);
  for (u32 i = 0; i < job.color_count; ++i) {
    job.color_formats[i] = to_mtl_format(desc.color_formats[i]);
    if (job.color_formats[i] == MTLPixelFormatInvalid) {
      nx::loge(log_rhi,
               "create_graphics_pipeline('{}'): colour format {} is not "
               "supported by Metal",
               desc.name, to_string(desc.color_formats[i]));
      return false;
    }
    job.blend[i] = desc.blend[i];
  }
  if (desc.depth_format != Format::Unknown) {
    job.depth_format = to_mtl_format(desc.depth_format);
    if (job.depth_format == MTLPixelFormatInvalid) {
      nx::loge(log_rhi,
               "create_graphics_pipeline('{}'): depth format {} is not "
               "supported by Metal",
               desc.name, to_string(desc.depth_format));
      return false;
    }
    job.depth_has_stencil = format_is_stencil(desc.depth_format);
  }
  job.topology = desc.topology;
  job.cull = desc.cull;
  job.front_face = desc.front_face;
  job.depth_test = desc.depth_test;
  job.depth_write = desc.depth_write;
  job.depth_compare = desc.depth_compare;
  job.sample_count = nx::max(desc.sample_count, 1u);
  out = std::move(job);
  return true;
}

[[nodiscard]] bool prepare_compute_pipeline(
    MetalDeviceState &state, const ComputePipelineDesc &desc,
    MetalComputePipelineJob &out) {
  MetalComputePipelineJob job;
  job.name = nx::string(desc.name);
  job.device = state.device;
  job.function = lookup_function(state, desc.compute, desc.name, "compute");
  if (job.function == nil)
    return false;
  job.archive = state.pipeline_archive;
  job.compile_lock = &state.pipeline_archive_lock;
  for (u32 i = 0; i < 3; ++i)
    job.group_size[i] = desc.group_size[i];
  if (job.group_size[0] == 0 || job.group_size[1] == 0 ||
      job.group_size[2] == 0) {
    const MetalShader *const shader =
        state.shaders.try_get(desc.compute.shader);
    if (shader == nullptr ||
        !shader->reflection.local_size(nx::id_string(desc.compute.entry_point),
                                       job.group_size)) {
      nx::loge(log_rhi,
               "create_compute_pipeline('{}'): no group_size given and the "
               "reflection sidecar does not state one for '{}'",
               desc.name, desc.compute.entry_point);
      return false;
    }
  }
  out = std::move(job);
  return true;
}

[[nodiscard]] MetalPipeline
compile_graphics_pipeline(const MetalGraphicsPipelineJob &job) {
  MetalPipeline pipeline;
  if (job.device == nil || job.vertex == nil || job.compile_lock == nullptr)
    return pipeline;

  const nx::scoped_lock<nx::mutex> held(*job.compile_lock);
  @autoreleasepool {
    MTLRenderPipelineDescriptor *info =
        [[MTLRenderPipelineDescriptor alloc] init];
    info.label = to_ns_string(job.name.view());
    info.vertexFunction = job.vertex;
    info.fragmentFunction = job.fragment;
    info.rasterSampleCount = job.sample_count;
    info.inputPrimitiveTopology = to_mtl_topology_class(job.topology);
    for (u32 i = 0; i < job.color_count; ++i) {
      info.colorAttachments[i].pixelFormat = job.color_formats[i];
      apply_blend(info.colorAttachments[i], job.blend[i]);
    }
    if (job.depth_format != MTLPixelFormatInvalid) {
      info.depthAttachmentPixelFormat = job.depth_format;
      if (job.depth_has_stencil)
        info.stencilAttachmentPixelFormat = job.depth_format;
    }
    if (job.archive != nil)
      info.binaryArchives = @[ job.archive ];

    NSError *error = nil;
    pipeline.render =
        [job.device newRenderPipelineStateWithDescriptor:info error:&error];
    if (pipeline.render == nil) {
      nx::loge(log_rhi, "create_graphics_pipeline('{}'): {}", job.name,
               error != nil ? [[error localizedDescription] UTF8String]
                            : "newRenderPipelineState failed");
      return {};
    }
    if (job.archive != nil) {
      NSError *add_error = nil;
      if (![job.archive addRenderPipelineFunctionsWithDescriptor:info
                                                           error:&add_error])
        nx::logd(log_rhi, "pipeline archive: '{}' not cached: {}", job.name,
                 add_error != nil
                     ? [[add_error localizedDescription] UTF8String]
                     : "unknown error");
    }
    if (job.depth_test || job.depth_write) {
      MTLDepthStencilDescriptor *depth =
          [[MTLDepthStencilDescriptor alloc] init];
      depth.label = to_ns_string(job.name.view());
      depth.depthCompareFunction = job.depth_test
                                       ? to_mtl_compare(job.depth_compare)
                                       : MTLCompareFunctionAlways;
      depth.depthWriteEnabled = job.depth_write ? YES : NO;
      pipeline.depth_stencil =
          [job.device newDepthStencilStateWithDescriptor:depth];
    }
  }
  pipeline.topology = to_mtl_topology(job.topology);
  pipeline.cull = to_mtl_cull(job.cull);
  pipeline.winding = to_mtl_winding(job.front_face);
  pipeline.is_compute = false;
  return pipeline;
}

[[nodiscard]] MetalPipeline
compile_compute_pipeline(const MetalComputePipelineJob &job) {
  MetalPipeline pipeline;
  if (job.device == nil || job.function == nil || job.compile_lock == nullptr)
    return pipeline;

  const nx::scoped_lock<nx::mutex> held(*job.compile_lock);
  @autoreleasepool {
    MTLComputePipelineDescriptor *info =
        [[MTLComputePipelineDescriptor alloc] init];
    info.label = to_ns_string(job.name.view());
    info.computeFunction = job.function;
    if (job.archive != nil)
      info.binaryArchives = @[ job.archive ];

    NSError *error = nil;
    pipeline.compute =
        [job.device newComputePipelineStateWithDescriptor:info
                                                  options:MTLPipelineOptionNone
                                               reflection:nullptr
                                                    error:&error];
    if (pipeline.compute == nil) {
      nx::loge(log_rhi, "create_compute_pipeline('{}'): {}", job.name,
               error != nil ? [[error localizedDescription] UTF8String]
                            : "newComputePipelineState failed");
      return {};
    }
    if (job.archive != nil) {
      NSError *add_error = nil;
      if (![job.archive addComputePipelineFunctionsWithDescriptor:info
                                                            error:&add_error])
        nx::logd(log_rhi, "pipeline archive: '{}' not cached: {}", job.name,
                 add_error != nil
                     ? [[add_error localizedDescription] UTF8String]
                     : "unknown error");
    }
    pipeline.threads_per_group =
        MTLSizeMake(job.group_size[0], job.group_size[1], job.group_size[2]);
    const NSUInteger declared = nx::cast<NSUInteger>(job.group_size[0]) *
                                job.group_size[1] * job.group_size[2];
    if (declared > pipeline.compute.maxTotalThreadsPerThreadgroup) {
      nx::loge(log_rhi,
               "create_compute_pipeline('{}'): {} threads per group exceeds "
               "this device's limit of {}",
               job.name, declared,
               pipeline.compute.maxTotalThreadsPerThreadgroup);
      return {};
    }
  }
  pipeline.is_compute = true;
  return pipeline;
}

}

void create_pipeline_archive(MetalDeviceState &state,
                             const nx::string_view path) {
  state.pipeline_archive_path = nx::string(path);
  if (path.empty())
    return;

  @autoreleasepool {
    MTLBinaryArchiveDescriptor *info =
        [[MTLBinaryArchiveDescriptor alloc] init];

    const bool have_existing = nx::fs::exists(path);
    if (have_existing)
      info.url = archive_url(path);

    NSError *error = nil;
    state.pipeline_archive =
        [state.device newBinaryArchiveWithDescriptor:info error:&error];

    if (state.pipeline_archive == nil && have_existing) {
      nx::logd(log_rhi, "pipeline archive is stale or unreadable, starting a "
                        "fresh one");
      info.url = nil;
      error = nil;
      state.pipeline_archive =
          [state.device newBinaryArchiveWithDescriptor:info error:&error];
    }

    if (state.pipeline_archive == nil) {
      nx::logw(log_rhi, "pipeline archive unavailable: {}",
               error != nil ? [[error localizedDescription] UTF8String]
                            : "unknown error");
      return;
    }
    state.pipeline_archive.label = @"nx2d pipelines";

    if (have_existing && info.url != nil)
      nx::logd(log_rhi, "pipeline archive: loaded {}", path);
  }
}

void save_pipeline_archive(MetalDeviceState &state) noexcept {
  if (state.pipeline_archive == nil || state.pipeline_archive_path.empty())
    return;

  const nx::scoped_lock<nx::mutex> held(state.pipeline_archive_lock);

  const nx::string_view parent =
      nx::fs::path::parent_path(state.pipeline_archive_path.view());
  if (!parent.empty() && !nx::fs::exists(parent))
    (void)nx::fs::create_directories(parent);

  @autoreleasepool {
    NSError *error = nil;
    if (![state.pipeline_archive
            serializeToURL:archive_url(state.pipeline_archive_path.view())
                     error:&error]) {
      nx::logd(log_rhi, "pipeline archive not written: {}",
               error != nil ? [[error localizedDescription] UTF8String]
                            : "unknown error");
      return;
    }
  }
  nx::logd(log_rhi, "pipeline archive: saved {}", state.pipeline_archive_path);
}

ShaderHandle MetalDevice::create_shader(const ShaderDesc &desc) {
  if (m_state == nullptr)
    return {};

  if (desc.code == nullptr || desc.code_size == 0) {
    nx::loge(log_rhi, "create_shader('{}'): empty blob", desc.name);
    return {};
  }

  MetalShader shader;
  @autoreleasepool {
    // The blob is the caller's and may go away as soon as this returns, so the
    // dispatch_data must own a copy. DISPATCH_DATA_DESTRUCTOR_DEFAULT is
    // exactly that: it copies the bytes.
    dispatch_data_t data =
        dispatch_data_create(desc.code, nx::cast<size_t>(desc.code_size),
                             nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);

    NSError *error = nil;
    shader.library = [m_state->device newLibraryWithData:data error:&error];
    if (shader.library == nil) {
      nx::loge(log_rhi, "create_shader('{}'): {}", desc.name,
               error != nil ? [[error localizedDescription] UTF8String]
                            : "newLibraryWithData failed");
      return {};
    }
    shader.library.label = to_ns_string(desc.name);
  }

  if (!desc.reflection_json.empty() &&
      !reflect_from_json(desc.reflection_json, shader.reflection))
    nx::logw(log_rhi, "create_shader('{}'): reflection sidecar unusable",
             desc.name);

  return m_state->shaders.insert(std::move(shader));
}

const ShaderReflection &
MetalDevice::reflection(const ShaderHandle handle) const noexcept {
  static const ShaderReflection empty;
  if (m_state == nullptr)
    return empty;
  const MetalShader *shader = m_state->shaders.try_get(handle);
  return shader != nullptr ? shader->reflection : empty;
}

void MetalDevice::destroy_shader(const ShaderHandle handle) noexcept {
  if (m_state == nullptr || !handle.valid())
    return;
  retire(*m_state, {.shader = handle});
}

PipelineHandle
MetalDevice::create_graphics_pipeline(const GraphicsPipelineDesc &desc) {
  if (m_state == nullptr)
    return {};
  MetalGraphicsPipelineJob job;
  if (!prepare_graphics_pipeline(*m_state, desc, job))
    return {};
  MetalPipeline pipeline = compile_graphics_pipeline(job);
  if (pipeline.render == nil)
    return {};
  return m_state->pipelines.insert(std::move(pipeline));
}

nx::future<CompiledPipeline> MetalDevice::create_graphics_pipeline_async(
    nx::thread_pool &workers, const GraphicsPipelineDesc &desc,
    const nx::task_priority priority) {
  if (m_state == nullptr)
    return {};
  MetalGraphicsPipelineJob job;
  if (!prepare_graphics_pipeline(*m_state, desc, job))
    return {};
  const nx::workload_ticket ticket = m_pipeline_compile_workload.submit();
  nx::future<CompiledPipeline> future = workers.async(
      priority, [this, ticket, job = std::move(job)]() mutable {
        return nx::measure_workload(
            m_pipeline_compile_workload, ticket, nx::cancellation_token{},
            [&job] {
              MetalPipeline native = compile_graphics_pipeline(job);
              if (native.render == nil)
                return CompiledPipeline{};
              return detail::CompiledPipelineAccess::adopt(std::move(native));
            });
      });
  if (!future.valid())
    m_pipeline_compile_workload.reject(ticket);
  return future;
}

PipelineHandle
MetalDevice::create_compute_pipeline(const ComputePipelineDesc &desc) {
  if (m_state == nullptr)
    return {};
  MetalComputePipelineJob job;
  if (!prepare_compute_pipeline(*m_state, desc, job))
    return {};
  MetalPipeline pipeline = compile_compute_pipeline(job);
  if (pipeline.compute == nil)
    return {};
  return m_state->pipelines.insert(std::move(pipeline));
}

nx::future<CompiledPipeline> MetalDevice::create_compute_pipeline_async(
    nx::thread_pool &workers, const ComputePipelineDesc &desc,
    const nx::task_priority priority) {
  if (m_state == nullptr)
    return {};
  MetalComputePipelineJob job;
  if (!prepare_compute_pipeline(*m_state, desc, job))
    return {};
  const nx::workload_ticket ticket = m_pipeline_compile_workload.submit();
  nx::future<CompiledPipeline> future = workers.async(
      priority, [this, ticket, job = std::move(job)]() mutable {
        return nx::measure_workload(
            m_pipeline_compile_workload, ticket, nx::cancellation_token{},
            [&job] {
              MetalPipeline native = compile_compute_pipeline(job);
              if (native.compute == nil)
                return CompiledPipeline{};
              return detail::CompiledPipelineAccess::adopt(std::move(native));
            });
      });
  if (!future.valid())
    m_pipeline_compile_workload.reject(ticket);
  return future;
}

PipelineHandle MetalDevice::publish_pipeline(CompiledPipeline &&compiled) {
  if (m_state == nullptr)
    return {};
  MetalPipeline native =
      detail::CompiledPipelineAccess::take<MetalPipeline>(compiled);
  if ((!native.is_compute && native.render == nil) ||
      (native.is_compute && native.compute == nil))
    return {};
  return m_state->pipelines.insert(std::move(native));
}

void MetalDevice::destroy_pipeline(const PipelineHandle handle) noexcept {
  if (m_state == nullptr || !handle.valid())
    return;
  retire(*m_state, {.pipeline = handle});
}

void MetalCommandContext::bind_pipeline(const PipelineHandle handle) {
  if (m_state == nullptr)
    return;

  const MetalPipeline *pipeline = m_state->pipelines.try_get(handle);
  if (pipeline == nullptr) {
    nx::loge(log_rhi, "bind_pipeline: handle is not live");
    return;
  }

  if (pipeline->is_compute) {
    if (ctx_slot().compute_encoder == nil) {
      nx::loge(log_rhi, "bind_pipeline: a compute pipeline needs a compute "
                        "pass (begin_compute_pass)");
      return;
    }
    [ctx_slot().compute_encoder setComputePipelineState:pipeline->compute];
    ctx_slot().bound_threads_per_group = pipeline->threads_per_group;
    ctx_slot().has_bound_pipeline = true;
    return;
  }

  id<MTLRenderCommandEncoder> encoder = ctx_slot().encoder;
  if (encoder == nil) {
    nx::loge(log_rhi, "bind_pipeline outside a render pass");
    return;
  }

  [encoder setRenderPipelineState:pipeline->render];
  [encoder setCullMode:pipeline->cull];
  [encoder setFrontFacingWinding:pipeline->winding];
  if (pipeline->depth_stencil != nil)
    [encoder setDepthStencilState:pipeline->depth_stencil];

  ctx_slot().bound_topology = pipeline->topology;
  ctx_slot().has_bound_pipeline = true;
}

void MetalCommandContext::push_constants(const void *data, const u32 size,
                                         const u32 offset) {
  if (m_state == nullptr || data == nullptr || size == 0)
    return;
  if (ctx_slot().encoder == nil && ctx_slot().compute_encoder == nil) {
    nx::loge(log_rhi, "push_constants outside a pass");
    return;
  }

  if (offset + size > PUSH_CONSTANT_SIZE) {
    nx::loge(log_rhi, "push_constants: {} bytes at offset {} exceeds the {} "
                      "byte push block",
             size, offset, PUSH_CONSTANT_SIZE);
    return;
  }

  std::memcpy(ctx_slot().push_constants + offset, data, size);

  if (ctx_slot().compute_encoder != nil) {
    [ctx_slot().compute_encoder setBytes:ctx_slot().push_constants
                                length:PUSH_CONSTANT_SIZE
                               atIndex:PUSH_CONSTANT_BUFFER_INDEX];
    return;
  }

  [ctx_slot().encoder setVertexBytes:ctx_slot().push_constants
                            length:PUSH_CONSTANT_SIZE
                           atIndex:PUSH_CONSTANT_BUFFER_INDEX];
  [ctx_slot().encoder setFragmentBytes:ctx_slot().push_constants
                              length:PUSH_CONSTANT_SIZE
                             atIndex:PUSH_CONSTANT_BUFFER_INDEX];
}

void MetalCommandContext::draw(const u32 vertex_count, const u32 instance_count,
                               const u32 first_vertex,
                               const u32 first_instance) {
  if (m_state == nullptr || ctx_slot().encoder == nil || vertex_count == 0)
    return;
  if (!ctx_slot().has_bound_pipeline) {
    nx::loge(log_rhi, "draw without a bound pipeline");
    return;
  }

  [ctx_slot().encoder drawPrimitives:ctx_slot().bound_topology
                       vertexStart:first_vertex
                       vertexCount:vertex_count
                     instanceCount:instance_count
                      baseInstance:first_instance];
}

void MetalCommandContext::bind_index_buffer(const BufferHandle handle,
                                            const IndexType type,
                                            const u64 offset) {
  if (m_state == nullptr)
    return;

  const MetalBuffer *buffer = m_state->buffers.try_get(handle);
  if (buffer == nullptr || buffer->buffer == nil) {
    nx::loge(log_rhi, "bind_index_buffer: handle is not live");
    return;
  }

  ctx_slot().index_buffer = buffer->buffer;
  ctx_slot().index_type =
      type == IndexType::U32 ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16;
  ctx_slot().index_stride = type == IndexType::U32 ? 4u : 2u;
  ctx_slot().index_buffer_offset = offset;
}

void MetalCommandContext::draw_indexed(const u32 index_count,
                                       const u32 instance_count,
                                       const u32 first_index,
                                       const i32 vertex_offset,
                                       const u32 first_instance) {
  if (m_state == nullptr || ctx_slot().encoder == nil || index_count == 0)
    return;
  if (!ctx_slot().has_bound_pipeline) {
    nx::loge(log_rhi, "draw_indexed without a bound pipeline");
    return;
  }
  if (ctx_slot().index_buffer == nil) {
    nx::loge(log_rhi, "draw_indexed without a bound index buffer");
    return;
  }

  // Vulkan takes first_index as an element count; Metal wants a byte offset
  // into the buffer, so it folds into the base offset here.
  const u64 offset = ctx_slot().index_buffer_offset +
                     nx::cast<u64>(first_index) * ctx_slot().index_stride;

  [ctx_slot().encoder drawIndexedPrimitives:ctx_slot().bound_topology
                               indexCount:index_count
                                indexType:ctx_slot().index_type
                              indexBuffer:ctx_slot().index_buffer
                        indexBufferOffset:offset
                            instanceCount:instance_count
                               baseVertex:vertex_offset
                             baseInstance:first_instance];
}

void MetalCommandContext::dispatch(const u32 groups_x, const u32 groups_y,
                                   const u32 groups_z) {
  if (m_state == nullptr || groups_x == 0 || groups_y == 0 || groups_z == 0)
    return;

  id<MTLComputeCommandEncoder> encoder = ctx_slot().compute_encoder;
  if (encoder == nil) {
    nx::loge(log_rhi, "dispatch outside a compute pass");
    return;
  }
  if (!ctx_slot().has_bound_pipeline) {
    nx::loge(log_rhi, "dispatch without a bound pipeline");
    return;
  }

  [encoder dispatchThreadgroups:MTLSizeMake(groups_x, groups_y, groups_z)
          threadsPerThreadgroup:ctx_slot().bound_threads_per_group];
}

void MetalCommandContext::draw_indirect(const BufferHandle handle,
                                        const u64 offset, const u32 draw_count,
                                        const u32 stride) {
  if (m_state == nullptr || ctx_slot().encoder == nil || draw_count == 0)
    return;
  if (!ctx_slot().has_bound_pipeline) {
    nx::loge(log_rhi, "draw_indirect without a bound pipeline");
    return;
  }
  const MetalBuffer *buffer = m_state->buffers.try_get(handle);
  if (buffer == nullptr || buffer->buffer == nil)
    return;

  const u64 step =
      stride != 0 ? stride : nx::cast<u64>(sizeof(DrawIndirectCommand));
  for (u32 i = 0; i < draw_count; ++i) {
    [ctx_slot().encoder drawPrimitives:ctx_slot().bound_topology
                       indirectBuffer:buffer->buffer
                 indirectBufferOffset:offset + nx::cast<u64>(i) * step];
  }
}

void MetalCommandContext::draw_indexed_indirect(const BufferHandle handle,
                                                const u64 offset,
                                                const u32 draw_count,
                                                const u32 stride) {
  if (m_state == nullptr || ctx_slot().encoder == nil || draw_count == 0)
    return;
  if (!ctx_slot().has_bound_pipeline) {
    nx::loge(log_rhi, "draw_indexed_indirect without a bound pipeline");
    return;
  }
  if (ctx_slot().index_buffer == nil) {
    nx::loge(log_rhi, "draw_indexed_indirect without an index buffer");
    return;
  }
  const MetalBuffer *buffer = m_state->buffers.try_get(handle);
  if (buffer == nullptr || buffer->buffer == nil)
    return;

  const u64 step =
      stride != 0 ? stride : nx::cast<u64>(sizeof(DrawIndexedIndirectCommand));
  for (u32 i = 0; i < draw_count; ++i) {
    [ctx_slot().encoder drawIndexedPrimitives:ctx_slot().bound_topology
                                    indexType:ctx_slot().index_type
                                  indexBuffer:ctx_slot().index_buffer
                            indexBufferOffset:ctx_slot().index_buffer_offset
                               indirectBuffer:buffer->buffer
                         indirectBufferOffset:offset +
                                              nx::cast<u64>(i) * step];
  }
}

void MetalCommandContext::dispatch_indirect(const BufferHandle handle,
                                            const u64 offset) {
  if (m_state == nullptr)
    return;
  id<MTLComputeCommandEncoder> encoder = ctx_slot().compute_encoder;
  if (encoder == nil) {
    nx::loge(log_rhi, "dispatch_indirect outside a compute pass");
    return;
  }
  if (!ctx_slot().has_bound_pipeline) {
    nx::loge(log_rhi, "dispatch_indirect without a bound pipeline");
    return;
  }
  const MetalBuffer *buffer = m_state->buffers.try_get(handle);
  if (buffer == nullptr || buffer->buffer == nil)
    return;

  [encoder dispatchThreadgroupsWithIndirectBuffer:buffer->buffer
                             indirectBufferOffset:offset
                            threadsPerThreadgroup:ctx_slot()
                                                      .bound_threads_per_group];
}

}
