
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

  MetalPipeline pipeline;
  @autoreleasepool {
    id<MTLFunction> vertex =
        lookup_function(*m_state, desc.vertex, desc.name, "vertex");
    if (vertex == nil)
      return {};

    id<MTLFunction> fragment = nil;
    if (desc.fragment.shader.valid()) {
      fragment = lookup_function(*m_state, desc.fragment, desc.name, "fragment");
      if (fragment == nil)
        return {};
    }

    MTLRenderPipelineDescriptor *info =
        [[MTLRenderPipelineDescriptor alloc] init];
    info.label = to_ns_string(desc.name);
    info.vertexFunction = vertex;
    info.fragmentFunction = fragment;
    info.rasterSampleCount = nx::max<u32>(1, desc.sample_count);
    info.inputPrimitiveTopology = to_mtl_topology_class(desc.topology);

    for (u32 i = 0; i < desc.color_count && i < MAX_COLOR_ATTACHMENTS; ++i) {
      const MTLPixelFormat format = to_mtl_format(desc.color_formats[i]);
      if (format == MTLPixelFormatInvalid) {
        nx::loge(log_rhi, "create_graphics_pipeline('{}'): colour format {} is "
                          "not supported by Metal",
                 desc.name, to_string(desc.color_formats[i]));
        return {};
      }
      info.colorAttachments[i].pixelFormat = format;
      apply_blend(info.colorAttachments[i], desc.blend[i]);
    }

    if (desc.depth_format != Format::Unknown) {
      const MTLPixelFormat format = to_mtl_format(desc.depth_format);
      info.depthAttachmentPixelFormat = format;
      if (format_is_stencil(desc.depth_format))
        info.stencilAttachmentPixelFormat = format;
    }

    // Offering the archive lets Metal reuse a previously compiled binary
    // instead of running the shader compiler. A miss is not an error - without
    // MTLPipelineOptionFailOnBinaryArchiveMiss it just compiles as usual.
    if (m_state->pipeline_archive != nil)
      info.binaryArchives = @[ m_state->pipeline_archive ];

    NSError *error = nil;
    pipeline.render =
        [m_state->device newRenderPipelineStateWithDescriptor:info
                                                        error:&error];
    if (pipeline.render == nil) {
      nx::loge(log_rhi, "create_graphics_pipeline('{}'): {}", desc.name,
               error != nil ? [[error localizedDescription] UTF8String]
                            : "newRenderPipelineState failed");
      return {};
    }

    if (m_state->pipeline_archive != nil) {
      NSError *add_error = nil;
      if (![m_state->pipeline_archive
              addRenderPipelineFunctionsWithDescriptor:info
                                                 error:&add_error]) {
        nx::logd(log_rhi, "pipeline archive: '{}' not cached: {}", desc.name,
                 add_error != nil
                     ? [[add_error localizedDescription] UTF8String]
                     : "unknown error");
      }
    }

    if (desc.depth_test || desc.depth_write) {
      MTLDepthStencilDescriptor *depth =
          [[MTLDepthStencilDescriptor alloc] init];
      depth.label = to_ns_string(desc.name);
      depth.depthCompareFunction = desc.depth_test
                                       ? to_mtl_compare(desc.depth_compare)
                                       : MTLCompareFunctionAlways;
      depth.depthWriteEnabled = desc.depth_write ? YES : NO;
      pipeline.depth_stencil =
          [m_state->device newDepthStencilStateWithDescriptor:depth];
    }
  }

  pipeline.topology = to_mtl_topology(desc.topology);
  pipeline.cull = to_mtl_cull(desc.cull);
  pipeline.winding = to_mtl_winding(desc.front_face);
  pipeline.is_compute = false;

  return m_state->pipelines.insert(std::move(pipeline));
}

PipelineHandle
MetalDevice::create_compute_pipeline(const ComputePipelineDesc &desc) {
  if (m_state == nullptr)
    return {};

  u32 group_size[3] = {desc.group_size[0], desc.group_size[1],
                       desc.group_size[2]};
  if (group_size[0] == 0 || group_size[1] == 0 || group_size[2] == 0) {
    const MetalShader *cs = m_state->shaders.try_get(desc.compute.shader);
    if (cs == nullptr ||
        !cs->reflection.local_size(nx::id_string(desc.compute.entry_point),
                                   group_size)) {
      nx::loge(log_rhi,
               "create_compute_pipeline('{}'): no group_size given and the "
               "reflection sidecar does not state one for '{}'",
               desc.name, desc.compute.entry_point);
      return {};
    }
  }

  MetalPipeline pipeline;
  @autoreleasepool {
    id<MTLFunction> function =
        lookup_function(*m_state, desc.compute, desc.name, "compute");
    if (function == nil)
      return {};

    MTLComputePipelineDescriptor *info =
        [[MTLComputePipelineDescriptor alloc] init];
    info.label = to_ns_string(desc.name);
    info.computeFunction = function;
    if (m_state->pipeline_archive != nil)
      info.binaryArchives = @[ m_state->pipeline_archive ];

    NSError *error = nil;
    pipeline.compute =
        [m_state->device newComputePipelineStateWithDescriptor:info
                                                       options:MTLPipelineOptionNone
                                                    reflection:nullptr
                                                         error:&error];
    if (pipeline.compute == nil) {
      nx::loge(log_rhi, "create_compute_pipeline('{}'): {}", desc.name,
               error != nil ? [[error localizedDescription] UTF8String]
                            : "newComputePipelineState failed");
      return {};
    }

    if (m_state->pipeline_archive != nil) {
      NSError *add_error = nil;
      if (![m_state->pipeline_archive
              addComputePipelineFunctionsWithDescriptor:info
                                                  error:&add_error]) {
        nx::logd(log_rhi, "pipeline archive: '{}' not cached: {}", desc.name,
                 add_error != nil
                     ? [[add_error localizedDescription] UTF8String]
                     : "unknown error");
      }
    }

    pipeline.threads_per_group = MTLSizeMake(
        group_size[0], group_size[1], group_size[2]);

    const NSUInteger declared = nx::cast<NSUInteger>(group_size[0]) *
                                group_size[1] * group_size[2];
    if (declared > pipeline.compute.maxTotalThreadsPerThreadgroup) {
      nx::loge(log_rhi, "create_compute_pipeline('{}'): {} threads per group "
                        "exceeds this device's limit of {}",
               desc.name, declared,
               pipeline.compute.maxTotalThreadsPerThreadgroup);
      return {};
    }
  }

  pipeline.is_compute = true;
  return m_state->pipelines.insert(std::move(pipeline));
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
