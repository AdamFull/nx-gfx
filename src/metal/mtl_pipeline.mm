/**
 * @file mtl_pipeline.mm
 * @brief Shader libraries and pipeline state objects.
 *
 * Metal splits what Vulkan calls one pipeline across three places: the render
 * pipeline state (shaders, attachment formats, blending), a separate
 * MTLDepthStencilState, and encoder calls for cull mode, winding and primitive
 * type. MetalPipeline holds all of it so bind_pipeline can apply the lot.
 */

#include "rendering/rhi/metal/mtl_internal.h"

namespace nxe::rhi {
namespace {

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

/// Metal also needs the topology *class* at pipeline creation, but only when
/// the shader writes a point size or the pipeline is used with a tessellator.
/// Reporting it unconditionally is harmless and keeps validation quiet.
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

/// Inverted on purpose. The backend renders with a Y-flipped viewport so that
/// Vulkan-convention clip space lands the same way on Metal (see
/// to_mtl_viewport), and mirroring the framebuffer reverses the winding of
/// every triangle. Flipping the front face back here cancels it out, so a
/// pipeline that culls correctly on Vulkan culls correctly here.
MTLWinding to_mtl_winding(const FrontFace face) noexcept {
  return face == FrontFace::Clockwise ? MTLWindingCounterClockwise
                                      : MTLWindingClockwise;
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
  case BlendMode::Multiply:
    attachment.sourceRGBBlendFactor = MTLBlendFactorDestinationColor;
    attachment.destinationRGBBlendFactor = MTLBlendFactorZero;
    attachment.sourceAlphaBlendFactor = MTLBlendFactorDestinationAlpha;
    attachment.destinationAlphaBlendFactor = MTLBlendFactorZero;
    break;
  case BlendMode::Opaque:
  default:
    attachment.blendingEnabled = NO;
    break;
  }
}

/// Looks an entry point up in a shader blob. Slang emits Metal entry points
/// under their original names, so the RHI's entry_point string maps straight
/// across.
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

} // namespace

// Shaders

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

  return m_state->shaders.insert(std::move(shader));
}

void MetalDevice::destroy_shader(const ShaderHandle handle) noexcept {
  if (m_state == nullptr || !handle.valid())
    return;
  // A library is only referenced while a pipeline is being built, but deferring
  // costs nothing and keeps every destroy path on the same rule.
  retire(*m_state, {.shader = handle});
}

// Pipelines

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

    // A fragment-less pipeline is legal (depth prepass), so an unset handle is
    // not an error - only a named entry point that fails to resolve is.
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

    // Vertices are pulled from SV_VertexID or from bindless buffers, never
    // from a vertex buffer, so there is deliberately no vertexDescriptor.

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

    // Depth state is a separate object in Metal, and only exists when the
    // pipeline actually tests or writes depth.
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

  MetalPipeline pipeline;
  @autoreleasepool {
    id<MTLFunction> function =
        lookup_function(*m_state, desc.compute, desc.name, "compute");
    if (function == nil)
      return {};

    NSError *error = nil;
    pipeline.compute =
        [m_state->device newComputePipelineStateWithFunction:function
                                                       error:&error];
    if (pipeline.compute == nil) {
      nx::loge(log_rhi, "create_compute_pipeline('{}'): {}", desc.name,
               error != nil ? [[error localizedDescription] UTF8String]
                            : "newComputePipelineState failed");
      return {};
    }

    // Metal dispatches threads, not groups, so the group size has to travel
    // with the pipeline for dispatch() to multiply it back out.
    const NSUInteger width = pipeline.compute.threadExecutionWidth;
    const NSUInteger total = pipeline.compute.maxTotalThreadsPerThreadgroup;
    pipeline.threads_per_group = MTLSizeMake(width, total / width, 1);
  }

  pipeline.is_compute = true;
  return m_state->pipelines.insert(std::move(pipeline));
}

void MetalDevice::destroy_pipeline(const PipelineHandle handle) noexcept {
  if (m_state == nullptr || !handle.valid())
    return;
  // A pipeline state is referenced by every command buffer that bound it, so
  // it outlives the destroy call by up to frames_in_flight frames.
  retire(*m_state, {.pipeline = handle});
}

// Command context: pipeline binding and draw submission.

void MetalCommandContext::bind_pipeline(const PipelineHandle handle) {
  if (m_state == nullptr)
    return;

  const MetalPipeline *pipeline = m_state->pipelines.try_get(handle);
  if (pipeline == nullptr) {
    nx::loge(log_rhi, "bind_pipeline: handle is not live");
    return;
  }

  if (pipeline->is_compute) {
    // Compute needs its own encoder, which cannot coexist with a render pass.
    nx::loge(log_rhi, "bind_pipeline: compute pipelines are not supported "
                      "inside a render pass");
    return;
  }

  id<MTLRenderCommandEncoder> encoder = m_state->encoder;
  if (encoder == nil) {
    nx::loge(log_rhi, "bind_pipeline outside a render pass");
    return;
  }

  [encoder setRenderPipelineState:pipeline->render];
  [encoder setCullMode:pipeline->cull];
  [encoder setFrontFacingWinding:pipeline->winding];
  if (pipeline->depth_stencil != nil)
    [encoder setDepthStencilState:pipeline->depth_stencil];

  m_state->bound_topology = pipeline->topology;
  m_state->has_bound_pipeline = true;
}

void MetalCommandContext::push_constants(const void *data, const u32 size,
                                         const u32 offset) {
  if (m_state == nullptr || m_state->encoder == nil || data == nullptr ||
      size == 0)
    return;

  if (offset + size > PUSH_CONSTANT_SIZE) {
    nx::loge(log_rhi, "push_constants: {} bytes at offset {} exceeds the {} "
                      "byte push block",
             size, offset, PUSH_CONSTANT_SIZE);
    return;
  }

  // Metal has no partial update for a setBytes argument: the whole buffer
  // binding is replaced each time. Shadow the block so a partial write keeps
  // whatever the previous writes put there.
  std::memcpy(m_state->push_constants + offset, data, size);

  [m_state->encoder setVertexBytes:m_state->push_constants
                            length:PUSH_CONSTANT_SIZE
                           atIndex:PUSH_CONSTANT_BUFFER_INDEX];
  [m_state->encoder setFragmentBytes:m_state->push_constants
                              length:PUSH_CONSTANT_SIZE
                             atIndex:PUSH_CONSTANT_BUFFER_INDEX];
}

void MetalCommandContext::draw(const u32 vertex_count, const u32 instance_count,
                               const u32 first_vertex,
                               const u32 first_instance) {
  if (m_state == nullptr || m_state->encoder == nil || vertex_count == 0)
    return;
  if (!m_state->has_bound_pipeline) {
    nx::loge(log_rhi, "draw without a bound pipeline");
    return;
  }

  [m_state->encoder drawPrimitives:m_state->bound_topology
                       vertexStart:first_vertex
                       vertexCount:vertex_count
                     instanceCount:instance_count
                      baseInstance:first_instance];
}

void MetalCommandContext::draw_indexed(const u32 index_count,
                                       const u32 instance_count,
                                       const u32 first_index,
                                       const i32 vertex_offset,
                                       const u32 first_instance) {
  // Metal takes the index buffer on the draw call itself rather than from a
  // bind, and the RHI has no bind_index_buffer yet - it arrives with the 2D
  // renderer, which is the first thing that needs one.
  (void)index_count;
  (void)instance_count;
  (void)first_index;
  (void)vertex_offset;
  (void)first_instance;
  nx::loge(log_rhi, "draw_indexed: not supported until the RHI has index "
                    "buffer binding");
}

void MetalCommandContext::dispatch(const u32 groups_x, const u32 groups_y,
                                   const u32 groups_z) {
  (void)groups_x;
  (void)groups_y;
  (void)groups_z;
  // A compute dispatch needs its own encoder, so it cannot be recorded between
  // begin_render_pass and end_render_pass. The render graph will own that
  // split; until then there is nothing sane to do here.
  nx::loge(log_rhi, "dispatch: compute passes are not implemented on Metal yet");
}

} // namespace nxe::rhi
