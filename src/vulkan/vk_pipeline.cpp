
#include "core/rendering/rhi/vulkan/vk_device.h"

#include "core/foundation/diagnostics/log.h"
#include "core/foundation/platform/filesystem.h"

#include <cstring>
#include <span>

namespace nxe::rhi {

/// Owner-thread snapshot of everything vkCreateGraphicsPipelines needs. Shader
/// modules are immutable Vulkan objects; their ShaderHandle owners remain live
/// until the worker finishes. No engine handle table is touched by the worker.
struct VulkanGraphicsPipelineCompileJob {
  nx::string name;
  nx::string vertex_entry;
  nx::string fragment_entry;
  VkShaderModule vertex = VK_NULL_HANDLE;
  VkShaderModule fragment = VK_NULL_HANDLE;
  Format color_formats[MAX_COLOR_ATTACHMENTS] = {};
  u32 color_count = 0;
  Format depth_format = Format::Unknown;
  BlendState blend[MAX_COLOR_ATTACHMENTS] = {};
  PrimitiveTopology topology = PrimitiveTopology::TriangleList;
  CullMode cull = CullMode::None;
  FrontFace front_face = FrontFace::CounterClockwise;
  bool depth_test = false;
  bool depth_write = false;
  CompareOp depth_compare = CompareOp::LessOrEqual;
  u32 sample_count = 1;
};

struct VulkanComputePipelineCompileJob {
  nx::string name;
  nx::string entry;
  VkShaderModule shader = VK_NULL_HANDLE;
};

/// RAII owner for an unpublished native pipeline. Cancellation and failed
/// publication therefore cannot leak the driver object.
class VulkanCompiledPipeline {
public:
  VulkanCompiledPipeline() = default;
  ~VulkanCompiledPipeline() { reset(); }

  VulkanCompiledPipeline(const VulkanCompiledPipeline &) = delete;
  VulkanCompiledPipeline &operator=(const VulkanCompiledPipeline &) = delete;

  VulkanCompiledPipeline(VulkanCompiledPipeline &&other) noexcept
      : m_device(other.m_device), m_pipeline(other.m_pipeline),
        m_bind_point(other.m_bind_point), m_name(std::move(other.m_name)) {
    other.m_device = VK_NULL_HANDLE;
    other.m_pipeline = VK_NULL_HANDLE;
  }

  VulkanCompiledPipeline &operator=(VulkanCompiledPipeline &&other) noexcept {
    if (this != &other) {
      reset();
      m_device = other.m_device;
      m_pipeline = other.m_pipeline;
      m_bind_point = other.m_bind_point;
      m_name = std::move(other.m_name);
      other.m_device = VK_NULL_HANDLE;
      other.m_pipeline = VK_NULL_HANDLE;
    }
    return *this;
  }

  [[nodiscard]] bool valid() const noexcept {
    return m_pipeline != VK_NULL_HANDLE;
  }

private:
  friend class VulkanDevice;

  void reset() noexcept {
    if (m_device != VK_NULL_HANDLE && m_pipeline != VK_NULL_HANDLE)
      vkDestroyPipeline(m_device, m_pipeline, nullptr);
    m_device = VK_NULL_HANDLE;
    m_pipeline = VK_NULL_HANDLE;
    m_name.clear();
  }

  VkDevice m_device = VK_NULL_HANDLE;
  VkPipeline m_pipeline = VK_NULL_HANDLE;
  VkPipelineBindPoint m_bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;
  nx::string m_name;
};

namespace {

const nx::log::Category log_rhi = nx::log::category("rhi");

VkPrimitiveTopology to_vk_topology(const PrimitiveTopology topology) noexcept {
  switch (topology) {
  case PrimitiveTopology::TriangleList:
    return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  case PrimitiveTopology::TriangleStrip:
    return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
  case PrimitiveTopology::LineList:
    return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
  case PrimitiveTopology::LineStrip:
    return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
  default:
    return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
  }
}

VkCullModeFlags to_vk_cull(const CullMode cull) noexcept {
  switch (cull) {
  case CullMode::Front:
    return VK_CULL_MODE_FRONT_BIT;
  case CullMode::Back:
    return VK_CULL_MODE_BACK_BIT;
  default:
    return VK_CULL_MODE_NONE;
  }
}

VkPipelineColorBlendAttachmentState
to_vk_blend(const BlendState &state) noexcept {
  VkPipelineColorBlendAttachmentState out{};
  out.colorWriteMask = 0;
  if (state.write_mask & 0x1)
    out.colorWriteMask |= VK_COLOR_COMPONENT_R_BIT;
  if (state.write_mask & 0x2)
    out.colorWriteMask |= VK_COLOR_COMPONENT_G_BIT;
  if (state.write_mask & 0x4)
    out.colorWriteMask |= VK_COLOR_COMPONENT_B_BIT;
  if (state.write_mask & 0x8)
    out.colorWriteMask |= VK_COLOR_COMPONENT_A_BIT;

  out.blendEnable = state.enabled ? VK_TRUE : VK_FALSE;
  out.colorBlendOp = VK_BLEND_OP_ADD;
  out.alphaBlendOp = VK_BLEND_OP_ADD;
  out.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  out.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;

  switch (state.mode) {
  case BlendMode::AlphaBlend:
    out.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    out.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    break;
  case BlendMode::Premultiplied:
    out.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    out.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    break;
  case BlendMode::Additive:
    out.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    out.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    out.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    break;
  case BlendMode::PremultipliedAdditive:
    out.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    out.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    out.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    break;
  case BlendMode::Multiply:
    out.srcColorBlendFactor = VK_BLEND_FACTOR_DST_COLOR;
    out.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    break;
  case BlendMode::Screen:
    out.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    out.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    break;
  case BlendMode::Opaque:
  case BlendMode::Count:
    out.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    out.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    break;
  }
  return out;
}

VkShaderModule create_module(VkDevice device, const void *code,
                             const usize size) noexcept {
  VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  info.codeSize = size;
  info.pCode = static_cast<const u32 *>(code);
  VkShaderModule module = VK_NULL_HANDLE;
  NX_VK_CHECK(vkCreateShaderModule(device, &info, nullptr, &module));
  return module;
}

} // namespace

bool VulkanDevice::create_pipeline_cache(const nx::string_view path) {
  m_pipeline_cache_path = path;

  VkPipelineCacheCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};

  nx::blob<u8> initial;
  if (!path.empty() && nx::fs::exists(path)) {
    if (auto read = nx::fs::file_read(path)) {
      initial = std::move(*read);
      VkPhysicalDeviceProperties props{};
      vkGetPhysicalDeviceProperties(m_physical_device, &props);
      if (initial.size() >= 16 + VK_UUID_SIZE) {
        const u8 *uuid = initial.data() + 16;
        if (std::memcmp(uuid, props.pipelineCacheUUID, VK_UUID_SIZE) != 0) {
          nx::logd(log_rhi, "pipeline cache is for another device, ignoring");
          initial = {};
        }
      }
    }
  }

  info.initialDataSize = initial.size();
  info.pInitialData = initial.size() ? initial.data() : nullptr;
  NX_VK_TRY(vkCreatePipelineCache(m_device, &info, nullptr, &m_pipeline_cache));

  if (initial.size())
    nx::logd(log_rhi, "pipeline cache: loaded {} bytes", initial.size());
  return true;
}

void VulkanDevice::save_pipeline_cache() noexcept {
  if (m_pipeline_cache == VK_NULL_HANDLE || m_pipeline_cache_path.empty())
    return;

  const nx::scoped_lock<nx::mutex> held(m_pipeline_cache_lock);

  usize size = 0;
  if (vkGetPipelineCacheData(m_device, m_pipeline_cache, &size, nullptr) !=
          VK_SUCCESS ||
      size == 0)
    return;

  nx::blob<u8> data(size);
  if (vkGetPipelineCacheData(m_device, m_pipeline_cache, &size, data.data()) !=
      VK_SUCCESS)
    return;

  const auto written =
      nx::fs::file_write(m_pipeline_cache_path, {data.data(), size});
  if (written)
    nx::logd(log_rhi, "pipeline cache: saved {} bytes", size);
  else
    nx::logw(log_rhi, "pipeline cache: could not write {}",
             m_pipeline_cache_path);
}

ShaderHandle VulkanDevice::create_shader(const ShaderDesc &desc) {
  if (desc.code == nullptr || desc.code_size == 0 || desc.code_size % 4 != 0) {
    nx::loge(log_rhi, "create_shader('{}'): invalid SPIR-V blob", desc.name);
    return {};
  }

  VulkanShader shader;
  shader.module = create_module(m_device, desc.code, desc.code_size);
  if (shader.module == VK_NULL_HANDLE)
    return {};

  if (!desc.reflection_json.empty() &&
      !reflect_from_json(desc.reflection_json, shader.reflection))
    nx::logw(log_rhi, "create_shader('{}'): reflection sidecar unusable",
             desc.name);

  for (const ReflectComputeEntry &e : shader.reflection.compute_entries) {
    VulkanShader::ComputeEntry entry;
    entry.name = nx::string(e.name.view());
    entry.local_size[0] = e.local_size[0];
    entry.local_size[1] = e.local_size[1];
    entry.local_size[2] = e.local_size[2];
    shader.compute_entries.push_back(std::move(entry));
  }

  set_object_name(vk_handle_value(shader.module), VK_OBJECT_TYPE_SHADER_MODULE,
                  desc.name);
  return m_shaders.insert(std::move(shader));
}

const ShaderReflection &
VulkanDevice::reflection(const ShaderHandle handle) const noexcept {
  static const ShaderReflection empty;
  const VulkanShader *shader = m_shaders.try_get(handle);
  return shader != nullptr ? shader->reflection : empty;
}

void VulkanDevice::destroy_shader(const ShaderHandle handle) noexcept {
  if (VulkanShader *shader = m_shaders.try_get(handle)) {
    vkDestroyShaderModule(m_device, shader->module, nullptr);
    m_shaders.erase(handle);
  }
}

PipelineHandle
VulkanDevice::create_graphics_pipeline(const GraphicsPipelineDesc &desc) {
  VulkanGraphicsPipelineCompileJob job;
  if (!prepare_graphics_pipeline(desc, job))
    return {};
  VulkanCompiledPipeline native = compile_graphics_pipeline(job);
  if (!native.valid())
    return {};
  return publish_pipeline(
      detail::CompiledPipelineAccess::adopt(std::move(native)));
}

nx::future<CompiledPipeline>
VulkanDevice::create_graphics_pipeline_async(nx::thread_pool &workers,
                                             const GraphicsPipelineDesc &desc,
                                             const nx::task_priority priority) {
  VulkanGraphicsPipelineCompileJob job;
  if (!prepare_graphics_pipeline(desc, job))
    return {};
  const nx::workload_ticket ticket = m_pipeline_compile_workload.submit();
  nx::future<CompiledPipeline> future =
      workers.async(priority, [this, ticket, job = std::move(job)]() mutable {
        return nx::measure_workload(
            m_pipeline_compile_workload, ticket, nx::cancellation_token{},
            [this, &job] {
              VulkanCompiledPipeline native = compile_graphics_pipeline(job);
              if (!native.valid())
                return CompiledPipeline{};
              return detail::CompiledPipelineAccess::adopt(std::move(native));
            });
      });
  if (!future.valid())
    m_pipeline_compile_workload.reject(ticket);
  return future;
}

bool VulkanDevice::prepare_graphics_pipeline(
    const GraphicsPipelineDesc &desc,
    VulkanGraphicsPipelineCompileJob &out) const {
  const VulkanShader *vs = m_shaders.try_get(desc.vertex.shader);
  if (vs == nullptr) {
    nx::loge(log_rhi, "create_graphics_pipeline('{}'): missing vertex shader",
             desc.name);
    return {};
  }

  const VulkanShader *fs = m_shaders.try_get(desc.fragment.shader);

  VulkanGraphicsPipelineCompileJob job;
  job.name = nx::string(desc.name);
  job.vertex_entry = nx::string(desc.vertex.entry_point);
  job.fragment_entry = nx::string(desc.fragment.entry_point);
  job.vertex = vs->module;
  job.fragment = fs != nullptr ? fs->module : VK_NULL_HANDLE;
  job.color_count = nx::min(desc.color_count, MAX_COLOR_ATTACHMENTS);
  for (u32 i = 0; i < job.color_count; ++i) {
    job.color_formats[i] = desc.color_formats[i];
    job.blend[i] = desc.blend[i];
  }
  job.depth_format = desc.depth_format;
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

VulkanCompiledPipeline VulkanDevice::compile_graphics_pipeline(
    const VulkanGraphicsPipelineCompileJob &job) const {
  if (job.vertex == VK_NULL_HANDLE)
    return {};

  VkPipelineShaderStageCreateInfo stages[2] = {};
  stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = job.vertex;
  stages[0].pName = job.vertex_entry.c_str();
  u32 stage_count = 1;
  if (job.fragment != VK_NULL_HANDLE) {
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = job.fragment;
    stages[1].pName = job.fragment_entry.c_str();
    stage_count = 2;
  }

  VkPipelineVertexInputStateCreateInfo vertex_input{
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};

  VkPipelineInputAssemblyStateCreateInfo input_assembly{
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  input_assembly.topology = to_vk_topology(job.topology);

  VkPipelineViewportStateCreateInfo viewport{
      VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  viewport.viewportCount = 1;
  viewport.scissorCount = 1;

  VkPipelineRasterizationStateCreateInfo raster{
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  raster.polygonMode = VK_POLYGON_MODE_FILL;
  raster.cullMode = to_vk_cull(job.cull);
  raster.frontFace = job.front_face == FrontFace::Clockwise
                         ? VK_FRONT_FACE_CLOCKWISE
                         : VK_FRONT_FACE_COUNTER_CLOCKWISE;
  raster.lineWidth = 1.f;

  VkPipelineMultisampleStateCreateInfo multisample{
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  multisample.rasterizationSamples =
      nx::cast<VkSampleCountFlagBits>(job.sample_count);

  VkPipelineDepthStencilStateCreateInfo depth_stencil{
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  depth_stencil.depthTestEnable = job.depth_test ? VK_TRUE : VK_FALSE;
  depth_stencil.depthWriteEnable = job.depth_write ? VK_TRUE : VK_FALSE;
  depth_stencil.depthCompareOp = to_vk_compare_op(job.depth_compare);
  depth_stencil.maxDepthBounds = 1.f;

  VkPipelineColorBlendAttachmentState blend_states[MAX_COLOR_ATTACHMENTS] = {};
  for (u32 i = 0; i < job.color_count; ++i)
    blend_states[i] = to_vk_blend(job.blend[i]);

  VkPipelineColorBlendStateCreateInfo color_blend{
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  color_blend.attachmentCount = job.color_count;
  color_blend.pAttachments = blend_states;

  const VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT,
                                           VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynamic{
      VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dynamic.dynamicStateCount = 2;
  dynamic.pDynamicStates = dynamic_states;

  VkFormat color_formats[MAX_COLOR_ATTACHMENTS] = {};
  for (u32 i = 0; i < job.color_count; ++i)
    color_formats[i] = to_vk_format(job.color_formats[i]);

  VkPipelineRenderingCreateInfo rendering{
      VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  rendering.colorAttachmentCount = job.color_count;
  rendering.pColorAttachmentFormats = color_formats;
  if (job.depth_format != Format::Unknown) {
    rendering.depthAttachmentFormat = to_vk_format(job.depth_format);
    if (format_is_stencil(job.depth_format))
      rendering.stencilAttachmentFormat = to_vk_format(job.depth_format);
  }

  VkGraphicsPipelineCreateInfo info{
      VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  info.pNext = &rendering;
  info.stageCount = stage_count;
  info.pStages = stages;
  info.pVertexInputState = &vertex_input;
  info.pInputAssemblyState = &input_assembly;
  info.pViewportState = &viewport;
  info.pRasterizationState = &raster;
  info.pMultisampleState = &multisample;
  info.pDepthStencilState = &depth_stencil;
  info.pColorBlendState = job.color_count > 0 ? &color_blend : nullptr;
  info.pDynamicState = &dynamic;
  info.layout = m_pipeline_layout;

  VulkanCompiledPipeline compiled;
  compiled.m_device = m_device;
  compiled.m_bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;
  compiled.m_name = job.name;
  const nx::scoped_lock<nx::mutex> held(m_pipeline_cache_lock);
  if (NX_VK_CHECK(vkCreateGraphicsPipelines(
          m_device, m_pipeline_cache, 1, &info, nullptr,
          &compiled.m_pipeline)) != VK_SUCCESS) {
    nx::loge(log_rhi, "create_graphics_pipeline('{}') failed", job.name);
    return {};
  }
  return compiled;
}

PipelineHandle
VulkanDevice::create_compute_pipeline(const ComputePipelineDesc &desc) {
  VulkanComputePipelineCompileJob job;
  if (!prepare_compute_pipeline(desc, job))
    return {};
  VulkanCompiledPipeline native = compile_compute_pipeline(job);
  if (!native.valid())
    return {};
  return publish_pipeline(
      detail::CompiledPipelineAccess::adopt(std::move(native)));
}

nx::future<CompiledPipeline>
VulkanDevice::create_compute_pipeline_async(nx::thread_pool &workers,
                                            const ComputePipelineDesc &desc,
                                            const nx::task_priority priority) {
  VulkanComputePipelineCompileJob job;
  if (!prepare_compute_pipeline(desc, job))
    return {};
  const nx::workload_ticket ticket = m_pipeline_compile_workload.submit();
  nx::future<CompiledPipeline> future =
      workers.async(priority, [this, ticket, job = std::move(job)]() mutable {
        return nx::measure_workload(
            m_pipeline_compile_workload, ticket, nx::cancellation_token{},
            [this, &job] {
              VulkanCompiledPipeline native = compile_compute_pipeline(job);
              if (!native.valid())
                return CompiledPipeline{};
              return detail::CompiledPipelineAccess::adopt(std::move(native));
            });
      });
  if (!future.valid())
    m_pipeline_compile_workload.reject(ticket);
  return future;
}

bool VulkanDevice::prepare_compute_pipeline(
    const ComputePipelineDesc &desc,
    VulkanComputePipelineCompileJob &out) const {
  const VulkanShader *cs = m_shaders.try_get(desc.compute.shader);
  if (cs == nullptr) {
    nx::loge(log_rhi, "create_compute_pipeline('{}'): missing shader",
             desc.name);
    return false;
  }

  for (const VulkanShader::ComputeEntry &declared : cs->compute_entries) {
    if (declared.name.view() != desc.compute.entry_point)
      continue;
    // A zero means the shader states its size through a specialisation
    // constant, so there is nothing to compare against.
    if (declared.local_size[0] == 0)
      break;
    if (desc.group_size[0] != 0 &&
        (desc.group_size[0] != declared.local_size[0] ||
         desc.group_size[1] != declared.local_size[1] ||
         desc.group_size[2] != declared.local_size[2])) {
      nx::logw(log_rhi,
               "create_compute_pipeline('{}'): group_size {}x{}x{} does not "
               "match the shader's [numthreads({}, {}, {})]. Vulkan ignores "
               "the field, but Metal dispatches with it and will be wrong",
               desc.name, desc.group_size[0], desc.group_size[1],
               desc.group_size[2], declared.local_size[0],
               declared.local_size[1], declared.local_size[2]);
    }
    break;
  }

  VulkanComputePipelineCompileJob job;
  job.name = nx::string(desc.name);
  job.entry = nx::string(desc.compute.entry_point);
  job.shader = cs->module;
  out = std::move(job);
  return true;
}

VulkanCompiledPipeline VulkanDevice::compile_compute_pipeline(
    const VulkanComputePipelineCompileJob &job) const {
  if (job.shader == VK_NULL_HANDLE)
    return {};

  VkComputePipelineCreateInfo info{
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  info.stage.module = job.shader;
  info.stage.pName = job.entry.c_str();
  info.layout = m_pipeline_layout;

  VulkanCompiledPipeline compiled;
  compiled.m_device = m_device;
  compiled.m_bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
  compiled.m_name = job.name;
  const nx::scoped_lock<nx::mutex> held(m_pipeline_cache_lock);
  if (NX_VK_CHECK(vkCreateComputePipelines(m_device, m_pipeline_cache, 1, &info,
                                           nullptr, &compiled.m_pipeline)) !=
      VK_SUCCESS) {
    nx::loge(log_rhi, "create_compute_pipeline('{}') failed", job.name);
    return {};
  }
  return compiled;
}

PipelineHandle VulkanDevice::publish_pipeline(CompiledPipeline &&opaque) {
  VulkanCompiledPipeline compiled =
      detail::CompiledPipelineAccess::take<VulkanCompiledPipeline>(opaque);
  if (!compiled.valid())
    return {};
  VulkanPipeline pipeline;
  pipeline.pipeline = compiled.m_pipeline;
  pipeline.bind_point = compiled.m_bind_point;
  const nx::string name = std::move(compiled.m_name);
  compiled.m_device = VK_NULL_HANDLE;
  compiled.m_pipeline = VK_NULL_HANDLE;
  const PipelineHandle handle = m_pipelines.insert(pipeline);
  if (!handle.valid()) {
    vkDestroyPipeline(m_device, pipeline.pipeline, nullptr);
    return {};
  }
  set_object_name(vk_handle_value(pipeline.pipeline), VK_OBJECT_TYPE_PIPELINE,
                  name.view());
  return handle;
}

void VulkanDevice::destroy_pipeline(const PipelineHandle handle) noexcept {
  if (!m_pipelines.contains(handle))
    return;
  m_pending_deletes.push_back({m_timeline_value, {}, {}, {}, handle});
}

const VulkanPipeline *
VulkanDevice::pipeline(const PipelineHandle handle) const noexcept {
  return m_pipelines.try_get(handle);
}

void VulkanCommandContext::bind_pipeline(const PipelineHandle handle) {
  if (m_device == nullptr)
    return;
  const VulkanPipeline *pipeline = m_device->pipeline(handle);
  if (pipeline == nullptr)
    return;
  vkCmdBindPipeline(m_cmd, pipeline->bind_point, pipeline->pipeline);
}

void VulkanCommandContext::push_constants(const void *data, const u32 size,
                                          const u32 offset) {
  if (m_device == nullptr)
    return;
  vkCmdPushConstants(m_cmd, m_device->pipeline_layout(), VK_SHADER_STAGE_ALL,
                     offset, size, data);
}

void VulkanCommandContext::bind_index_buffer(const BufferHandle handle,
                                             const IndexType type,
                                             const u64 offset) {
  if (m_device == nullptr)
    return;
  const VulkanBuffer *buffer = m_device->buffer(handle);
  if (buffer == nullptr || buffer->buffer == VK_NULL_HANDLE) {
    nx::loge(log_rhi, "bind_index_buffer: handle is not live");
    return;
  }
  vkCmdBindIndexBuffer(m_cmd, buffer->buffer, offset,
                       type == IndexType::U32 ? VK_INDEX_TYPE_UINT32
                                              : VK_INDEX_TYPE_UINT16);
}

void VulkanCommandContext::draw(const u32 vertex_count,
                                const u32 instance_count,
                                const u32 first_vertex,
                                const u32 first_instance) {
  vkCmdDraw(m_cmd, vertex_count, instance_count, first_vertex, first_instance);
}

void VulkanCommandContext::draw_indexed(const u32 index_count,
                                        const u32 instance_count,
                                        const u32 first_index,
                                        const i32 vertex_offset,
                                        const u32 first_instance) {
  vkCmdDrawIndexed(m_cmd, index_count, instance_count, first_index,
                   vertex_offset, first_instance);
}

void VulkanCommandContext::dispatch(const u32 groups_x, const u32 groups_y,
                                    const u32 groups_z) {
  vkCmdDispatch(m_cmd, groups_x, groups_y, groups_z);
}

void VulkanCommandContext::draw_indirect(const BufferHandle buffer,
                                         const u64 offset, const u32 draw_count,
                                         const u32 stride) {
  const VulkanBuffer *buf = m_device->buffer(buffer);
  if (buf == nullptr)
    return;
  vkCmdDrawIndirect(m_cmd, buf->buffer, offset, draw_count,
                    stride != 0 ? stride
                                : nx::cast<u32>(sizeof(DrawIndirectCommand)));
}

void VulkanCommandContext::draw_indexed_indirect(const BufferHandle buffer,
                                                 const u64 offset,
                                                 const u32 draw_count,
                                                 const u32 stride) {
  const VulkanBuffer *buf = m_device->buffer(buffer);
  if (buf == nullptr)
    return;
  vkCmdDrawIndexedIndirect(
      m_cmd, buf->buffer, offset, draw_count,
      stride != 0 ? stride : nx::cast<u32>(sizeof(DrawIndexedIndirectCommand)));
}

void VulkanCommandContext::dispatch_indirect(const BufferHandle buffer,
                                             const u64 offset) {
  const VulkanBuffer *buf = m_device->buffer(buffer);
  if (buf == nullptr)
    return;
  vkCmdDispatchIndirect(m_cmd, buf->buffer, offset);
}

void VulkanCommandContext::begin_compute_pass(const nx::string_view name) {
  if (m_in_render_pass) {
    nx::loge(log_rhi, "begin_compute_pass('{}') inside a render pass", name);
    return;
  }
  push_debug_group(name);
}

void VulkanCommandContext::end_compute_pass() { pop_debug_group(); }

} // namespace nxe::rhi
