/**
 * @file vk_pipeline.cpp
 * @brief Shader modules, graphics/compute pipelines, and the disk cache.
 */

#include "rhi/vulkan/vk_device.h"

#include "foundation/diagnostics/log.h"
#include "foundation/platform/filesystem.h"

#include <cstring>

namespace nxe::rhi {
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

/// Expands a blend preset into a colour-attachment blend state.
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
  case BlendMode::Multiply:
    out.srcColorBlendFactor = VK_BLEND_FACTOR_DST_COLOR;
    out.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    break;
  default: // Opaque
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

// Pipeline cache

bool VulkanDevice::create_pipeline_cache(const nx::string_view path) {
  m_pipeline_cache_path = path;

  VkPipelineCacheCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};

  nx::blob<u8> initial;
  if (!path.empty() && nx::fs::exists(path)) {
    if (auto read = nx::fs::file_read(path)) {
      initial = std::move(*read);
      // A cache blob is tied to the exact driver/GPU. The loader ignores an
      // incompatible one, but checking the UUID up front avoids feeding it
      // megabytes it will only discard.
      VkPhysicalDeviceProperties props{};
      vkGetPhysicalDeviceProperties(m_physical_device, &props);
      if (initial.size() >= 16 + VK_UUID_SIZE) {
        const u8 *uuid = initial.data() + 16;
        if (std::memcmp(uuid, props.pipelineCacheUUID, VK_UUID_SIZE) != 0) {
          nx::logi(log_rhi, "pipeline cache is for another device, ignoring");
          initial = {};
        }
      }
    }
  }

  info.initialDataSize = initial.size();
  info.pInitialData = initial.size() ? initial.data() : nullptr;
  NX_VK_TRY(vkCreatePipelineCache(m_device, &info, nullptr, &m_pipeline_cache));

  if (initial.size())
    nx::logi(log_rhi, "pipeline cache: loaded {} bytes", initial.size());
  return true;
}

void VulkanDevice::save_pipeline_cache() noexcept {
  if (m_pipeline_cache == VK_NULL_HANDLE || m_pipeline_cache_path.empty())
    return;

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
    nx::logi(log_rhi, "pipeline cache: saved {} bytes", size);
  else
    nx::logw(log_rhi, "pipeline cache: could not write {}",
             m_pipeline_cache_path);
}

// Shaders

ShaderHandle VulkanDevice::create_shader(const ShaderDesc &desc) {
  if (desc.code == nullptr || desc.code_size == 0 || desc.code_size % 4 != 0) {
    nx::loge(log_rhi, "create_shader('{}'): invalid SPIR-V blob", desc.name);
    return {};
  }

  VulkanShader shader;
  shader.module = create_module(m_device, desc.code, desc.code_size);
  if (shader.module == VK_NULL_HANDLE)
    return {};

  const ShaderHandle handle = m_shaders.insert(shader);
  set_object_name(vk_handle_value(shader.module), VK_OBJECT_TYPE_SHADER_MODULE,
                  desc.name);
  return handle;
}

void VulkanDevice::destroy_shader(const ShaderHandle handle) noexcept {
  // A shader module is consumed at pipeline-creation time and never touched by
  // the GPU, so it can go immediately.
  if (VulkanShader *shader = m_shaders.try_get(handle)) {
    vkDestroyShaderModule(m_device, shader->module, nullptr);
    m_shaders.erase(handle);
  }
}

// Pipelines

PipelineHandle
VulkanDevice::create_graphics_pipeline(const GraphicsPipelineDesc &desc) {
  const VulkanShader *vs = m_shaders.try_get(desc.vertex.shader);
  const VulkanShader *fs = m_shaders.try_get(desc.fragment.shader);
  if (vs == nullptr || fs == nullptr) {
    nx::loge(log_rhi, "create_graphics_pipeline('{}'): missing shader",
             desc.name);
    return {};
  }

  const nx::string vs_entry(desc.vertex.entry_point);
  const nx::string fs_entry(desc.fragment.entry_point);

  VkPipelineShaderStageCreateInfo stages[2] = {};
  stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vs->module;
  stages[0].pName = vs_entry.c_str();
  stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = fs->module;
  stages[1].pName = fs_entry.c_str();

  // No vertex input: geometry is pulled from buffers by device address.
  VkPipelineVertexInputStateCreateInfo vertex_input{
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};

  VkPipelineInputAssemblyStateCreateInfo input_assembly{
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  input_assembly.topology = to_vk_topology(desc.topology);

  VkPipelineViewportStateCreateInfo viewport{
      VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  viewport.viewportCount = 1;
  viewport.scissorCount = 1;

  VkPipelineRasterizationStateCreateInfo raster{
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  raster.polygonMode = VK_POLYGON_MODE_FILL;
  raster.cullMode = to_vk_cull(desc.cull);
  raster.frontFace = desc.front_face == FrontFace::Clockwise
                         ? VK_FRONT_FACE_CLOCKWISE
                         : VK_FRONT_FACE_COUNTER_CLOCKWISE;
  raster.lineWidth = 1.f;

  VkPipelineMultisampleStateCreateInfo multisample{
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  multisample.rasterizationSamples = nx::cast<VkSampleCountFlagBits>(
      desc.sample_count == 0 ? 1 : desc.sample_count);

  VkPipelineDepthStencilStateCreateInfo depth_stencil{
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  depth_stencil.depthTestEnable = desc.depth_test ? VK_TRUE : VK_FALSE;
  depth_stencil.depthWriteEnable = desc.depth_write ? VK_TRUE : VK_FALSE;
  depth_stencil.depthCompareOp = to_vk_compare_op(desc.depth_compare);
  depth_stencil.maxDepthBounds = 1.f;

  VkPipelineColorBlendAttachmentState blend_states[MAX_COLOR_ATTACHMENTS] = {};
  for (u32 i = 0; i < desc.color_count && i < MAX_COLOR_ATTACHMENTS; ++i)
    blend_states[i] = to_vk_blend(desc.blend[i]);

  VkPipelineColorBlendStateCreateInfo color_blend{
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  color_blend.attachmentCount = desc.color_count;
  color_blend.pAttachments = blend_states;

  const VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT,
                                           VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynamic{
      VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dynamic.dynamicStateCount = 2;
  dynamic.pDynamicStates = dynamic_states;

  VkFormat color_formats[MAX_COLOR_ATTACHMENTS] = {};
  for (u32 i = 0; i < desc.color_count && i < MAX_COLOR_ATTACHMENTS; ++i)
    color_formats[i] = to_vk_format(desc.color_formats[i]);

  VkPipelineRenderingCreateInfo rendering{
      VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  rendering.colorAttachmentCount = desc.color_count;
  rendering.pColorAttachmentFormats = color_formats;
  if (desc.depth_format != Format::Unknown) {
    rendering.depthAttachmentFormat = to_vk_format(desc.depth_format);
    if (format_is_stencil(desc.depth_format))
      rendering.stencilAttachmentFormat = to_vk_format(desc.depth_format);
  }

  VkGraphicsPipelineCreateInfo info{
      VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  info.pNext = &rendering;
  info.stageCount = 2;
  info.pStages = stages;
  info.pVertexInputState = &vertex_input;
  info.pInputAssemblyState = &input_assembly;
  info.pViewportState = &viewport;
  info.pRasterizationState = &raster;
  info.pMultisampleState = &multisample;
  info.pDepthStencilState = &depth_stencil;
  info.pColorBlendState = &color_blend;
  info.pDynamicState = &dynamic;
  info.layout = m_pipeline_layout;

  VulkanPipeline pipeline;
  pipeline.bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;
  if (NX_VK_CHECK(vkCreateGraphicsPipelines(
          m_device, m_pipeline_cache, 1, &info, nullptr, &pipeline.pipeline)) !=
      VK_SUCCESS) {
    nx::loge(log_rhi, "create_graphics_pipeline('{}') failed", desc.name);
    return {};
  }

  const PipelineHandle handle = m_pipelines.insert(pipeline);
  set_object_name(vk_handle_value(pipeline.pipeline), VK_OBJECT_TYPE_PIPELINE,
                  desc.name);
  return handle;
}

PipelineHandle
VulkanDevice::create_compute_pipeline(const ComputePipelineDesc &desc) {
  const VulkanShader *cs = m_shaders.try_get(desc.compute.shader);
  if (cs == nullptr) {
    nx::loge(log_rhi, "create_compute_pipeline('{}'): missing shader",
             desc.name);
    return {};
  }

  const nx::string entry(desc.compute.entry_point);

  VkComputePipelineCreateInfo info{
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  info.stage.module = cs->module;
  info.stage.pName = entry.c_str();
  info.layout = m_pipeline_layout;

  VulkanPipeline pipeline;
  pipeline.bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
  if (NX_VK_CHECK(vkCreateComputePipelines(m_device, m_pipeline_cache, 1, &info,
                                           nullptr, &pipeline.pipeline)) !=
      VK_SUCCESS) {
    nx::loge(log_rhi, "create_compute_pipeline('{}') failed", desc.name);
    return {};
  }

  const PipelineHandle handle = m_pipelines.insert(pipeline);
  set_object_name(vk_handle_value(pipeline.pipeline), VK_OBJECT_TYPE_PIPELINE,
                  desc.name);
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

// Command recording

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

} // namespace nxe::rhi
