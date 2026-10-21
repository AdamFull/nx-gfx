
#include "rendering/rhi/vulkan/vk_device.h"

#include "core/foundation/core/scope_guard.h"
#include "core/foundation/diagnostics/log.h"
#include "core/foundation/platform/filesystem.h"

#include <cstring>
#include <iterator>
#include <span>

namespace nx::gfx::detail {

/// Owner-thread snapshot of everything vkCreateGraphicsPipelines needs. Shader
/// modules are immutable Vulkan objects; their gfx::shader_handle owners remain
/// live until the worker finishes. No engine handle table is touched by the
/// worker.
struct VulkanGraphicsPipelineCompileJob {
  nx::string name;
  /// The vertex or mesh stage.
  VkShaderStageFlagBits primitive_stage = VK_SHADER_STAGE_VERTEX_BIT;
  nx::string primitive_entry;
  nx::string fragment_entry;
  VkShaderModule primitive = VK_NULL_HANDLE;
  VkShaderModule fragment = VK_NULL_HANDLE;
  gfx::format color_formats[MAX_COLOR_ATTACHMENTS] = {};
  u32 color_count = 0;
  gfx::format depth_format = gfx::format::Unknown;
  gfx::blend_state blend[MAX_COLOR_ATTACHMENTS] = {};
  gfx::primitive_topology topology = gfx::primitive_topology::TriangleList;
  gfx::cull_mode cull = gfx::cull_mode::None;
  gfx::front_face front_face = gfx::front_face::CounterClockwise;
  bool depth_test = false;
  bool depth_write = false;
  gfx::compare_op depth_compare = gfx::compare_op::LessOrEqual;
  u32 sample_count = 1;
  bool sample_shading = false;
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
        m_bind_point(other.m_bind_point), m_cull(other.m_cull),
        m_front_face(other.m_front_face), m_name(std::move(other.m_name)) {
    other.m_device = VK_NULL_HANDLE;
    other.m_pipeline = VK_NULL_HANDLE;
  }

  VulkanCompiledPipeline &operator=(VulkanCompiledPipeline &&other) noexcept {
    if (this != &other) {
      reset();
      m_device = other.m_device;
      m_pipeline = other.m_pipeline;
      m_bind_point = other.m_bind_point;
      m_cull = other.m_cull;
      m_front_face = other.m_front_face;
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
  VkCullModeFlags m_cull = VK_CULL_MODE_NONE;
  VkFrontFace m_front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  nx::string m_name;
};

/// A gfx::pipeline_job's payload: the snapshot to compile and, once compiled,
/// the native pipeline it made.
struct VulkanPipelineJob {
  const VulkanDevice *device = nullptr;
  bool compute = false;
  VulkanGraphicsPipelineCompileJob graphics;
  VulkanComputePipelineCompileJob compute_work;
  VulkanCompiledPipeline compiled;
};

namespace {

VkPrimitiveTopology
to_vk_topology(const gfx::primitive_topology topology) noexcept {
  switch (topology) {
  case gfx::primitive_topology::TriangleList:
    return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  case gfx::primitive_topology::TriangleStrip:
    return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
  case gfx::primitive_topology::LineList:
    return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
  case gfx::primitive_topology::LineStrip:
    return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
  default:
    return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
  }
}

VkCullModeFlags to_vk_cull(const gfx::cull_mode cull) noexcept {
  switch (cull) {
  case gfx::cull_mode::Front:
    return VK_CULL_MODE_FRONT_BIT;
  case gfx::cull_mode::Back:
    return VK_CULL_MODE_BACK_BIT;
  default:
    return VK_CULL_MODE_NONE;
  }
}

VkFrontFace to_vk_front_face(const gfx::front_face face) noexcept {
  return face == gfx::front_face::Clockwise ? VK_FRONT_FACE_CLOCKWISE
                                            : VK_FRONT_FACE_COUNTER_CLOCKWISE;
}

VkPipelineColorBlendAttachmentState
to_vk_blend(const gfx::blend_state &state) noexcept {
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
  case gfx::blend_mode::AlphaBlend:
    out.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    out.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    break;
  case gfx::blend_mode::Premultiplied:
    out.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    out.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    break;
  case gfx::blend_mode::Additive:
    out.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    out.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    out.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    break;
  case gfx::blend_mode::PremultipliedAdditive:
    out.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    out.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    out.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    break;
  case gfx::blend_mode::Multiply:
    out.srcColorBlendFactor = VK_BLEND_FACTOR_DST_COLOR;
    out.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    break;
  case gfx::blend_mode::Screen:
    out.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    out.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    break;
  case gfx::blend_mode::Opaque:
  case gfx::blend_mode::Count:
    out.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    out.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    break;
  }
  return out;
}

VkShaderModule create_module(const VulkanDevice &owner, VkDevice device,
                             const void *code, const usize size) noexcept {
  VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  info.codeSize = size;
  info.pCode = static_cast<const u32 *>(code);
  VkShaderModule module = VK_NULL_HANDLE;
  NX_VK_DEVICE_CHECK(owner,
                     vkCreateShaderModule(device, &info, nullptr, &module));
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
          nx::logd(nx::log::rhi, "pipeline cache is for another device, ignoring");
          initial = {};
        }
      }
    }
  }

  info.initialDataSize = initial.size();
  info.pInitialData = initial.size() ? initial.data() : nullptr;
  NX_VK_DEVICE_TRY(*this, vkCreatePipelineCache(m_device, &info, nullptr,
                                                &m_pipeline_cache));
  set_object_name(vk_handle_value(m_pipeline_cache),
                  VK_OBJECT_TYPE_PIPELINE_CACHE, "pipeline cache");

  if (initial.size())
    nx::logd(nx::log::rhi, "pipeline cache: loaded {} bytes", initial.size());
  return true;
}

bool VulkanDevice::save_pipeline_cache() noexcept {
  if (device_status() == gfx::gpu_device_status::Lost ||
      m_pipeline_cache == VK_NULL_HANDLE || m_pipeline_cache_path.empty())
    return false;

  const nx::scoped_lock<nx::mutex> held(m_pipeline_cache_lock);

  // Compiles do not take the lock and keep growing the device's cache, so its
  // size can change between the size query and the copy as often as they
  // finish. A merge takes a snapshot into a cache no compile touches.
  const VkPipelineCacheCreateInfo info{
      VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
  VkPipelineCache snapshot = VK_NULL_HANDLE;
  if (NX_VK_DEVICE_CHECK(*this, vkCreatePipelineCache(m_device, &info, nullptr,
                                                      &snapshot)) != VK_SUCCESS)
    return false;
  const nx::scope_guard destroy_snapshot([this, snapshot] {
    vkDestroyPipelineCache(m_device, snapshot, nullptr);
  });
  usize size = 0;
  if (NX_VK_DEVICE_CHECK(*this, vkMergePipelineCaches(m_device, snapshot, 1,
                                                      &m_pipeline_cache)) !=
          VK_SUCCESS ||
      NX_VK_DEVICE_CHECK(
          *this, vkGetPipelineCacheData(m_device, snapshot, &size, nullptr)) !=
          VK_SUCCESS ||
      size == 0)
    return false;
  nx::blob<u8> data(size);
  if (NX_VK_DEVICE_CHECK(*this, vkGetPipelineCacheData(m_device, snapshot,
                                                       &size, data.data())) !=
      VK_SUCCESS)
    return false;

  if (!nx::fs::file_write_atomic(m_pipeline_cache_path, {data.data(), size})) {
    nx::logw(nx::log::rhi, "pipeline cache: could not write {}",
             m_pipeline_cache_path);
    return false;
  }
  nx::logd(nx::log::rhi, "pipeline cache: saved {} bytes", size);
  return true;
}

namespace {

/// Names every OpEntryPoint in @p code. Leaves @p out empty if the blob is not
/// SPIR-V this can walk, which the callers read as "do not check".
void collect_entry_points(const void *const code, const usize bytes,
                          nx::small_vector<nx::string, 4> &out) {
  constexpr u32 SPIRV_MAGIC = 0x07230203u;
  constexpr u32 OP_ENTRY_POINT = 15u;
  constexpr usize HEADER_WORDS = 5;
  // Execution model, entry id, then the name as a NUL-padded literal.
  constexpr u32 ENTRY_PREFIX_WORDS = 3;

  const auto *const words = static_cast<const u32 *>(code);
  const usize count = bytes / sizeof(u32);
  if (count <= HEADER_WORDS || words[0] != SPIRV_MAGIC)
    return;

  for (usize i = HEADER_WORDS; i < count;) {
    const u32 op = words[i] & 0xFFFFu;
    const usize length = words[i] >> 16u;
    if (length == 0 || i + length > count)
      return;
    if (op == OP_ENTRY_POINT && length > ENTRY_PREFIX_WORDS) {
      const auto *const name =
          reinterpret_cast<const char *>(&words[i + ENTRY_PREFIX_WORDS]);
      const usize room = (length - ENTRY_PREFIX_WORDS) * sizeof(u32);
      usize n = 0;
      while (n < room && name[n] != '\0')
        ++n;
      out.push_back(nx::string(nx::string_view(name, n)));
    }
    i += length;
  }
}

[[nodiscard]] bool declares_entry(const VulkanShader &shader,
                                  const nx::string_view entry) noexcept {
  if (shader.entry_points.empty())
    return true;
  for (const nx::string &name : shader.entry_points)
    if (name.view() == entry)
      return true;
  return false;
}

} // namespace

gfx::shader_handle VulkanDevice::create_shader(const gfx::shader_desc &desc) {
  if (desc.code == nullptr || desc.code_size == 0 || desc.code_size % 4 != 0) {
    nx::loge(nx::log::rhi, "create_shader('{}'): invalid SPIR-V blob", desc.name);
    return {};
  }

  VulkanShader shader;
  shader.module = create_module(*this, m_device, desc.code, desc.code_size);
  if (shader.module == VK_NULL_HANDLE)
    return {};

  collect_entry_points(desc.code, desc.code_size, shader.entry_points);

  if (desc.reflection != nullptr)
    shader.reflection = *desc.reflection;

  for (const gfx::reflect_compute_entry &e :
       shader.reflection.compute_entries) {
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

const gfx::shader_reflection &
VulkanDevice::reflection(const gfx::shader_handle handle) const noexcept {
  static const gfx::shader_reflection empty;
  const VulkanShader *shader = m_shaders.try_get(handle);
  return shader != nullptr ? shader->reflection : empty;
}

void VulkanDevice::destroy_shader(const gfx::shader_handle handle) noexcept {
  if (VulkanShader *shader = m_shaders.try_get(handle)) {
    vkDestroyShaderModule(m_device, shader->module, nullptr);
    m_shaders.erase(handle);
  }
}

gfx::pipeline_job
VulkanDevice::prepare_pipeline(const gfx::graphics_pipeline_desc &desc) {
  VulkanPipelineJob job{.device = this};
  if (!prepare_graphics_pipeline(desc, job.graphics))
    return {};
  return detail::pipeline_job_access::adopt(
      std::move(job), &VulkanDevice::compile_job, &m_pipeline_compile_workload);
}

gfx::pipeline_job
VulkanDevice::prepare_pipeline(const gfx::compute_pipeline_desc &desc) {
  VulkanPipelineJob job{.device = this, .compute = true};
  if (!prepare_compute_pipeline(desc, job.compute_work))
    return {};
  return detail::pipeline_job_access::adopt(
      std::move(job), &VulkanDevice::compile_job, &m_pipeline_compile_workload);
}

bool VulkanDevice::compile_job(void *const payload) {
  VulkanPipelineJob &job = *static_cast<VulkanPipelineJob *>(payload);
  job.compiled = job.compute
                     ? job.device->compile_compute_pipeline(job.compute_work)
                     : job.device->compile_graphics_pipeline(job.graphics);
  return job.compiled.valid();
}

bool VulkanDevice::prepare_graphics_pipeline(
    const gfx::graphics_pipeline_desc &desc,
    VulkanGraphicsPipelineCompileJob &out) const {
  const bool mesh = desc.mesh.shader.valid();
  if (mesh && desc.vertex.shader.valid()) {
    nx::loge(nx::log::rhi,
             "create_graphics_pipeline('{}'): both a vertex and a mesh shader",
             desc.name);
    return false;
  }
  if (mesh && !m_caps.mesh_shaders) {
    nx::loge(nx::log::rhi,
             "create_graphics_pipeline('{}'): the device has no mesh shaders",
             desc.name);
    return false;
  }
  const gfx::shader_entry &primitive = mesh ? desc.mesh : desc.vertex;
  const nx::string_view stage_name = mesh ? "mesh" : "vertex";
  const VulkanShader *ps = m_shaders.try_get(primitive.shader);
  if (ps == nullptr) {
    nx::loge(nx::log::rhi, "create_graphics_pipeline('{}'): missing {} shader",
             desc.name, stage_name);
    return false;
  }

  const VulkanShader *fs = m_shaders.try_get(desc.fragment.shader);

  if (!declares_entry(*ps, primitive.entry_point)) {
    nx::loge(nx::log::rhi,
             "create_graphics_pipeline('{}'): the {} shader declares no "
             "entry point '{}'",
             desc.name, stage_name, primitive.entry_point);
    return false;
  }
  if (fs != nullptr && !desc.fragment.entry_point.empty() &&
      !declares_entry(*fs, desc.fragment.entry_point)) {
    nx::loge(nx::log::rhi,
             "create_graphics_pipeline('{}'): the fragment shader declares no "
             "entry point '{}'",
             desc.name, desc.fragment.entry_point);
    return false;
  }

  VulkanGraphicsPipelineCompileJob job;
  job.name = nx::string(desc.name);
  job.primitive_stage =
      mesh ? VK_SHADER_STAGE_MESH_BIT_EXT : VK_SHADER_STAGE_VERTEX_BIT;
  job.primitive_entry = nx::string(primitive.entry_point);
  job.fragment_entry = nx::string(desc.fragment.entry_point);
  job.primitive = ps->module;
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
  job.sample_shading =
      desc.sample_shading && m_caps.sample_shading && job.sample_count > 1;
  out = std::move(job);
  return true;
}

VulkanCompiledPipeline VulkanDevice::compile_graphics_pipeline(
    const VulkanGraphicsPipelineCompileJob &job) const {
  if (job.primitive == VK_NULL_HANDLE)
    return {};

  VkPipelineShaderStageCreateInfo stages[2] = {};
  stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage = job.primitive_stage;
  stages[0].module = job.primitive;
  stages[0].pName = job.primitive_entry.c_str();
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
  raster.frontFace = to_vk_front_face(job.front_face);
  raster.lineWidth = 1.f;

  VkPipelineMultisampleStateCreateInfo multisample{
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  multisample.rasterizationSamples =
      nx::cast<VkSampleCountFlagBits>(job.sample_count);
  multisample.sampleShadingEnable = job.sample_shading ? VK_TRUE : VK_FALSE;
  multisample.minSampleShading = job.sample_shading ? 1.f : 0.f;

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

  // Culling is dynamic so that draws differing only in it, such as mirrored
  // and two-sided instances, share one pipeline.
  const VkDynamicState dynamic_states[] = {
      VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
      VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE};
  VkPipelineDynamicStateCreateInfo dynamic{
      VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dynamic.dynamicStateCount = nx::cast<u32>(std::size(dynamic_states));
  dynamic.pDynamicStates = dynamic_states;

  VkFormat color_formats[MAX_COLOR_ATTACHMENTS] = {};
  for (u32 i = 0; i < job.color_count; ++i)
    color_formats[i] = to_vk_format(job.color_formats[i]);

  VkPipelineRenderingCreateInfo rendering{
      VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  rendering.colorAttachmentCount = job.color_count;
  rendering.pColorAttachmentFormats = color_formats;
  if (job.depth_format != gfx::format::Unknown) {
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
  compiled.m_cull = raster.cullMode;
  compiled.m_front_face = raster.frontFace;
  compiled.m_name = job.name;
  // The cache is created without EXTERNALLY_SYNCHRONIZED, so the driver
  // guards it: a lock here would run every compile one at a time.
  if (NX_VK_DEVICE_CHECK(
          *this, vkCreateGraphicsPipelines(m_device, m_pipeline_cache, 1, &info,
                                           nullptr, &compiled.m_pipeline)) !=
      VK_SUCCESS) {
    nx::loge(nx::log::rhi, "create_graphics_pipeline('{}') failed", job.name);
    return {};
  }
  return compiled;
}

bool VulkanDevice::prepare_compute_pipeline(
    const gfx::compute_pipeline_desc &desc,
    VulkanComputePipelineCompileJob &out) const {
  const VulkanShader *cs = m_shaders.try_get(desc.compute.shader);
  if (cs == nullptr) {
    nx::loge(nx::log::rhi, "create_compute_pipeline('{}'): missing shader",
             desc.name);
    return false;
  }

  if (!declares_entry(*cs, desc.compute.entry_point)) {
    nx::loge(nx::log::rhi,
             "create_compute_pipeline('{}'): the shader declares no entry "
             "point '{}'",
             desc.name, desc.compute.entry_point);
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
      nx::logw(nx::log::rhi,
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
  if (NX_VK_DEVICE_CHECK(
          *this, vkCreateComputePipelines(m_device, m_pipeline_cache, 1, &info,
                                          nullptr, &compiled.m_pipeline)) !=
      VK_SUCCESS) {
    nx::loge(nx::log::rhi, "create_compute_pipeline('{}') failed", job.name);
    return {};
  }
  return compiled;
}

gfx::pipeline_handle
VulkanDevice::publish_pipeline(gfx::pipeline_job &&opaque) {
  VulkanPipelineJob job =
      detail::pipeline_job_access::take<VulkanPipelineJob>(opaque);
  VulkanCompiledPipeline &compiled = job.compiled;
  if (job.device != this || !compiled.valid())
    return {};
  VulkanPipeline pipeline;
  pipeline.pipeline = compiled.m_pipeline;
  pipeline.bind_point = compiled.m_bind_point;
  pipeline.cull = compiled.m_cull;
  pipeline.front_face = compiled.m_front_face;
  const nx::string name = std::move(compiled.m_name);
  compiled.m_device = VK_NULL_HANDLE;
  compiled.m_pipeline = VK_NULL_HANDLE;
  gfx::pipeline_handle handle;
  {
    const nx::scoped_lock<nx::mutex> held(m_garbage_lock);
    handle = m_pipelines.insert(VulkanPipeline(pipeline));
  }
  if (!handle.valid()) {
    vkDestroyPipeline(m_device, pipeline.pipeline, nullptr);
    return {};
  }
  set_object_name(vk_handle_value(pipeline.pipeline), VK_OBJECT_TYPE_PIPELINE,
                  name.view());
  return handle;
}

void VulkanDevice::destroy_pipeline(
    const gfx::pipeline_handle handle) noexcept {
  const nx::scoped_lock<nx::mutex> held(m_garbage_lock);
  if (!m_pipelines.contains(handle))
    return;
  retire({.pipeline = handle});
}

const VulkanPipeline *
VulkanDevice::pipeline(const gfx::pipeline_handle handle) const noexcept {
  return m_pipelines.try_get(handle);
}

void VulkanCommandContext::bind_pipeline(const gfx::pipeline_handle handle) {
  if (m_device == nullptr)
    return;
  const VulkanPipeline *pipeline = m_device->pipeline(handle);
  if (pipeline == nullptr)
    return;
  vkCmdBindPipeline(m_cmd, pipeline->bind_point, pipeline->pipeline);
  if (pipeline->bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS) {
    vkCmdSetCullModeEXT(m_cmd, pipeline->cull);
    vkCmdSetFrontFaceEXT(m_cmd, pipeline->front_face);
  }
}

void VulkanCommandContext::set_cull_mode(const gfx::cull_mode cull) {
  if (m_cmd != VK_NULL_HANDLE)
    vkCmdSetCullModeEXT(m_cmd, to_vk_cull(cull));
}

void VulkanCommandContext::set_front_face(const gfx::front_face face) {
  if (m_cmd != VK_NULL_HANDLE)
    vkCmdSetFrontFaceEXT(m_cmd, to_vk_front_face(face));
}

void VulkanCommandContext::push_constants(const void *data, const u32 size,
                                          const u32 offset) {
  if (m_device == nullptr)
    return;
  vkCmdPushConstants(m_cmd, m_device->pipeline_layout(), VK_SHADER_STAGE_ALL,
                     offset, size, data);
}

void VulkanCommandContext::bind_index_buffer(const gfx::buffer_handle handle,
                                             const gfx::index_type type,
                                             const u64 offset) {
  if (m_device == nullptr)
    return;
  const VulkanBuffer *buffer = m_device->buffer(handle);
  if (buffer == nullptr || buffer->buffer == VK_NULL_HANDLE) {
    nx::loge(nx::log::rhi, "bind_index_buffer: handle is not live");
    return;
  }
  vkCmdBindIndexBuffer(m_cmd, buffer->buffer, offset,
                       type == gfx::index_type::U32 ? VK_INDEX_TYPE_UINT32
                                                    : VK_INDEX_TYPE_UINT16);
}

void VulkanCommandContext::draw(const u32 vertex_count,
                                const u32 instance_count,
                                const u32 first_vertex,
                                const u32 first_instance) {
  if (m_device == nullptr)
    return;
  vkCmdDraw(m_cmd, vertex_count, instance_count, first_vertex, first_instance);
}

void VulkanCommandContext::draw_indexed(const u32 index_count,
                                        const u32 instance_count,
                                        const u32 first_index,
                                        const i32 vertex_offset,
                                        const u32 first_instance) {
  if (m_device == nullptr)
    return;
  vkCmdDrawIndexed(m_cmd, index_count, instance_count, first_index,
                   vertex_offset, first_instance);
}

void VulkanCommandContext::dispatch(const u32 groups_x, const u32 groups_y,
                                    const u32 groups_z) {
  if (m_device == nullptr)
    return;
  vkCmdDispatch(m_cmd, groups_x, groups_y, groups_z);
}

void VulkanCommandContext::draw_indirect(const gfx::buffer_handle buffer,
                                         const u64 offset, const u32 draw_count,
                                         const u32 stride) {
  if (m_device == nullptr)
    return;
  const VulkanBuffer *buf = m_device->buffer(buffer);
  if (buf == nullptr)
    return;
  vkCmdDrawIndirect(
      m_cmd, buf->buffer, offset, draw_count,
      stride != 0 ? stride : nx::cast<u32>(sizeof(gfx::draw_indirect_command)));
}

void VulkanCommandContext::draw_mesh_tasks_indirect(
    const gfx::buffer_handle buffer, const u64 offset, const u32 draw_count,
    const u32 stride) {
  if (m_device == nullptr || !m_device->caps().mesh_shaders)
    return;
  const VulkanBuffer *buf = m_device->buffer(buffer);
  if (buf == nullptr)
    return;
  vkCmdDrawMeshTasksIndirectEXT(
      m_cmd, buf->buffer, offset, draw_count,
      stride != 0
          ? stride
          : nx::cast<u32>(sizeof(gfx::draw_mesh_tasks_indirect_command)));
}

void VulkanCommandContext::draw_indexed_indirect(
    const gfx::buffer_handle buffer, const u64 offset, const u32 draw_count,
    const u32 stride) {
  if (m_device == nullptr)
    return;
  const VulkanBuffer *buf = m_device->buffer(buffer);
  if (buf == nullptr)
    return;
  vkCmdDrawIndexedIndirect(
      m_cmd, buf->buffer, offset, draw_count,
      stride != 0 ? stride
                  : nx::cast<u32>(sizeof(gfx::draw_indexed_indirect_command)));
}

void VulkanCommandContext::dispatch_indirect(const gfx::buffer_handle buffer,
                                             const u64 offset) {
  if (m_device == nullptr)
    return;
  const VulkanBuffer *buf = m_device->buffer(buffer);
  if (buf == nullptr)
    return;
  vkCmdDispatchIndirect(m_cmd, buf->buffer, offset);
}

void VulkanCommandContext::begin_compute_pass(const nx::string_view name) {
  if (m_in_render_pass) {
    nx::loge(nx::log::rhi, "begin_compute_pass('{}') inside a render pass", name);
    return;
  }
  push_debug_group(name);
}

void VulkanCommandContext::end_compute_pass() { pop_debug_group(); }

} // namespace nx::gfx::detail
