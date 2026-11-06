#include "src/vulkan/vk_device.h"

#include "core/foundation/containers/small_vector.h"
#include "core/foundation/diagnostics/log.h"
#include "nx/gfx/ray.h"

#include <cstddef>

namespace nx::gfx::detail {

static_assert(sizeof(gfx::ray_instance) ==
              sizeof(VkAccelerationStructureInstanceKHR));
static_assert(offsetof(gfx::ray_instance, bottom_level) ==
              offsetof(VkAccelerationStructureInstanceKHR,
                       accelerationStructureReference));

namespace {

/// What one build of a structure is made of: its geometries and, for each,
/// how many primitives it holds.
struct BuildInput {
  nx::small_vector<VkAccelerationStructureGeometryKHR, 4> geometries;
  nx::small_vector<u32, 4> counts;
  VkAccelerationStructureBuildGeometryInfoKHR info{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
};

[[nodiscard]] VkBuildAccelerationStructureFlagsKHR
to_vk_build_flags(const gfx::acceleration_structure_flags flags) noexcept {
  VkBuildAccelerationStructureFlagsKHR out = 0;
  if (flags & gfx::acceleration_structure_flag::PreferFastTrace)
    out |= VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  if (flags & gfx::acceleration_structure_flag::PreferFastBuild)
    out |= VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
  if (flags & gfx::acceleration_structure_flag::AllowUpdate)
    out |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
  if (flags & gfx::acceleration_structure_flag::AllowCompaction)
    out |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
  return out;
}

[[nodiscard]] VkGeometryFlagsKHR
to_vk_geometry_flags(const gfx::ray_geometry_flags flags) noexcept {
  VkGeometryFlagsKHR out = 0;
  if (flags & gfx::ray_geometry_flag::Opaque)
    out |= VK_GEOMETRY_OPAQUE_BIT_KHR;
  if (flags & gfx::ray_geometry_flag::NoDuplicateAnyHit)
    out |= VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
  return out;
}

/// False, logged, for a description no device can build.
[[nodiscard]] bool describe(const gfx::device_caps &caps,
                            const gfx::bottom_level_desc &desc,
                            BuildInput &out) {
  if (desc.geometries.empty() ||
      desc.geometries.size() > caps.max_ray_geometries) {
    nx::loge(nx::log::rhi, "a bottom level of {} geometries (1 to {})",
             desc.geometries.size(), caps.max_ray_geometries);
    return false;
  }
  for (const gfx::ray_triangles &triangles : desc.geometries) {
    if (triangles.triangle_count == 0 || triangles.vertex_count == 0 ||
        triangles.triangle_count > caps.max_ray_primitives) {
      nx::loge(nx::log::rhi,
               "a bottom-level geometry of {} triangles over {} vertices",
               triangles.triangle_count, triangles.vertex_count);
      return false;
    }
    VkAccelerationStructureGeometryKHR geometry{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags = to_vk_geometry_flags(triangles.flags);
    VkAccelerationStructureGeometryTrianglesDataKHR &data =
        geometry.geometry.triangles;
    data.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    data.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    data.vertexData.deviceAddress = triangles.vertex_address;
    data.vertexStride = triangles.vertex_stride;
    data.maxVertex = triangles.vertex_count - 1;
    data.indexType = triangles.index_address == 0 ? VK_INDEX_TYPE_NONE_KHR
                     : triangles.indices == gfx::index_type::U16
                         ? VK_INDEX_TYPE_UINT16
                         : VK_INDEX_TYPE_UINT32;
    data.indexData.deviceAddress = triangles.index_address;
    data.transformData.deviceAddress = triangles.transform_address;
    out.geometries.push_back(geometry);
    out.counts.push_back(triangles.triangle_count);
  }
  out.info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  out.info.flags = to_vk_build_flags(desc.flags);
  out.info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  out.info.geometryCount = nx::cast<u32>(out.geometries.size());
  out.info.pGeometries = out.geometries.data();
  return true;
}

[[nodiscard]] bool describe(const gfx::device_caps &caps,
                            const gfx::top_level_desc &desc, BuildInput &out) {
  if (desc.instance_count == 0 ||
      desc.instance_count > caps.max_ray_instances) {
    nx::loge(nx::log::rhi, "a top level of {} instances (1 to {})",
             desc.instance_count, caps.max_ray_instances);
    return false;
  }
  VkAccelerationStructureGeometryKHR geometry{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  geometry.geometry.instances.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
  geometry.geometry.instances.arrayOfPointers = VK_FALSE;
  geometry.geometry.instances.data.deviceAddress = desc.instance_address;
  out.geometries.push_back(geometry);
  out.counts.push_back(desc.instance_count);
  out.info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  out.info.flags = to_vk_build_flags(desc.flags);
  out.info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  out.info.geometryCount = 1;
  out.info.pGeometries = out.geometries.data();
  return true;
}

[[nodiscard]] gfx::acceleration_structure_sizes
query_sizes(const VkDevice device, BuildInput &input) noexcept {
  VkAccelerationStructureBuildSizesInfoKHR sizes{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetAccelerationStructureBuildSizesKHR(
      device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &input.info,
      input.counts.data(), &sizes);
  return {.structure = sizes.accelerationStructureSize,
          .build_scratch = sizes.buildScratchSize,
          .update_scratch = sizes.updateScratchSize};
}

} // namespace

gfx::acceleration_structure_sizes VulkanDevice::acceleration_structure_sizes(
    const gfx::bottom_level_desc &desc) const {
  BuildInput input;
  if (!m_caps.acceleration_structures || !describe(m_caps, desc, input))
    return {};
  return query_sizes(m_device, input);
}

gfx::acceleration_structure_sizes VulkanDevice::acceleration_structure_sizes(
    const gfx::top_level_desc &desc) const {
  BuildInput input;
  if (!m_caps.acceleration_structures || !describe(m_caps, desc, input))
    return {};
  return query_sizes(m_device, input);
}

gfx::acceleration_structure_handle VulkanDevice::create_acceleration_structure(
    const gfx::acceleration_structure_desc &desc) {
  if (!m_caps.acceleration_structures || desc.size == 0) {
    nx::loge(nx::log::rhi, "create_acceleration_structure('{}'): {}", desc.name,
             desc.size == 0 ? "zero size"
                            : "the device has no acceleration structures");
    return {};
  }
  VulkanAccelerationStructure structure;
  structure.kind = desc.kind;
  structure.size = desc.size;
  structure.flags = desc.flags;
  structure.buffer =
      create_buffer({.name = desc.name,
                     .size = desc.size,
                     .usage = gfx::buffer_usage::DeviceAddress,
                     .memory = gfx::memory_usage::DeviceLocal},
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
  const VulkanBuffer *const storage = buffer(structure.buffer);
  if (storage == nullptr)
    return {};

  VkAccelerationStructureCreateInfoKHR info{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  info.buffer = storage->buffer;
  info.size = desc.size;
  info.type = desc.kind == gfx::acceleration_structure_kind::Top
                  ? VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR
                  : VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  if (NX_VK_DEVICE_CHECK(*this, vkCreateAccelerationStructureKHR(
                                    m_device, &info, nullptr,
                                    &structure.structure)) != VK_SUCCESS) {
    destroy_buffer(structure.buffer);
    return {};
  }
  VkAccelerationStructureDeviceAddressInfoKHR where{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
  where.accelerationStructure = structure.structure;
  structure.address =
      vkGetAccelerationStructureDeviceAddressKHR(m_device, &where);
  set_object_name(vk_handle_value(structure.structure),
                  VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR, desc.name);

  const nx::scoped_lock<nx::mutex> held(m_garbage_lock);
  return m_acceleration_structures.insert(
      VulkanAccelerationStructure(structure));
}

void VulkanDevice::destroy_acceleration_structure(
    const gfx::acceleration_structure_handle handle) noexcept {
  const nx::scoped_lock<nx::mutex> held(m_garbage_lock);
  const VulkanAccelerationStructure *const structure =
      m_acceleration_structures.try_get(handle);
  if (structure == nullptr)
    return;
  retire({.buffer = structure->buffer, .acceleration_structure = handle});
}

namespace {

/// Records one build, or an update in place, into @p cmd once the checks
/// every kind shares pass.
[[nodiscard]] bool record_build(const VkCommandBuffer cmd,
                                const VulkanDevice &device,
                                const bool in_render_pass,
                                const gfx::acceleration_structure_handle target,
                                const gfx::acceleration_structure_kind kind,
                                const bool update, BuildInput &input,
                                const u64 scratch_address) {
  const gfx::device_caps &caps = device.caps();
  const VulkanAccelerationStructure *const structure =
      device.acceleration_structure(target);
  const u32 alignment =
      nx::max(caps.acceleration_structure_scratch_alignment, 1u);
  const char *why = nullptr;
  if (structure == nullptr)
    why = "no such structure";
  else if (structure->kind != kind)
    why = "a structure of the other kind";
  else if (to_vk_build_flags(structure->flags) != input.info.flags)
    why = "a structure made with other flags";
  else if (update &&
           !(structure->flags & gfx::acceleration_structure_flag::AllowUpdate))
    why = "an update of a structure not made AllowUpdate";
  else if (in_render_pass)
    why = "inside a render pass";
  else if (scratch_address == 0 || scratch_address % alignment != 0)
    why = "scratch missing or misaligned";
  else if (query_sizes(device.vk_device(), input).structure > structure->size)
    why = "the structure is too small";
  if (why != nullptr) {
    nx::loge(nx::log::rhi, "build_acceleration_structure: {}", why);
    return false;
  }

  input.info.dstAccelerationStructure = structure->structure;
  input.info.scratchData.deviceAddress = scratch_address;
  if (update) {
    input.info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
    input.info.srcAccelerationStructure = structure->structure;
  }
  nx::small_vector<VkAccelerationStructureBuildRangeInfoKHR, 4> ranges;
  for (const u32 count : input.counts)
    ranges.push_back({.primitiveCount = count});
  const VkAccelerationStructureBuildRangeInfoKHR *const range = ranges.data();
  vkCmdBuildAccelerationStructuresKHR(cmd, 1, &input.info, &range);
  return true;
}

} // namespace

bool VulkanCommandContext::build_acceleration_structure(
    const gfx::acceleration_structure_handle target,
    const gfx::bottom_level_desc &desc, const u64 scratch_address) {
  if (m_device == nullptr || m_cmd == VK_NULL_HANDLE)
    return false;
  BuildInput input;
  if (!m_device->caps().acceleration_structures ||
      !describe(m_device->caps(), desc, input))
    return false;
  return record_build(m_cmd, *m_device, m_in_render_pass, target,
                      gfx::acceleration_structure_kind::Bottom, false, input,
                      scratch_address);
}

bool VulkanCommandContext::build_acceleration_structure(
    const gfx::acceleration_structure_handle target,
    const gfx::top_level_desc &desc, const u64 scratch_address) {
  if (m_device == nullptr || m_cmd == VK_NULL_HANDLE)
    return false;
  BuildInput input;
  if (!m_device->caps().acceleration_structures ||
      !describe(m_device->caps(), desc, input))
    return false;
  if (desc.instance_address % 16 != 0) {
    nx::loge(nx::log::rhi, "build_acceleration_structure: instances at an "
                           "address not 16 aligned");
    return false;
  }
  return record_build(m_cmd, *m_device, m_in_render_pass, target,
                      gfx::acceleration_structure_kind::Top, false, input,
                      scratch_address);
}

bool VulkanCommandContext::update_acceleration_structure(
    const gfx::acceleration_structure_handle target,
    const gfx::bottom_level_desc &desc, const u64 scratch_address) {
  if (m_device == nullptr || m_cmd == VK_NULL_HANDLE)
    return false;
  BuildInput input;
  if (!m_device->caps().acceleration_structures ||
      !describe(m_device->caps(), desc, input))
    return false;
  return record_build(m_cmd, *m_device, m_in_render_pass, target,
                      gfx::acceleration_structure_kind::Bottom, true, input,
                      scratch_address);
}

bool VulkanCommandContext::update_acceleration_structure(
    const gfx::acceleration_structure_handle target,
    const gfx::top_level_desc &desc, const u64 scratch_address) {
  if (m_device == nullptr || m_cmd == VK_NULL_HANDLE)
    return false;
  BuildInput input;
  if (!m_device->caps().acceleration_structures ||
      !describe(m_device->caps(), desc, input))
    return false;
  if (desc.instance_address % 16 != 0) {
    nx::loge(nx::log::rhi, "update_acceleration_structure: instances at an "
                           "address not 16 aligned");
    return false;
  }
  return record_build(m_cmd, *m_device, m_in_render_pass, target,
                      gfx::acceleration_structure_kind::Top, true, input,
                      scratch_address);
}

gfx::query_pool_handle
VulkanDevice::create_compacted_size_pool(const u32 count) {
  if (!m_caps.acceleration_structures || count == 0)
    return {};
  VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
  info.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
  info.queryCount = count;
  VulkanQueryPool queries;
  queries.count = count;
  queries.type = info.queryType;
  if (NX_VK_DEVICE_CHECK(*this, vkCreateQueryPool(m_device, &info, nullptr,
                                                  &queries.pool)) != VK_SUCCESS)
    return {};
  set_object_name(vk_handle_value(queries.pool), VK_OBJECT_TYPE_QUERY_POOL,
                  "compacted sizes");
  return m_query_pools.insert(queries);
}

bool VulkanDevice::read_compacted_sizes(const gfx::query_pool_handle handle,
                                        const std::span<u64> out) const {
  const VulkanQueryPool *const queries = m_query_pools.try_get(handle);
  if (queries == nullptr || out.empty() ||
      queries->type != VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR)
    return false;
  const u32 count = nx::min(queries->count, nx::cast<u32>(out.size()));
  return NX_VK_DEVICE_CHECK(*this, vkGetQueryPoolResults(
                                       m_device, queries->pool, 0, count,
                                       nx::cast<usize>(count) * sizeof(u64),
                                       out.data(), sizeof(u64),
                                       VK_QUERY_RESULT_64_BIT)) == VK_SUCCESS;
}

void VulkanCommandContext::write_compacted_sizes(
    const std::span<const gfx::acceleration_structure_handle> structures,
    const gfx::query_pool_handle pool, const u32 first) {
  if (m_device == nullptr || m_cmd == VK_NULL_HANDLE || structures.empty())
    return;
  const VulkanQueryPool *const queries = m_device->query_pool(pool);
  const char *why = nullptr;
  if (queries == nullptr ||
      queries->type != VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR)
    why = "not a pool of compacted sizes";
  else if (first > queries->count || structures.size() > queries->count - first)
    why = "more sizes than the pool holds";
  else if (m_in_render_pass)
    why = "inside a render pass";
  nx::small_vector<VkAccelerationStructureKHR, 8> natives;
  for (const gfx::acceleration_structure_handle handle : structures) {
    if (why != nullptr)
      break;
    const VulkanAccelerationStructure *const structure =
        m_device->acceleration_structure(handle);
    if (structure == nullptr ||
        !(structure->flags & gfx::acceleration_structure_flag::AllowCompaction))
      why = "a structure not made AllowCompaction";
    else
      natives.push_back(structure->structure);
  }
  if (why != nullptr) {
    nx::loge(nx::log::rhi, "write_compacted_sizes: {}", why);
    return;
  }
  vkCmdWriteAccelerationStructuresPropertiesKHR(
      m_cmd, nx::cast<u32>(natives.size()), natives.data(),
      VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, queries->pool,
      first);
}

bool VulkanCommandContext::copy_acceleration_structure(
    const gfx::acceleration_structure_handle source,
    const gfx::acceleration_structure_handle target, const bool compact) {
  if (m_device == nullptr || m_cmd == VK_NULL_HANDLE)
    return false;
  const VulkanAccelerationStructure *const from =
      m_device->acceleration_structure(source);
  const VulkanAccelerationStructure *const to =
      m_device->acceleration_structure(target);
  const char *why = nullptr;
  if (from == nullptr || to == nullptr)
    why = "no such structure";
  else if (from->kind != to->kind || from->flags != to->flags)
    why = "structures of another kind or flags";
  else if (compact &&
           !(from->flags & gfx::acceleration_structure_flag::AllowCompaction))
    why = "compacting a structure not made AllowCompaction";
  else if (!compact && to->size < from->size)
    why = "the target is smaller than the source";
  else if (m_in_render_pass)
    why = "inside a render pass";
  if (why != nullptr) {
    nx::loge(nx::log::rhi, "copy_acceleration_structure: {}", why);
    return false;
  }
  VkCopyAccelerationStructureInfoKHR info{
      VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR};
  info.src = from->structure;
  info.dst = to->structure;
  info.mode = compact ? VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR
                      : VK_COPY_ACCELERATION_STRUCTURE_MODE_CLONE_KHR;
  vkCmdCopyAccelerationStructureKHR(m_cmd, &info);
  return true;
}

} // namespace nx::gfx::detail
