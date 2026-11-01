#include "src/vulkan/vk_resource_ledger.h"

#include "core/foundation/platform/os.h"

#include <cstring>

namespace nx::gfx::detail {

void VulkanResourceLedger::created(const gpu_resource_kind kind, const u64 id,
                                   const u64 bytes,
                                   const nx::string_view name) noexcept {
  Live &live = m_live[static_cast<u32>(kind)];
  live.bytes.fetch_add(bytes, std::memory_order_relaxed);
  live.count.fetch_add(1, std::memory_order_relaxed);
  note(kind, id, bytes, name, true);
}

void VulkanResourceLedger::destroyed(const gpu_resource_kind kind, const u64 id,
                                     const u64 bytes,
                                     const nx::string_view name) noexcept {
  Live &live = m_live[static_cast<u32>(kind)];
  live.bytes.fetch_sub(bytes, std::memory_order_relaxed);
  live.count.fetch_sub(1, std::memory_order_relaxed);
  note(kind, id, bytes, name, false);
}

void VulkanResourceLedger::note(const gpu_resource_kind kind, const u64 id,
                                const u64 bytes, const nx::string_view name,
                                const bool created) noexcept {
  if (!m_on.load(std::memory_order_acquire))
    return;
  gpu_resource_event event;
  event.at_ns = nx::os::ticks_ns();
  event.bytes = bytes;
  event.id = id;
  event.kind = kind;
  event.created = created;
  event.name_size = nx::cast<u8>(nx::min(name.size(), GPU_RESOURCE_NAME_BYTES));
  if (event.name_size != 0)
    std::memcpy(event.name, name.data(), event.name_size);

  const nx::scoped_lock<nx::mutex> held(m_lock);
  if (m_count == m_events.size()) {
    m_dropped.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  m_events[(m_first + m_count) % m_events.size()] = event;
  ++m_count;
}

void VulkanResourceLedger::set_events(const bool on) {
  if (on) {
    const nx::scoped_lock<nx::mutex> held(m_lock);
    if (m_events.empty())
      m_events.resize(GPU_RESOURCE_EVENTS_HELD);
  }
  m_on.store(on, std::memory_order_release);
}

usize VulkanResourceLedger::read(
    const std::span<gpu_resource_event> out) noexcept {
  const nx::scoped_lock<nx::mutex> held(m_lock);
  const usize count = nx::min(out.size(), m_count);
  for (usize i = 0; i < count; ++i)
    out[i] = m_events[(m_first + i) % m_events.size()];
  if (count != 0) {
    m_first = (m_first + count) % m_events.size();
    m_count -= count;
  }
  return count;
}

void VulkanResourceLedger::totals(gpu_memory_stats &out) const noexcept {
  const auto totals = [this](const gpu_resource_kind kind) {
    const Live &live = m_live[static_cast<u32>(kind)];
    return gpu_resource_totals{live.bytes.load(std::memory_order_relaxed),
                               live.count.load(std::memory_order_relaxed)};
  };
  out.buffers = totals(gpu_resource_kind::Buffer);
  out.textures = totals(gpu_resource_kind::Texture);
  out.memory_heaps = totals(gpu_resource_kind::MemoryHeap);
}

} // namespace nx::gfx::detail
