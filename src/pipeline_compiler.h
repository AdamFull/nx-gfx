#pragma once

#include "core/foundation/core/foundation.h"

#include <utility>

namespace nxe::rhi {

namespace detail {
struct CompiledPipelineAccess;
}

/// Backend-owned native pipeline produced away from the render owner. The
/// payload is deliberately opaque: descriptor translation, native object
/// lifetime, and publication remain RHI responsibilities rather than leaking
/// into the application layer.
class CompiledPipeline {
public:
  CompiledPipeline() = default;
  ~CompiledPipeline() { reset(); }

  CompiledPipeline(const CompiledPipeline &) = delete;
  CompiledPipeline &operator=(const CompiledPipeline &) = delete;

  CompiledPipeline(CompiledPipeline &&other) noexcept
      : m_payload(other.m_payload), m_destroy(other.m_destroy),
        m_type(other.m_type) {
    other.m_payload = nullptr;
    other.m_destroy = nullptr;
    other.m_type = nullptr;
  }

  CompiledPipeline &operator=(CompiledPipeline &&other) noexcept {
    if (this != &other) {
      reset();
      m_payload = other.m_payload;
      m_destroy = other.m_destroy;
      m_type = other.m_type;
      other.m_payload = nullptr;
      other.m_destroy = nullptr;
      other.m_type = nullptr;
    }
    return *this;
  }

  [[nodiscard]] bool valid() const noexcept { return m_payload != nullptr; }

private:
  friend struct detail::CompiledPipelineAccess;

  template <typename T>
  [[nodiscard]] static CompiledPipeline adopt(T &&payload) {
    using Value = std::decay_t<T>;
    Value *const stored = nx::allocate<Value>(std::forward<T>(payload));
    if (stored == nullptr)
      return {};
    CompiledPipeline out;
    out.m_payload = stored;
    out.m_destroy = [](void *value) noexcept {
      nx::release(static_cast<Value *>(value));
    };
    out.m_type = type_id<Value>();
    return out;
  }

  template <typename T> [[nodiscard]] T take() noexcept {
    if (m_payload == nullptr || m_type != type_id<T>())
      return {};
    T out = std::move(*static_cast<T *>(m_payload));
    m_destroy(m_payload);
    m_payload = nullptr;
    m_destroy = nullptr;
    m_type = nullptr;
    return out;
  }

  template <typename T> [[nodiscard]] static const void *type_id() noexcept {
    static const u8 token = 0;
    return &token;
  }

  void reset() noexcept {
    if (m_payload != nullptr)
      m_destroy(m_payload);
    m_payload = nullptr;
    m_destroy = nullptr;
    m_type = nullptr;
  }

  void *m_payload = nullptr;
  void (*m_destroy)(void *) noexcept = nullptr;
  const void *m_type = nullptr;
};

namespace detail {

/// Backend implementation bridge. Kept out of the public device contract so
/// common rendering code can only move, inspect, and publish opaque results.
struct CompiledPipelineAccess {
  template <typename T>
  [[nodiscard]] static CompiledPipeline adopt(T &&payload) {
    return CompiledPipeline::adopt(std::forward<T>(payload));
  }

  template <typename T>
  [[nodiscard]] static T take(CompiledPipeline &pipeline) noexcept {
    return pipeline.take<T>();
  }
};

} // namespace detail

} // namespace nxe::rhi
