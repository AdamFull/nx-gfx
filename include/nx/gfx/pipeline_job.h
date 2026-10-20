#pragma once

#include "core/foundation/core/foundation.h"
#include "core/foundation/core/type_index.h"
#include "core/foundation/threading/thread_pool.h"
#include "core/foundation/threading/workload.h"

#include <utility>

namespace nx::gfx {

namespace detail {
struct pipeline_job_access;
}

/// A pipeline on its way to the device: prepare_pipeline() makes it on the
/// device's owner thread, copying every input; compile() runs the driver
/// compilation on any thread, touching no device table; publish_pipeline()
/// turns it into a handle on the owner thread. Dropping a job cancels it.
/// The payload is the backend's own, so native objects never reach callers.
class pipeline_job {
public:
  pipeline_job() = default;
  ~pipeline_job() { reset(); }

  pipeline_job(const pipeline_job &) = delete;
  pipeline_job &operator=(const pipeline_job &) = delete;

  pipeline_job(pipeline_job &&other) noexcept { take_from(other); }

  pipeline_job &operator=(pipeline_job &&other) noexcept {
    if (this != &other) {
      reset();
      take_from(other);
    }
    return *this;
  }

  /// Holds a job: prepare_pipeline() accepted the description.
  [[nodiscard]] bool valid() const noexcept { return m_payload != nullptr; }
  /// compile() ran and the driver made the pipeline.
  [[nodiscard]] bool compiled() const noexcept { return m_compiled; }

  /// The driver compilation, once; later calls answer the first's result.
  bool compile() {
    if (m_payload == nullptr || m_compile == nullptr)
      return m_compiled;
    nx::workload_tracker *const workload = std::exchange(m_workload, nullptr);
    void *const payload = m_payload;
    const auto run = std::exchange(m_compile, nullptr);
    m_compiled = workload != nullptr
                     ? nx::measure_workload(
                           *workload, m_ticket, nx::cancellation_token{},
                           [payload, run] { return run(payload); })
                     : run(payload);
    return m_compiled;
  }

private:
  friend struct detail::pipeline_job_access;

  void take_from(pipeline_job &other) noexcept {
    m_payload = std::exchange(other.m_payload, nullptr);
    m_destroy = std::exchange(other.m_destroy, nullptr);
    m_compile = std::exchange(other.m_compile, nullptr);
    m_type = std::exchange(other.m_type, 0);
    m_workload = std::exchange(other.m_workload, nullptr);
    m_ticket = other.m_ticket;
    m_compiled = std::exchange(other.m_compiled, false);
  }

  void reset() noexcept {
    if (m_workload != nullptr)
      m_workload->reject(m_ticket);
    if (m_payload != nullptr)
      m_destroy(m_payload);
    m_payload = nullptr;
    m_destroy = nullptr;
    m_compile = nullptr;
    m_type = 0;
    m_workload = nullptr;
    m_compiled = false;
  }

  void *m_payload = nullptr;
  void (*m_destroy)(void *) noexcept = nullptr;
  bool (*m_compile)(void *) = nullptr;
  // Identifies the payload type by value (nx::type_index), not by the
  // address of a per-TU static - see msvc-internal-linkage-trap in project
  // memory for why an address-identity scheme is not safe on this toolchain.
  u32 m_type = 0;
  /// Counts the compilation from submission; rejected if it never runs.
  nx::workload_tracker *m_workload = nullptr;
  nx::workload_ticket m_ticket;
  bool m_compiled = false;
};

/// Compiles @p job on @p workers and hands it back for publication.
[[nodiscard]] inline nx::future<pipeline_job> compile_pipeline_async(
    nx::thread_pool &workers, pipeline_job &&job,
    const nx::task_priority priority = nx::task_priority::Normal) {
  if (!job.valid())
    return {};
  return workers.async(priority, [job = std::move(job)]() mutable {
    (void)job.compile();
    return std::move(job);
  });
}

namespace detail {

/// Backend implementation bridge. Kept out of the public device contract so
/// common rendering code can only move, compile, and publish opaque jobs.
struct pipeline_job_access {
  /// @p compile runs on a worker with the payload; true when it made the
  /// native pipeline, which the payload then holds.
  template <typename T>
  [[nodiscard]] static pipeline_job adopt(T &&payload, bool (*compile)(void *),
                                          nx::workload_tracker *workload) {
    using Value = std::decay_t<T>;
    Value *const stored = nx::allocate<Value>(std::forward<T>(payload));
    if (stored == nullptr)
      return {};
    pipeline_job out;
    out.m_payload = stored;
    out.m_destroy = [](void *value) noexcept {
      nx::release(static_cast<Value *>(value));
    };
    out.m_compile = compile;
    out.m_type = nx::type_index<Value>();
    if (workload != nullptr) {
      out.m_workload = workload;
      out.m_ticket = workload->submit();
    }
    return out;
  }

  /// The payload of a compiled job, or a default one when @p job is not
  /// compiled or holds another type; the job is emptied only on success.
  template <typename T>
  [[nodiscard]] static T take(pipeline_job &job) noexcept {
    if (job.m_payload == nullptr || !job.m_compiled ||
        job.m_type != nx::type_index<T>())
      return {};
    T out = std::move(*static_cast<T *>(job.m_payload));
    job.reset();
    return out;
  }
};

} // namespace detail

} // namespace nx::gfx
