#include "framework/nxtest.h"

#include "nx/gfx/pipeline_job.h"

namespace gfx = nx::gfx;

namespace {

struct Payload {
  int value = 0;
  int compiles = 0;
};

bool compile_payload(void *const payload) {
  Payload &p = *static_cast<Payload *>(payload);
  ++p.compiles;
  p.value += 1;
  return p.value > 0;
}

[[nodiscard]] gfx::pipeline_job make_job(const int value,
                                         nx::workload_tracker *workload) {
  return gfx::detail::pipeline_job_access::adopt(Payload{.value = value},
                                                 &compile_payload, workload);
}

} // namespace

TEST_CASE("pipeline job: compile runs the backend step once, then take "
          "recovers the payload") {
  gfx::pipeline_job job = make_job(41, nullptr);
  REQUIRE(job.valid());
  CHECK_FALSE(job.compiled());

  CHECK(job.compile());
  CHECK(job.compile());
  CHECK(job.compiled());

  const Payload payload = gfx::detail::pipeline_job_access::take<Payload>(job);
  CHECK(payload.value == 42);
  CHECK(payload.compiles == 1);
  CHECK_FALSE(job.valid());
}

TEST_CASE("pipeline job: take refuses another type and an uncompiled job "
          "without consuming either") {
  gfx::pipeline_job job = make_job(41, nullptr);
  REQUIRE(job.valid());

  CHECK(gfx::detail::pipeline_job_access::take<Payload>(job).value == 0);
  CHECK(job.valid());

  REQUIRE(job.compile());
  CHECK(gfx::detail::pipeline_job_access::take<int>(job) == 0);
  CHECK(job.valid());

  CHECK(gfx::detail::pipeline_job_access::take<Payload>(job).value == 42);
  CHECK_FALSE(job.valid());
}

TEST_CASE("pipeline job: a failed compile is not compiled and gives up no "
          "payload") {
  gfx::pipeline_job job = make_job(-5, nullptr);
  CHECK_FALSE(job.compile());
  CHECK_FALSE(job.compiled());
  CHECK(gfx::detail::pipeline_job_access::take<Payload>(job).value == 0);
}

TEST_CASE("pipeline job: the compile workload counts a compiled job once and "
          "rejects one dropped before it compiles") {
  nx::workload_tracker workload{"test::compile"};
  {
    gfx::pipeline_job job = make_job(1, &workload);
    gfx::pipeline_job moved = std::move(job);
    CHECK(moved.compile());
  }
  {
    gfx::pipeline_job dropped = make_job(1, &workload);
    gfx::pipeline_job moved = std::move(dropped);
  }
  const nx::workload_statistics stats = workload.statistics();
  CHECK(stats.submitted == 2);
  CHECK(stats.completed == 1);
  CHECK(stats.rejected == 1);
  CHECK(stats.queued == 0);
}

TEST_CASE("pipeline job: compile_pipeline_async compiles on the pool and "
          "hands the job back") {
  nx::thread_pool workers(
      {.thread_count = 1, .name = "nx-test-job", .collect_statistics = false});
  nx::future<gfx::pipeline_job> pending =
      gfx::compile_pipeline_async(workers, make_job(41, nullptr));
  REQUIRE(pending.valid());
  gfx::pipeline_job job = pending.get();
  CHECK(job.compiled());
  CHECK(gfx::detail::pipeline_job_access::take<Payload>(job).value == 42);

  CHECK_FALSE(
      gfx::compile_pipeline_async(workers, gfx::pipeline_job{}).valid());
}
