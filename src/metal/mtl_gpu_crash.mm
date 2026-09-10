#include "core/rendering/rhi/metal/mtl_gpu_crash.h"

#include "core/rendering/rhi/gpu_crash.h"
#include "core/rendering/rhi/metal/mtl_internal.h"

#include "core/foundation/diagnostics/log.h"
#include "core/foundation/strings/format.h"

#include <atomic>
#include <utility>

namespace nxe::rhi {
namespace {

nx::string utf8(NSString *text) {
  const char *const chars = text != nil ? [text UTF8String] : nullptr;
  return nx::string(chars != nullptr ? chars : "");
}

nx::string_view encoder_state_name(const MTLCommandEncoderErrorState state) {
  switch (state) {
  case MTLCommandEncoderErrorStateAffected:
    return "affected";
  case MTLCommandEncoderErrorStateCompleted:
    return "completed";
  case MTLCommandEncoderErrorStateFaulted:
    return "faulted";
  case MTLCommandEncoderErrorStatePending:
    return "pending";
  default:
    return "unknown";
  }
}

} // namespace
} // namespace nxe::rhi

@interface NXMetalGpuCrashObserver : NSObject {
  std::atomic<bool> _lost;
  std::atomic<bool> _reported;
  nx::string _device;
  bool _reportingEnabled;
}
- (instancetype)initWithDeviceName:(nx::string_view)deviceName
                   reportingEnabled:(bool)reportingEnabled;
- (bool)deviceLost;
- (void)inspect:(id<MTLCommandBuffer>)buffer
          label:(nx::string_view)label
          frame:(u64)frame;
@end

@implementation NXMetalGpuCrashObserver

- (instancetype)initWithDeviceName:(const nx::string_view)deviceName
                   reportingEnabled:(const bool)reportingEnabled {
  self = [super init];
  if (self != nil) {
    _lost.store(false, std::memory_order_relaxed);
    _reported.store(false, std::memory_order_relaxed);
    _device = deviceName;
    _reportingEnabled = reportingEnabled;
  }
  return self;
}

- (bool)deviceLost {
  return _lost.load(std::memory_order_acquire);
}

- (void)inspect:(id<MTLCommandBuffer>)buffer
          label:(const nx::string_view)label
          frame:(const u64)frame {
  NSError *const error = buffer.error;
  if (error == nil)
    return;
  _lost.store(true, std::memory_order_release);
  if (!_reportingEnabled)
    return;
  bool expected = false;
  if (!_reported.compare_exchange_strong(expected, true,
                                         std::memory_order_acq_rel))
    return;

  try {
    nxe::rhi::GpuCrashReport report;
    report.backend = nxe::rhi::GpuBackend::Metal;
    report.device = _device;
    report.operation = label;
    report.result = nx::format("{} code={} status={}",
                               nxe::rhi::utf8(error.domain), error.code,
                               nx::cast<usize>(buffer.status));
    report.description = nxe::rhi::utf8(error.localizedDescription);
    if (error.localizedFailureReason != nil)
      report.description.append(
          nx::format("; {}", nxe::rhi::utf8(error.localizedFailureReason)));
    report.frame = frame;

    if (@available(macOS 11.0, iOS 14.0, *)) {
      NSArray<id<MTLCommandBufferEncoderInfo>> *const infos =
          error.userInfo[MTLCommandBufferEncoderInfoErrorKey];
      for (id<MTLCommandBufferEncoderInfo> info in infos) {
        nxe::rhi::GpuEncoderFault encoder;
        encoder.label = nxe::rhi::utf8(info.label);
        encoder.state = nxe::rhi::encoder_state_name(info.errorState);
        for (NSString *const signpost in info.debugSignposts)
          encoder.debug_signposts.push_back(nxe::rhi::utf8(signpost));
        report.encoders.push_back(std::move(encoder));
      }
    }

    if (!nxe::rhi::report_gpu_crash(report))
      nx::loge(nxe::rhi::log_rhi,
               "Metal GPU crash report could not be persisted");
  } catch (...) {
    nx::loge(nxe::rhi::log_rhi,
             "Metal GPU crash diagnostics could not be collected");
  }
}

@end

namespace nxe::rhi {

void configure_gpu_crash_reporting(MetalDeviceState &state,
                                   const GpuCrashMode mode,
                                   const nx::string_view device_name) {
  const bool enabled = mode != GpuCrashMode::Disabled;
  state.gpu_crash_detailed = mode == GpuCrashMode::Detailed;
  state.gpu_crash_observer = [[NXMetalGpuCrashObserver alloc]
      initWithDeviceName:device_name
         reportingEnabled:enabled];
}

GpuDeviceStatus gpu_device_status(const MetalDeviceState &state) noexcept {
  NXMetalGpuCrashObserver *const observer = state.gpu_crash_observer;
  return observer != nil && [observer deviceLost] ? GpuDeviceStatus::Lost
                                                   : GpuDeviceStatus::Ready;
}

id<MTLCommandBuffer> make_command_buffer(MetalDeviceState &state,
                                         const nx::string_view label,
                                         const u64 frame) {
  id<MTLCommandBuffer> buffer = nil;
  if (state.gpu_crash_detailed) {
    if (@available(macOS 11.0, iOS 14.0, *)) {
      MTLCommandBufferDescriptor *const descriptor =
          [[MTLCommandBufferDescriptor alloc] init];
      descriptor.errorOptions =
          MTLCommandBufferErrorOptionEncoderExecutionStatus;
      buffer = [state.queue commandBufferWithDescriptor:descriptor];
    }
  }
  if (buffer == nil)
    buffer = [state.queue commandBuffer];
  if (buffer == nil)
    return nil;

  buffer.label = to_ns_string(label);
  NXMetalGpuCrashObserver *const observer = state.gpu_crash_observer;
  if (observer != nil) {
    const nx::string owned_label(label);
    [buffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
      [observer inspect:completed label:owned_label.view() frame:frame];
    }];
  }
  return buffer;
}

} // namespace nxe::rhi
