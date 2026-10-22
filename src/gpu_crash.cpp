#include "nx/gfx/crash.h"

#include "core/foundation/diagnostics/crash.h"
#include "core/foundation/diagnostics/log.h"
#include "core/foundation/serialization/json.h"
#include "core/foundation/strings/format.h"

#include <utility>

namespace nx::gfx {
namespace {

nx::string_view backend_name(const gpu_backend backend) noexcept {
  return backend == gpu_backend::Metal ? "Metal" : "Vulkan";
}

nx::string hex(const u64 value) { return nx::format("0x{:016x}", value); }

nx::json::value string_array(const nx::vector<nx::string> &values) {
  nx::json::value out = nx::json::value::make_array();
  out.reserve(values.size());
  for (const nx::string &value : values)
    out.push_back(nx::json::value(value));
  return out;
}

} // namespace

nx::string gpu_crash_json(const gpu_crash_report &report) {
  nx::json::value root = nx::json::value::make_object();
  root.insert("schema_version", nx::json::value(1u));
  root.insert("kind", nx::json::value("gpu_crash"));
  root.insert("backend", nx::json::value(backend_name(report.backend)));
  root.insert("device", nx::json::value(report.device));
  root.insert("operation", nx::json::value(report.operation));
  root.insert("result", nx::json::value(report.result));
  root.insert("frame", nx::json::value(report.frame));
  root.insert("description", nx::json::value(report.description));

  nx::json::value addresses = nx::json::value::make_array();
  addresses.reserve(report.addresses.size());
  for (const gpu_fault_address &fault : report.addresses) {
    nx::json::value entry = nx::json::value::make_object();
    entry.insert("kind", nx::json::value(fault.kind));
    entry.insert("address", nx::json::value(hex(fault.address)));
    entry.insert("precision", nx::json::value(fault.precision));
    addresses.push_back(std::move(entry));
  }
  root.insert("fault_addresses", std::move(addresses));

  nx::json::value objects = nx::json::value::make_array();
  objects.reserve(report.objects.size());
  for (const gpu_object_identity &object : report.objects) {
    nx::json::value entry = nx::json::value::make_object();
    entry.insert("object_type", nx::json::value(object.object_type));
    entry.insert("object_name", nx::json::value(object.object_name));
    entry.insert("object_handle", nx::json::value(hex(object.object)));
    objects.push_back(std::move(entry));
  }
  root.insert("objects", std::move(objects));

  nx::json::value bindings = nx::json::value::make_array();
  bindings.reserve(report.bindings.size());
  for (const gpu_address_binding &binding : report.bindings) {
    nx::json::value entry = nx::json::value::make_object();
    entry.insert("operation",
                 nx::json::value(binding.bound ? "bind" : "unbind"));
    entry.insert("object_type", nx::json::value(binding.object_type));
    entry.insert("object_name", nx::json::value(binding.object_name));
    entry.insert("object_handle", nx::json::value(hex(binding.object)));
    entry.insert("address", nx::json::value(hex(binding.address)));
    entry.insert("size", nx::json::value(binding.size));
    entry.insert("internal", nx::json::value(binding.internal));
    bindings.push_back(std::move(entry));
  }
  root.insert("address_bindings", std::move(bindings));

  nx::json::value vendor_faults = nx::json::value::make_array();
  vendor_faults.reserve(report.vendor_faults.size());
  for (const gpu_vendor_fault &fault : report.vendor_faults) {
    nx::json::value entry = nx::json::value::make_object();
    entry.insert("description", nx::json::value(fault.description));
    entry.insert("code", nx::json::value(hex(fault.code)));
    entry.insert("data", nx::json::value(hex(fault.data)));
    vendor_faults.push_back(std::move(entry));
  }
  root.insert("vendor_faults", std::move(vendor_faults));

  nx::json::value encoders = nx::json::value::make_array();
  encoders.reserve(report.encoders.size());
  for (const gpu_encoder_fault &encoder : report.encoders) {
    nx::json::value entry = nx::json::value::make_object();
    entry.insert("label", nx::json::value(encoder.label));
    entry.insert("state", nx::json::value(encoder.state));
    entry.insert("debug_signposts", string_array(encoder.debug_signposts));
    encoders.push_back(std::move(entry));
  }
  root.insert("command_encoders", std::move(encoders));
  root.insert("vendor_binary_attachment",
              nx::json::value(report.vendor_binary.empty()
                                  ? nx::string_view{}
                                  : nx::string_view("gpu-vendor.bin")));
  nx::json::value attachments = nx::json::value::make_array();
  attachments.reserve(report.attachments.size());
  for (const gpu_crash_attachment &attachment : report.attachments)
    attachments.push_back(nx::json::value(attachment.name));
  root.insert("attachments", std::move(attachments));
  return nx::json::write(root, {.indent = 2});
}

bool report_gpu_crash(const gpu_crash_report &report) noexcept {
  try {
    const nx::string json = gpu_crash_json(report);
    nx::vector<nx::crash::Attachment> attachments;
    attachments.push_back({
        .name = "gpu-report.json",
        .bytes = {reinterpret_cast<const u8 *>(json.data()), json.size()},
    });
    if (!report.vendor_binary.empty()) {
      attachments.push_back({
          .name = "gpu-vendor.bin",
          .bytes = {report.vendor_binary.data(), report.vendor_binary.size()},
      });
    }
    for (const gpu_crash_attachment &attachment : report.attachments)
      attachments.push_back(
          {.name = attachment.name,
           .bytes = {attachment.bytes.data(), attachment.bytes.size()}});

    return nx::crash::report({.reason = "GPU command execution failed",
                              .detail = "Structured diagnostics are attached "
                                        "as gpu-report.json.",
                              .kind = "gpu-crash",
                              .attachments = attachments});
  } catch (...) {
    nx::loge(nx::log::gpu,
             "GPU crash diagnostics could not be assembled or persisted");
    return false;
  }
}

} // namespace nx::gfx
