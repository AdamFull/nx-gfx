#pragma once

#include "nx/gfx/reflection.h"

namespace nxe::rhi {

// The names before nx::gfx. Only the backend's own sources and projects made
// before nx::gfx (through app/engine.h) use them.
using ReflectKind = nx::gfx::reflect_kind;
using ReflectMember = nx::gfx::reflect_member;
using ReflectBlock = nx::gfx::reflect_block;
using ReflectComputeEntry = nx::gfx::reflect_compute_entry;
using ShaderReflection = nx::gfx::shader_reflection;

using nx::gfx::reflect_from_json;

} // namespace nxe::rhi
