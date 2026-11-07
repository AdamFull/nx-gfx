# nx gfx guide

The contracts behind the API in `include/nx/gfx/`: what a caller may rely on,
and what it must do in return. The headers carry the per-function details;
this file is what spans them.

## Devices and handles

`create_device` makes a `device` on the calling thread, which owns it from
then on; `destroy_device` waits for the GPU, then frees it and whatever it
still holds. A device hands out generational handles (`texture_handle`,
`buffer_handle`, `sampler_handle`, `shader_handle`, `pipeline_handle`,
`heap_handle`, `query_pool_handle`, `swapchain_handle`): a handle names one
object of one device, a destroyed handle never comes back as another object,
and every destruction is deferred past the work already submitted.

Code above the library chooses from capabilities (`caps(dev)`), never from the
backend: maximum texture size, timestamps, device-local memory, the optional
features below. MSAA is a set, not a maximum: `device_caps::msaa_sample_counts`
holds the counts every attachment format takes, and a device may lack one
below its largest (Mali has 4 but not 2). Pick a count with
`samples_at_most`.

The threading contract is stated per function in the headers. In short: the
owner thread makes and frees the device and swapchains, begins and ends frames
and publishes pipelines; any thread creates and destroys resources, asks for
bindless indices, maps, uploads and compiles pipelines; a `command_list` has
one recorder at a time.

## Vulkan routes

The Vulkan backend needs Vulkan 1.1 with buffer device addresses, timeline
semaphores, descriptor indexing, scalar block layout, shader draw parameters,
synchronization2 and extended dynamic state. It picks core command routes from
the lower of the loader's and the device's API versions; a feature promoted
past that version stays on its extension route. Required device commands are
checked after the dispatch table loads, and initialization fails with a
diagnostic rather than let a null command pointer reach a frame.

A device records passes through dynamic rendering where it has it, and
otherwise through render passes over imageless framebuffers (the render-pass
route); `device_caps::dynamic_rendering` says which, as does the device's log
line. A caller sees no difference: every attachment begins and ends a pass in
the state its barriers left it in, so a render pass transitions nothing, and a
pipeline is made against a compatible pass from the formats and sample counts
it declares. A device with neither is refused.

`device_desc::disable_routes`, or `NX_GFX_DISABLE_ROUTES=dynamic_rendering`,
puts a device that has dynamic rendering on the fallback, so the fallback is
tested on any machine; Shipping builds ignore both. `nx_gfx_tests` runs every
case again that way (ctest label `routes`). `ray_queries` (`route::RayQueries`)
turns off ray queries and the acceleration structures they trace, so a caller's
own fallback for them is tested on a device that has them; `nx_gfx_tests`
checks it with a device of its own rather than another pass of the suite.

Bindless slots live in one update-after-bind descriptor set, or, where a
device has `VK_EXT_descriptor_heap` and it is asked for
(`device_desc::enable_routes`, or `NX_GFX_ENABLE_ROUTES=descriptor_heap`
outside Shipping), in a resource heap and a sampler heap: host-visible
buffers the slots are written into and bound by address at the start of
every command buffer. Pipelines on that route have no layout; each stage maps
the set's four bindings onto constant heap offsets, so the same SPIR-V runs on
both, and push constants become push data. `device_caps::descriptor_heap`
says which. The route is opt-in because no driver passes the suite with it
yet. On NVIDIA 610.62 a sampled read returns zeros when the same shader also
reads that image without a sampler (a load or a size query) and its heap index
is not a constant; the same SPIR-V is correct on the descriptor set, validation
is silent, and neither the resource masks, the mapping's sampler fields nor
`NonUniform` change it. That is a driver fault, and the route waits for a fixed
driver. `nx_gfx_tests` runs again on it
(`descriptor heaps: ...`).

Outside Shipping, `NX_GFX_PIPELINE_STATISTICS` (any value) has every pipeline
compile log what the driver reports of its executables, a line each
(`pipeline '<name>' <executable>, <statistic> <value>, ...`), where the device
has `VK_KHR_pipeline_executable_properties`: on NVIDIA, among others, the
registers a thread takes, which bound how many warps a multiprocessor holds.
It is for profiling: capturing the statistics may slow compiles. Metal reports
nothing. `NX_GFX_PIPELINE_REPRESENTATIONS=<folder>` logs them too and writes
what the driver says it compiled each executable into, a file each
representation (`<pipeline>.<executable>.<representation>.txt`, or `.bin`),
where the driver offers any: Mesa's drivers do; NVIDIA's offers none.

## Bindless resources

Every texture, storage view and sampler gets a slot in one bindless table,
asked for with `texture_index`, `rw_texture_index` (one per mip) and
`sampler_index`. A texture's slot holds a view of its own type, so a cube's
sampled slot is a cube; its storage slots are 2D arrays of its faces, as Vulkan
has no cube storage view. Sampler slot 0 (`DEFAULT_SAMPLER`) is the device's
own bilinear, clamp-to-edge sampler (`sampler_desc`'s defaults), made before
any other, so a handle made with sampler 0 samples through a real one rather
than an unwritten descriptor. Metal leaves cube, array, 3D and multisampled
textures out until its 3D track and reports them absent.

`buffer_address` gives a buffer's device address for `NxPtr<T>`.
`replace_texture` gives a handle a new image under the same bindless index
(mip streaming, hot reload).

## Shaders and the shared types

The Slang library is four modules. `nxg_core` holds the bindless tables and
their accessors, the null page and `nx_readable`; `nxg_ptr` holds `NxPtr<T>`;
`nxg_texture` holds the texture handles: `NxSampler`, `NxTexture2D<T>`,
`NxRWTexture2D<T>`, and on Vulkan `NxTextureCube<T>`, `NxTexture2DArray<T>`,
`NxTexture3D<T>`, `NxTexture2DMS<T>`, `NxRWTexture2DArray<T>` and
`NxRWTexture3D<T>`; `nxg_ray` holds inline ray tracing (see Acceleration
structures). Shaders reach textures through these handles, never raw
indices. Each is one word with a C++ half under the same name
(`nx/gfx/shaders/texture.hpp`) and the same makers and readers:
`from_indices(texture, sampler, flags)`, `from_packed`, `none()`,
`from_index` for storage, `texture_index()`, `is_valid()`.
`NxTexture2D<float>::gather` (and `<uint>`) reads the four texels bilinear
filtering would blend, unfiltered, in Gather's order.
`NxTexture2DArray<T>::sample_grad(uv, layer, dx, dy)` samples a layer with
explicit screen gradients, as a compute shader with none of its own must.

A struct C++ and Slang share is declared once, in a header that starts from
`nx/gfx/shaders/interop.h`, whose comment states the rules: fields of the
shared types (`float4`, `uint`, `NxPtr<T>`, the texture handles), constants
made with `NX_CONST`, functions marked `NX_SHARED_FN` (`shared.h`, which also
spells `sqrt`, `asuint` and `asfloat` for both), no `#ifdef` and no visibility
keywords. A shared struct lays out alike in C++, in Slang's natural layout
(what `NxPtr` reads) and in std430 (push constants), so fields are 4, 8 and 16
byte types placed where all three agree. `NX_SHARED_SIZE` holds a struct to
its size; `gfx interop:` in `nx_gfx_tests` compares the library's own types
field by field with the shaders' reflection, and a consumer does the same for
its structs.

C++'s `float4` and the rest are plain structs unless `NX_GFX_INTEROP_MATH_TYPES`
names a header supplying the consumer's own (with
`NX_GFX_INTEROP_MATH_LIBRARIES` for what it needs); the build defines it for
every target linking `nx::gfx_formats`, so every translation unit agrees.

`NxPtr<T>` is a device address, read and written with `load`, `store` and, on
`NxPtr<uint>` and `NxPtr<int>`, `atomic_add` (on `NxPtr<uint>`, also
`atomic_compare_exchange`; on `NxPtr<uint>` and
`NxPtr<uint64_t>`, `atomic_max`, the 64-bit one only where
`device_caps::buffer_int64_atomics`). A load through a null `NxPtr` reads
zeros: Mali compilers move a load above the check that guards it, so loads
read a zeroed null page (`NX_NULL_PAGE_BYTES`) in place of address zero.
`is_valid()` still means absent, and a store or atomic through null still
faults. Never put a load under its own pointer's `is_valid()`
(`p.is_valid() ? p.load(i) : 0`): inside the check the compiler knows the
address is not zero, folds the null page away and may still hoist the load,
so Mali reads address zero and loses the device. Load unguarded and let the
null page read zero. A shader that reads a raw address offsets it with
`nx_readable`.

Compile shaders with `nx_gfx_compile_slang` (`cmake/NxGfxShaders.cmake`) or
the flags it uses. SPIR-V 1.3 runs on every Vulkan 1.1 device; a shader that
needs more (mesh shaders and ray queries need 1.4) is loaded only where the
device has what it needs.

Every SPIR-V module then goes through `normalize_spirv` (`nx/gfx/spirv.h`, in
`nx::gfx_formats`) before a device loads it; `nx_gfx_compile_slang` runs it
through the `nx_gfx_spirv` host tool, and a consumer with its own pipeline
calls it where it compiles. It answers the module unchanged, byte for byte,
except for what it exists to fix: slangc 2026.8 declares `SPV_KHR_ray_tracing`
for an acceleration structure made from an address
(`OpConvertUToAccelerationStructureKHR`, which `SPV_KHR_ray_query` provides as
well), and a device without `VK_KHR_ray_tracing_pipeline`, which is every
device this library makes, then refuses the module
(`VUID-VkShaderModuleCreateInfo-pCode-08742`). A module with no `RayTracingKHR`
capability loses that declaration. `spirv-opt --trim-capabilities` is no
substitute: it drops `SPV_KHR_physical_storage_buffer` instead. A build that
cannot run its own executables names a host build in `NX_GFX_SPIRV_TOOL`.

## Pipelines

A pipeline is made in three steps. `prepare_pipeline` copies every input of a
description, on the owner thread, into a `pipeline_job`, or answers an invalid
one when it refuses the description. `pipeline_job::compile` runs the driver
compilation once, on any thread, touching no device table;
`compile_pipeline_async` does that on an `nx::thread_pool`. `publish_pipeline`
inserts a compiled job into the device's table on the owner thread, and
answers an invalid handle for a job that did not compile. A job dropped before
it compiles is cancelled, its native object destroyed and its compilation
counted as rejected in `pipeline_compile_statistics`. There is no synchronous
create: a caller that wants one compiles the job where it stands. Debug names
are not part of a pipeline's identity.

`device_desc::pipeline_cache_path` is where the driver's cache (Vulkan) or
binary archive (Metal) lives between runs; `save_pipeline_cache` writes it.

## Frames and swapchains

A device has one swapchain per window it draws: `create_swapchain`, or
`create_offscreen_swapchain` for one with no surface. `begin_frame` starts GPU
frame N (its slot wait, garbage and uploads) without touching any swapchain;
`acquire(swapchain)` then takes that swapchain's image, once per frame, and
`current_backbuffer` names it. A swapchain that is out of date, minimised or
not asked for is simply not acquired: the frame still submits and the others
still present. `end_frame` submits once, waiting on every acquired image, and
presents all of them in one call, so FIFO waits once per frame however many
windows there are. A present that finds a surface changed marks the swapchain
`swapchain_stale` for its owner to resize. Every swapchain shares
`swapchain_format`, chosen by the first one made and kept after it goes, so
pipelines built for it stay valid. A frame that acquired only offscreen
swapchains is waited for in `end_frame`. `begin_headless_frame` and
`end_headless_frame` record and submit with no swapchain at all.

`split_frame` divides a frame into lists recorded on several threads and
submitted in order; a pass never spans two lists.

`swapchain_desc::alpha` asks for an opaque or a premultiplied surface, and
`swapchain_alpha` says what the surface granted. Vulkan takes only a
premultiplied composite mode for a transparent request, never post-multiplied
or `INHERIT` (which Intel's Win32 driver offers and then ignores). When a
Win32 surface offers no premultiplied mode, frames reach the screen through
DirectComposition: D3D11 on the same adapter allocates the images, Vulkan
imports them as the backbuffers, and two shared fences hand each frame over.
Callers see an ordinary swapchain. Metal sets the layer's `opaque` flag.

## GPU markers

`set_gpu_markers(dev, true)` times the frame's debug groups from the next
`begin_frame`, and `read_gpu_markers` hands them over: name, depth and times on
`nx::os::ticks_ns`'s clock. Nothing here knows who reads them or waits for the
GPU: a frame's markers are read when its slot comes round, a frame or two later,
and nobody reading loses the oldest once `GPU_MARKERS_HELD` wait. Off, a debug
group costs one test, and no query pool exists until the first frame on.

A frame's lists are timed, split lists too and in the order they are submitted, up
to 1024 groups a frame, 32 lists and `GPU_MARKER_MOST_DEPTH` deep; `gpu_markers_stats`
counts what was left out. A marker's `frame` tells frames apart. Copies on the copy
queue are timed as well, up to 256 waiting, when that queue has timestamp bits and the
device has host query reset, since a copy queue cannot reset a query itself. A device
with memoryless attachments, a tiling GPU, does not time groups inside a render pass.
Both ends of a group sit at the bottom of the pipe, so groups in a row never overlap. Calibrated timestamps place them on the host's clock, read several
times about once a second and the tightest reading kept; without them each frame is
placed by when it was submitted, which can be up to a frame early, and
`gpu_marker_stats::calibrated` says which. A Shipping build enables neither calibrated
timestamps nor host query reset, so it times no copies and places frames by their
submit, but a pass's duration is as good. Metal returns nothing yet.

## Retirement holds

The device defers every destruction past the work already submitted, but a
resource can also be named by work not yet recorded, such as a frame packet a
render thread has not reached. `hold_retirements(serial)` and
`release_retirements(serial)` cover that: what is destroyed while a hold is
active outlives it. `replace_texture` honours the same holds, so the swap
waits until no held work can still record against the old image, and the new
image starts undrawn. Without holds a destruction waits only for submitted
work, and a replacement swaps at once.

## Uploads

`upload_texture`, `upload_buffer` and `clear_buffer` go through the device's
copy queue (or the graphics queue where there is none) and return an
`upload_ticket`; `upload_complete`,
`wait_upload` and `wait_uploads` observe them, and `upload_stats` counts the
work; `read_buffer` and `read_texture` read back, after the frames submitted
before them on the GPU, so a readback sees what those frames and the client
queues they wait on wrote. A texture upload may take a
`texture_data`: a byte span and its subresource
table, validated (`validate_image_upload`) before anything is copied.

## Optional features

`device_caps` reports what a device may use beyond the floor: mesh shaders,
64-bit buffer and image atomics, fragment barycentrics, acceleration
structures and ray queries, the subgroup operations a compute shader has, the
subgroup size and the range a pipeline may ask for. Vulkan enables each at
device creation when the device has it and reports it only then, so a shader
may use whatever is reported. Where `mesh_shaders` is reported a
`graphics_pipeline_desc` may name a `mesh` entry instead of a `vertex` one;
such a pipeline draws only through `draw_mesh_tasks_indirect`. A module with a
mesh entry holds no fragment entry a pipeline uses: Intel's compiler crashes
building a fragment stage whose module holds a mesh entry too.

## Acceleration structures

Where `acceleration_structures` is reported, `nx/gfx/ray.h` builds them, and
where `ray_queries` is, a shader traces them inline (`RayQuery`) from any
stage; there are no ray-tracing pipelines. A shader reaches a top level by its
address (`acceleration_structure_address`), passed as a 64-bit word like any
device address, and traces it through `nxg_ray`: `NxScene::from_address`,
an `NxRay`, and `nx_trace<Flags>(scene, ray, mask, shade)`, which calls the
caller's `IRayShade` back, `any_hit` for each triangle of a geometry not made
opaque (a cut-out texel answers false), then `closest_hit` with the nearest
that counted, or `miss`. Consumers write those callbacks rather than a query
loop, so another implementation behind `nx_trace` changes no consumer. No
descriptor is bound. `NxRayInstance` (`nx/gfx/shaders/ray_layout.h`, which
`gfx::ray_instance` names) is the instance record both C++ and shaders
write, and the `NX_RAY_` constants are a trace's flags. On Metal every trace
misses until its 3D track: Slang makes no acceleration structure from an
address there.

A structure owns the buffer it lives in (`acceleration_structure_buffer`).
Barriers and render-graph imports name that buffer, in three states:
`AccelerationStructureInput` for what a build reads (vertices, indices,
transforms, instances; their buffers made `AccelerationStructureInput`),
`AccelerationStructureBuild` for the structure and the scratch a build writes,
and `AccelerationStructureRead` for a structure a shader traces or a top level
built over it reads. The caller places those barriers; a scratch reused by the
next build needs a barrier from `AccelerationStructureBuild` to itself.

`sizes` answers what a description needs: the structure's bytes, its build
scratch and its update scratch. The scratch's address is a multiple of
`device_caps::acceleration_structure_scratch_alignment`. A structure is made
with the flags every build, update and copy into it names
(`acceleration_structure_desc::flags`); a build of other flags is refused. So
is one into the other kind, one too small, a misaligned scratch, top-level
instances not 16 aligned, and anything inside a render pass; a refused build
records nothing and logs why. Positions are three `f32` each; instances are
`ray_instance` records, 64 bytes, laid out as Vulkan's.

`update_acceleration_structure` refits a structure made `AllowUpdate` in
place, for the same counts and flags with moved positions or instances; a top
level over an updated bottom one is built or updated after it. Compaction takes
a structure made `AllowCompaction`: its size is written into a pool
(`create_compacted_size_pool`, reset before each write) once it is built,
read back without waiting (`read_compacted_sizes`) frames later, and the
structure copied compacted (`copy_acceleration_structure`) into one made that
size, the first then destroyed. Destruction is deferred past submitted work and
past retirement holds, as a buffer's is.

Metal reports neither feature until its 3D track, and its functions answer
invalid, zero or false.

## The software BVH

Where a device has no ray queries, a caller's fallback traces levels the
library builds on the CPU (`nx/gfx/bvh.h`, in `nx::gfx_formats`, so asset
tools build them too): 4-wide BVHs in one block each, whose layout C++ and
shaders share (`nx/gfx/shaders/bvh_layout.h`). A block is a header, nodes in
level order, its primitives in leaf order, the level table, and, with
`bvh_options::keep_bounds`, each node's exact box for refitting. A bottom
level's primitives are triangles with their corners inline. A top level's
(`build_bvh_top`, `NX_BVH_TOP`) are instances: the inverse of each
instance's transform, its bottom level's address, custom index, mask and
flags, and its place in the input, so a ray is carried into each
instance's space unnormalized and a distance means the same in both
levels. A node holds four children's boxes quantized to a byte an axis on
a grid whose steps are powers of two; a decoded box always holds its child,
checked with the arithmetic a shader decodes with.

`build_bvh_here` builds on the calling thread; `build_bvh` on an
`nx::thread_pool`, a future the caller continues on, cancelled through a
token (the future then reports cancelled). Both give the same bytes for the
same input and options whatever the workers: a binary tree by binned SAH
whose node slots follow from the triangles they span, so subtrees built at
once write apart, then a level-order collapse to four children. Past a
depth limit a node splits at its median, which bounds the levels and so a
traversal's stack. Input with no triangles, counts not multiples of three,
an index past the positions, a position not finite or options out of range
is refused, never built.

Shaders trace a software top level through `nxg_ray` as they trace a
hardware one: `nx_trace` over an `NxSoftwareScene` (the top block's address)
rather than an `NxScene`, with the same `IRayShade`, flags, masks, instance
flags and facing, so one consumer source serves both. It picks the scene
type itself (`typealias Scene = ...` under a define of its own), since a
macro does not reach into an imported module, and builds its software
variant as a file that sets that define and includes the source; that
variant needs no ray-query capability and compiles for Metal. The walk keeps
one stack of `NX_BVH_STACK` entries for both levels and drops what does not
fit; its triangle test is watertight in float alone, without the CPU's
double fallback for an edge function of exactly zero: each edge function is
computed from the edge's lesser corner first (`nx_bvh_edge`), so the two
triangles sharing an edge get each other's negation exactly even where a
compiler fuses the products into a multiply-add, which Slang's `precise`
does not stop in SPIR-V.

A bottom level whose vertices move and whose triangles do not (skinning,
morphing) is refitted on the GPU rather than rebuilt, through `nxg_bvh`:
`nx_bvh_refit_triangle` rewrites one triangle record's corners from where
its geometry's vertices are now (`NxBvhRefitGeometry`: positions and the
indices it was built with), and `nx_bvh_refit_node` re-boxes one node from
its children and quantizes them again on the builder's conservative grid,
the root's box into the header too. The consumer writes the entry points:
one thread a triangle, then one dispatch a level, deepest first (the level
table is in the block), a barrier on the block between each. It needs a
block built with `keep_bounds`, as a quantized box cannot be refitted from
quantized children. Refitting vertices that did not move gives back the
bytes the CPU built. Every record lies on 16 bytes, as `NxPtr` reads and
writes 16 at a time; that is why `NxBvhBounds` is padded to 32.

`validate_bvh` checks a block from anywhere before it is trusted: offsets
and counts that account for every byte, nodes after their parents and in
their levels, leaves that cover every triangle exactly once, finite
corners. `trace_bvh` and `trace_bvh_scene` (through a top level, with a
ray mask) are the CPU reference a shader traversal is tested against:
nearest hit, every triangle opaque, watertight (Woop, Benthin and Wald), a
box's far distance widened by a few ulps.

## Native interop

A library that works on the backend's own objects (an upscaler SDK, a video
decoder) reaches them through `nx/gfx/native.h` and, on Vulkan,
`nx/gfx/vulkan.h`, never the backend's types. `native(dev)` hands out the
instance, adapter, device, graphics queue and the loader's proc-address
functions as opaque pointers tagged with their API; `native(dev, texture)` a
texture's image, sampled view and format; `native_layout` the image layout a
resource state stands for. A texture in a copy state may take any transfer
command in that layout, a clear, blit or resolve as well as a copy: the
state's barriers cover them all. Between render passes `begin_native` hands out the
command buffer and `end_native` takes it back, binding again what the library may have
bound over the device's own. A library that frees at once what earlier frames
used is first given `wait_submitted`, which waits for the frames submitted so
far and, unlike `wait_idle`, frees nothing and needs no other thread to stop.

What such a library must enable before the device exists it asks through a
`native_device_client` in `device_desc::native_clients`: instance extensions,
device extensions for the chosen adapter, then a patch of the features the
device is made with (Vulkan: the `VkPhysicalDeviceFeatures2` heading the pNext
chain). The device enables the extensions it has and leaves out the rest; if
it cannot be made with the clients' features, it is made with none of the
clients' requests, and `native_extension` says what it was made with. A client
may ask for a queue of its own (`queue_family`), which the device makes only
in a family none of its own queues use; `native(dev, client)` hands it back.
Vulkan makes the host synchronize each queue, so every submit and present to
a device queue holds the queue's lock around the call alone, never across a
wait; native code submitting to a device queue goes through
`vk::submit_graphics`, which takes the same lock. A client's own queue is the
client's to synchronize.

An image the library did not make can be adopted: `vk::wrap` takes a 2D,
single-mip Vulkan image shared (concurrent) with every family
`vk::queue_families` lists, and gives it a handle, a bindless slot and views
of the device's own. It retires as any texture does; once no submitted or
held work can use it, the device drops its views and calls the image's
`release` instead of freeing it. A wrapped image is sampled and may be copied
from, never uploaded into: uploads hand images between the copy and graphics
families.

A client whose queue writes what frames read makes frames wait on the GPU
rather than waiting on the CPU itself. `vk::acquire_frame_wait` hands it a
timeline semaphore the device owns, at most eight per device; the client
signals it on its own queue and names each value with `vk::wait_before_frame`,
after submitting that signal. Every frame submitted from then on waits for the
newest value named, which costs nothing once reached. `vk::release_frame_wait`
stops the waits, and the device destroys the semaphore once the first frame
submitted without it has completed, or with the device.

## GPU crash reporting

`device_desc::gpu_crash_mode` selects the crash service. `Disabled` collects
nothing; `Basic` records the device, the failed native operation, its result
and the last submitted frame; `Detailed` adds diagnostics that may cost
driver time. Reports go to nx-foundation's crash directory as the
`gpu-crash` kind, with a versioned `gpu-report.json` sidecar and opaque vendor
data as a separate bounded attachment; the JSON names the GPU objects involved,
resolving late address-binding events against a bounded inventory.

Device loss is terminal for the device: submission, presentation, uploads,
swapchain recreation and idle waits stop, `device_status` says so, and the
caller is expected to shut down in order. Teardown frees host-side and native
ownership without waiting on the lost device. Vulkan prefers
`VK_KHR_device_fault`, falls back to `VK_EXT_device_fault`, and otherwise
still writes the basic report; Detailed mode adds vendor binary data and a
bounded history from `VK_EXT_device_address_binding_report`. Metal observes
every command buffer through its completion handler, and Detailed mode adds
encoder labels, states and debug signposts. Either backend publishes at most
one report per device. A report lists the further files it carries in the
JSON's `attachments`.

On Vulkan, Detailed mode also starts NVIDIA Nsight Aftermath where the build
found its SDK (`NX_GFX_AFTERMATH_SDK`, found under an installed Nsight
Graphics by default) and its library loads at run time; elsewhere it is off
and costs nothing. Aftermath must start before the device is made: the device
then enables `VK_NV_device_diagnostics_config` and checkpoints, and each debug
group marks a checkpoint resolved to its label. Where `VK_AMD_buffer_marker`
exists, each group also has the GPU write when it starts and when it is done,
to memory that a lost device leaves readable, so the report lists the last groups
and which were running. Every shader module's SPIR-V
is kept, and the driver's debug information for shaders is collected as
deferred callbacks, so on device loss the report adds the decoded
`gpu-aftermath.json` (faulting warps resolved to a shader, its instruction
and, for SPIR-V built with source debug information, the file and line), the
raw `gpu-aftermath.nv-gpudmp` and the `.nvdbg` files Nsight Graphics opens it
with. The wait for the dump is bounded.

## Testing

`nx_gfx_tests` covers the API, the pipeline job, crash reports, native
interop, acceleration structures traced by a ray query, `normalize_spirv` and
the shared types' layout, and runs again on the render-pass route.
A case that needs a device skips where none can be made. Metal is maintained
with the Vulkan backend but runs in none of these tests off Apple hardware.
