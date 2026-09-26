#include "rendering/rhi/vulkan/vk_composition_win32.h"

#include "core/foundation/containers/small_vector.h"
#include "core/foundation/diagnostics/log.h"

#include <windows.h>
// Do not reorder!
#include <d3d11_4.h>
#include <dcomp.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <cstring>

namespace nxe::rhi {

using Microsoft::WRL::ComPtr;

struct CompositionPresenter::State {
  ComPtr<ID3D11Device5> device;
  ComPtr<ID3D11DeviceContext4> context;
  ComPtr<IDXGIFactory2> factory;
  ComPtr<IDXGISwapChain1> swapchain;
  ComPtr<IDCompositionDevice> composition;
  ComPtr<IDCompositionTarget> target;
  ComPtr<IDCompositionVisual> visual;
  ComPtr<ID3D11Fence> rendered;
  ComPtr<ID3D11Fence> copied;
  nx::small_vector<ComPtr<ID3D11Texture2D>, 4> images;
  DXGI_FORMAT format = DXGI_FORMAT_B8G8R8A8_UNORM;
  u32 count = 0;

  [[nodiscard]] bool make_images(u32 width, u32 height);
};

namespace {

[[nodiscard]] bool failed(const HRESULT result, const char *what) {
  if (SUCCEEDED(result))
    return false;
  nx::logw(nx::log::rhi, "composition: {} failed (0x{:08x})", what,
           nx::cast<u32>(result));
  return true;
}

} // namespace

bool CompositionPresenter::State::make_images(const u32 width,
                                              const u32 height) {
  images.clear();
  D3D11_TEXTURE2D_DESC desc{};
  desc.Width = width;
  desc.Height = height;
  desc.MipLevels = 1;
  desc.ArraySize = 1;
  desc.Format = format;
  desc.SampleDesc.Count = 1;
  desc.Usage = D3D11_USAGE_DEFAULT;
  desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
  desc.MiscFlags =
      D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_SHARED;
  for (u32 i = 0; i < count; ++i) {
    ComPtr<ID3D11Texture2D> image;
    if (failed(device->CreateTexture2D(&desc, nullptr, &image),
               "CreateTexture2D"))
      return false;
    images.push_back(std::move(image));
  }
  return true;
}

CompositionPresenter::CompositionPresenter() = default;
CompositionPresenter::~CompositionPresenter() {
  if (m_state && m_state->context)
    m_state->context->Flush();
}

bool CompositionPresenter::create(void *const hwnd, const u8 (&luid)[8],
                                  const u32 width, const u32 height,
                                  const u32 image_count, const bool rgba) {
  auto s = nx::make_unique<State>();
  s->count = image_count;
  s->format = rgba ? DXGI_FORMAT_R8G8B8A8_UNORM : DXGI_FORMAT_B8G8R8A8_UNORM;

  ComPtr<IDXGIFactory4> factory;
  if (failed(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)),
             "CreateDXGIFactory2"))
    return false;
  LUID adapter_luid{};
  std::memcpy(&adapter_luid, luid, sizeof(adapter_luid));
  ComPtr<IDXGIAdapter1> adapter;
  if (failed(factory->EnumAdapterByLuid(adapter_luid, IID_PPV_ARGS(&adapter)),
             "EnumAdapterByLuid"))
    return false;

  ComPtr<ID3D11Device> device;
  ComPtr<ID3D11DeviceContext> context;
  constexpr D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1,
                                          D3D_FEATURE_LEVEL_11_0};
  if (failed(D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr,
                               D3D11_CREATE_DEVICE_BGRA_SUPPORT, levels,
                               nx::cast<UINT>(std::size(levels)),
                               D3D11_SDK_VERSION, &device, nullptr, &context),
             "D3D11CreateDevice") ||
      failed(device.As(&s->device), "ID3D11Device5") ||
      failed(context.As(&s->context), "ID3D11DeviceContext4") ||
      failed(factory.As(&s->factory), "IDXGIFactory2"))
    return false;

  if (failed(s->device->CreateFence(0, D3D11_FENCE_FLAG_SHARED,
                                    IID_PPV_ARGS(&s->rendered)),
             "CreateFence") ||
      failed(s->device->CreateFence(0, D3D11_FENCE_FLAG_SHARED,
                                    IID_PPV_ARGS(&s->copied)),
             "CreateFence"))
    return false;

  DXGI_SWAP_CHAIN_DESC1 desc{};
  desc.Width = width;
  desc.Height = height;
  desc.Format = s->format;
  desc.SampleDesc.Count = 1;
  desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
  desc.BufferCount = 2;
  desc.Scaling = DXGI_SCALING_STRETCH;
  desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
  desc.AlphaMode = DXGI_ALPHA_MODE_PREMULTIPLIED;
  if (failed(s->factory->CreateSwapChainForComposition(s->device.Get(), &desc,
                                                       nullptr, &s->swapchain),
             "CreateSwapChainForComposition"))
    return false;

  ComPtr<IDXGIDevice> dxgi_device;
  if (failed(s->device.As(&dxgi_device), "IDXGIDevice") ||
      failed(DCompositionCreateDevice(dxgi_device.Get(),
                                      IID_PPV_ARGS(&s->composition)),
             "DCompositionCreateDevice") ||
      failed(s->composition->CreateTargetForHwnd(static_cast<HWND>(hwnd), TRUE,
                                                 &s->target),
             "CreateTargetForHwnd") ||
      failed(s->composition->CreateVisual(&s->visual), "CreateVisual") ||
      failed(s->visual->SetContent(s->swapchain.Get()), "SetContent") ||
      failed(s->target->SetRoot(s->visual.Get()), "SetRoot") ||
      failed(s->composition->Commit(), "Commit"))
    return false;

  if (!s->make_images(width, height))
    return false;
  m_state = std::move(s);
  return true;
}

bool CompositionPresenter::resize(const u32 width, const u32 height) {
  State &s = *m_state;
  s.images.clear();
  s.context->ClearState();
  s.context->Flush();
  if (failed(
          s.swapchain->ResizeBuffers(0, width, height, DXGI_FORMAT_UNKNOWN, 0),
          "ResizeBuffers"))
    return false;
  return s.make_images(width, height);
}

u32 CompositionPresenter::image_count() const noexcept {
  return m_state ? m_state->count : 0;
}

void *CompositionPresenter::image_handle(const u32 index) const {
  ComPtr<IDXGIResource1> resource;
  HANDLE handle = nullptr;
  if (failed(m_state->images[index].As(&resource), "IDXGIResource1") ||
      failed(resource->CreateSharedHandle(nullptr,
                                          DXGI_SHARED_RESOURCE_READ |
                                              DXGI_SHARED_RESOURCE_WRITE,
                                          nullptr, &handle),
             "CreateSharedHandle"))
    return nullptr;
  return handle;
}

void *CompositionPresenter::rendered_fence_handle() const {
  HANDLE handle = nullptr;
  return failed(m_state->rendered->CreateSharedHandle(nullptr, GENERIC_ALL,
                                                      nullptr, &handle),
                "CreateSharedHandle")
             ? nullptr
             : handle;
}

void *CompositionPresenter::copied_fence_handle() const {
  HANDLE handle = nullptr;
  return failed(m_state->copied->CreateSharedHandle(nullptr, GENERIC_ALL,
                                                    nullptr, &handle),
                "CreateSharedHandle")
             ? nullptr
             : handle;
}

bool CompositionPresenter::present(const u32 index, const u64 value,
                                   const bool vsync) {
  State &s = *m_state;
  ComPtr<ID3D11Texture2D> backbuffer;
  if (failed(s.context->Wait(s.rendered.Get(), value), "Wait") ||
      failed(s.swapchain->GetBuffer(0, IID_PPV_ARGS(&backbuffer)), "GetBuffer"))
    return false;
  s.context->CopyResource(backbuffer.Get(), s.images[index].Get());
  if (failed(s.context->Signal(s.copied.Get(), value), "Signal"))
    return false;
  const HRESULT presented = s.swapchain->Present(vsync ? 1 : 0, 0);
  return !failed(presented, "Present");
}

} // namespace nxe::rhi
