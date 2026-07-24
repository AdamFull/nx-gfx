/**
 * @file stb_image_impl.cpp
 * @brief The single translation unit that compiles stb_image's implementation.
 */

// No SIMD intrinsics path: keep the decoder portable across the desktop and
// mobile targets the RHI is built for, at a small cost to decode speed.
// TODO: STB will not be used as a delivery format
#define STBI_NO_SIMD
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
