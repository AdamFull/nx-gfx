/**
 * @file stb_image_resize_impl.cpp
 * @brief The single translation unit that compiles stb_image_resize2's
 * implementation, used to build mip chains at cook time.
 */

#define STBIR_NO_SIMD
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>
