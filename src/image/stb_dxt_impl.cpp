/**
 * @file stb_dxt_impl.cpp
 * @brief The single translation unit that compiles stb_dxt's BC1-5 block
 * compressor, used to cook block-compressed textures.
 */

#include <cstring> // stb_dxt's implementation calls memcpy without including it

#define STB_DXT_IMPLEMENTATION
#include <stb_dxt.h>
