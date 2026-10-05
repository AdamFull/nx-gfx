// Slang reserves the module name `core` and resolves imports by file name.
// Keep the engine implementation in core.slang while giving it an unambiguous
// module identity for imports.
module nx_core;

__exported import nx_texture_layout;

#include "core.slang"
