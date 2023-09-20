#ifndef __ADAMANT_CONFIG
#define __ADAMANT_CONFIG

#include <cstdint>

namespace adamant
{

constexpr uint32_t ADM_MEM_MIN_ALLOC = 8;
constexpr uint32_t ADM_MEM_STATIC_BUFFER = 4096;

constexpr uint32_t ADM_DB_OBJ_BLOCKSIZE = 64;

constexpr uint32_t ADM_META_BASE_BLOCKSIZE = 256;
constexpr uint32_t ADM_META_STACK_BLOCKSIZE = 256;
constexpr uint32_t ADM_META_STACK_DEPTH = 8;
constexpr uint32_t ADM_META_STACK_NAMEL = 16;
constexpr uint32_t ADM_META_STACK_NAMES = ADM_META_STACK_NAMEL*ADM_META_STACK_DEPTH;

constexpr uint32_t ADM_ELF_NAMES_BLOCKSIZE = 4096;
constexpr uint32_t ADM_ELF_HASHES_BLOCKSIZE = 256;

constexpr uint8_t ADM_ELF_SYMS_MAPLOG = 7;
constexpr uint8_t ADM_ELF_LIBS_MAPLOG = 4;

constexpr uint32_t ADM_MAX_PATH = 256;
constexpr uint32_t ADM_MAX_STRING = 256;

#define ADM_WARNING (0)
#define ADM_DEBUG (0)
#define ADM_API (0)

}

#endif
