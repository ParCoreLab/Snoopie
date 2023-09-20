#ifndef __ADAMANT_COMMON
#define __ADAMANT_COMMON

#include <iostream>
#include <fstream>

namespace adamant
{

#define ADM_VISIBILITY __attribute__ ((visibility ("internal")))

//ADM_VISIBILITY extern unsigned char adm_tracing;
ADM_VISIBILITY extern const char* adm_module;

ADM_VISIBILITY
void adm_out(const void* buffer, const unsigned int size);

ADM_VISIBILITY
std::streamoff adm_conf_line(const char* var, char* line, std::streamoff offset=0);

ADM_VISIBILITY
bool adm_conf_string(const char* var, const char* find);

template <typename M, typename... V> static inline void dbgp(const M m, V... v) { std::cerr << m; dbgp(v...); };
template <typename M> static inline void dbgp(const M m) { std::cerr << m << std::endl; };

#if ADM_WARNING==1
#define adm_warning(...) dbgp<"ADAMANT WARNING:",__VA_ARGS__>()
#else
#define adm_warning(...) do{}while(0)
#endif

#if ADM_DEBUG==1
#define adm_debug(...) dbgp<"ADAMANT DEBUG:",__VA_ARGS__>()
#else
#define adm_debug(...) do{}while(0)
#endif

#if ADM_API==1
#define adm_api(...) dbgp<"ADAMANT API:",__VA_ARGS__>()
#else
#define adm_api(...) do{}while(0)
#endif

ADM_VISIBILITY extern void* (*malloc_ptr)(size_t);
ADM_VISIBILITY extern void  (*free_ptr)(void*);
ADM_VISIBILITY extern void* (*realloc_ptr)(void *,size_t);
ADM_VISIBILITY extern void* (*calloc_ptr)(size_t,size_t);
ADM_VISIBILITY extern void* (*valloc_ptr)(size_t);
ADM_VISIBILITY extern void* (*pvalloc_ptr)(size_t);
ADM_VISIBILITY extern void* (*memalign_ptr)(size_t,size_t);
ADM_VISIBILITY extern void* (*aligned_alloc_ptr)(size_t,size_t);
ADM_VISIBILITY extern int   (*posix_memalign_ptr)(void**,size_t,size_t);
ADM_VISIBILITY extern void* (*mmap_ptr)(void*,size_t,int,int,int,off_t);
ADM_VISIBILITY extern void* (*mmap64_ptr)(void*,size_t,int,int,int,off64_t);
ADM_VISIBILITY extern int   (*munmap_ptr)(void*,size_t);

ADM_VISIBILITY
void adm_posix_init();

ADM_VISIBILITY
void adm_posix_fini();

}

#endif
