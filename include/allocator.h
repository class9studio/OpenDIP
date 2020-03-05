
#ifndef _OPENDIP_ALLOCATOR_H
#define _OPENDIP_ALLOCATOR_H

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#endif

#include <stdlib.h>
#include <list>
#include <vector>

namespace opendip {

// the alignment of all the allocated buffers
#define MALLOC_ALIGN    16

// Aligns a pointer to the specified number of bytes
// ptr Aligned pointer
// n Alignment size that must be a power of two
template<typename _Tp> static inline _Tp* alignPtr(_Tp* ptr, int n=(int)sizeof(_Tp))
{
    return (_Tp*)(((size_t)ptr + n-1) & -n);
}

// Aligns a buffer size to the specified number of bytes
// The function returns the minimum number that is greater or equal to sz and is divisible by n
// sz Buffer size to align
// n Alignment size that must be a power of two
static inline size_t alignSize(size_t sz, int n)
{
    return (sz + n-1) & -n;
}

static inline void* fastMalloc(size_t size)
{
    unsigned char* udata = (unsigned char*)malloc(size + sizeof(void*) + MALLOC_ALIGN);
    if (!udata)
        return 0;
    unsigned char** adata = alignPtr((unsigned char**)udata + 1, MALLOC_ALIGN);
    adata[-1] = udata;
    return adata;
}

static inline void fastFree(void* ptr)
{
    if (ptr)
    {
        unsigned char* udata = ((unsigned char**)ptr)[-1];
        free(udata);
    }
}

static inline int OPENDIP_XADD(int* addr, int delta) { int tmp = *addr; *addr += delta; return tmp; }

class Allocator
{
public:
    ~Allocator()
    {

    };

    static inline void* fastMalloc(size_t size)
    {
        unsigned char* udata = (unsigned char*)malloc(size + sizeof(void*) + MALLOC_ALIGN);
        if (!udata)
            return 0;
        unsigned char** adata = alignPtr((unsigned char**)udata + 1, MALLOC_ALIGN);
        adata[-1] = udata;
        return adata;
    };
    void fastFree(void* ptr)
    {
        if (ptr)
        {
            unsigned char* udata = ((unsigned char**)ptr)[-1];
            free(udata);
        }
    }
};


} // namespace opendip

#endif // _OPENDIP_ALLOCATOR_H
