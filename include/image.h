#ifndef __IMAGE_H__
#define __IMAGE_H__

#include "allocator.h"

namespace opendip{

typedef enum OpenDIP_Image_FILE_Type_s
{
    OPENDIP_IMAGE_UNKOWN = 0x0,
    OPENDIP_IMAGE_RAW,
    OPENDIP_IMAGE_JPG,
    OPENDIP_IMAGE_TIF,
    OPENDIP_IMAGE_PNG,
    OPENDIP_IMAGE_BMP,
    OPENDIP_IMAGE_GIP,
    OPENDIP_IMAGE_ICO,
    OPENDIP_IMAGE_NUM,
}OpenDIP_Image_FILE_Type_e;

enum OpenDIP_Image_Type
{
    OPENDIP_IMAGE_RGB       = 1,
    OPENDIP_IMAGE_BGR       = 2,
    OPENDIP_IMAGE_GRAY      = 3,
    OPENDIP_IMAGE_RGBA      = 4,
};

extern OpenDIP_Image_FILE_Type_e GetImageTypeFromFile(char *filename);

class Image
{
public:
    // empty
    Image();
    // vec
    Image(int w, size_t elemsize = 4u, Allocator* allocator = 0);
    // image
    Image(int w, int h, size_t elemsize = 4u, Allocator* allocator = 0);
    // dim
    Image(int w, int h, int c, size_t elemsize = 4u, Allocator* allocator = 0);
    // packed vec
    Image(int w, size_t elemsize, int elempack, Allocator* allocator = 0);
    // packed image
    Image(int w, int h, size_t elemsize, int elempack, Allocator* allocator = 0);
    // packed dim
    Image(int w, int h, int c, size_t elemsize, int elempack, Allocator* allocator = 0);
    // copy
    Image(const Image& m);
    // external vec
    Image(int w, void* data, size_t elemsize = 4u, Allocator* allocator = 0);
    // external image
    Image(int w, int h, void* data, size_t elemsize = 4u, Allocator* allocator = 0);
    // external dim
    Image(int w, int h, int c, void* data, size_t elemsize = 4u, Allocator* allocator = 0);
    // external packed vec
    Image(int w, void* data, size_t elemsize, int elempack, Allocator* allocator = 0);
    // external packed image
    Image(int w, int h, void* data, size_t elemsize, int elempack, Allocator* allocator = 0);
    // external packed dim
    Image(int w, int h, int c, void* data, size_t elemsize, int elempack, Allocator* allocator = 0);
    // release
    ~Image();


    // assign
    Image& operator=(const Image& m);
    // set all
    void fill(float v);
    void fill(int v);

    template <typename T> void fill(T v);
    // deep copy
    Image clone(Allocator* allocator = 0) const;
    // reshape vec
    Image reshape(int w, Allocator* allocator = 0) const;
    // reshape image
    Image reshape(int w, int h, Allocator* allocator = 0) const;
    // reshape dim
    Image reshape(int w, int h, int c, Allocator* allocator = 0) const;
    // allocate vec
    void create(int w, size_t elemsize = 4u, Allocator* allocator = 0);
    // allocate image
    void create(int w, int h, size_t elemsize = 4u, Allocator* allocator = 0);
    // allocate dim
    void create(int w, int h, int c, size_t elemsize = 4u, Allocator* allocator = 0);
    // allocate packed vec
    void create(int w, size_t elemsize, int elempack, Allocator* allocator = 0);
    // allocate packed image
    void create(int w, int h, size_t elemsize, int elempack, Allocator* allocator = 0);
    // allocate packed dim
    void create(int w, int h, int c, size_t elemsize, int elempack, Allocator* allocator = 0);
    // allocate like
    void create_like(const Image& m, Allocator* allocator = 0);

    // refcount++
    void addref();
    // refcount--
    void release();

    bool empty() const;
    size_t total() const;

    // data reference
    Image channel(int c);
    const Image channel(int c) const;
    float* row(int y);
    const float* row(int y) const;
    template<typename T> T* row(int y);
    template<typename T> const T* row(int y) const;

    // range reference
    Image channel_range(int c, int channels);
    const Image channel_range(int c, int channels) const;
    Image row_range(int y, int rows);
    const Image row_range(int y, int rows) const;
    Image range(int x, int n);
    const Image range(int x, int n) const;

    // access raw data
    template<typename T> operator T*();
    template<typename T> operator const T*() const;

    // convenient access float vec element
    float& operator[](int i);
    const float& operator[](int i) const;

    // substract channel-wise mean values, then multiply by normalize values, pass 0 to skip
    void substract_mean_normalize(const float* mean_vals, const float* norm_vals);

    // convenient construct from half precisoin floating point data
    static Image from_float16(const unsigned short* data, int size);

    // pointer to the data
    void* data;

    // pointer to the reference counter
    // when points to user-allocated data, the pointer is NULL
    int* refcount;

    // element size in bytes
    // 4 = float32/int32
    // 2 = float16
    // 1 = int8/uint8
    // 0 = empty
    size_t elemsize;

    // packed count inside element
    // c/1-h-w-1  h/1-w-1  w/1-1  scalar
    // c/4-h-w-4  h/4-w-4  w/4-4  sse/neon
    // c/8-h-w-8  h/8-w-8  w/8-8  avx/fp16
    int elempack;

    // the allocator
    Allocator* allocator;

    // the dimensionality
    int dims;

    int w;
    int h;
    int c;

    size_t cstep;
};


inline Image::Image()
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
}

inline Image::Image(int _w, size_t _elemsize, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _elemsize, _allocator);
}

inline Image::Image(int _w, int _h, size_t _elemsize, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _h, _elemsize, _allocator);
}

inline Image::Image(int _w, int _h, int _c, size_t _elemsize, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _h, _c, _elemsize, _allocator);
}

inline Image::Image(int _w, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _elemsize, _elempack, _allocator);
}

inline Image::Image(int _w, int _h, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _h, _elemsize, _elempack, _allocator);
}

inline Image::Image(int _w, int _h, int _c, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0)
{
    create(_w, _h, _c, _elemsize, _elempack, _allocator);
}

inline Image::Image(const Image& m)
    : data(m.data), refcount(m.refcount), elemsize(m.elemsize), elempack(m.elempack), allocator(m.allocator), dims(m.dims), w(m.w), h(m.h), c(m.c), cstep(m.cstep)
{
    if (refcount)
        OPENDIP_XADD(refcount, 1);
}

inline Image::Image(int _w, void* _data, size_t _elemsize, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(1), w(_w), h(1), c(1)
{
    cstep = w;
}

inline Image::Image(int _w, int _h, void* _data, size_t _elemsize, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(2), w(_w), h(_h), c(1)
{
    cstep = w * h;
}

inline Image::Image(int _w, int _h, int _c, void* _data, size_t _elemsize, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(1), allocator(_allocator), dims(3), w(_w), h(_h), c(_c)
{
    cstep = alignSize(w * h * elemsize, 16) / elemsize;
}

inline Image::Image(int _w, void* _data, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(1), w(_w), h(1), c(1)
{
    cstep = w;
}

inline Image::Image(int _w, int _h, void* _data, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(2), w(_w), h(_h), c(1)
{
    cstep = w * h;
}

inline Image::Image(int _w, int _h, int _c, void* _data, size_t _elemsize, int _elempack, Allocator* _allocator)
    : data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(3), w(_w), h(_h), c(_c)
{
    cstep = alignSize(w * h * elemsize, 16) / elemsize;
}

inline Image::~Image()
{
    release();
}

inline Image& Image::operator=(const Image& m)
{
    if (this == &m)
        return *this;

    if (m.refcount)
        OPENDIP_XADD(m.refcount, 1);

    release();

    data = m.data;
    refcount = m.refcount;
    elemsize = m.elemsize;
    elempack = m.elempack;
    allocator = m.allocator;

    dims = m.dims;
    w = m.w;
    h = m.h;
    c = m.c;

    cstep = m.cstep;

    return *this;
}

inline void Image::fill(float _v)
{
    int size = (int)total();
    float* ptr = (float*)data;
    int remain = size;
    for (; remain>0; remain--)
    {
        *ptr++ = _v;
    }
}

inline void Image::fill(int _v)
{
    int size = (int)total();
    int* ptr = (int*)data;
    int remain = size;
    for (; remain>0; remain--)
    {
        *ptr++ = _v;
    }
}



template <typename T>
inline void Image::fill(T _v)
{
    int size = total();
    T* ptr = (T*)data;
    for (int i=0; i<size; i++)
    {
        ptr[i] = _v;
    }
}

inline Image Image::clone(Allocator* allocator) const
{
    if (empty())
        return Image();

    Image m;
    if (dims == 1)
        m.create(w, elemsize, elempack, allocator);
    else if (dims == 2)
        m.create(w, h, elemsize, elempack, allocator);
    else if (dims == 3)
        m.create(w, h, c, elemsize, elempack, allocator);

    if (total() > 0)
    {
        memcpy(m.data, data, total() * elemsize);
    }

    return m;
}

inline Image Image::reshape(int _w, Allocator* _allocator) const
{
    if (w * h * c != _w)
        return Image();

    if (dims == 3 && cstep != (size_t)w * h)
    {
        Image m;
        m.create(_w, elemsize, elempack, _allocator);

        // flatten
        for (int i=0; i<c; i++)
        {
            const void* ptr = (unsigned char*)data + i * cstep * elemsize;
            void* mptr = (unsigned char*)m.data + i * w * h * elemsize;
            memcpy(mptr, ptr, w * h * elemsize);
        }

        return m;
    }

    Image m = *this;

    m.dims = 1;
    m.w = _w;
    m.h = 1;
    m.c = 1;

    m.cstep = _w;

    return m;
}

inline Image Image::reshape(int _w, int _h, Allocator* _allocator) const
{
    if (w * h * c != _w * _h)
        return Image();

    if (dims == 3 && cstep != (size_t)w * h)
    {
        Image m;
        m.create(_w, _h, elemsize, elempack, _allocator);

        // flatten
        for (int i=0; i<c; i++)
        {
            const void* ptr = (unsigned char*)data + i * cstep * elemsize;
            void* mptr = (unsigned char*)m.data + i * w * h * elemsize;
            memcpy(mptr, ptr, w * h * elemsize);
        }

        return m;
    }

    Image m = *this;

    m.dims = 2;
    m.w = _w;
    m.h = _h;
    m.c = 1;

    m.cstep = _w * _h;

    return m;
}

inline Image Image::reshape(int _w, int _h, int _c, Allocator* _allocator) const
{
    if (w * h * c != _w * _h * _c)
        return Image();

    if (dims < 3)
    {
        if ((size_t)_w * _h != alignSize(_w * _h * elemsize, 16) / elemsize)
        {
            Image m;
            m.create(_w, _h, _c, elemsize, elempack, _allocator);

            // align channel
            for (int i=0; i<_c; i++)
            {
                const void* ptr = (unsigned char*)data + i * _w * _h * elemsize;
                void* mptr = (unsigned char*)m.data + i * m.cstep * m.elemsize;
                memcpy(mptr, ptr, _w * _h * elemsize);
            }

            return m;
        }
    }
    else if (c != _c)
    {
        // flatten and then align
        Image tmp = reshape(_w * _h * _c, _allocator);
        return tmp.reshape(_w, _h, _c, _allocator);
    }

    Image m = *this;

    m.dims = 3;
    m.w = _w;
    m.h = _h;
    m.c = _c;

    m.cstep = alignSize(_w * _h * elemsize, 16) / elemsize;

    return m;
}

inline void Image::create(int _w, size_t _elemsize, Allocator* _allocator)
{
    if (dims == 1 && w == _w && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 1;
    w = _w;
    h = 1;
    c = 1;

    cstep = w;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

inline void Image::create(int _w, int _h, size_t _elemsize, Allocator* _allocator)
{
    if (dims == 2 && w == _w && h == _h && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 2;
    w = _w;
    h = _h;
    c = 1;

    cstep = w * h;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

inline void Image::create(int _w, int _h, int _c, size_t _elemsize, Allocator* _allocator)
{
    if (dims == 3 && w == _w && h == _h && c == _c && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = 1;
    allocator = _allocator;

    dims = 3;
    w = _w;
    h = _h;
    c = _c;

    cstep = alignSize(w * h * elemsize, 16) / elemsize;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

inline void Image::create(int _w, size_t _elemsize, int _elempack, Allocator* _allocator)
{
    if (dims == 1 && w == _w && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 1;
    w = _w;
    h = 1;
    c = 1;

    cstep = w;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

inline void Image::create(int _w, int _h, size_t _elemsize, int _elempack, Allocator* _allocator)
{
    if (dims == 2 && w == _w && h == _h && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 2;
    w = _w;
    h = _h;
    c = 1;

    cstep = w * h;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

inline void Image::create(int _w, int _h, int _c, size_t _elemsize, int _elempack, Allocator* _allocator)
{
    if (dims == 3 && w == _w && h == _h && c == _c && elemsize == _elemsize && elempack == _elempack && allocator == _allocator)
        return;

    release();

    elemsize = _elemsize;
    elempack = _elempack;
    allocator = _allocator;

    dims = 3;
    w = _w;
    h = _h;
    c = _c;

    cstep = alignSize(w * h * elemsize, 16) / elemsize;

    if (total() > 0)
    {
        size_t totalsize = alignSize(total() * elemsize, 4);
        if (allocator)
            data = allocator->fastMalloc(totalsize + (int)sizeof(*refcount));
        else
            data = fastMalloc(totalsize + (int)sizeof(*refcount));
        refcount = (int*)(((unsigned char*)data) + totalsize);
        *refcount = 1;
    }
}

inline void Image::create_like(const Image& m, Allocator* _allocator)
{
    if (m.dims == 1)
        create(m.w, m.elemsize, m.elempack, _allocator);
    else if (m.dims == 2)
        create(m.w, m.h, m.elemsize, m.elempack, _allocator);
    else if (m.dims == 3)
        create(m.w, m.h, m.c, m.elemsize, m.elempack, _allocator);
}

inline void Image::addref()
{
    if (refcount)
        OPENDIP_XADD(refcount, 1);
}

inline void Image::release()
{
    if (refcount && OPENDIP_XADD(refcount, -1) == 1)
    {
        if (allocator)
            allocator->fastFree(data);
        else
            fastFree(data);
    }

    data = 0;

    elemsize = 0;
    elempack = 0;

    dims = 0;
    w = 0;
    h = 0;
    c = 0;

    cstep = 0;

    refcount = 0;
}

inline bool Image::empty() const
{
    return data == 0 || total() == 0;
}

inline size_t Image::total() const
{
    return cstep * c;
}

inline Image Image::channel(int _c)
{
    return Image(w, h, (unsigned char*)data + cstep * _c * elemsize, elemsize, elempack, allocator);
}

inline const Image Image::channel(int _c) const
{
    return Image(w, h, (unsigned char*)data + cstep * _c * elemsize, elemsize, elempack, allocator);
}

inline float* Image::row(int y)
{
    return (float*)((unsigned char*)data + w * y * elemsize);
}

inline const float* Image::row(int y) const
{
    return (const float*)((unsigned char*)data + w * y * elemsize);
}

template <typename T>
inline T* Image::row(int y)
{
    return (T*)((unsigned char*)data + w * y * elemsize);
}

template <typename T>
inline const T* Image::row(int y) const
{
    return (const T*)((unsigned char*)data + w * y * elemsize);
}

inline Image Image::channel_range(int _c, int channels)
{
    return Image(w, h, channels, (unsigned char*)data + cstep * _c * elemsize, elemsize, elempack, allocator);
}

inline const Image Image::channel_range(int _c, int channels) const
{
    return Image(w, h, channels, (unsigned char*)data + cstep * _c * elemsize, elemsize, elempack, allocator);
}

inline Image Image::row_range(int y, int rows)
{
    return Image(w, rows, (unsigned char*)data + w * y * elemsize, elemsize, elempack, allocator);
}

inline const Image Image::row_range(int y, int rows) const
{
    return Image(w, rows, (unsigned char*)data + w * y * elemsize, elemsize, elempack, allocator);
}

inline Image Image::range(int x, int n)
{
    return Image(n, (unsigned char*)data + x * elemsize, elemsize, elempack, allocator);
}

inline const Image Image::range(int x, int n) const
{
    return Image(n, (unsigned char*)data + x * elemsize, elemsize, elempack, allocator);
}

template <typename T>
inline Image::operator T*()
{
    return (T*)data;
}

template <typename T>
inline Image::operator const T*() const
{
    return (const T*)data;
}

inline float& Image::operator[](int i)
{
    return ((float*)data)[i];
}

inline const float& Image::operator[](int i) const
{
    return ((const float*)data)[i];
}



}// namespace opendip

#endif