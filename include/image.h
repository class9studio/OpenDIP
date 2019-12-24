#ifndef __IMAGE_H__
#define __IMAGE_H__

#include "allocator.h"

namespace opendip{

enum OpenDIP_Image_FILE_Type
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
};

enum OpenDIP_Image_Type
{
    OPENDIP_IMAGE_RGB       = 1,
    OPENDIP_IMAGE_BGR       = 2,
    OPENDIP_IMAGE_GRAY      = 3,
    OPENDIP_IMAGE_RGBA      = 4,
};

enum OpenDIP_Channel_Type
{
    OPENDIP_CHANNEL_R = 0,
    OPENDIP_CHANNEL_G,
    OPENDIP_CHANNEL_B,
    OPENDIP_CHANNEL_NUM,
};

enum OpenDIP_ColorCvtGray_Type
{
    OPENDIP_COLORCVTGRAY_MAXMIN = 0,   //  最大最小平均法
    OPENDIP_COLORCVTGRAY_AVERAGE,      //  平均值
    OPENDIP_COLORCVTGRAY_WEIGHTED,     //  加权平均法
};

class Image
{
public:
    // empty
    Image(size_t elemsize = 4u, Allocator* allocator = 0);
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
    // allocate empty
    void create(size_t elemsize = 4u, Allocator* allocator = 0);    
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

	//image file type
	OpenDIP_Image_FILE_Type ftype;

    // stb-image or not
    bool is_stbimage;
};




}// namespace opendip

#endif