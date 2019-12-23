#include <cstring>
#include <iostream>
#include "image.h"
#include "common.h"

namespace opendip {

  Image::Image(size_t _elemsize, Allocator* _allocator)
	: data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0), ftype(OPENDIP_IMAGE_UNKOWN), is_stbimage(false)
{
	create(_elemsize, _allocator);
}

  Image::Image(int _w, size_t _elemsize, Allocator* _allocator)
	: data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0), ftype(OPENDIP_IMAGE_UNKOWN), is_stbimage(false)
{
	create(_w, _elemsize, _allocator);
}

  Image::Image(int _w, int _h, size_t _elemsize, Allocator* _allocator)
	: data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0), ftype(OPENDIP_IMAGE_UNKOWN), is_stbimage(false)
{
	create(_w, _h, _elemsize, _allocator);
}

  Image::Image(int _w, int _h, int _c, size_t _elemsize, Allocator* _allocator)
	: data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0), ftype(OPENDIP_IMAGE_UNKOWN), is_stbimage(false)
{
	create(_w, _h, _c, _elemsize, _allocator);
}

  Image::Image(int _w, size_t _elemsize, int _elempack, Allocator* _allocator)
	: data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0), ftype(OPENDIP_IMAGE_UNKOWN), is_stbimage(false)
{
	create(_w, _elemsize, _elempack, _allocator);
}

  Image::Image(int _w, int _h, size_t _elemsize, int _elempack, Allocator* _allocator)
	: data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0), ftype(OPENDIP_IMAGE_UNKOWN), is_stbimage(false)
{
	create(_w, _h, _elemsize, _elempack, _allocator);
}

  Image::Image(int _w, int _h, int _c, size_t _elemsize, int _elempack, Allocator* _allocator)
	: data(0), refcount(0), elemsize(0), elempack(0), allocator(0), dims(0), w(0), h(0), c(0), cstep(0), ftype(OPENDIP_IMAGE_UNKOWN), is_stbimage(false)
{
	create(_w, _h, _c, _elemsize, _elempack, _allocator);
}

inline Image::Image(const Image& m)
	: data(m.data), refcount(m.refcount), elemsize(m.elemsize), elempack(m.elempack), allocator(m.allocator), dims(m.dims), w(m.w), h(m.h), c(m.c), \
	 cstep(m.cstep),ftype(m.ftype),is_stbimage(m.is_stbimage)
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

  Image::Image(int _w, void* _data, size_t _elemsize, int _elempack, Allocator* _allocator)
	: data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(1), w(_w), h(1), c(1)
{
	cstep = w;
}

  Image::Image(int _w, int _h, void* _data, size_t _elemsize, int _elempack, Allocator* _allocator)
	: data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(2), w(_w), h(_h), c(1)
{
	cstep = w * h;
}

  Image::Image(int _w, int _h, int _c, void* _data, size_t _elemsize, int _elempack, Allocator* _allocator)
	: data(_data), refcount(0), elemsize(_elemsize), elempack(_elempack), allocator(_allocator), dims(3), w(_w), h(_h), c(_c)
{
	cstep = alignSize(w * h * elemsize, 16) / elemsize;
}

  Image::~Image()
{
	release();
}

  Image& Image::operator=(const Image& m)
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
	ftype = m.ftype;
	is_stbimage = m.is_stbimage;

	dims = m.dims;
	w = m.w;
	h = m.h;
	c = m.c;

	cstep = m.cstep;

	return *this;
}

  void Image::fill(float _v)
{
	int size = (int)total();
	float* ptr = (float*)data;
	int remain = size;
	for (; remain > 0; remain--)
	{
		*ptr++ = _v;
	}
}

  void Image::fill(int _v)
{
	int size = (int)total();
	int* ptr = (int*)data;
	int remain = size;
	for (; remain > 0; remain--)
	{
		*ptr++ = _v;
	}
}



template <typename T>
  void Image::fill(T _v)
{
	int size = total();
	T* ptr = (T*)data;
	for (int i = 0; i < size; i++)
	{
		ptr[i] = _v;
	}
}

  Image Image::clone(Allocator* allocator) const
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

  Image Image::reshape(int _w, Allocator* _allocator) const
{
	if (w * h * c != _w)
		return Image();

	if (dims == 3 && cstep != (size_t)w * h)
	{
		Image m;
		m.create(_w, elemsize, elempack, _allocator);

		// flatten
		for (int i = 0; i < c; i++)
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

  Image Image::reshape(int _w, int _h, Allocator* _allocator) const
{
	if (w * h * c != _w * _h)
		return Image();

	if (dims == 3 && cstep != (size_t)w * h)
	{
		Image m;
		m.create(_w, _h, elemsize, elempack, _allocator);

		// flatten
		for (int i = 0; i < c; i++)
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

  Image Image::reshape(int _w, int _h, int _c, Allocator* _allocator) const
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
			for (int i = 0; i < _c; i++)
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

 void Image::create(size_t _elemsize, Allocator* _allocator)
{
	if (dims == 1 && w == 1 && elemsize == _elemsize && elempack == 1 && allocator == _allocator)
		return;

	release();

	elemsize = _elemsize;
	elempack = 1;
	allocator = _allocator;

	dims = 1;
	w = 1;
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

  void Image::create(int _w, size_t _elemsize, Allocator* _allocator)
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

  void Image::create(int _w, int _h, size_t _elemsize, Allocator* _allocator)
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

  void Image::create(int _w, int _h, int _c, size_t _elemsize, Allocator* _allocator)
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

  void Image::create(int _w, size_t _elemsize, int _elempack, Allocator* _allocator)
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

  void Image::create(int _w, int _h, size_t _elemsize, int _elempack, Allocator* _allocator)
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

  void Image::create(int _w, int _h, int _c, size_t _elemsize, int _elempack, Allocator* _allocator)
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

  void Image::create_like(const Image& m, Allocator* _allocator)
{
	if (m.dims == 1)
		create(m.w, m.elemsize, m.elempack, _allocator);
	else if (m.dims == 2)
		create(m.w, m.h, m.elemsize, m.elempack, _allocator);
	else if (m.dims == 3)
		create(m.w, m.h, m.c, m.elemsize, m.elempack, _allocator);
}

  void Image::addref()
{
	if (refcount)
		OPENDIP_XADD(refcount, 1);
}

  void Image::release()
{
	if (refcount && OPENDIP_XADD(refcount, -1) == 1)
	{
		if(is_stbimage)
		{
			StbFree(data);
		}
		else
		{
			if (allocator)
				allocator->fastFree(data);
			else
				fastFree(data);  
		}
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

  bool Image::empty() const
{
	return data == 0 || total() == 0;
}

  size_t Image::total() const
{
	return cstep * c;
}

  Image Image::channel(int _c)
{
	return Image(w, h, (unsigned char*)data + cstep * _c * elemsize, elemsize, elempack, allocator);
}

  const Image Image::channel(int _c) const
{
	return Image(w, h, (unsigned char*)data + cstep * _c * elemsize, elemsize, elempack, allocator);
}

  float* Image::row(int y)
{
	return (float*)((unsigned char*)data + w * y * elemsize);
}

  const float* Image::row(int y) const
{
	return (const float*)((unsigned char*)data + w * y * elemsize);
}

template <typename T>
  T* Image::row(int y)
{
	return (T*)((unsigned char*)data + w * y * elemsize);
}

template <typename T>
  const T* Image::row(int y) const
{
	return (const T*)((unsigned char*)data + w * y * elemsize);
}

  Image Image::channel_range(int _c, int channels)
{
	return Image(w, h, channels, (unsigned char*)data + cstep * _c * elemsize, elemsize, elempack, allocator);
}

  const Image Image::channel_range(int _c, int channels) const
{
	return Image(w, h, channels, (unsigned char*)data + cstep * _c * elemsize, elemsize, elempack, allocator);
}

  Image Image::row_range(int y, int rows)
{
	return Image(w, rows, (unsigned char*)data + w * y * elemsize, elemsize, elempack, allocator);
}

  const Image Image::row_range(int y, int rows) const
{
	return Image(w, rows, (unsigned char*)data + w * y * elemsize, elemsize, elempack, allocator);
}

  Image Image::range(int x, int n)
{
	return Image(n, (unsigned char*)data + x * elemsize, elemsize, elempack, allocator);
}

  const Image Image::range(int x, int n) const
{
	return Image(n, (unsigned char*)data + x * elemsize, elemsize, elempack, allocator);
}

template <typename T>
  Image::operator T*()
{
	return (T*)data;
}

template <typename T>
  Image::operator const T*() const
{
	return (const T*)data;
}

  float& Image::operator[](int i)
{
	return ((float*)data)[i];
}

  const float& Image::operator[](int i) const
{
	return ((const float*)data)[i];
}


}  //namespace opendip