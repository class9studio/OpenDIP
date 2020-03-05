#ifndef __POINT_H__
#define __POINT_H__

namespace opendip{


template<typename _Tp> class Point_
{
public:
    typedef _Tp value_type;

    //! default constructor
    Point_();
    Point_(_Tp _x, _Tp _y);
    Point_(const Point_& pt);

    Point_& operator = (const Point_& pt);


    _Tp x; //!< x coordinate of the point
    _Tp y; //!< y coordinate of the point
};

typedef Point_<int> Point2i;
typedef Point_<float> Point2f;
typedef Point_<double> Point2d;
typedef Point2i Point;

template<typename _Tp> inline
Point_<_Tp>::Point_()
    : x(0), y(0) {}

template<typename _Tp> inline
Point_<_Tp>::Point_(_Tp _x, _Tp _y)
    : x(_x), y(_y) {}

template<typename _Tp> inline
Point_<_Tp>::Point_(const Point_& pt)
    : x(pt.x), y(pt.y) {}


template<typename _Tp> inline
Point_<_Tp>& Point_<_Tp>::operator = (const Point_& pt)
{
    x = pt.x; y = pt.y;
    return *this;
}

}

#endif //__POINT_H__