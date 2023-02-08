#ifndef AKNN_VEC_H
#define AKNN_VEC_H

#include <array>
#include <vector>
#include <algorithm>
#include <initializer_list>
#include <limits>
#include <iostream>

//! Vector of values of specified type T with element count equal to Dim
template<typename T, int Dim>
struct Vec
{
    //! values of the vector
    std::array<T, Dim> v;

    //! Zero vector
    Vec() : Vec(0) {}
    //! Fill with value
    Vec(T value) { for (int i = 0; i < Dim; ++i) { v[i] = value; } }
    //! Initialize with specified values
    Vec(std::initializer_list<T> il) { int i = 0; for (T val : il) { v[i] = val; ++i; } }

    //! Array access operator, read only
    const T& operator[](int i) const { return v[i]; }
    //! Array access operator
    T& operator[](int i) { return v[i]; }

    //! Subtraction of 2 vectors
    Vec<T, Dim> operator-(const Vec<T, Dim>& other) const {
        Vec<T, Dim> res;
        for (int i = 0; i < Dim; ++i) {
            res[i] = v[i] - other[i];
        }
        return res;
    }

    //! Returns true if values inside vectors are equal
    bool operator==(const Vec<T, Dim>& other) const {
        for (int i = 0; i < Dim; ++i)
            if (v[i] != other[i])
                return false;
        return true;
    }

    //! Squared euclidean norm of the vector
    T LengthSquared() const {
        T res = 0;
        for (int i = 0; i < Dim; ++i) {
            res += v[i] * v[i];
        }
        return res;
    }

    //! Squared euclidean distance between two vectors
    T DistSquared(const Vec<T, Dim>& other) const {
        return (other - *this).LengthSquared();
    }
};

//! Just squares the value
template<typename T>
T Square(T v) {
    return v * v;
}

//! Empty struct, used for omiting optional data
struct Empty {};

//! Object represented by a point. Has optional field data, by default with type Empty, in which case it should be ignored.
template<typename FloatT, int Dim, typename ObjData = Empty>
struct PointObj
{
    //! Point representing the object
    Vec<FloatT, Dim> point;
    //! Optional custom data (color, normal, etc.)
    ObjData data;
};

//! Axis aligned bounding box
template<typename FloatT, int Dim>
struct Box;

//! Represents split of some bounding box at some dimension to 2 smaller boxes
template<typename FloatT, int Dim>
struct BoxSplit
{
    //! split dimension
    int dim;
    //! position of the split at the split dimension
    FloatT value;
    //! left split result
    Box<FloatT, Dim> left;
    //! right split result
    Box<FloatT, Dim> right;
};

//! Axis aligned bounding box
template<typename FloatT, int Dim>
struct Box
{
    //! Min corner of the box
    Vec<FloatT, Dim> min;
    //! Max corner of the box
    Vec<FloatT, Dim> max;

    //! Initializes to invalid box, representing empty volume
    Box() : min(std::numeric_limits<FloatT>::infinity()), max(-std::numeric_limits<FloatT>::infinity()) {}
    //! Initializes to specified values
    Box(Vec<FloatT, Dim> min, Vec<FloatT, Dim> max) : min(min), max(max) {}

    //! Check if boxes are equal
    bool operator==(const Box<FloatT, Dim>& other) const {
        return min == other.min && max == other.max;
    }

    //! Create tight bounding box from array of points
    template<typename ObjData = Empty>
    static Box GetBoundingBox(const std::vector<PointObj<FloatT, Dim, ObjData>>& objs) {
        Box bbox;
        std::for_each(objs.begin(), objs.end(), [&bbox](const PointObj<FloatT, Dim, ObjData>& obj) { bbox.Include(obj.point); });
        return bbox;
    }

    //! Size of the box in specified dimension
    FloatT GetSize(int dim) const { return max[dim] - min[dim]; }

    //! Computes squared euclidean distnace of specified point to this box
    FloatT SquaredDistance(const Vec<FloatT, Dim>& point) const
    {
        FloatT dist = 0;
        for (int d = 0; d < Dim; ++d) {
            dist += Square<FloatT>(std::max((FloatT)0, min[d] - point[d]));
            dist += Square<FloatT>(std::max((FloatT)0, point[d] - max[d]));
        }
        return dist;
    }

    //! Check if point is inside the box
    bool Includes(const Vec<FloatT, Dim>& point) const
    {
        for (int d = 0; d < Dim; ++d) {
            if (point[d] < min[d] || point[d] > max[d])
                return false;
        }
        return true;
    };

    //! Extend the box, so that the specified point is inside
    void Include(const Vec<FloatT, Dim>& point)
    {
        for (int d = 0; d < Dim; ++d) {
            min[d] = std::min(min[d], point[d]);
            max[d] = std::max(max[d], point[d]);
        }
    }

    //! Splits the box in 2 halfs at dimension of the greatest size
    BoxSplit<FloatT, Dim> Split() const {
        int splitDim = 0;
        FloatT maxSize = 0;
        for (int d = 0; d < Dim; ++d) {
            FloatT dSize = GetSize(d);
            if (dSize > maxSize) {
                maxSize = dSize;
                splitDim = d;
            }
        }
        Box left = *this;
        Box right = *this;
        FloatT value = min[splitDim] + maxSize / 2;
        left.max[splitDim] = value;
        right.min[splitDim] = value;
        return {splitDim, value, left, right};
    }
};

//! Used for printing vector of values
template<typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
    os << "[";
    for(int i = 0; i < v.size(); ++i) {
        os << v[i] << (i + 1 == v.size() ? "" : ", ");
    }
    os << "]";
    return os;
}
//! Used for printing array of values
template<typename T, int N>
inline std::ostream& operator<<(std::ostream& os, const std::array<T, N>& v) {
    os << "[";
    for(int i = 0; i < v.size(); ++i) {
        os << v[i] << (i + 1 == v.size() ? "" : ", ");
    }
    os << "]";
    return os;
}
//! Used for printing Vec
template<typename FloatT, int Dim>
inline std::ostream& operator<<(std::ostream& os, const Vec<FloatT, Dim>& v) {
    os << "(";
    for(int d = 0; d < Dim; ++d) {
        os << v[d] << (d + 1 == Dim ? "" : ", ");
    }
    os << ")";
    return os;
}
//! Used for printing Box
template<typename FloatT, int Dim>
inline std::ostream& operator<<(std::ostream& os, const Box<FloatT, Dim>& box) {
    os << "{min: " << box.min << ", max: " << box.max << "}";
    return os;
}
//! Simply converts PointObj to Vec
template<typename FloatT, int Dim, typename ObjData = Empty>
std::vector<Vec<FloatT, Dim>> ObjsToVec(const std::vector<PointObj<FloatT, Dim, ObjData>>& objs) {
    std::vector<Vec<FloatT, Dim>> res;
    res.resize(objs.size());
    std::transform(objs.begin(), objs.end(), res.begin(), [&](const PointObj<FloatT, Dim, ObjData>& obj) { return obj.point; });
    return std::move(res);
}
//! Sorts array of points by distance to specified point
template<typename FloatT, int Dim>
void SortByDistanceToPoint(std::vector<Vec<FloatT, Dim>>& points, const Vec<FloatT, Dim>& point) {
    std::sort(points.begin(), points.end(), [&point](const Vec<FloatT, Dim>& p1, const Vec<FloatT, Dim>& p2) {
        return point.DistSquared(p1) < point.DistSquared(p2);
    });
}

// Few redeclaration to simplify some testing code

template<int Dim>
using VecF = Vec<float, Dim>;
template<int Dim>
using BoxF = Box<float, Dim>;
template<int Dim>
using PointObjF = PointObj<float, Dim>;

template<int Dim>
using VecD = Vec<double, Dim>;
template<int Dim>
using BoxD = Box<double, Dim>;
template<int Dim>
using PointObjD = PointObj<double, Dim>;

using VecF1 = Vec<float, 1>;
using VecF2 = Vec<float, 2>;
using VecF3 = Vec<float, 3>;
using VecF4 = Vec<float, 4>;

using VecD1 = Vec<double, 1>;
using VecD2 = Vec<double, 2>;
using VecD3 = Vec<double, 3>;
using VecD4 = Vec<double, 4>;

using BoxF1 = Box<float, 1>;
using BoxF2 = Box<float, 2>;
using BoxF3 = Box<float, 3>;
using BoxF4 = Box<float, 4>;

using BoxD1 = Box<double, 1>;
using BoxD2 = Box<double, 2>;
using BoxD3 = Box<double, 3>;
using BoxD4 = Box<double, 4>;

using PointObjF1 = PointObj<float, 1>;
using PointObjF2 = PointObj<float, 2>;
using PointObjF3 = PointObj<float, 3>;
using PointObjF4 = PointObj<float, 4>;

using PointObjD1 = PointObj<double, 1>;
using PointObjD2 = PointObj<double, 2>;
using PointObjD3 = PointObj<double, 3>;
using PointObjD4 = PointObj<double, 4>;

#endif // AKNN_VEC_H
