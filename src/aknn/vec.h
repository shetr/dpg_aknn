#ifndef AKNN_VEC_H
#define AKNN_VEC_H

#include <array>
#include <vector>
#include <algorithm>
#include <initializer_list>
#include <limits>
#include <iostream>

template<typename T, int Dim>
struct Vec
{
    // values of the vector
    std::array<T, Dim> v;

    Vec() : Vec(0) {}
    
    Vec(T value) { for (int i = 0; i < Dim; ++i) { v[i] = value; } }

    Vec(std::initializer_list<T> il) { int i = 0; for (T val : il) { v[i] = val; ++i; } }

    const T& operator[](int i) const { return v[i]; }
    T& operator[](int i) { return v[i]; }

    Vec<T, Dim> operator-(const Vec<T, Dim>& other) const {
        Vec<T, Dim> res;
        for (int i = 0; i < Dim; ++i) {
            res[i] = v[i] - other[i];
        }
        return res;
    }

    bool operator==(const Vec<T, Dim>& other) const {
        for (int i = 0; i < Dim; ++i)
            if (v[i] != other[i])
                return false;
        return true;
    }

    T LengthSquared() const {
        T res = 0;
        for (int i = 0; i < Dim; ++i) {
            res += v[i] * v[i];
        }
        return res;
    }

    T DistSquared(const Vec<T, Dim>& other) const {
        return (other - *this).LengthSquared();
    }
};

template<typename FloatT>
FloatT Square(FloatT v) {
    return v * v;
}

// Object represented by a point
template<typename FloatT, int Dim>
struct PointObj
{
    Vec<FloatT, Dim> point;
    void* customData = nullptr;
};

template<typename FloatT, int Dim>
struct Box;

template<typename FloatT, int Dim>
struct BoxSplit
{
    int dim;
    FloatT value;
    Box<FloatT, Dim> left;
    Box<FloatT, Dim> right;
};

// Axis aligned bounding box
template<typename FloatT, int Dim>
struct Box
{
    Vec<FloatT, Dim> min;
    Vec<FloatT, Dim> max;

    Box() : min(std::numeric_limits<FloatT>::infinity()), max(-std::numeric_limits<FloatT>::infinity()) {}
    Box(Vec<FloatT, Dim> min, Vec<FloatT, Dim> max) : min(min), max(max) {}

    bool operator==(const Box<FloatT, Dim>& other) const {
        return min == other.min && max == other.max;
    }

    static Box GetBoundingBox(const std::vector<PointObj<FloatT, Dim>>& objs) {
        Box bbox;
        std::for_each(objs.begin(), objs.end(), [&bbox](const PointObj<FloatT, Dim>& obj) { bbox.Include(obj.point); });
        return bbox;
    }

    FloatT GetSize(int dim) const { return max[dim] - min[dim]; }

    FloatT SquaredDistance(const Vec<FloatT, Dim>& point) const
    {
        FloatT dist = 0;
        for (int d = 0; d < Dim; ++d) {
            dist += Square(std::max(0, min[d] - point[d]));
            dist += Square(std::max(0, point[d] - max[d]));
        }
        return dist;
    }

    void Include(const Vec<FloatT, Dim>& point)
    {
        for (int d = 0; d < Dim; ++d) {
            min[d] = std::min(min[d], point[d]);
            max[d] = std::max(max[d], point[d]);
        }
    }

    BoxSplit<FloatT, Dim> FairSplit() const {
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

template<typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
    os << "[";
    for(int i = 0; i < v.size(); ++i) {
        os << v[i] << (i + 1 == v.size() ? "" : ", ");
    }
    os << "]";
    return os;
}

template<typename T, int N>
inline std::ostream& operator<<(std::ostream& os, const std::array<T, N>& v) {
    os << "[";
    for(int i = 0; i < v.size(); ++i) {
        os << v[i] << (i + 1 == v.size() ? "" : ", ");
    }
    os << "]";
    return os;
}

template<typename FloatT, int Dim>
inline std::ostream& operator<<(std::ostream& os, const Vec<FloatT, Dim>& v) {
    os << "(";
    for(int d = 0; d < Dim; ++d) {
        os << v[d] << (d + 1 == Dim ? "" : ", ");
    }
    os << ")";
    return os;
}

template<typename FloatT, int Dim>
inline std::ostream& operator<<(std::ostream& os, const Box<FloatT, Dim>& box) {
    os << "{min: " << box.min << ", max: " << box.max << "}";
    return os;
}

template<typename FloatT, int Dim>
std::vector<Vec<FloatT, Dim>> ObjsToVec(const std::vector<PointObj<FloatT, Dim>>& objs) {
    std::vector<Vec<FloatT, Dim>> res;
    res.resize(objs.size());
    std::transform(objs.begin(), objs.end(), res.begin(), [&](const PointObj<FloatT, Dim>& obj) { return obj.point; });
    return res;
}

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
