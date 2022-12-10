#ifndef AKNN_VEC_H
#define AKNN_VEC_H

#include <array>
#include <vector>
#include <algorithm>
#include <initializer_list>
#include <limits>

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

template<typename FloatT, int Dim>
struct Box;

template<typename FloatT>
struct BoxSplit
{
    int dim;
    FloatT value;
    Box left;
    Box right;
};

// Axis aligned bounding box
template<typename FloatT, int Dim>
struct Box
{
    Vec<FloatT, Dim> min;
    Vec<FloatT, Dim> max;

    Box() : min(std::numeric_limits<FloatT>::infinity()), max(-std::numeric_limits<FloatT>::infinity()) {}

    static Box GetBoundingBox(const std::vector<Vec<FloatT, Dim>>& points) {
        Box bbox;
        std::for_each(points.begin(), points.end(), [&bbox](const Vec<FloatT, Dim>& point) { bbox.Include(point); });
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

    BoxSplit FairSplit() const {
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

// Base class for objects represented by a point
template<typename FloatT, int Dim>
class PointObject
{
public:
    Vec<FloatT, Dim> point;
};

#endif // AKNN_VEC_H
