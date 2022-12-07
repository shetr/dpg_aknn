#ifndef AKNN_VEC_H
#define AKNN_VEC_H

#include <array>
#include <initializer_list>

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
struct Box
{
    Vec<FloatT, Dim> min;
    Vec<FloatT, Dim> max;

    FloatT GetSize(int dim) const { return max[dim] - min[dim]; }

    FloatT SquaredDistance(const Vec<FloatT, Dim>& point)
    {
        FloatT dist = 0;
        for (int d = 0; d < Dim; ++d) {
            dist += Square(std::max(0, min[d] - point[d]));
            dist += Square(std::max(0, point[d] - max[d]));
        }
        return dist;
    }
};

#endif // AKNN_VEC_H
