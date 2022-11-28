#ifndef AKNN_VEC_H
#define AKNN_VEC_H

#include <array>

template<typename FloatT, int Dim>
struct Vec
{
    // values of the vector
    std::array<FloatT, Dim> v;

    const FloatT& operator[](int i) const { return v[i]; }
    FloatT& operator[](int i) { return v[i]; }
};

template<typename FloatT, int Dim>
struct AABB
{
    Vec<FloatT, Dim> min;
    Vec<FloatT, Dim> max;
};

#endif // AKNN_VEC_H
