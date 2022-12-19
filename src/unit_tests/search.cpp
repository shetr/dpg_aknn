
#include <gtest/gtest.h>
#include <aknn/search.h>

#include "test_data.h"
/*
template<typename FloatT, int Dim>
void SortByDistanceToPoint(std::vector<Vec<FloatT, Dim>>& points, const Vec<FloatT, Dim>& point) {
    std::sort(points.begin(), points.end(), [&point](const Vec<FloatT, Dim>& p1, const Vec<FloatT, Dim>& p2) {
        return point.DistSquared(p1) < point.DistSquared(p2);
    });
}*/

TEST(Search_LinearFindNN, line2d10) {
    VecD2 expected(5.0);
    PointObjD2 res = LinearFindNearestNeighbor<double, 2>(TestData::Get().line2d10, VecD2(5.1));
    EXPECT_EQ(expected, res.point);
}
TEST(Search_LinearFindNN, line3d10) {
    VecD3 expected(5.0);
    PointObjD3 res = LinearFindNearestNeighbor<double, 3>(TestData::Get().line3d10, VecD3(5.1));
    EXPECT_EQ(expected, res.point);
}
TEST(Search_LinearFindNN, line4d10) {
    VecD4 expected(5.0);
    PointObjD4 res = LinearFindNearestNeighbor<double, 4>(TestData::Get().line4d10, VecD4(5.1));
    EXPECT_EQ(expected, res.point);
}
TEST(Search_LinearFindNN, grid2d10) {
    VecD2 expected({5.0, 4.0});
    PointObjD2 res = LinearFindNearestNeighbor<double, 2>(TestData::Get().grid2d10, VecD2({5.1, 3.9}));
    EXPECT_EQ(expected, res.point);
}
TEST(Search_LinearFindNN, grid3d10) {
    VecD3 expected({5.0, 4.0, 8.0});
    PointObjD3 res = LinearFindNearestNeighbor<double, 3>(TestData::Get().grid3d10, VecD3({5.1, 3.9, 7.6}));
    EXPECT_EQ(expected, res.point);
}
TEST(Search_LinearFindNN, grid4d10) {
    VecD4 expected({5.0, 4.0, 8.0, 6.0});
    PointObjD4 res = LinearFindNearestNeighbor<double, 4>(TestData::Get().grid4d10, VecD4({5.1, 3.9, 7.6, 6.3}));
    EXPECT_EQ(expected, res.point);
}

TEST(Search_LinearFindKNN, line2d10_k1) {
    std::vector<VecD2> expected = {{5.0, 5.0}};
    std::vector<VecD2> res = ObjsToVec(LinearFindKNearestNeighbors<double, 2>(TestData::Get().line2d10, VecD2(5.0), 1));
    EXPECT_EQ(expected, res);
}
TEST(Search_LinearFindKNN, line2d10_k2) {
    std::vector<VecD2> expected = {{5.0, 5.0}, {6.0, 6.0}};
    std::vector<VecD2> res = ObjsToVec(LinearFindKNearestNeighbors<double, 2>(TestData::Get().line2d10, VecD2(5.5), 2));
    std::sort(res.begin(), res.end(), [&](const VecD2& v1, const VecD2& v2) { return v1[0] < v2[0]; });
    EXPECT_EQ(expected, res);
}
TEST(Search_LinearFindKNN, line2d10_k3) {
    std::vector<VecD2> expected = {{4.0, 4.0}, {5.0, 5.0}, {6.0, 6.0}};
    std::vector<VecD2> res = ObjsToVec(LinearFindKNearestNeighbors<double, 2>(TestData::Get().line2d10, VecD2(5.0), 3));
    std::sort(res.begin(), res.end(), [&](const VecD2& v1, const VecD2& v2) { return v1[0] < v2[0]; });
    EXPECT_EQ(expected, res);
}
TEST(Search_LinearFindKNN, line2d10_k4) {
    std::vector<VecD2> expected = {{4.0, 4.0}, {5.0, 5.0}, {6.0, 6.0}, {7.0, 7.0}};
    std::vector<VecD2> res = ObjsToVec(LinearFindKNearestNeighbors<double, 2>(TestData::Get().line2d10, VecD2(5.5), 4));
    std::sort(res.begin(), res.end(), [&](const VecD2& v1, const VecD2& v2) { return v1[0] < v2[0]; });
    EXPECT_EQ(expected, res);
}