
#include <gtest/gtest.h>
#include <aknn/vec.h>

#include "test_data.h"

TEST(Box_GetBoundingBox, line2d10) {
    BoxD2 expected(VecD2(0.0), VecD2(10.0));
    BoxD2 result = BoxD2::GetBoundingBox(TestData::Get().line2d10);
    EXPECT_EQ(expected, result);
}
TEST(Box_GetBoundingBox, line3d10) {
    BoxD3 expected(VecD3(0.0), VecD3(10.0));
    BoxD3 result = BoxD3::GetBoundingBox(TestData::Get().line3d10);
    EXPECT_EQ(expected, result);
}
TEST(Box_GetBoundingBox, line4d10) {
    BoxD4 expected(VecD4(0.0), VecD4(10.0));
    BoxD4 result = BoxD4::GetBoundingBox(TestData::Get().line4d10);
    EXPECT_EQ(expected, result);
}

TEST(Box_GetBoundingBox, grid2d10) {
    BoxD2 expected(VecD2(0.0), VecD2(10.0));
    BoxD2 result = BoxD2::GetBoundingBox(TestData::Get().grid2d10);
    EXPECT_EQ(expected, result);
}
TEST(Box_GetBoundingBox, grid3d10) {
    BoxD3 expected(VecD3(0.0), VecD3(10.0));
    BoxD3 result = BoxD3::GetBoundingBox(TestData::Get().grid3d10);
    EXPECT_EQ(expected, result);
}
TEST(Box_GetBoundingBox, grid4d10) {
    BoxD4 expected(VecD4(0.0), VecD4(10.0));
    BoxD4 result = BoxD4::GetBoundingBox(TestData::Get().grid4d10);
    EXPECT_EQ(expected, result);
}