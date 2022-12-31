
#include <gtest/gtest.h>
#include <aknn/search.h>

#include "test_data.h"

template<int Dim>
void TestFindNN(const std::vector<NNTestCase<Dim>>& testCases, FindNNFunc<Dim> findNN)
{
    int testCaseNum = 0;
    for (const NNTestCase<Dim>& testCase : testCases) {
        PointObj<double, Dim> res = findNN(*testCase.inputObjs, testCase.queryPoint);
        EXPECT_EQ(testCase.expectedRes, res.point) << "Test case " << testCaseNum << ": incorrect result";
        ++testCaseNum;
    }
}

template<int Dim>
void TestFindKNN(const std::vector<KNNTestCase<Dim>>& testCases, FindKNNFunc<Dim> findKNN)
{
    int testCaseNum = 0;
    for (const KNNTestCase<Dim>& testCase : testCases) {
        std::vector<Vec<double, Dim>> res = ObjsToVec(findKNN(*testCase.inputObjs, testCase.queryPoint, testCase.k));
        SortByDistanceToPoint(res, testCase.queryPoint);
        EXPECT_EQ(testCase.expectedRes.size(), res.size()) << "Test case " << testCaseNum << ": incorrect size";
        EXPECT_EQ(testCase.expectedRes, res) << "Test case " << testCaseNum << ": incorrect results";
        ++testCaseNum;
    }
}

/*
template<int Dim>
void TestFindNNWithBBDTree(const std::vector<NNTestCase<Dim>>& testCases, FindNNFunc<Dim> findNN)
{

}*/

TEST(TestLinearFindNN, dim2) {
    TestFindNN<2>(TestData::Get().nnTestCases2d, LinearFindNearestNeighbor<double, 2>);
}
TEST(TestLinearFindNN, dim3) {
    TestFindNN<3>(TestData::Get().nnTestCases3d, LinearFindNearestNeighbor<double, 3>);
}
TEST(TestLinearFindNN, dim4) {
    TestFindNN<4>(TestData::Get().nnTestCases4d, LinearFindNearestNeighbor<double, 4>);
}

TEST(TestFindNN, dim2_fairSplit_leafSize1) {
    TestFindNN<2>(TestData::Get().nnTestCases2d, [](const std::vector<PointObj<double, 2>> objs, const Vec<double, 2>& queryPoint){
        BBDTree<double, 2> tree = BBDTree<double, 2>::BuildFairSplitTree(1, objs);
        return FindNearestNeighbor(tree, queryPoint);
    });
}

TEST(TestLinearFindKNN, dim2) {
    TestFindKNN<2>(TestData::Get().knnTestCases2d, LinearFindKNearestNeighbors<double, 2>);
}
TEST(TestLinearFindKNN, dim3) {
    TestFindKNN<3>(TestData::Get().knnTestCases3d, LinearFindKNearestNeighbors<double, 3>);
}
TEST(TestLinearFindKNN, dim4) {
    TestFindKNN<4>(TestData::Get().knnTestCases4d, LinearFindKNearestNeighbors<double, 4>);
}