
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

template<int Dim>
void TestFindNNWithBasicSplitTree(const std::vector<NNTestCase<Dim>>& testCases, int leafSize)
{
    TestFindNN<Dim>(testCases, [&](const std::vector<PointObj<double, Dim>> objs, const Vec<double, Dim>& queryPoint){
        BBDTree<double, Dim> tree = BBDTree<double, Dim>::BuildBasicSplitTree(leafSize, objs);
        return FindNearestNeighbor(tree, queryPoint);
    });
}

template<int Dim>
void TestFindKNNWithBasicSplitTree(const std::vector<KNNTestCase<Dim>>& testCases, int leafSize, FixedPriQueue<DistObj<double, Dim>>& knnQueue)
{
    TestFindKNN<Dim>(testCases, [&](const std::vector<PointObj<double, Dim>> objs, const Vec<double, Dim>& queryPoint, int k){
        BBDTree<double, Dim> tree = BBDTree<double, Dim>::BuildBasicSplitTree(leafSize, objs);
        return FindKNearestNeighbors(tree, queryPoint, k, knnQueue);
    });
}

template<int Dim>
void TestFindNNWithMidpointSplitTree(const std::vector<NNTestCase<Dim>>& testCases, int leafSize)
{
    TestFindNN<Dim>(testCases, [&](const std::vector<PointObj<double, Dim>> objs, const Vec<double, Dim>& queryPoint){
        BBDTree<double, Dim> tree = BBDTree<double, Dim>::BuildMidpointSplitTree(leafSize, objs);
        return FindNearestNeighbor(tree, queryPoint);
    });
}

template<int Dim>
void TestFindKNNWithMidpointSplitTree(const std::vector<KNNTestCase<Dim>>& testCases, int leafSize, FixedPriQueue<DistObj<double, Dim>>& knnQueue)
{
    TestFindKNN<Dim>(testCases, [&](const std::vector<PointObj<double, Dim>> objs, const Vec<double, Dim>& queryPoint, int k){
        BBDTree<double, Dim> tree = BBDTree<double, Dim>::BuildMidpointSplitTree(leafSize, objs);
        return FindKNearestNeighbors(tree, queryPoint, k, knnQueue);
    });
}

TEST(TestLinearFindNN, dim2) {
    TestFindNN<2>(TestData::Get().nnTestCases2d, LinearFindNearestNeighbor<double, 2>);
}
TEST(TestLinearFindNN, dim3) {
    TestFindNN<3>(TestData::Get().nnTestCases3d, LinearFindNearestNeighbor<double, 3>);
}
TEST(TestLinearFindNN, dim4) {
    TestFindNN<4>(TestData::Get().nnTestCases4d, LinearFindNearestNeighbor<double, 4>);
}

TEST(TestFindNN, dim2_BasicSplit_leafSize1) {
    TestFindNNWithBasicSplitTree<2>(TestData::Get().nnTestCases2d, 1);
}
TEST(TestFindNN, dim3_BasicSplit_leafSize1) {
    TestFindNNWithBasicSplitTree<3>(TestData::Get().nnTestCases3d, 1);
}
TEST(TestFindNN, dim4_BasicSplit_leafSize1) {
    TestFindNNWithBasicSplitTree<4>(TestData::Get().nnTestCases4d, 1);
}

TEST(TestFindNN, dim2_MidpointSplit_leafSize1) {
    TestFindNNWithMidpointSplitTree<2>(TestData::Get().nnTestCases2d, 1);
}
TEST(TestFindNN, dim3_MidpointSplit_leafSize1) {
    TestFindNNWithMidpointSplitTree<3>(TestData::Get().nnTestCases3d, 1);
}
TEST(TestFindNN, dim4_MidpointSplit_leafSize1) {
    TestFindNNWithMidpointSplitTree<4>(TestData::Get().nnTestCases4d, 1);
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

TEST(TestFindKNN, dim2_BasicSplit_leafSize1_linearPriQueue) {
    TestFindKNNWithBasicSplitTree<2>(TestData::Get().knnTestCases2d, 1, LinearPriQueue<DistObj<double, 2>>());
}
TEST(TestFindKNN, dim3_BasicSplit_leafSize1_linearPriQueue) {
    TestFindKNNWithBasicSplitTree<3>(TestData::Get().knnTestCases3d, 1, LinearPriQueue<DistObj<double, 3>>());
}
TEST(TestFindKNN, dim4_BasicSplit_leafSize1_linearPriQueue) {
    TestFindKNNWithBasicSplitTree<4>(TestData::Get().knnTestCases4d, 1, LinearPriQueue<DistObj<double, 4>>());
}

TEST(TestFindKNN, dim2_MidpointSplit_leafSize1_linearPriQueue) {
    TestFindKNNWithMidpointSplitTree<2>(TestData::Get().knnTestCases2d, 1, LinearPriQueue<DistObj<double, 2>>());
}
TEST(TestFindKNN, dim3_MidpointSplit_leafSize1_linearPriQueue) {
    TestFindKNNWithMidpointSplitTree<3>(TestData::Get().knnTestCases3d, 1, LinearPriQueue<DistObj<double, 3>>());
}
TEST(TestFindKNN, dim4_MidpointSplit_leafSize1_linearPriQueue) {
    TestFindKNNWithMidpointSplitTree<4>(TestData::Get().knnTestCases4d, 1, LinearPriQueue<DistObj<double, 4>>());
}
