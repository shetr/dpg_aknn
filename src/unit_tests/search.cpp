
#include <gtest/gtest.h>
#include <aknn/search.h>

#include "test_data.h"

template<int Dim>
using FindNNFunc = std::function<PointObj<double, Dim>(const std::vector<PointObj<double, Dim>>& objs, const Vec<double, Dim>& queryPoint)>;

template<int Dim>
using FindANNFunc = std::function<PointObj<double, Dim>(const std::vector<PointObj<double, Dim>>& objs, const Vec<double, Dim>& queryPoint, double epsilon)>;

template<int Dim>
using FindKNNFunc = std::function<std::vector<PointObj<double, Dim>>(const std::vector<PointObj<double, Dim>>& objs, const Vec<double, Dim>& queryPoint, int k)>;

template<int Dim>
using FindAKNNFunc = std::function<std::vector<PointObj<double, Dim>>(const std::vector<PointObj<double, Dim>>& objs, const Vec<double, Dim>& queryPoint, int k, double epsilon)>;

template<int Dim>
using BuildBBDTreeFunc = std::function<BBDTree<double, Dim>(int leafSize, const std::vector<PointObj<double, Dim>>& objs)>;

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
void TestFindANN(const std::vector<ANNTestCase<Dim>>& testCases, FindANNFunc<Dim> findANN)
{
    int testCaseNum = 0;
    for (const ANNTestCase<Dim>& testCase : testCases) {
        PointObj<double, Dim> res = findANN(*testCase.inputObjs, testCase.queryPoint, testCase.epsilon);
        double nnDistEpsilon = (1 + testCase.epsilon) * testCase.queryPoint.DistSquared(testCase.nn);
        double resDist = testCase.queryPoint.DistSquared(res.point);
        EXPECT_LE(resDist, nnDistEpsilon) << "Test case " << testCaseNum << ": incorrect results";
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
void TestFindAKNN(const std::vector<AKNNTestCase<Dim>>& testCases, FindAKNNFunc<Dim> findAKNN)
{
    int testCaseNum = 0;
    for (const AKNNTestCase<Dim>& testCase : testCases) {
        std::vector<Vec<double, Dim>> res = ObjsToVec(findAKNN(*testCase.inputObjs, testCase.queryPoint, testCase.k, testCase.epsilon));
        EXPECT_EQ(testCase.k, res.size()) << "Test case " << testCaseNum << ": incorrect size";
        double kth_nnDistEpsilon = (1 + testCase.epsilon) * testCase.queryPoint.DistSquared(testCase.kth_nn);
        for (const Vec<double, Dim>& v : res) {
            double vDist = testCase.queryPoint.DistSquared(v);
            EXPECT_LE(vDist, kth_nnDistEpsilon) << "Test case " << testCaseNum << ": incorrect results";
        }
        ++testCaseNum;
    }
}

template<int Dim>
void TestFindNNWithBBDTree(const std::vector<NNTestCase<Dim>>& testCases, BuildBBDTreeFunc<Dim> buildFunc, int leafSize)
{
    TestFindNN<Dim>(testCases, [&](const std::vector<PointObj<double, Dim>>& objs, const Vec<double, Dim>& queryPoint){
        BBDTree<double, Dim> tree = buildFunc(leafSize, objs);
        return FindNearestNeighbor(tree, queryPoint);
    });
}

template<int Dim>
void TestFindANNWithBBDTree(const std::vector<ANNTestCase<Dim>>& testCases, BuildBBDTreeFunc<Dim> buildFunc, int leafSize)
{
    TestFindANN<Dim>(testCases, [&](const std::vector<PointObj<double, Dim>>& objs, const Vec<double, Dim>& queryPoint, double epsilon){
        BBDTree<double, Dim> tree = buildFunc(leafSize, objs);
        return FindAproximateNearestNeighbor(tree, queryPoint, epsilon);
    });
}

template<int Dim>
void TestFindKNNWithBBDTree(const std::vector<KNNTestCase<Dim>>& testCases, BuildBBDTreeFunc<Dim> buildFunc, int leafSize, FixedPriQueue<DistObj<double, Dim>>& knnQueue)
{
    TestFindKNN<Dim>(testCases, [&](const std::vector<PointObj<double, Dim>>& objs, const Vec<double, Dim>& queryPoint, int k){
        BBDTree<double, Dim> tree = buildFunc(leafSize, objs);
        return FindKNearestNeighbors(tree, queryPoint, k, knnQueue);
    });
}

template<int Dim>
void TestFindAKNNWithBBDTree(const std::vector<AKNNTestCase<Dim>>& testCases, BuildBBDTreeFunc<Dim> buildFunc, int leafSize, FixedPriQueue<DistObj<double, Dim>>& aknnQueue)
{
    TestFindAKNN<Dim>(testCases, [&](const std::vector<PointObj<double, Dim>>& objs, const Vec<double, Dim>& queryPoint, int k, double epsilon){
        BBDTree<double, Dim> tree = buildFunc(leafSize, objs);
        return FindKAproximateNearestNeighbors(tree, queryPoint, k, epsilon, aknnQueue);
    });
}

template<int Dim>
void RandomTestsFindAKNNWithBBDTree()
{
    int datasetsCount = 4;
    int queryCount = 10;
    int datasetSize = 1000;
    std::vector<int> leafSizes = {1, 2, 10, 100};
    std::vector<int> ks = {1, 2, 5, 10, 25, 100};
    std::vector<double> epsilons = {0, 1, 2, 3, 5, 8, 10};
    std::vector<std::unique_ptr<FixedPriQueue<DistObj<double, Dim>>>> fixedQueues;
    fixedQueues.push_back(std::unique_ptr<FixedPriQueue<DistObj<double, Dim>>>(new LinearPriQueue<DistObj<double, Dim>>()));
    fixedQueues.push_back(std::unique_ptr<FixedPriQueue<DistObj<double, Dim>>>(new HeapPriQueue<DistObj<double, Dim>>()));
    fixedQueues.push_back(std::unique_ptr<FixedPriQueue<DistObj<double, Dim>>>(new StdPriQueue<DistObj<double, Dim>>()));

    std::vector<PointObjD<Dim>> dataset;
    dataset.resize(datasetSize, PointObjD<Dim>());
    for (int datasetNum = 0; datasetNum < datasetsCount; ++datasetNum)
    {
        for (int i = 0; i < datasetSize; ++i) {
            dataset[i] = PointObjD<Dim>({TestData::Get().GenRandVec<Dim>()});
        }

        for (int leafSize : leafSizes)
        {
            BBDTree<double, Dim> tree = BBDTree<double, Dim>::BuildMidpointSplitTree(leafSize, dataset);
            for (int query = 0; query < queryCount; ++query) {
                VecD<Dim> queryPoint = TestData::Get().GenRandVec<Dim>();
                for (int k : ks) {
                    std::vector<Vec<double, Dim>> knn = ObjsToVec(LinearFindKNearestNeighbors<double, Dim>(dataset, queryPoint, k));
                    SortByDistanceToPoint(knn, queryPoint);
                    double lastKnnDist = queryPoint.DistSquared(knn.back());
                    for (double epsilon : epsilons) {
                        double lastKnnDistEpsilon = (1 + epsilon) * lastKnnDist;
                        for (int queue = 0; queue < (int)fixedQueues.size(); ++queue) {
                            std::vector<Vec<double, Dim>> aknn = ObjsToVec(FindKAproximateNearestNeighbors(tree, queryPoint, k, epsilon, *fixedQueues[queue]));
                            EXPECT_EQ(k, aknn.size()) << "Incorrect size: d" << datasetNum << ", l" << leafSize << ", p" << queryPoint << ", k" << k << ", e" << epsilon << ", q" << queue;
                            for (int v = 0; v < (int)aknn.size(); ++v) {
                                double vDist = queryPoint.DistSquared(aknn[v]);
                                EXPECT_LE(vDist, lastKnnDistEpsilon) << "Incorrect result: d" << datasetNum << ", l" << leafSize << ", p" << queryPoint << ", k" << k << ", e" << epsilon << ", q" << queue << ", v" << v;
                            }
                        }
                    }
                }
            }
        }
        

    }
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
    TestFindNNWithBBDTree<2>(TestData::Get().nnTestCases2d, BBDTree<double, 2>::BuildBasicSplitTree, 1);
}
TEST(TestFindNN, dim3_BasicSplit_leafSize1) {
    TestFindNNWithBBDTree<3>(TestData::Get().nnTestCases3d, BBDTree<double, 3>::BuildBasicSplitTree, 1);
}
TEST(TestFindNN, dim4_BasicSplit_leafSize1) {
    TestFindNNWithBBDTree<4>(TestData::Get().nnTestCases4d, BBDTree<double, 4>::BuildBasicSplitTree, 1);
}

TEST(TestFindNN, dim2_MidpointSplit_leafSize1) {
    TestFindNNWithBBDTree<2>(TestData::Get().nnTestCases2d, BBDTree<double, 2>::BuildMidpointSplitTree, 1);
}
TEST(TestFindNN, dim3_MidpointSplit_leafSize1) {
    TestFindNNWithBBDTree<3>(TestData::Get().nnTestCases3d, BBDTree<double, 3>::BuildMidpointSplitTree, 1);
}
TEST(TestFindNN, dim4_MidpointSplit_leafSize1) {
    TestFindNNWithBBDTree<4>(TestData::Get().nnTestCases4d, BBDTree<double, 4>::BuildMidpointSplitTree, 1);
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
    TestFindKNNWithBBDTree<2>(TestData::Get().knnTestCases2d, BBDTree<double, 2>::BuildBasicSplitTree, 1, LinearPriQueue<DistObj<double, 2>>());
}
TEST(TestFindKNN, dim3_BasicSplit_leafSize1_linearPriQueue) {
    TestFindKNNWithBBDTree<3>(TestData::Get().knnTestCases3d, BBDTree<double, 3>::BuildBasicSplitTree, 1, LinearPriQueue<DistObj<double, 3>>());
}
TEST(TestFindKNN, dim4_BasicSplit_leafSize1_linearPriQueue) {
    TestFindKNNWithBBDTree<4>(TestData::Get().knnTestCases4d, BBDTree<double, 4>::BuildBasicSplitTree, 1, LinearPriQueue<DistObj<double, 4>>());
}

TEST(TestFindKNN, dim2_MidpointSplit_leafSize1_linearPriQueue) {
    TestFindKNNWithBBDTree<2>(TestData::Get().knnTestCases2d, BBDTree<double, 2>::BuildMidpointSplitTree, 1, LinearPriQueue<DistObj<double, 2>>());
}
TEST(TestFindKNN, dim3_MidpointSplit_leafSize1_linearPriQueue) {
    TestFindKNNWithBBDTree<3>(TestData::Get().knnTestCases3d, BBDTree<double, 3>::BuildMidpointSplitTree, 1, LinearPriQueue<DistObj<double, 3>>());
}
TEST(TestFindKNN, dim4_MidpointSplit_leafSize1_linearPriQueue) {
    TestFindKNNWithBBDTree<4>(TestData::Get().knnTestCases4d, BBDTree<double, 4>::BuildMidpointSplitTree, 1, LinearPriQueue<DistObj<double, 4>>());
}


TEST(TestFindANN, dim2_MidpointSplit) {
    for(int leafSize : TestData::Get().leafSizes)
        TestFindANNWithBBDTree<2>(TestData::Get().annTestCases2d, BBDTree<double, 2>::BuildMidpointSplitTree, leafSize);
}
TEST(TestFindANN, dim3_MidpointSplit) {
    for(int leafSize : TestData::Get().leafSizes)
        TestFindANNWithBBDTree<3>(TestData::Get().annTestCases3d, BBDTree<double, 3>::BuildMidpointSplitTree, leafSize);
}
TEST(TestFindANN, dim4_MidpointSplit) {
    for(int leafSize : TestData::Get().leafSizes)
        TestFindANNWithBBDTree<4>(TestData::Get().annTestCases4d, BBDTree<double, 4>::BuildMidpointSplitTree, leafSize);
}

TEST(TestFindAKNN, dim2_MidpointSplit) {
    for (const DataStructureConfig<2>& cfg : TestData::Get().dataStructureConfigs2d)
        TestFindAKNNWithBBDTree<2>(TestData::Get().aknnTestCases2d, BBDTree<double, 2>::BuildMidpointSplitTree, cfg.leafSize, *cfg.knnQueue);
}
TEST(TestFindAKNN, dim3_MidpointSplit) {
    for (const DataStructureConfig<3>& cfg : TestData::Get().dataStructureConfigs3d)
        TestFindAKNNWithBBDTree<3>(TestData::Get().aknnTestCases3d, BBDTree<double, 3>::BuildMidpointSplitTree, cfg.leafSize, *cfg.knnQueue);
}
TEST(TestFindAKNN, dim4_MidpointSplit) {
    for (const DataStructureConfig<4>& cfg : TestData::Get().dataStructureConfigs4d)
        TestFindAKNNWithBBDTree<4>(TestData::Get().aknnTestCases4d, BBDTree<double, 4>::BuildMidpointSplitTree, cfg.leafSize, *cfg.knnQueue);
}

TEST(RandomTestFindAKNN, dim2) {
    RandomTestsFindAKNNWithBBDTree<2>();
}
TEST(RandomTestFindAKNN, dim3) {
    RandomTestsFindAKNNWithBBDTree<3>();
}
TEST(RandomTestFindAKNN, dim4) {
    RandomTestsFindAKNNWithBBDTree<4>();
}
