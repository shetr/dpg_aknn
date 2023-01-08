#ifndef UNIT_TEST_DATA_H
#define UNIT_TEST_DATA_H

#include <algorithm>
#include <memory>
#include <mutex>
#include <random>

#include <aknn/vec.h>
#include <aknn/search.h>

template<int Dim>
struct NNTestCase
{
    const std::vector<PointObjD<Dim>>* inputObjs;
    VecD<Dim> queryPoint;
    VecD<Dim> expectedRes;
};

template<int Dim>
struct ANNTestCase
{
    const std::vector<PointObjD<Dim>>* inputObjs;
    VecD<Dim> queryPoint;
    double epsilon;
    VecD<Dim> nn;
};

template<int Dim>
struct KNNTestCase
{
    const std::vector<PointObjD<Dim>>* inputObjs;
    VecD<Dim> queryPoint;
    int k;
    std::vector<VecD<Dim>> expectedRes;
};

template<int Dim>
struct AKNNTestCase
{
    const std::vector<PointObjD<Dim>>* inputObjs;
    VecD<Dim> queryPoint;
    int k;
    double epsilon;
    VecD<Dim> kth_nn;
};

template<int Dim>
struct DataStructureConfig
{
    int leafSize;
    FixedPriQueue<DistObj<double, Dim>>* knnQueue;
};

class TestData
{
public:
    // Test points

    std::vector<int> basicInts;

    std::vector<PointObjD<2>> cube2d;
    std::vector<PointObjD<3>> cube3d;
    std::vector<PointObjD<4>> cube4d;
    
    std::vector<PointObjD<2>> line2d10;
    std::vector<PointObjD<3>> line3d10;
    std::vector<PointObjD<4>> line4d10;

    std::vector<PointObjD<2>> grid2d10;
    std::vector<PointObjD<3>> grid3d10;
    std::vector<PointObjD<4>> grid4d10;

    // Test cases

    std::vector<NNTestCase<2>> nnTestCases2d;
    std::vector<NNTestCase<3>> nnTestCases3d;
    std::vector<NNTestCase<4>> nnTestCases4d;
    
    std::vector<ANNTestCase<2>> annTestCases2d;
    std::vector<ANNTestCase<3>> annTestCases3d;
    std::vector<ANNTestCase<4>> annTestCases4d;
    
    std::vector<KNNTestCase<2>> knnTestCases2d;
    std::vector<KNNTestCase<3>> knnTestCases3d;
    std::vector<KNNTestCase<4>> knnTestCases4d;
    
    std::vector<AKNNTestCase<2>> aknnTestCases2d;
    std::vector<AKNNTestCase<3>> aknnTestCases3d;
    std::vector<AKNNTestCase<4>> aknnTestCases4d;

    // data structure configs

    std::vector<DataStructureConfig<2>> dataStructureConfigs2d;
    std::vector<DataStructureConfig<3>> dataStructureConfigs3d;
    std::vector<DataStructureConfig<4>> dataStructureConfigs4d;

    std::vector<int> leafSizes;

    std::vector<FixedPriQueue<DistObj<double, 2>>*> fixedQueues2d;
    std::vector<FixedPriQueue<DistObj<double, 3>>*> fixedQueues3d;
    std::vector<FixedPriQueue<DistObj<double, 4>>*> fixedQueues4d;

    static const TestData& Get();

private:
    std::mt19937 gen;

    static std::unique_ptr<TestData> s_testData;
    static std::mutex s_testDataMutex;

    TestData();
    
    template<int Dim>
    std::vector<PointObjD<Dim>> GenLine(int max) {
        std::vector<PointObjD<Dim>> line;
        for (int i = 0; i <= max; ++i) {
            line.push_back({VecD<Dim>((double)i)});
        }
        std::shuffle(line.begin(), line.end(), gen);
        return line;
    }

    template<int Dim>
    std::vector<PointObjD<Dim>> GenGrid(int max) {
        std::vector<PointObjD<Dim>> grid;
        int size = max + 1;
        int sizeI = 1;
        for (int d = 0; d < Dim; ++d) {
            sizeI *= size;
        }
        for (int i = 0; i < sizeI; ++i) {
            PointObjD<Dim> obj;
            int di = i;
            for (int d = 0; d < Dim; ++d) {
                obj.point[d] = di % size;
                di /= size;
            }
            grid.push_back(obj);
        }
        std::shuffle(grid.begin(), grid.end(), gen);
        return grid;
    }
};

#endif // UNIT_TEST_DATA_H