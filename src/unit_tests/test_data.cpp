#include "test_data.h"


std::unique_ptr<TestData> TestData::s_testData;
std::mutex TestData::s_testDataMutex;

const TestData& TestData::Get()
{
    if(!s_testData) {
        std::lock_guard<std::mutex> testDataLock(s_testDataMutex);
        if(!s_testData) {
            s_testData = std::unique_ptr<TestData>(new TestData());
        }
    }
    return *s_testData;
}


TestData::TestData() : gen(5)
{
    basicInts = {9, 2, 5, 10, 3, 1, 7, 8, 4, 6};

    cube2d = GenGrid<2>(1);
    cube3d = GenGrid<3>(1);
    cube4d = GenGrid<4>(1);

    line2d10 = GenLine<2>(10);
    line3d10 = GenLine<3>(10);
    line4d10 = GenLine<4>(10);
    
    grid2d10 = GenGrid<2>(10);
    grid3d10 = GenGrid<3>(10);
    grid4d10 = GenGrid<4>(10);

    nnTestCases2d = {
        NNTestCase<2>({&line2d10, VecD2(5.1), VecD2(5.0)}),
        NNTestCase<2>({&grid2d10, VecD2({5.1, 3.9}), VecD2({5.0, 4.0})}),
        NNTestCase<2>({&line2d10, VecD2(100.0), VecD2(10.0)}),
        NNTestCase<2>({&grid2d10, VecD2(100.0), VecD2(10.0)})
    };

    nnTestCases3d = {
        NNTestCase<3>({&line3d10, VecD3(5.1), VecD3(5.0)}),
        NNTestCase<3>({&grid3d10, VecD3({5.1, 3.9, 7.6}), VecD3({5.0, 4.0, 8.0})}),
        NNTestCase<3>({&line3d10, VecD3(100.0), VecD3(10.0)}),
        NNTestCase<3>({&grid3d10, VecD3(100.0), VecD3(10.0)})
    };

    nnTestCases4d = {
        NNTestCase<4>({&line4d10, VecD4(5.1), VecD4(5.0)}),
        NNTestCase<4>({&grid4d10, VecD4({5.1, 3.9, 7.6, 6.3}), VecD4({5.0, 4.0, 8.0, 6.0})}),
        NNTestCase<4>({&line4d10, VecD4(100.0), VecD4(10.0)}),
        NNTestCase<4>({&grid4d10, VecD4(100.0), VecD4(10.0)})
    };

    knnTestCases2d = {
        KNNTestCase<2>({&line2d10, VecD2(5.0), 1, { VecD2(5.0) }}),
        KNNTestCase<2>({&line2d10, VecD2(5.4), 2, { VecD2(5.0), VecD2(6.0) }}),
        KNNTestCase<2>({&line2d10, VecD2(5.1), 3, { VecD2(5.0), VecD2(6.0), VecD2(4.0) }}),
        KNNTestCase<2>({&line2d10, VecD2(5.6), 4, { VecD2(6.0), VecD2(5.0), VecD2(7.0), VecD2(4.0) }})
    };
    
    knnTestCases3d = {
        KNNTestCase<3>({&line3d10, VecD3(5.0), 1, { VecD3(5.0) }}),
        KNNTestCase<3>({&line3d10, VecD3(5.4), 2, { VecD3(5.0), VecD3(6.0) }}),
        KNNTestCase<3>({&line3d10, VecD3(5.1), 3, { VecD3(5.0), VecD3(6.0), VecD3(4.0) }}),
        KNNTestCase<3>({&line3d10, VecD3(5.6), 4, { VecD3(6.0), VecD3(5.0), VecD3(7.0), VecD3(4.0) }})
    };
    
    knnTestCases4d = {
        KNNTestCase<4>({&line4d10, VecD4(5.0), 1, { VecD4(5.0) }}),
        KNNTestCase<4>({&line4d10, VecD4(5.4), 2, { VecD4(5.0), VecD4(6.0) }}),
        KNNTestCase<4>({&line4d10, VecD4(5.1), 3, { VecD4(5.0), VecD4(6.0), VecD4(4.0) }}),
        KNNTestCase<4>({&line4d10, VecD4(5.6), 4, { VecD4(6.0), VecD4(5.0), VecD4(7.0), VecD4(4.0) }})
    };
}