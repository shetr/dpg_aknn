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
    basicInts = {10, 2, 5, 7, 3, 1, 9, 8, 4, 6};

    cube2d = GenGrid<2>(1);
    cube3d = GenGrid<3>(1);
    cube4d = GenGrid<4>(1);

    line2d10 = GenLine<2>(10);
    line3d10 = GenLine<3>(10);
    line4d10 = GenLine<4>(10);
    
    grid2d10 = GenGrid<2>(10);
    grid3d10 = GenGrid<3>(10);
    grid4d10 = GenGrid<4>(10);
}