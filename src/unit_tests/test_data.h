#ifndef UNIT_TEST_DATA_H
#define UNIT_TEST_DATA_H

#include <algorithm>
#include <memory>
#include <mutex>
#include <random>

#include <aknn/vec.h>

class TestData
{
public:
    std::vector<int> basicInts;

    std::vector<PointObj<double, 2>> cube2d;
    std::vector<PointObj<double, 3>> cube3d;
    std::vector<PointObj<double, 4>> cube4d;
    
    std::vector<PointObj<double, 2>> line2d10;
    std::vector<PointObj<double, 3>> line3d10;
    std::vector<PointObj<double, 4>> line4d10;

    std::vector<PointObj<double, 2>> grid2d10;
    std::vector<PointObj<double, 3>> grid3d10;
    std::vector<PointObj<double, 4>> grid4d10;

    static const TestData& Get();

private:
    std::mt19937 gen;

    static std::unique_ptr<TestData> s_testData;
    static std::mutex s_testDataMutex;

    TestData();
    
    template<int Dim>
    std::vector<PointObj<double, Dim>> GenLine(int max) {
        std::vector<PointObj<double, Dim>> line;
        for (int i = 0; i <= max; ++i) {
            line.push_back({Vec<double, Dim>((double)i)});
        }
        std::shuffle(line.begin(), line.end(), gen);
        return line;
    }

    template<int Dim>
    std::vector<PointObj<double, Dim>> GenGrid(int max) {
        std::vector<PointObj<double, Dim>> grid;
        int size = max + 1;
        int sizeI = 1;
        for (int d = 0; d < Dim; ++d) {
            sizeI *= size;
        }
        for (int i = 0; i < sizeI; ++i) {
            PointObj<double, Dim> obj;
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