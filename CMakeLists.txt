cmake_minimum_required(VERSION 3.11...3.16)

project(DPG_AKNN
    VERSION 0.1
    DESCRIPTION "DPG aknn"
    LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE APP_SRC
"src/*.cpp"
"src/*.h"
)

set(APP_INCLUDE 
"src"
)

add_executable(aknn ${APP_SRC})

target_include_directories(aknn PUBLIC ${APP_INCLUDE})
