cmake_minimum_required(VERSION 3.11...3.16)

project(DPG_AKNN
    VERSION 0.1
    DESCRIPTION "DPG aknn"
    LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(src/extern/googletest)

set(INCLUDE_DIR "src")

file(GLOB_RECURSE AKNN_SRC "src/aknn/*.cpp" "src/aknn/*.h")
add_library(aknn STATIC ${AKNN_SRC})
target_include_directories(aknn PUBLIC ${INCLUDE_DIR})

file(GLOB_RECURSE APP_SRC "src/app/*.cpp" "src/app/*.h")
add_executable(app ${APP_SRC})
target_include_directories(app PUBLIC ${INCLUDE_DIR} "src/extern/argumentum/include")
target_link_libraries(app PUBLIC aknn)

add_custom_command(TARGET app POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:app> "${CMAKE_CURRENT_SOURCE_DIR}/exe/")

file(GLOB_RECURSE UNIT_TESTS_SRC "src/unit_tests/*.cpp" "src/unit_tests/*.h")
add_executable(unit_tests ${UNIT_TESTS_SRC})
target_include_directories(unit_tests PUBLIC ${INCLUDE_DIR} "src/extern/googletest/googletest/include")
target_link_libraries(unit_tests PUBLIC aknn)
target_link_libraries(unit_tests PUBLIC gtest gtest_main)

find_package(Doxygen)
    if(Doxygen_FOUND)
add_subdirectory(doc)
  else()
message(STATUS "Doxygen not found, not building docs")
  endif()
