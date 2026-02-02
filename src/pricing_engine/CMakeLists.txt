cmake_minimum_required(VERSION 3.14)
project(PricingEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(pricing_engine main.cpp)

target_link_libraries(pricing_engine PRIVATE Eigen3::Eigen)
target_compile_options(pricing_engine PRIVATE -O3 -march=native -Wall -Wextra)
