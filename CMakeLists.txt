cmake_minimum_required(VERSION 3.5)
project(SGD)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp SGD.h LR.h utils.h)
add_executable(SGD ${SOURCE_FILES})