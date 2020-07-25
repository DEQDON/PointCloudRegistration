cmake_minimum_required(VERSION 2.8.3)

project(pcAlignment)

find_package(PCL 1.8 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})

add_executable(pcAlignment src/align.cpp)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR})

target_link_libraries(pcAlignment ${PCL_LIBRARIES}) 