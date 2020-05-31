cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(opennnxor VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_PREFIX_PATH "${CMAKE_SRC_DIR}../../libs/opennn/opennn/")

link_directories(${CMAKE_PREFIX_PATH})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_SRC_DIR}../../libs/opennn/opennn/)

add_library(opennnxor SHARED src/OpenNNXOR.cpp)

target_link_libraries(opennnxor opennn)

set_property(TARGET opennnxor PROPERTY CXX_STANDARD 14)

set_target_properties( opennnxor
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SRC_DIR}../"
)

















