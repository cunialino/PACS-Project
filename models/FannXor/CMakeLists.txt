cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(fannxor VERSION 1.0.0 LANGUAGES CXX)

#set(CMAKE_PREFIX_PATH "${CMAKE_SRC_DIR}../../libs/opennn/opennn/")

#link_directories(${CMAKE_PREFIX_PATH})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include) # ${CMAKE_SRC_DIR}../../libs/opennn/opennn/)

add_library(fannxor SHARED src/FannXor.cpp)

target_link_libraries(fannxor fann)

set_property(TARGET fannxor PROPERTY CXX_STANDARD 14)

set_target_properties( fannxor
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SRC_DIR}../"
)

















