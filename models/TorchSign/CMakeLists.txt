cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(torchsign VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_PREFIX_PATH "${CMAKE_SRC_DIR}../../libs/libtorch/")

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(torchsign SHARED src/TorchSign.cpp)

target_link_libraries(torchsign "${TORCH_LIBRARIES}")

set_property(TARGET torchsign PROPERTY CXX_STANDARD 14)

set_target_properties( torchsign
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SRC_DIR}../"
)
















