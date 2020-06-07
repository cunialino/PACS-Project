cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(torchxorpaper VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_PREFIX_PATH "${CMAKE_SRC_DIR}../../libs/pytorch/")

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(torchxorpaper SHARED src/TorchXor.cpp)

target_link_libraries(torchxorpaper "${TORCH_LIBRARIES}")

set_property(TARGET torchxorpaper PROPERTY CXX_STANDARD 14)

set_target_properties( torchxorpaper
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SRC_DIR}../"
)
















