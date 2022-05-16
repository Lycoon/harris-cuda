cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-g -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project(harris)

include_directories(
    src
)

file(GLOB all_SRCS
    "src/include/*.hh"
    "src/*.cc"

    main.cc
)

add_executable(harris ${all_SRCS})

find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})
target_link_libraries(harris ${PNG_LIBRARY})
