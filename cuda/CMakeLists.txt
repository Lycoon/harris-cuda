cmake_minimum_required (VERSION 3.14)
project (GPGPU CUDA CXX)

option(PROF "Reference build" NO)
include(FetchContent)

FetchContent_Declare(
  GoogleBenchmark
  URL https://github.com/google/benchmark/archive/v1.5.2.tar.gz
  )

FetchContent_Declare(
  CLI11
  URL https://github.com/CLIUtils/CLI11/archive/v1.9.1.tar.gz
  )

FetchContent_Declare(
  spdlog
  URL https://github.com/gabime/spdlog/archive/v1.9.2.tar.gz
  )


if (NOT GoogleBenchmark_POPULATED)
  FetchContent_Populate(GoogleBenchmark)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "From Gtest")
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "From Gtest")
  add_subdirectory(${googlebenchmark_SOURCE_DIR} ${googlebenchmark_BINARY_DIR})
endif()

FetchContent_MakeAvailable(spdlog)
FetchContent_MakeAvailable(CLI11)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} )
find_package(PNG REQUIRED)
# find_package(CUDA REQUIRED)
find_package(TBB REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -fsanitize=address")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")


add_library(harris_cu)
target_compile_features(harris_cu PUBLIC cxx_std_14)
target_link_libraries(harris_cu PRIVATE spdlog::spdlog)


target_sources(harris_cu PRIVATE src/include/harris.hh src/harris.cu)
set_target_properties(harris_cu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# add_library(render_cpu_ref SHARED IMPORTED)
# target_link_libraries(render_cpu_ref INTERFACE tbb)
# set_target_properties(render_cpu_ref PROPERTIES IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/librender_cpu_ref.so)

file(GLOB all_SRCS
    "src/include/*.hh"
    "src/main.cc"
    "src/png.cc"

    main.cc
)

add_executable(harris ${all_SRCS})
target_compile_features(harris PUBLIC cxx_std_17)
target_link_libraries(harris PRIVATE PNG::PNG CLI11::CLI11 spdlog::spdlog harris_cu)

# add_executable(bench src/bench.cc)
# target_compile_features(bench PUBLIC cxx_std_17)
# target_link_libraries(bench PRIVATE spdlog::spdlog benchmark harris_cu)
