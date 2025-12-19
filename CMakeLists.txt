cmake_minimum_required(VERSION 3.17)
include(CheckLanguage)
check_language(CUDA)
project(efaxx LANGUAGES CXX C CUDA)
set(CMAKE_CXX_STANDARD 20)
include(cmake/packages.cmake)



include_directories(
  /opt/amazon/efa/include
  /usr/local/cuda/include
)
link_directories(
  /opt/amazon/efa/lib
  /usr/local/cuda/lib64
  /opt/nccl/build/lib
)

add_subdirectory(src)
add_subdirectory(experiments)
add_subdirectory(tests)
