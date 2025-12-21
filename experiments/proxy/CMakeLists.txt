file(GLOB src *.cu *.cc)
set(test_target proxy)
set(CMAKE_CUDA_ARCHITECTURES 80 90)

add_executable(${test_target} ${src})
set_property(TARGET ${test_target} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
find_package(MPI REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
find_library(NVML_LIBRARIES nvidia-ml PATHS /usr/local/cuda/lib64/stubs)
pkg_check_modules(HWLOC REQUIRED hwloc)
target_compile_options(${test_target} PRIVATE -Wall -O3 -g)
target_include_directories(${test_target} PRIVATE
  "${PROJECT_SOURCE_DIR}/src/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${HWLOC_INCLUDE_DIRS}"
  "${MPI_INCLUDE_PATH}"
)
target_link_libraries(${test_target} PRIVATE
  Threads::Threads
  CUDA::cudart
  CUDA::cuda_driver
  efaxx::efaxx
  spdlog::spdlog
  "${MPI_LIBRARIES}"
  "${HWLOC_LIBRARIES}"
  "${NVML_LIBRARIES}"
)
