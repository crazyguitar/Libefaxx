file(GLOB src *.cu)
set(target queue)
set(CMAKE_CUDA_ARCHITECTURES 80 90)

add_executable(${target} ${src})
set_property(TARGET ${target} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
find_package(MPI REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
find_library(NVML_LIBRARIES nvidia-ml PATHS /usr/local/cuda/lib64/stubs)
find_library(GDR_LIBRARY gdrapi PATHS /opt/gdrcopy/lib /usr/local/lib /usr/lib)
pkg_check_modules(HWLOC REQUIRED hwloc)
target_compile_options(${target} PRIVATE -Wall -O3 -g)
target_include_directories(${target} PRIVATE
  "${PROJECT_SOURCE_DIR}/src/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${HWLOC_INCLUDE_DIRS}"
  "${MPI_INCLUDE_PATH}"
  "/opt/gdrcopy/include"
)
target_link_libraries(${target} PRIVATE
  Threads::Threads
  CUDA::cudart
  CUDA::cuda_driver
  efaxx::efaxx
  spdlog::spdlog
  "${MPI_LIBRARIES}"
  "${HWLOC_LIBRARIES}"
  "${NVML_LIBRARIES}"
  "${GDR_LIBRARY}"
)
