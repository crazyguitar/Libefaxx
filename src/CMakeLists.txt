file(GLOB src *.cc *.cu)
set(target efaxx)
set(alias efaxx::efaxx)
set(ENV{PKG_CONFIG_PATH} "/opt/amazon/efa/lib/pkgconfig")
set(CMAKE_CUDA_ARCHITECTURES 80 90)

add_library(${target} ${src})
add_library(${alias} ALIAS ${target})
set_property(TARGET ${target} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_library(NVML_LIBRARIES nvidia-ml PATHS /usr/local/cuda/lib64/stubs)
pkg_check_modules(Fabric IMPORTED_TARGET libfabric)
pkg_check_modules(HWLOC REQUIRED hwloc)
target_compile_options(${target} PRIVATE -Wall -O3 -g)
target_include_directories(${target} PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${HWLOC_INCLUDE_DIRS}"
)

target_link_libraries(${target} PRIVATE
  Threads::Threads
  CUDA::cudart
  CUDA::cuda_driver
  spdlog::spdlog
  PkgConfig::Fabric
  "${HWLOC_LIBRARIES}"
  "${NVML_LIBRARIES}"
)
