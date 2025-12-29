file(GLOB src *.cu *.cc)
set(target efa)

add_executable(${target} ${src})
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
find_library(NVML_LIBRARIES nvidia-ml PATHS /usr/local/cuda/lib64/stubs)
pkg_check_modules(HWLOC REQUIRED hwloc)
pkg_check_modules(Fabric IMPORTED_TARGET libfabric)
target_compile_options(${target} PRIVATE -Wall -O3 -g)
target_include_directories(${target} PRIVATE
  "${PROJECT_SOURCE_DIR}/src/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${HWLOC_INCLUDE_DIRS}"
)
target_link_libraries(${target} PRIVATE
  Threads::Threads
  efaxx::efaxx
  spdlog::spdlog
  PkgConfig::Fabric
  "${HWLOC_LIBRARIES}"
  "${NVML_LIBRARIES}"
)
