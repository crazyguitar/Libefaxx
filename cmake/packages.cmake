include(FetchContent)

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.15.1
)

FetchContent_MakeAvailable(spdlog)

# Common dependencies
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(MPI REQUIRED)

pkg_check_modules(HWLOC REQUIRED hwloc)
set(ENV{PKG_CONFIG_PATH} "/opt/amazon/efa/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
pkg_check_modules(Fabric REQUIRED IMPORTED_TARGET libfabric)
pkg_check_modules(EFAVerbs IMPORTED_TARGET libefa)
find_library(NVML_LIBRARIES nvidia-ml PATHS /usr/local/cuda/lib64/stubs)
find_library(GDR_LIBRARY gdrapi PATHS /opt/gdrcopy/lib /usr/local/lib /usr/lib)
