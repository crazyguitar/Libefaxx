file(GLOB src *.cu *.cc)
set(target efa)

add_executable(${target} ${src})
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
