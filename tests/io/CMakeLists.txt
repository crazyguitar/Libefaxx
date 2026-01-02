set(test_target io_test)

add_executable(${test_target}
  io_test.cc
  ${PROJECT_SOURCE_DIR}/src/handle.cc
  ${PROJECT_SOURCE_DIR}/src/selector.cc
)
target_compile_options(${test_target} PRIVATE -Wall -O3 -g)
target_include_directories(${test_target} PRIVATE
  "${PROJECT_SOURCE_DIR}/src/include"
  "${HWLOC_INCLUDE_DIRS}"
)
target_link_libraries(${test_target} PRIVATE
  Threads::Threads
  CUDA::cudart
  CUDA::cuda_driver
  spdlog::spdlog
  "${HWLOC_LIBRARIES}"
  "${NVML_LIBRARIES}"
)
