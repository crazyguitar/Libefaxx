file(GLOB test_src *.cu)
set(test_target queue_test)
set(CMAKE_CUDA_ARCHITECTURES 80 90)

add_executable(${test_target} ${test_src})
set_property(TARGET ${test_target} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_library(GDR_LIBRARY gdrapi PATHS /opt/gdrcopy/lib)
target_compile_options(${test_target} PRIVATE -Wall -O3 -g)
target_include_directories(${test_target} PRIVATE
  "${PROJECT_SOURCE_DIR}/src/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  /opt/gdrcopy/include
)
target_link_libraries(${test_target} PRIVATE
  Threads::Threads
  CUDA::cudart
  CUDA::cuda_driver
  efaxx::efaxx
  spdlog::spdlog
  ${GDR_LIBRARY}
)
