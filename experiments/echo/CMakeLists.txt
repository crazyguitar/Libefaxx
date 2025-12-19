file(GLOB src *.cu *.cc)
set(test_target echo)
set(CMAKE_CUDA_ARCHITECTURES 80 90)

add_executable(${test_target} ${src})
set_property(TARGET ${test_target} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)
target_compile_options(${test_target} PRIVATE -Wall -O3 -g)
target_include_directories(${test_target} PRIVATE
  "${PROJECT_SOURCE_DIR}/src/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(${test_target} PRIVATE Threads::Threads CUDA::cudart efaxx::efaxx spdlog::spdlog)
