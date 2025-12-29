file(GLOB src *.cu *.cc)
set(target echo)
set(CMAKE_CUDA_ARCHITECTURES 80 90)

add_executable(${target} ${src})
set_property(TARGET ${target} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)
target_compile_options(${target} PRIVATE -Wall -O3 -g)
target_include_directories(${target} PRIVATE
  "${PROJECT_SOURCE_DIR}/src/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(${target} PRIVATE Threads::Threads CUDA::cudart efaxx::efaxx spdlog::spdlog)
