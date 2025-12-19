set(test_target io_test)

add_executable(${test_target}
  io_test.cc
  ${PROJECT_SOURCE_DIR}/src/handle.cc
  ${PROJECT_SOURCE_DIR}/src/selector.cc
)
find_package(Threads REQUIRED)
target_compile_options(${test_target} PRIVATE -Wall -O3 -g)
target_include_directories(${test_target} PRIVATE
  "${PROJECT_SOURCE_DIR}/src/include"
)
target_link_libraries(${test_target} PRIVATE Threads::Threads spdlog::spdlog)
