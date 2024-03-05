set(${PROJECT_NAME}_BENCHMARK
  ur5
  profile
  )


set(${PROJECT_NAME}_BENCHMARK_TOOLS
  timings.hpp
  timings.cpp
  )


# Attempt to find Pinocchio
find_package(pinocchio REQUIRED)

# Check if Pinocchio was found
if(pinocchio_FOUND)
  message(STATUS "Pinocchio found: ${pinocchio_INCLUDE_DIRS}")
else()
  message(FATAL_ERROR "Pinocchio not found")
endif()



add_library(${PROJECT_NAME}_BENCHMARK_TOOLS SHARED ${${PROJECT_NAME}_BENCHMARK_TOOLS})


foreach(BENCHMARK_NAME ${${PROJECT_NAME}_BENCHMARK})
  add_executable(${BENCHMARK_NAME} ${BENCHMARK_NAME}.cpp)
  target_link_libraries(${BENCHMARK_NAME} ${PROJECT_NAME}
                        example-robot-data::example-robot-data
                        ${PROJECT_NAME}_BENCHMARK_TOOLS)
  add_custom_target("benchmarks-cpp-${BENCHMARK_NAME}" ${BENCHMARK_NAME}
                                                       \${INPUT})
endforeach(BENCHMARK_NAME ${${PROJECT_NAME}_BENCHMARK})