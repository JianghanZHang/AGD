cmake_minimum_required(VERSION 3.10)

project(grg)
set(PROJECT_NAME grg)
set(PROJECT_NAMESPACE JianghanZHang)
set(PROJECT_URL https://github.com/${PROJECT_NAMESPACE}/${PROJECT_NAME})



SET(PROJECT_USE_KEYWORD_LINK_LIBRARIES TRUE)

SET(CXX_DISABLE_WERROR True)

SET(CMAKE_VERBOSE_MAKEFILE True)

set(PROJECT_USE_CMAKE_EXPORT TRUE)
set(CUSTOM_HEADER_DIR ${PROJECT_NAME})


option(BUILD_PYTHON_INTERFACE "Build the python binding" ON)

option(BUILD_BENCHMARKS "Build the benchmarks" OFF)
option(BUILD_UNITTEST "Build the unit test" OFF)


# Check if the submodule cmake have been initialized
set(JRL_CMAKE_MODULES "${CMAKE_CURRENT_LIST_DIR}/cmake")
IF(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake/base.cmake")
  MESSAGE(FATAL_ERROR "\ncmake/base.cmake not found. Please run the following command first:\ngit submodule update --init\n")
ENDIF()


include(${CMAKE_CURRENT_LIST_DIR}/cmake/base.cmake)
# Project definition
COMPUTE_PROJECT_ARGS(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})

include(${CMAKE_CURRENT_LIST_DIR}/cmake/boost.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/apple.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/ide.cmake)
include(CMakeDependentOption)

# Project dependencies
find_package(console_bridge REQUIRED)
find_package(crocoddyl REQUIRED)
find_package(Boost REQUIRED COMPONENTS unit_test_framework)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# set(crocoddyl_INCLUDE_DIRS "/opt/openrobots/include")
# set(CROCODDYL_LIBRARY "/opt/openrobots/lib/")

message(STATUS "Crocoddyl include directory: ${crocoddyl_INCLUDE_DIRS}")
message(STATUS "Crocoddyl library: ${crocoddyl_LIBRARIES}")

# Assuming you know crocoddyl was compiled with multithreading support

add_project_dependency(example-robot-data 4.0.7 REQUIRED PKG_CONFIG_REQUIRES
"example-robot-data >= 4.0.7")
add_project_dependency(crocoddyl 2.0.0 REQUIRED)
add_project_dependency(eigenpy 2.7.10 REQUIRED)
add_project_dependency(OpenMP 3.0.0 REQUIRED)


if(BUILD_PYTHON_INTERFACE)
  string(REGEX REPLACE "-" "_" PY_NAME ${PROJECT_NAME})
  set(${PY_NAME}_INSTALL_DIR ${PYTHON_SITELIB}/${PY_NAME})
endif()

set(${PROJECT_NAME}_HEADERS
  include/${PROJECT_NAME}/grg.hpp
  include/${PROJECT_NAME}/grg_adahessian.hpp
  include/${PROJECT_NAME}/grg_nonlinear.hpp


  )

set(${PROJECT_NAME}_SOURCES
  src/grg.cpp
  src/grg_adahessian.cpp
  src/grg_nonlinear.cpp
  )

add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SOURCES} ${${PROJECT_NAME}_HEADERS})
target_include_directories(${PROJECT_NAME} PUBLIC $<INSTALL_INTERFACE:include>)


set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "\$ORIGIN/../lib")

if(BUILD_PYTHON_INTERFACE)
  add_subdirectory(bindings)
endif()

if(BUILD_UNITTEST)
  add_subdirectory(unittest)
endif()


target_link_libraries(${PROJECT_NAME} PUBLIC crocoddyl::crocoddyl)

# Benchmarks
if(BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

install(TARGETS ${PROJECT_NAME} EXPORT ${TARGETS_EXPORT_NAME} DESTINATION lib)
install(FILES package.xml DESTINATION share/${PROJECT_NAME})

