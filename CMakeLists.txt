cmake_minimum_required(VERSION 3.10)

project(grg)
set(PROJECT_NAME grg)
set(PROJECT_NAMESPACE JianghanZHang)
set(PROJECT_URL https://github.com/${PROJECT_NAMESPACE}/${PROJECT_NAME})

include_directories("${PROJECT_SOURCE_DIR}/include")


SET(PROJECT_USE_KEYWORD_LINK_LIBRARIES TRUE)

SET(CXX_DISABLE_WERROR True)

SET(CMAKE_VERBOSE_MAKEFILE True)
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "/usr/lib/x86_64-linux-gnu")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

option(BUILD_PYTHON_INTERFACE "Build the python binding" ON)
option(BUILD_BENCHMARKS "Build the benchmarks" ON)

# Check if the submodule cmake have been initialized
set(JRL_CMAKE_MODULES "${CMAKE_CURRENT_LIST_DIR}/cmake")
IF(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake/base.cmake")
  MESSAGE(FATAL_ERROR "\ncmake/base.cmake not found. Please run the following command first:\ngit submodule update --init\n")
ENDIF()

# JRL-cmakemodule setup
include(${CMAKE_CURRENT_LIST_DIR}/cmake/base.cmake)


# Project dependencies
find_package(Boost REQUIRED COMPONENTS unit_test_framework)
find_package(crocoddyl REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
set(crocoddyl_INCLUDE_DIRS "/opt/openrobts/include")
message(STATUS "Crocoddyl include directory: ${crocoddyl_INCLUDE_DIRS}")
message(STATUS "Crocoddyl library: ${crocoddyl_LIBRARIES}")

# Assuming you know crocoddyl was compiled with multithreading support

add_project_dependency(example-robot-data 4.0.7 REQUIRED PKG_CONFIG_REQUIRES
"example-robot-data >= 4.0.7")

set(${PROJECT_NAME}_HEADERS
  include/${PROJECT_NAME}/grg.hpp
  include/${PROJECT_NAME}/grg_adahessian.hpp

  )

set(${PROJECT_NAME}_SOURCES
  src/grg.cpp
  src/grg_adahessian.cpp
  )

add_project_dependency(crocoddyl 2.0.0 REQUIRED)
add_project_dependency(eigenpy 2.7.10 REQUIRED)
add_project_dependency(OpenMP 3.0.0 REQUIRED)



if(BUILD_PYTHON_INTERFACE)
  string(REGEX REPLACE "-" "_" PY_NAME ${PROJECT_NAME})
  set(${PY_NAME}_INSTALL_DIR ${PYTHON_SITELIB}/${PY_NAME})
endif()

# set(${PROJECT_NAME}_HEADERS
#   include/${PROJECT_NAME}/grg.hpp
#   )

# set(${PROJECT_NAME}_SOURCES
#   src/grg.cpp
#   )

link_directories(/usr/lib/x86_64-linux-gnu)


if(BUILD_PYTHON_INTERFACE)
  add_subdirectory(bindings)
endif()

add_subdirectory(unittest)


add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SOURCES} ${${PROJECT_NAME}_HEADERS})
target_include_directories(${PROJECT_NAME} PUBLIC $<INSTALL_INTERFACE:include>)

target_link_libraries(${PROJECT_NAME} PUBLIC crocoddyl::crocoddyl)

# Benchmarks
if(BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
