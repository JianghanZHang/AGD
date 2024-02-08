cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME grg)

project(${PROJECT_NAME})

SET(PROJECT_USE_KEYWORD_LINK_LIBRARIES TRUE)

SET(CXX_DISABLE_WERROR True)

SET(CMAKE_VERBOSE_MAKEFILE True)
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "/usr/lib/x86_64-linux-gnu")

option(BUILD_PYTHON_INTERFACE "Build the python binding" ON)
option(BUILD_BENCHMARKS "Build the benchmarks" OFF)

# Check if the submodule cmake have been initialized
set(JRL_CMAKE_MODULES "${CMAKE_CURRENT_LIST_DIR}/cmake")
IF(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake/base.cmake")
  MESSAGE(FATAL_ERROR "\ncmake/base.cmake not found. Please run the following command first:\ngit submodule update --init\n")
ENDIF()

# JRL-cmakemodule setup
include(${CMAKE_CURRENT_LIST_DIR}/cmake/base.cmake)


# Project dependencies
find_package(Boost REQUIRED COMPONENTS unit_test_framework)

add_project_dependency(example-robot-data 4.0.7 REQUIRED PKG_CONFIG_REQUIRES
"example-robot-data >= 4.0.7")

add_project_dependency(crocoddyl 2.0.0 REQUIRED)
add_project_dependency(eigenpy 2.7.10 REQUIRED)

if(BUILD_PYTHON_INTERFACE)
  string(REGEX REPLACE "-" "_" PY_NAME ${PROJECT_NAME})
  set(${PY_NAME}_INSTALL_DIR ${PYTHON_SITELIB}/${PY_NAME})
endif()

set(${PROJECT_NAME}_HEADERS
  include/${PROJECT_NAME}/grg.hpp
  )

set(${PROJECT_NAME}_SOURCES
  src/grg.cpp
  )

link_directories(/usr/lib/x86_64-linux-gnu)

if(BUILD_PYTHON_INTERFACE)
  add_subdirectory(bindings)
endif()

# Benchmarks
add_subdirectory(benchmarks)
