# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Produce verbose output by default.
VERBOSE = 1

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jianghan/Devel/workspace/src/GRG

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jianghan/Devel/workspace/src/GRG/build

# Utility rule file for benchmarks-cpp-ur5.

# Include the progress variables for this target.
include benchmarks/CMakeFiles/benchmarks-cpp-ur5.dir/progress.make

benchmarks/CMakeFiles/benchmarks-cpp-ur5:
	cd /home/jianghan/Devel/workspace/src/GRG/build/benchmarks && ./ur5 ${INPUT}

benchmarks-cpp-ur5: benchmarks/CMakeFiles/benchmarks-cpp-ur5
benchmarks-cpp-ur5: benchmarks/CMakeFiles/benchmarks-cpp-ur5.dir/build.make

.PHONY : benchmarks-cpp-ur5

# Rule to build all files generated by this target.
benchmarks/CMakeFiles/benchmarks-cpp-ur5.dir/build: benchmarks-cpp-ur5

.PHONY : benchmarks/CMakeFiles/benchmarks-cpp-ur5.dir/build

benchmarks/CMakeFiles/benchmarks-cpp-ur5.dir/clean:
	cd /home/jianghan/Devel/workspace/src/GRG/build/benchmarks && $(CMAKE_COMMAND) -P CMakeFiles/benchmarks-cpp-ur5.dir/cmake_clean.cmake
.PHONY : benchmarks/CMakeFiles/benchmarks-cpp-ur5.dir/clean

benchmarks/CMakeFiles/benchmarks-cpp-ur5.dir/depend:
	cd /home/jianghan/Devel/workspace/src/GRG/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jianghan/Devel/workspace/src/GRG /home/jianghan/Devel/workspace/src/GRG/benchmarks /home/jianghan/Devel/workspace/src/GRG/build /home/jianghan/Devel/workspace/src/GRG/build/benchmarks /home/jianghan/Devel/workspace/src/GRG/build/benchmarks/CMakeFiles/benchmarks-cpp-ur5.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : benchmarks/CMakeFiles/benchmarks-cpp-ur5.dir/depend
