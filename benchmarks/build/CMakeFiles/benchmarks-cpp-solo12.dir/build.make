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
CMAKE_SOURCE_DIR = /home/jianghan/Devel/workspace/src/mim_solvers/benchmarks

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jianghan/Devel/workspace/src/mim_solvers/benchmarks/build

# Utility rule file for benchmarks-cpp-solo12.

# Include the progress variables for this target.
include CMakeFiles/benchmarks-cpp-solo12.dir/progress.make

CMakeFiles/benchmarks-cpp-solo12:
	./solo12 ${INPUT}

benchmarks-cpp-solo12: CMakeFiles/benchmarks-cpp-solo12
benchmarks-cpp-solo12: CMakeFiles/benchmarks-cpp-solo12.dir/build.make

.PHONY : benchmarks-cpp-solo12

# Rule to build all files generated by this target.
CMakeFiles/benchmarks-cpp-solo12.dir/build: benchmarks-cpp-solo12

.PHONY : CMakeFiles/benchmarks-cpp-solo12.dir/build

CMakeFiles/benchmarks-cpp-solo12.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/benchmarks-cpp-solo12.dir/cmake_clean.cmake
.PHONY : CMakeFiles/benchmarks-cpp-solo12.dir/clean

CMakeFiles/benchmarks-cpp-solo12.dir/depend:
	cd /home/jianghan/Devel/workspace/src/mim_solvers/benchmarks/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jianghan/Devel/workspace/src/mim_solvers/benchmarks /home/jianghan/Devel/workspace/src/mim_solvers/benchmarks /home/jianghan/Devel/workspace/src/mim_solvers/benchmarks/build /home/jianghan/Devel/workspace/src/mim_solvers/benchmarks/build /home/jianghan/Devel/workspace/src/mim_solvers/benchmarks/build/CMakeFiles/benchmarks-cpp-solo12.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/benchmarks-cpp-solo12.dir/depend

