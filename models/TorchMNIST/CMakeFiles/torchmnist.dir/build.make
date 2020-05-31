# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/elia/Università/PACS/PACS-Project/models/TorchMNIST

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/elia/Università/PACS/PACS-Project/models/TorchMNIST

# Include any dependencies generated for this target.
include CMakeFiles/torchmnist.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/torchmnist.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/torchmnist.dir/flags.make

CMakeFiles/torchmnist.dir/src/TorchMNIST.cpp.o: CMakeFiles/torchmnist.dir/flags.make
CMakeFiles/torchmnist.dir/src/TorchMNIST.cpp.o: src/TorchMNIST.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/elia/Università/PACS/PACS-Project/models/TorchMNIST/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/torchmnist.dir/src/TorchMNIST.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/torchmnist.dir/src/TorchMNIST.cpp.o -c /home/elia/Università/PACS/PACS-Project/models/TorchMNIST/src/TorchMNIST.cpp

CMakeFiles/torchmnist.dir/src/TorchMNIST.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/torchmnist.dir/src/TorchMNIST.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/elia/Università/PACS/PACS-Project/models/TorchMNIST/src/TorchMNIST.cpp > CMakeFiles/torchmnist.dir/src/TorchMNIST.cpp.i

CMakeFiles/torchmnist.dir/src/TorchMNIST.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/torchmnist.dir/src/TorchMNIST.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/elia/Università/PACS/PACS-Project/models/TorchMNIST/src/TorchMNIST.cpp -o CMakeFiles/torchmnist.dir/src/TorchMNIST.cpp.s

# Object files for target torchmnist
torchmnist_OBJECTS = \
"CMakeFiles/torchmnist.dir/src/TorchMNIST.cpp.o"

# External object files for target torchmnist
torchmnist_EXTERNAL_OBJECTS =

/home/elia/Università/PACS/PACS-Project/models/libtorchmnist.so: CMakeFiles/torchmnist.dir/src/TorchMNIST.cpp.o
/home/elia/Università/PACS/PACS-Project/models/libtorchmnist.so: CMakeFiles/torchmnist.dir/build.make
/home/elia/Università/PACS/PACS-Project/models/libtorchmnist.so: /home/elia/Università/PACS/PACS-Project/libs/pytorch/lib/libtorch.so
/home/elia/Università/PACS/PACS-Project/models/libtorchmnist.so: /home/elia/Università/PACS/PACS-Project/libs/pytorch/lib/libc10.so
/home/elia/Università/PACS/PACS-Project/models/libtorchmnist.so: /home/elia/Università/PACS/PACS-Project/libs/pytorch/lib/libc10.so
/home/elia/Università/PACS/PACS-Project/models/libtorchmnist.so: CMakeFiles/torchmnist.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/elia/Università/PACS/PACS-Project/models/TorchMNIST/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library /home/elia/Università/PACS/PACS-Project/models/libtorchmnist.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/torchmnist.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/torchmnist.dir/build: /home/elia/Università/PACS/PACS-Project/models/libtorchmnist.so

.PHONY : CMakeFiles/torchmnist.dir/build

CMakeFiles/torchmnist.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/torchmnist.dir/cmake_clean.cmake
.PHONY : CMakeFiles/torchmnist.dir/clean

CMakeFiles/torchmnist.dir/depend:
	cd /home/elia/Università/PACS/PACS-Project/models/TorchMNIST && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/elia/Università/PACS/PACS-Project/models/TorchMNIST /home/elia/Università/PACS/PACS-Project/models/TorchMNIST /home/elia/Università/PACS/PACS-Project/models/TorchMNIST /home/elia/Università/PACS/PACS-Project/models/TorchMNIST /home/elia/Università/PACS/PACS-Project/models/TorchMNIST/CMakeFiles/torchmnist.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/torchmnist.dir/depend
