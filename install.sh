#!/bin/bash

cd libs

#Install pytorch 
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
rm libtorch-shared-with-deps-latest.zip

#Install OpenNN and make it as SHARED library
cd opennn 
cat opennn/CMakeLists.txt | awk -F"(" '{if($1 == "add_library") print $0" SHARED"; else print $0}' > opennn/CMakeLists_tmp.txt
mv opennn/CMakeLists_tmp.txt opennn/CMakeLists.txt
cmake .
make opennn
