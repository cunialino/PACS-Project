# PACS-Project

Elia Cunial (elia.cunial@mail.polimi.it) Advnaced programming for scientific computing project.
The project is an advanced and flexible implementation of the work in [https://arxiv.org/abs/1708.02276](https://arxiv.org/abs/1708.02276).

## Requirements
Tools:
* make version 4.3
* cmake version 3.18.2
* g++ version 10.2.0
* Open MPI version 4.0.4
Utilities:
* wget
* unzip
* awk

## Install
To install everything you can run the install.sh script that will download and compile the Neural netowrks library used in the examples inside models folder. 


## Usage
you can run mpirun build/apps/program --help to list all the possible options that you can tweak from command line, the model you wish to run is selected at run time from command line among all the .so files in the models folder.

