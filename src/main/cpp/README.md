<!--
{% comment %}
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements.  See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to you under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
{% endcomment %}
-->


By default, SystemML implements all its matrix operations in Java.
This simplifies deployment especially in a distributed environment.

In some cases (such as deep learning), the user might want to use native BLAS
rather than SystemML's internal Java library for performing single-node
operations such matrix multiplication, convolution, etc.
To enable that, the user has to build `systemml.cpp` and other C++ files (available in this folder)
as a shared library and make it available through `LD_LIBRARY_PATH` (on linux)
or `PATH` (on Windows). If the shared library is not accessible, SystemML
falls back to its internal Java library.

In the below section, we describe the steps to build `systemml.cpp` using native BLAS.

# Step 1: Install BLAS

## Option 1: Install Intel MKL (recommended)

1. Download and install the [community version of Intel MKL](https://software.intel.com/sites/campaigns/nest/).
Intel requires you to first register your email address and then sends the download link to your email address
with license key.

	* Linux users will have to extract the downloaded `.tgz` file and execute `install.sh`.
	* Windows users will have to execute the downloaded `.exe` file and follow the guided setup.

2. Set `MKLROOT` enviroment variable to point to the installed location.

	* Linux: By default, Intel MKL will be installed in `/opt/intel/mkl/`.

		```bash
		export MKLROOT=/opt/intel/mkl/
		```

	* Windows: By default, Intel MKL will be installed in `C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2017\windows\mkl`.

To add a new enviroment variable on Windows, the user has to right-click on `Computer` and then click `Properties > Advanced system settings > Environment Variables > New`.

## Option 2: Install OpenBLAS  

1. RedHat / CentOS:
	```bash
	sudo yum install openblas
	```

2. Ubuntu:
	```bash
	sudo apt-get install openblas
	```

3. Windows:
Download [pre-built binaries](https://sourceforge.net/projects/openblas/) or install from [the source](https://github.com/xianyi/OpenBLAS).

4. Set `OPENBLASROOT` enviroment variable to point to the installed location.

# Compiling systemml.cpp

## With CMake

### Intro to CMake
If you are familiar with cmake, skip this section.<br/>
In a regular project with a Makefile, the compiled object files are placed in the same directory as the source.
Sometimes we don't want to pollute the source tree. We might also want to have different binaries for different configurations. For instance, if we want to link a binary with separate libraries.
CMake supports out of source tree builds. As an illustration, you can create a directory called "BUILD" and invoke cmake like so : `cmake <path/to/source>`. The makefile and other config files are placed in this "BUILD" directory. You can now say `make` and the compiled objects and binary files are created in this directory. You can then create another "BUILD2" directory and repeat the process.
You can pass options to cmake as well. In this instance, it might be to specify whether to build with Intel MKL or OpenBLAS. This can be done from the command line with a "-D" appended to it, but more interestingly, it can also be done form a n-curses GUI which is invoked as `ccmake <path/to/source>`. (You may need to install this separately).
Also, the C, C++ compilers and their flags are picked up by cmake when set in standard environment variables. These are respectively `CC`, `CXX`, `CFLAGS` & `CXFLAGS`. As an example, they may be specified as:
```
CXX=gcc-6 cmake ..
```
For this project, I typically make a directory in the `cpp` folder (this folder) and name it the config I use. For instance, `INTEL` for Intel MKL and `OPENBLAS` for OpenBLAS.


### On MacOS
The version of clang that ships with Mac does not come with OpenMP. `brew install` either `llvm` or `g++`. The instructions that follow are for llvm:

1. Intel MKL - CMake should detect the MKL installation path, otherwise it can specified by the environment variable `MKLROOT`. To use (with gcc-6):
```
mkdir INTEL && cd INTEL
CXX=/usr/local/opt/llvm/bin/clang++ CC=/usr/local/opt/llvm/bin/clang LDFLAGS=-L/usr/local/opt/llvm/lib CPPFLAGS=-I/usr/local/opt/llvm/include cmake  -DUSE_INTEL_MKL=ON ..
make
```
2. OpenBLAS - CMake should be able to detect the path of OpenBLAS. If it can't, set the `OpenBLAS` environment variable. If using `brew` to install OpenBLAS, set the `OpenBLAS_HOME` environment variable to `/usr/local/opt/openblas/`. To use (with gcc-6):
```
export OpenBLAS_HOME=/usr/local/opt/openblas/
mkdir OPENBLAS && cd OPENBLAS
CXX=/usr/local/opt/llvm/bin/clang++ CC=/usr/local/opt/llvm/bin/clang LDFLAGS=-L/usr/local/opt/llvm/lib CPPFLAGS=-I/usr/local/opt/llvm/include cmake  -DUSE_OPEN_BLAS=ON ..
make
```

### On Linux
With the appropriate prerequisites (C++ compiler with OpenMP, OpenBLAS or IntelMKL)
1. Intel MKL
```
mkdir INTEL && cd INTEL
cmake -DUSE_INTEL_MKL=ON ..
make
```
2. OpenBLAS - If CMake cannot detect your OpenBLAS installation, set the `OpenBLAS_HOME` environment variable to the OpenBLAS Home.
```
mkdir OPENBLAS && cd OPENBLAS
cmake -DUSE_OPEN_BLAS=ON ..
make
```

After the shared library object is generated, it may need to be renamed to what the Java code expects.


### On Windows
- Install MKL or Download the OpenBlas Binary
- Install Visual Studio Community Edition (tested on VS 2017)
- Use the CMake GUI, select the source directory, the output directory
- Press the `configure` button, set the `generator` and `use default native compilers` option
- By default, `USE_INTEL_MKL` is selected, if you wanted to use OpenBLAS, unselect the `USE_INTEL_MKL`, select the `USE_OPEN_BLAS`.
- You might run into errors a couple of times, select the appropriate library and include files/directories (For MKL or OpenBLAS) a couple of times, and all the errors should go away. 
- Then press generate. This will generate Visual Studio project files, which you can open in VS2017 to compile the libraries.

## Manually

### 64-bit Linux

```bash
g++ -o libpreload_systemml-linux-x86_64.so preload_systemml.cpp  -I$JAVA_HOME/include -I$JAVA_HOME/include/linux -lm -ldl -m64 -fopenmp -O3 -shared -fPIC
```

1. Intel MKL:

	```bash
	# -Wl,--as-needed
	g++ -o libsystemml_mkl-linux-x86_64.so *.cpp  -I$JAVA_HOME/include -I$MKLROOT/include -I$JAVA_HOME/include/linux -DUSE_MKL_THREADING_GNU -lmkl_rt -lpthread -lm -ldl -L$MKLROOT/lib/intel64 -m64 -fopenmp -O3 -shared -fPIC
	```

2. OpenBLAS:

	```bash
	# -I$OPENBLASROOT/include -L$OPENBLASROOT/lib
	g++ -o libsystemml_openblas-linux-x86_64.so *.cpp  -I$JAVA_HOME/include  -I$JAVA_HOME/include/linux -lopenblas -lpthread -lm -ldl -DUSE_OPEN_BLAS -fopenmp -O3 -shared -fPIC
	```

### 64-bit Windows

Comment `#include "config.h"` in systemml.cpp

1. Intel MKL:

	```bash
	"C:\\Program Files (x86)\\Microsoft Visual Studio 12.0\\VC\\vcvarsall.bat" amd64
	"%MKLROOT%"\bin\mklvars.bat intel64
	set JAVA_HOME=C:\Program Files\Java\jdk1.8.0_25
	cl *.cpp -I. -I"%MKLROOT%"\include -I"%JAVA_HOME%"\include -I"%JAVA_HOME%"\include\win32 -Fesystemml_mkl-windows-x86_64.dll -MD -LD  "%MKLROOT%"\lib\intel64_win\mkl_intel_thread_dll.lib "%MKLROOT%"\lib\intel64_win\mkl_core_dll.lib "%MKLROOT%"\lib\intel64_win\mkl_intel_lp64_dll.lib
	```


2. OpenBLAS:

Download and extract [64-bit OpenBLAS](https://sourceforge.net/projects/openblas/files/v0.2.19/OpenBLAS-v0.2.19-Win64-int32.zip/download)

	```bash
	"C:\\Program Files (x86)\\Microsoft Visual Studio 12.0\\VC\\vcvarsall.bat" amd64
	set JAVA_HOME=C:\Program Files\Java\jdk1.8.0_25
	cl *.cpp -I. -I"%OPENBLASROOT%"\include -I"%JAVA_HOME%"\include -I"%JAVA_HOME%"\include\win32 -DUSE_OPEN_BLAS -Fesystemml_openblas-windows-x86_64.dll -MD -LD "%OPENBLASROOT%"\lib\libopenblas.dll.a
	```
