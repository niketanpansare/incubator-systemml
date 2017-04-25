## CMAKE Build Instructions

### On MacOS
The version of clang that ships with Mac does not come with OpenMP. `brew install` either `llvm` or `g++`. The instructions that follow are for llvm:

1. Intel MKL - CMake should detect the MKL installation path, otherwise it can specified by the environment variable `MKLROOT`. To use (with clang):
```
mkdir INTEL && cd INTEL
CXX=/usr/local/opt/llvm/bin/clang++ CC=/usr/local/opt/llvm/bin/clang LDFLAGS=-L/usr/local/opt/llvm/lib CPPFLAGS=-I/usr/local/opt/llvm/include cmake  -DUSE_INTEL_MKL=ON ..
make install
```

2. OpenBLAS - CMake should be able to detect the path of OpenBLAS. If it can't, set the `OpenBLAS` environment variable. If using `brew` to install OpenBLAS, set the `OpenBLAS_HOME` environment variable to `/usr/local/opt/openblas/`. To use (with clang):
```
export OpenBLAS_HOME=/usr/local/opt/openblas/
mkdir OPENBLAS && cd OPENBLAS
CXX=/usr/local/opt/llvm/bin/clang++ CC=/usr/local/opt/llvm/bin/clang LDFLAGS=-L/usr/local/opt/llvm/lib CPPFLAGS=-I/usr/local/opt/llvm/include cmake  -DUSE_OPEN_BLAS=ON -DCMAKE_BUILD_TYPE=Release ..
make install
```

### On Linux
With the appropriate prerequisites (C++ compiler with OpenMP, OpenBLAS or IntelMKL)
1. Intel MKL
```
mkdir INTEL && cd INTEL
cmake -DUSE_INTEL_MKL=ON -DCMAKE_BUILD_TYPE=Release ..
make install
```
2. OpenBLAS - If CMake cannot detect your OpenBLAS installation, set the `OpenBLAS_HOME` environment variable to the OpenBLAS Home.
```
mkdir OPENBLAS && cd OPENBLAS
cmake -DUSE_OPEN_BLAS=ON -DCMAKE_BUILD_TYPE=Release ..
make install
```

The generated library files are placed in src/main/cpp/lib. This location can be changed from the CMakeLists.txt file.


### On Windows
- Install MKL or Download the OpenBlas Binary
- Install Visual Studio Community Edition (tested on VS 2017)
- Use the CMake GUI, select the source directory, the output directory
- Press the `configure` button, set the `generator` and `use default native compilers` option
- Set the `CMAKE_BUILD_TYPE` to `Release`, this sets the appropriate optimization flags
- By default, `USE_INTEL_MKL` is selected, if you wanted to use OpenBLAS, unselect the `USE_INTEL_MKL`, select the `USE_OPEN_BLAS`.
- You might run into errors a couple of times, select the appropriate library and include files/directories (For MKL or OpenBLAS) a couple of times, and all the errors should go away.
- Then press generate. This will generate Visual Studio project files, which you can open in VS2017 to compile the libraries.
