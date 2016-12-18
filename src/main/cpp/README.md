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
To enable that, the user has to build `systemml.cpp` (available in this folder)
as a shared library and make it available through `LD_LIBRARY_PATH` (on linux)
or `PATH` (on Windows). If the shared library is not accessible, SystemML
falls back to its internal Java library.

In the below section, we describe the steps to build `systemml.cpp` using native BLAS.

# Step 1: Install BLAS

## Option 1: Install Intel MKL (Recommended for performance)

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

```bash
git clone https://github.com/xianyi/OpenBLAS
cd OpenBLAS
make FC=gfortran
sudo make PREFIX=/opt/openblas install
``` 

# Step 2: Set environment variables.

1. *Required:*  SystemML expects `JAVA_HOME` to be set to appropriate JDK.

2. Optional: Set USE_BLAS enviroment variable to either `mkl` (default) or `openblas`.

3. Optional: The user may chose pip to compile systemml by setting `COMPILE_NATIVE=1`.

4. Additional step for Windows users:

	* Ensure that the Visual Studio compiler `cl.exe` is accessible.
	
	* Ensure that standard include files are accessible (assumption: 32-bit VC 12):
		```bash
		"C:\\Program Files (x86)\\Microsoft Visual Studio 12.0\\VC\\vcvarsall.bat"
		"%MKLROOT%"\bin\mklvars.bat ia32
		```
	* You may have to copy the dll `C:\Program Files (x86)\Common Files\Intel\Shared Libraries\redist\ia32_win\compiler\libiomp5md.dll` to your path.
	
By default, if the user doesnot specify `USE_BLAS`, SystemML compiles using Intel MKL with the path provided by `MKLROOT`. 
Also, if the user doesnot specify `COMPILE_NATIVE=1`, then during pip installation we output the command the user can use to 
compile SystemML library post-installation. Here are some examples of the output after completing step 3:

	* Example 1: Intel MKL on Linux:
		
		```bash
		==========================================================
	  	Executing following command to compile native systemml library:
	  	g++ -o /home/[username]/libsystemml.so systemml-cpp/systemml.cpp  -I/[path-java-home]/include -Isystemml-cpp -I/opt/intel/mkl/include -I/[path-java-home]/include/linux -lmkl_rt -lpthread -lm -ldl -L/opt/intel/mkl/lib/intel64 -m64 -Wl,--no-as-needed -fopenmp -O3 -shared -fPIC
	  	==========================================================
		```
	
	* Example 2: OpenBLAS on Linux:
		
		```bash
		==========================================================
	  	Executing following command to compile native systemml library:
	  	g++ -o /home/[username]/libsystemml.so systemml-cpp/systemml.cpp  -I/[path-java-home]/include -Isystemml-cpp -I/opt/openblas/include -I/[path-java-home]/include/linux -lopenblas -lpthread -lm -ldl -DUSE_OPEN_BLAS -L/opt/openblas/lib -fopenmp -O3 -shared -fPIC
	  	==========================================================
		```


# Step 3: Install SystemML using pip

```bash
git checkout https://github.com/apache/incubator-systemml.git
cd incubator-systemml
mvn clean package -P distribution
pip install target/systemml-0.12.0-incubating-SNAPSHOT-python.tgz -v
```