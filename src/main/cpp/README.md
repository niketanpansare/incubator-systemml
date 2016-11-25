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

This directory contains the custom kernels used for GPU backend and also C++ stubs for native invocation.

# Running SystemML in native mode:

## Compile SystemML shared library

Though, SystemML will work with other open-source BLAS version, we recommend using Intel MKL.

1. Download and install community version of [Intel MKL](https://software.intel.com/sites/campaigns/nest/). 
Intel requires you to first register your email address and then sends the download link to your email address
with license key. 

2. Compile systemml shared library

	* You can either compile using CMake:
		```bash
		cmake .
		make
		```
		
	* Or on Linux with g++ compiler and Intel MKL:
	
		```bash
		export MKLROOT=/opt/intel/mkl
		export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.111-1.b15.el7_2.x86_64
		g++ -shared -fPIC -o libsystemml.so systemml.cpp -I. -I$MKLROOT/include -I$JAVA_HOME/include -I$JAVA_HOME/include/linux -fopenmp -L$MKLROOT/lib/intel64/ -lmkl_rt -lm
		```
	
	* Or on Windows (using 32-bit JVM) with cl.exe compiler and Intel MKL:
	
		* Change `cblas.h` to `mkl.h`.
		
		* Ensure that standard include files are accessible:
		
			```bash
			"C:\\Program Files (x86)\\Microsoft Visual Studio 12.0\\VC\\vcvarsall.bat"
			"%MKLROOT%"\bin\mklvars.bat ia32
			```
			
		* Then compile systemml using cl.exe. This assumes that you have set MKLROOT (likely in `C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2017\windows\mkl`)
		and JAVA_HOME environment variable.
		
			```bash
			cl systemml.cpp -I. -I"%MKLROOT%"\include -I"%JAVA_HOME%"\include -I"%JAVA_HOME%"\include\win32 -Fesystemml.dll -MD -LD "%MKLROOT%"\lib\ia32_win\mkl_intel_c_dll.lib "%MKLROOT%"\lib\ia32_win\mkl_intel_thread_dll.lib "%MKLROOT%"\lib\ia32_win\mkl_core_dll.lib 
			```
		
		* You may have to copy the dll `C:\Program Files (x86)\Common Files\Intel\Shared Libraries\redist\ia32_win\compiler\libiomp5md.dll` to your path.
	
	

3. Make sure that MKL and systemml shared library are available to Java.
 
	When using in Cluster mode, you will have to ensure that the shared library and dependent libraries are available on every node. 
	
	```bash
	export LD_LIBRARY_PATH=$MKL_ROOT/lib/intel64:.:$LD_LIBRARY_PATH
	```

## Run SystemML in native mode

	Provide `-native` flag to SystemML when invoked via commandline.
	```bash
	-native: <mode> (optional) execution mode (blas, blas_dense, blas_loop, blas_loop_dense)
	```