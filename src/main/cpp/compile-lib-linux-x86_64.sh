#!/bin/bash
# This shell script compiles the required shared libraries for 64-bit Linux on x86 machine

# yum whatprovides libgcc_s.so.1
# GNU Standard C++ Library: libstdc++.so.6
# GCC version 4.8 shared support library: libgcc_s.so.1
# The GNU libc libraries: libm.so.6, libdl.so.2, libc.so.6, libpthread.so.0
# GCC OpenMP v3.0 shared support library: libgomp.so.1
gcc_toolkit="libgcc_s.so\|libm.so\|libstdc++\|libc.so\|libdl.so\|libgomp.so\|libpthread.so"
linux_loader="linux-vdso.so\|ld-linux-x86-64.so"
intel_mkl="libmkl_rt.so"

# Fortran runtime: libgfortran.so.3
# GCC __float128 shared support library: libquadmath.so.0
openblas="libopenblas.so\|libgfortran.so\|libquadmath.so"

if [[ -z "${JAVA_HOME}" ]]; then
	echo "Error: Expected JAVA_HOME to be set to the JDK"
else
	# Assumes that MKL is installed
	mkdir INTEL && cd INTEL
	cmake -DUSE_INTEL_MKL=ON ..
	make
	cp libpreload.so ../lib/libpreload_systemml-linux-x86_64.so
	cp libsystemml.so ../lib/libsystemml_mkl-linux-x86_64.so
	cd ..
	rm -rf INTEL

	# Assumes that OpenBLAS is installed with GNU OpenMP
	# git clone https://github.com/xianyi/OpenBLAS.git
	# cd OpenBLAS/
	# make clean
	# make USE_OPENMP=1
	# sudo make install
	mkdir OPENBLAS && cd OPENBLAS
	cmake -DUSE_OPEN_BLAS=ON ..
	make
	cp libsystemml.so ../lib/libsystemml_openblas-linux-x86_64.so
	# If you get an error: '/usr/lib64/libopenblas.so: No such file or directory', please execute:
	# sudo ln -s /opt/OpenBLAS/lib/libopenblas.so /usr/lib64/libopenblas.so
	cd ..
	rm -rf OPENBLAS

	echo "-----------------------------------------------------------------------"
	echo "Check for unexpected dependencies added after code change or new setup:"
	echo "Non-standard dependencies for libpreload_systemml-linux-x86_64.so"
	ldd lib/libpreload_systemml-linux-x86_64.so | grep -v $gcc_toolkit"\|"$linux_loader
	echo "Non-standard dependencies for libsystemml_mkl-linux-x86_64.so"
	ldd lib/libsystemml_mkl-linux-x86_64.so | grep -v $gcc_toolkit"\|"$linux_loader"\|"$intel_mkl
	echo "Non-standard dependencies for libsystemml_openblas-linux-x86_64.so"
	ldd lib/libsystemml_openblas-linux-x86_64.so | grep -v $gcc_toolkit"\|"$linux_loader"\|"$openblas
	echo "-----------------------------------------------------------------------"
fi
