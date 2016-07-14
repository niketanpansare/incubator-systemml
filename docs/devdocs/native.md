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

# Initial testing of native 

## Compile on linux
Uses CMake.

Set JAVA\_HOME to your java installation. For instance, on Ubuntu, with Oracle Java 8, JAVA\_HOME=/usr/lib/jvm/java-8-oracle
<br/>
Set EIGEN3\_INCLUDE\_DIR to where the Eigen3 header files can be found. For instance, on my machine, EIGEN3\_HOME\_DIR=/usr/local/include/eigen3/
<br/>
Run cmake to generate the Makefile, run the makefile
```
cmake .
make
```

On Mac, GCC supports OpenMP. Install it with homebrew (or any other method). 
/usr/bin/gcc invokes the clang/LLVM
```
CC=gcc-6 cmake .
make
```

