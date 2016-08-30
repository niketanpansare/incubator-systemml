#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
# 
#   http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------

cmake_minimum_required(VERSION 2.6.2)
include_directories( "$ENV{JAVA_HOME}/include")
include_directories( "$ENV{JAVA_HOME}/include/linux")
include_directories( "$ENV{JAVA_HOME}/include/darwin")

enable_language(CXX)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(Eigen3 REQUIRED)
include_directories(EIGEN3_INCLUDE_DIR)

project (SystemML)
add_library(SystemML SHARED SystemML.cpp)