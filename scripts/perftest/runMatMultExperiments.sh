#!/bin/bash
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

#-------------------------------------------------------------
# STEP 1: Make sure the current directory contains:
# libsystemml.so_mkl, libsystemml.so_openblas, SystemML.jar, matmult.dml, lib/*, log4j.properties

# STEP 2: Set CUDA Library Path if you plan to test GPU performance:
export LD_LIBRARY_PATH=. # Include jcuda and cuda lib if you plan to test GPU
# export CUDA_HOME=..

# STEP 3: Set appropriate size so that you are not measuring only garbage collection time
JAVA_OPTS="-Xmx20g -Xms20g -Xmn2048m -server"
OUTPUT_SYSTEMML_STATS="-stats"
#-------------------------------------------------------------


JAR_PATH="."
for j in `ls $JCUDA_PATH/*.jar`
do
        JAR_PATH=$JAR_PATH":"$j
done
for j in `ls lib/*.jar`
do
        JAR_PATH=$JAR_PATH":"$j
done

invoke_systemml() {
        iter=$4
        setup=$5
        gpu_flag=""
        if [ "$setup" == "GPU" ]
        then
        	gpu_flag="-gpu"	
       	fi
        echo "Testing "$setup" with "$iter" iterations and using setup ["$1", "$2"] %*% ["$2", "$3"]"
        tstart=$(date +%s.%N) #$SECONDS
        echo $JAVA_OPTS $OUTPUT_SYSTEMML_STATS
        java $JAVA_OPTS -classpath $JAR_PATH:SystemML.jar org.apache.sysml.api.DMLScript -f matmult.dml $OUTPUT_SYSTEMML_STATS $gpu_flag -args $1 $2 $3 $4
        ttime=$(echo "$(date +%s.%N) - $tstart" | bc)
        echo $setup","$iter","$1","$2","$3","$ttime >> time.txt
}


rm time1.txt time.txt
iter=1000
echo "BLAS,IsSingleThreaded,M,N,K,Time" > time1.txt
echo "-------------------------"
for i in 1000 # 1 10 100 1000 2000 5000 10000
do
for j in 1000 # 1 10 100 1000 2000 5000 10000
do
for k in 1000 # 1 10 100 1000 2000 5000 10000
do
        # Intel MKL
        export USE_BLAS=mkl
        cp libsystemml.so_mkl libsystemml.so
        invoke_systemml $i $j $k $iter IntelMKL
        
        # OpenBLAS
        export USE_BLAS=openblas
        cp libsystemml.so_openblas libsystemml.so
        invoke_systemml $i $j $k $iter OpenBLAS
        
        # Java
        mv libsystemml.so tmp.so
        invoke_systemml $i $j $k $iter Java
        
        # GPU
        # invoke_systemml $i $j $k $iter GPU
done
done
done
