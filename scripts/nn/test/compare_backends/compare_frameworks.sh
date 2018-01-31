#!/usr/bin/bash
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

# Build tensorflow from the source with CUDA 8 and CuDNN 5 for fair comparison.
# git clone https://github.com/tensorflow/tensorflow
# cd tensorflow
# git checkout r1.5
# Use CUDA 8 and CuDNN 5
# ./configure
# bazel build --config=opt --config=cuda ./tensorflow/tools/pip_package:build_pip_package

rm time.txt &> /dev/null
if [ ! -d logs ]; then
	mkdir logs
fi
for framework in keras systemml
do
	for epochs in 10 100
	do
		for num_gpus in 0 1 2
		do
			model='lenet'
			data='mnist'
			batch_size=64
			$SPARK_HOME/bin/spark-submit --driver-memory 20g compare_frameworks.py --model=$model --data=$data --epochs=$epochs --batch_size=$batch_size --num_gpus=$num_gpus --framework=$framework &> logs/'log_'$model'_'$data'_'$epochs'_'$batch_size'_'$num_gpus'_'$framework'.txt'
		done
	done
done
