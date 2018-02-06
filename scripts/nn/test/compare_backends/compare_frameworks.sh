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

# Install Keras:
# pip install keras==2.1.3

# Build tensorflow from the source with CUDA 8 and CuDNN 5 for fair comparison.
# git clone https://github.com/tensorflow/tensorflow
# cd tensorflow
# git checkout r1.5
# Use CUDA 8, CuDNN 5 and enable XLA.
# ./configure
# bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
# bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/nike/sysml_experiments/
# pip install --upgrade ~/nike/sysml_experiments/tensorflow-1.5.0-cp27-cp27mu-linux_x86_64.whl

# BigDL:
# wget https://s3-ap-southeast-1.amazonaws.com/bigdl-download/dist-spark-2.1.1-scala-2.11.8-all-0.4.0-dist.zip
# pip install bigdl==0.4.0

SPARK_PARAMS="--driver-memory 50g"
declare -a EPOCHS=(10 100)

if [[ -z "${SPARK_HOME}" ]]; then
	echo "Please set the environment variable SPARK_HOME"
else
	echo 'framework,model,data,epochs,batch_size,num_gpus,precision,blas,phase,codegen,total_time,data_loading_time,model_loading_time,fit_time,spark_init_time,predict_time' > time.txt
	if [ ! -d logs ]; then
		mkdir logs
	fi
	pip install keras==2.1.3
	
	for codegen in enabled disabled
	do
	
	for phase in train # test
	do
	# elephas failed for synchronous execution. Hence, we are not included it into the test suite.
	for framework in keras systemml
	do
		for epochs in "${EPOCHS[@]}"
		do
			model='lenet'
			data='mnist'
			batch_size=64
			for num_gpus in 0 1 # 2 - We are not testing multi-GPU setup
			do
				for precision in single double
				do
					if [ "$framework" == 'systemml' ]; then
						if [ "$num_gpus" == 0 ]; then
							declare -a SUPPORTED_BLAS=("none" "mkl" "openblas")
						else
							declare -a SUPPORTED_BLAS=("openblas")
						fi
					else
						declare -a SUPPORTED_BLAS=("eigen")
					fi
					for blas in "${SUPPORTED_BLAS[@]}"
					do
						$SPARK_HOME/bin/spark-submit --driver-memory 50g compare_frameworks.py --model=$model --data=$data --epochs=$epochs --batch_size=$batch_size --num_gpus=$num_gpus --framework=$framework --precision=$precision --blas=$blas --phase=$phase --codegen=$codegen &> logs/'log_'$model'_'$data'_'$epochs'_'$batch_size'_'$num_gpus'_'$framework'_'$precision'_'$blas'_'$phase'_codegen-'$codegen'.txt'
					done
				done
			done
		done
	done
	done
	done
	framework="bigdl"
	num_gpus=0
	batch_size=60 # must be multiple of number of cores
	blas="mkl"
	codegen='disabled'
	
	# BigDL doesnot support latest Keras version
	pip uninstall keras --yes
	pip install keras==1.2.2
	for phase in train # test
        do
        for epochs in "${EPOCHS[@]}"
        do
        	model='lenet'
        	data='mnist'
               	for precision in single double
		do
			BIGDL_PARAMS="--driver-class-path bigdl-SPARK_2.1-0.4.0-jar-with-dependencies.jar  --conf spark.shuffle.reduceLocality.enabled=false --conf spark.shuffle.blockTransferService=nio --conf spark.scheduler.minRegisteredResourcesRatio=0.0 --conf spark.scheduler.minRegisteredResourcesRatio=1.0 --conf spark.speculation=false"
                	$SPARK_HOME/bin/spark-submit $SPARK_PARAMS $BIGDL_PARAMS  compare_frameworks.py --model=$model --data=$data --epochs=$epochs --batch_size=$batch_size --num_gpus=$num_gpus --framework=$framework --precision=$precision --blas=$blas --phase=$phase --codegen=$codegen &> logs/'log_'$model'_'$data'_'$epochs'_'$batch_size'_'$num_gpus'_'$framework'_'$precision'_'$blas'_'$phase'_codegen-'$codegen'.txt'
		done
       	done
	done
	pip uninstall keras --yes
	pip install keras==2.1.3
fi
