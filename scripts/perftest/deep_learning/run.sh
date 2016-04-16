#!/bin/bash

# Set appropriate size so that you are not measuring only garbage collection time
JAVA_OPTS="-Xmx20g -Xms20g -Xmn2048m -server"
OUTPUT_SYSTEMML_STATS="-stats"

invoke_systemml() {
	iter=$1
	setup=$2
	echo "Testing SystemML's conv2d with "$iter" iterations and using setup "$setup
	tstart=$(date +%s.%N) #$SECONDS
	echo $JAVA_OPTS $OUTPUT_SYSTEMML_STATS
	java $JAVA_OPTS -jar systemml-0.10.0-incubating-SNAPSHOT-standalone.jar -f test_conv2d.dml $OUTPUT_SYSTEMML_STATS -exec singlenode -args $iter $setup
	ttime=$(echo "$(date +%s.%N) - $tstart" | bc)
	echo "SystemML,"$iter","$setup","ttime >> time.txt
}

invoke_tensorflow() {
	iter=$1
	setup=$2
	echo "Testing SystemML's conv2d with "$iter" iterations and using setup "$setup
	tstart=$(date +%s.%N) #$SECONDS
	python test_tf_conv2d.py $iter $setup
	ttime=$(echo "$(date +%s.%N) - $tstart" | bc)
	echo "Tensorflow,"$iter","$setup","ttime >> time.txt
}

iter=1000
echo "-------------------------"
invoke_systemml $iter 1
invoke_tensorflow $iter 1
invoke_systemml $iter 2
invoke_tensorflow $iter 2
invoke_systemml $iter 3
invoke_tensorflow $iter 3
invoke_systemml $iter 4
invoke_tensorflow $iter 4
