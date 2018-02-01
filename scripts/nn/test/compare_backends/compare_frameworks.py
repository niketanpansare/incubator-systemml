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

# To check usage, use -h option.


import time
start = time.time()
data_loading = 0.0
import os, argparse, sys

parser=argparse.ArgumentParser()
parser.add_argument('--model', help='Supported values are: lenet. Default: lenet', type=str, default='lenet')
parser.add_argument('--data', help='Supported values are: mnist. Default: mnist', type=str, default='mnist')
parser.add_argument('--epochs', help='Number of epochs. Default: 10', type=int, default=10)
parser.add_argument('--batch_size', help='Batch size. Default: 64', type=int, default=64)
parser.add_argument('--num_gpus', help='Number of GPUs. Default: 0', type=int, default=0)
parser.add_argument('--framework', help='Supported values are: systemml, keras. Default: systemml', type=str, default='systemml')
args=parser.parse_args()

config = {'model':args.model, 'data':args.data, 'epochs':args.epochs, 'batch_size':args.batch_size, 'num_gpus':args.num_gpus, 'framework':args.framework, 'display':100}
num_gpus = int(config['num_gpus'])

if config['framework'] == 'systemml' or num_gpus == 0:
	# When framework is systemml, force any tensorflow allocation to happen on CPU to avoid GPU OOM
	os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
	os.environ['CUDA_VISIBLE_DEVICES'] = ''
elif num_gpus == 1:
	os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

input_shapes = { 'mnist':(1,28,28) }
num_labels = {'mnist':10 }
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
import math
import numpy as np
from sklearn.utils import shuffle
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout,Flatten
from keras import backend as K
from keras.models import Model
from keras.utils import np_utils
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

# For fair comparison
if config['framework'] == 'keras':
	tf_config = tf.ConfigProto()
	#tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3
	tf_config.gpu_options.allow_growth = True
	set_session(tf.Session(config=tf_config))

#K.set_floatx('float64')
def get_data():
	if config['data'] == 'mnist':
		from mlxtend.data import mnist_data
		X, y = mnist_data()
		X_train, y_train =  shuffle(X, y)
		# Scale the input features
		scale = 0.00390625
		X_train = X_train*scale
		return X_train, y_train
	else:
		raise ValueError('Unsupported data:' + str(config['data']))

def get_keras_input_shape(input_shape):
	if K.image_data_format() == 'channels_first':
		return input_shape
	else:
		return ( input_shape[1], input_shape[2], input_shape[0] )

def get_keras_model():
	if config['model'] == 'lenet':
		keras_model = Sequential()
		keras_model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=get_keras_input_shape(input_shapes[config['data']]), padding='same'))
		keras_model.add(MaxPooling2D(pool_size=(2, 2)))
		keras_model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
		keras_model.add(MaxPooling2D(pool_size=(2, 2)))
		keras_model.add(Flatten())
		keras_model.add(Dense(512, activation='relu'))
		keras_model.add(Dropout(0.5))
		keras_model.add(Dense(10, activation='softmax'))
	else:
		raise ValueError('Unsupported model:' + str(config['model']))
	#if type(keras_model) == keras.models.Sequential:
	#	# Convert the sequential model to functional model
	#	if keras_model.model is None:
	#		keras_model.build()
	#	keras_model = keras_model.model
	return keras_model

t0 = time.time()
X, y = get_data() 
data_loading = data_loading + time.time() - t0

#X = X.astype(np.float64)
epochs = int(config['epochs'])
batch_size = int(config['batch_size'])
num_samples = X.shape[0]
max_iter = int(epochs*math.ceil(num_samples/batch_size))
display = int(config['display'])

def get_framework_model(framework):
	keras_model = get_keras_model()
	if framework == 'systemml':
		from systemml.mllearn import Keras2DML
		sysml_model = Keras2DML(spark, keras_model, input_shape=input_shapes[config['data']], batch_size=batch_size, max_iter=max_iter, test_iter=0, display=display)
		sysml_model.setConfigProperty("sysml.native.blas", "openblas")
		sysml_model.setStatistics(True).setStatisticsMaxHeavyHitters(100)
		sysml_model.setConfigProperty("sysml.gpu.sync.postProcess", "false")
		sysml_model.setConfigProperty("sysml.stats.finegrained", "true")
		#sysml_model.setConfigProperty("sysml.gpu.eager.cudaFree", "true")
		sysml_model.setConfigProperty("sysml.floating.point.precision", "single")
		#sysml_model.setConfigProperty("sysml.codegen.enabled", "true").setConfigProperty("sysml.codegen.plancache", "true")
		if num_gpus >= 1:
			sysml_model.setGPU(True).setForceGPU(True)
		if num_gpus > 1:
			sysml_model.set(train_algo="allreduce_parallel_batches", parallel_batches=num_gpus)
		return sysml_model
	elif framework == 'keras':
		if num_gpus >= 2:
			from keras.utils import multi_gpu_model
			keras_model = multi_gpu_model(keras_model, gpus=num_gpus)
		keras_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.95, decay=5e-4, nesterov=True))
		return keras_model
	else:
		raise ValueError('Unsupported framework:' + str(framework))

def get_framework_data(framework):
	if framework == 'systemml':
		return X, y
	elif framework == 'keras':
		input_shape=get_keras_input_shape(input_shapes[config['data']])
		return X.reshape((-1, input_shape[0], input_shape[1], input_shape[2])), np_utils.to_categorical(y, num_labels[config['data']])


framework = config['framework']
t0 = time.time()
framework_model = get_framework_model(framework)
t1 = time.time()
framework_X, framework_y = get_framework_data(framework)
t2 = time.time()
data_loading = data_loading + t2 - t1
model_loading = t1 - t0 

print("Starting fit for the framework:" + framework)
if framework == 'systemml':
	framework_model.fit(framework_X, framework_y)
elif framework == 'keras':
	framework_model.fit(framework_X, framework_y, epochs=epochs, batch_size=batch_size)
	K.clear_session()
else:
	raise ValueError('Unsupported framework:' + str(framework))
end = time.time()
with open('time.txt', 'a') as f:
	f.write(config['framework'] + ',' + config['model'] + ',' + config['data'] + ',' + str(config['epochs']) + ',' + str(config['batch_size']) + ',' + str(config['num_gpus']) + ',' + str(end-start) + ',' + str(data_loading) + ',' + str(model_loading) + '\n')

