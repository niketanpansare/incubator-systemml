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

# The framework elephas (as of Feb 5th, 2018) fails with TypeError: 'LabeledPoint' object is not iterable

parser=argparse.ArgumentParser()
parser.add_argument('--model', help='Supported values are: lenet. Default: lenet', type=str, default='lenet')
parser.add_argument('--data', help='Supported values are: mnist. Default: mnist', type=str, default='mnist')
parser.add_argument('--epochs', help='Number of epochs. Default: 10', type=int, default=10)
parser.add_argument('--batch_size', help='Batch size. Default: 64', type=int, default=64)
parser.add_argument('--num_gpus', help='Number of GPUs. Default: 0', type=int, default=0)
parser.add_argument('--framework', help='Supported values are: systemml, keras, elephas, bigdl. Default: systemml', type=str, default='systemml')
parser.add_argument('--precision', help='Supported values are: single, double. Default: single', type=str, default='single')
parser.add_argument('--blas', help='Supported values are: openblas, mkl, none. Default: openblas', type=str, default='openblas')
parser.add_argument('--phase', help='Supported values are: train, test. Default: train', type=str, default='train')
parser.add_argument('--codegen', help='Supported values are: enabled, disabled. Default: enabled', type=str, default='enabled')
args=parser.parse_args()

config = {'model':args.model, 'data':args.data, 'epochs':args.epochs, 'batch_size':args.batch_size, 'num_gpus':args.num_gpus, 'framework':args.framework, 'display':100}
num_gpus = int(config['num_gpus'])

if args.precision != 'single' and args.precision != 'double':
        raise ValueError('Incorrect precision:' + args.precision)

if config['framework'] == 'systemml' or num_gpus == 0:
	# When framework is systemml, force any tensorflow allocation to happen on CPU to avoid GPU OOM
	os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
	os.environ['CUDA_VISIBLE_DEVICES'] = ''
elif num_gpus == 1:
	os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

input_shapes = { 'mnist':(1,28,28) }
num_labels = {'mnist':10 }

t0 = time.time()
from pyspark import SparkContext
sc = SparkContext()
sc.parallelize([1, 2, 3, 4, 5]).count()
if config['framework'] != 'elephas':
	# As elephas only support Spark 1.6 or less as of Feb 5th 2018
	from pyspark.sql import SparkSession
	spark = SparkSession.builder.getOrCreate()
	spark.createDataFrame([(1, 4), (2, 5), (3, 6)], ["A", "B"]).count()
spark_init_time = time.time() - t0

import math
import numpy as np
from sklearn.utils import shuffle
import keras
from keras.models import Sequential
from keras.layers import Input, Dense,MaxPooling2D, Dropout,Flatten
from keras import backend as K
from keras.models import Model
from keras.utils import np_utils
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import regularizers

if config['framework'] == 'bigdl':
	# bigdl only supports keras 1.2
	from keras.layers import Convolution2D
else:
	from keras.layers import Conv2D

# For fair comparison
if config['framework'] == 'keras':
	
	tf_config = tf.ConfigProto()
	#tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3
	tf_config.gpu_options.allow_growth = True
	if args.codegen == 'enabled':
		tf_config.graph_options.optimizer_options.global_jit_level = K.tf.OptimizerOptions.ON_1
	else:
		tf_config.graph_options.optimizer_options.global_jit_level = 0
	set_session(tf.Session(config=tf_config))

if args.precision == 'double' and config['framework'] == 'keras':
	K.set_floatx('float64')
	#print('double precision is not supported in Keras. See https://stackoverflow.com/questions/48552508/running-keras-with-double-precision-fails/')

def get_data():
	if config['data'] == 'mnist':
		# 5K dataset
		#from mlxtend.data import mnist_data
		#X, y = mnist_data()
		# 60K dataset
		import mnist
		X = mnist.train_images().reshape(60000, -1)
		y = mnist.train_labels()
		X_train, y_train =  shuffle(X, y)
		# Scale the input features
		scale = 0.00390625
		X_train = X_train*scale
		return X_train, y_train
	else:
		raise ValueError('Unsupported data:' + str(config['data']))

def get_keras_input_shape(input_shape):
	#if K.image_data_format() == 'channels_first':
	#	return input_shape
	#else:
	#	return ( input_shape[1], input_shape[2], input_shape[0] )
	return ( input_shape[1], input_shape[2], input_shape[0] )

def conv2d(nb_filter, nb_row, nb_col, activation='relu', padding='same', input_shape=None):
	if input_shape != None:
		if config['framework'] == 'bigdl':
			return Convolution2D(nb_filter, nb_row, nb_col, activation=activation, input_shape=input_shape, W_regularizer=regularizers.l2(0.01), border_mode=padding)
		else:
			return Conv2D(nb_filter, kernel_size=(nb_row, nb_col), activation=activation, input_shape=input_shape, kernel_regularizer=regularizers.l2(0.01), padding=padding)
	else:
		if config['framework'] == 'bigdl':
                        return Convolution2D(nb_filter, nb_row, nb_col, activation=activation, W_regularizer=regularizers.l2(0.01), border_mode=padding)
                else:
                        return Conv2D(nb_filter, kernel_size=(nb_row, nb_col), activation=activation, kernel_regularizer=regularizers.l2(0.01), padding=padding)

def dense(num_out, activation='relu'):
	if config['framework'] == 'bigdl':
		return Dense(num_out, activation=activation, W_regularizer=regularizers.l2(0.01))
	else:
		return Dense(num_out, activation=activation, kernel_regularizer=regularizers.l2(0.01))
	
def get_keras_model():
	if config['model'] == 'lenet':
		keras_model = Sequential()
		keras_model.add(conv2d(32, 5, 5, input_shape=get_keras_input_shape(input_shapes[config['data']])))
		keras_model.add(MaxPooling2D(pool_size=(2, 2)))
		keras_model.add(conv2d(64, 5, 5))
		keras_model.add(MaxPooling2D(pool_size=(2, 2)))
		keras_model.add(Flatten())
		keras_model.add(dense(512))
		keras_model.add(Dropout(0.5))
		keras_model.add(dense(10, activation='softmax'))
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
if args.precision == 'double':
	X = X.astype(np.float64)
data_loading = data_loading + time.time() - t0

epochs = int(config['epochs'])
batch_size = int(config['batch_size'])
num_samples = X.shape[0]
max_iter = int(epochs*math.ceil(num_samples/batch_size))
display = int(config['display'])

def get_framework_model(framework):
	keras_model = get_keras_model()
	if framework == 'systemml':
		from systemml.mllearn import Keras2DML
		load_keras_weights = True if args.phase == 'test' else False
		sysml_model = Keras2DML(spark, keras_model, input_shape=input_shapes[config['data']], batch_size=batch_size, max_iter=max_iter, test_iter=0, display=display, load_keras_weights=load_keras_weights)
		sysml_model.setStatistics(True).setStatisticsMaxHeavyHitters(100)
		sysml_model.setConfigProperty("sysml.gpu.sync.postProcess", "false")
		sysml_model.setConfigProperty("sysml.stats.finegrained", "true")
		#sysml_model.setConfigProperty("sysml.gpu.eager.cudaFree", "true")
		# From configuration:
		sysml_model.setConfigProperty("sysml.native.blas", args.blas)
		sysml_model.setConfigProperty("sysml.floating.point.precision", args.precision)
		if args.codegen == 'enabled':
			sysml_model.setConfigProperty("sysml.codegen.enabled", "true").setConfigProperty("sysml.codegen.optimizer", "fuse_no_redundancy")
		# .setConfigProperty("sysml.codegen.plancache", "true")
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
	elif  framework == 'elephas':
		keras_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.95, decay=5e-4, nesterov=True))
		from elephas.spark_model import SparkModel
		from elephas import optimizers as elephas_optimizers
		optim = elephas_optimizers.SGD(lr=0.01, momentum=0.95, decay=5e-4, nesterov=True)
		spark_model = SparkModel(sc, keras_model, optimizer=optim,mode='synchronous') #, num_workers=2)
		return spark_model
	elif  framework == 'bigdl':
		keras_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.95, decay=5e-4, nesterov=True))
		model_json = keras_model.to_json()
		path = "model.json"
		with open(path, "w") as json_file:
			json_file.write(model_json)
		from bigdl.nn.layer import *
		bigdl_model = Model.load_keras(json_path=path)
		return bigdl_model
	else:
		raise ValueError('Unsupported framework:' + str(framework))

def get_framework_data(framework):
	if framework == 'systemml':
		return X, y
	elif framework == 'keras':
		input_shape=get_keras_input_shape(input_shapes[config['data']])
		return X.reshape((-1, input_shape[0], input_shape[1], input_shape[2])), np_utils.to_categorical(y, num_labels[config['data']])
	elif framework == 'elephas':
		#from elephas.utils.rdd_utils import to_labeled_point
		#return to_labeled_point(sc, X, y, categorical=True), None
		from elephas.utils.rdd_utils import to_labeled_point
		return to_labeled_point(sc, X, y, categorical=True), None
	elif framework == 'bigdl':
		if config['data'] == 'mnist':
			input_shape=get_keras_input_shape(input_shapes[config['data']])
			images, labels = sc.parallelize(X.reshape((-1, input_shape[0], input_shape[1], input_shape[2]))), sc.parallelize(y + 1)  # Target start from 1 in BigDL
			return images.zip(labels).map(lambda t: Sample.from_ndarray(t[0], t[1])), None
		return X, y
	else:
		raise ValueError('Unsupported framework:' + str(framework))


framework = config['framework']
t0 = time.time()
framework_model = get_framework_model(framework)
t1 = time.time()
framework_X, framework_y = get_framework_data(framework)
t2 = time.time()
data_loading = data_loading + t2 - t1
model_loading = t1 - t0 

if args.phase == 'train':
        print("Starting fit for the framework:" + framework)
        if framework == 'systemml':
                framework_model.fit(framework_X, framework_y)
        elif framework == 'keras':
                framework_model.fit(framework_X, framework_y, epochs=epochs, batch_size=batch_size)
                K.clear_session()
        elif framework == 'elephas':
                framework_model.train(framework_X, nb_epoch=epochs, batch_size=batch_size)
        elif framework =='bigdl':
                from bigdl.examples.keras.keras_utils import *
                from bigdl.optim.optimizer import *
                from bigdl.util.common import *
                from bigdl.nn.criterion import *
                init_engine()
                #batch_size = 60 # make this divisible by number of cores
                #optim = SGD(learningrate=0.01, momentum=0.95, learningrate_decay=5e-4, nesterov=True, dampening=0) # Fails
                optim = Adam()
                bigdl_type = "float" if args.precision == 'single' else "double"
                optimizer = Optimizer(model=framework_model, training_rdd=framework_X, end_trigger=MaxEpoch(epochs), batch_size=batch_size, optim_method=Adam(), criterion=ClassNLLCriterion(logProbAsInput=False), bigdl_type=bigdl_type)
                optimizer.optimize()
        else:
                raise ValueError('Unsupported framework:' + str(framework))
t3 = time.time()
if args.phase == 'test':
	print("Starting predict for the framework:" + framework)
	if framework == 'systemml':
        	preds = framework_model.predict(framework_X)
		if hasattr(preds, '_jdf'):
			preds.count()
	elif framework == 'keras':
        	framework_model.predict(framework_X)
	        K.clear_session()
	elif framework == 'elephas':
        	framework_model.predict(framework_X).count()
	elif framework =='bigdl':
		from bigdl.util.common import *
        	init_engine()
	        #batch_size = 60 # make this divisible by number of cores
        	#optim = SGD(learningrate=0.01, momentum=0.95, learningrate_decay=5e-4, nesterov=True, dampening=0) # Fails
        	bigdl_type = "float" if args.precision == 'single' else "double"
		framework_model.predict(framework_X).count()
	else:
        	raise ValueError('Unsupported framework:' + str(framework))
end = time.time()
with open('time.txt', 'a') as f:
	f.write(config['framework'] + ',' + config['model'] + ',' + config['data'] + ',' + str(config['epochs']) + ',' + str(config['batch_size']) + ',' + str(config['num_gpus']) + ',' + args.precision + ',' + args.blas + ',' + args.phase + ',' + args.codegen + ',' + str(end-start) + ',' + str(data_loading) + ',' + str(model_loading) + ',' + str(t3-t2) + ',' + str(spark_init_time) + ',' + str(end-t3) + '\n')

