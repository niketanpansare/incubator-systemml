#!/usr/bin/python
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

# Assumption: pip install keras
#
# This test validates SystemML's deep learning APIs (Keras2DML, Caffe2DML and nn layer) by comparing the results with that of keras.
#
# To run:
#   - Python 2: `PYSPARK_PYTHON=python2 spark-submit --master local[*] --driver-memory 10g  --driver-class-path ../../../../target/SystemML.jar,../../../../target/systemml-*-extra.jar test_nn_numpy.py`
#   - Python 3: `PYSPARK_PYTHON=python3 spark-submit --master local[*] --driver-memory 10g --driver-class-path SystemML.jar,systemml-*-extra.jar test_nn_numpy.py`

# Make the `systemml` package importable
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
sys.path.insert(0, path)

import unittest

import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, LSTM, UpSampling2D, SimpleRNN, Activation
from keras.optimizers import SGD
from keras import backend as K
from keras.models import Model
from systemml.mllearn import Keras2DML
from pyspark.sql import SparkSession
from keras.utils import np_utils
from scipy import stats
from operator import mul

batch_size = 32
input_shape = (3, 28, 28)
input_shape2d = (15, 10)
K.set_image_data_format('channels_first')
# K.set_image_dim_ordering("th")

def get_tensor(shape, random=True):
    if random:
        return stats.zscore(np.random.randint(100, size=shape))
    else:
        size = reduce(mul, list(shape), 1)
        return stats.zscore(np.arange(size).reshape(shape))

keras_3dtensor = get_tensor((batch_size, input_shape[0], input_shape[1], input_shape[2]), random=False)
keras_2dtensor = get_tensor((batch_size, input_shape2d[0], input_shape2d[1]), random=False)
sysml_3dmatrix = keras_3dtensor.reshape((batch_size, -1))
sysml_2dmatrix = keras_2dtensor.reshape((batch_size, -1))
num_labels = 10
tmp_dir = 'tmp_dir'

spark = SparkSession.builder.getOrCreate()

def initialize_weights(model):
    for l in range(len(model.layers)):
        if model.layers[l].get_weights() is not None or len(model.layers[l].get_weights()) > 0:
            model.layers[l].set_weights([get_tensor(elem.shape, random=True) for elem in
                                         model.layers[l].get_weights()])
    return model

def get_output_shape(layers):
    tmp_keras_model = Sequential()
    for layer in layers:
        tmp_keras_model.add(layer)
    return tmp_keras_model.layers[-1].output_shape

def get_one_hot_encoded_labels(output_shape):
    output_cells = reduce(mul, list(output_shape[1:]), 1)
    y = np.array(np.random.choice(output_cells, batch_size))
    y[0] = output_cells - 1
    one_hot_labels = np_utils.to_categorical(y, num_classes=output_cells)
    return one_hot_labels

def get_sysml_model(keras_model, sysml_input_shape):
    sysml_model = Keras2DML(spark, keras_model, input_shape=sysml_input_shape, weights=tmp_dir, test_iter=0, max_iter=1, batch_size=batch_size, weight_decay=0)
    # For apples-to-apples comparison of output probabilities:
    # By performing one-hot encoding outside, we ensure that the ordering of the TF columns
    # matches that of SystemML
    sysml_model.set(train_algo='batch', perform_one_hot_encoding=False)
    return sysml_model

def base_test(layers, add_dense=False, test_backward=True):
    layers = [layers] if not isinstance(layers, list) else layers
    output_shape = get_output_shape(layers)
    # --------------------------------------
    # Create Keras model
    keras_model = Sequential()
    for layer in layers:
        keras_model.add(layer)
    if len(output_shape) > 2:
        # 3-D model
        sysml_input_shape = input_shape
        sysml_matrix = sysml_3dmatrix
        keras_tensor = keras_3dtensor
        keras_model.add(Flatten())
    elif len(output_shape) == 2:
        # 2-D model
        sysml_input_shape = (input_shape2d[0], input_shape2d[1], 1)
        sysml_matrix = sysml_2dmatrix
        keras_tensor = keras_2dtensor
    else:
        raise Exception('Model with output shape ' + str(output_shape) + ' is not supported.')
    if add_dense:
        keras_model.add(Dense(num_labels, activation='softmax'))
    else:
        keras_model.add(Activation('softmax'))
    keras_model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1, decay=0, momentum=0, nesterov=False))
    print(keras_model.summary())
    # --------------------------------------
    keras_model = initialize_weights(keras_model)
    sysml_model = get_sysml_model(keras_model, sysml_input_shape)
    # --------------------------------------
    sysml_preds = sysml_model.predict_proba(sysml_matrix).flatten()
    keras_preds = keras_model.predict(keras_tensor).flatten()
    if test_backward:
        one_hot_labels = get_one_hot_encoded_labels(keras_model.layers[-1].output_shape)
        sysml_model.fit(sysml_matrix, one_hot_labels)
        keras_model.train_on_batch(keras_tensor, one_hot_labels)
        sysml_preds = sysml_model.predict_proba(sysml_matrix).flatten()
        keras_preds = keras_model.predict(keras_tensor).flatten()
    return sysml_preds, keras_preds

def test_forward(layers):
    sysml_preds, keras_preds = base_test(layers, test_backward=False)
    print('Forward:' + str(sysml_preds) + ' == ' + str(keras_preds) + '?')
    return np.allclose(sysml_preds, keras_preds)

def test_backward(layers):
    sysml_preds, keras_preds = base_test(layers, test_backward=True)
    print('Backward:' + str(sysml_preds) + ' == ' + str(keras_preds) + '?')
    return np.allclose(sysml_preds, keras_preds)

class TestNNLibrary(unittest.TestCase):

    def test_flatten_forward(self):
        self.failUnless(test_forward(Flatten(input_shape=input_shape)))

    def test_flatten_backward(self):
        self.failUnless(test_backward(Flatten(input_shape=input_shape)))

if __name__ == '__main__':
    unittest.main()
