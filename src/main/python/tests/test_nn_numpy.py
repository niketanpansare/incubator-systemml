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

batch_size = 32
input_shape = (3, 64, 64)
input_shape2d = (15, 10)
K.set_image_data_format('channels_first')
# K.set_image_dim_ordering("th")
keras_3dtensor = np.random.rand(batch_size, input_shape[0], input_shape[1], input_shape[2])
keras_2dtensor = np.random.rand(batch_size, input_shape2d[0], input_shape2d[1])
sysml_3dmatrix = keras_3dtensor.reshape((batch_size, -1))
sysml_2dmatrix = keras_2dtensor.reshape((batch_size, -1))
num_labels = 10
tmp_dir = 'tmp_dir'

spark = SparkSession.builder.getOrCreate()

def are_predictions_all_close(keras_model, sysml_input_shape, is2D=False, rtol=1e-05, atol=1e-08):
    output_cells = 1
    output_shape = keras_model.layers[-1].output_shape
    for i in range(len(output_shape) - 1):
        output_cells = output_cells * output_shape[i + 1]
    one_hot_labels = np_utils.to_categorical(np.array(np.random.choice(output_cells, batch_size)), num_classes=output_cells)
    sysml_model = Keras2DML(spark, keras_model, input_shape=sysml_input_shape, weights=tmp_dir, test_iter=0, max_iter=1, batch_size=batch_size, weight_decay=0)
    keras_tensor = keras_2dtensor if is2D else keras_3dtensor
    sysml_matrix = sysml_2dmatrix if is2D else sysml_3dmatrix
    keras_preds = keras_model.predict(keras_tensor  + 0.001).flatten()
    sysml_preds = sysml_model.predict_proba(sysml_matrix + 0.001).flatten()
    # For apples-to-apples comparison of output probabilities:
    # By performing one-hot encoding outside, we ensure that the ordering of the TF columns
    # matches that of SystemML
    sysml_model.set(train_algo='batch', perform_one_hot_encoding=False)
    sysml_model.fit(sysml_matrix, one_hot_labels)
    keras_model.train_on_batch(keras_tensor, one_hot_labels)
    keras_preds1 = keras_model.predict(keras_tensor + 0.002).flatten()
    sysml_preds1 = sysml_model.predict_proba(sysml_matrix + 0.002).flatten()
    print('Keras output before training:' + str(keras_preds))
    print('SystemML output before training:' + str(sysml_preds))
    print('Keras output after training:' + str(keras_preds1))
    print('SystemML output after training:' + str(sysml_preds1))
    print('Difference in the output before training:' + str(np.abs(sysml_preds-keras_preds)))
    print('Difference in the output after training:' + str(np.abs(sysml_preds1 - keras_preds1)))
    is_forward_loading_correct =  np.allclose(keras_preds, sysml_preds, rtol=rtol, atol=atol)
    is_backward_correct = np.allclose(keras_preds1, sysml_preds1, rtol=rtol, atol=atol)
    if is_forward_loading_correct and not is_backward_correct:
        print('Model loading and forward pass is correct, but not the backward and update pass')
    return is_forward_loading_correct and is_backward_correct

def transform_weight(w, use_constant_weights=False):
    return np.ones(w.shape)*0.1 if use_constant_weights else stats.zscore(w)

def set_weights(model):
    for l in range(len(model.layers)):
        if model.layers[l].get_weights() is not None or len(model.layers[l].get_weights()) > 0:
            model.layers[l].set_weights([transform_weight(np.arange(elem.size).reshape(elem.shape) + 1) for elem in
                                         model.layers[l].get_weights()])

def get_model(layers, add_flatten, add_dense):
    keras_model = Sequential()
    for layer in layers:
        keras_model.add(layer)
    if add_flatten:
        keras_model.add(Flatten())
    if add_dense:
        keras_model.add(Dense(num_labels, activation='softmax'))
    else:
        keras_model.add(Activation('softmax'))
    # Test it with simplest optimizer setting:
    keras_model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=1, decay=0, momentum=0, nesterov=False))
    set_weights(keras_model)
    return keras_model

class TestNNLibrary(unittest.TestCase):
    def run_test(self, layers, is2D=False, rtol=1e-05, atol=1e-08):
        if not isinstance(layers, list):
            layers = [layers]
        shape = (input_shape2d[0], input_shape2d[1], 1) if is2D else input_shape
        keras_model = get_model(layers, True, False)
        simple_layer_test = are_predictions_all_close(keras_model, shape, is2D=is2D, rtol=rtol, atol=atol)
        #simple_layer_test = True  # TODO
        #keras_model = get_model(layers, True, True)
        #classify_layer_test = are_predictions_all_close(keras_model, shape, is2D=is2D, rtol=rtol, atol=atol)
        classify_layer_test = True
        self.failUnless(simple_layer_test and classify_layer_test)

    def test_conv2d(self):
        self.run_test(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

if __name__ == '__main__':
    unittest.main()
