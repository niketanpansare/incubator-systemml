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

# To run:
#   - Python 2: `PYSPARK_PYTHON=python2 spark-submit --master local[*] --driver-memory 10g --driver-class-path SystemML.jar,systemml-*-extra.jar test_nn_numpy.py`
#   - Python 3: `PYSPARK_PYTHON=python3 spark-submit --master local[*] --driver-memory 10g --driver-class-path SystemML.jar,systemml-*-extra.jar test_nn_numpy.py`

# Make the `systemml` package importable
import os
import sys
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
sys.path.insert(0, path)

import unittest

import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout,Flatten
from keras import backend as K
from keras.models import Model
from systemml.mllearn import Keras2DML
from pyspark.sql import SparkSession

batch_size = 32
input_shape = (3,64,64)
K.set_image_data_format("channels_first")
keras_tensor = np.random.rand(batch_size,input_shape[0], input_shape[1], input_shape[2])
sysml_matrix = keras_tensor.reshape((batch_size, -1))
tmp_dir = 'tmp_dir'

spark = SparkSession.builder.getOrCreate()

# Currently not integrated with JUnit test
# ~/spark-1.6.1-scala-2.11/bin/spark-submit --master local[*] --driver-class-path SystemML.jar test.py
class TestNNLibrary(unittest.TestCase):
    def test_conv2d(self):
        keras_model = Sequential()
        keras_model.add(Conv2D(32, kernel_size=(3, 3), activation='softmax', input_shape=input_shape, padding='valid'))
        sysml_model = Keras2DML(spark, keras_model, input_shape=input_shape, weights=tmp_dir)
        keras_preds = keras_model.predict(keras_tensor)
        print(str(keras_preds))
        keras_preds = keras_preds.flatten()
        sysml_preds = sysml_model.predict_proba(sysml_matrix)
        print(str(sysml_preds))
        sysml_preds = sysml_preds.flatten()
        self.failUnless(np.allclose(keras_preds, sysml_preds))

    
if __name__ == '__main__':
    unittest.main()
