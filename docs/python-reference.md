---
layout: global
title: Reference Guide for Python Users
description: Reference Guide for Python Users
---
<!--
{% comment %}
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements.  See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to you under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
{% endcomment %}
-->

* This will become a table of contents (this text will be scraped).
{:toc}

<br/>

## Introduction

SystemML enables flexible, scalable machine learning. This flexibility is achieved through the specification of a high-level declarative machine learning language that comes in two flavors, 
one with an R-like syntax (DML) and one with a Python-like syntax (PyDML).

Algorithm scripts written in DML and PyDML can be run on Hadoop, on Spark, or in Standalone mode. 
No script modifications are required to change between modes. SystemML automatically performs advanced optimizations 
based on data and cluster characteristics, so much of the need to manually tweak algorithms is largely reduced or eliminated.
To understand more about DML and PyDML, we recommend that you read [Beginner's Guide to DML and PyDML](https://apache.github.io/incubator-systemml/beginners-guide-to-dml-and-pydml.html).

For convenience of Python users, SystemML exposes several language-level APIs that allow Python users to use SystemML
and its algorithms without the need to know DML or PyDML. We explain these APIs in the below sections.

## matrix API

The matrix class allows users to perform linear algebra operations in SystemML using a NumPy-like interface.
This class supports several arithmetic operators (such as +, -, *, /, ^, etc).

matrix class is a python wrapper that implements basic matrix
operators, matrix functions as well as converters to common Python
types (for example: Numpy arrays, PySpark DataFrame and Pandas
DataFrame).

The operators supported are:

1.  Arithmetic operators: +, -, *, /, //, %, \** as well as dot
    (i.e. matrix multiplication)
2.  Indexing in the matrix
3.  Relational/Boolean operators: \<, \<=, \>, \>=, ==, !=, &, \|

In addition, following functions are supported for matrix:

1.  transpose
2.  Aggregation functions: sum, mean, var, sd, max, min, argmin,
    argmax, cumsum
3.  Global statistical built-In functions: exp, log, abs, sqrt,
    round, floor, ceil, sin, cos, tan, asin, acos, atan, sign, solve

For all the above functions, we always return a two dimensional matrix, especially for aggregation functions with axis. 
For example: Assuming m1 is a matrix of (3, n), NumPy returns a 1d vector of dimension (3,) for operation m1.sum(axis=1)
whereas SystemML returns a 2d matrix of dimension (3, 1).

Note: an evaluated matrix contains a data field computed by eval
method as DataFrame or NumPy array.

It is important to note that matrix class also supports most of NumPy's universal functions (i.e. ufuncs).
The current version of NumPy explicitly disables overriding ufunc, but this should be enabled in next release. 
Until then to test above code, please use:

```bash
git clone https://github.com/niketanpansare/numpy.git
cd numpy
python setup.py install
```

This will enable NumPy's functions to invoke matrix class:

```python
import systemml as sml
import numpy as np
m1 = sml.matrix(np.ones((3,3)) + 2)
m2 = sml.matrix(np.ones((3,3)) + 3)
np.add(m1, m2)
``` 

The matrix class doesnot support following ufuncs:

- Complex number related ufunc (for example: `conj`)
- Hyperbolic/inverse-hyperbolic functions (for example: sinh, arcsinh, cosh, ...)
- Bitwise operators
- Xor operator
- Infinite/Nan-checking (for example: isreal, iscomplex, isfinite, isinf, isnan)
- Other ufuncs: copysign, nextafter, modf, frexp, trunc.

This class also supports several input/output formats such as NumPy arrays, Pandas DataFrame, SciPy sparse matrix and PySpark DataFrame.

By default, the operations are evaluated lazily to avoid conversion overhead and also to maximize optimization scope.
To disable lazy evaluation, please us `set_lazy` method:

```python
>>> import systemml as sml
>>> import numpy as np
>>> m1 = sml.matrix(np.ones((3,3)) + 2)

Welcome to Apache SystemML!

>>> m2 = sml.matrix(np.ones((3,3)) + 3)
>>> np.add(m1, m2) + m1
# This matrix (mVar4) is backed by below given PyDML script (which is not yet evaluated). To fetch the data of this matrix, invoke toNumPy() or toDF() or toPandas() methods.
mVar2 = load(" ", format="csv")
mVar1 = load(" ", format="csv")
mVar3 = mVar1 + mVar2
mVar4 = mVar3 + mVar1
save(mVar4, " ")


>>> sml.set_lazy(False)
>>> m1 = sml.matrix(np.ones((3,3)) + 2)
>>> m2 = sml.matrix(np.ones((3,3)) + 3)
>>> np.add(m1, m2) + m1
# This matrix (mVar8) is backed by NumPy array. To fetch the NumPy array, invoke toNumPy() method.
``` 

### Usage:

```python
import systemml as sml
import numpy as np
m1 = sml.matrix(np.ones((3,3)) + 2)
m2 = sml.matrix(np.ones((3,3)) + 3)
m2 = m1 * (m2 + m1)
m4 = 1.0 - m2
m4.sum(axis=1).toNumPy()
```

Output:

```bash
array([[-60.],
       [-60.],
       [-60.]])
```


### Design Decisions:

1.  Until eval() method is invoked, we create an AST (not exposed to
    the user) that consist of unevaluated operations and data
    required by those operations. As an anology, a spark user can
    treat eval() method similar to calling RDD.persist() followed by
    RDD.count().
2.  The AST consist of two kinds of nodes: either of type matrix or
    of type DMLOp. Both these classes expose \_visit method, that
    helps in traversing the AST in DFS manner.
3.  A matrix object can either be evaluated or not. If evaluated,
    the attribute 'data' is set to one of the supported types (for
    example: NumPy array or DataFrame). In this case, the attribute
    'op' is set to None. If not evaluated, the attribute 'op' which
    refers to one of the intermediate node of AST and if of type
    DMLOp. In this case, the attribute 'data' is set to None.

5.  DMLOp has an attribute 'inputs' which contains list of matrix
    objects or DMLOp.

6.  To simplify the traversal, every matrix object is considered
    immutable and an matrix operations creates a new matrix object.
    As an example: m1 = sml.matrix(np.ones((3,3))) creates a matrix
    object backed by 'data=(np.ones((3,3))'. m1 = m1 \* 2 will
    create a new matrix object which is now backed by 'op=DMLOp( ...
    )' whose input is earlier created matrix object.

7.  Left indexing (implemented in \_\_setitem\_\_ method) is a
    special case, where Python expects the existing object to be
    mutated. To ensure the above property, we make deep copy of
    existing object and point any references to the left-indexed
    matrix to the newly created object. Then the left-indexed matrix
    is set to be backed by DMLOp consisting of following pydml:
    left-indexed-matrix = new-deep-copied-matrix
    left-indexed-matrix[index] = value

8.  Please use m.print\_ast() and/or type m for debugging. Here is a
    sample session:

        >>> npm = np.ones((3,3))
        >>> m1 = sml.matrix(npm + 3)
        >>> m2 = sml.matrix(npm + 5)
        >>> m3 = m1 + m2
        >>> m3
        mVar2 = load(" ", format="csv")
        mVar1 = load(" ", format="csv")
        mVar3 = mVar1 + mVar2
        save(mVar3, " ")
        >>> m3.print_ast()
        - [mVar3] (op).
          - [mVar1] (data).
          - [mVar2] (data).    


## MLContext API

The Spark MLContext API offers a programmatic interface for interacting with SystemML from Spark using languages such as Scala, Java, and Python. 
As a result, it offers a convenient way to interact with SystemML from the Spark Shell and from Notebooks such as Jupyter and Zeppelin.

### Usage

The below example demonstrates how to invoke the algorithm [scripts/algorithms/MultiLogReg.dml](https://github.com/apache/incubator-systemml/blob/master/scripts/algorithms/MultiLogReg.dml)
using Python [MLContext API](https://apache.github.io/incubator-systemml/spark-mlcontext-programming-guide).

```python
from sklearn import datasets, neighbors
from pyspark.sql import DataFrame, SQLContext
import systemml as sml
import pandas as pd
import os, imp
sqlCtx = SQLContext(sc)
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target + 1
n_samples = len(X_digits)
# Split the data into training/testing sets and convert to PySpark DataFrame
X_df = sqlCtx.createDataFrame(pd.DataFrame(X_digits[:.9 * n_samples]))
y_df = sqlCtx.createDataFrame(pd.DataFrame(y_digits[:.9 * n_samples]))
ml = sml.MLContext(sc)
# Get the path of MultiLogReg.dml
scriptPath = os.path.join(imp.find_module("systemml")[1], 'systemml-java', 'scripts', 'algorithms', 'MultiLogReg.dml')
script = sml.dml(scriptPath).input(X=X_df, Y_vec=y_df).output("B_out")
beta = ml.execute(script).get('B_out').toNumPy()
```


## mllearn API

### Usage

```python
# Scikit-learn way
from sklearn import datasets, neighbors
from systemml.mllearn import LogisticRegression
from pyspark.sql import SQLContext
sqlCtx = SQLContext(sc)
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target 
n_samples = len(X_digits)
X_train = X_digits[:.9 * n_samples]
y_train = y_digits[:.9 * n_samples]
X_test = X_digits[.9 * n_samples:]
y_test = y_digits[.9 * n_samples:]
logistic = LogisticRegression(sqlCtx)
print('LogisticRegression score: %f' % logistic.fit(X_train, y_train).score(X_test, y_test))
```

Output:

```bash
LogisticRegression score: 0.922222
```

### Reference documentation

 *class*`systemml.mllearn.estimators.LinearRegression`(*sqlCtx*, *fit\_intercept=True*, *normalize=False*, *max\_iter=100*, *tol=1e-06*, *C=float("inf")*, *solver='newton-cg'*, *transferUsingDF=False*)(#systemml.mllearn.estimators.LinearRegression "Permalink to this definition")
:   Bases: `systemml.mllearn.estimators.BaseSystemMLRegressor`{.xref .py
    .py-class .docutils .literal}

    Performs linear regression to model the relationship between one
    numerical response variable and one or more explanatory (feature)
    variables.

        >>> import numpy as np
        >>> from sklearn import datasets
        >>> from systemml.mllearn import LinearRegression
        >>> from pyspark.sql import SQLContext
        >>> # Load the diabetes dataset
        >>> diabetes = datasets.load_diabetes()
        >>> # Use only one feature
        >>> diabetes_X = diabetes.data[:, np.newaxis, 2]
        >>> # Split the data into training/testing sets
        >>> diabetes_X_train = diabetes_X[:-20]
        >>> diabetes_X_test = diabetes_X[-20:]
        >>> # Split the targets into training/testing sets
        >>> diabetes_y_train = diabetes.target[:-20]
        >>> diabetes_y_test = diabetes.target[-20:]
        >>> # Create linear regression object
        >>> regr = LinearRegression(sqlCtx, solver='newton-cg')
        >>> # Train the model using the training sets
        >>> regr.fit(diabetes_X_train, diabetes_y_train)
        >>> # The mean square error
        >>> print("Residual sum of squares: %.2f" % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))

 *class*`systemml.mllearn.estimators.LogisticRegression`(*sqlCtx*, *penalty='l2'*, *fit\_intercept=True*, *normalize=False*,  *max\_iter=100*, *max\_inner\_iter=0*, *tol=1e-06*, *C=1.0*, *solver='newton-cg'*, *transferUsingDF=False*)(#systemml.mllearn.estimators.LogisticRegression "Permalink to this definition")
:   Bases: `systemml.mllearn.estimators.BaseSystemMLClassifier`{.xref
    .py .py-class .docutils .literal}

    Performs both binomial and multinomial logistic regression.

    Scikit-learn way

        >>> from sklearn import datasets, neighbors
        >>> from systemml.mllearn import LogisticRegression
        >>> from pyspark.sql import SQLContext
        >>> sqlCtx = SQLContext(sc)
        >>> digits = datasets.load_digits()
        >>> X_digits = digits.data
        >>> y_digits = digits.target + 1
        >>> n_samples = len(X_digits)
        >>> X_train = X_digits[:.9 * n_samples]
        >>> y_train = y_digits[:.9 * n_samples]
        >>> X_test = X_digits[.9 * n_samples:]
        >>> y_test = y_digits[.9 * n_samples:]
        >>> logistic = LogisticRegression(sqlCtx)
        >>> print('LogisticRegression score: %f' % logistic.fit(X_train, y_train).score(X_test, y_test))

    MLPipeline way

        >>> from pyspark.ml import Pipeline
        >>> from systemml.mllearn import LogisticRegression
        >>> from pyspark.ml.feature import HashingTF, Tokenizer
        >>> from pyspark.sql import SQLContext
        >>> sqlCtx = SQLContext(sc)
        >>> training = sqlCtx.createDataFrame([
        >>>     (0L, "a b c d e spark", 1.0),
        >>>     (1L, "b d", 2.0),
        >>>     (2L, "spark f g h", 1.0),
        >>>     (3L, "hadoop mapreduce", 2.0),
        >>>     (4L, "b spark who", 1.0),
        >>>     (5L, "g d a y", 2.0),
        >>>     (6L, "spark fly", 1.0),
        >>>     (7L, "was mapreduce", 2.0),
        >>>     (8L, "e spark program", 1.0),
        >>>     (9L, "a e c l", 2.0),
        >>>     (10L, "spark compile", 1.0),
        >>>     (11L, "hadoop software", 2.0)
        >>> ], ["id", "text", "label"])
        >>> tokenizer = Tokenizer(inputCol="text", outputCol="words")
        >>> hashingTF = HashingTF(inputCol="words", outputCol="features", numFeatures=20)
        >>> lr = LogisticRegression(sqlCtx)
        >>> pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
        >>> model = pipeline.fit(training)
        >>> test = sqlCtx.createDataFrame([
        >>>     (12L, "spark i j k"),
        >>>     (13L, "l m n"),
        >>>     (14L, "mapreduce spark"),
        >>>     (15L, "apache hadoop")], ["id", "text"])
        >>> prediction = model.transform(test)
        >>> prediction.show()

 *class*`systemml.mllearn.estimators.SVM`(*sqlCtx*, *fit\_intercept=True*, *normalize=False*, *max\_iter=100*, *tol=1e-06*, *C=1.0*, *is\_multi\_class=False*, *transferUsingDF=False*)(#systemml.mllearn.estimators.SVM "Permalink to this definition")
:   Bases: `systemml.mllearn.estimators.BaseSystemMLClassifier`{.xref
    .py .py-class .docutils .literal}

    Performs both binary-class and multiclass SVM (Support Vector
    Machines).

        >>> from sklearn import datasets, neighbors
        >>> from systemml.mllearn import SVM
        >>> from pyspark.sql import SQLContext
        >>> sqlCtx = SQLContext(sc)
        >>> digits = datasets.load_digits()
        >>> X_digits = digits.data
        >>> y_digits = digits.target 
        >>> n_samples = len(X_digits)
        >>> X_train = X_digits[:.9 * n_samples]
        >>> y_train = y_digits[:.9 * n_samples]
        >>> X_test = X_digits[.9 * n_samples:]
        >>> y_test = y_digits[.9 * n_samples:]
        >>> svm = SVM(sqlCtx, is_multi_class=True)
        >>> print('LogisticRegression score: %f' % svm.fit(X_train, y_train).score(X_test, y_test))

 *class*`systemml.mllearn.estimators.NaiveBayes`(*sqlCtx*, *laplace=1.0*, *transferUsingDF=False*)(#systemml.mllearn.estimators.NaiveBayes "Permalink to this definition")
:   Bases: `systemml.mllearn.estimators.BaseSystemMLClassifier`{.xref
    .py .py-class .docutils .literal}

    Performs Naive Bayes.

        >>> from sklearn.datasets import fetch_20newsgroups
        >>> from sklearn.feature_extraction.text import TfidfVectorizer
        >>> from systemml.mllearn import NaiveBayes
        >>> from sklearn import metrics
        >>> from pyspark.sql import SQLContext
        >>> sqlCtx = SQLContext(sc)
        >>> categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
        >>> newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
        >>> newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
        >>> vectorizer = TfidfVectorizer()
        >>> # Both vectors and vectors_test are SciPy CSR matrix
        >>> vectors = vectorizer.fit_transform(newsgroups_train.data)
        >>> vectors_test = vectorizer.transform(newsgroups_test.data)
        >>> nb = NaiveBayes(sqlCtx)
        >>> nb.fit(vectors, newsgroups_train.target)
        >>> pred = nb.predict(vectors_test)
        >>> metrics.f1_score(newsgroups_test.target, pred, average='weighted')


## Utility classes (used internally)

### systemml.classloader 

 `systemml.classloader.createJavaObject`(*sc*, *obj\_type*)[](#systemml.classloader.createJavaObject "Permalink to this definition")
:   Performs appropriate check if SystemML.jar is available and returns
    the handle to MLContext object on JVM

    sc: SparkContext
    :   SparkContext

    obj\_type: Type of object to create ('mlcontext' or 'dummy')

### systemml.converters

 `systemml.converters.getNumCols`(*numPyArr*)[](#systemml.converters.getNumCols "Permalink to this definition")
:   

 `systemml.converters.convertToMatrixBlock`(*sc*, *src*)[](#systemml.converters.convertToMatrixBlock "Permalink to this definition")
:   

 `systemml.converters.convertToNumPyArr`(*sc*, *mb*)[](#systemml.converters.convertToNumPyArr "Permalink to this definition")
:   

 `systemml.converters.convertToPandasDF`(*X*)[](#systemml.converters.convertToPandasDF "Permalink to this definition")
:   

 `systemml.converters.convertToLabeledDF`(*sqlCtx*, *X*, *y=None*)[](#systemml.converters.convertToLabeledDF "Permalink to this definition")
:  

### Other classes from systemml.defmatrix

 *class*`systemml.defmatrix.DMLOp`(*inputs*, *dml=None*)[](#systemml.defmatrix.DMLOp "Permalink to this definition")
:   Bases: `object`{.xref .py .py-class .docutils .literal}

    Represents an intermediate node of Abstract syntax tree created to
    generate the PyDML script


## Troubleshooting Python APIs

#### Unable to load SystemML.jar into current pyspark session.

While using SystemML's Python package through pyspark or notebook (SparkContext is not previously created in the session), the
below method is not required. However, if the user wishes to use SystemML through spark-submit and has not previously invoked 

 `systemml.defmatrix.setSparkContext`(*sc*)
:   Before using the matrix, the user needs to invoke this function if SparkContext is not previously created in the session.

    sc: SparkContext
    :   SparkContext

Example:

```python
import systemml as sml
import numpy as np
sml.setSparkContext(sc)
m1 = sml.matrix(np.ones((3,3)) + 2)
m2 = sml.matrix(np.ones((3,3)) + 3)
m2 = m1 * (m2 + m1)
m4 = 1.0 - m2
m4.sum(axis=1).toNumPy()
```

If SystemML was not installed via pip, you may have to download SystemML.jar and provide it to pyspark via `--driver-class-path` and `--jars`. 

#### matrix API is running slow when set_lazy(False) or when eval() is called often.

This is a known issue. The matrix API is slow in this scenario due to slow Py4J conversion from Java MatrixObject or Java RDD to Python NumPy or DataFrame.
To resolve this for now, we recommend writing the matrix to FileSystemML and using `load` function.

#### maximum recursion depth exceeded

SystemML matrix is backed by lazy evaluation and uses a recursive Depth First Search (DFS).
Python can throw `RuntimeError: maximum recursion depth exceeded` when the recursion of DFS exceeds beyond the limit 
set by Python. There are two ways to address it:

1. Increase the limit in Python:
 
	```python
	import sys
	some_large_number = 2000
	sys.setrecursionlimit(some_large_number)
	```

2. Evaluate the intermeditate matrix to cut-off large recursion.