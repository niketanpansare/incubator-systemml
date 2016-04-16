# Initial prototype for Deep Learning

Please note: until this fork is merged into trunk, this prototype does not represent SystemML's deep learning engine.
So, if you do benchmark against this fork, please DO NOT cite SystemML.

## Representing tensor and images in SystemML

In this prototype, we represent a tensor as a matrix stored in a row-major format,
where first dimension of tensor and matrix are exactly the same. For example, a tensor (with all zeros)
of shape [3, 2, 4, 5] can be instantiated by following DML statement:
```sh
A = matrix(0, rows=3, cols=2*4*5) 
```
### Tensor functions:

#### Element-wise arithmetic operators:
Following operators work out-of-the box when both tensors X and Y have same shape:

* Element-wise exponentiation: `X ^ Y`
* Element-wise unary minus: `-X`
* Element-wise integer division: `X %/% Y`
* Element-wise modulus operation: `X %% Y`
* Element-wise multiplication: `X * Y`
* Element-wise division: `X / Y`
* Element-wise addition: `X + Y`
* Element-wise subtraction: `X - Y`

SystemML does not support implicit broadcast for above tensor operations, however one can write a DML-bodied function to do so.
For example: to perform the above operations with broadcasting on second dimensions, one can use the below `rep(Z, n)` function:
``` python
rep = function(matrix[double] Z, int C) return (matrix[double] ret) {
	ret = Z
	parfor(i in 2:C) {
		ret = cbind(ret, Z)
	}
}
```
Using the above `rep(Z, n)` function, we can realize the element-wise arithmetic operation with broadcasting. Here are some examples:
* X of shape [N, C, H, W] and Y of shape [1, C, H, W]: `X + Y` (Note: SystemML does implicit broadcasting in this case because of the way 
it represents the tensor)
* X of shape [1, C, H, W] and Y of shape [N, C, H, W]: `X + Y` (Note: SystemML does implicit broadcasting in this case because of the way 
it represents the tensor)
* X of shape [N, C, H, W] and Y of shape [N, 1, H, W]: `X + rep(Y, C)`
* X of shape [N, C, H, W] and Y of shape [1, 1, H, W]: `X + rep(Y, C)`
* X of shape [N, 1, H, W] and Y of shape [N, C, H, W]: `rep(X, C) + Y`
* X of shape [1, 1, H, W] and Y of shape [N, C, H, W]: `rep(X, C) + Y`

TODO: Map the NumPy tensor calls to DML expressions.

## Representing images in SystemML

The images are assumed to be stored NCHW format, where N = batch size, C = #channels, H = height of image and W = width of image. 
Hence, the images are internally represented as a matrix with dimension (N, C * H * W).

## Convolution and Pooling built-in functions

This prototype also contains initial implementation of forward/backward functions for 2D convolution and pooling:
* `conv2d(x, w, ...)`
* `conv2d_backward_filter(x, dout, ...)` and `conv2d_backward_data(w, dout, ...)`
* `max_pool(x, ...)` and `max_pool_backward(x, dout, ...)`

The required arguments for all above functions are:
* stride=[stride_h, stride_w]
* padding=[pad_h, pad_w]
* input_shape=[numImages, numChannels, height_image, width_image]

The additional required argument for conv2d/conv2d_backward_filter/conv2d_backward_data functions is:
* filter_shape=[numFilters, numChannels, height_filter, width_filter]

The additional required argument for max_pool/avg_pool functions is:
* pool_size=[height_pool, width_pool]

### Border mode:
* To perform valid padding, use `padding = (input_shape-filter_shape)*(stride-1)/ 2`. (Hint: for stride length of 1, `padding = [0, 0]` performs valid padding).

* To perform full padding, use `padding = ((stride-1)*input_shape + (stride+1)*filter_shape - 2*stride) / 2`. (Hint: for stride length of 1, `padding = [filter_h-1, filter_w-1]` performs full padding).

* To perform same padding, use `padding = (input_shape*(stride-1) + filter_shape - stride)/2`. (Hint: for stride length of 1, `padding = [(filter_h-1)/2, (filter_w-1)/2]` performs same padding).

### Explanation of backward functions for conv2d

The function `conv2d_backward_data(filter, dout, zero padding)` performs the operation `conv2d(dout, rotate_4Dtensor(filter), full padding)`
and `conv2d_backward_filter(x, dout)` performs `flipFirst2Dim( conv2d(flipFirst2Dim(x), flipFirst2Dim(dout) )`. 
The results of these functions are consistent with Nvidia's CuDNN library.

The function `rotate_4Dtensor` can be implemented using following DML script:

``` python
rotate_matrix = function(matrix[double] filter) return (matrix[double] ret) {
        pc = table(seq(1, ncol(filter)), seq(ncol(filter), 1, -1))
        out1 = filter %*% pc
        pr = table(seq(1, ncol(filter)), seq(ncol(filter), 1, -1))
        ret = pr %*% out1
}

rotate_4Dtensor = function(matrix[double] filter, int dim1, int dim2, int dim3, int dim4) return (matrix[double] ret) {
        ret = matrix(0, rows=dim2, cols=dim1*dim3*dim4)
        for(k in 0:(dim1-1)) {
                for(c in 0:(dim2-1)) {
                        # ---------------------------------------------------
                        # Since the tensor is stored in row-major format, the indexing flips the first and second dimensions
                        outIndex1 = k*dim3*dim4
                        outIndex2 = (k+1)*dim3*dim4-1
                        inIndex1 = c*dim3*dim4
                        inIndex2 = (c+1)*dim3*dim4-1
                        # ---------------------------------------------------
                        # Now, extract one channel of one filter and rotate it
                        oneFilterMap = filter[k+1, (inIndex1+1):(inIndex2+1)]
                        rotatedOneFilterMap = rotate_matrix(oneFilterMap)
                        # ---------------------------------------------------
                        # Place the rotated filter map back
                        ret[c+1, (outIndex1+1):(outIndex2+1)] = rotatedOneFilterMap
                }
        }
}

flipFirst2Dim = function(matrix[double] filter, int dim1, int dim2, int dim3, int dim4) return (matrix[double] ret) {
        ret = matrix(0, rows=dim2, cols=dim1*dim3*dim4)
        for(k in 0:(dim1-1)) {
                for(c in 0:(dim2-1)) {
                        # ---------------------------------------------------
                        # Since the tensor is stored in row-major format, the indexing flips the first and second dimensions
                        outIndex1 = k*dim3*dim4
                        outIndex2 = (k+1)*dim3*dim4-1
                        inIndex1 = c*dim3*dim4
                        inIndex2 = (c+1)*dim3*dim4-1
                        # Place the rotated filter map back
                        ret[c+1, (outIndex1+1):(outIndex2+1)] = filter[k+1, (inIndex1+1):(inIndex2+1)]
                }
        }
}

# Example invocations:
rotated_filter = rotate_4Dtensor(filter, K, C, R, S)
dx1 = conv2d(dout, rotated_filter, stride=[stride_h, stride_w], padding=[R-1, S-1], input_shape=[N, K, P, Q], filter_shape=[C, K, R, S])
dx2 = conv2d_backward_data(filter, dout, stride=[stride_h, stride_w], padding=[pad_h, pad_w], input_shape=[N, C, H, W], filter_shape=[K, C, R, S])
# dx1 is same as dx2

r_x = flipFirst2Dim(x, N, C, H, W)
r_d = flipFirst2Dim(dout, N, K, P, Q)
out = conv2d(r_x, r_d, stride=[stride_h, stride_w], padding=[pad_h, pad_w], input_shape=[C, N, H, W], filter_shape=[K, N, P, Q])
dfilter1 = flipFirst2Dim(out, C, K, R, S)
dfilter2 = conv2d_backward_filter(x, dout, stride=[stride_h, stride_w], padding=[pad_h, pad_w], input_shape=[N, C, H, W], filter_shape=[K, C, R, S])
# dfilter1 is same as dfilter2
``` 