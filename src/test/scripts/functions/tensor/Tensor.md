# Initial prototype

It is important to note that lot of assumptions and hacks have been baked into this prototype for rapid prototyping, often at cost of performance. So, pretty please, donot cite or benchmark against this fork as it is not guaranteed that this will ever be integrated in SystemML. Once the feasibility study is done and after detailed discussion in SystemML forum, we will start our work on Deep Learning.

In this prototype, a dense tensor is stored internally as a dense matrix in a row-major format, where first dimension of tensor and matrix are exactly the same. Hence, a tensor with shape [W, X, Y, Z]  is stored internally as a matrix with dimension (W, X * Y * Z). This translation is done at parser level and no engine support has been added. So, if one tries to use non-element-wise tensor algebra, he/she may get incorrect result.

A restricted indexing is supported in this prototype. For example:
```sh
A=tensor("1 2 3 4 5 6 7 8 9 10 11 12", shape=[3, 2, 2]) # A = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape((3, 2, 2))
B=A[1:2, ] 								# B = A[0:2, :, :]
```

The images are assumed to be stored NCHW format, where N = batch size, C = #channels, H = height of image and W = width of image. Hence, the images are internally represented as a matrix with dimension (N, C * H * W).

This prototype also contains initial implementation of forward/backward functions for 2D convolution and pooling:
* `conv2d(x, w, ...)`
* `conv2d_backward_filter(x, dout, ...)` and `conv2d_backward_data(w, dout, ...)`
* `max_pool(x, ...)`

The required arguments for all above functions are:
* stride=[stride_h, stride_w]
* padding=[pad_h, pad_W]
* input_shape=[numImages, numChannels, height_image, width_image]

The additional required argument for conv2d/conv2d_backward_filter/conv2d_backward_data functions is:
* filter_shape=[numFilters, numChannels, height_filter, width_filter]

The additional required argument for max_pool/avg_pool functions is:
* pool_size=[height_pool, width_pool]

The function `conv2d_backward_data(filter, dout, zero padding)` performs the operation `conv2d(dout, rotate_4Dtensor(filter), full padding)`
and `conv2d_backward_filter(x, dout)` performs `rotate_4Dtensor( conv2d(rotate_4Dtensor(x), rotate_4Dtensor(dout) )`.

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
# Example invocations:
# dx1 = conv2d(dout, rotated_filter, stride=[stride_h, stride_w], padding=[R-1, S-1], input_shape=[N, K, P, Q], filter_shape=[C, K, R, S])
# dx2 = conv2d_backward_data(filter, dout, stride=[stride_h, stride_w], padding=[pad_h, pad_w], input_shape=[N, C, H, W], filter_shape=[K, C, R, S])
``` 

For full padding, use pad_h = filter_height-1 and pad_w = filter_weight-1.