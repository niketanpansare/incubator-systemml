# Initial prototype

It is important to note that lot of assumptions and hacks have been baked into this prototype for rapid prototyping, often at cost of performance. So, pretty please, donot cite or benchmark against this fork as it is not guaranteed that this will ever be integrated in SystemML. Once the feasibility study is done and after detailed discussion in SystemML forum, we will start our work on Deep Learning.

In this prototype, a dense tensor is stored internally as a dense matrix in a row-major format, where first dimension of tensor and matrix are exactly the same. Hence, a tensor with shape [W, X, Y, Z]  is stored internally as a matrix with dimension (W, X * Y * Z). This translation is done at parser level and no engine support has been added. So, if one tries to use non-element-wise tensor algebra, he/she may get incorrect result.

A restricted indexing is supported in this prototype and since we donot have a deeper integration yet, the shape of tensor needs to be provided while indexing. For example:
```sh
A=tensor("1 2 3 4 5 6 7 8 9 10 11 12", shape=[3, 2, 2]) # A = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape((3, 2, 2))
B=A[1:2, shape=[3, 2, 2]] 								# B = A[0:2, :, :]
C=A[2:, shape=[3, 2, 2]]								# C = A[1:, :,:][0:1, :,:]
```

The images are assumed to be stored NCHW format, where N = batch size, C = #channels, H = height of image and W = width of image. Hence, the images are internally represented as a matrix with dimension (N, C * H * W).

This prototype also contains initial implementation of forward/backward functions for 2D convolution and pooling:
* conv2d(x, w, ...)
* conv2d_backward_filter(x, dout, ...) and conv2d_backward_data(w, dout, ...)
* max_pool2d(x, ...), avg_pool2d(x, ...)

The required arguments for all above functions are:
* stride=[stride_h, stride_w]
* padding=[pad_h, pad_W]
* input_shape=[numImages, numChannels, height_image, width_image]

The additional required argument for conv2d/conv2d_backward_filter/conv2d_backward_data functions is:
* filter_shape=[numFilters, numChannels, height_filter, width_filter]

The additional required argument for max_pool2d/avg_pool2d functions is:
* pool_size=[height_pool, width_pool]