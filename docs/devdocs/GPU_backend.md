# Initial prototype for GPU backend

A GPU backend implements two important abstract classes:
* `org.apache.sysml.runtime.controlprogram.context.GPUContext`
* `org.apache.sysml.runtime.controlprogram.context.GPUPointer`

The GPUContext is responsible for GPU memory management and gets call-backs from SystemML's bufferpool on following methods:
*  `prepare(MatrixBlock mat, boolean isInput, boolean lock)`: This function should prepares GPUPointer for processing of GPUInstruction.
If the MatrixBlock is to be used as input, then it should copy the data from host to device. Locking the MatrixBlock ensures that it won't be evicted during the execution.
*  `remove(MatrixBlock mat)`: This method removes GPUPointers and related information from GPUContext.

## JCudaContext:
The current prototype supports Nvidia's CUDA libraries using JCuda wrapper. The implementation for the above classes can be found in:
* `org.apache.sysml.runtime.controlprogram.context.JCudaContext`
* `org.apache.sysml.runtime.controlprogram.context.JCudaPointer`

### Setup instructions for JCudaContext:

1. Install CUDA 7.5
2. Install CuDNN v4 from http://developer.download.nvidia.com/compute/redist/cudnn/v4/cudnn-7.0-win-x64-v4.0-prod.zip
3. Download JCuda binaries version 0.7.5b and JCudnn version 0.7.5. 
* For Windows: Copy the DLLs into C:\lib (or /lib) directory. Link: http://www.jcuda.org/downloads/downloads.html
* For Mac/Linux: TODO !! 