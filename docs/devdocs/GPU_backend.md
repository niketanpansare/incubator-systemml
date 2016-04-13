# Initial prototype for GPU backend

A GPU backend implements two important abstract classes:
* `org.apache.sysml.runtime.controlprogram.context.GPUContext`
* `org.apache.sysml.runtime.controlprogram.context.GPUPointer`

The GPUContext is responsible for GPU memory management and gets call-backs from SystemML's bufferpool on following methods:
*  `prepare(MatrixBlock mat, boolean isInput, boolean lock)`: This function should prepares GPUPointer for processing of GPUInstruction.
If the MatrixBlock is to be used as input, then it should copy the data from host to device. Locking the MatrixBlock ensures that it won't be evicted during the execution.
*  `remove(MatrixBlock mat)`: This method removes GPUPointers and related information from GPUContext.

The current prototype supports Nvidia's CUDA libraries using JCuda wrapper. The implementation for the above classes can be found in:
* `org.apache.sysml.runtime.controlprogram.context.JCudaContext`
* `org.apache.sysml.runtime.controlprogram.context.JCudaPointer`