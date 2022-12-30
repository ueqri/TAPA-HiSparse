# HiSparse Multi-HBM Sparse Matrix-Sparse Vector Multiplication

Sparse Matrix-Sparse Vector Multiplication (SpMSpV) exploits the sparsity in the input vector and only loads the necessary columns of the sparse matrix.

We accelerated SpMSpV on HBM-Equipped FPGAs using the following architecture and achieved ~5X speedup than FPGA'22 HiSparse SpMV accelerator when vector sparsity is above 99.9% (which was true in the first few iterations of many graph applications).

![spmspv_xcel_arch](https://user-images.githubusercontent.com/56567688/210109265-dc0ee806-ed7e-422f-b1df-86cf6f2f88d3.svg)

### Vector Loader

Load the *sparse* vector from device HBM memory, and feed the loaded element into a certain matrix loader (determined by the column index of the element).

### Matrix Loader

Use the column index sent by Vector Loader to load the required columns of sparse matrix from off-chip HBM memory via AXI4 Master interface, and assign the payloads to the merge unit.

### Input Merge Unit

Merge all the payload streams from Matrix Loaders (only some are busy generating payloads since Vector Loader only feeds certain Matrix Loaders at a time), and forward a fixed number of payloads to the Shuffle Unit for further processing.

### Shuffle Unit

Re-order the payloads (i.e., matrix non-zeros) transferred from the Input Merge Unit and then assign them to PEs. It resolves the bank conflicts on the PE output buffers in a non-blocking way.

### Processing Engine (PE)

Multiply the incoming non-zero value from the matrix with the corresponding vector value, and accumulate the product to the output buffer location indicated by the row index. It is fully pipelined with load-store forwarding (introduced in FPGA'22 HiSparse).

### Result Drain Unit

Collect outputs from the PEs and write the results into the device HBM memory.
