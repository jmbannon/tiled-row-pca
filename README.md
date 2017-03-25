# Tiled Row Principal Component Analysis
A work-in-progress implementation of a distributed variant of traditional principal component analysis.

## Development Instructions
In the `build/` directory, copy your respective OS make.inc file into the root directory. Run make to compile.

## TODO
### Initialization
1. Distribute the global matrix evenly amongst all compute nodes.
2. For every node n<sub>i</sub>, load its local matrix into GPU memory
### Matrix normalization
We must subtract the global column means from every column.
1. Compute local column means using CUDA
2. Compute global column means using ReduceAll on local column means using MPI
3. Load global column means into GPU memory, subtract global column means from every column.

### Computing global covariance matrix
1. Compute QR decomposition on the normalized, local matrix using cuBLAS. Disregard Q.
 - The second paper computes QR decomp. w/ +25% GLOPS and scales. To be implemented later
2. Perform the following Reduce function
 - Row-bind R from two nodes n</sub>i</sub>, n<sub>j</sub>
 - Compute QR decomp. on the row-binded matrices. Disregard Q.
3. Construct the covariance matrix on n<sub>0</sub> using method found in [1](http://www.math.cuhk.edu.hk/~rchan/paper/grid.pdf)

### Computing PCA
1. Compute SVD via cuBLAS on n<sub>0</sub>, disregard everything besides first k right singular vectors in V; the principal components
2. Broadcast principal components
3. Multiply original local matrix with principal components

