# Tiled Row Principal Component Analysis
A work-in-progress implementation of a distributed variant of traditional principal component analysis.

## Development Instructions
In the `build/` directory, copy your respective OS make.inc file into the root directory. Run make to compile.

Current TODO:
- Finish last sub-function of Tile QR
- Test sub-function Tile QR validation, scalability, etc
- Use OpenMP for piecing Tile QR sub-functions together in parallel
- Have MPI recognize each node as a single rank
- Research parallel partial SVD algorithms
- Tile Matrix Multiplication
