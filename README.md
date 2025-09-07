# Performance Comparison: Matrix Multiplication

## Execution Times

| Implementation                | Optimization | Transpose | Time (s)  |
|-------------------------------|--------------|-----------|-----------|
| C (manual multiplication)     | O0           | No        | ~130.0000 |
| C (manual multiplication)     | O3           | No        |  ~21.0000 |
| C (manual multiplication)     | O3           | Yes       |  ~18.5000 |
| Python NumPy                  | -            | -         |    0.2286 |
| Python PyTorch (CPU)          | -            | -         |    0.2424 |
| Python TensorFlow (CPU)       | -            | -         |    0.2285 |
| Python JAX (CPU)              | -            | -         |    0.1015 |


## Further Development and Optimization

Further improvement of matrix multiplication performance can be achieved using:

- **Cache-aware algorithms**
- **Divide-and-conquer algorithms** 
- **Sub-cubic algorithms**
- **SIMD operations**
- **Multithreading and parallelization**  

More information can be found here: [Matrix multiplication algorithm](https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm)

