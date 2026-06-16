# Day 1: Vectors, Matrices, and Multi-Dimensional Tensors in Python

## 🎯 Goal of the Day
Understand how multi-dimensional arrays (tensors) are represented in memory and how to manipulate them in Python and NumPy. You will learn the difference between row-major and column-major memory layouts and implement a matrix multiplication from scratch.

## 🧠 Key Concepts
* **Scalar, Vector, Matrix, Tensor**: Dimensional hierarchy of data representations.
* **Memory Layout**: C-contiguous (row-major) vs. Fortran-contiguous (column-major) layouts.
* **Strides**: The number of bytes to step in each dimension when traversing a tensor.
* **Broadcasting**: Element-wise operations on arrays of different shapes.

## 📖 Mini Explanation
Tensors are stored as a contiguous 1D block of memory. To read a tensor of shape `(2, 3)` (a matrix with 2 rows and 3 columns), we map a multi-dimensional index `(i, j)` to a 1D index using **strides**.
For a row-major matrix containing 32-bit floats (4 bytes each), the strides are `(12, 4)`. Moving to the next row requires jumping 12 bytes (3 elements * 4 bytes), while moving to the next column requires jumping 4 bytes.

## 📝 Practical Exercise
In today's Jupyter Notebook, you will:
1. Initialize matrices as nested Python lists.
2. Implement standard matrix multiplication using nested loops.
3. Compare execution speeds between your scratch implementation and NumPy's vectorized `@` operator.

## ⚡ Optional Challenge
Calculate strides manually for a 3D tensor of shape `(2, 3, 4)` and implement a function `get_element(tensor_data, strides, indices)` to retrieve elements from a flattened 1D representation.

## 📚 Resources
* [NumPy Strides Explained](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.strides.html)
* [Essence of Linear Algebra - 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)

## 💬 Daily Reflection
* Why is vectorized computation in NumPy faster than Python loops?
* What happens to strides when you transpose a matrix?
