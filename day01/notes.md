# Technical Notes: Vectors, Matrices, and Tensors

## Dimensionality
* **Scalar** (0D Tensor): Single number. e.g., `x = 5.0`
* **Vector** (1D Tensor): 1D array of numbers. e.g., `v = [1.0, 2.0, 3.0]`
* **Matrix** (2D Tensor): 2D array of numbers. e.g., `M = [[1, 2], [3, 4]]`
* **Tensor** (ND Tensor): ND array of numbers.

## Memory Layouts
Computers store data linearly. A 2D matrix:
```
[[1, 2],
 [3, 4]]
```
Is stored in memory as:
* **Row-Major (C style)**: `[1, 2, 3, 4]` (contiguous rows)
* **Column-Major (Fortran style)**: `[1, 3, 2, 4]` (contiguous columns)

## Indexing Formula
For a 2D tensor with strides `(S0, S1)` and index `(i, j)`:
$$\text{Offset} = i \times S0 + j \times S1$$
