````markdown
# Matrix Manipulation Library in C++

## Overview
This **Matrix Manipulation Library** provides a comprehensive set of functions for creating, displaying, and manipulating matrices in C++.  
It supports a wide range of operations such as arithmetic, LU decomposition, determinant calculation, inversion, trace, transpose, and eigenvalue/eigenvector computations.  
The library includes robust error handling and a user-friendly command-line interface (CLI) for interactive use.

---

## Features
- **Matrix Creation**: Create matrices of any numeric type (`int`, `double`, etc.) with specified dimensions and names.  
- **Matrix Display**: Print matrices to the console for easy visualization.  
- **Arithmetic Operations**: Perform addition, subtraction, multiplication, and element-wise multiplication.  
- **Advanced Operations**:
  - LU Decomposition  
  - Determinant Calculation  
  - Inverse Calculation  
  - Trace  
  - Transpose  
  - Rank  
  - Eigenvalues and Eigenvectors  
- **Error Handling**: Comprehensive error messages for invalid operations (e.g., non-square matrices for determinant or inverse).  
- **Interactive CLI**: A friendly command-line interface to perform matrix operations interactively.

---

## Getting Started

### Prerequisites
- A C++ Compiler (e.g., **GCC**)
- **Standard Template Library (STL)**

### Building the Library
Clone the repository:
```bash
git clone https://github.com/yourusername/matrix-manipulation-library.git
cd matrix-manipulation-library
````

Compile the code:

```bash
g++ -o matrix_library main.cpp matrix.cpp
```

Run the executable:

```bash
./matrix_library
```

---

## Usage

### Creating a Matrix

Use the interactive CLI to create a new matrix.
You will be prompted to specify:

* Number of rows
* Number of columns
* Matrix name

### Performing Operations

The CLI allows you to:

* **Unary Operations**: Operate on a single matrix (e.g., transpose, trace, LU decomposition).
* **Binary Operations**: Perform operations on two matrices (e.g., addition, subtraction, multiplication).
* **Matrix Print**: Display a stored matrix.

#### Example CLI Menu

```plaintext
1. Create a new matrix
2. Print a stored matrix
3. Unary Matrix Operations
4. Binary Matrix Operations
5. Addition of Array of Matrices
6. Exit
```

---

## Example Code

### Matrix Creation

```cpp
Matrix<int> matrixA(3, 3, "A");
matrixA.set(0, 0, 1);
matrixA.set(0, 1, 2);
matrixA.set(0, 2, 3);
// ... (populate other elements)
```

### Matrix Operations

```cpp
// Transpose
Matrix<int> transposed = transpose(matrixA);

// Determinant
double det = determinant(matrixA);

// LU Decomposition
auto [L, U] = luDecomposition(matrixA);
```

---

## Error Handling

The library includes **extensive error handling** to prevent invalid operations.
For example:

* Attempting to calculate the determinant of a non-square matrix
* Performing binary operations on mismatched matrix sizes

Such operations will result in descriptive error messages, ensuring safe and reliable matrix manipulations.

---
