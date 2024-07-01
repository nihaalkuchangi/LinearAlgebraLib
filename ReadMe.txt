Matrix Manipulation Library in C++
Overview
This Matrix Manipulation Library provides a comprehensive set of functions for creating, displaying, and manipulating matrices in C++. The library supports various operations such as arithmetic, LU decomposition, determinant calculation, inversion, trace, transpose, and eigenvalue/eigenvector computations. It includes robust error handling and a user-friendly command-line interface for interactive use.

Features
Matrix Creation: Easily create matrices of any numeric type (int, double, etc.) with specified dimensions and names.
Matrix Display: Print matrices to the console for easy visualization.
Arithmetic Operations: Perform addition, subtraction, multiplication, and element-wise multiplication.
Advanced Operations:
LU Decomposition
Determinant Calculation
Inverse Calculation
Trace
Transpose
Rank
Eigenvalues and Eigenvectors
Error Handling: Comprehensive error messages for invalid operations (e.g., non-square matrices for certain operations).
Interactive CLI: User-friendly command-line interface to perform matrix operations interactively.
Getting Started
Prerequisites
C++ Compiler (e.g., GCC)
Standard Template Library (STL)
Building the Library
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/matrix-manipulation-library.git
cd matrix-manipulation-library
Compile the code:

bash
Copy code
g++ -o matrix_library main.cpp matrix.cpp
Run the executable:

bash
Copy code
./matrix_library
Usage
Creating a Matrix
To create a new matrix, follow the prompts in the command-line interface. You will be asked to specify the number of rows, columns, and the name of the matrix.

Performing Operations
The CLI will guide you through various operations you can perform on the matrices:

Unary Operations: Operations on a single matrix, such as transpose, trace, LU decomposition, etc.
Binary Operations: Operations involving two matrices, such as addition, subtraction, and multiplication.
Matrix Print: Display a stored matrix.
Example CLI Menu
plaintext
Copy code
1. Create a new matrix
2. Print a stored matrix
3. Unary Matrix Operations
4. Binary Matrix Operations
5. Addition of Array of Matrices
6. Exit
Example Code
Matrix Creation
cpp
Copy code
Matrix<int> matrixA(3, 3, "A");
matrixA.set(0, 0, 1);
matrixA.set(0, 1, 2);
matrixA.set(0, 2, 3);
// ... (populate other elements)
Matrix Operations
cpp
Copy code
// Transpose
Matrix<int> transposed = transpose(matrixA);

// Determinant
double det = determinant(matrixA);

// LU Decomposition
auto [L, U] = luDecomposition(matrixA);
Error Handling
The library includes extensive error handling to ensure invalid operations are not performed. For example, attempting to calculate the determinant of a non-square matrix will result in an appropriate error message.
