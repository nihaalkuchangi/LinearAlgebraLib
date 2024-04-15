#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
using namespace std;

template <typename T>
struct Matrix
{
    int rows;
    int cols;
    vector<T> data;
    template <typename U>
    friend void displayMatrix(Matrix<U> &matrix);
    // Constructor with input validation
    Matrix(int r, int c) : rows(r), cols(c)
    {
        if (r <= 0 || c <= 0)
        {
            throw invalid_argument("Matrix dimensions must be positive.");
        }
        data.resize(r * c); // Allocate memory for data
    }

    // Function to get element at a specific row and column
T& get(int i, int j) {
  if (i < 0 || i >= rows || j < 0 || j >= cols) {
    throw out_of_range("Index out of bounds for matrix access.");
  }
  // Return a reference to the element, allowing modification
  return data[i * cols + j];
}

    // Function to set element at a specific row and column
    void set(int i, int j, const T &value)
    {
        get(i, j) = value;
    }

    // Function to take input values from the user
   void takeInput() {
    cout << "Enter elements for the matrix:" << endl;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            while (!(cin >> data[i * cols + j])) {
                // Clear the error state from cin
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n'); // Discard invalid input
                cout << "Invalid input. Please enter a number: ";
            }
        }
    }
}

    template <typename K>
    static T determinant2x2(Matrix<K> &matrix)
    {
        return matrix.get(0, 0) * matrix.get(1, 1) - matrix.get(0, 1) * matrix.get(1, 0);
    }
};

// Function to create a matrix of user-specified type and dimensions
#include <iostream>
#include <limits>

template <typename T>
Matrix<T> createMatrix() {
    int rows, cols;

    while (true) {
        cout << "Enter the number of rows and columns for the matrix:" << endl;
        cout << "Rows:";

        if (!(cin >> rows)) {
            // Clear the input stream
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cout << "Invalid input. Please enter a positive integer for rows." << endl;
            continue;
        }

        if (rows <= 0) {
            cout << "Invalid input. Please enter a positive integer for rows." << endl;
            continue;
        }

        cout << "Cols:";

        if (!(cin >> cols)) {
            // Clear the input stream
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cout << "Invalid input. Please enter a positive integer for columns." << endl;
            continue;
        }

        if (cols <= 0) {
            cout << "Invalid input. Please enter a positive integer for columns." << endl;
            continue;
        }

        break; // Exit the loop if valid input is received
    }

    Matrix<T> matrix(rows, cols);
    matrix.takeInput();
    cout << "The Inputed Matrix is:" << endl;
    displayMatrix(matrix);
    return matrix;
}

template <typename T>
void displayMatrix(Matrix<T> &matrix)
{
    // Print top border
    for (int j = 0; j < matrix.cols; ++j)
    {
        cout << "+---";
    }
    cout << "+" << endl;

    // Print each row with elements and separators
    for (int i = 0; i < matrix.rows; ++i)
    {
        cout << "| ";
        for (int j = 0; j < matrix.cols; ++j)
        {
            cout << matrix.get(i, j) << " | ";
        }
        cout << endl;

        // Print separator for each row (except the last)
        if (i < matrix.rows - 1)
        {
            for (int j = 0; j < matrix.cols; ++j)
            {
                cout << "+---";
            }
            cout << "+" << endl;
        }
    }

    // Print bottom border
    for (int j = 0; j < matrix.cols; ++j)
    {
        cout << "+---";
    }
    cout << "+" << endl;
}

template <typename T>
bool isColumnVector(const Matrix<T>& matrix) {
  return matrix.cols == 1;
}


template <typename T>
T norm(Matrix<T>& matrix) {
  if (!isColumnVector(matrix)) {
    throw invalid_argument("norm function is only defined for column vectors");
  }

  T sum = 0;
  for (int i = 0; i < matrix.rows; ++i) {
    auto x = matrix.get(i,0);
    sum += (x * x); // Square each element and add to the sum
  }

  return sqrt(sum); // Return the square root of the sum (Euclidean norm)
}

template <typename T>
T vectorDotProduct(Matrix<T>& matrix1,Matrix<T>& matrix2) {
  if (matrix1.rows != matrix2.rows) {
    throw invalid_argument("Matrices must have the same number of rows for dot product");
  }

  T product = 0;
  for (int i = 0; i < matrix1.rows; ++i) {
    auto x = matrix1.get(i,0);
    product += x*x; // Multiply corresponding elements and add to the product
  }

  return product;
}

// Function to calculate the trace of a matrix
template <typename T>
void matrixTrace(Matrix<T> &matrix)
{
    // Check if the matrix is square
    if (matrix.rows != matrix.cols)
    {
        cout << "Trace is only defined for square matrices." << endl;
        return;
    }

    // Calculate the trace
    T sum = 0;
    for (int i = 0; i < matrix.rows; ++i)
    {
        sum += matrix.get(i, i);
    }

    // Print the trace
    cout << "Trace of the matrix: " << sum << endl;
}

// Function to calculate the mean of each column and store them in an array
template <typename T>
void calculateColumnMeans(Matrix<T> &matrix)
{
    // Create a 1xcols matrix to store column means
    Matrix<double> means(1, matrix.cols);

    // Calculate sum of each column
    for (int i = 0; i < matrix.rows; ++i)
    {
        for (int j = 0; j < matrix.cols; ++j)
        {
            means.get(0, j) += matrix.get(i, j); // Access element in means matrix
        }
    }

    // Calculate average for each column
    for (int j = 0; j < matrix.cols; ++j)
    {
        means.get(0, j) /= matrix.rows;
    }

    cout << "Means of each column in the matrix:" << endl;
    displayMatrix(means);
}

// Function to find the transpose of a matrix
template <typename T>
void transpose(Matrix<T> &matrix)
{
    // Create a new matrix with swapped dimensions (rows become columns and vice versa)
    Matrix<T> transposed(matrix.cols, matrix.rows);
    // Fill the transposed matrix by swapping elements
    for (int i = 0; i < matrix.rows; ++i)
    {
        for (int j = 0; j < matrix.cols; ++j)
        {
            transposed.set(j, i, matrix.get(i, j));
        }
    }

    cout << "The Transpose of the given Matrix is:" << endl;
    displayMatrix(transposed);
}

// Custom implementation of magnitude function for doubles
template <typename T>
T magnitude(const T &value)
{
    return value >= 0 ? value : -value;
}

// Custom implementation of row_swap function
template <typename T>
void row_swap(T &a, T &b)
{
    T temp = a;
    a = b;
    b = temp;
}// Function to calculate the inverse of a matrix
template <typename T>
Matrix<T> matrixInverse(Matrix<T>& matrix) {
  // Check if the matrix is square
  if (matrix.rows != matrix.cols) {
    throw invalid_argument("Matrix inverse is only defined for square matrices.");
  }

  int n = matrix.rows;

  // Check if the matrix is singular by calculating its determinant
  auto det = determinant(matrix);
  if (det == 0) {
    throw invalid_argument("The input matrix is singular, and its inverse does not exist.");
  }

  // Perform LU decomposition of the matrix
  pair<Matrix<T>, Matrix<T>> lu = luDecomposition(matrix);
  Matrix<T> L = lu.first;
  Matrix<T> U = lu.second;

  // Create an identity matrix with the same dimensions as the input matrix
  Matrix<T> identity(n, n);
  for (int i = 0; i < n; ++i) {
    identity.set(i, i, 1);
  }

  // Solve the system of equations L * y = identity for each column of the identity matrix
  // This gives us the columns of the inverse matrix
  Matrix<T> inverse(n, n);
  for (int j = 0; j < n; ++j) {
    Matrix<T> y(n, 1); // Create a column vector to store the solution

    // Forward substitution
    for (int i = 0; i < n; ++i) {
      T sum = 0;
      // Check for division by zero in U.get(i, i) before division
      if (abs(U.get(i, i)) < std::numeric_limits<T>::epsilon()) {
        throw invalid_argument("Matrix is nearly singular, inverse calculation may be inaccurate.");
      }
      for (int k = 0; k < i; ++k) {
        sum += L.get(i, k) * y.get(k, 0);
      }
      y.set(i, 0, identity.get(i, j) - sum);
    }

    // Backward substitution
    for (int i = n - 1; i >= 0; --i) {
      T sum = 0;
      // Check for division by zero in U.get(i, i) before division
      if (abs(U.get(i, i)) < std::numeric_limits<T>::epsilon()) {
        throw invalid_argument("Matrix is nearly singular, inverse calculation may be inaccurate.");
      }
      for (int k = i + 1; k < n; ++k) {
        sum += U.get(i, k) * inverse.get(k, j);
      }
      inverse.set(i, j, (y.get(i, 0) - sum) / U.get(i, i));
    }
  }

  return inverse;
}


// Function to perform LU decomposition of a matrix
template <typename T>
pair<Matrix<T>, Matrix<T>> luDecomposition(Matrix<T>& matrix) {
    // Check if the matrix is square
    if (matrix.rows != matrix.cols) {
        throw invalid_argument("LU decomposition is only defined for square matrices.");
    }

    // Create matrices for L and U with the same dimensions as the input matrix
    int n = matrix.rows;
    Matrix<T> lower(n, n);
    Matrix<T> upper(n, n);

    // Decomposing matrix into Upper and Lower triangular matrix
    for (int i = 0; i < n; ++i) {
        // Upper Triangular
        for (int k = i; k < n; ++k) {
            T sum = 0.0;
            for (int j = 0; j < i; ++j) {
                sum += lower.get(i, j) * upper.get(j, k);
            }
            upper.set(i, k, matrix.get(i, k) - sum);
        }

        // Lower Triangular
        for (int k = i; k < n; ++k) {
            if (i == k) {
                lower.set(i, i, 1.0);  // Diagonal as 1
            } else {
                T sum = 0.0;
                for (int j = 0; j < i; ++j) {
                    sum += lower.get(k, j) * upper.get(j, i);
                }
                lower.set(k, i, (matrix.get(k, i) - sum) / upper.get(i, i));
            }
        }
    }

    return pair<Matrix<T>, Matrix<T>>(lower, upper);
}

// Function to perform LU decomposition

template <typename T>
int matrixRank(Matrix<T> &matrix)
{
    Matrix<T> temp(matrix); // Create a copy of the matrix to avoid modifying the original

    int rank = 0;
    int n = temp.rows;
    int m = temp.cols;
    int x = n<m?n:m;
    for (int col = 0; col < x; ++col)
    {
        int pivotRow = col;

        // Find the pivot element (largest element in magnitude in the current column below the diagonal)
        for (int i = col + 1; i < n; ++i)
        {
            if (magnitude(temp.get(i, col)) > magnitude(temp.get(pivotRow, col)))
            {
                pivotRow = i;
            }
        }

        // If the pivot element is zero, the column is linearly dependent and the rank doesn't increase
        if (magnitude(temp.get(pivotRow, col)) == 0)
        {
            continue;
        }

        // Swap rows if necessary to ensure a non-zero pivot
        if (pivotRow != col)
        {
            for (int j = 0; j < m; ++j)
            {
                row_swap(temp.get(col, j), temp.get(pivotRow, j));
            }
        }

        // Eliminate elements below the diagonal in the current column using row operations
        for (int i = col + 1; i < n; ++i)
        {
            T factor = temp.get(i, col) / temp.get(col, col);
            for (int j = col + 1; j < m; ++j)
            {
                temp.set(i, j, temp.get(i, j) - factor * temp.get(col, j));
            }
        }

        rank++; // Increment rank if a non-zero pivot element was found
    }

    return rank;
}
template <typename T>
std::pair<Matrix<T>, Matrix<T>> eigenValuesAndVectors(Matrix<T>& matrix) {
  // Check if the matrix is square
  if (matrix.rows != matrix.cols) {
    throw invalid_argument("Eigenvalues and eigenvectors are only defined for square matrices.");
  }

  int n = matrix.rows;

  // Choose an initial guess vector (e.g., the first column of the matrix)
  Matrix<T> x(n, 1);
  for (int i = 0; i < n; ++i) {
    x.set(i, 0, matrix.get(i, 0));
  }

  // Iterate until convergence (eigenvalue estimate stabilizes)
  int max_iterations = 100;
  T eigenvalue_old, eigenvalue_new;
  for (int iter = 0; iter < max_iterations; ++iter) {
    // Perform matrix-vector multiplication (y = A * x)
    Matrix<T> y(n, 1);
    for (int i = 0; i < n; ++i) {
      T sum = 0;
      for (int j = 0; j < n; ++j) {
        sum += matrix.get(i, j) * x.get(j, 0);
      }
      y.set(i, 0, sum);
    }

    // Normalize the resulting vector (x = y / ||y||)
    T norm_value = norm(y); // Call the external norm function
    for (int i = 0; i < n; ++i) {
      x.set(i, 0, y.get(i, 0) / norm_value);
    }

    // Calculate the Rayleigh quotient (estimate of the eigenvalue)
    eigenvalue_new = vectorDotProduct(matrix, x); // Call the external vectorDotProduct function

    // Check for convergence (significant change in eigenvalue estimate)
    if (iter > 0 && abs(eigenvalue_new - eigenvalue_old) / abs(eigenvalue_old) < 1e-6) {
      break;
    }

    eigenvalue_old = eigenvalue_new;
  }

  // Create a matrix to store the eigenvector
  Matrix<T> eigenvector(n, 1);
  for (int i = 0; i < n; ++i) {
    eigenvector.set(i, 0, x.get(i, 0));
  }

  // Create a result pair to hold eigenvalues (as a diagonal matrix) and eigenvectors
  Matrix<T> eigenvalues(1, 1);
  eigenvalues.set(0, 0, eigenvalue_new);
  return make_pair(eigenvalues, eigenvector);
}

template <typename T>
T determinant(Matrix<T> &matrix)
{
    // Base case: 1x1 matrix
    if (matrix.rows == 2 && matrix.cols == 2)
    {
        return Matrix<T>::determinant2x2(matrix);
    }
    if (matrix.rows == 1)
    {
        return matrix.get(0, 0);
    }

    // Recursive case: Calculate determinant using cofactor expansion
    T determinant_sum = 0;
    for (int col = 0; col < matrix.cols; ++col)
    {
        // Create a minor matrix by excluding the first row and the current column
        Matrix<T> minor(matrix.rows - 1, matrix.cols - 1);
        for (int i = 1; i < matrix.rows; ++i)
        {
            int minor_col = 0;
            for (int j = 0; j < matrix.cols; ++j)
            {
                if (j != col)
                {
                    minor.set(i - 1, minor_col, matrix.get(i, j));
                    minor_col++;
                }
            }
        }

        // Calculate the cofactor and add it to the determinant sum
        determinant_sum += pow(-1, col) * matrix.get(0, col) * determinant(minor);
    }

    return determinant_sum;
}

template <typename T>
std::pair<Matrix<T>, Matrix<T>> getMatrices() {
  cout << "Matrix A:" << endl;
  Matrix<T> matrix_A = createMatrix<T>();

  cout << "Matrix B:" << endl;
  Matrix<T> matrix_B = createMatrix<T>();

  return std::make_pair(matrix_A, matrix_B);
}



template <typename T>
Matrix<T> addMatrices(Matrix<T>& matrix_A, Matrix<T>& matrix_B) {
    if (matrix_A.rows != matrix_B.rows || matrix_A.cols != matrix_B.cols) {
        throw std::invalid_argument("Matrices must have the same dimensions for addition.");
    }

    Matrix<T> result(matrix_A.rows, matrix_A.cols);

    for (int i = 0; i < matrix_A.rows; ++i) {
        for (int j = 0; j < matrix_A.cols; ++j) {
            result.set(i, j, matrix_A.get(i, j) + matrix_B.get(i, j));
        }
    }

    return result;
}

template <typename T>
Matrix<T> addMatrices( std::vector<Matrix<T>>& matrices) {
    if (matrices.size() < 2) {
        throw std::invalid_argument("At least two matrices are required for addition.");
    }
    // Use fold expression to add matrices
    Matrix<T> result = matrices[0];
    // Loop to add each matrix to the result
    for (size_t i = 1; i < matrices.size(); ++i) {
        result = addMatrices(result, matrices[i]);
    }

    return result;
}



template <typename T>
Matrix<T> subMatrices(Matrix<T>& matrix_A, Matrix<T>& matrix_B) 
{
  // Check if matrices have the same dimensions
  if (matrix_A.rows != matrix_B.rows || matrix_A.cols != matrix_B.cols) {
    throw invalid_argument("Matrices must have the same dimensions for subtraction.");
  }

  // Create a new matrix with the same dimensions as A and B
  Matrix<T> matrix_C(matrix_A.rows, matrix_A.cols);

  // Add corresponding elements of A and B
  for (int i = 0; i < matrix_A.rows; ++i) {
    for (int j = 0; j < matrix_A.cols; ++j) {
      matrix_C.set(i, j, matrix_A.get(i, j) - matrix_B.get(i, j));
    }
  }

  // Return the resulting sum matrix
  return matrix_C;
}

template <typename T>
Matrix<T> ewmMatrices(Matrix<T>& matrix_A, Matrix<T>& matrix_B) 
{
  // Check if matrices have the same dimensions
  if (matrix_A.rows != matrix_B.rows || matrix_A.cols != matrix_B.cols) {
    throw invalid_argument("Matrices must have the same dimensions for Element-wise Multiplication.");
  }

  // Create a new matrix with the same dimensions as A and B
  Matrix<T> matrix_C(matrix_A.rows, matrix_A.cols);

  // Add corresponding elements of A and B
  for (int i = 0; i < matrix_A.rows; ++i) {
    for (int j = 0; j < matrix_A.cols; ++j) {
      matrix_C.set(i, j, matrix_A.get(i, j) * matrix_B.get(i, j));
    }
  }

  // Return the resulting sum matrix
  return matrix_C;
}

template <typename T>
Matrix<T> multiplyMatrices(Matrix<T>& matrix_A, Matrix<T>& matrix_B) {
  // Check if column count of A matches row count of B for multiplication
  if (matrix_A.cols != matrix_B.rows) {
    throw invalid_argument("Incompatible matrix dimensions for multiplication. Number of columns in A must equal number of rows in B.");
  }

  // Create a new result matrix C with dimensions (rows of A, columns of B)
  int rows_C = matrix_A.rows;
  int cols_C = matrix_B.cols;
  Matrix<T> matrix_C(rows_C, cols_C);

  // Perform matrix multiplication using nested loops
  for (int i = 0; i < rows_C; ++i) {
    for (int j = 0; j < cols_C; ++j) {
      T sum = 0;
      for (int k = 0; k < matrix_A.cols; ++k) { // Inner loop iterates over columns of A (also rows of B)
        sum += matrix_A.get(i, k) * matrix_B.get(k, j);
      }
      matrix_C.set(i, j, sum);
    }
  }

  // Return the resulting product matrix
  return matrix_C;
}

#include <iostream>
#include <limits> // for numeric_limits<streamsize>::max()
#include <sstream> // for stringstream

using namespace std;

// Function to solve a system of linear equationstemplate <typename T>
template <typename T>
Matrix<T> solveSystemOfLinearEquations(Matrix<T>& A, Matrix<T>& b) {
    try {
        // Calculate the determinant of the coefficient matrix A
        auto det = determinant(A);

        if (det == 0) {
            // If the determinant is zero, the matrix is singular
            // Check if the system has a solution or not
            pair<Matrix<T>, Matrix<T>> lu = luDecomposition(A);
            Matrix<T>& L = lu.first;
            Matrix<T>& U = lu.second;

            // Create a vector y using forward substitution
            Matrix<T> y(A.rows, 1);
            for (int i = 0; i < A.rows; ++i) {
                y.set(i, 0, b.get(i, 0));
                for (int j = 0; j < i; ++j) {
                    y.set(i, 0, y.get(i, 0) - L.get(i, j) * y.get(j, 0));
                }
            }

            // Check if y is the zero vector
            bool isZeroVector = true;
            for (int i = 0; i < A.rows; ++i) {
                if (y.get(i, 0) != 0) {
                    isZeroVector = false;
                    break;
                }
            }

            if (isZeroVector) {
                // If y is the zero vector, the system has infinitely many solutions
                cout << "The system of equations has infinitely many solutions." << endl;
                return Matrix<T>(A.rows, 1); // Return an arbitrary solution
            } else {
                // If y is not the zero vector, the system has no solution
                cout << "The system of equations has no solution." << endl;
                return Matrix<T>(0, 0); // Return an empty matrix
            }
        } else {
            // If the determinant is not zero, calculate the inverse of A
            Matrix<T> inverseA = matrixInverse(A);

            // Multiply the inverse of A with the constant vector b
            return multiplyMatrices(inverseA, b);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return Matrix<T>(0, 0);
    }
}
void solveSystemOfLinearEquations() {
  int num_variables;

  // Get user input for the number of variables with error handling
  while (true) {
    cout << "Enter the number of variables (same as the number of equations): ";
    if (!(cin >> num_variables)) {
      // Handle invalid input (clear input buffer)
      cin.clear();
      cin.ignore(numeric_limits<streamsize>::max(), '\n');
      cout << "Invalid input. Please enter a positive integer." << endl;
      continue;
    }

    if (num_variables <= 0) {
      cout << "Error: The number of variables must be positive." << endl;
      continue;
    }

    break; // Exit the loop if valid input is received
  }

  // Create matrices for coefficients and constants (same number of rows)
  Matrix<double> A(num_variables, num_variables);
  Matrix<double> b(num_variables, 1);

  // Print table header for coefficients
  std::cout << endl << "Enter the coefficients of the equations (row-wise):" << endl;

  // Take input for coefficients and constant in a single line
  for (int i = 0; i < num_variables; ++i) {
    std::cout << "Eq " << i + 1 << " : ";
    for (int j = 0; j < num_variables; ++j) {
      double value;
      cout << "Coeff of x" << j + 1 << ": ";
      cin >> value;
      A.set(i, j, value);
    }
    cout << "Const of Eq" << i + 1 << ": ";
    double constant;
    cin >> constant;
    b.set(i, 0, constant);
    cout << endl;
  }

  cout << "Coefficient Matrix A:" << endl;
  displayMatrix(A); // Assuming your Matrix class has a printMatrix() method
  cout << "Constant Vector b:" << endl;
  displayMatrix(b);
  // Solve the system of linear equations (assuming a suitable library function exists)
  Matrix<double> solution(1,num_variables);
  try {
    solution = solveSystemOfLinearEquations(A,b); 

    // Check if a solution was found
    if (solution.rows == 0) {
      cout << "The system of equations has no solution." << endl;
    } else {
      cout << "Solution to the system of linear equations:" << endl;
      for (int i = 0; i < num_variables; ++i) {
        cout << "x" << i + 1 << " = " << solution.get(i, 0) << endl;
      }
    }
  } catch (const std::exception& e) {
    cerr << "Error: " << e.what() << endl;
  }

}

int main() {
    while (true) {
        cout << "\nMenu:" << endl;
        cout << "1. Unary Matrix Operations" << endl;
        cout << "2. Binary Matrix Operations" << endl;
        cout << "3. Addition of a Array of Matrices " << endl;
        cout << "4. Solving System of Linear Equations " << endl;
        cout << "5. Exit" << endl;
        int choice;
        cout << "Enter your choice: ";
        if (!(std::cin >> choice)) {
      std::cin.clear(); // Clear the error state from cin
      std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Discard invalid input
      std::cout << "Invalid input. Please enter a number (1-5)." << std::endl;
      continue;
    }

    // Check if choice is within valid range (1-4)
    if (choice < 1 || choice > 5) {
      std::cout << "Invalid choice. Please enter a number between 1 and 5." << std::endl;
      continue;
    }
        
        
        
        switch (choice) {
            case 1:
                // Submenu for unary matrix operations
                while (true) {
                    cout << "\nUnary Matrix Operations:" << endl;
                    cout << "1. Matrix Trace" << endl;
                    cout << "2. Matrix Average" << endl;
                    cout << "3. Matrix Transpose " << endl;
                    cout << "4. Matrix LU Decomposition " << endl;
                    cout << "5. Matrix Inverse " << endl;
                    cout << "6. Matrix Determinant " << endl;
                    cout << "7. Matrix Rank " << endl;
                    cout << "8. Matrix Eigen Vectors and Values " << endl;
                    cout << "9. Go Back to Main Menu" << endl;

                    int subChoice;
                    cout << "Enter your choice: ";
                    if (!(std::cin >> subChoice)) {
                        std::cin.clear(); // Clear the error state from cin
                        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Discard invalid input
                        std::cout << "Invalid input. Please enter a number (1-9)." << std::endl;
                        continue;
                    }

                    // Check if choice is within valid range (1-4)
                    if (subChoice < 1 || subChoice > 9) {
                    std::cout << "Invalid choice. Please enter a number between 1 and 9." << std::endl;
                    continue;
                    }

                    switch (subChoice) {
                        case 1:
                            {
                                Matrix<double> matrix = createMatrix<double>();
                                matrixTrace(matrix);
                                break;
                            }
                        case 2:
                            {
                                Matrix<double> matrix = createMatrix<double>();
                                calculateColumnMeans(matrix);
                                break;
                            }
                        case 3:
                            {
                                Matrix<double> matrix = createMatrix<double>();
                                transpose(matrix);
                                break;
                            }
                        case 4:
                            {
                                Matrix<double> matrix = createMatrix<double>();
                                try {
                                    pair<Matrix<double>, Matrix<double>> lu = luDecomposition(matrix);
                                    cout << "Lower triangular matrix (L):" << endl;
                                    displayMatrix(lu.first);
                                    cout << "Upper triangular matrix (U):" << endl;
                                    displayMatrix(lu.second);
                                } catch (const invalid_argument& e) {
                                    cerr << "Error: " << e.what() << endl;
                                }
                                break;
                            }
                        case 5:
                            {
                                Matrix<double> matrix = createMatrix<double>();
                                try {
                                    Matrix<double> inverse = matrixInverse(matrix);
                                    cout << "Inverse of the matrix:" << endl;
                                    displayMatrix(inverse);
                                } catch (const invalid_argument& e) {
                                    cerr << "Error: " << e.what() << endl;
                                }
                                break;
                            }
                        case 6:
                            {
                                Matrix<double> matrix = createMatrix<double>();
                                try {
                                    auto det = determinant(matrix);
                                    cout << "Determinant of the matrix: " << det << endl;
                                } catch (const invalid_argument& e) {
                                    cerr << "Error: " << e.what() << endl;
                                }
                                break;
                            }
                        case 7:
                            {
                                Matrix<double> matrix = createMatrix<double>();
                                int r = matrixRank(matrix);
                                cout << "Rank of the matrix: " << r << endl;
                                break;
                            }
                        case 8:
                            {
                                Matrix<double> matrix = createMatrix<double>();
                                try {
                                    if (matrix.rows != matrix.cols) {
                                        throw invalid_argument("Eigenvalues and eigenvectors are only defined for square matrices.");
                                    }
                                    pair<Matrix<double>, Matrix<double>> result = eigenValuesAndVectors(matrix);
                                    cout << "Eigenvalues:\n";
                                    displayMatrix(result.first);
                                    cout << "Eigenvectors:\n";
                                    displayMatrix(result.second);
                                } catch (const invalid_argument& e) {
                                    cerr << "Error: " << e.what() << endl;
                                }
                                break;
                            }
                        case 9:
                            {
                                // Exit the unary operations submenu and go back to main menu
                                break;
                            }
                        default:
                            cout << "Invalid choice. Please try again." << endl;
                    }
                    if (subChoice == 9) {
                        break;
                    }
                }
                break;
            case 2:
            {
                cout << "\nSelect Binary Operation:" << endl;
                cout << "1. Matrix Addition" << endl;
                cout << "2. Matrix Subtraction" << endl;
                cout << "3. Matrix Multiplication" <<endl;
                cout << "4. Matrix Element-wise Multiplication"<<endl;
                cout << "5. Go Back to Main Menu" << endl;

                int subChoice;
                    cout << "Enter your choice: ";
                    if (!(std::cin >> subChoice)) {
                        std::cin.clear(); // Clear the error state from cin
                        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Discard invalid input
                        std::cout << "Invalid input. Please enter a number (1-5)." << std::endl;
                        continue;
                    }

                    // Check if choice is within valid range (1-4)
                    if (subChoice < 1 || subChoice > 5) {
                    std::cout << "Invalid choice. Please enter a number between 1 and 5." << std::endl;
                    continue;
                    }

               
                switch (subChoice) {
                case 1: {
                    cout << "Matrix Addition is only performed on Matrices with same dimensions" << endl;

                    try {
                        // Get two matrices from the user using getMatrices function
                        std::pair<Matrix<double>, Matrix<double>> matricesPair = getMatrices<double>();

                        // Access the matrices from the pair
                        Matrix<double> matrix_A = matricesPair.first;
                        Matrix<double> matrix_B = matricesPair.second;

                        // Check if dimensions are compatible before addition
                        if (matrix_A.rows != matrix_B.rows || matrix_A.cols != matrix_B.cols) {
                            throw invalid_argument("Matrices must have the same dimensions for addition.");
                        }

                        // Create an array of matrices and populate it with the two matrices
                       // Matrix<double> matrices[2] = {matrix_A, matrix_B};

                        // Perform matrix addition using the new addMatrices function
                        Matrix<double> matrix_C = addMatrices(matrix_A, matrix_B);
                        
                        cout << "Sum of matrices:" << endl;
                        displayMatrix(matrix_C);

                    } catch (const invalid_argument& e) {
                        cerr << "Error: " << e.what() << endl;
                    }

                    break;
                }

                case 2: {
                    cout << "Matrix Subtraction is only performed on Matrices with same dimensions"<<endl;
                    // Get two matrices from the user using getMatrices function
                    std::pair<Matrix<double>, Matrix<double>> matrices = getMatrices<double>();

                    // Access the matrices from the pair
                    Matrix<double> matrix_A = matrices.first;
                    Matrix<double> matrix_B = matrices.second;

                    try {
                        // Check if dimensions are compatible before addition
                        if (matrix_A.rows != matrix_B.rows || matrix_A.cols != matrix_B.cols) {
                        throw invalid_argument("Matrices must have the same dimensions for subtraction.");
                        }

                        // Perform matrix addition
                        Matrix<double> matrix_C = subMatrices(matrix_A, matrix_B);
                        cout << "Diffrences of matrices:" << endl;
                        displayMatrix(matrix_C);
                    } catch (const invalid_argument& e) {
                        cerr << "Error: " << e.what() << endl;
                    }
                    break;
                    }
                case 3: {
                    // Get two matrices from the user using getMatrices function
                    std::pair<Matrix<double>, Matrix<double>> matrices = getMatrices<double>();

                    // Access the matrices from the pair
                    Matrix<double> matrix_A = matrices.first;
                    Matrix<double> matrix_B = matrices.second;

                    try {
                        // Check if dimensions are compatible for multiplication (A columns = B rows)
                        if (matrix_A.cols != matrix_B.rows) {
                        throw invalid_argument("Incompatible matrix dimensions for multiplication. Number of columns in A must equal number of rows in B.");
                        }

                        // Perform matrix multiplication
                        Matrix<double> matrix_C = multiplyMatrices(matrix_A, matrix_B);
                        cout << "Product of matrices (C = A * B):" << endl;
                        displayMatrix(matrix_C);
                    } catch (const invalid_argument& e) {
                        cerr << "Error: " << e.what() << endl;
                    }
                    break;
                    }
                case 4: {
                    cout << "MatrixElement-wise Multiplication is only performed on Matrices with same dimensions"<<endl;
                    // Get two matrices from the user using getMatrices function
                    std::pair<Matrix<double>, Matrix<double>> matrices = getMatrices<double>();

                    // Access the matrices from the pair
                    Matrix<double> matrix_A = matrices.first;
                    Matrix<double> matrix_B = matrices.second;

                    try {
                        // Check if dimensions are compatible before addition
                        if (matrix_A.rows != matrix_B.rows || matrix_A.cols != matrix_B.cols) {
                        throw invalid_argument("Matrices must have the same dimensions for Element-wise Multiplication.");
                        }

                        // Perform matrix addition
                        Matrix<double> matrix_C = ewmMatrices(matrix_A, matrix_B);
                        cout << "Product of Element-wise Multiplication:" << endl;
                        displayMatrix(matrix_C);
                    } catch (const invalid_argument& e) {
                        cerr << "Error: " << e.what() << endl;
                    }
                    break;
                    }

                    case 5:
                            {
                                // Exit the unary operations submenu and go back to main menu
                                break;
                            }
                        default:
                            cout << "Invalid choice. Please try again." << endl;
                    }
                    if (subChoice == 5) {
                        break;
                    }
                }
                break;
          case 3:
            // Operations on an Array of Matrices submenu
            {
                std::vector<Matrix<double>> matrices;
                int matrixCount;
                std::cout << "Enter the number of matrices to add (min 2): ";
                if (!(std::cin >> matrixCount) || matrixCount < 2) {
                    std::cin.clear();
                    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                    std::cout << "Invalid input. Please enter a number greater than 1." << std::endl;
                    break;
                }

                for (int i = 0; i < matrixCount; ++i) {
                    std::cout << "Enter matrix " << i + 1 << " (rows cols): ";
                    int rows, cols;
                    if (!(std::cin >> rows >> cols)) {
                        std::cin.clear();
                        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                        std::cout << "Invalid input. Please enter rows and cols." << std::endl;
                        break;
                    }

                    Matrix<double> matrix(rows, cols);
                    std::cout << "Enter matrix elements:" << std::endl;
                    for (int r = 0; r < rows; ++r) {
                        for (int c = 0; c < cols; ++c) {
                            double element;
                            if (!(std::cin >> element)) {
                                std::cin.clear();
                                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                                std::cout << "Invalid input. Please enter matrix elements." << std::endl;
                                break;
                            }
                            matrix.set(r, c, element);
                        }
                    }
                    matrices.push_back(matrix);
                }

                try {
                    Matrix<double> result = addMatrices(matrices);  // Use the updated addMatrices function
                    std::cout << "Result of adding matrices:" << std::endl;
                    displayMatrix(result);  // Display the result matrix
                } catch (const std::exception& e) {
                    std::cerr << "Error: " << e.what() << std::endl;
                }
            }
            break;

            case 4: 
            {
                solveSystemOfLinearEquations();
                break;

            }
            case 5:
                cout << "Exiting..." << endl;
                return 0;
            default:
                cout << "Invalid choice. Please try again." << endl;
        }
    }
    return 0;
}

