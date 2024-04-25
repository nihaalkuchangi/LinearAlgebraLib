#include <iostream>
#include <unordered_map>
#include <string>
#include <cmath>
#include <limits>
#include <type_traits>
#include <sstream>
#include <iomanip>

using namespace std;

// Global 2D int array
int res[4][8] = {1};

template <typename T1, typename T2>
class Pair
{
public:
    // Default constructor
    Pair() : first(), second() {}

    // Parameterized constructor
    Pair(const T1 &f, const T2 &s) : first(f), second(s) {}

    // Copy constructor
    Pair(const Pair &other) : first(other.first), second(other.second) {}

    // Destructor (default)
    ~Pair() = default;

    // Getter for first element
    T1 getFirst() const
    {
        return first;
    }

    // Setter for first element
    void setFirst(const T1 &f)
    {
        first = f;
    }

    // Getter for second element
    T2 getSecond() const
    {
        return second;
    }

    // Setter for second element
    void setSecond(const T2 &s)
    {
        second = s;
    }

private:
    T1 first;
    T2 second;
};

template <typename T>
class Vector
{
private:
    T *data;
    size_t capacity;
    size_t size;

public:
    Vector();
    Vector(size_t initialSize);
    Vector(const Vector &other);
    ~Vector();
    Vector &operator=(const Vector &other);
    size_t Size() const;
    size_t Capacity() const;
    void PushBack(const T &value);
    T &operator[](size_t index);
    const T &operator[](size_t index) const;
    class iterator
    {
    private:
        T *ptr;

    public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = T;
        using pointer = T *;
        using reference = T &;

        iterator(pointer p) : ptr(p) {}

        reference operator*() const { return *ptr; }
        pointer operator->() { return ptr; }

        // Pre-increment
        iterator &operator++()
        {
            ++ptr;
            return *this;
        }

        // Post-increment
        iterator operator++(int)
        {
            iterator temp = *this;
            ++ptr;
            return temp;
        }

        bool operator==(const iterator &other) const { return ptr == other.ptr; }
        bool operator!=(const iterator &other) const { return ptr != other.ptr; }
    };

    iterator begin() { return iterator(data); }
    iterator end() { return iterator(data + size); }
};

template <typename T>
Vector<T>::Vector() : data(nullptr), capacity(0), size(0) {}
template <typename T>
Vector<T>::Vector(size_t initialSize) : data(nullptr), capacity(initialSize), size(initialSize)
{
    data = new T[capacity];
}
template <typename T>
Vector<T>::Vector(const Vector &other) : capacity(other.capacity), size(other.size)
{
    data = new T[capacity];
    for (size_t i = 0; i < size; ++i)
    {
        data[i] = other.data[i];
    }
}

template <typename T>
Vector<T>::~Vector()
{
    delete[] data;
}

template <typename T>
Vector<T> &Vector<T>::operator=(const Vector &other)
{
    if (this != &other)
    {
        delete[] data;
        capacity = other.capacity;
        size = other.size;
        data = new T[capacity];
        for (size_t i = 0; i < size; ++i)
        {
            data[i] = other.data[i];
        }
    }
    return *this;
}

template <typename T>
size_t Vector<T>::Size() const
{
    return size;
}

template <typename T>
size_t Vector<T>::Capacity() const
{
    return capacity;
}

template <typename T>
void Vector<T>::PushBack(const T &value)
{
    if (size == capacity)
    {
        capacity = (capacity == 0) ? 1 : capacity * 2;
        T *newData = new T[capacity];
        for (size_t i = 0; i < size; ++i)
        {
            newData[i] = data[i];
        }
        delete[] data;
        data = newData;
    }
    data[size++] = value;
}

template <typename T>
T &Vector<T>::operator[](size_t index)
{
    return data[index];
}

template <typename T>
const T &Vector<T>::operator[](size_t index) const
{
    return data[index];
}

template <typename T>
struct Matrix
{
    static_assert(std::is_arithmetic<T>::value, "Matrix type must be arithmetic.");

    int rows;
    int cols;
    string name;
    Vector<T> data;

    template <typename U>
    friend void displayMatrix(Matrix<U> &matrix);

    template <typename U>
    friend Matrix<U> createMatrix();

    Matrix() : rows(0), cols(0), name("Matrix0") {}

    Matrix(int r, int c, const string &n) : rows(r), cols(c), name(n), data(r * c)
    {
        if (r <= 0 || c <= 0)
        {
            throw invalid_argument("Matrix dimensions must be positive.");
        }
    }

    T &get(int i, int j);
    void set(int i, int j, const T &value);
    void takeInput();

    template <typename K>
    static T determinant2x2(Matrix<K> &matrix);
};

template <typename T>
unordered_map<string, Matrix<T>> matrices;

template <typename T>
unordered_map<string, Matrix<T>> &getMatrices()
{
    return matrices<T>;
}

template <typename T>
T &Matrix<T>::get(int i, int j)
{
    if (i < 0 || i >= rows || j < 0 || j >= cols)
    {
        throw out_of_range("Index out of bounds for matrix access.");
    }
    return data[i * cols + j];
}

template <typename T>
void Matrix<T>::set(int i, int j, const T &value)
{
    get(i, j) = value;
}

template <typename T>
void Matrix<T>::takeInput()
{
    cout << "Enter elements for the matrix:" << endl;

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            while (!(cin >> data[i * cols + j]))
            {
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
                cout << "Invalid input. Please enter a number: ";
            }
        }
    }
}
template <typename T>
void displayMatrix(Matrix<T> &matrix)
{
    for (int j = 0; j < matrix.cols; ++j)
    {
        cout << "+---";
    }
    cout << "+" << endl;

    for (int i = 0; i < matrix.rows; ++i)
    {
        cout << "| ";
        for (int j = 0; j < matrix.cols; ++j)
        {
            cout << matrix.get(i, j) << " | ";
        }
        cout << endl;

        if (i < matrix.rows - 1)
        {
            for (int j = 0; j < matrix.cols; ++j)
            {
                cout << "+---";
            }
            cout << "+" << endl;
        }
    }

    for (int j = 0; j < matrix.cols; ++j)
    {
        cout << "+---";
    }
    cout << "+" << endl;
}

template <typename T>
Matrix<T> createMatrix()
{
    int rows, cols;
    string name;

    while (true)
    {
        cout << "Rows: ";

        if (!(cin >> rows) || rows <= 0)
        {
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cout << "Invalid input. Please enter a positive integer for rows." << endl;
            continue;
        }

        cout << "Cols: ";

        if (!(cin >> cols) || cols <= 0)
        {
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cout << "Invalid input. Please enter a positive integer for columns." << endl;
            continue;
        }

        cout << "Enter a name for the matrix: ";
        cin >> name;

        if (name.empty())
        {
            cout << "Matrix name cannot be empty. Please enter a valid name." << endl;
            continue;
        }

        if (getMatrices<double>().count(name))
        {
            cout << "The name " << name << " is already being used by another matrix, give a new name to the matrix" << endl;
            continue;
        }

        if (name == "new")
        {
            cout << "The name new is already inavlid, give another name to the matrix" << endl;
            continue;
        }

        break;
    }

    Matrix<T> matrix(rows, cols, name);
    matrix.takeInput();
    cout << "The inputted matrix '" << matrix.name << "' is:" << endl;
    displayMatrix(matrix);

    getMatrices<T>()[matrix.name] = matrix; // Store the matrix in the unordered_map

    return matrix;
}
template <typename T>
Matrix<T> addMatrices(Matrix<T> &matrix)
{
    return matrix;
}

template <typename T, typename... Matrices>
Matrix<T> addMatrices(Matrix<T> &matrix, Matrix<T> &nextMatrix, Matrices &...matrices)
{
    Matrix<T> temp = addTwoMatrices(matrix, nextMatrix);
    return addMatrices(temp, matrices...);
}

template <typename T>
Matrix<T> addTwoMatrices(Matrix<T> &mat1, Matrix<T> &mat2)
{
    // Create a result matrix with the same dimensions as the input matrices
    Matrix<T> result(mat1.rows, mat1.cols, "temp");
    initializeToZero(result);

    // Perform addition element-wise
    for (int i = 0; i < mat1.rows; ++i)
    {
        for (int j = 0; j < mat1.cols; ++j)
        {
            result.set(i, j, mat1.get(i, j) + mat2.get(i, j));
        }
    }

    return result;
}

template <typename T>
template <typename K>
T Matrix<T>::determinant2x2(Matrix<K> &matrix)
{
    return matrix.get(0, 0) * matrix.get(1, 1) - matrix.get(0, 1) * matrix.get(1, 0);
}
// Unanry Operations
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

template <typename T>
void calculateColumnMeans(Matrix<T> &matrix)
{
    // Create a 1xcols matrix to store column means
    std::ostringstream oss;
    res[0][1]++;
    oss << "Mean" << res[0][1];
    string meanname = oss.str();

    Matrix<double> means(1, matrix.cols, meanname); // Using double for more accurate mean calculation

    // Initialize means matrix to zeros
    for (int j = 0; j < matrix.cols; ++j)
    {
        means.get(0, j) = 0.0;
    }

    // Calculate sum of each column
    for (int i = 0; i < matrix.rows; ++i)
    {
        for (int j = 0; j < matrix.cols; ++j)
        {
            means.get(0, j) += static_cast<double>(matrix.get(i, j)); // Cast to double for floating-point addition
        }
    }

    // Calculate average for each column
    for (int j = 0; j < matrix.cols; ++j)
    {
        means.get(0, j) /= static_cast<double>(matrix.rows); // Cast to double for floating-point division
    }

    getMatrices<T>()[means.name] = means;
    cout << "Means of each column in the matrix: " << matrix.name << "(" << means.name << ")" << endl; // Assuming matrix has a name member
    displayMatrix(means);
}
template <typename T>
void transpose(Matrix<T> &matrix)
{
    std::ostringstream oss;
    res[0][2]++;
    oss << "Transpose" << res[0][2];
    string resname = oss.str();
    Matrix<T> transposed(matrix.cols, matrix.rows, resname);
    // Fill the transposed matrix by swapping elements
    for (int i = 0; i < matrix.rows; ++i)
    {
        for (int j = 0; j < matrix.cols; ++j)
        {
            transposed.set(j, i, matrix.get(i, j));
        }
    }

    getMatrices<T>()[transposed.name] = transposed;
    cout << "Transpose of the matrix: " << matrix.name << "(" << transposed.name << ")" << endl; // Assuming matrix has a name member
    displayMatrix(transposed);
}

// Lambda template for magnitude
template <typename T>
auto magnitude = [](T val)
{
    return std::abs(val);
};

template <>
auto magnitude<float> = [](float val)
{
    return std::fabs(val);
};

template <>
auto magnitude<double> = [](double val)
{
    return std::fabs(val);
};
template <typename T>
T norm(Matrix<T> &vec)
{
    T sum = 0;
    for (int i = 0; i < vec.rows; ++i)
    {
        sum += vec.get(i, 0) * vec.get(i, 0);
    }
    return std::sqrt(sum);
}
template <typename T>
T vectorDotProduct(Matrix<T> &vec1, Matrix<T> &vec2)
{
    if (vec1.rows != vec2.rows)
    {
        throw std::invalid_argument("Vector dimensions must match for dot product.");
    }
    T result = 0;
    for (int i = 0; i < vec1.rows; ++i)
    {
        result += vec1.get(i, 0) * vec2.get(i, 0);
    }
    return result;
}
template <typename T>
int matrixRank(Matrix<T> &matrix)
{
    Matrix<T> temp(matrix); // Create a copy of the matrix to avoid modifying the original

    int rank = 0;
    int n = temp.rows;
    int m = temp.cols;
    int x = n < m ? n : m;
    for (int col = 0; col < x; ++col)
    {
        int pivotRow = col;

        // Find the pivot element (largest element in magnitude in the current column below the diagonal)
        for (int i = col + 1; i < n; ++i)
        {
            if (magnitude<T>(temp.get(i, col)) > magnitude<T>(temp.get(pivotRow, col)))
            {
                pivotRow = i;
            }
        }

        // If the pivot element is zero, the column is linearly dependent and the rank doesn't increase
        if (magnitude<T>(temp.get(pivotRow, col)) == 0)
        {
            continue;
        }

        // Swap rows if necessary to ensure a non-zero pivot
        if (pivotRow != col)
        {
            for (int j = 0; j < m; ++j)
            {
                std::swap(temp.get(col, j), temp.get(pivotRow, j));
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
pair<Matrix<T>, Matrix<T>> luDecomposition(Matrix<T> &matrix, int n)
{

    // Create matrices for L and U with the same dimensions as the input matrix
    Matrix<T> lower(n, n, "lower");
    Matrix<T> upper(n, n, "upper");

    // Initialize matrices to zero
    initializeToZero(lower);
    initializeToZero(upper);

    // Decomposing matrix into Upper and Lower triangular matrix
    for (int i = 0; i < n; ++i)
    {
        // Upper Triangular
        for (int k = i; k < n; ++k)
        {
            T sum = 0.0;
            for (int j = 0; j < i; ++j)
            {
                sum += lower.get(i, j) * upper.get(j, k);
            }
            upper.set(i, k, matrix.get(i, k) - sum);

            // Round to 4 decimal places
            upper.set(i, k, std::round(upper.get(i, k) * 10000.0) / 10000.0);
        }

        // Lower Triangular
        for (int k = i; k < n; ++k)
        {
            if (i == k)
            {
                lower.set(i, i, 1.0); // Diagonal as 1
            }
            else
            {
                T sum = 0.0;
                for (int j = 0; j < i; ++j)
                {
                    sum += lower.get(k, j) * upper.get(j, i);
                }
                lower.set(k, i, (matrix.get(k, i) - sum) / upper.get(i, i));

                // Round to 4 decimal places
                lower.set(k, i, std::round(lower.get(k, i) * 10000.0) / 10000.0);
            }
        }
    }

    // Return the pair of Lower and Upper matrices
    return {lower, upper};
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
        std::ostringstream oss;
        res[0][3]++;
        oss << "Minor" << res[0][3];
        string resname = oss.str();
        Matrix<T> minor(matrix.rows - 1, matrix.cols - 1, resname);
        std::ostringstream minorOss;
        res[0][4]++; // Assuming res is a global 2D array
        minorOss << matrix.name << "Minor" << res[0][4];
        std::string minorName = minorOss.str();

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
        determinant_sum += std::round(pow(-1, col) * matrix.get(0, col) * determinant(minor) * 10000.0) / 10000.0;
    }

    return determinant_sum;
}

template <typename T>
Matrix<T> matrixInverse(Matrix<T> &matrix)
{
    // Check if the matrix is square
    if (matrix.rows != matrix.cols)
    {
        throw std::invalid_argument("Matrix inverse is only defined for square matrices.");
    }

    int n = matrix.rows;

    // Check if the matrix is singular by calculating its determinant
    auto det = determinant(matrix);
    if (det == 0)
    {
        throw std::invalid_argument("The input matrix is singular, and its inverse does not exist.");
    }

    // Perform LU decomposition of the matrix
    pair<Matrix<T>, Matrix<T>> lu = luDecomposition(matrix, n);
    Matrix<T> L = lu.first;
    Matrix<T> U = lu.second;

    // Create an identity matrix with the same dimensions as the input matrix
    Matrix<T> identity(n, n, "I");
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (i != j)
                identity.set(i, j, 0.0);
            else
                identity.set(i, j, 1);
        }
    }

    // Initialize the inverse matrix to zero
    std::ostringstream inverseOss;
    res[0][5]++; // Assuming res is a global 2D array
    inverseOss << matrix.name << "Inverse" << res[0][5];
    std::string inverseName = inverseOss.str();
    Matrix<T> inverse(n, n, inverseName);

    initializeToZero(inverse);

    // Solve the system of equations L * y = identity for each column of the identity matrix
    // This gives us the columns of the inverse matrix
    for (int j = 0; j < n; ++j)
    {
        Matrix<T> y(n, 1, "col1"); // Create a column vector to store the solution

        // Forward substitution
        for (int i = 0; i < n; ++i)
        {
            T sum = 0;
            // Check for division by zero in U.get(i, i) before division
            if (std::abs(U.get(i, i)) < std::numeric_limits<T>::epsilon())
            {
                throw std::invalid_argument("Matrix is nearly singular, inverse calculation may be inaccurate.");
            }
            for (int k = 0; k < i; ++k)
            {
                sum += L.get(i, k) * y.get(k, 0);
            }
            y.set(i, 0, identity.get(i, j) - sum);
        }

        // Backward substitution
        for (int i = n - 1; i >= 0; --i)
        {
            T sum = 0;
            // Check for division by zero in U.get(i, i) before division
            if (std::abs(U.get(i, i)) < std::numeric_limits<T>::epsilon())
            {
                throw std::invalid_argument("Matrix is nearly singular, inverse calculation may be inaccurate.");
            }
            for (int k = i + 1; k < n; ++k)
            {
                sum += U.get(i, k) * inverse.get(k, j);
            }
            inverse.set(i, j, (y.get(i, 0) - sum) / U.get(i, i));
        }
    }

    // Add the inverse matrix to the matrices collection
    getMatrices<T>()[inverseName] = inverse;

    // Print the inverse matrix
    cout << "Inverse of matrix " << matrix.name << ": " << inverse.name << endl;
    displayMatrix(inverse);
    return inverse;
}
template <typename T>
void eigenValuesAndVectors(Matrix<T> &matrix)
{
    // Check if the matrix is square
    if (matrix.rows != matrix.cols)
    {
        throw std::invalid_argument("Eigenvalues and eigenvectors are only defined for square matrices.");
    }

    int n = matrix.rows;

    // Choose an initial guess vector (e.g., the first column of the matrix)
    Matrix<T> x(n, 1, "X");
    for (int i = 0; i < n; ++i)
    {
        x.set(i, 0, matrix.get(i, 0));
    }

    // Iterate until convergence (eigenvalue estimate stabilizes)
    int max_iterations = 100;
    double eigenvalue_old, eigenvalue_new;
    for (int iter = 0; iter < max_iterations; ++iter)
    {
        // Perform matrix-vector multiplication (y = A * x)
        Matrix<T> y(n, 1, "y");
        for (int i = 0; i < n; ++i)
        {
            T sum = 0;
            for (int j = 0; j < n; ++j)
            {
                sum += matrix.get(i, j) * x.get(j, 0);
            }
            y.set(i, 0, sum);
        }

        // Normalize the resulting vector (x = y / ||y||)
        T norm_value = norm(y); // Assuming a norm function is available
        for (int i = 0; i < n; ++i)
        {
            x.set(i, 0, y.get(i, 0) / norm_value);
        }

        // Calculate the Rayleigh quotient (estimate of the eigenvalue)
        eigenvalue_new = vectorDotProduct(matrix, x); // Assuming a vectorDotProduct function is available

        // Check for convergence (significant change in eigenvalue estimate)
        if (iter > 0 && std::abs(eigenvalue_new - eigenvalue_old) / std::abs(eigenvalue_old) < 1e-6)
        {
            break;
        }

        eigenvalue_old = eigenvalue_new;
    }

    // Create a matrix to store the eigenvector with a name
    std::ostringstream eigenvectorOss;
    res[0][3]++; // Assuming res is a global 2D array
    eigenvectorOss << matrix.name << "Eigenvector" << res[0][3];
    std::string eigenvectorName = eigenvectorOss.str();
    Matrix<T> eigenvector(n, 1, eigenvectorName);
    for (int i = 0; i < n; ++i)
    {
        eigenvector.set(i, 0, x.get(i, 0));
    }

    // Create a matrix to store the eigenvalue with a name
    std::ostringstream eigenvalueOss;
    res[0][4]++;
    eigenvalueOss << matrix.name << "Eigenvalue" << res[0][4];
    std::string eigenvalueName = eigenvalueOss.str();
    Matrix<T> eigenvalue(1, 1, eigenvalueName);
    eigenvalue.set(0, 0, eigenvalue_new);

    // Add matrices to global matrices
    getMatrices<T>()[eigenvectorName] = eigenvector;
    getMatrices<T>()[eigenvalueName] = eigenvalue;

    // Display matrices
    cout << "Eigenvector of the matrix: " << matrix.name << endl;
    displayMatrix(eigenvector);
    cout << "Eigenvalue of the matrix: " << matrix.name << endl;
    displayMatrix(eigenvalue);
}
template <typename T>
void luDecomposition(Matrix<T> &matrix)
{
    // Check if the matrix is square
    if (matrix.rows != matrix.cols)
    {
        throw std::invalid_argument("LU decomposition is only defined for square matrices.");
    }

    // Create matrices for L and U with the same dimensions as the input matrix
    int n = matrix.rows;
    std::ostringstream lowerOss, upperOss;
    res[0][2]++; // Assuming res is a global 2D array
    lowerOss << "Lower" << res[0][2];
    upperOss << "Upper" << res[0][2];
    std::string lowerName = lowerOss.str();
    std::string upperName = upperOss.str();
    Matrix<T> lower(n, n, lowerName);
    Matrix<T> upper(n, n, upperName);

    // Initialize matrices to zero
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            lower.set(i, j, 0.0);
            upper.set(i, j, 0.0);
        }
    }

    // Decomposing matrix into Upper and Lower triangular matrix
    for (int i = 0; i < n; ++i)
    {
        // Upper Triangular
        for (int k = i; k < n; ++k)
        {
            T sum = 0.0;
            for (int j = 0; j < i; ++j)
            {
                sum += lower.get(i, j) * upper.get(j, k);
            }
            upper.set(i, k, matrix.get(i, k) - sum);

            // Round to 4 decimal places
            upper.set(i, k, std::round(upper.get(i, k) * 10000.0) / 10000.0);
        }

        // Lower Triangular
        for (int k = i; k < n; ++k)
        {
            if (i == k)
            {
                lower.set(i, i, 1.0); // Diagonal as 1
            }
            else
            {
                T sum = 0.0;
                for (int j = 0; j < i; ++j)
                {
                    sum += lower.get(k, j) * upper.get(j, i);
                }
                lower.set(k, i, (matrix.get(k, i) - sum) / upper.get(i, i));

                // Round to 4 decimal places
                lower.set(k, i, std::round(lower.get(k, i) * 10000.0) / 10000.0);
            }
        }
    }

    // Save lower and upper matrices
    getMatrices<T>()[lowerName] = lower;
    getMatrices<T>()[upperName] = upper;

    // Print lower and upper matrices
    cout << "Lower matrix: " << lower.name << endl;
    displayMatrix(lower);
    cout << "Upper matrix: " << upper.name << endl;
    displayMatrix(upper);
}

// Binary Operations:
template <typename T>
void addMatrix(Matrix<T> &matrix_A, Matrix<T> &matrix_B)
{
    std::ostringstream name;
    res[1][0]++; // Assuming res is a global 2D array
    name << "Add" << res[1][0];
    std::string name1 = name.str();
    Matrix<T> result(matrix_A.rows, matrix_A.cols, name1);

    for (int i = 0; i < matrix_A.rows; ++i)
    {
        for (int j = 0; j < matrix_A.cols; ++j)
        {
            result.set(i, j, matrix_A.get(i, j) + matrix_B.get(i, j));
        }
    }

    // Add result matrix to matrices map
    getMatrices<T>()[name1] = result;

    // Display result matrix
    cout << "Result matrix: " << result.name << endl;
    displayMatrix(result);
}
template <typename T>
void subMatrix(Matrix<T> &matrix_A, Matrix<T> &matrix_B)
{
    std::ostringstream name;
    res[1][1]++;
    name << "Sub" << res[1][1];
    std::string name1 = name.str();
    Matrix<T> result(matrix_A.rows, matrix_A.cols, name1);

    for (int i = 0; i < matrix_A.rows; ++i)
    {
        for (int j = 0; j < matrix_A.cols; ++j)
        {
            result.set(i, j, matrix_A.get(i, j) - matrix_B.get(i, j));
        }
    }

    // Add result matrix to matrices map
    getMatrices<T>()[name1] = result;

    // Display result matrix
    cout << "Result matrix: " << result.name << endl;
    displayMatrix(result);
}

template <typename T>
void emMulMatrix(Matrix<T> &matrix_A, Matrix<T> &matrix_B)
{
    std::ostringstream name;
    res[1][3]++;
    name << "emMul" << res[1][3];
    std::string name1 = name.str();
    Matrix<T> result(matrix_A.rows, matrix_A.cols, name1);

    for (int i = 0; i < matrix_A.rows; ++i)
    {
        for (int j = 0; j < matrix_A.cols; ++j)
        {
            result.set(i, j, matrix_A.get(i, j) * matrix_B.get(i, j));
        }
    }

    // Add result matrix to matrices map
    getMatrices<T>()[name1] = result;

    // Display result matrix
    cout << "Result matrix: " << result.name << endl;
    displayMatrix(result);
}

template <typename T>
void mulMatrix(Matrix<T> &matrix_A, Matrix<T> &matrix_B)
{
    std::ostringstream name;
    res[1][2]++;
    name << "Mul" << res[1][2];
    std::string name1 = name.str();
    Matrix<T> result(matrix_A.rows, matrix_B.cols, name1);
    initializeToZero(result);
    // Perform matrix multiplication using nested loops
    for (int i = 0; i < matrix_A.rows; ++i)
    {
        for (int j = 0; j < matrix_B.cols; ++j)
        {
            T sum = 0;
            for (int k = 0; k < matrix_A.cols; ++k)
            {
                sum += matrix_A.get(i, k) * matrix_B.get(k, j);
            }
            result.set(i, j, sum);
        }
    }

    // Add result matrix to matrices map
    getMatrices<T>()[name1] = result;

    // Display result matrix
    cout << "Result matrix: " << result.name << endl;
    displayMatrix(result);
}

void printMatrixfromMatrices(string matrixName)
{
    // Check if the matrix exists in the map
    if (getMatrices<double>().count(matrixName))
    {
        auto &matrix = getMatrices<double>()[matrixName]; // Access matrix using its name
        cout << "Printing matrix '" << matrixName << "':" << endl;
        displayMatrix<double>(matrix);
    }
    else
    {
        cout << "Matrix '" << matrixName << "' not found." << endl;
    }
}

string getMatrixName()
{
    string name;

    while (true)
    {
        cout << "Enter the name of the matrix (or 'new' to create a new one): ";
        cin >> name;

        if (name == "new")
        {
            return ""; // Indicate new matrix creation
        }
        else if (getMatrices<double>().count(name))
        {
            // Matrix with the name exists, return it
            return name;
        }
        else
        {
            cout << "Matrix '" << name << "' does not exist." << endl;
        }
    }
}

string validName()
{
    while (true)
    {
        string matrixName = getMatrixName();
        if (matrixName.empty())
        {
            // Create a new matrix
            Matrix<double> newMatrix = createMatrix<double>();
            matrixName = newMatrix.name; // Store the new matrix
            cout << "New matrix created successfully." << endl;
            return matrixName;
        }
        else
        {
            // Use the existing matrix with the given name
            if (getMatrices<double>().count(matrixName))
            {
                // Access and perform operations on the existing matrix
                cout << "Using matrix '" << matrixName << "' for operations." << endl;
                printMatrixfromMatrices(matrixName);
                return matrixName;
            }
            else
            {
                cout << "Error: Matrix '" << matrixName << "' not found after creation attempt." << endl;
            }
        }
    }
}

void printMatrixNames()
{
    cout << "Matrix names:" << endl;

    // Assuming matrices() returns a map or unordered_map with string keys
    auto &matrixMap = getMatrices<double>();
    for (const auto &pair : matrixMap)
    {
        cout << pair.first << endl;
    }
}
template <typename T>
bool checkDimensions(Matrix<T> &matrix1, Matrix<T> &matrix2)
{
    return matrix1.rows == matrix2.rows && matrix1.cols == matrix2.cols;
}
template <typename T>
void initializeToZero(Matrix<T> &matrix)
{
    for (int i = 0; i < matrix.rows; ++i)
    {
        for (int j = 0; j < matrix.cols; ++j)
        {
            matrix.set(i, j, 0);
        }
    }
}
template <typename T>
Matrix<T> solveSystemOfLinearEquations(Matrix<T> &A, Matrix<T> &b)
{
    try
    {
        // Calculate the determinant of the coefficient matrix A
        auto det = determinant(A);

        if (det == 0)
        {
            // If the determinant is zero, the matrix is singular
            // Check if the system has a solution or not
            pair<Matrix<T>, Matrix<T>> lu = luDecomposition(A, A.rows);
            Matrix<T> &L = lu.first;
            Matrix<T> &U = lu.second;

            // Create a vector y using forward substitution
            Matrix<T> y(A.rows, 1, "y");
            for (int i = 0; i < A.rows; ++i)
            {
                y.set(i, 0, b.get(i, 0));
                for (int j = 0; j < i; ++j)
                {
                    y.set(i, 0, y.get(i, 0) - L.get(i, j) * y.get(j, 0));
                }
            }

            // Check if y is the zero vector
            bool isZeroVector = true;
            for (int i = 0; i < A.rows; ++i)
            {
                if (y.get(i, 0) != 0)
                {
                    isZeroVector = false;
                    break;
                }
            }

            if (isZeroVector)
            {
                // If y is the zero vector, the system has infinitely many solutions
                cout << "The system of equations has infinitely many solutions." << endl;
                return Matrix<T>(A.rows, 1, "temp"); // Return an arbitrary solution
            }
            else
            {
                // If y is not the zero vector, the system has no solution
                cout << "The system of equations has no solution." << endl;
                return Matrix<T>(0, 0, "temp"); // Return an empty matrix
            }
        }
        else
        {
            // If the determinant is not zero, calculate the inverse of A
            Matrix<T> inverseA = matrixInverse(A);
            Matrix<T> result(inverseA.rows, b.cols, "name1");
            initializeToZero(result);
            // Perform matrix multiplication using nested loops
            for (int i = 0; i < inverseA.rows; ++i)
            {
                for (int j = 0; j < b.cols; ++j)
                {
                    T sum = 0;
                    for (int k = 0; k < inverseA.cols; ++k)
                    {
                        sum += inverseA.get(i, k) * b.get(k, j);
                    }
                    result.set(i, j, sum);
                }
            }

            // Multiply the inverse of A with the constant vector b
            return result;
        }
    }
    catch (const std::exception &e)
    {
        // std::cerr << "Error: " << e.what() << std::endl;
        return Matrix<T>(0, 0, "zero");
    }
}
void solveSystemOfLinearEquations()
{
    int num_variables;

    // Get user input for the number of variables with error handling
    while (true)
    {
        cout << "Enter the number of variables (same as the number of equations): ";
        if (!(cin >> num_variables))
        {
            // Handle invalid input (clear input buffer)
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cout << "Invalid input. Please enter a positive integer." << endl;
            continue;
        }

        if (num_variables <= 0)
        {
            cout << "Error: The number of variables must be positive." << endl;
            continue;
        }

        break; // Exit the loop if valid input is received
    }

    // Create matrices for coefficients and constants (same number of rows)
    Matrix<double> A(num_variables, num_variables, "Coeff");
    Matrix<double> b(num_variables, 1, "Const");

    // Print table header for coefficients
    std::cout << endl
              << "Enter the coefficients of the equations (row-wise):" << endl;

    // Take input for coefficients and constant in a single line
    for (int i = 0; i < num_variables; ++i)
    {
        std::cout << "Eq " << i + 1 << " : " << endl;
        for (int j = 0; j < num_variables; ++j)
        {
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
    displayMatrix(A);
    cout << "Constant Vector b:" << endl;
    displayMatrix(b);

    Matrix<double> solution(1, num_variables, "Sol");
    try
    {
        solution = solveSystemOfLinearEquations(A, b);

        // Check if a solution was found
        if (solution.rows == 0)
        {
            cout << "The system of equations has no or infinite solution." << endl;
        }
        else
        {
            cout << "Solution to the system of linear equations:" << endl;
            for (int i = 0; i < num_variables; ++i)
            {
                cout << "x" << i + 1 << " = " << solution.get(i, 0) << endl;
            }
        }
    }
    catch (const std::exception &e)
    {
        // cerr << "Error: " << e.what() << endl;
    }
}

int main()
{
    char choice;

    while (true)
    {
        cout << "\nMenu:" << endl;

        cout << "1. Create a new matrix" << endl;
        cout << "2. Print a stored matrix" << endl;
        cout << "3. Unary Matrix Operations" << endl;            // Placeholder
        cout << "4. Binary Matrix Operations" << endl;           // Placeholder
        cout << "5. Addition of Array of Matrices" << endl;      // Placeholder
        cout << "6. Solving System of Linear Equations" << endl; // Placeholder
        cout << "7. Exit" << endl;

        cout << "Enter your choice: ";
        std::cin >> choice;

        // Error handling for invalid input (non-char)
        if (!std::cin)
        {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid input. Please enter a letter (1-7)." << std::endl;
            continue;
        }

        // Validate choice within range (1-7)
        if (choice < '1' || choice > '7')
        {
            std::cout << "Invalid choice. Please enter a letter between 1 and 7." << std::endl;
            continue;
        }

        switch (choice)
        {
        case '1':
        {
            // Call createMatrix to create a new matrix of any numeric type (int, double, etc.)
            cout << "\nCreating a new matrix...\n";
            createMatrix<double>(); // You can change the template argument here for different numeric types
            break;
        }
        case '2':
        {
            string matrixName;
            printMatrixNames();
            cout << "\nEnter the name of the matrix to print: ";
            cin >> matrixName;
            printMatrixfromMatrices(matrixName);
            break;
        }

        case '3':
            // Submenu for unary matrix operations
            while (true)
            {
                cout << "\nUnary Matrix Operations:" << endl;
                cout << "1. Matrix Trace" << endl;
                cout << "2. Matrix Average" << endl;
                cout << "3. Matrix Transpose " << endl;
                cout << "4. Matrix LU Decomposition " << endl;
                cout << "5. Matrix Determinant " << endl;
                cout << "6. Matrix Inverse " << endl;
                cout << "7. Matrix Rank " << endl;
                cout << "8. Matrix Eigen Vectors and Values " << endl;
                cout << "9. Go Back to Main Menu" << endl;

                int subChoice;
                cout << "Enter your choice: ";
                if (!(std::cin >> subChoice))
                {
                    std::cin.clear();                                                   // Clear the error state from cin
                    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Discard invalid input
                    std::cout << "Invalid input. Please enter a number (1-9)." << std::endl;
                    continue;
                }

                // Check if choice is within valid range (1-4)
                if (subChoice < 1 || subChoice > 9)
                {
                    std::cout << "Invalid choice. Please enter a number between 1 and 9." << std::endl;
                    continue;
                }

                if (subChoice != 9)
                {
                    string nameofA = validName();
                    auto &matrixA = getMatrices<double>()[nameofA];

                    switch (subChoice)
                    {
                    case 1:
                    {
                        matrixTrace(matrixA);
                        break;
                    }
                    case 2:
                    {
                        calculateColumnMeans(matrixA);
                        break;
                    }
                    case 3:
                    {
                        transpose(matrixA);
                        break;
                    }
                    case 4:
                    {
                        if (matrixA.rows == matrixA.cols)
                        {
                            luDecomposition(matrixA);
                            break;
                        }
                        else
                        {
                            cout << "The matrix is not a sqaure matrix and hence LU Decompostion can not be performed" << endl;
                            break;
                        }
                    }
                    case 5:
                    {
                        auto det = determinant(matrixA);
                        cout << "Determinant of the matrix: " << det << endl;
                        break;
                    }
                    case 6:
                    {
                        if (matrixA.rows == matrixA.cols)
                        {
                            auto x = matrixInverse(matrixA);
                            break;
                        }
                        else
                        {
                            cout << "The matrix is not a sqaure matrix and hence Inverse of the matrix does not exist" << endl;
                            break;
                        }
                    }
                    case 7:
                    {

                        int r = matrixRank(matrixA);
                        cout << "Rank of the matrix: " << r << endl;
                        break;
                    }
                    case 8:
                    {
                        if (matrixA.rows == matrixA.cols)
                        {
                            eigenValuesAndVectors(matrixA);
                            break;
                        }
                        else
                        {
                            cout << "The matrix is not a sqaure matrix " << endl;
                        }
                    }
                    case 9:
                    {
                        // Exit the unary operations submenu and go back to main menu
                        break;
                    }
                    default:
                        cout << "Invalid choice. Please try again." << endl;
                    }
                }
                if (subChoice == 9)
                {
                    break;
                }
            }
            break;
        case '4':
        {
            // Binary matrix operations submenu
            while (true)
            {
                cout << "\nBinary Matrix Operation:" << endl;
                cout << "1. Matrix Addition" << endl;
                cout << "2. Matrix Subtraction" << endl;
                cout << "3. Matrix Multiplication" << endl;
                cout << "4. Matrix Element-wise Multiplication" << endl;
                cout << "5. Go Back to Main Menu" << endl;

                int subChoice;
                cout << "Enter your choice: ";
                if (!(std::cin >> subChoice))
                {
                    std::cin.clear();
                    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                    std::cout << "Invalid input. Please enter a number (1-5)." << std::endl;
                    continue;
                }

                if (subChoice < 1 || subChoice > 5)
                {
                    std::cout << "Invalid choice. Please enter a number between 1 and 5." << std::endl;
                    continue;
                }

                if (subChoice != 5)
                {
                    cout << "Matrix 1:" << endl;
                    string nameofA = validName();
                    auto &matrixA = getMatrices<double>()[nameofA];
                    cout << "Matrix 2:" << endl;
                    string nameofB = validName();
                    auto &matrixB = getMatrices<double>()[nameofB];

                    switch (subChoice)
                    {
                    case 1:
                    {
                        if (checkDimensions(matrixA, matrixB))
                        {
                            addMatrix(matrixA, matrixB);
                        }
                        else
                        {
                            cout << "The matrices should be of equal dimensions " << endl;
                        }

                        break;
                    }

                    case 2:
                    {
                        if (checkDimensions(matrixA, matrixB))
                        {
                            subMatrix(matrixA, matrixB);
                        }
                        else
                        {
                            cout << "The matrices should be of equal dimensions " << endl;
                        }

                        break;
                    }
                    case 3:
                    {
                        if (matrixB.rows == matrixA.cols)
                        {
                            mulMatrix(matrixA, matrixB);
                        }
                        else
                        {
                            cout << "Dimensions are not compatible" << endl;
                        }
                        break;
                    }
                    case 4:
                    {
                        if (checkDimensions(matrixA, matrixB))
                        {
                            emMulMatrix(matrixA, matrixB);
                        }
                        else
                        {
                            cout << "The matrices should be of equal dimensions " << endl;
                        }
                        break;
                    }
                    default:
                        cout << "Invalid choice. Please try again." << endl;
                    }
                }

                if (subChoice == 5)
                {
                    break;
                }
            }
            break;
        }
        break;
        case '5':
        {
            // Addition of array of matrices
            int numMatrices;
            do
            {
                std::cout << "Enter the number of matrices to add (should be 2 or more): ";
                if (!(std::cin >> numMatrices))
                {
                    std::cin.clear();
                    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                    std::cout << "Invalid input. Please enter a valid number (2 or more)." << std::endl;
                    continue;
                }

                if (numMatrices < 2)
                {
                    std::cout << "Number of matrices should be 1 or more. Please try again." << std::endl;
                }
                else
                {
                    break;
                }
            } while (true);
            Vector<string> matrixNames;
            for (int i = 0; i < numMatrices; ++i)
            {
                std::string name;
                std::cout << "Enter the name of matrix " << i + 1 << ": ";
                std::cin >> name;
                matrixNames.PushBack(name);
            }

            // Fetch the matrices from the map
            Vector<Matrix<double>> Tmatrices;
            for (size_t i = 0; i < matrixNames.Size(); ++i)
            {
                if (getMatrices<double>().find(matrixNames[i]) == getMatrices<double>().end())
                {
                    cout << ("Matrix " + matrixNames[i] + " does not exist.");
                    continue;;
                }
                Tmatrices.PushBack(getMatrices<double>()[matrixNames[i]]);
            }
            int x = 0;
            // Check if all matrices have the same dimensions
            for (size_t i = 1; i < Tmatrices.Size(); ++i)
            {
                if (!checkDimensions(Tmatrices[0], Tmatrices[i]))
                {
                    cout << ("Matrices have different dimensions.") << endl;
                    x = 1;
                    break;
                }
            }
            if (x==1)
            {
                break;
            }
            size_t size = Tmatrices.Size();

            if (size > 2)
            {
                Matrix<double> result = addMatrices(Tmatrices[0], Tmatrices[1], Tmatrices[size - 1]);
                static int resultCount = 0;
                std::string resultName = "Result" + std::to_string(resultCount++);
                // Store the result matrix in the matrices map
                getMatrices<double>()[resultName] = result;
                // Display the resulting matrix
                displayMatrix(result);
            }
            else
            {
                if(size==2)
                {
                    addMatrix(Tmatrices[0], Tmatrices[1]);
                }
                else
                {
                    cout<<"The number of valid inputs are less than 2"<<endl;
                }
            }

            break;
        }
        // break;
        case '6':
            solveSystemOfLinearEquations();
            break;

        case '7':
            cout << "Exiting..." << endl;
            return 0;

        default:
            cout << "Invalid choice. Please try again." << std::endl;
        }
    }
    return 0;
}
