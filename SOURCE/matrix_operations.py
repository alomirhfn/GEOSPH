from pysph.sph.equation import Equation
from compyle.api import declare
from math import pow, sqrt, pi, acos, atan2, cos, sin, fabs, isinf, isnan

def vector_norm(mat=[1.0, 1.0, 1.0]):
    """
    Returns the L2 norm of a 3 element vector
    """
    i = declare("int")

    norm = 0
    for i in range(3):
        norm += mat[i]*mat[i]

    return sqrt(norm)


def matrix_norm(mat=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]):
    """
    Returns the L2 norm of a 3x3 matrix
    """
    i = declare("int")

    norm = 0
    for i in range(9):
        norm += mat[i]*mat[i]

    return sqrt(norm)


def vector_normalize(vec=[1.0, 1.0, 1.0], vec_size=3):
    r"""
    Returns a normalized vector, i.e. with unity norm.

    Parameters
    ----------
    :param vec: list representing a vector or vector of vectors
    :param vec_size: integer representing the number elements in the vector

    Output
    -----------
    :return: None
    """
    i, j, num_vecs = declare("int", 3)

    num_vecs = int(vec_size / 3)
    for i in range(num_vecs):
        vec_sum2 = 0.0
        for j in range(3):
            vec_sum2 += pow(vec[3*i + j], 2)
        vec_sum = sqrt(vec_sum2)
        for j in range(3):
            vec[3*i + j] /= vec_sum


def vector_outer_product(vec1=[1.0, 1.0, 1.0], vec2=[1.0, 1.0, 1.0],
                         mat=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                         n=2):
    r"""
    Given 2 vectors (1x3), calculates the outer product, which is a matrix of
    size (3x3).

    Parameters
    ----------
    :param vec1: list representing the first vector
    :param vec2: list representing the second vector
    :param mat: resultant matrix
    :param n: dimension of the vector

    Output
    -----------
    :return: None
    """
    i, j = declare("int", 2)

    for i in range(n):
        for j in range(n):
            mat[n*i + j] = vec1[i]*vec2[j]


def matrix_multiply(mat1=[1.0, 1.0], mat2=[1.0, 1.0], res=[1.0, 1.0], n=2):
    r"""
    Multiply two square matrices. Stores the result in 'res'.

    Parameters
    ----------
    :param mat1: list representing matrix 1
    :param mat2: list representing matrix 2
    :param res: list to hold the result
    :param n: integer representing the number of rows (or columns)

    Output
    -----------
    :return: None
    """
    i, j, k = declare('int', 3)

    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += mat1[n*i + k] * mat2[n*k + j]
            res[n*i + j] = s


def matrix_trace(mat=[1.0, 1.0, 1.0, 1.0], n=2):
    r"""
    Returns the sum of the diagonal elements of a square matrix.

    Parameters
    ----------
    :param mat: list representing a matrix
    :param n: integer representing the number of rows (or columns)

    Output
    -----------
    :return: float corresponding to the matrix's trace
    """
    if n == 2:
        return mat[0] + mat[3]

    else:
        return mat[0] + mat[4] + mat[8]


def matrix_determinant_2x2(mat=[1.0, 1.0, 1.0, 1.0]):
    r"""
    Returns the determinant of a square 2x2 matrix.

    Parameters
    ----------
    :param mat: list representing a matrix

    Output
    -----------
    :return: float corresponding to the matrix's determinant
    """
    return mat[0]*mat[3] - mat[1]*mat[2]


def matrix_determinant(mat=[1.0, 1.0, 1.0, 1.0], n=2):
    r"""
    Returns the determinant of 3x3 matrix.

    Parameters
    ----------
    :param mat: list representing a matrix
    :param n: integer representing the number of rows (or columns)

    Output
    -----------
    :return: float corresponding to the matrix's determinant
    """
    i = declare("int")
    sub_a, sub_b, sub_c = declare("matrix(4)", 3)

    if n == 2:
        return matrix_determinant_2x2(mat)

    else:
        res = 0.0

        # Matrix co-factors
        a_coff, b_coff, c_coff = mat[0:3]

        # Extract sub-matrices
        for i in range(2):
            sub_a[i] = mat[i+4]
            sub_a[i+2] = mat[i+7]
            sub_b[i] = mat[2*i + 3]
            sub_b[i+2] = mat[2*i + 6]
            sub_c[i] = mat[i+3]
            sub_c[i+2] = mat[i+6]

        # Determinant calculation
        res += (
            a_coff*matrix_determinant_2x2(sub_a)
            - b_coff*matrix_determinant_2x2(sub_b)
            + c_coff*matrix_determinant_2x2(sub_c)
        )

        return res


def matrix_transpose(mat=[1.0, 1.0, 1.0, 1.0], res=[1.0, 1.0, 1.0, 1.0], n=2):
    r"""
    Transpose a square matrix, by flipping rows by columns of same index.
    Stores the result in 'res'.

    Parameters
    ----------
    :param mat: list representing matrix
    :param res: list to hold the result
    :param n: integer representing the number of rows (or columns)

    Output
    -----------
    :return: None
    """
    i, j, idx, m = declare("int", 4)

    if n == 2:
        res[0] = mat[0]
        res[1] = mat[2]
        res[2] = mat[1]
        res[3] = mat[3]

    else:
        m = n + 1
        for i in range(n):
            for j in range(n):
                idx = n*i + j
                mij = mat[idx]
                if idx % m == 0:
                    res[idx] = mat[idx]
                else:
                    res[idx] = mat[n*j + i]


def matrix_eigenvalues(mat=[1.0, 1.0, 1.0, 1.0], eigvals=[1.0, 1.0], n=2):
    r"""
    Algorithm based on the closed form solution of eigenvalues of a symmetric
    3x3 matrix proposed in:

    Smith, O.K. (1961) Eigenvalues of a symmetric 3x3 matrix. Communications
    of the ACM 4(4), p168.

    (https://dl.acm.org/citation.cfm?doid=355578.366316 - Accessed 02/21/2019)

    For an explanation on the meaning of each term in the code below, refer to
    the paper.

    Parameters
    -----------
    :param mat: list representing the matrix in Voight notation
    :param eigvals: list with eigenvalues sorted in ascending order
    :param n: int representing the number or rows (columns) in 'mat'

    Output
    -----------
    :return: None
    """
    i = declare("int")
    eigvals_temp = declare("matrix(3)")
    mat_b = declare("matrix(9)")

    if n == 2:
        mat_tr = matrix_trace(mat, 2)  # Trace of the matrix
        s4ac = sqrt(pow(mat_tr, 2.0) - 4*matrix_determinant(mat, 2))
        eigvals[0] = (mat_tr - s4ac) / 2.0
        eigvals[1] = (mat_tr + s4ac) / 2.0
        eigvals[2] = 0.0

    else:
        m = matrix_trace(mat, 3) / 3.0
        for i in range(9):
            mat_b[i] = mat[i]
            if i % 4 == 0:
                mat_b[i] -= m

        q = matrix_determinant(mat_b, 3) / 2.0

        p = 0.0
        for i in range(9):
            p += pow(mat_b[i], 2.0) / 6.0

        c = pow(p, 3.0) - pow(q, 2.0)
        if c < 0.0:
            c = 0.0
        phi = atan2(sqrt(c), q) / 3.0

        eigvals_temp[0] = m + 2.0*sqrt(p) * cos(phi)
        eigvals_temp[1] = m - sqrt(p)*(cos(phi) + sqrt(3.0)*sin(phi))
        eigvals_temp[2] = m - sqrt(p) * (cos(phi) - sqrt(3.0) * sin(phi))

        # Sorting the eigenvalues
        if eigvals_temp[0] < eigvals_temp[1]:
            if eigvals_temp[1] < eigvals_temp[2]:
                eigvals[0] = eigvals_temp[0]
                eigvals[1] = eigvals_temp[1]
                eigvals[2] = eigvals_temp[2]
            elif eigvals_temp[0] < eigvals_temp[2]:
                eigvals[0] = eigvals_temp[0]
                eigvals[1] = eigvals_temp[2]
                eigvals[2] = eigvals_temp[1]
            else:
                eigvals[0] = eigvals_temp[2]
                eigvals[1] = eigvals_temp[0]
                eigvals[2] = eigvals_temp[1]
        else:
            if eigvals_temp[2] < eigvals_temp[1]:
                eigvals[0] = eigvals_temp[2]
                eigvals[1] = eigvals_temp[1]
                eigvals[2] = eigvals_temp[0]
            elif eigvals_temp[0] < eigvals_temp[2]:
                eigvals[0] = eigvals_temp[1]
                eigvals[1] = eigvals_temp[0]
                eigvals[2] = eigvals_temp[2]
            else:
                eigvals[0] = eigvals_temp[1]
                eigvals[1] = eigvals_temp[2]
                eigvals[2] = eigvals_temp[0]


def matrix_eigenvectors(mat=[1.0, 1.0, 1.0, 1.0], eigvals=[1.0, 1.0],
                        eigvecs=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], n=2):
    r"""
    Algorithms taken from:
        https://www.math.harvard.edu/archive/21b_fall_04/exhibits/2dmatrices/
    as of 03/07/2019.

    and

        https://en.wikipedia.org/wiki/Eigenvalue_algorithm as of 02/20/2019.

    Parameters
    -----------
    :param mat: list representing the matrix in Voight notation
    :param eigvals: list with eigenvalues sorted in ascending order
    :param eigvecs: list holding the resulting eigenvectors for 'mat'
    :param n: int representing the number or rows (columns) in 'mat'

    Output
    -----------
    :return: None
    """
    tol = 1e-12
    i, j, k, max_index = declare("int", 4)
    col_sum = declare("matrix(3)")
    temp_a, temp_b, temp_c = declare("matrix(9)", 3)

    if n == 2:
        # Check for diagonal matrices.
        diag_sum = fabs(mat[1]) + fabs(mat[2])
        if diag_sum < tol:
            eigvecs[0] = 1.0
            eigvecs[2] = 1.0

        else:
            if mat[2] != 0:
                eigvecs[0] = eigvals[0] - mat[3]
                eigvecs[1] = mat[2]
                eigvecs[3] = eigvals[1] - mat[3]
                eigvecs[4] = mat[2]
            elif mat[1] != 0:
                eigvecs[0] = mat[1]
                eigvecs[1] = eigvals[0] - mat[0]
                eigvecs[3] = mat[1]
                eigvecs[4] = eigvals[1] - mat[0]
            else:
                eigvecs[0] = 0.0
                eigvecs[1] = 1.0
                eigvecs[3] = 1.0
                eigvecs[4] = 0.0

            eigvecs[2] = 0.0
            eigvecs[5] = 0.0
            vector_normalize(eigvecs, 3 * n)

    else:

        # Check for diagonal matrices.
        diag_sum = 0.0
        for i in range(9):
            if i % 4 != 0.0:
                diag_sum += fabs(mat[i])
        if diag_sum < tol:
            eigvecs[0] = 1.0
            eigvecs[4] = 1.0
            eigvecs[8] = 1.0

        else:
            for i in range(3):
                max_value = -1.0e32

                for j in range(9):
                    temp_a[j] = mat[j]
                    temp_b[j] = mat[j]

                if i == 0:
                    for j in range(0, 9, 4):
                        temp_a[j] -= eigvals[1]
                        temp_b[j] -= eigvals[2]

                    matrix_multiply(temp_a, temp_b, temp_c, 3)

                    # Find column with maximum sum
                    for j in range(3):
                        col_sum[j] = 0.0
                        for k in range(3):
                            col_sum[j] += abs(temp_c[3*k + j])
                    for j in range(3):
                        if col_sum[j] > max_value:
                            max_index = j
                            max_value = col_sum[j]

                    # Assign values to the vector list
                    for k in range(3):
                        eigvecs[k] = temp_c[3*k + max_index]

                elif i == 1:
                    for j in range(0, 9, 4):
                        temp_a[j] -= eigvals[0]
                        temp_b[j] -= eigvals[2]
                    matrix_multiply(temp_a, temp_b, temp_c, 3)

                    for j in range(3):
                        col_sum[j] = 0.0
                        for k in range(3):
                            col_sum[j] += abs(temp_c[3*k + j])
                    for j in range(3):
                        if col_sum[j] > max_value:
                            max_index = j
                            max_value = col_sum[j]

                    for k in range(3):
                        eigvecs[k+3] = temp_c[3*k + max_index]

                else:
                    for j in range(0, 9, 4):
                        temp_a[j] -= eigvals[0]
                        temp_b[j] -= eigvals[1]
                    matrix_multiply(temp_a, temp_b, temp_c, 3)

                    for j in range(3):
                        col_sum[j] = 0.0
                        for k in range(3):
                            col_sum[j] += abs(temp_c[3*k + j])
                    for j in range(3):
                        if col_sum[j] > max_value:
                            max_index = j
                            max_value = col_sum[j]

                    for k in range(3):
                        eigvecs[k+6] = temp_c[3*k + max_index]

            vector_normalize(eigvecs, 3*n)

def matrix_add(mat1=[1.0, 1.0, 1.0, 1.0], mat2=[1.0, 1.0, 1.0, 1.0],
               res=[1.0, 1.0, 1.0, 1.0], n=2, s=1.0):
    r"""
    This equation adds two matrices of equal dimension (mat1, mat2) and returns
    the result in a second matrix (res).

    :param mat1:
    :param mat2:
    :param res:
    :param s: -1.0 if subtraction
    :param n:
    :return: None
    """
    i, r = declare("int", 2)
    r = n * n
    for i in range(r):
        res[i] = mat1[i] + s*mat2[i]


def matrix_multiply_vector(mat=[1.0, 1.0, 1.0, 1.0], vec=[0.0, 0.0],
                           res=[0.0, 0.0], n=2):
    r"""
    This equation multiplies a square matrix (mat) by a vector (vec) and
    returns the result in a second vector (res).

    :param mat:
    :param vec:
    :param res:
    :param n:
    :return:
    """
    i, j = declare("int", 2)

    # Initialize result vector
    for i in range(n):
        res[i] = 0.0

    for i in range(n):
        for j in range(n):
            res[i] += mat[n*i + j]*vec[j]


def matrix_exponentiation(mat=[0.0, 0.0], res=[0.0, 0.0], exp=0, n=2):
    r"""
    This function returns the input matrix (mat) to the power of (exp), i.e.
    it multiplies the matrix by itself exp-times.

    :param mat: List (double)
    :param res: List (double)
    :param exp: (int)
    :param n: (int)
    :return: None
    """
    i, j, k, m = declare("int", 4)
    res_old = declare("matrix(36)")

    for i in range(n*n):
        res[i] = 0.0

    m = n + 1
    if exp == 0:
        for i in range(n*n):
            if i % m == 0:
                res[i] = 1.0

    else:
        matrix_exponentiation(mat, res, exp-1, n)

        for i in range(n*n):
            res_old[i] = res[i]

        for i in range(n):
            for j in range(n):
                s = 0.0
                for k in range(n):
                    s += mat[n*i + k]*res_old[n*k + j]
                res[n*i + j] = s


def augment_voigt_tensor(mat=[0.0, 0.0, 0.0], res=[0.0, 0.0, 0.0], n=2):
    r"""

    :param mat:
    :param res:
    :param n:
    :return:
    """
    i = declare("int")

    for i in range(2*n):
        m = mat[i]
        if i < n:
            res[4*i] = m
        elif i < 2*n-1:
            res[i-2] = m
            res[(i-2)*n] = m
        else:
            res[i] = m
            res[i+2] = m


def basic_matrix_inverse(mat=[0.0, 0.0], inv=[0.0, 0.0]):
    """
    This function performs a brute-force calculation of the inverse of a
    3x3 matrix.
    """

    a = mat[0]
    b = mat[1]
    c = mat[2]
    d = mat[3]
    e = mat[4]
    f = mat[5]
    g = mat[6]
    h = mat[7]
    i = mat[8]

    det = (a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g))

    inv[0] = 1.0/det * (e*i - f*h)
    inv[1] = 1.0/det * (c*h - b*i)
    inv[2] = 1.0/det * (b*f - c*e)
    inv[3] = 1.0/det * (f*g - d*i)
    inv[4] = 1.0/det * (a*i - c*g)
    inv[5] = 1.0/det * (c*d - a*f)
    inv[6] = 1.0/det * (d*h - e*g)
    inv[7] = 1.0/det * (b*g - a*h)
    inv[8] = 1.0/det * (a*e - b*d)


def matrix_inverse(mat=[0.0, 0.0, 0.0, 0.0], res=[0.0, 0.0, 0.0, 0.0], n=2):
    r"""
    This function takes a symmetric, positive-definite matrix (mat) in Voigt
    notation and returns its inverse. It first performs a LDL (or Cholesky)
    decomposition to make the matrix lower triangular and then invert it.

    References
    -----------
    https://en.wikipedia.org/wiki/Cholesky_decomposition

    https://math.stackexchange.com/questions/1003801/...
        ...inverse-of-an-invertible-upper-triangular-matrix-of-order-3

    (As of 04/7/2020)

    :param mat:
    :param res:
    :param n:
    :return: None
    """
    i, j, k, idx, m, r = declare("int", 6)
    low_mat, low_inv, temp, diag_inv, band, mat_t, d = declare("matrix(36)", 7)
    ten = declare("matrix(15)")
    matv, temp2 = declare("matrix(36)", 2)

    # TODO: First test if matrix is positive-definite!

    # This is necessary to convert a tensor in Voigt notation to a full square
    #  tensor
    if n % 2 > 0:
        r = int((n*n + n)/2)
        for i in range(r):
            ten[i] = mat[i]
        augment_voigt_tensor(ten, matv, n)
    else:
        for i in range(n*n):
            matv[i] = mat[i]

    # Initialize lower triangular matrix, low_mat
    m = int(n + 1)
    for i in range(n*n):
        low_mat[i] = 0.0
        diag_inv[i] = 0.0
        band[i] = 0.0
        temp[i] = 0.0
        temp2[i] = 0.0
        d[i] = 0.0

    # # Perform Cholesky decomposition
    # for i in range(n):
    #     for j in range(n):
    #         idx = n*i + j
    #         lsum = 0.0
    #
    #         # Diagonal terms
    #         if idx % (n+1) == 0:
    #             for k in range(j):
    #                 ljk = low_mat[n*i + k]
    #                 lsum += ljk*ljk
    #             low_mat[idx] = sqrt(matv[idx] - lsum)
    #
    #         # Off-diagonal terms
    #         elif i > j:
    #             for k in range(j):
    #                 lsum += low_mat[n*i + k]*low_mat[n*j + k]
    #             low_mat[idx] = (matv[idx] - lsum) / low_mat[(n+1)*j]

    # Perform LDL decomposition
    for i in range(n*n):
        if i % m == 0:
            low_mat[i] = 1.0

    for i in range(n):
        for j in range(n):
            idx = n*i + j
            lsum = 0.0

            # Diagonal terms
            if idx % m == 0:
                for k in range(j):
                    ljk = low_mat[n*i + k]
                    lsum += ljk*ljk*d[m*k]
                d[idx] = matv[idx] - lsum

            # Off-diagonal terms
            elif i > j:
                for k in range(j):
                    lsum += low_mat[n*i + k]*low_mat[n*j + k]*d[m*k]
                low_mat[idx] = (matv[idx] - lsum) / d[m*j]

    # Perform diagonalization
    for i in range(n):
        d[i*m] = sqrt(d[i*m])
    matrix_multiply(low_mat, d, temp, n)
    for i in range(n*n):
        low_mat[i] = temp[i]

    # Inversion of the lower diagonal matrix
    for i in range(n*n):
        if i % m == 0:
            diag_inv[i] = -1.0/low_mat[i]  # Matrix with inverse diag terms
        else:
            band[i] = low_mat[i]  # Matrix diagonal terms equal to zero

    matrix_multiply(diag_inv, band, mat_t, n)

    for i in range(n):
        matrix_exponentiation(mat_t, temp, i, n)
        for j in range(n*n):
            temp2[j] += temp[j]

    matrix_multiply(temp2, diag_inv, low_inv, n)
    matrix_transpose(low_inv, temp, n)
    matrix_multiply(temp, low_inv, temp2, n)

    # Restore vector back to Voigt notation
    if n % 2 > 0:
        tensor_voight(temp2, res, n)


def full_matrix_inverse(mat=[0.0, 0.0, 0.0, 0.0], res=[0.0, 0.0, 0.0, 0.0],
                        n=2):
    r"""
    This function takes a symmetric, positive-definite full matrix (mat) - nxn,
    and returns its inverse in Voigt notation. It first performs a LDL
    (or Cholesky) decomposition to make the matrix lower triangular and then
    invert it.

    References
    -----------
    https://en.wikipedia.org/wiki/Cholesky_decomposition

    https://math.stackexchange.com/questions/1003801/...
        ...inverse-of-an-invertible-upper-triangular-matrix-of-order-3

    (As of 04/7/2020)

    :param mat:
    :param res:
    :param n:
    :return: None
    """
    i, j, k, idx, m, r = declare("int", 6)
    low_mat, low_inv, temp, diag_inv, band, mat_t, d = declare("matrix(36)", 7)
    temp2 = declare("matrix(36)")

    # TODO: First test if matrix is positive-definite!

    # Initialize lower triangular matrix, low_mat
    m = n + 1
    for i in range(n*n):
        low_mat[i] = 0.0
        diag_inv[i] = 0.0
        band[i] = 0.0
        temp[i] = 0.0
        temp2[i] = 0.0
        d[i] = 0.0

    # # Perform Cholesky decomposition
    # for i in range(n):
    #     for j in range(n):
    #         idx = n*i + j
    #         lsum = 0.0
    #
    #         # Diagonal terms
    #         if idx % m == 0:
    #             for k in range(j):
    #                 ljk = low_mat[n*i + k]
    #                 lsum += ljk*ljk
    #             low_mat[idx] = sqrt(mat[idx] - lsum)
    #
    #         # Off-diagonal terms
    #         elif i > j:
    #             for k in range(j):
    #                 lsum += low_mat[n*i + k]*low_mat[n*j + k]
    #             low_mat[idx] = (mat[idx] - lsum) / low_mat[(n+1)*j]

    # Perform LDL decomposition
    for i in range(n*n):
        if i % m == 0:
            low_mat[i] = 1.0

    for i in range(n):
        for j in range(n):
            idx = n*i + j
            lsum = 0.0

            # Diagonal terms
            if idx % m == 0:
                for k in range(j):
                    ljk = low_mat[n*i + k]
                    lsum += ljk*ljk*d[m*k]
                d[idx] = mat[idx] - lsum

            # Off-diagonal terms
            elif i > j:
                for k in range(j):
                    lsum += low_mat[n*i + k]*low_mat[n*j + k]*d[m*k]
                low_mat[idx] = (mat[idx] - lsum) / d[m*j]

    # Perform diagonalizing
    for i in range(n):
        d[i*m] = sqrt(d[i*m])
    matrix_multiply(low_mat, d, temp, n)
    for i in range(n*n):
        low_mat[i] = temp[i]

    # Inversion of the lower diagonal matrix
    for i in range(n*n):
        if i % m == 0:
            diag_inv[i] = -1.0/low_mat[i]  # Matrix with inverse diag terms
        else:
            band[i] = low_mat[i]  # Matrix diagonal terms equal to zero

    matrix_multiply(diag_inv, band, mat_t, n)

    for i in range(n):
        matrix_exponentiation(mat_t, temp, i, n)
        for j in range(n*n):
            temp2[j] += temp[j]

    matrix_multiply(temp2, diag_inv, low_inv, n)
    matrix_transpose(low_inv, temp, n)
    matrix_multiply(temp, low_inv, temp2, n)

    for i in range(n*n):
        res[i] = temp2[i]

def tensor_voight(A=[0.0, 0.0, 0.0, 0.0], res=[0.0, 0.0, 0.0], n=2):
    r"""
    This function transforms a full nxn square symmetric tensor in its Voight
    form, i.e., a vector of dimensions 1 by (n^2 + n) / 2.

    :param A:
    :param res:
    :param n:
    :return:
    """
    i, j, m, idx, idx2 = declare("int", 5)

    m = n + 1
    idx = 0
    idx2 = 0
    for i in range(n*n):
        if i % m == 0:
            res[idx] = A[i]
            idx += 1
        elif i < idx*n:
            res[n + idx2] = A[i]
            idx2 += 1

def matrix_inverse_exact(mat=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         res=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         n=2, dbg_vec=[0.0, 0.0, 0.0]):
    r"""
        This function calculates the inverse of a symmetric 3x3 (or 4x4) matrix
         written in flattened form, i.e., indices of entries are in sequence,
         such that for a 3x3 matrix:
            A(0) = Axx
            A(1) = Axy
            A(2) = Axz
            A(3) = Ayx = Axy
            A(4) = Ayy
            A(5) = Ayz
            A(6) = Azx = Axz
            A(7) = Azy = Ayz
            A(8) = Azz

        The analytical solution was obtained using Wolfram alpha on August 4th,
        2023.

        .. math::

            \begin{bmatrix}
                a & b & c & d \\
                b & e & f & g\\
                c & f & h & p\\
                d & g & p & q
            \end{bmatrix}^{-1} = \text{det}^{-1}
            \begin{bmatrix}
                ehq - ep^2 - f^2q + 2fgp - g^2h &
                -bhq + bp^2 + cfq - cgp - dfp + dgh &
                bfq - bgp - ceq + cg^2 + dep - dfg &
                -bfp + bgh + cep - cfg - deh + df^2 \\
                - bhq + bp^2 + cfq - cgp - dfp + dgh &
                ahq - ap^2 - c^2q + 2cdp - d^2h &
                -afq + agp + bcq - bdp - cdg + d^2f &
                afp - agh - bcp + bdh + c^2g - cdf \\
                bfq - bgp - ceq + cg^2 + dep - dfg &
                -afq + agp + bcq - bdp - cdg + d^2f &
                aeq - ag^2 - b^2q + 2bdg - d^2e &
                -aep + afg + b^2p - bcg - bdf + cde \\
                -bfp + bgh + cep - cfg - deh + df^2 &
                afp - agh - bcp + bdh + c^2g - cdf &
                -aep + afg + b^2p - bcg - bdf + cde &
                aeh - af^2 - b^2h + 2bcf - c^2e
            \end{bmatrix}

        with

        .. math::

            \text{det}^{-1} = \frac{-1}{aehq - aep^2 - af^2q + 2afgp - ag^2h -
            b^2hq + b^2p^2 + 2bcfq - 2bcgp - 2bdfp + 2bdgh - c^2eq + c^2g^2 +
            2cdep - 2cdfg - d^2eh + d^2f^2}

        The calculation of the condition number was taken from this Wikipedia
        article on 02/01/2024: https://en.wikipedia.org/wiki/Condition_number

        Input:
        -----------
            :param list mat: a flattened matrix (list or array) of size 4
              (2x2 matrix), 9 (3x3 matrix), or 16 (4x4)
            :param list res: a flattened resultant matrix (list or array) of
              size 9 (for 2x2 or 3x3 input) or size 16 (for 4x4 input)
            :param int n: matrix dimension 2 for 2x2, 3 for 3x3, or 4 for 4x4
            :param dbg_vec: a flattened matrix (list or array) of size 2
              (1x2 matrix) where the first entry holds the error/debug flag,
              and the second entry holds the value of the determinant

        Output
        -----------
        :return: None
    """
    evals = declare("matrix(3)")

    # Read the correct components of the input matrix depending on the matrix
    #  shape
    if n == 2:
        a = mat[0]
        b = mat[1]
        e = mat[3]
        c = d = f = g = p = 0.0
        h = q = 1.0

    elif n == 3:
        a = mat[0]
        b = mat[1]
        c = mat[2]
        e = mat[4]
        f = mat[5]
        h = mat[8]
        d = g = p = 0.0
        q = 1.0

    else:
        a = mat[0]
        b = mat[1]
        c = mat[2]
        d = mat[3]
        e = mat[5]
        f = mat[6]
        g = mat[7]
        h = mat[10]
        p = mat[11]
        q = mat[15]

    # Determinant
    det = (a*e*h*q - a*e*p*p - a*f*f*q + 2*a*f*g*p - a*g*g*h - b*b*h*q +
           b*b*p*p + 2*b*c*f*q - 2*b*c*g*p - 2*b*d*f*p + 2*b*d*g*h - c*c*e*q +
           c*c*g*g + 2*c*d*e*p - 2*c*d*f*g - d*d*e*h + d*d*f*f)

    dbg_vec[1] = det

    # Condition number if the matrix is 2x2 or 3x3
    k = 1e12
    if det > 0.0 and isinf(det) == False and isnan(det) == False:
        if n < 4:
            eig_vals_3x3_analytical(mat, evals)
            e1 = fabs(evals[0])
            e2 = fabs(evals[1])
            e3 = fabs(evals[2])
            k = max(max(e1, e2), e3) / min(min(e1, e2), e3)

        else:
            # TODO: This needs to be implemented, i.e., eigvals for 4x4 matrix
            #  as in else statement below (01/30/2024)
            k = 1.0

    dbg_vec[2] = k

    # Check for too high a condition number which overcorrects the kernel grad.
    #
    # NOTE (03/26/2024): After some 2D & 3D testing, and based on an email
    # exchange with PySPH people, to limit overcorrecting and not to correct
    # the kernel gradient particles on free surfaces, a condition number of up
    # to 2 was chosen as the normal range, which in turn, limits the correction
    # matrix to a factor around 2. A condition number up to 10 can be used with
    # corrections of the order of 10, but anything above 2 can generate
    # numerical issues when particles get very disordered or without enough
    # neighbors.
    #
    # This value of 2 is (approximately) equivalent to the tolerance of 1.0 set
    # for the PySPH correction version.
    if k > 2.0:

        # Result for 2x2 and 3x3 matrices
        if n < 4:
            res[0] = 1.0
            res[1] = res[3] = 0
            res[2] = res[6] = 0
            res[4] = 1.0
            res[5] = res[7] = 0
            res[8] = 1.0

        # Result for 4x4 matrix
        else:
            res[0] = 1.0
            res[5] = 1.0
            res[10] = 1.0
            res[15] = 1.0
            res[1] = res[4] = 0.0
            res[2] = res[8] = 0.0
            res[3] = res[12] = 0.0
            res[6] = res[9] = 0.0
            res[7] = res[13] = 0.0
            res[11] = res[14] = 0.0

        dbg_vec[0] = -1.0

    else:

        if n < 4:
            res[0] = (e*h*q - e*p*p - f*f*q + 2*f*g*p - g*g*h) / det
            res[1] = res[3] = (-b*h*q + b*p*p + c*f*q - c*g*p - d*f*p +
                               d*g*h) / det
            res[2] = res[6] = (b*f*q - b*g*p - c*e*q + c*g*g + d*e*p -
                               d*f*g) / det
            res[4] = (a*h*q - a*p*p - c*c*q + 2*c*d*p - d*d*h) / det
            res[5] = res[7] = (-a*f*q + a*g*p + b*c*q - b*d*p - c*d*g +
                               d*d*f) / det
            res[8] = (a*e*q - a*g*g - b*b*q + 2*b*d*g - d*d*e) / det

        else:
            res[0] = (e*h*q - e*p*p - f*f*q + 2*f*g*p - g*g*h) / det
            res[1] = res[4] = (-b*h*q + b*p*p + c*f*q - c*g*p - d*f*p +
                               d*g*h) / det
            res[2] = res[8] = (b*f*q - b*g*p - c*e*q + c*g*g + d*e*p -
                               d*f*g) / det
            res[3] = res[12] = (-b*f*p + b*g*h + c*e*p - c*f*g - d*e*h +
                                d*f*f) / det
            res[5] = (a*h*q - a*p*p - c*c*q + 2*c*d*p - d*d*h) / det
            res[6] = res[9] = (-a*f*q + a*g*p + b*c*q - b*d*p - c*d*g +
                               d*d*f) / det
            res[7] = res[13] = (a*f*p - a*g*h - b*c*p + b*d*h + c*c*g -
                                c*d*f) / det
            res[10] = (a*e*q - a*g*g - b*b*q + 2*b*d*g - d*d*e) / det
            res[11] = res[14] = (-a*e*p + a*f*g + b*b*p - b*c*g - b*d*f +
                                 c*d*e) / det
            res[15] = (a*e*h - a*f*f - b*b*h + 2*b*c*f - c*c*e) / det

        dbg_vec[0] = 0.0

def cross_product(vec1=[0.0, 0.0], vec2=[0.0, 0.0], res=[0.0, 0.0]):
    res[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1]
    res[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2]
    res[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0]

def eig_vals_3x3_analytical(mat=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            eig_vals=[0.0, 0.0, 0.0]):
    r"""
    Calculates the eigenvalues of a symmetric 3x3 matrix using the enhanced and
     efficient analytical algorithm proposed by Harari and Albocher (2022),
     which provides accurate eigenvalues when two of the eigenvalues are close
     to each other (coalescing).

    Paper link: https://onlinelibrary.wiley.com/doi/full/10.1002/nme.7153

    @param mat: Symmetric 3x3 matrix is flattened form (1D array)
    @param eig_vals: Vector of eigenvalues (1x3)
    @return: None
    """
    m_dev, t, sm_dev, res = declare("matrix(9)", 4)

    # Read matrix terms
    a11 = mat[0]
    a12 = mat[1]
    a13 = mat[2]
    a22 = mat[4]
    a23 = mat[5]
    a33 = mat[8]

    # Compute matrix invariants I1 and J2, and coefficient s
    i1 = (a11 + a22 + a33) / 3.0
    j2 = ((pow(a11 - a22, 2) + pow(a22 - a33, 2) + pow(a33 - a11, 2)) / 6 +
          pow(a12, 2) + pow(a13, 2) + pow(a23, 2))
    s = sqrt(j2 / 3.0)

    # Check if matrix was isotropic
    if s < 1e-9:
        eig_vals[0] = eig_vals[1] = eig_vals[2] = i1

    else:
        # Calculate the deviatoric part of the matrix
        m_dev[0] = a11 - i1
        m_dev[4] = a22 - i1
        m_dev[8] = a33 - i1
        m_dev[1] = m_dev[3] = a12
        m_dev[2] = m_dev[6] = a13
        m_dev[5] = m_dev[7] = a23

        # Calculate the matrix T
        matrix_multiply(m_dev, m_dev, t, 3)

        t[0] -= 2 * j2 / 3
        t[4] -= 2 * j2 / 3
        t[8] -= 2 * j2 / 3

        # Calculate s * deviatoric matrix
        matrix_multiply_scalar(m_dev, s, sm_dev, 3)

        # Calculate the "determinant" d
        matrix_add(t, sm_dev, res, 3, -1.0)
        norm1 = matrix_norm(res)

        matrix_add(t, sm_dev, res, 3, 1.0)
        norm2 = matrix_norm(res)

        d = norm1 / norm2

        # Determine the sign of the determinant
        sj = sign(1 - d)

        # Check for singularity
        if sj * (1 - d) < 1e-9:
            eig_vals[0] = sqrt(j2) + i1
            eig_vals[1] = i1
            eig_vals[2] = -sqrt(j2) + i1

        else:
            # Calculate Lode's angle
            alpha = 2 * atan2(pow(norm1, sj), pow(norm2, sj)) / 3.0

            # Coefficients cd and sd
            cd = sj * s * cos(alpha)
            sd = sqrt(j2) * sin(alpha)

            # Final eigenvalues
            eig_vals[0] = 2 * cd + i1
            eig_vals[1] = -cd + sd + i1
            eig_vals[2] = -cd - sd + i1

        # print("\n")
        # print("##### MATRIX L #####\n")
        # print(mat[0], mat[1], mat[2])
        # print(mat[3], mat[4], mat[5])
        # print(mat[6], mat[7], mat[8])
        # print('\n')
        # print("##### MATRIX L Dev #####\n")
        # print(m_dev[0], m_dev[1], m_dev[2])
        # print(m_dev[3], m_dev[4], m_dev[5])
        # print(m_dev[6], m_dev[7], m_dev[8])
        # print('\n')
        # print("I1: ", i1)
        # print("J2: ", j2)
        # print("s: ", s)
        # print('\n')
        # print("Norm 1: ", norm1)
        # print("Norm 2: ", norm2)
        # print("Determinant, d: ", d)
        # print("sj: ", sj)
        # print("Check sj * (1 - d) < tol: ", sj * (1 - d))
        # print("alpha: ", alpha)
        # print("cd: ", cd)
        # print("sd: ", sd)
        # print('\n')
        # print("Eigv1, Eigv2, Eigv3: ", eig_vals[0], eig_vals[1], eig_vals[2])
        # print("\n")

def rot_mat_3x3_analytical(
        mat=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        eigvals=[0.0,0.0,0.0],
        rot_mat=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
):

    linv = declare("matrix(9)")

    # Inverse matrix in principal directions (diagonal)
    linv[0] = linv[4] = linv[8] = 0.0
    linv[1] = linv[2] = linv[3] = 0.0
    linv[5] = linv[6] = linv[7] = 0.0

    if eigvals[0] != 0.0:
        linv[0] = 1.0 / eigvals[0]
    if eigvals[1] != 0.0:
        linv[4] = 1.0 / eigvals[1]
    if eigvals[2] != 0.0:
        linv[8] = 1.0 / eigvals[2]

    matrix_multiply(mat, linv, rot_mat, 3)

# ========================== Auxiliary Functions ==============================
def max(a=0.0, b=0.0):
    if a > b:
        return a
    return b

def min(a=0.0, b=0.0):
    if a < b:
        return a
    return b

def matrix_multiply_scalar(mat=[1.0, 1.0, 1.0, 1.0], a=0.0,
                           res=[1.0, 1.0, 1.0, 1.0], n=2):
    r"""
    This equation multiplies a matrix (mat) by a scalar value (a) and returns
    the result in a second matrix (res).

    :param mat:
    :param a:
    :param res:
    :param n:
    :return:
    """
    i, r = declare("int", 2)
    r = n * n
    for i in range(r):
        res[i] = a*mat[i]

def sign(a=0.0):
    if a < 0.0:
        return -1.0
    return 1.0

# =============================================================================
# =========================== ERICK ADDED FUNCTIONS ===========================
# =============================================================================

def eig_vals_3x3_analytical_erick(
        mat=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        eig_vals=[0.0, 0.0, 0.0]
):
    """
    Erick: apenas alterei para o eig_vals sair em ordem do maior para o menor.
    Calculates the eigenvalues of a symmetric 3x3 matrix using the enhanced and
     efficient analytical algorithm proposed by Harari and Albocher (2022),
     which provides accurate eigenvalues when two of the eigenvalues are close
     to each other (coalescing).

    Paper link: https://onlinelibrary.wiley.com/doi/full/10.1002/nme.7153

    @param mat: Symmetric 3x3 matrix is flattened form (1D array) (sxx,sxy,sxz,
                 syx,syy,syz,szx,szy,szz)
    @param eig_vals: Vector of eigenvalues (1x3), (s1,s2,s3) s1>s2>s3
    @return: None
    """
    m_dev, t, sm_dev, res = declare("matrix(9)", 4)

    # Read matrix terms
    a11 = mat[0]
    a12 = mat[1]
    a13 = mat[2]
    a22 = mat[4]
    a23 = mat[5]
    a33 = mat[8]

    # Compute matrix invariants I1 and J2, and coefficient s
    i1 = (a11 + a22 + a33) / 3.0
    j2 = (
            (pow(a11 - a22, 2) + pow(a22 - a33, 2) +
             pow(a33 - a11, 2)) / 6 + pow(a12, 2) + pow(a13, 2) +
            pow(a23, 2)
    )
    s = sqrt(j2 / 3.0)

    # Check if matrix was isotropic
    if s < 1e-9:
        eig_vals[0] = eig_vals[1] = eig_vals[2] = i1

    else:
        # Calculate the deviatoric part of the matrix
        m_dev[0] = a11 - i1
        m_dev[4] = a22 - i1
        m_dev[8] = a33 - i1
        m_dev[1] = m_dev[3] = a12
        m_dev[2] = m_dev[6] = a13
        m_dev[5] = m_dev[7] = a23

        # Calculate the matrix T
        matrix_multiply(m_dev, m_dev, t, 3)

        t[0] -= 2 * j2 / 3
        t[4] -= 2 * j2 / 3
        t[8] -= 2 * j2 / 3

        # Calculate s * deviatoric matrix
        matrix_multiply_scalar(m_dev, s, sm_dev, 3)

        # Calculate the "determinant" d
        matrix_add(t, sm_dev, res, 3, -1.0)
        norm1 = matrix_norm(res)

        matrix_add(t, sm_dev, res, 3, 1.0)
        norm2 = matrix_norm(res)

        d = norm1 / norm2

        # Determine the sign of the determinant
        sj = sign(1 - d)

        # Check for singularity
        if sj * (1 - d) < 1e-9:
            eig_vals[0] = sqrt(j2) + i1
            eig_vals[1] = i1
            eig_vals[2] = -sqrt(j2) + i1

        else:
            # Calculate Lode's angle
            alpha = 2 * atan2(pow(norm1, sj), pow(norm2, sj)) / 3.0

            # Coefficients cd and sd
            cd = sj * s * cos(alpha)
            sd = sqrt(j2) * sin(alpha)

            # Final eigenvalues
            eig_vals[0] = 2 * cd + i1
            eig_vals[1] = -cd + sd + i1
            eig_vals[2] = -cd - sd + i1

    # Erick: adicionei isso para oganizar do maior para o menor
    s1 = 0
    s2 = 0
    s3 = 0

    if eig_vals[0] <= eig_vals[1] and eig_vals[0] <= eig_vals[2]:
        s1 = eig_vals[0]
        if eig_vals[1] <= eig_vals[2]:
            s2 = eig_vals[1]
            s3 = eig_vals[2]
        else:
            s2 = eig_vals[2]
            s3 = eig_vals[1]
    elif eig_vals[1] <= eig_vals[2]:
        s1 = eig_vals[1]
        if eig_vals[0] <= eig_vals[2]:
            s2 = eig_vals[0]
            s3 = eig_vals[2]
        else:
            s2 = eig_vals[2]
            s3 = eig_vals[0]
    else:
        s1 = eig_vals[2]
        if eig_vals[0] <= eig_vals[1]:
            s2 = eig_vals[0]
            s3 = eig_vals[1]
        else:
            s2 = eig_vals[1]
            s3 = eig_vals[0]

    eig_vals[0] = s1
    eig_vals[1] = s2
    eig_vals[2] = s3


def matrix_eigenvectors_erick(mat=[1.0, 1.0, 1.0, 1.0], eigvals=[1.0, 1.0],
                              eigvecs=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], n=2):
    """
    Algorithms taken from:
        https://www.math.harvard.edu/archive/21b_fall_04/exhibits/2dmatrices/
    as of 03/07/2019.

    and

        https://en.wikipedia.org/wiki/Eigenvalue_algorithm as of 02/20/2019.

    Parameters
    -----------
    :param mat: Symmetric 3x3 matrix is flattened form (1D array) (sxx,sxy,sxz,
                 syx,syy,syz,szx,szy,szz)
    :param eigvals: list with eigenvalues sorted in any order
    :param eigvecs: list holding the resulting eigenvectors for 'mat' (in the
                     same order as eigvals list)
    :param n: int representing the number or rows (columns) in 'mat'

    Output
    -----------
    :return: None
    """

    tol = 1e-2
    i, j, k, max_index, e, f = declare("int", 6)
    col_sum = declare("matrix(3)")
    temp_a, temp_b, temp_c = declare("matrix(9)", 3)

    if n == 2:
        # Check for diagonal matrices.
        diag_sum = fabs(mat[1]) + fabs(mat[2])
        if diag_sum < tol:
            eigvecs[0] = 1.0
            eigvecs[2] = 1.0
        else:
            if mat[2] != 0:
                eigvecs[0] = eigvals[0] - mat[3]
                eigvecs[1] = mat[2]
                eigvecs[3] = eigvals[1] - mat[3]
                eigvecs[4] = mat[2]
            elif mat[1] != 0:
                eigvecs[0] = mat[1]
                eigvecs[1] = eigvals[0] - mat[0]
                eigvecs[3] = mat[1]
                eigvecs[4] = eigvals[1] - mat[0]
            else:
                eigvecs[0] = 0.0
                eigvecs[1] = 1.0
                eigvecs[3] = 1.0
                eigvecs[4] = 0.0
            eigvecs[2] = 0.0
            eigvecs[5] = 0.0
            vector_normalize(eigvecs, 3 * n)
    else:
        # for i in range(9):
        #    eigvecs[i] = 0.0
        # Check for diagonal matrices.
        diag_sum = 0.0
        for i in range(9):
            if i % 4 != 0.0:
                diag_sum += fabs(mat[i])
        if diag_sum < tol:
            # Erick: alterei esse trecho para garantir que a ordem dos
            # eigenvectors obedeça  a ordem do eigenvalues
            e = 0
            f = 0

            if fabs(eigvals[0] - mat[0]) < tol:
                eigvecs[0] = 1.0
                e = 0
            elif fabs(eigvals[0] - mat[4]) < tol:
                eigvecs[1] = 1.0
                e = 1
            elif fabs(eigvals[0] - mat[8]) < tol:
                eigvecs[2] = 1.0
                e = 2
            if e != 0 and fabs(eigvals[1] - mat[0]) < tol:
                eigvecs[3] = 1.0
                f = 0
            elif e != 1 and fabs(eigvals[1] - mat[4]) < tol:
                eigvecs[4] = 1.0
                f = 1
            elif e != 2 and fabs(eigvals[1] - mat[8]) < tol:
                eigvecs[5] = 1.0
                f = 2
            if e != 0 and f != 0 and fabs(eigvals[2] - mat[0]) < tol:
                eigvecs[6] = 1.0
            elif e != 1 and f != 1 and fabs(eigvals[2] - mat[4]) < tol:
                eigvecs[7] = 1.0
            elif e != 2 and f != 2 and fabs(eigvals[2] - mat[8]) < tol:
                eigvecs[8] = 1.0
            # for i in range(9):
            #    eigvecs[i] = 1.0

        else:
            for i in range(3):
                max_value = -1.0e32

                for j in range(9):
                    temp_a[j] = mat[j]
                    temp_b[j] = mat[j]

                if i == 0:
                    for j in range(0, 9, 4):
                        temp_a[j] -= eigvals[1]
                        temp_b[j] -= eigvals[2]

                    matrix_multiply(temp_a, temp_b, temp_c, 3)

                    # Find column with maximum sum
                    for j in range(3):
                        col_sum[j] = 0.0
                        for k in range(3):
                            col_sum[j] += abs(temp_c[3 * k + j])
                    for j in range(3):
                        if col_sum[j] > max_value:
                            max_index = j
                            max_value = col_sum[j]

                    # Assign values to the vector list
                    for k in range(3):
                        eigvecs[k] = temp_c[3 * k + max_index]
                elif i == 1:
                    for j in range(0, 9, 4):
                        temp_a[j] -= eigvals[0]
                        temp_b[j] -= eigvals[2]
                    matrix_multiply(temp_a, temp_b, temp_c, 3)

                    for j in range(3):
                        col_sum[j] = 0.0
                        for k in range(3):
                            col_sum[j] += abs(temp_c[3 * k + j])
                    for j in range(3):
                        if col_sum[j] > max_value:
                            max_index = j
                            max_value = col_sum[j]

                    for k in range(3):
                        eigvecs[k + 3] = temp_c[3 * k + max_index]
                else:
                    for j in range(0, 9, 4):
                        temp_a[j] -= eigvals[0]
                        temp_b[j] -= eigvals[1]
                    matrix_multiply(temp_a, temp_b, temp_c, 3)

                    for j in range(3):
                        col_sum[j] = 0.0
                        for k in range(3):
                            col_sum[j] += abs(temp_c[3 * k + j])
                    for j in range(3):
                        if col_sum[j] > max_value:
                            max_index = j
                            max_value = col_sum[j]

                    for k in range(3):
                        eigvecs[k + 6] = temp_c[3 * k + max_index]

        vector_normalize(eigvecs, 3 * n)


def voight_to_flat(voight=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   flat=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
    flat[0] = voight[0]  # sxx
    flat[1] = voight[5]  # sxy
    flat[2] = voight[4]  # sxz
    flat[3] = flat[1]  # syx
    flat[4] = voight[1]  # syy
    flat[5] = voight[3]  # syz
    flat[6] = flat[2]  # szx
    flat[7] = flat[5]  # szy
    flat[8] = voight[2]  # szy

def flat_to_voight(flat=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   voight=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
    voight[0] = flat[0]  # sxx
    voight[1] = flat[4]  # syy
    voight[2] = flat[8]  # szz
    voight[3] = flat[5]  # syz
    voight[4] = flat[2]  # sxz
    voight[5] = flat[1]  # sxy

def matrix_zero(mat=[0.0, 0.0], n=2):
    i = declare("int")
    for i in range(n):
        mat[i] = 0.0
