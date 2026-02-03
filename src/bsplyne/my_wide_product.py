import numpy as np
import scipy.sparse as sps
import numba as nb


@nb.njit(cache=True)
def _wide_product_max_nnz(
    a_indptr: np.ndarray, b_indptr: np.ndarray, height: int
) -> int:
    """
    Compute the maximum number of nonzeros in the result.

    Parameters
    ----------
    a_indptr : np.ndarray
        CSR pointer array for matrix A.
    b_indptr : np.ndarray
        CSR pointer array for matrix B.
    height : int
        Number of rows (must be the same for both A and B).

    Returns
    -------
    max_nnz : int
        Total number of nonzero elements in the resulting matrix.
    """
    max_nnz = 0
    for i in range(height):
        nnz_a = a_indptr[i + 1] - a_indptr[i]
        nnz_b = b_indptr[i + 1] - b_indptr[i]
        max_nnz += nnz_a * nnz_b
    return max_nnz


@nb.njit(cache=True)
def _wide_product_row(
    a_data: np.ndarray,
    a_indices: np.ndarray,
    b_data: np.ndarray,
    b_indices: np.ndarray,
    b_width: int,
    out_data: np.ndarray,
    out_indices: np.ndarray,
) -> int:
    """
    Compute the wide product for one row.

    For each nonzero in the row of A and each nonzero in the row of B, it computes:
        out_index = a_indices[i] * b_width + b_indices[j]
        out_value = a_data[i] * b_data[j]

    Parameters
    ----------
    a_data : np.ndarray
        Nonzero values for the row in A.
    a_indices : np.ndarray
        Column indices for the row in A.
    b_data : np.ndarray
        Nonzero values for the row in B.
    b_indices : np.ndarray
        Column indices for the row in B.
    b_width : int
        Number of columns in B.
    out_data : np.ndarray
        Preallocated output array for the row's data.
    out_indices : np.ndarray
        Preallocated output array for the row's indices.

    Returns
    -------
    off : int
        Number of nonzero entries computed for this row.
    """
    off = 0
    for i in range(a_data.shape[0]):
        for j in range(b_data.shape[0]):
            out_indices[off] = a_indices[i] * b_width + b_indices[j]
            out_data[off] = a_data[i] * b_data[j]
            off += 1
    return off


@nb.njit(cache=True)
def _wide_product_numba(
    height: int,
    a_data: np.ndarray,
    a_indices: np.ndarray,
    a_indptr: np.ndarray,
    a_width: int,
    b_data: np.ndarray,
    b_indices: np.ndarray,
    b_indptr: np.ndarray,
    b_width: int,
):
    """
    Compute the row-wise wide (Khatri-Rao) product for two CSR matrices.

    For each row i, the result[i, :] = kron(A[i, :], B[i, :]), i.e. the Kronecker product
    of the i-th rows of A and B.

    Parameters
    ----------
    height : int
        Number of rows in A and B.
    a_data : np.ndarray
        Data array for matrix A (CSR format).
    a_indices : np.ndarray
        Indices array for matrix A.
    a_indptr : np.ndarray
        Index pointer array for matrix A.
    a_width : int
        Number of columns in A.
    b_data : np.ndarray
        Data array for matrix B (CSR format).
    b_indices : np.ndarray
        Indices array for matrix B.
    b_indptr : np.ndarray
        Index pointer array for matrix B.
    b_width : int
        Number of columns in B.

    Returns
    -------
    out_data : np.ndarray
        Data array for the resulting CSR matrix.
    out_indices : np.ndarray
        Indices array for the resulting CSR matrix.
    out_indptr : np.ndarray
        Index pointer array for the resulting CSR matrix.
    total_nnz : int
        Total number of nonzero entries computed.
    """
    max_nnz = _wide_product_max_nnz(a_indptr, b_indptr, height)
    out_data = np.empty(max_nnz, dtype=a_data.dtype)
    out_indices = np.empty(max_nnz, dtype=a_indices.dtype)
    out_indptr = np.empty(height + 1, dtype=a_indptr.dtype)

    off = 0
    for i in range(height):
        out_indptr[i] = off
        a_start = a_indptr[i]
        a_end = a_indptr[i + 1]
        b_start = b_indptr[i]
        b_end = b_indptr[i + 1]

        row_nnz = _wide_product_row(
            a_data[a_start:a_end],
            a_indices[a_start:a_end],
            b_data[b_start:b_end],
            b_indices[b_start:b_end],
            b_width,
            out_data[off:],
            out_indices[off:],
        )
        off += row_nnz
    out_indptr[height] = off
    return out_data[:off], out_indices[:off], out_indptr, off


def my_wide_product(A: sps.spmatrix, B: sps.spmatrix) -> sps.csr_matrix:
    """
    Compute a "1D" Kronecker product row by row.

    For each row i, the result C[i, :] = kron(A[i, :], B[i, :]).
    Matrices A and B must have the same number of rows.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Input sparse matrix A in CSR format.
    B : scipy.sparse.spmatrix
        Input sparse matrix B in CSR format.

    Returns
    -------
    C : scipy.sparse.csr_matrix
        Resulting sparse matrix in CSR format with shape (A.shape[0], A.shape[1]*B.shape[1]).
    """
    if A.shape[0] != B.shape[0]:
        raise ValueError("A and B must have the same number of rows")

    # Ensure matrices are in CSR format for fast row slicing.
    if not sps.isspmatrix_csr(A):
        A = A.tocsr()
    if not sps.isspmatrix_csr(B):
        B = B.tocsr()

    height = A.shape[0]
    a_width = A.shape[1]
    b_width = B.shape[1]

    out_data, out_indices, out_indptr, total_nnz = _wide_product_numba(
        height,
        A.data,
        A.indices,
        A.indptr,
        a_width,
        B.data,
        B.indices,
        B.indptr,
        b_width,
    )

    # Build the resulting CSR matrix.
    C = sps.csr_matrix(
        (out_data, out_indices, out_indptr), shape=(height, a_width * b_width)
    )

    return C
