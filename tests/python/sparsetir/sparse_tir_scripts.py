from tvm.script import tir as T


@T.prim_func
def csrmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    k: T.int32,
    nnz: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    I = T.dense_fixed(m)
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
    K = T.dense_fixed(k)
    A = T.match_sparse_buffer(a, (I, J), "float32")
    B = T.match_sparse_buffer(b, (T.dense(J), K), "float32")
    C = T.match_sparse_buffer(c, (I, K), "float32")
    with T.iter([I, K, J], "SSR", "csrmm") as [vi, vk, vj]:
        with T.init():
            C[vi, vk] = 0.0
        C[vi, vk] = C[vi, vk] + A[vi, vj] * B[vj, vk]


@T.prim_func
def csrmm_dense_iter(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    k: T.int32,
    nnz: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    I = T.dense_fixed(m)
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
    K = T.dense_fixed(k)
    A = T.match_sparse_buffer(a, (I, J), "float32")
    B = T.match_sparse_buffer(b, (T.dense(J), K), "float32")
    C = T.match_sparse_buffer(c, (I, K), "float32")
    with T.iter([I, T.dense(J), K], "SRS", "csrmm") as [vi, vj, vk]:
        with T.init():
            C[vi, vk] = 0.0
        C[vi, vk] = C[vi, vk] + A[vi, vj] * B[vj, vk]


@T.prim_func
def segment_reduce(
    a: T.handle,
    b: T.handle,
    indptr: T.handle,
    n: T.int32,
    nnz: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    I = T.dense_fixed(n)
    J = T.dense_variable(I, (100, nnz), indptr, "int32")
    A = T.match_sparse_buffer(a, (I, J), "float32")
    B = T.match_sparse_buffer(b, (I,), "float32")
    with T.iter([I, J], "SR", "segment_reduce") as [vi, vj]:
        with T.init():
            B[vi] = 0.
        B[vi] = B[vi] + A[vi, vj]


@T.prim_func
def csr_reduce(
    a: T.handle,
    b: T.handle,
    indptr: T.handle,
    indices: T.handle,
    n: T.int32,
    m: T.int32,
    nnz: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    I = T.dense_fixed(n)
    J = T.sparse_variable(I, (m, nnz), (indptr, indices), "int32")
    A = T.match_sparse_buffer(a, (I, J), "float32")
    B = T.match_sparse_buffer(b, (I,), "float32")
    with T.iter([I, J], "SR", "csr_reduce") as [vi, vj]:
        with T.init():
            B[vi] = 0.0
        B[vi] = B[vi] + A[vi, vj]


@T.prim_func
def bsrmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    nb: T.int32,
    mb: T.int32,
    nnzb: T.int32,
    blk: T.int32,
    feat_size: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    I = T.dense_fixed(nb)
    J = T.sparse_variable(I, (mb, nnzb), (indptr, indices), "int32")
    BI = T.dense_fixed(blk)
    BJ = T.dense_fixed(blk)
    F = T.dense_fixed(feat_size)
    A = T.match_sparse_buffer(a, (I, J, BI, BJ), "float32")
    B = T.match_sparse_buffer(b, (T.dense(J), BJ, F), "float32")
    C = T.match_sparse_buffer(c, (I, BI, F), "float32")

    with T.iter([I, BI, BJ, F, J], "SSRSR", "bsrmm") as [
        vi,
        vbi,
        vbj,
        vf,
        vj,
    ]:
        with T.init():
            C[vi, vbi, vf] = 0.0
        C[vi, vbi, vf] = C[vi, vbi, vf] + A[vi, vj, vbi, vbj] * B[vj, vbj, vf]


@T.prim_func
def ellmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indices: T.handle,
    nb: T.int32,
    mb: T.int32,
    feat_size: T.int32,
    col: T.int32,
    blk: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    I = T.dense_fixed(nb)
    J = T.sparse_fixed(I, (mb, col), indices, "int32")
    F = T.dense_fixed(feat_size)
    BI = T.dense_fixed(blk)
    BJ = T.dense_fixed(blk)
    A = T.match_sparse_buffer(a, (I, J, BI, BJ), "float32")
    B = T.match_sparse_buffer(b, (T.dense(J), BJ, F), "float32")
    C = T.match_sparse_buffer(c, (I, BI, F), "float32")

    with T.iter([I, J, BI, BJ, F], "SRSRS", "ellmm") as [
        vi,
        vj,
        vbi,
        vbj,
        vf,
    ]:
        with T.init():
            C[vi, vbi, vf] = 0.0
        C[vi, vbi, vf] = C[vi, vbi, vf] + A[vi, vj, vbi, vbj] * B[vj, vbj, vf]


@T.prim_func
def csr_element_wise(
    a: T.handle,
    b: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    nnz: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    I = T.dense_fixed(m)
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
    A = T.match_sparse_buffer(a, (I, J), "float32")
    B = T.match_sparse_buffer(b, (I, J), "float32")

    with T.iter([I, J], "SS", "csr_element_wise") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.5


@T.prim_func
def bmm(
    x: T.handle,
    y: T.handle,
    z: T.handle,
    indptr_i: T.handle,
    indptr_j: T.handle,
    indptr_k: T.handle,
    indptr_ij: T.handle,
    indptr_jk: T.handle,
    indptr_ik: T.handle,
    batch_size: T.int32,
    nnz_i: T.int32,
    nnz_j: T.int32,
    nnz_k: T.int32,
    nnz_ij: T.int32,
    nnz_jk: T.int32,
    nnz_ik: T.int32
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    B = T.dense_fixed(batch_size)
    I = T.dense_variable(B, (32768, nnz_i), indptr_i, "int32")
    J = T.dense_variable(B, (32768, nnz_j), indptr_j, "int32")
    K = T.dense_variable(B, (32768, nnz_k), indptr_k, "int32")
    IJ = T.attach_axis(I, J, nnz_ij, indptr_ij, "int32")
    JK = T.attach_axis(J, K, nnz_jk, indptr_jk, "int32")
    IK = T.attach_axis(I, K, nnz_ik, indptr_ik, "int32")
    X = T.match_sparse_buffer(x, (B, I, IJ), "float32")
    Y = T.match_sparse_buffer(y, (B, J, JK), "float32")
    Z = T.match_sparse_buffer(z, (B, I, IK), "float32")
    with T.iter([B, I, J, K], "SSRS", "bmm") as [vb, vi, vj, vk]:
        with T.init():
            Z[vb, vi, vk] = 0.
        Z[vb, vi, vk] = Z[vb, vi, vk] + X[vb, vi, vj] * Y[vb, vj, vk]


@T.prim_func
def sddmm(a: T.handle, b: T.handle, c: T.handle, indptr: T.handle, indices: T.handle, m: T.int32, n: T.int32, k: T.int32, nnz: T.int32) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    I = T.dense_fixed(m)
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
    K = T.dense_fixed(k)
    A = T.match_sparse_buffer(a, (I, K), "float32")
    B = T.match_sparse_buffer(b, (T.dense(J), K), "float32")
    C = T.match_sparse_buffer(c, (I, J), "float32")

    with T.iter([I, J, K], "SSR", "sddmm") as [vi, vj, vk]:
        with T.init():
            C[vi, vj] = 0.
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def fused_sddmm(a: T.handle, b: T.handle, c: T.handle, indptr: T.handle, indices: T.handle, m: T.int32, n: T.int32, k: T.int32, nnz: T.int32) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    I = T.dense_fixed(m)
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
    K = T.dense_fixed(k)
    A = T.match_sparse_buffer(a, (I, K), "float32")
    B = T.match_sparse_buffer(b, (T.dense(J), K), "float32")
    C = T.match_sparse_buffer(c, (I, J), "float32")

    with T.iter([T.fuse(I, J), K], "SSR", "sddmm") as [vi, vj, vk]:
        with T.init():
            C[vi, vj] = 0.
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def square_sum(a: T.handle, b: T.handle, indptr_j: T.handle, indices_j: T.handle, indptr_k: T.handle, indices_k: T.handle, nnz_j: T.int32, nnz_k: T.int32, M: T.int32, N1: T.int32, N2: T.int32):
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    I = T.dense_fixed(M)
    J = T.sparse_variable(I, (N1, nnz_j), (indptr_j, indices_j), "int32")
    K = T.sparse_variable(J, (N2, nnz_k), (indptr_k, indices_k), "int32")
    A = T.match_sparse_buffer(a, (I, J, K), "float32")
    B = T.match_sparse_buffer(b, (I,), "float32")

    with T.iter([I, J, K], "SRR", "square_sum") as [vi, vj, vk]:
        with T.init():
            B[vi] = 0.0
        B[vi] = B[vi] + A[vi, vj, vk]


@T.prim_func
def square_sum_two_K(a: T.handle, b: T.handle, indptr_j: T.handle, indices_j: T.handle, indptr_k0: T.handle, indices_k0: T.handle, indptr_k1: T.handle, indices_k1: T.handle, nnz_j: T.int32, nnz_k: T.int32, M: T.int32, N1: T.int32, N2: T.int32):
    # Used only for testing `GetIndicesRange()`.
    # Currently it is ensured that `indptr_k0` is the same as `indptr_k1`, and `indices_k0` is the
    # same as `indices_k1`.
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    I = T.dense_fixed(M)
    J = T.sparse_variable(I, (N1, nnz_j), (indptr_j, indices_j), "int32")
    K0 = T.sparse_variable(J, (N2, nnz_k), (indptr_k0, indices_k0), "int32")
    K1 = T.sparse_variable(J, (N2, nnz_k), (indptr_k1, indices_k1), "int32")
    A = T.match_sparse_buffer(a, (I, J, K0), "float32")
    B = T.match_sparse_buffer(b, (I,), "float32")

    with T.iter([I, J, K1], "SRR", "square_sum") as [vi, vj, vk]:
        with T.init():
            B[vi] = 0.0
        B[vi] = B[vi] + A[vi, vj, vk]


@T.prim_func
def fused_reduction_4d_2d(
    x: T.handle,
    y: T.handle,
    indptr_j: T.handle,
    indptr_k: T.handle,
    indptr_l: T.handle,
    n: T.int32,
    nnz_j: T.int32,
    nnz_k: T.int32,
    nnz_l: T.int32) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    I = T.dense_fixed(n)
    J = T.dense_variable(I, (32768, nnz_j), indptr_j, "int32")
    K = T.dense_variable(J, (32768, nnz_k), indptr_k, "int32")
    L = T.dense_variable(K, (32768, nnz_l), indptr_l, "int32")
    X = T.match_sparse_buffer(x, (I, J, K, L), "float32")
    Y = T.match_sparse_buffer(y, (I, J), "float32")
    with T.iter([T.fuse(I, J), K, L], "SSRR", "reduction_4d_2d") as [vi, vj, vk, vl]:
        with T.init():
            Y[vi, vj] = 0.0
        Y[vi, vj] = Y[vi, vj] + X[vi, vj, vk, vl]
    

@T.prim_func
def fused_reduction_4d_3d(
    x: T.handle,
    y: T.handle,
    indptr_j: T.handle,
    indptr_k: T.handle,
    indptr_l: T.handle,
    n: T.int32,
    nnz_j: T.int32,
    nnz_k: T.int32,
    nnz_l: T.int32) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    I = T.dense_fixed(n)
    J = T.dense_variable(I, (32768, nnz_j), indptr_j, "int32")
    K = T.dense_variable(J, (32768, nnz_k), indptr_k, "int32")
    L = T.dense_variable(K, (32768, nnz_l), indptr_l, "int32")
    X = T.match_sparse_buffer(x, (I, J, K, L), "float32")
    Y = T.match_sparse_buffer(y, (I, J, K), "float32")
    with T.iter([T.fuse(I, J, K), L], "SSSR", "reduction_4d_3d") as [vi, vj, vk, vl]:
        with T.init():
            Y[vi, vj, vk] = 0.0
        Y[vi, vj, vk] = Y[vi, vj, vk] + X[vi, vj, vk, vl]
 

@T.prim_func
def rgcn_forward(
    etype: T.handle,
    w: T.handle,
    x: T.handle,
    y: T.handle,
    indptr: T.handle,
    indices: T.handle,
    n: T.int32,
    r: T.int32,
    feat_size: T.int32,
    nnz: T.int32
):
    I = T.dense_fixed(n)
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
    R = T.dense_fixed(r)
    F_in = T.dense_fixed(feat_size)
    F_out = T.dense_fixed(feat_size)
    E = T.match_sparse_buffer(etype, (I, J), "int32")
    W = T.match_sparse_buffer(w, (R, F_out, F_in), "float32")
    X = T.match_sparse_buffer(x, (T.dense(J), F_in), "float32")
    Y = T.match_sparse_buffer(y, (I, F_out), "float32")
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    with T.iter([I, F_out, J, F_in], "SSRR", "rgcn-forward") as [
        vi, vout, vj, vin,
    ]:
        with T.init():
            Y[vi, vout] = 0.
        Y[vi, vout] = Y[vi, vout] + W[E[vi, vj], vout, vin] * X[vj, vin]
