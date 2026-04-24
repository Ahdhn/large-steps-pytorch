from largesteps.solvers import CholeskySolver, ConjugateGradientSolver, solve
import weakref
import os
import csv
import numpy as np
import torch

# Cache for the system solvers
_cache = {}

# State for dumping matrices and rhs vectors to disk for external analysis.
# Populated via enable_matrix_dump(); consumed by _register_matrix (below) and
# by _record_solve (called from largesteps.solvers).
_dump_state = {
    "dir": None,
    "matrices": {},   # id(M) -> entry dict (see _register_matrix)
    "next_idx": 0,
}


def enable_matrix_dump(output_dir):
    """
    Enable dumping of each unique system matrix M (one per remesh stage) along
    with the first forward and first backward rhs seen for it, plus solve
    counts. Files are written as MatrixMarket (.mtx).

    Call once before optimize_shape(); call flush_matrix_dump() afterwards to
    write the counts.csv summary.
    """
    os.makedirs(output_dir, exist_ok=True)
    _dump_state["dir"] = output_dir
    _dump_state["matrices"] = {}
    _dump_state["next_idx"] = 0


def disable_matrix_dump():
    _dump_state["dir"] = None
    _dump_state["matrices"] = {}
    _dump_state["next_idx"] = 0


def _mmwrite_sparse_coo(path, indices, values, shape):
    """Write a sparse COO matrix to a MatrixMarket file."""
    try:
        from scipy.io import mmwrite
        from scipy.sparse import coo_matrix
        rows = indices[0]
        cols = indices[1]
        mat = coo_matrix((values, (rows, cols)), shape=shape)
        mmwrite(path, mat, symmetry="general")
        return
    except ImportError:
        pass

    rows = indices[0]
    cols = indices[1]
    nnz = values.shape[0]
    with open(path, "w") as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"{shape[0]} {shape[1]} {nnz}\n")
        for r, c, v in zip(rows, cols, values):
            f.write(f"{int(r) + 1} {int(c) + 1} {float(v):.17g}\n")


def _mmwrite_dense(path, array):
    """Write a dense 2D array to a MatrixMarket file."""
    arr = np.asarray(array)
    if arr.ndim == 1:
        arr = arr[:, None]
    try:
        from scipy.io import mmwrite
        mmwrite(path, arr, symmetry="general")
        return
    except ImportError:
        pass

    with open(path, "w") as f:
        f.write("%%MatrixMarket matrix array real general\n")
        f.write(f"{arr.shape[0]} {arr.shape[1]}\n")
        # MatrixMarket array format is column-major.
        for j in range(arr.shape[1]):
            for i in range(arr.shape[0]):
                f.write(f"{float(arr[i, j]):.17g}\n")


def _register_matrix(L):
    """
    Assign an index to a newly-seen matrix, dump it to disk, and create its
    bookkeeping entry. Idempotent per matrix id.
    """
    if _dump_state["dir"] is None:
        return None

    key = id(L)
    if key in _dump_state["matrices"]:
        return _dump_state["matrices"][key]

    idx = _dump_state["next_idx"]
    _dump_state["next_idx"] += 1

    L_cpu = L.detach().cpu().coalesce()
    indices = L_cpu.indices().numpy()
    values = L_cpu.values().numpy()
    shape = tuple(L_cpu.shape)

    matrix_file = f"mat_{idx:03d}.mtx"
    _mmwrite_sparse_coo(
        os.path.join(_dump_state["dir"], matrix_file),
        indices, values, shape,
    )

    entry = {
        "idx": idx,
        "n": shape[0],
        "fwd_count": 0,
        "bwd_count": 0,
        "fwd_saved": False,
        "bwd_saved": False,
        "matrix_file": matrix_file,
        "fwd_rhs_file": "",
        "bwd_rhs_file": "",
    }
    _dump_state["matrices"][key] = entry
    return entry


def _record_solve(L, b, backward):
    """
    Record a solve against matrix L with right-hand side b. The first forward
    and first backward rhs for each matrix are persisted to disk; counts are
    always incremented.
    """
    if _dump_state["dir"] is None:
        return

    entry = _dump_state["matrices"].get(id(L))
    if entry is None:
        entry = _register_matrix(L)
        if entry is None:
            return

    if backward:
        entry["bwd_count"] += 1
        if not entry["bwd_saved"]:
            fname = f"mat_{entry['idx']:03d}_rhs_bwd.mtx"
            _mmwrite_dense(
                os.path.join(_dump_state["dir"], fname),
                b.detach().cpu().numpy(),
            )
            entry["bwd_saved"] = True
            entry["bwd_rhs_file"] = fname
    else:
        entry["fwd_count"] += 1
        if not entry["fwd_saved"]:
            fname = f"mat_{entry['idx']:03d}_rhs_fwd.mtx"
            _mmwrite_dense(
                os.path.join(_dump_state["dir"], fname),
                b.detach().cpu().numpy(),
            )
            entry["fwd_saved"] = True
            entry["fwd_rhs_file"] = fname


def flush_matrix_dump():
    """
    Write a counts.csv summary into the dump directory. Safe to call even if
    dumping is disabled (it becomes a no-op).
    """
    if _dump_state["dir"] is None:
        return

    csv_path = os.path.join(_dump_state["dir"], "counts.csv")
    entries = sorted(_dump_state["matrices"].values(), key=lambda e: e["idx"])
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "idx", "n_vertices", "fwd_count", "bwd_count",
            "matrix_file", "fwd_rhs_file", "bwd_rhs_file",
        ])
        for e in entries:
            writer.writerow([
                e["idx"], e["n"], e["fwd_count"], e["bwd_count"],
                e["matrix_file"], e["fwd_rhs_file"], e["bwd_rhs_file"],
            ])


def cache_put(key, value, A):
    # Called when 'A' is garbage collected
    def cleanup_callback(wr):
        del _cache[key]

    wr = weakref.ref(
        A,
        cleanup_callback
    )

    _cache[key] = (value, wr)

def to_differential(L, v):
    """
    Convert vertex coordinates to the differential parameterization.

    Parameters
    ----------
    L : torch.sparse.Tensor
        (I + l*L) matrix
    v : torch.Tensor
        Vertex coordinates
    """
    return L @ v

def from_differential(L, u, method='Cholesky'):
    """
    Convert differential coordinates back to Cartesian.

    If this is the first time we call this function on a given matrix L, the
    solver is cached. It will be destroyed once the matrix is garbage collected.

    Parameters
    ----------
    L : torch.sparse.Tensor
        (I + l*L) matrix
    u : torch.Tensor
        Differential coordinates
    method : {'Cholesky', 'CG'}
        Solver to use.
    """
    key = (id(L), method)
    if key not in _cache.keys():
        if method == 'Cholesky':
            solver = CholeskySolver(L)
        elif method == 'CG':
            solver = ConjugateGradientSolver(L)
        else:
            raise ValueError(f"Unknown solver type '{method}'.")

        cache_put(key, solver, L)
        _register_matrix(L)
    else:
        solver = _cache[key][0]

    return solve(solver, u)
