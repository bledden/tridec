"""Shared Tanner-graph CSR/CSC edge layout for a GF2 check matrix.

Every backend (numpy, torch, Triton) consumes this ONE layout, so their
gather/scatter index sets are identical by construction — the CPU reference and
the GPU kernels walk the same edges in the same order.
"""
import numpy as np


def tanner_adjacency(H):
    """CSR (check->bit) + CSC (bit->check) adjacency for a GF2 check matrix H.

    This is the SHARED gather/scatter layout: a single set of (check, bit) edges,
    addressable both check-major (CSR, the gather of a check's incident bit
    messages) and bit-major (CSC, the scatter back to bits). Both index the SAME
    edge set, so an edge-indexed message array ports directly to a GPU
    gather/scatter kernel.

    Returns a dict with:
      n_checks, n_bits,
      check_to_bit_ptr (n_checks+1), check_to_bit_idx (nnz)  -- CSR,
      bit_to_check_ptr (n_bits+1),   bit_to_check_idx (nnz)  -- CSC.
    """
    H = np.asarray(H, dtype=np.uint8) % 2
    n_checks, n_bits = H.shape

    # CSR check->bit (row-major nonzeros, columns ascending within each row).
    rows, cols = np.nonzero(H)
    order = np.lexsort((cols, rows))           # sort by (row, then col)
    rows, cols = rows[order], cols[order]
    chk_ptr = np.zeros(n_checks + 1, dtype=np.int64)
    np.add.at(chk_ptr, rows + 1, 1)
    chk_ptr = np.cumsum(chk_ptr)
    chk_idx = cols.astype(np.int64)

    # CSC bit->check (col-major nonzeros, rows ascending within each col).
    order2 = np.lexsort((rows, cols))          # sort by (col, then row)
    rows2, cols2 = rows[order2], cols[order2]
    bit_ptr = np.zeros(n_bits + 1, dtype=np.int64)
    np.add.at(bit_ptr, cols2 + 1, 1)
    bit_ptr = np.cumsum(bit_ptr)
    bit_idx = rows2.astype(np.int64)

    return dict(
        n_checks=int(n_checks), n_bits=int(n_bits),
        check_to_bit_ptr=chk_ptr, check_to_bit_idx=chk_idx,
        bit_to_check_ptr=bit_ptr, bit_to_check_idx=bit_idx,
    )
