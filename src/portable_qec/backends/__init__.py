"""Decoder backends.

``bp_numpy`` is always importable (numpy/scipy only). ``bp_torch`` requires
torch; ``bp_triton`` / ``relay_triton`` additionally require triton and a GPU
to RUN (they import without one — the kernels compile only where triton
exists). The API layer (``portable_qec.api``) imports the optional backends
lazily, so a missing extra never breaks the core package.
"""
from .bp_numpy import BpBaseline

__all__ = ["BpBaseline"]
