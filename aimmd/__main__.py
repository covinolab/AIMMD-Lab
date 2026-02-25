"""
AIMMD multiprocessing entry point.

This module exists so that ``python -m aimmd`` is a valid invocation and, more
importantly, so that ``multiprocessing.spawn`` can safely re-execute the main
module without accidentally triggering heavy side effects or user code.

Context
-------
On platforms and configurations that use the "spawn" start method, Python starts
a fresh interpreter and imports the main module. Providing a minimal, stable
``aimmd.__main__`` helps avoid import-time surprises.

Current behavior
----------------
The entry point intentionally does nothing. AIMMD's functional entry points are
expected to live elsewhere (e.g., CLI wrappers or higher-level launchers).
"""

def main():
    """
    Minimal entry point for ``python -m aimmd``.

    Notes
    -----
    This is intentionally a no-op. Its primary purpose is compatibility with
    multiprocessing spawn semantics.
    """
    # nothing to do here – real work happens elsewhere
    pass

if __name__ == "__main__":
    # Standard module execution guard: required for multiprocessing spawn safety.
    main()
