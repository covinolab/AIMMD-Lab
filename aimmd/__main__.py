"""
AIMMD multiprocessing entry point.

This module exists so that Python's multiprocessing "spawn" start method can
safely re-execute the package entry point.

Why this file exists
--------------------
On some platforms and configurations, multiprocessing uses the "spawn" method
to start child processes. In that case, Python imports the main module in the
new interpreter. Having a lightweight, import-safe entry point avoids:

- executing heavyweight AIMMD imports at spawn time,
- side effects during interpreter bootstrap,
- accidental re-initialization of global state.

Design
------
- `main()` intentionally does nothing.
- Real work is expected to be triggered from other modules (e.g., Worker,
  Launcher, scripts) that are explicitly invoked by the user.

Notes
-----
If you ever need a CLI in the future, this file is a safe place to parse
arguments and dispatch, provided it remains spawn-friendly (minimal imports
at module scope).
"""

def main():
    """
    Minimal entry point for ``python -m aimmd``.
    Intentionally empty: keep multiprocessing spawn re-import cheap and safe.
    Real execution is orchestrated by AIMMD components invoked elsewhere.
    """
    pass

if __name__ == "__main__":
    # Standard guard: prevents accidental execution on import.
    main()
