"""Stand-ins that let autodoc import ``aimmd.cgaimmd`` without its heavy dependencies.

``conf.py`` replaces torch (and the other heavy runtime dependencies) by small
stubs when they cannot be imported, as on Read the Docs, where only
``docs/requirements.txt`` is installed. The C-GAIMMD modules use more of
torch at import time than the rest of AIMMD (``torch.nn.functional``,
``torch.nn.init``, ``torch.Generator``, ``torch.device``, dtypes, and
``torch.no_grad`` as a decorator), so ``conf.py`` calls :func:`install` right
after its own stubs. It extends the torch stub and does nothing when the
real torch is importable.

bioemu2 and RDKit (the ``bioemu2`` extra) are not stubbed here: ``conf.py``
lists them in ``autodoc_mock_imports``, so autodoc replaces them by Sphinx's
mock modules while it imports :mod:`aimmd.cgaimmd.bioemu2`, whether or not
they are installed.
"""

import contextlib
import sys
import types

MOCKED_IMPORTS = ["bioemu2", "rdkit"]
"""The modules of the ``bioemu2`` extra that autodoc mocks (``autodoc_mock_imports``)."""


def _module(name):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    return module


class _GradMode(contextlib.ContextDecorator):
    """``torch.no_grad`` / ``torch.inference_mode`` stand-in: a context manager
    that is also usable as ``@torch.no_grad()`` and ``@torch.no_grad``."""

    def __new__(cls, func=None, *args, **kwargs):
        if callable(func):                      # bare decorator: @torch.no_grad
            return func
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Generator:
    """``torch.Generator`` stand-in."""

    def __init__(self, device=None):
        self.device = device

    def manual_seed(self, seed):
        return self

    def initial_seed(self):
        return 0


class _Device:
    """``torch.device`` stand-in."""

    def __init__(self, type="cpu", index=None):
        if isinstance(type, str) and ":" in type:
            type, index = type.split(":", 1)
            index = int(index)
        self.type = type
        self.index = index

    def __repr__(self):
        suffix = f", index={self.index})" if self.index is not None else ")"
        return f"device(type='{self.type}'" + suffix

    def __str__(self):
        return self.type if self.index is None else f"{self.type}:{self.index}"

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))


def install():
    """Extend the torch stub of ``conf.py`` for the C-GAIMMD modules (no-op with real torch)."""
    torch = sys.modules.get("torch")
    if torch is None or getattr(torch, "__file__", None) is not None:
        return                                  # the real torch: nothing to stub

    nn = _module("torch.nn")
    torch.nn = nn
    functional = _module("torch.nn.functional")
    nn.functional = functional
    base = getattr(nn, "Module", object)
    for name in ("SiLU", "GELU", "Sequential", "ModuleList", "ModuleDict", "Embedding",
                 "Identity", "Linear", "ReLU", "LayerNorm", "ParameterList", "Dropout"):
        if not hasattr(nn, name):
            setattr(nn, name, type(name, (base,), {}))
    init = _module("torch.nn.init")
    nn.init = init
    for name in ("zeros_", "normal_", "xavier_uniform_", "uniform_", "ones_"):
        setattr(init, name, lambda *args, **kwargs: None)
    torch.no_grad = _GradMode
    torch.inference_mode = _GradMode
    torch.enable_grad = _GradMode
    torch.Generator = _Generator
    torch.device = _Device
    for name in ("float16", "float32", "float64", "int64", "int32", "bool", "uint8"):
        if not hasattr(torch, name):
            setattr(torch, name, name)
    torch.is_tensor = lambda obj: isinstance(obj, torch.Tensor)
