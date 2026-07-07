"""Sphinx configuration for the AIMMD documentation."""

from __future__ import annotations

import contextlib
import importlib
import os
import shutil
import sys
import types
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent / "_ext"))  # local Sphinx extensions


def _ensure_module(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    return module


def _install_doc_stubs() -> None:
    """Install lightweight dependency stubs so autodoc can import AIMMD."""

    if not (shutil.which("gmx") or shutil.which("gmx_mpi")):
        shutil.which = lambda cmd, _orig=shutil.which: (
            "/usr/bin/false" if cmd in {"gmx", "gmx_mpi"} else _orig(cmd)
        )

    def _can_import(name: str) -> bool:
        try:
            importlib.import_module(name)
            return True
        except Exception:
            return False

    if not _can_import("tqdm"):
        tqdm_mod = _ensure_module("tqdm")

        class _DummyTqdm:
            def __init__(self, iterable=None, total=None, **kwargs):
                self.iterable = iterable
                self.total = total
                self.n = 0

            def __iter__(self):
                if self.iterable is None:
                    return iter(())
                return iter(self.iterable)

            def update(self, n=1):
                self.n += n

            def close(self):
                return None

            def set_description(self, *args, **kwargs):
                return None

            def refresh(self):
                return None

        tqdm_mod.tqdm = _DummyTqdm
        tqdm_auto = _ensure_module("tqdm.auto")
        tqdm_auto.tqdm = _DummyTqdm
        tqdm_notebook = _ensure_module("tqdm.notebook")
        tqdm_notebook.tqdm = _DummyTqdm

    if not _can_import("filelock"):
        filelock = _ensure_module("filelock")

        class FileLock:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        filelock.FileLock = FileLock

    if not _can_import("psutil"):
        psutil = _ensure_module("psutil")

        class _Memory:
            available = 2**30

        class _Process:
            def cpu_affinity(self, cpus=None):
                return list(range(4)) if cpus is None else None

        psutil.virtual_memory = lambda: _Memory()
        psutil.cpu_count = lambda logical=False: 4
        psutil.Process = _Process

    if not _can_import("matplotlib"):
        matplotlib = _ensure_module("matplotlib")
        pyplot = _ensure_module("matplotlib.pyplot")
        pyplot.figure = lambda *args, **kwargs: None
        pyplot.subplots = lambda *args, **kwargs: (None, None)
        pyplot.plot = lambda *args, **kwargs: None
        pyplot.savefig = lambda *args, **kwargs: None
        pyplot.close = lambda *args, **kwargs: None
        patches = _ensure_module("matplotlib.patches")

        class Patch:
            def __init__(self, *args, **kwargs):
                pass

        patches.Patch = Patch
        matplotlib.pyplot = pyplot
        matplotlib.patches = patches

    if not _can_import("torch"):
        torch = _ensure_module("torch")

        class Tensor:
            pass

        class Module:
            def __init__(self, *args, **kwargs):
                pass

            def parameters(self):
                return iter(())

            def state_dict(self, *args, **kwargs):
                return {}

            def load_state_dict(self, *args, **kwargs):
                return None

            def reset_parameters(self):
                return None

            def train(self, *args, **kwargs):
                return self

            def eval(self):
                return self

            def to(self, *args, **kwargs):
                return self

            def __call__(self, *args, **kwargs):
                return None

        class Parameter:
            def __init__(self, data=None, *args, **kwargs):
                self.data = data

        class Linear(Module):
            pass

        class ReLU(Module):
            pass

        class Adam:
            def __init__(self, *args, **kwargs):
                pass

            def zero_grad(self):
                return None

            def step(self):
                return None

        nn = _ensure_module("torch.nn")
        nn.Module = Module
        nn.Parameter = Parameter
        nn.Linear = Linear
        nn.ReLU = ReLU

        optim = _ensure_module("torch.optim")
        optim.Adam = Adam

        cuda = types.SimpleNamespace(device_count=lambda: 0)

        torch.Tensor = Tensor
        torch.nn = nn
        torch.optim = optim
        torch.cuda = cuda
        torch.float = "float"
        torch.long = "long"
        torch.set_num_interop_threads = lambda *args, **kwargs: None
        torch.set_num_threads = lambda *args, **kwargs: None
        torch.no_grad = contextlib.nullcontext
        torch.tensor = lambda *args, **kwargs: None
        torch.zeros = lambda *args, **kwargs: None
        torch.ones = lambda *args, **kwargs: None
        torch.empty = lambda *args, **kwargs: None
        torch.from_numpy = lambda *args, **kwargs: None
        torch.cat = lambda *args, **kwargs: None
        torch.stack = lambda *args, **kwargs: None
        torch.load = lambda *args, **kwargs: {}
        torch.save = lambda *args, **kwargs: None
        _ensure_module("torch._dynamo")

    if not _can_import("MDAnalysis"):
        mda = _ensure_module("MDAnalysis")

        class _ReaderBase:
            def __del__(self):
                return None

            def close(self):
                return None

            def __getitem__(self, item):
                return self

            def __len__(self):
                return 0

        class Timestep:
            positions = None
            _velocities = None

        class Universe:
            def __init__(self, *args, **kwargs):
                self.trajectory = types.SimpleNamespace(
                    ts=Timestep(), n_atoms=0, __iter__=lambda self: iter(())
                )

            @classmethod
            def empty(cls, *args, **kwargs):
                return cls()

        class Writer:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def write(self, *args, **kwargs):
                return None

            def close(self):
                return None

        def _reader(*args, **kwargs):
            return _ReaderBase()

        coordinates = _ensure_module("MDAnalysis.coordinates")
        coordinates_base = _ensure_module("MDAnalysis.coordinates.base")
        coordinates_base.ReaderBase = _ReaderBase
        coordinates_base.Timestep = Timestep
        coordinates_core = _ensure_module("MDAnalysis.coordinates.core")
        coordinates_core.reader = _reader
        coordinates_memory = _ensure_module("MDAnalysis.coordinates.memory")

        class MemoryReader(_ReaderBase):
            pass

        coordinates_memory.MemoryReader = MemoryReader
        coordinates_timestep = _ensure_module("MDAnalysis.coordinates.timestep")
        coordinates_timestep.Timestep = Timestep
        coordinates.base = coordinates_base
        coordinates.core = coordinates_core
        coordinates.memory = coordinates_memory
        coordinates.timestep = coordinates_timestep

        lib = _ensure_module("MDAnalysis.lib")
        distances = _ensure_module("MDAnalysis.lib.distances")
        distances.calc_dihedrals = lambda *args, **kwargs: 0.0
        distances.calc_bonds = lambda *args, **kwargs: 0.0
        lib.distances = distances

        transformations = _ensure_module("MDAnalysis.transformations")
        mda.Universe = Universe
        mda.Writer = Writer
        mda.coordinates = coordinates
        mda.lib = lib
        mda.transformations = transformations

    if not _can_import("mdtraj"):
        mdtraj = _ensure_module("mdtraj")

        class Trajectory:
            pass

        mdtraj.Trajectory = Trajectory

    if not _can_import("torch_geometric"):
        torch_geometric = _ensure_module("torch_geometric")
        tg_data = _ensure_module("torch_geometric.data")

        class Data:
            pass

        class Batch:
            @staticmethod
            def from_data_list(*args, **kwargs):
                return None

        tg_data.Data = Data
        tg_data.Batch = Batch
        tg_nn = _ensure_module("torch_geometric.nn")
        tg_nn.radius_graph = lambda *args, **kwargs: None
        torch_geometric.data = tg_data
        torch_geometric.nn = tg_nn

    if not _can_import("mlcolvar"):
        mlcolvar = _ensure_module("mlcolvar")
        mlcolvar_data = _ensure_module("mlcolvar.data")
        mlcolvar_dataset = _ensure_module("mlcolvar.data.dataset")

        class DictDataset(dict):
            pass

        mlcolvar_dataset.DictDataset = DictDataset
        mlcolvar_graph = _ensure_module("mlcolvar.data.graph")
        mlcolvar_graph_atomic = _ensure_module("mlcolvar.data.graph.atomic")

        class Configurations:
            pass

        mlcolvar_graph_atomic.Configurations = Configurations
        mlcolvar_graph_utils = _ensure_module("mlcolvar.data.graph.utils")
        mlcolvar_graph_utils.create_dataset_from_configurations = (
            lambda *args, **kwargs: None
        )
        mlcolvar_utils = _ensure_module("mlcolvar.utils")
        mlcolvar_utils_io = _ensure_module("mlcolvar.utils.io")
        mlcolvar_utils_io._configures_from_trajectory = lambda *args, **kwargs: None
        mlcolvar_utils_io._topology_from_selection = lambda *args, **kwargs: None
        mlcolvar_data.dataset = mlcolvar_dataset
        mlcolvar_data.graph = mlcolvar_graph
        mlcolvar_graph.atomic = mlcolvar_graph_atomic
        mlcolvar_graph.utils = mlcolvar_graph_utils
        mlcolvar.utils = mlcolvar_utils
        mlcolvar_utils.io = mlcolvar_utils_io
        mlcolvar.data = mlcolvar_data


_install_doc_stubs()


# -- Project information -----------------------------------------------------

project = "AIMMD"
copyright = "2025, Gianmarco Lazzeri, Simon Lichtinger"
author = "Gianmarco Lazzeri, Simon Lichtinger"
release = "0.1.0"
version = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_nb",           # registers myst_parser as well — do NOT also list myst_parser
    "sphinx_copybutton",
    "sphinx_design",
    "aimmd_params",      # local extension in _ext/ (generated Params field reference)
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# -- autodoc / autosummary / napoleon ----------------------------------------

autosummary_generate = True
autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_inherit_docstrings = True
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": False,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# -- MyST / notebooks (myst-nb) ----------------------------------------------

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
# The example notebooks are committed with their saved outputs. Read the Docs
# has neither GROMACS nor a GPU, so never execute at build time — the docs
# render the stored outputs instead.
nb_execution_mode = "off"

# -- intersphinx / copybutton ------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- HTML output (Furo) ------------------------------------------------------

html_theme = "furo"
html_title = "AIMMD"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
# Enable these once the assets exist in _static/ (a missing file fails a -W build):
# html_logo = "_static/logo.svg"
# html_favicon = "_static/favicon.ico"
html_theme_options = {
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#0f8b8d",
        "color-brand-content": "#0f8b8d",
    },
    "dark_css_variables": {
        "color-brand-primary": "#2bb3b5",
        "color-brand-content": "#2bb3b5",
    },
    "source_repository": "https://github.com/covinolab/AIMMD",
    "source_branch": "main",
    "source_directory": "docs/source/",
}


# -- Docstring normalization -------------------------------------------------
# Several docstrings begin with an RST title + underline and mix prose with
# loose reStructuredText. This handler runs *after* napoleon (conf.py's setup()
# is connected last), so it only cleans residual title underlines / stray
# indentation left after napoleon has converted the NumPy sections — it does not
# touch napoleon's generated field lists.

_UNDERLINE_RE = re.compile(r"^[=\-~`:#*^]{3,}\s*$")


def _sanitize_docstring(app, what, name, obj, options, lines):
    """Normalize docstrings that mix prose and strict reStructuredText badly."""

    if len(lines) >= 2 and _UNDERLINE_RE.match(lines[1].strip()):
        del lines[:2]

    cleaned = []
    for line in lines:
        stripped = line.strip()
        if _UNDERLINE_RE.match(stripped):
            continue
        if stripped:
            line = line.replace("\t", "    ")
            line = re.sub(r"^\s{4,}", "", line)
        cleaned.append(line.rstrip())

    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()

    # Ensure bullet/enumerated lists are separated from surrounding prose by
    # blank lines, which docutils requires (the raw docstrings frequently omit
    # them, producing "list ends without a blank line" warnings under autodoc).
    def _is_item(s):
        return bool(re.match(r"\s*[-*+]\s", s)) or bool(re.match(r"\s*\d+[.)]\s", s))

    padded = []
    for i, line in enumerate(cleaned):
        prev = padded[-1] if padded else ""
        # blank line before a list that directly follows non-list prose
        if _is_item(line) and prev.strip() and not _is_item(prev):
            padded.append("")
        padded.append(line)
        nxt = cleaned[i + 1] if i + 1 < len(cleaned) else ""
        # blank line after a list item when the next line dedents to base prose
        if (_is_item(line) and nxt.strip() and not _is_item(nxt)
                and not nxt[:1].isspace()):
            padded.append("")

    lines[:] = padded


def setup(app):
    app.connect("autodoc-process-docstring", _sanitize_docstring)
