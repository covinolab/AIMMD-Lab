# Contributing to AIMMD

Thanks for your interest in improving AIMMD! This is a short guide; see the
[developer guide](https://aimmd-lab.readthedocs.io/en/latest/developer_guide.html)
for the architecture and a suggested reading order.

## Development setup

AIMMD needs a working GROMACS (`gmx` or `gmx_mpi` on `PATH`) for the full test
suite; the toy-engine tests need nothing extra. We recommend a clean conda
environment:

```bash
conda create -n aimmd python=3.13
conda activate aimmd
git clone https://github.com/covinolab/AIMMD.git
cd AIMMD
pip install -e ".[tests,docs]"
```

## Running the tests

```bash
pytest tests/
```

`tests/test_toy_1d.py` and `tests/test_multi_system.py` run end-to-end on the
toy engine (no GROMACS); `tests/test_retinal.py` exercises a realistic
GROMACS-based workflow.

## Building the documentation

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs/source docs/build/html
```

The docs import `aimmd` through dependency stubs in `docs/source/conf.py`, so
the build does not require GROMACS or the heavy runtime dependencies. If you
touch documented code, build the docs to confirm the API reference still
renders.

## Style and conventions

- **Docstrings:** NumPy style (rendered by `sphinx.ext.napoleon`). New public
  functions, classes, and `Params` fields should be documented — the parameter
  reference and API pages are generated from the code.
- **Mixin architecture:** the major classes (`Params`, `Path`, `PathEnsemble`,
  `Worker`, `Launcher`) are composed from numbered private mixin modules
  (`_fields.py`, `_io.py`, `_methods.py`, `_properties.py`, …). Keep that
  layering when extending them.
- **Backward compatibility:** new parameters should default to the previous
  behavior so existing `params.py` files keep working.

## Pull requests

1. Create a feature branch.
2. Make your change, with tests where practical.
3. Ensure `pytest tests/` passes and the docs build.
4. Open a pull request against `main` at
   <https://github.com/covinolab/AIMMD>.

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
