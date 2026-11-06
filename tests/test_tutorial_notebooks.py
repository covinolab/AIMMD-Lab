"""The tutorial notebooks: the rendered copies match the examples, the drafts are valid, and
tutorial 3 (C-GAIMMD on the 1-D toy, CPU only) runs from top to bottom.

The notebooks live in ``examples/notebooks/``; the documentation renders them
from their saved outputs through the links in ``docs/source/tutorials/`` (it
never executes them). Tutorials 4 to 6 need a GPU, GROMACS and, for 5 and 6, the
BioEmu2 weights; they are checked here for valid notebook structure and
Python syntax only.

Executing tutorial 3 takes one to two minutes (``slow``). It needs nbformat,
nbclient and ipykernel (the ``tests`` extra); the kernel runs on this
interpreter, and the notebook's temporary folders go to the test's own
folder.
"""

import ast
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / 'examples' / 'notebooks'
RENDERED = ROOT / 'docs' / 'source' / 'tutorials'
TOY_TUTORIAL = '3_cgaimmd_toy_1d.ipynb'
GPU_TUTORIALS = ('4_cgaimmd_painn_nanma.ipynb', '5_cgaimmd_bioemu2_chignolin.ipynb',
                 '6_cgaimmd_bioemu2_trypsin_benzamidine.ipynb')


def _notebooks():
    return sorted(path.name for path in EXAMPLES.glob('*.ipynb'))


def test_every_tutorial_is_rendered_from_its_example():
    """docs/source/tutorials links every example notebook (or holds an identical copy), and no other."""
    assert set(_notebooks()) >= {TOY_TUTORIAL, *GPU_TUTORIALS}
    assert sorted(path.name for path in RENDERED.glob('*.ipynb')) == _notebooks()
    for name in _notebooks():
        assert (RENDERED / name).read_bytes() == (EXAMPLES / name).read_bytes(), name


@pytest.mark.parametrize('name', [TOY_TUTORIAL, *GPU_TUTORIALS])
def test_tutorial_code_is_valid_python(name):
    """Every code cell parses, and the notebook file is a version 4 notebook with a title."""
    notebook = json.loads((EXAMPLES / name).read_text())
    assert notebook['nbformat'] == 4
    cells = notebook['cells']
    assert cells[0]['cell_type'] == 'markdown' and ''.join(cells[0]['source']).startswith('# ')
    for index, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            try:
                ast.parse(source)
            except SyntaxError as exc:                                  # pragma: no cover
                pytest.fail(f'{name}, cell {index}: {exc}')


@pytest.mark.parametrize('name', GPU_TUTORIALS)
def test_gpu_tutorials_are_drafts_without_outputs(name):
    """Until a GPU run executes them, the GPU tutorials carry no outputs and say so on top."""
    notebook = json.loads((EXAMPLES / name).read_text())
    assert not any(cell.get('outputs') for cell in notebook['cells'] if cell['cell_type'] == 'code')
    assert 'Outputs pending' in ''.join(notebook['cells'][1]['source'])


@pytest.mark.slow
def test_toy_tutorial_runs(tmp_path):
    """Tutorial 3 runs from top to bottom on the CPU, with the outputs its text describes."""
    nbformat = pytest.importorskip('nbformat')
    nbclient = pytest.importorskip('nbclient')
    pytest.importorskip('ipykernel')
    pytest.importorskip('matplotlib')
    from jupyter_client.kernelspec import KernelSpecManager
    from jupyter_client.manager import KernelManager

    kernels = tmp_path / 'kernels'
    (kernels / 'this_python').mkdir(parents=True)
    (kernels / 'this_python' / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Python 3', 'language': 'python'}))
    work = tmp_path / 'work'
    work.mkdir()
    env = {**os.environ, 'TMPDIR': str(work)}
    notebook = nbformat.read(str(EXAMPLES / TOY_TUTORIAL), as_version=4)
    manager = KernelManager(kernel_name='this_python',
                            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
    client = nbclient.NotebookClient(notebook, km=manager, timeout=600,
                                     resources={'metadata': {'path': str(work)}})
    client.execute(env=env)

    text = '\n'.join(''.join(output.get('text', '')) for cell in notebook.cells
                     if cell.cell_type == 'code' for output in cell.get('outputs', []))
    assert 'preflight OK' in text
    assert "4. networkARB.json: {'round': 1, 'n_trained_shots': 4, 'status': 'trained'}" in text
    assert "'path000008.xtc']" in text                       # eight shots
    assert "generator: uniform_in_R {'high': 1.8, 'low': 1.2}" in text
    figures = [output for cell in notebook.cells if cell.cell_type == 'code'
               for output in cell.get('outputs', []) if 'image/png' in output.get('data', {})]
    assert len(figures) == 1
