"""The tutorial notebooks: the rendered copies match the examples, the drafts are valid, and
tutorials 3 (C-GAIMMD on the 1-D toy) and 7 (analysing a C-GAIMMD run), both CPU only, run
from top to bottom.

The notebooks live in ``examples/notebooks/``; the documentation renders them
from their saved outputs through the links in ``docs/source/tutorials/`` (it
never executes them). Tutorials 4 to 6 need a GPU, GROMACS and, for 5 and 6, the
BioEmu2 weights; they are checked here for valid notebook structure and
Python syntax only.

Executing tutorial 3 takes one to two minutes and tutorial 7 about four
(``slow``). They need nbformat, nbclient and ipykernel (the ``tests`` extra);
the kernel runs on this interpreter, and the notebooks' temporary folders go to
the test's own folder.
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
ANALYSIS_TUTORIAL = '7_cgaimmd_analysis.ipynb'
GPU_TUTORIALS = ('4_cgaimmd_painn_nanma.ipynb', '5_cgaimmd_bioemu2_chignolin.ipynb',
                 '6_cgaimmd_bioemu2_trypsin_benzamidine.ipynb')


def _notebooks():
    return sorted(path.name for path in EXAMPLES.glob('*.ipynb'))


def test_every_tutorial_is_rendered_from_its_example():
    """docs/source/tutorials links every example notebook (or holds an identical copy), and no other."""
    assert set(_notebooks()) >= {TOY_TUTORIAL, ANALYSIS_TUTORIAL, *GPU_TUTORIALS}
    assert sorted(path.name for path in RENDERED.glob('*.ipynb')) == _notebooks()
    for name in _notebooks():
        assert (RENDERED / name).read_bytes() == (EXAMPLES / name).read_bytes(), name


@pytest.mark.parametrize('name', [TOY_TUTORIAL, ANALYSIS_TUTORIAL, *GPU_TUTORIALS])
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


def _execute(name, tmp_path):
    """Run the notebook `name` of examples/notebooks on a kernel of this interpreter."""
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
    notebook = nbformat.read(str(EXAMPLES / name), as_version=4)
    manager = KernelManager(kernel_name='this_python',
                            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
    client = nbclient.NotebookClient(notebook, km=manager, timeout=600,
                                     resources={'metadata': {'path': str(work)}})
    client.execute(env=env)
    outputs = [output for cell in notebook.cells if cell.cell_type == 'code'
               for output in cell.get('outputs', [])]
    text = '\n'.join(''.join(output.get('text', '')) for output in outputs)
    figures = [output for output in outputs if 'image/png' in output.get('data', {})]
    return text, figures, work


@pytest.mark.slow
def test_toy_tutorial_runs(tmp_path):
    """Tutorial 3 runs from top to bottom on the CPU, with the outputs its text describes."""
    text, figures, _ = _execute(TOY_TUTORIAL, tmp_path)
    assert 'preflight OK' in text
    assert "4. networkARB.json: {'round': 1, 'n_trained_shots': 4, 'status': 'trained'}" in text
    assert "'path000008.xtc']" in text                       # eight shots
    assert "generator: uniform_in_R {'high': 1.8, 'low': 1.2}" in text
    assert len(figures) == 1


@pytest.mark.slow
def test_analysis_tutorial_runs(tmp_path):
    """Tutorial 7 makes its run, analyses it with every tool its text names, saves its results
    outside the run folder and shows its six figures; no temporary path is printed."""
    text, figures, work = _execute(ANALYSIS_TUTORIAL, tmp_path)
    for line in ('shots in 4 chains', "the trainer's last estimate", 'reweight_run, cached values',
                 'values recomputed with run/networkARB.h5', 'free simulations alone',
                 'share by the last end state visited', 'against the exact free energy',
                 'distance MLP trained on', 'C-GAIMMD model (last round)', 'data ceiling',
                 'near-copy line', 'under the final check', 'frames to shoot', 'dial slope',
                 'settings in 17 sections'):
        assert line in text, line
    for saved in ('committor_parity.png', 'distance_mlp.pt', 'rates.json', 'run_card.tex',
                  'validation_frames'):
        assert saved in text, saved
    assert str(work) not in text
    assert len(figures) == 6
    (folder,) = list(work.glob('cgaimmd_analysis_*'))
    for saved in ('rates.*', 'run_card.*', 'committor_parity.*', 'distance_mlp.pt',
                  'frames_table.json'):
        assert list((folder / 'analysis').rglob(saved)), saved
        assert not list((folder / 'run').rglob(saved)), saved
