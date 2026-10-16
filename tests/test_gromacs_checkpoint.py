"""
Tests for aimmd.engines.gromacs: reading GROMACS checkpoints with gmx dump.

The fixtures in tests/data are real ``gmx dump -cp`` outputs (GROMACS 2025.3
dumping checkpoints written by GROMACS 2025.4 on JUPITER), trimmed to the header
and the output-file section: the coordinate and velocity listings are left out.

- ``gmx_dump_cp_append.txt``: ``chainR3/forw.cpt`` of the kcmpd17 campaign, a
  shooting half (appending run); its xtc offset is above 2**32.
- ``gmx_dump_cp_noappend.txt``: ``freeA/traj000001.cpt`` of the same campaign,
  a free run (``-noappend``), whose output files carry the ``.partNNNN`` names.

No GROMACS is needed: ``gmx`` is replaced by small shell scripts.
"""

import hashlib
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from aimmd.engines.gromacs import (Checkpoint, CheckpointError, CheckpointOutput,
                                   checksum_matches, format_ps, gmx_command,
                                   parse_checkpoint_dump, read_checkpoint)

DATA = Path(__file__).parent / 'data'
APPEND = DATA / 'gmx_dump_cp_append.txt'
NOAPPEND = DATA / 'gmx_dump_cp_noappend.txt'


def _fake_gmx(tmp_path, body):
    """An executable shell script standing in for `gmx`."""
    script = tmp_path / 'fake_gmx'
    script.write_text('#!/bin/sh\n' + body)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def test_parses_the_real_dump_of_an_appending_run():
    with open(APPEND) as fh:
        cpt = parse_checkpoint_dump(fh, 'forw.cpt')
    assert (cpt.part, cpt.step, cpt.time) == (1, 110839300, 221678.6)
    assert [o.name for o in cpt.outputs] == ['forw.log', 'forw.xtc', 'forw.edr']
    xtc = cpt.output('run_nativecontact_A1/chainR3/forw.xtc')  # by base name
    # file_offset_high = 1, file_offset_low = 1509227540: above 2**32, and the
    # size of the trajectory on disk (the run stopped gracefully)
    assert xtc.offset == 2**32 + 1509227540 == 5804194836
    assert xtc.checksum_size == 1048576
    assert xtc.checksum == '7fe1ade499bc2da603c804600e1a448b'
    assert cpt.output('forw.trr') is None
    assert cpt.describe() == 'part 1, step 110839300, t = 221678.6 ps'


def test_parses_the_real_dump_of_a_noappend_run():
    with open(NOAPPEND) as fh:
        cpt = parse_checkpoint_dump(fh, 'traj000001.cpt')
    assert (cpt.part, cpt.step, cpt.time) == (1, 113950400, 227900.8)
    xtc = cpt.output('freeA/traj000001.part0001.xtc')
    assert xtc.offset == 2**32 + 1672411900
    assert cpt.output('traj000001.part0002.xtc') is None


def _dump(low='1509227540', high='1', n_outputs='1', with_outputs=True,
          checksum='7fe1ade499bc2da603c804600e1a448b'):
    lines = ['GROMACS version = 2025.4', 'simulation part # = 3',
             'step = 1000', 't = 2.000000', 'x (2):', '   x[0]= 1.0e+00',
             '   x[1]= 2.0e+00', 'energy_nsteps = 7']
    if with_outputs:
        lines += [f'number of output files = {n_outputs}',
                  'output filename = back.xtc', f'file_offset_high = {high}',
                  f'file_offset_low = {low}', 'file_checksum_size = 1048576',
                  f'file_checksum = {checksum}']
    return [line + '\n' for line in lines]


def test_offset_with_a_negative_low_half():
    # GROMACS prints the low 32 bits as a signed int
    cpt = parse_checkpoint_dump(_dump(low='-1294967296', high='2'))
    assert cpt.output('back.xtc').offset == 2 * 2**32 + 3000000000
    assert (cpt.part, cpt.step, cpt.time) == (3, 1000, 2.0)


def test_a_checksum_that_is_not_md5_counts_as_unrecorded():
    cpt = parse_checkpoint_dump(_dump(checksum='(not computed)'))
    output = cpt.output('back.xtc')
    assert (output.checksum_size, output.checksum) == (-1, '')


@pytest.mark.parametrize('lines, match', [
    (_dump(with_outputs=False), 'ends before its list of output files'),
    (_dump(n_outputs='2'), 'announces 2 output files but lists 1'),
    ([line for line in _dump() if not line.startswith('t =')], 'no time line'),
    ([line for line in _dump() if 'offset_low' not in line], 'no complete offset'),
    ([], 'no part / step / time'),
])
def test_incomplete_dumps_raise(lines, match):
    with pytest.raises(CheckpointError, match=match):
        parse_checkpoint_dump(lines, 'x.cpt')


# ---------------------------------------------------------------------------
# running gmx dump
# ---------------------------------------------------------------------------

def test_read_checkpoint_runs_gmx_dump(tmp_path):
    cpt = tmp_path / 'my run.cpt'  # quoted on the command line
    cpt.write_bytes(b'not read by the fake gmx')
    gmx = _fake_gmx(tmp_path, f'[ "$1 $2" = "dump -cp" ] || exit 3\n'
                              f'[ -f "$3" ] || exit 4\n'
                              f'cat "{APPEND}"\n'
                              f'echo "GROMACS reminds you: ..." >&2\n')
    checkpoint = read_checkpoint(str(cpt), gmx)
    assert isinstance(checkpoint, Checkpoint)
    assert checkpoint.fname == str(cpt)
    assert checkpoint.output('forw.xtc').offset == 5804194836


def test_read_checkpoint_reports_the_gromacs_error(tmp_path):
    cpt = tmp_path / 'back.cpt'
    cpt.write_bytes(b'x')
    gmx = _fake_gmx(tmp_path, (
        'echo "simulation part # = 1"\n'          # a partial dump on stdout
        'cat >&2 <<EOF\n'
        '-------------------------------------------------------\n'
        'Program:     gmx dump, version 2025.3\n'
        '\n'
        'Fatal error:\n'
        'Checkpoint file corrupted/truncated, or maybe you are out of disk\n'
        'space?\n'
        '\n'
        'For more information and tips for troubleshooting, please check\n'
        '-------------------------------------------------------\n'
        'EOF\n'
        'exit 1\n'))
    with pytest.raises(CheckpointError) as info:
        read_checkpoint(str(cpt), gmx)
    message = str(info.value)
    assert 'failed with exit code 1' in message
    assert 'Checkpoint file corrupted/truncated, or maybe you are out of disk ' \
           'space?' in message


def test_read_checkpoint_rejects_a_dump_cut_short(tmp_path):
    cpt = tmp_path / 'back.cpt'
    cpt.write_bytes(b'x')
    gmx = _fake_gmx(tmp_path, f'head -n 30 "{APPEND}"\n')
    with pytest.raises(CheckpointError, match='ends before its list of output'):
        read_checkpoint(str(cpt), gmx)


def test_read_checkpoint_missing_file_and_missing_gmx(tmp_path):
    with pytest.raises(CheckpointError, match='does not exist'):
        read_checkpoint(str(tmp_path / 'none.cpt'), 'gmx')
    cpt = tmp_path / 'back.cpt'
    cpt.write_bytes(b'x')
    with pytest.raises(CheckpointError, match='exit code 127'):
        read_checkpoint(str(cpt), str(tmp_path / 'no_such_gmx'))


def test_read_checkpoint_times_out(tmp_path):
    cpt = tmp_path / 'back.cpt'
    cpt.write_bytes(b'x')
    gmx = _fake_gmx(tmp_path, 'echo "simulation part # = 1"\nsleep 30\n')
    with pytest.raises(CheckpointError, match='did not finish within 0.5 s'):
        read_checkpoint(str(cpt), gmx, timeout=0.5)


# ---------------------------------------------------------------------------
# which gmx, and checksums
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('grompp, expected', [
    ('/opt/gromacs/bin/gmx grompp -maxwarn 1', '/opt/gromacs/bin/gmx'),
    ('srun -n 1 gmx_mpi grompp', 'srun -n 1 gmx_mpi'),
    (['gmx_a grompp', 'gmx_b grompp'], 'gmx_a'),   # multi-system list
])
def test_gmx_command_follows_grompp(grompp, expected):
    assert gmx_command(SimpleNamespace(gmx_grompp=grompp)) == expected


@pytest.mark.parametrize('grompp', ['None grompp -maxwarn 1', 'gmx_grompp_wrapper',
                                    None])
def test_gmx_command_falls_back_to_the_resolved_gromacs(monkeypatch, grompp):
    from aimmd import _config
    monkeypatch.setattr(_config, 'GROMACS', '/resolved/gmx', raising=False)
    assert gmx_command(SimpleNamespace(gmx_grompp=grompp)) == '/resolved/gmx'
    monkeypatch.setattr(_config, 'GROMACS', None)
    assert gmx_command(None) == 'gmx'


def _record(fname, offset):
    """The CheckpointOutput GROMACS would write for the first `offset` bytes."""
    size = min(offset, 1 << 20)
    with open(fname, 'rb') as fh:
        fh.seek(offset - size)
        digest = hashlib.md5(fh.read(size)).hexdigest()
    return CheckpointOutput(os.path.basename(fname), offset, size, digest)


def test_checksum_matches_like_gromacs(tmp_path):
    fname = tmp_path / 'back.xtc'
    fname.write_bytes(os.urandom(3 << 20))
    output = _record(fname, (5 << 19) + 7)          # 2.5 MiB + 7 bytes
    assert checksum_matches(fname, output) is True
    # bytes after the offset do not matter: mdrun truncates them
    with open(fname, 'r+b') as fh:
        fh.seek(output.offset + 10)
        fh.write(b'changed')
    assert checksum_matches(fname, output) is True
    # bytes inside the checksummed MiB do
    with open(fname, 'r+b') as fh:
        fh.seek(output.offset - 100)
        fh.write(b'changed')
    assert checksum_matches(fname, output) is False
    # a file shorter than the offset fails, as in GROMACS
    os.truncate(fname, output.offset - 1)
    assert checksum_matches(fname, output) is False


def test_checksum_of_a_short_file_and_without_checksum(tmp_path):
    fname = tmp_path / 'back.xtc'
    fname.write_bytes(b'abc' * 1000)
    assert checksum_matches(fname, _record(fname, 3000)) is True
    assert checksum_matches(fname, CheckpointOutput('back.xtc', 3000)) is None


def test_format_ps():
    assert format_ps(221678.6) == '221678.6'
    assert format_ps(300.0) == '300'
    assert format_ps(0.002) == '0.002'
