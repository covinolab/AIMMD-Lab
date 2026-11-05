# Files of the C-GAIMMD production runs of October 2026, for the old-name tests

C-GAIMMD was developed as "flow shooting" (F-AIMMD). The production runs of
that time wrote the old names (`chain_type = 'flow'`, the `flow_*` fields,
imports from `aimmd.flow`) into their params files. These files are copies of
what those runs used, so that the tests can check that the old names are
still read (for one release, `aimmd.cgaimmd._legacy`).

| file | source | used by |
|------|--------|---------|
| `params_chignolin_faimmd_equilibrium_free.py` | the params file of the online BioEmu2 chignolin campaign of 27 Oct 2026, verbatim (md5 `07b257c75fee8c3d869ff2d0bf1973c8`; archived on plutus in `bioemu2_campaign_backups/2026-10-27_chignolin-faimmd-equilibrium-free_full_run/`) | `tests/test_cgaimmd_archived_params.py`, with NANMA stand-ins for the chignolin system and stubs for `chignolin_states` and `bioemu2_committor` |
| `archived_chignolin_validity_hook.py` | lines 122-160 of the same file (the validity hook around `aimmd.flow.validate._polish`), verbatim, with the objects it uses set by the test | `tests/test_cgaimmd_polished_validity.py` and `tests/_cgaimmd_polish_golden.py` |

The params file still holds the paths of the JUPITER project folder (the
default of `BIOEMU2_COMMITTOR_CHIGNOLIN`, read by the params file itself);
the test sets that variable to its stand-in folder, so they are never used.
