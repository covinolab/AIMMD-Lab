Tutorials
=========

These Jupyter notebooks are rendered here from their **saved outputs**; to
run them yourself, launch them from ``examples/notebooks/`` in a clone of the
repository.

AIMMD
-----

Tutorials 1 and 2 run end to end on the built-in toy engine, with no GROMACS
and no GPU.

.. toctree::
   :maxdepth: 1

   1_toy_1d
   2_multi_system

- **1D toy system** (:doc:`1_toy_1d`) — a complete single-system AIMMD run on a
  1-D double-well toy model, from configuration through launching, training, and
  analysis of the resulting path ensemble.
- **Multi-system run** (:doc:`2_multi_system`) — two toy systems with different
  atom counts in one run, trained with a single shared committor network
  (contrasted with per-system networks), including per-system committor and rate
  analysis.

C-GAIMMD
--------

Committor-conditioned Generative AIMMD (:doc:`../cgaimmd`). Tutorial 3 runs on
a CPU in about a minute. Tutorials 4 to 6 are the production workflows of real
systems; they need a GPU and GROMACS, and tutorials 5 and 6 the BioEmu2
weights. Tutorial 7, the analysis of a run, runs on a CPU in about four
minutes.

.. toctree::
   :maxdepth: 1

   3_cgaimmd_toy_1d
   4_cgaimmd_painn_nanma
   5_cgaimmd_bioemu2_chignolin
   6_cgaimmd_bioemu2_trypsin_benzamidine
   7_cgaimmd_analysis

- **C-GAIMMD on the 1-D toy** (:doc:`3_cgaimmd_toy_1d`) — the C-GAIMMD loop on
  the toy engine with the minimal flow-matching network: the params file, the
  ``'ode'`` generator, the target-committor law, the preflight, shots before
  and after a training round, the final acceptance check (by default no
  committor filter), the sidecars and the shot table, and a generator of your
  own registered in about ten lines.
- **PaiNN C-GAIMMD on alanine dipeptide** (:doc:`4_cgaimmd_painn_nanma`) — the
  production workflow with GROMACS: generator atoms, unbiased end-state MD,
  the PaiNN network and its end-state pretraining, the params file, the
  preflight, a local launch and a job script, monitoring, restarts and
  post-training, rates and the committor against brute-force shooting.
  *Rendered from a GPU run: until that run is added, the page shows the code
  and the explanations without outputs.*
- **BioEmu2 C-GAIMMD on chignolin** (:doc:`5_cgaimmd_bioemu2_chignolin`) — the
  BioEmu2 system folder (atom map, completion of hydrogens, trunk embeddings),
  the params file with the sequential fit, the polish, re-solvation and
  minimization, the initial network, the preflight, a local launch and a
  cluster job script, restarts and monitoring.
  *Rendered from a GPU run: until that run is added, the page shows the code
  and the explanations without outputs.*
- **BioEmu2 C-GAIMMD on trypsin with benzamidine**
  (:doc:`6_cgaimmd_bioemu2_trypsin_benzamidine`) — what a ligand changes: the
  ligand atom map from its SMILES string, the completion of hydrogens and of
  the Ca²⁺ ion, a system-specific validity check of the calcium site, the
  pooled features and pair adapters, and the ligand weight of the fit.
  *Rendered from a GPU run: until that run is added, the page shows the code
  and the explanations without outputs.*
- **Analysing a C-GAIMMD run** (:doc:`7_cgaimmd_analysis`) — a two-minute
  C-GAIMMD run of a particle in a two-channel potential, then every analysis a
  study of a run needs: the shot table and the run progression; rates from the
  reweighting, checked against the transitions of the free simulations;
  free energies with the generated shooting points on top; the convergence of
  both; the committor against brute-force shooting next to a distance-matrix
  baseline; generator adherence and sample quality; shooting validation of
  generated configurations through an AIMMD sweep; and the run card. The
  reasoning behind the estimators is on :doc:`../cgaimmd_analysis`.
