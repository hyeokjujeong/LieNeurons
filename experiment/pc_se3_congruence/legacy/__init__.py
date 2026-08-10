"""Superseded entry points, kept for reproducibility of past reported numbers.

Nothing in here is on the current path.  New work goes through the modules that
remain in ``experiment/pc_se3_congruence/`` (see their module docstrings).  These
scripts are retained because published tables in the report ``.md`` files were
produced by them, and because several of them are the NEGATIVE CONTROLS that
justify the current design -- deleting them would delete the evidence.

Generation 1 -- global vector pooling (2026-07-28 .. 2026-08-04).
Abandoned: pooling the point axis into global vector channels forces every
channel into Fix_H(rho), which collapses rank(K) from 6 to 3 on centro- and
C2-symmetric clouds.  Superseded by the pointwise pipeline
(``pointwise_models.py`` / ``pointwise_graph.py``), which keeps the point axis
until the Gram so a symmetry may PERMUTE factors instead of having to fix them.

    verify.py                  structural checks O/L/A/B/C for ModelA/B/C.
                               -> verify_pointwise.py (checks A-F)
    train.py                   ModelB training sanity run.
                               -> blockage_bench.py
    visualize_expB.py          spatial/body congruence animations for ModelB.
    analyze_klein_gate.py      vector-vs-covector equivalence, cross-slot
                               blockage, Klein gate/ReLU study.  The source of
                               the "Klein gate yes, Klein norm no" decision.
    check_killing_degeneracy.py  se(3) Killing form rank-3 degeneracy.

Generation 2 -- compact-kernel second moment (2026-08-06 .. 2026-08-07).
Abandoned: the backbone indexed channels by NEIGHBOUR RANK, which is not
invariant under relabelling an equal-distance shell, so exact distance ties
(cubic lattice, tetrahedral orbits) break equivariance.  Measured permutation
error 1.9e-01 vs 3.0e-16 for the pointwise model.

    symmetry_rank_diagnostic.py  rank collapse on symmetric clouds.
    run_tensor_kernel_suite.py   object-suite driver.
                                 -> run_pointwise_suite.py

The library modules these depend on (``models.py``, ``encoders.py``,
``data_synth.py``) deliberately stay in the parent package: ``blockage_bench.py``
still imports the superseded models from them as COMPARISON ARMS, so they are
live code, not legacy.  Each is split into current/legacy sections internally.

All of these still run from the repository root, e.g.

    python experiment/pc_se3_congruence/legacy/verify.py
"""
