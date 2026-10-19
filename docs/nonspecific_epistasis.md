# Nonspecific epistasis: the `--nonspecific_term fx` result (and why to revisit it)

Status: **the single best cell-level result recorded in this repo so far, on one seed — unconfirmed, and the
code is not on any live branch.** This document records what was measured, what the term learned, why it is
plausible, and the experiment that would settle it. Written 2026-10-18.

## 1. What the run was

| | |
|---|---|
| arm | `bl_w2a4_m16a16_capped_link_ns` |
| flags | released basis (`bl_w2a4_m16a16_capped`) + `--link softclamp` + `--nonspecific_term fx`, seed 1, 8 epochs, `--micro_batch_size 64`, `--val_cycle_passes` |
| control | `bl_w2a4_m16a16_capped_link` — *identical* flags without `--nonspecific_term`, same seed, same 8 epochs |
| code | `claude/main-baseline` @ `70be32e` (the control ran at `8cc1be2a`) |
| ran | 2026-10-14 03:12 → 07:30 |
| logs | `run_logs/bl_w2a4_m16a16_capped_link_ns.log`, `training_logs/bl_w2a4_m16a16_capped_link_ns/0/metrics.csv`, val dumps `val_dump_e0..e7.npz` (carry `ns_term` + `link_summary`) |

**The code is not on a live branch.** `70be32e` is reachable only through the tag
`refs/tags/archive/main-baseline-2026-10-14` (the `claude/main-baseline` worktree is at `03a6fff` and does
*not* contain it). Any revisit must start by resurrecting that commit (`git show 70be32e`; it touches
`config.py`, `link.py`, `training.py`, `tests/test_link.py`, +145/−6 lines).

## 2. What it computes

`f(x)` is a learned **piecewise-linear function on fixed knots** (default −3…5 kcal/mol every 0.5,
initialised at 0, constant beyond the outer knots), added to a **multi-mutant's latent** and never to a
single's:

* `x = dG_wt + the additive latent ddG` (the forced WT half of the double, **detached**), i.e. f is a
  function of the *additive expectation* of the double, not of its identity;
* f is added **before** the link, so the link still models the readout alone;
* at validation the comb readout gains f and the MT readout 2f, so `comb = (WT + MT)/2` still holds;
* `f` has its own optimizer group (no warm-up, never cut by a plateau), lives in checkpoints as
  `ns_head.*`, is logged every step as `ns/f_at_x*`, and is written into the val dumps as `ns_term`;
* requires `--link`; **`inference_scripts` do not read it yet**, so a model trained with it cannot currently
  be shipped without finishing that plumbing.

## 3. What was measured (mean of the last 3 epochs, epoch 7 of 8)

Same basis, same seed, same length — only the flag differs:

| metric | control (`+link softclamp`) | **`+ns fx`** | delta |
|---|---|---|---|
| `val_epi_cell_rank_mt` | 0.2017 | **0.2126** | +0.0109 |
| `val_epi_cell_mag_comb` | 0.2844 | **0.3038** | +0.0194 |
| `val_epi_cell_flipacc_mt` | 0.8652 | **0.8707** | +0.0056 |
| `val_epi_beyond_rho_mt` | +0.5048 | **+0.5639** | +0.0591 |
| `val_epi_beyond_rho_comb` | +0.4833 | **+0.5499** | +0.0665 |
| `val_rho_doubles_combined_avg` (guard) | 0.7134 | 0.7172 | +0.0038 |
| `val_rho_combined_avg` (guard) | 0.8383 | 0.8401 | +0.0018 |
| `val_epi_global_bias_comb` | −0.559 | −0.578 | −0.019 |
| `val_epi_naive_rho_mt` (not a target) | 0.3595 | 0.5713 | +0.2118 |

It beats **every** arm of its own series on the cell metrics and on the beyond-additive level:

| arm (same basis, seed 1, 8 epochs) | cell_rank_mt | cell_mag_comb | flipacc_mt | beyond_rho_mt | beyond_rho_comb |
|---|---|---|---|---|---|
| `+link softclamp` (control) | 0.2017 | 0.2844 | 0.8652 | +0.5048 | +0.4833 |
| **`+nonspecific_term fx`** | **0.2126** | **0.3038** | **0.8707** | **+0.5639** | **+0.5499** |
| `+link floor` (per-library floors, val) | 0.1972 | 0.2721 | 0.8389 | +0.5147 | +0.4959 |
| `--tf_singles latent_link` | 0.1841 | 0.2694 | 0.8559 | +0.4752 | +0.4752 |
| `--tf_singles predicted` | 0.1902 | 0.2789 | 0.8387 | +0.5276 | +0.4903 |

For completeness, the dc-series canonical and the released model on the same libraries (`*_comb` level,
last 3 epochs; the released arm's cell metrics are offline rescoring of its dumps, on the raw latent scale,
so it is informative rather than a matched control):

| arm | rank_comb | mag_comb | flip_comb | beyond_comb |
|---|---|---|---|---|
| released, no link (rescored) | 0.1557 | **0.3131** | 0.8600 | +0.4816 |
| `+ns fx` | **0.1694** | 0.3038 | **0.8708** | **+0.5499** |
| dc17 canonical | 0.1597 | 0.2704 | 0.8497 | +0.5041 |

## 4. What f learned

`ns/f_at_x*` at the end of training (kcal/mol, added to the latent):

| x (additive expectation) | −1 | 0 | +1 | +2 | +3 |
|---|---|---|---|---|---|
| f(x), epoch 0 | +0.072 | +0.037 | +0.031 | +0.044 | −0.030 |
| f(x), epoch 7 | **+0.504** | **+0.167** | −0.075 | −0.182 | −0.189 |

f is **monotone decreasing in x**: doubles whose additive expectation is low (stabilising) are *rescued*
(+0.5 at x = −1), doubles whose additive expectation is high are made *more destabilising* (−0.19 at
x ≥ +2, then flat). That is the shape of a saturation/floor effect, learned from data, and its size
(≈0.5 kcal/mol at the destabilising end) matches the independent Oct-8 finding that trained models
under-predict the nonspecific term by 0.5–0.7 kcal/mol.

## 5. Why it is plausible (the argument behind the term)

From the 2026-10-08 variance analysis (kept as a record in `docs/EXPERIMENTS.md` §2): of the variance of measured
dddG, the largest share — 56% — is a function of the **additive expectation alone**; pair offsets are 22%,
line effects 14%, and only ~6% is the specific combination. The floor/saturation makes even a perfectly
additive model look epistatic on the observed scale, so a predictor that knows *only* the saturation scores
0.58 on the naive dddG correlation, more than one that knows every epistasis component but not the
saturation (0.42). Giving that dependence its own learned term is therefore not a heuristic: it is the
dominant term of the measured variance, and the previous architecture asked the link and the MT adapter to
absorb it implicitly.

## 6. Caveats (why this is a lead, not a result)

1. **One seed.** Between-run spread on this basis is not small: across the four matched arms above, the
   spread of the last-3-epoch mean is ≈0.013 (`cell_rank_mt`), 0.015 (`cell_mag_comb`), 0.037
   (`beyond_rho_mt`), 0.015 (`flipacc_mt`). The `ns` advantage is ~0.9 SD, 1.3 SD, **1.6 SD** and 0.4 SD
   respectively — the robust part is `beyond_rho_*`, not the rank/flip cell metrics.
2. **The control is at a different commit** of the same line, and the term changes the training dynamics
   (its own optimizer group, never decayed).
3. **Much of the gain is on `naive_rho_mt` (+0.21)**, the metric the project deliberately does not
   optimise because saturation leaks into it. That is consistent with the term absorbing the
   nonspecific/saturation component — which is the point — but it also means part of the apparent gain is
   the confounded level moving, and the honest summary is that the **saturation-free** level
   (`beyond_rho_*`) moved by +0.06 while the naive level moved by +0.21.
4. **`global_bias_comb` got slightly worse** (−0.019): the term improved interaction ranking without fixing
   the absolute scale. Whatever it learned is not the link's residual bias.

## 7. The experiment to run (revisit plan)

**Status 2026-10-18: step 1 (the port) is DONE; the rest is planned, not executed.** The port is on branch
`feature/nonspecific-term` (`7db75a7`), worktree `.claude/worktrees/nonspecific-term`, parent `b34093f` (the
dc17/dc18/dc19 basis). `tests/test_nonspecific_term.py` covers the table, the comb/MT/epi identity, x's bias
accounting with and without `--per_mutation_bias`, that f learns from doubles only, and that the term neither
trains the WT half nor is trained by it. The arms, the queue mechanics and the decision rule are in
[`queue_takeover_ns.md`](queue_takeover_ns.md); the task below is what that plan schedules.

Goal: establish whether the term helps on the *current* canonical basis, and separate "nonspecific
(saturation-shaped)" from "extra capacity".

1. **Port it** — done, see above (the original was `70be32e` on `main-baseline`; the port touches
   `config.py`, `link.py`, `training.py` and adds its own test file).
2. **Matched arms, two seeds each**, 10 epochs, exact dc17 flags plus one change:
   * `ns1_sN` — dc17 + `--nonspecific_term fx`;
   * `ctrl_s2` — dc17 alone, seed 2 (dc17 seed 1 already exists as `dc17_3dec_nowtdbl_pmb_w2a4_m16a16_s1`).
3. **Ablations that make the result interpretable** (only after step 2 shows the effect):
   * **knot resolution** — default 0.5 vs a 2-knot (floor/ceiling-only) version: if the 2-knot version captures
     most of the gain, the term is a saturation model, not a curve;
   * **freeze f after epoch 2** vs learn throughout: guards against the term drifting to absorb the target
     late (its optimizer group is never decayed by design);
   * **x definition** — the detached additive latent vs the measured additivity, which is a stricter test of
     the "function of the additive expectation" claim;
   * with and without the link floor pin (dc18) — the two saturation devices may be redundant.
4. **Read it on**: mean of the last 3 epochs of `val_epi_beyond_rho_mt` / `_comb` and `cell_mag_comb`
   (the metrics that moved beyond their spread), then `cell_rank_mt`, `cell_flipacc_mt`; guards
   `val_rho_wt_valid_avg`, `val_epi_beyond_rho_comb` and `val_rmse_combined_avg` must not regress; report
   `val_epi_global_bias_comb` and the final `ns/f_at_x*` curve. Two seeds, and treat < 2 × between-seed SD
   as nothing.
5. **Before any adoption**: teach `inference_scripts/esm_msr_testing.py` to load `ns_head.*` from the
   checkpoint (it currently ignores it), and add the term to `esm_msr_testing.py`'s scoring, or the external
   benchmarks will silently be scored without it.

## 8. Related

* `docs/validation_metrics.md` — what each metric in §3 measures.
* `docs/anova_epistasis_report.md` — the variance partition and the saturation simulation behind §5.
* `docs/epistasis_metrics_report.html` — the report version, with figures.
* `docs/EXPERIMENTS.md` — where the run sits in the sequence, and the dc-series baselines to compare against.
