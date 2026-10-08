# Validation metrics

Everything is computed on the validation libraries of the split (the capped split: 9 libraries with designed doubles, 24 position-pair
matrices, 8,494 doubles with both singles measured) after every epoch and once before training (the zero-shot row, dump tag `zs`).
The epistasis metrics are defined in `src/esm_msr/epi_metrics.py` (its docstring is the short version of this page); the analysis behind the
choice is in the epistasis metrics report (`docs/epistasis_metrics_report.html`).

## What is logged (20 numbers + 2 per library)

| group | name | read it as | higher is |
|---|---|---|---|
| ddG product | `val_rho_combined_avg` | ranking of all measured items by the reported prediction, mean over libraries | better |
| | `val_rmse_combined_avg` | calibration of the reported prediction (observed scale under `--link`), kcal/mol | worse |
| | `val_rho_wt_valid_avg` | the WT head on singles (drives the WT plateau and `--wt_early_stop_patience`) | better |
| epistasis, holistic | `val_epi_naive_rho_comb` | Spearman of predicted vs measured dddG, all doubles. **Mostly saturation under a link** | (summary only) |
| | `val_epi_naive_rho_add` | the same for the saturation-only control | (control) |
| global | `val_epi_global_err_comb` | RMSE of the predicted against the measured E[dddG \| x] curve, kcal/mol | worse |
| | `val_epi_global_err_add` | the same for the saturation-only control: what the link alone reproduces | (control) |
| | `val_epi_global_bias_comb` | mean (predicted - measured) of that curve; negative = nonspecific epistasis under-predicted | closer to 0 |
| beyond global | `val_epi_beyond_rho_comb` | everything above the global level at once (pair offsets dominate it) | better |
| | `val_epi_beyond_rho_ctx` | the same for the WT-in-context control | (control) |
| pair | `val_epi_pair_rho_comb` | pair offsets beyond the global curve, across 24 pairs. **Report, do not select on it** | better |
| line | `val_epi_line_rho_comb` | a substitution's average coupling with the other position, beyond the pair offset | better |
| | `val_epi_line_rho_ctx` | control | (control) |
| cell | `val_epi_cell_mag_comb` | double-centred residual magnitudes (the specific combination), per complete block | better |
| | `val_epi_cell_rank_mt` | **the primary target**: double-centred within-column ranks of the MT pass (saturation-free) | better |
| | `val_epi_cell_rank_comb` | the same for the reported prediction | better |
| | `val_epi_cell_rank_ctx` | the same for the WT-in-context control | (control) |
| | `val_epi_cell_flipacc_mt` | share of confident measured order reversals whose direction the MT pass predicts (0.5 = chance) | better |
| | `val_epi_cell_flipacc_ctx` | control | (control) |
| diagnostic | `val_epi_cell_sigsd_mt` | how much partner-dependent order the MT pass predicts at all (0 = additive readout) | (not a skill) |
| per library | `val_rho_combined/<lib>`, `val_rmse_combined/<lib>` | the ddG product per library | |

Counts, logged every validation: `val_epi_n_doubles`, `val_epi_n_pairs`, `val_epi_n_blocks`, `val_epi_n_flips` (8,494 / 24 / 24 / 2,509 on
the capped split; a change means the validation set changed).

The checkpoint name reads `val_rho_combined_avg`; `scripts/run_arm.sh` ranks checkpoints by `val_epi_cell_rank_mt`; the learning-rate
plateaus read `val_rho_wt_valid_avg` (WT adapter and calibration) and `val_epi_cell_rank_mt` (MT adapter and calibration).

## The hierarchy

A validation double (residue a at position i < j, residue b at j) of a library with wild-type stability c = dG_wt has a measured ddG_AB, both
measured singles, and dddG = ddG_AB - ddG_A - ddG_B. Its **additive expectation** is x = c + ddG_A + ddG_B: the dG it would have if the two
substitutions did not interact. The doubles of one position pair form a matrix (rows a, columns b), so dddG divides into

| level | what it is | share of dddG variance (validation, in-sample) |
|---|---|---|
| global | E[dddG \| x]: the assay's floor and ceiling and any nonspecific trend, a function of x only | 56% |
| pair | the matrix mean beyond the global curve (70% of it is between libraries) | 22% |
| line | a row or column mean beyond the pair mean: one substitution's average coupling with the other position | 14% |
| cell | the double-centred remainder: what depends on the specific combination | 6% |

The cell level is about the size of the doubles' own measurement noise (SD about 0.2 to 0.27 kcal/mol), and the line level contains the
measurement noise of the single every cell of the line shares. Both are therefore noisy targets; a model can still correlate with them because
part of each is real (trained models reach 0.24-0.31 at the cell level on magnitudes and 0.80-0.88 accuracy on confident reversals).

## How each metric keeps saturation and the other levels out

Saturation is a monotone measurement h of the latent dG. It adds a term to dddG that is a function of x (large below the floor: a double whose
additive expectation is below the floor is measured near the practical floor, so its dddG is about floor - x) and it attenuates real epistasis
where h is flat. On the observed scale h(c + a + b) is not additive in (a, b), so it leaks into every level of raw dddG: the old raw levels gave a
predictor that knows only the saturation 0.71 (pair), 0.48 (line) and 0.10 (double-centred cell) in simulation, and 0.50, 0.53 and 0.26 on the
real validation set. Two devices remove it:

* **Beyond-global residuals** (magnitude metrics: `beyond_rho`, `pair_rho`, `line_rho`, `cell_mag`). Measured: dG_AB - E[dG_AB | x], which equals
  dddG - E[dddG | x] (a 25-bin piecewise-linear smoother of the measured x). Predicted, for a head: its observed-scale double minus its own
  additive prediction pushed through its own link, e = P_AB - obs(L_A + L_B + o) (o, the head's median second difference, removes a
  calibration bias), then minus E[e | x_hat]. A head that is additive before its link has e = 0 and scores exactly 0 at every level here,
  whatever its link does to the singles. Pair offsets are the matrix means of these residuals, line effects their row and column means centred on
  the matrix mean, cells the double-centred residuals of a complete block.
* **Within-column order** (`cell_rank`, `cell_flipacc`, `cell_sigsd`). Inside one column (one library, so one c and one h; one partner) a
  monotone assay cannot change which of two doubles is more stable, so the order of ddG_AB there is saturation-free on the measured side, exactly.
  `cell_rank` rank-transforms every column of the measured and of the head's latent ddG_AB on a complete block, double-centres both, and
  correlates them (both orientations, mean per block, mean over blocks); any predictor whose order inside a column does not depend on the partner
  scores exactly 0. `cell_flipacc` takes the 2 x 2 sub-blocks whose measured order reverses between two columns by more than 0.6 kcal/mol on both
  sides (a monotone assay cannot reverse an order, so the reversal fixes the sign of the latent interaction contrast) and scores the share in
  which the head's latent contrast L_ij - L_i'j - L_ij' + L_i'j' has that sign; pair, line and single effects cancel exactly in the contrast, so
  an additive head scores 0.5.

Specificity in simulation (the real validation design, known components, a saturating assay, noise; mean of 16 replicates):

| predictor knows | naive | beyond | pair | line | cell_mag | cell_rank | flipacc |
|---|---|---|---|---|---|---|---|
| saturation only (additive + true link) | 0.58 | -0.04 | -0.05 | -0.03 | 0.01 | 0.00 | 0.55 |
| saturation imitated with noise, no link | 0.54 | 0.00 | 0.07 | -0.01 | 0.00 | 0.00 | 0.51 |
| + pair offsets | 0.72 | 0.63 | 0.84 | 0.11 | 0.02 | 0.00 | 0.54 |
| + line effects | 0.61 | 0.25 | 0.08 | 0.50 | 0.01 | 0.01 | 0.54 |
| + cell interaction | 0.60 | 0.18 | 0.10 | 0.13 | 0.37 | 0.29 | 0.89 |
| everything, with the link | 0.75 | 0.70 | 0.83 | 0.54 | 0.36 | 0.29 | 0.89 |
| everything, no saturation | 0.42 | 0.71 | 0.84 | 0.53 | 0.36 | 0.26 | 0.90 |

Read across a row: each level credits its own component. The naive metric credits saturation above everything else (a model that knows all of
the epistasis but not the saturation scores 0.42, one that knows only the saturation 0.58). Residual limits: the flip accuracy gives 0.54-0.55
to a predictor whose LATENT has a nonspecific curvature (noise reversals happen more often in the floor-compressed column); `cell_mag` can carry a
little of the line level through a sharp floor (line effects pushed through curvature are non-additive on the observed scale); `cell_rank` has
neither leak.

## Heads and controls

| head | latent ddG of an item | role |
|---|---|---|
| `comb` | (WT + ~MT)/2 | the reported prediction (inference `combined_pred`) |
| `mt` | ~MT, the MT pass alone | the epistasis readout (inference `epistasis_pred`); its non-additive part is comb's, doubled and without the additive WT half that reorders columns |
| `ctx` | (WT + ~WT)/2, the WT adapter also read on the mutated sequence | CONTROL: what the backbone plus single-mutant training already know. The MT adapter's contribution is `mt` or `comb` minus `ctx`. Needs `--val_cycle_passes` (now the default; about 1-2 min per validation) |
| `add` | WT, additive by construction | CONTROL for saturation: through the link it scores only by saturation. Logged for the naive and global metrics only; on every other one it is 0 (exactly at the residual levels, within 0.002 on cell_rank, 0.47-0.53 on the reversal accuracy across 67 past dumps) |

## Typical values and noise (capped split)

Mean over the devel arms (link, MT rank 2) and the released baseline (no link, MT rank 16) at epoch 2; `rep SD` is the SD across the three
runs of the same configuration (two seeds and a rerun), the noise to beat when comparing two configurations.

| metric | zero-shot | devel e2 | baseline e2 | rep SD | gain / rep SD |
|---|---|---|---|---|---|
| `naive_rho_comb` | 0.38 | 0.42 | 0.32 | 0.013 | 0.8 |
| `global_err_comb` | 0.55 | 0.70 | 0.85 | 0.031 | (worsens) |
| `global_bias_comb` | -0.35 | -0.52 | -0.65 | 0.030 | |
| `beyond_rho_comb` | 0.28 | 0.42 | 0.46 | 0.004 | 30 |
| `pair_rho_comb` | 0.31 | 0.46 | 0.52 | 0.020 | 7 |
| `line_rho_comb` | 0.21 | 0.39 | 0.39 | 0.013 | 13 |
| `cell_mag_comb` | 0.16 | 0.24 | 0.31 | 0.007 | 12 |
| `cell_rank_mt` | 0.12 | 0.16 | 0.20 | 0.005 | 9 |
| `cell_rank_comb` | 0.09 | 0.13 | 0.16 | 0.004 | 11 |
| `cell_rank_ctx` | 0.09 | 0.14 | 0.14 | (jitter 0.003) | |
| `cell_flipacc_mt` | 0.72 | 0.80 | 0.88 | 0.010 | 8 |
| `cell_flipacc_ctx` | 0.72 | 0.80 | 0.78 | (jitter 0.007) | |
| `cell_sigsd_mt` | 0.11 | 0.09 | 0.11 | 0.003 | |

Absolute values carry the sampling uncertainty of a 24-matrix validation set (90% pair-bootstrap interval of `cell_rank_mt` about +/-0.05);
paired comparisons on the same set are far tighter (the rep SD above). A difference between two single-seed runs needs about 2.8 x rep SD to be
believed (cell_rank_mt 0.014, cell_mag 0.02, flipacc 0.03, beyond 0.011, line 0.04, pair 0.06); averaging the last two or three epochs helps
(epoch-to-epoch jitter is about the size of the rep SD).

## What to optimise and stop on

1. **Select and early-stop on `val_epi_cell_rank_mt`.** It measures the level only epistasis-specific learning can move, it is exactly immune to
   saturation and to every higher level, and it has the best noise of the cell-level metrics. It replaces `val_rho_flip_pair_mt_avg` (r = 0.89
   across past runs).
2. **Confirm with `val_epi_cell_mag_comb` and `val_epi_cell_flipacc_mt`** (r = 0.80 and 0.82 with it; together they are the cell level seen on
   magnitudes, in rank space and as reversals). For HPO across many arms, an average of the three, each divided by its rep SD, is less noisy than
   any one of them.
3. **Attribute with the controls:** `cell_rank_mt - cell_rank_ctx` is what the MT adapter adds over the WT adapter read in context (about +0.02 in
   the devel arms, +0.06 in the released baseline).
4. **Guards:** `val_rmse_combined_avg` and `val_rho_wt_valid_avg` must not regress; `val_epi_beyond_rho_comb` (above the global level overall)
   should not fall when the cell level rises.
5. **Do not optimise** `naive_rho` (under the link it is mostly the link; it is uncorrelated or negatively correlated with every clean level
   across past runs), `pair_rho` (24 pairs from 9 libraries) or `cell_sigsd` (not a skill).
6. **`global_bias_comb`** tracks the saturation model: every trained model so far under-predicts the nonspecific epistasis by 0.5-0.7 kcal/mol
   because doubles far below the floor are measured near dG 0 to +0.5 (library-dependent), not at the link's -1. Use it to judge changes to the
   link or to floor censoring.

## Retired names (old -> new)

| old | now |
|---|---|
| `val_rho_flip_pair_mt_{avg,pooled}`, `val_rho_flip_pair_mt/<lib>` | `val_epi_cell_rank_mt` (doubles instead of conditional items, so every head and control is scored on the same cells) |
| `val_rho_flip_pair_wt_*` | `val_epi_cell_rank_ctx` |
| `val_rho_flip_mt_*` | retired: pooling partner positions mixed the line and cell levels |
| `val_rho_colrank_{mt,wt,wt_blind}_*` | retired: the within-column order is mostly the single-effect consensus (an additive head scores 0.58) |
| `val_epi_global_rho_<head>` | `val_epi_naive_rho_{comb,add}` |
| `val_epi_global_rmse_*` | `val_epi_global_err_*`, `val_epi_global_bias_comb` |
| `val_epi_pair_effect_*` | `val_epi_pair_rho_comb` (residualised; the raw version was saturation) |
| `val_epi_partner_effect_*`, `*_beyond_single` | `val_epi_line_rho_{comb,ctx}` |
| `val_epi_interaction_rank_raw_per_matrix_*` | `val_epi_cell_mag_comb` (residualised; raw gave the saturation-only control 0.26) |
| `val_epi_interaction_rank_ranked_per_matrix_*` | `val_epi_cell_rank_*` (ranks ddG_AB, not dddG: ranking dddG inside a column is not saturation-free) |
| `val_epi_matrix_rank_*`, `val_epi_partner_context_rank_*`, `val_epi_identity_effect_*`, `*_beyond_add` | retired (identity_effect gave the saturation-only control 0.6) |
| `val_rho_mt_valid_avg`, `val_auc_dead_{wt,mt}_*`, every `*_pooled` ddG metric, `val_rho_wt_valid/<lib>` | no longer logged; `stats.compute_metrics` still returns them, and every validation dump allows recomputing them |

## Offline recomputation

Every validation writes `training_logs/<run>/0/val_dump_<tag>.npz` (all items' latent scores, the reverse leg, dG_wt, the link). Score any dump,
past runs included, with the current definitions, optionally with pair-bootstrap intervals:

```
PYTHONPATH=src python scripts/epi_from_dump.py training_logs/<run>/0/val_dump_e5.npz [...] [--boot 200] [--csv out.csv]
```

Dumps written before this change have no dG_wt; the script then reads it from the library caches (`--cache`, default `cache_v7`).
