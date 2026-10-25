# Validation metrics

Everything is computed on the validation libraries of the split (the capped split: 9 libraries with designed doubles, 24 position-pair
matrices, 8,494 doubles with both singles measured) after every epoch and once before training (the zero-shot row, dump tag `zs`).
The epistasis metrics are defined in `src/esm_msr/epi_metrics.py` (its docstring is the short version of this page); the analysis behind the
choice is in the epistasis metrics report (`docs/epistasis_metrics_report.html`).

## Authoritative epistasis definition and 2026-10-24 metric overhaul

As of **2026-10-24**, epistasis computed as:
$$\text{epi\_pred} = \sim\text{MT}(AB) - \text{WT}(AB)$$
is **authoritative across all training, validation, inference, and analysis scripts**.

### Rationale for the Overhaul:
1. **Authoritative Formulation ($\sim\text{MT}(AB) - \text{WT}(AB)$)**: Under `--per_mutation_bias`, the WT adapter on a double variant is
   additive by construction ($\text{WT}(AB) = \text{WT}(A) + \text{WT}(B)$). Evaluating the MT head relative to this additive wild-type
   baseline reads the exact non-additive increment the model predicts, at the cost of evaluating the double alone. The MT head is never asked
   for singles, avoiding the single-mutant shrinkage confound.
2. **Renaming `_mt` $\to$ `_epi`**: In all epistasis metrics, the head is evaluated as $\sim\text{MT}(AB) - \text{WT}(AB)$ (observed scale
   $P_{\text{epi}}(AB) - P_{\text{wt}}(AB)$). Calling this head `_mt` was misleading because it implied raw MT predictions; renaming to `_epi`
   explicitly identifies it as the epistasis prediction head.
3. **Dropping `_comb` from Epistasis Residuals**: The combined head mixes 50% additive WT head with 50% MT head. Halving the epistatic
   contrast compresses residuals by 2x (forcing $>97\%$ of variants into the $[-0.6, +0.6]$ kcal/mol neutral band), severely distorting
   residual classification and correlation without adding any signal. `comb` is retained only for `val_epi_naive_rho_comb` (the observed-scale
   product benchmark).
4. **Anchoring Global Error & Bias to `epi`**: `val_epi_global_err_epi` and `val_epi_global_bias_epi` track the RMSE and mean error of the
   calibrated and linked double-mutant predictions $P_{\text{epi}}$ directly against measured double stabilities. The flawed `global_dddg_bias*`
   metrics are deleted: because doubles partner with single mutants at destabilizing positions where models shrink toward the mean
   ($\sim +0.2$ kcal/mol per single), subtracting observed singles confounded epistasis bias with single-mutant shrinkage.
5. **New Residual Statistics**:
   - `val_epi_beyond_rho_pairavg_{epi,ctx}`: Mean across position pairs of within-pair residual Spearman correlation.
   - `val_epi_beyond_lib_{epi,ctx}`: Spearman correlation of library-centered residuals, stripping inter-library offsets.
   - `val_epi_class3_acc_{epi,ctx}`, `val_epi_class3_pos_recall_{epi,ctx}`, `val_epi_class3_neg_recall_{epi,ctx}`: 3-class classification
     accuracy (negative $< -0.6$, neutral $[-0.6, 0.6]$, positive $> +0.6$ kcal/mol) and confident positive/negative recall on residuals.

## Out-of-distribution benchmark sets are excluded (2026-10-19)

The three external benchmark libraries (`ptmuld`, `s461`, `ssym`) are **out of the training distribution** and are no
longer inside any aggregate. They are still scored and logged per library (`val_<metric>/<name>`) for diagnostics, and
summarised on their own as:

* `val_rho_combined_external` — mean `rho_combined` over those loaders
* `val_rmse_combined_external` — mean `rmse_combined` over those loaders

Everything below — every `val_*_avg`, `val_rho_ungrouped_*`, the NDCG@k aggregates and the whole `val_epi_*` hierarchy —
covers **in-distribution libraries only**. The list is the `--external_val_sets` flag (default `ptmuld,s461,ssym`), so it
travels with each run and with generated frozen-WT blocks.

## What is logged (28 epistasis metrics + 4 counts + 2 per library)

| group | name | read it as | higher is |
|---|---|---|---|
| ddG product | `val_rho_combined_avg` | ranking of all measured items by the reported prediction, mean over libraries | better |
| | `val_rmse_combined_avg` | calibration of the reported prediction (observed scale under `--link`), kcal/mol | worse |
| | `val_rho_wt_valid_avg` | the WT head on singles (drives the WT plateau and `--wt_early_stop_patience`) | better |
| epistasis, holistic | `val_epi_naive_rho_comb` | Spearman of predicted vs measured dddG, all doubles. **Mostly saturation under a link** | (summary only) |
| | `val_epi_naive_rho_epi` | the same for the epistasis head | better |
| | `val_epi_naive_rho_ctx` | the same for the context control | (control) |
| | `val_epi_naive_rho_add` | the same for the saturation-only control | (control) |
| global calibration | `val_epi_global_err_epi` | RMSE of calibrated $P_{\text{epi}}$ on measured doubles, kcal/mol | worse |
| | `val_epi_global_bias_epi` | mean ($P_{\text{epi}} - y$) on measured doubles; negative = under-predicted | closer to 0 |
| beyond global | `val_epi_beyond_rho_epi` | residual Spearman over all doubles: everything above the global curve at once | better |
| | `val_epi_beyond_rho_ctx` | the same for the WT-in-context control | (control) |
| | `val_epi_beyond_lib_epi` | residual Spearman with per-library means removed (no library offset) | better |
| | `val_epi_beyond_lib_ctx` | the same for the context control | (control) |
| | `val_epi_beyond_rho_pairavg_epi` | mean across position pairs of within-pair residual correlation | better |
| | `val_epi_beyond_rho_pairavg_ctx` | the same for the context control | (control) |
| pair | `val_epi_pair_rho_epi` | pair offsets beyond the global curve, across 24 pairs. **Report, do not select on it** | better |
| | `val_epi_pair_rho_ctx` | context control across pairs | (control) |
| line | `val_epi_line_rho_epi` | a substitution's average coupling with the other position, beyond the pair offset | better |
| | `val_epi_line_rho_ctx` | context control across lines | (control) |
| cell | `val_epi_cell_mag_epi` | double-centred residual magnitudes (the specific combination), per complete block | better |
| | `val_epi_cell_mag_ctx` | context control on cell magnitudes | (control) |
| | `val_epi_cell_rank_epi` | **the primary target**: double-centred within-column ranks of the epistasis pass (saturation-free) | better |
| | `val_epi_cell_rank_ctx` | context control on cell ranks | (control) |
| | `val_epi_cell_flipacc_epi` | share of confident measured order reversals whose direction the epistasis pass predicts (0.5 = chance) | better |
| | `val_epi_cell_flipacc_ctx` | context control on order reversals | (control) |
| classification | `val_epi_class3_acc_epi` | 3-class accuracy on residuals ($r < -0.6$, neutral $[-0.6, 0.6]$, $r > 0.6$ kcal/mol) | better |
| | `val_epi_class3_acc_ctx` | context control 3-class accuracy | (control) |
| | `val_epi_class3_pos_recall_epi` | recall on confident positive epistasis ($r > 0.6$ kcal/mol) | better |
| | `val_epi_class3_pos_recall_ctx` | context control positive recall | (control) |
| | `val_epi_class3_neg_recall_epi` | recall on confident negative epistasis ($r < -0.6$ kcal/mol) | better |
| | `val_epi_class3_neg_recall_ctx` | context control negative recall | (control) |
| per library | `val_rho_combined/<lib>`, `val_rmse_combined/<lib>` | the ddG product per library | |

Counts, logged every validation: `val_epi_n_doubles`, `val_epi_n_pairs`, `val_epi_n_blocks`, `val_epi_n_flips` (8,494 / 24 / 24 / 2,509 on
the capped split; a change means the validation set changed).

The checkpoint name reads `val_rho_combined_avg`; checkpoints are tracked by `val_epi_cell_rank_epi`; the learning-rate
plateaus read `val_rho_wt_valid_avg` (WT adapter) and `--mt_convergence_metric` (default `val_epi_beyond_rho_epi`).

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
| `epi` | ~MT, evaluated relative to WT: $\sim\text{MT}(AB) - \text{WT}(AB)$ | **AUTHORITATIVE EPISTASIS**: the MT pass alone beyond the WT additive baseline. Evaluates the non-additive interaction the model would actually report, without querying singles from the MT head. |
| `ctx` | ~WT, evaluated relative to WT: $\sim\text{WT}(AB) - \text{WT}(AB)$ | CONTROL: what the backbone plus single-mutant training already know about epistasis. Built from the reverse leg `wt_rev` on the mutated sequence. Needs `--val_cycle_passes`. |
| `comb` | (WT + ~MT)/2 | the reported prediction (inference `combined_pred`). Logged only for `val_epi_naive_rho_comb` (overall observed-scale benchmark). |
| `add` | WT, additive by construction | CONTROL for saturation: through the link it scores only by saturation. Logged for `val_epi_naive_rho_add` only; on every other level it is 0. |

## What to optimise and stop on

1. **Select on `val_epi_cell_rank_epi`.** It measures the level only epistasis-specific learning can move, it is exactly immune to
   saturation and to every higher level, and it has the best noise of the cell-level metrics.
2. **Monitor `val_epi_beyond_rho_epi` and `val_epi_beyond_rho_pairavg_epi`.** These measure epistasis correlation beyond the global curve
   over all doubles and averaged across pairs. `--mt_convergence_metric` defaults to `val_epi_beyond_rho_epi`.
3. **Confirm with `val_epi_cell_mag_epi`, `val_epi_cell_flipacc_epi`, and `val_epi_class3_acc_epi`.** Together they confirm cell epistasis on
   magnitudes, reversal classification ($|\Delta| > 0.6$ kcal/mol), and 3-class residual categorization.
4. **Attribute with the controls:** `cell_rank_epi - cell_rank_ctx` is what the MT adapter adds over the WT adapter read in context.
5. **Guards:** `val_rmse_combined_avg` and `val_rho_wt_valid_avg` must not regress; `val_epi_global_err_epi` tracks epistasis calibration.
6. **Do not optimise** `naive_rho` (under the link it is mostly the link), `pair_rho` (24 pairs from 9 libraries).

## Retired names (old -> now)

| old | now | rationale |
|---|---|---|
| `val_epi_*_mt` | `val_epi_*_epi` | Clarify that this head is the epistasis prediction ($\sim\text{MT} - \text{WT}$), not raw MT sequence output. |
| `val_epi_global_err_comb`, `global_bias_comb` | `val_epi_global_err_epi`, `global_bias_epi` | Anchored directly to the epistasis adapter predictions on double mutants. |
| `val_epi_global_dddg_bias*` | retired | Subtracted single mutants, which sit at destabilizing positions and shrink toward the mean ($\sim +0.2$ kcal/mol), corrupting epistasis bias with single shrinkage. |
| `val_epi_{beyond,pair,line,cell}_*_comb` | retired (kept only `naive_rho_comb`) | Mixing 50% additive WT head halved residual epistasis scale, compressing $>97\%$ of variants into the neutral band and obscuring epistasis skill. |
| `val_epi_cell_sigsd_*` | retired | Measure of spread / variation, not a predictive skill score. |
| `val_rho_flip_pair_mt_{avg,pooled}` | `val_epi_cell_rank_epi` | Standardized onto doubles population. |
| `val_rho_flip_pair_wt_*` | `val_epi_cell_rank_ctx` | Context control. |

## In `inference_scripts/esm_msr_testing.py`

Inference scripts output the authoritative epistasis prediction as `mt_lora_dddg_pred` and `epi_pred`, defined as:
$$\text{epi\_pred} = \sim\text{MT}(AB) - \text{WT}(AB)$$
For double mutants, $\text{WT}(AB)$ is additive by construction ($\text{WT}(A) + \text{WT}(B)$ under `--per_mutation_bias`), so this directly
evaluates the epistasis component the model predicts on the double alone. The legacy `combined_dddg_pred` ($0.5 \times (\text{MT} - \text{WT})$)
is preserved for backwards compatibility with earlier combined-head tables, but `mt_lora_dddg_pred` / `epi_pred` is the authoritative epistasis prediction.

### `--recalibrate linear|nonlinear` (proof of concept)

Replaces the checkpoint's calibration / link with one fitted to each whole evaluated dataset (a MegaScale scaffold, an external benchmark, a DMS,
the domainome set), per head, before any statistic or prediction file: `linear` is y = s * latent + b; `nonlinear` is
y = h(dG_wt + s * latent + b) - dG_wt with a freshly fitted monotone link h of the training link's family (floor, ceiling and softness all
learned; dG_wt = 0 where there is no dG scale). The latent columns are kept as `*_uncal`, the `*_dddg_pred` columns are recomputed on the new
scale, the fits and the RMSE before / after go to `<stats>_Recalibration.json`, and every output name gets `_recal-<mode>`. The stats files now
also carry `rmse_all`, `rmse_singles`, `rmse_doubles`, since rank statistics within a library cannot change under a monotone map. Fitted on
the evaluated data, it is an upper bound on what a recalibration can recover. On the capped validation libraries (o6g6_c10_hier, epoch 5) the
combined head's RMSE goes 0.745 -> 0.632 (linear) -> 0.585 (nonlinear), and the refitted floor lands at +0.29 kcal/mol, not the training
link's -0.99: the practical floor of the report, found independently.

## The training objective that goes with the cell level: `--lambda_mt_flip`

The confident-reversal loss (default off) trains the MT pass on the same information `cell_flipacc` scores: for every 2 x 2 sub-block of one
position pair inside a micro-batch whose measured within-column order reverses by more than `--flip_delta` (0.6 kcal/mol) on both sides, a
logistic loss `softplus(-sign * contrast / --flip_scale)` on the MT pass's latent contrast. Pair, line and single effects cancel and the
assay cannot create a reversal, so it pressures only the interaction, and only where the measurement is reliable. Columns of a pair travel
together when it is on (as for the `--mt_comp_*` weights). Logged: `train/L_flip_mt` (mean loss per reversal; log 2 = 0.69 for an additive
prediction), `train/flip_rev` (reversals in the batch) and `train/flip_rev_acc` (the share the MT pass already orders right). Its gradient
scale against the other terms is uncalibrated: probe it with `scripts/grad_share_probe.py` before choosing a weight. With three columns per
pair in a micro-batch a unit holds about 1.4 reversals on average (30% of units hold any), so the signal per step is thin.

## Whole-matrix losses: `--pack_pair_matrices`

Without it, the structured MT losses see only what one 64-row micro-batch holds: about three columns of a pair matrix. The component
regression then splits the error of a 19 x 3 slice (its "offset" is the pair offset plus three column effects, its "line" effects are means
over three partners), and the reversal loss finds about 1.4 confident reversals per unit. With `--pack_pair_matrices` (needs `--batch_size`
>= 361; use 400) every oriented pair matrix (one scored position x one partner position, up to 19 x 19 conditional items) travels whole in one
batch, and when the `--mt_comp_*` components or `--lambda_mt_flip` are on:

1. each batch first runs a no-grad MT forward over its matrix rows (a cache of latent predictions);
2. every MT micro-batch computes the components and the reversal loss on the WHOLE matrices of the batch, with its own rows live and the others
   from the cache. Its gradient reaches only its own rows, so the sum over micro-batches is the gradient of the whole-matrix losses exactly
   (tested against one micro-batch holding the whole batch), up to the dropout draw of the cached rows;
3. the logged `train/L_comp_*`, `train/L_flip_mt`, `train/flip_rev` come from the cache once per batch.

The components then are the hierarchy's levels (matrix offset, row and column effects over all partners, double-centred 19 x 19 interaction),
and every confident reversal of a matrix reaches the reversal loss. Costs: one extra no-grad MT forward per matrix row (about a third of the
MT pass's training cost on those rows), and fewer, larger steps per epoch (batch 400 instead of 256). The within-column ListMLE is unchanged
(its columns were already whole). Caveat: the `--reg_balance` constants were measured on 3-column blocks; whole-matrix components have a
different gradient scale (the offset of a 361-cell block is a mean over many more cells), so re-probe with `scripts/grad_share_probe.py`.

## Offline recomputation

Every validation writes `training_logs/<run>/0/val_dump_<tag>.npz` (all items' latent scores, the reverse leg, dG_wt, the link). Score any dump,
past runs included, with the current definitions, optionally with pair-bootstrap intervals:

```
PYTHONPATH=src python scripts/epi_from_dump.py training_logs/<run>/0/val_dump_e5.npz [...] [--boot 200] [--csv out.csv]
```

Dumps written before this change have no dG_wt; the script then reads it from the library caches (`--cache`, default `cache_v7`).
