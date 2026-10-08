# Validation metrics

Everything is computed on the validation libraries of the split (the capped split: 28 loaders with measured values) after every epoch and once
before training (the zero-shot row). `*_avg` is the mean over libraries, `*_pooled` pools the items of all libraries first.

## 1. The epistasis hierarchy (`val_epi_<level>_<head>`)

Defined in `src/esm_msr/epi_hierarchy.py`; the full definitions are in its docstring. Target: the measured dddG of each double,
`ddG_AB - ddG_A - ddG_B`. Prediction: the same difference of a head's **observed-scale** values (`h(dG_wt + latent) - dG_wt`, so saturation is
modelled), `obs_AB - obs_A - obs_B`. Doubles are grouped as all doubles > position-pair matrix > column (one residue fixed, the other varying)
> cell. Each nesting level has an EFFECT (variation of group means, minus the parent's mean) and a RANK (ordering inside each group, averaged over groups).

| level | what is correlated |
|---|---|
| `global_rmse` | RMSE of predicted vs measured dddG, all doubles pooled (global effect, saturation included) |
| `global_rho` | Spearman over all doubles pooled (global rank) |
| `pair_effect` | across position pairs: mean predicted vs mean measured dddG of the matrix |
| `matrix_rank` | Spearman over the cells of one matrix, averaged over matrices |
| `partner_effect` | across all columns (both orientations): [column mean - matrix mean], predicted vs measured |
| `partner_context_rank` | Spearman within one column (a fixed partner, the scored substitution varying), averaged over columns |
| `identity_effect` | each complete matrix double-centred, then averaged per (residue at i, residue at j) over matrices; predicted table vs measured table |
| `interaction_rank_{ranked,raw}_per_matrix` | double-centre each complete matrix, correlate predicted with measured, average the per-matrix correlations. `ranked`: rank within columns first (immune to monotone distortion of a column's order); `raw`: the values. Strongly correlated across matrices (0.8 on the baseline) but not interchangeable: ranking discards magnitude, so `raw` is systematically higher. The `pooled` variants were dropped (within 0.03 of `per_matrix` everywhere) |
| `global_rho_beyond_add` | PARTIAL: `global_rho` given the additive observed-scale score of each double (the `wt_add` value of the double itself, not its second difference) |
| `matrix_rank_beyond_add` | PARTIAL: `matrix_rank` given the same additive score, inside each matrix |
| `partner_effect_beyond_single` | PARTIAL: `partner_effect` given the measured single ddG of the line's fixed mutation, centred on the matrix's lines like the effects (lines whose fixed mutation has no uncensored single are left out of the partial only) |

Heads (`<head>`), with `WT`/`MT` the adapter and `~` the opposite direction (mutated sequence in, reverse mutation asked, sign flipped):

| head | latent score of an item | needs |
|---|---|---|
| `wt_add` | `WT` (WT adapter, wild-type sequence in). Additive before the link: it scores only through saturation, so it is the CONTROL | nothing extra |
| `comb` | `(WT + ~MT)/2`: each adapter in its native direction; the reported prediction | nothing extra |
| `wt_ctx` | `(WT + ~WT)/2`: the WT adapter in both directions. The WT adapter never trains on a mutated sequence, so this is a transfer probe, not a trained predictor | `--val_cycle_passes` |

`(MT + ~MT)/2` (`mt_ctx`) was dropped: the forward legs of both adapters read the wild-type sequence in one pass and are additive (their second differences are a constant plus rounding), so the non-additive content of `comb` and of `mt_ctx` is the same `~MT` leg and the two have identical ranks (Spearman 1.0000 of their second differences on the released-code baseline; slightly different through the link on devel). The MT forward pass is no longer run in validation.

### Partial-correlation columns

A partial Spearman correlation is the correlation of predicted and measured dddG after both have been rank-regressed on a control; it asks whether the skill survives holding the control fixed.

* `*_beyond_add` (control: the double's additive observed-scale score). Saturation makes dddG a function of how far the additive prediction already sits from the assay floor (measured dddG correlates about -0.46 with the additive score on the baseline), and a head can score by learning that alone. What is left is skill about the residue pair. On devel the control passes through the link, so it contains the saturation; `wt_add` is then the full saturation control. The additive score is a proxy (a noisy prediction), so the partial under-corrects.
* `partner_effect_beyond_single` (control: the measured single ddG of the fixed mutation). A line's shift is partly how destabilising its fixed mutation is (their correlation was -0.58 on the baseline) and partly which residue it pairs with; this keeps the second.

Reading them: compare a head's partial with its plain value. A large drop means the plain value was mostly the confounder (on the baseline the global rho of `comb` falls from 0.35 to 0.24 and its lead over `wt_ctx` disappears; the within-matrix leads survive).

Also logged once: `val_epi_n_doubles`, `val_epi_n_pairs`, `val_epi_n_columns`, `val_epi_n_complete_blocks`, `val_epi_n_identities`. Counts are small
(about 24 pairs on the capped split): read the pair-level numbers with intervals (bootstrap over pairs), not decimals.

How saturation is handled: neither the per-unit correlations nor the parent-mean subtraction can remove saturation that bends dddG inside a
unit, so read each level against `wt_add`.

## 2. Training-aligned metrics (unchanged, scored on the conditional labels the MT head trains on)

`rho_combined`, `rmse_combined` (measured items: singles, doubles, native-background), `rho_wt_valid` (WT head, singles), `rho_mt_valid`
(MT head, conditional items), `rho_colrank_{mt,wt,wt_blind}` (within-column Spearman on conditional labels; `wt` is the WT head with the partner in its
context, `wt_blind` the WT head's score of the plain single), `rho_flip_{mt}`, `rho_flip_pair_{mt,wt}`, `auc_dead_{wt,mt}`. The MT learning-rate
plateau reads `rho_flip_pair_mt_avg`.

Retired (replaced by section 1): `rho_epi_full`, `rho_pair_offset`, `rho_subst_effect`.

## 3. Offline recomputation

`PYTHONPATH=src python scripts/epi_from_dump.py training_logs/<run>/0/val_dump_e7.npz ...` recomputes every level for `wt_add`, `wt_ctx` and `comb` from a dump (the baseline's flat format or devel's per-library one) with the current definitions, so the metric set can change without re-running validation.
