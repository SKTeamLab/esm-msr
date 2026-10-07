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
| `interaction_rank_{ranked,raw}_{per_matrix,pooled}` | double-centre each complete matrix, correlate predicted with measured. `ranked`: rank within columns first (immune to monotone distortion); `raw`: the values. `per_matrix`: mean of per-matrix correlations; `pooled`: one correlation over all matrices |

Heads (`<head>`), with `WT`/`MT` the adapter and `~` the opposite direction (mutated sequence in, reverse mutation asked, sign flipped):

| head | latent score of an item | needs |
|---|---|---|
| `wt_add` | `WT` (WT adapter, wild-type sequence in). Additive before the link: it scores only through saturation, so it is the CONTROL | nothing extra |
| `comb` | `(WT + ~MT)/2`: each adapter in its native direction; the reported prediction | nothing extra |
| `wt_ctx` | `(WT + ~WT)/2`: the WT adapter in both directions | `--val_cycle_passes` |
| `mt_ctx` | `(MT + ~MT)/2`: the MT adapter in both directions (the forward MT input is one it never trains on: a probe) | `--val_cycle_passes` |

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
