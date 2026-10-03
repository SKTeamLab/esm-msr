# Dual-adapter design: data sources, routing, and the assay's limits

How the two LoRA adapters divide the work, which measurements train which adapter, and
the three places the training data will mislead a model that takes it at face value.

`esm_msr/routing.py` is the executable version of the first two sections. If this document
and that module disagree, the module is right.

## 1. What each pass computes

Both passes score a mutation set as a summed log-likelihood ratio over the mutated
positions, mapped to ddG by a scalar calibration head:

```
LLR = sum_i [ logit(to_residue_i) - logit(from_residue_i) ]
```

They differ only in which sequence is fed to ESM3, and therefore which residue is visible
at the position being scored.

| | input sequence | adapter | what it measures |
|---|---|---|---|
| **WT pass** | `wt_sequence_tokens` — the "before" state | `peft_wt` | the effect of each mutation in the *before* background |
| **MT pass** | `mt_sequence_tokens` — the "after" state, carrying every mutation | `peft_mt` | the effect of each mutation *given the others are present* |

Sign convention is Tsuboyama's: `ddG = dG(after) - dG(before)`, dG is unfolding free
energy, so **positive ddG is stabilising**.

For a double mutant AB this gives the key identity. The WT pass sees wild type at both
sites, so it can only sum wild-type-context effects:

```
WT pass  ~  ddG_A + ddG_B                        (additive; structurally blind to epistasis)
MT pass  ~  ddG(A|B) + ddG(B|A)  =  2*ddG_AB - (ddG_A + ddG_B)
0.5*WT + 0.5*MT  =  ddG_AB                       (exact)
```

That is why the ensemble is a half-and-half average rather than a tuned blend: it is the
average of the two thermodynamic paths A->AB and B->AB. For k mutations it stays exact as
long as epistasis is at most pairwise; third-order terms come out 1.5x too large, which
matters for screening sets where many variants carry three or more mutations.

## 2. Subsets and routing

Produced by `data.MutationStabilityDataset`, routed by `routing.head_for`.

| subset_type | head | MT-pass input | target | measured? |
|---|---|---|---|---|
| `single` | WT | wild type + X | ddG_X | measured |
| `double` | ensemble | AB | ddG_AB | measured |
| `native_cond` | MT | bg + X (code `1A0N_L7S`) | ddG(X \| bg) | measured |
| `cond` | MT | AB | ddG_AB - ddG_B | derived |
| `reversion` | none | wild type | -ddG_X | derived |

Three things worth knowing:

**`cond` is one item per ordered pair.** A double with both singles measured yields
ddG(A|B) and ddG(B|A). It replaces the earlier `mut_ctx` / `mut_ctx_rev` pair, which
encoded the same two quantities twice over — same sequence, same position, score and target
both negated — differing only in which structure they conditioned on. That choice is now
`cond_structure`. Enabling both subsets used to count every double four times.

**The MT pass always sees the maximally mutated sequence.** For `cond` the input is
byte-identical to the parent double's input, and for `native_cond` it is background +
target. This is what makes one-forward-per-variant inference possible: a single MT forward
on a k-mutant yields all k conditional terms at once. The alternative — feeding the
leave-one-out sequence and scoring the held-out mutation — needs k forwards.

**`reversion` has no head, deliberately.** Its before-state is a mutant sequence (so the WT
pass would violate "the WT pass sees the real wild type") and its after-state is the wild
type (so the MT pass would be doing the WT pass's job). The algebra makes this concrete:
the WT pass on a reversion computes `-(mt-marginal)` against `-ddG`, which is the
constraint `mt-marginal ~ ddG` — exactly what `--mt_single_anchor_weight` asks of the MT
adapter, using ordinary `single` items. Training the WT adapter on reversions therefore
adds no thermodynamic information; it hands the MT adapter's task to the WT adapter and
spends WT capacity on an input distribution the WT adapter never sees at inference. Prefer
the anchor. Reversions remain available (`--incl_reversions`) for experiments.

Antisymmetry of the readout, incidentally, is free: the score is antisymmetric in
`wt_id`/`mt_id` by construction, so no augmentation can teach it.

## 3. Three ways the data will mislead you

### 3.1 The assay's dynamic range is not the stated one

`dG_ML` comes from a bounded maximum-likelihood fit and the measured distribution has
**zero mass** outside [-1, +5] kcal/mol — for singles and doubles alike. But -1 is not the
operative limit. dG is very nearly a monotone function of the K50 gap
(`log10 K50 - log10 K50_unfolded`; r = 0.983 across ~700k variants), and as that gap closes
dG stops being identifiable rather than being clipped. The authors say so:

> "ΔG values become unreliable if K50 approaches K50,F or K50,U … ΔG becomes very sensitive
> to K50 and its uncertainty increases relative to the uncertainty in K50."

In the lowest gap bin their own 95% interval runs to ~19 kcal/mol. For a marginally stable
domain the practical floor is nearer dG ~ +0.5 than -1.

Worked example, 2KVT. Wild type sits at dG 4.0-4.2 with a K50 gap of 2.5-3.1 and intervals
near 0.1. The singles `S35L` and `H37K` have gaps of 0.10 and 0.15, intervals of 0.44-0.66,
and reported dG of -0.61 and -0.11. The double `S35L:H37K` has a gap of 0.20 and dG
**+0.07** — nominally more folded than either single. Subtracting gives dddG of **+4.95
kcal/mol** of apparent stabilising epistasis. All three states are unfolded and all three
dG values are unconstrained. Every value is numeric and passes the authors' tabulated
filters.

### 3.2 Derived epistasis inherits that, as a bias

21% of doubles with both singles measured have an additive prediction
`dG_wt + ddG_A + ddG_B` below -1. The fit cannot report a value there, so for that fifth
the measured double is *obliged* to come back high and the shortfall surfaces as spurious
stabilising epistasis:

| additive prediction | n | share | mean dddG |
|---|---|---|---|
| dG > +0.5 (comfortably measurable) | 55,605 | 42% | +0.33 |
| dG in [-1, +0.5] | 48,384 | 37% | +0.89 |
| dG < -1 (unreachable) | 27,154 | 21% | **+2.06** |

Hence `--min_additive_dG` (default **-1.0**), applied only to double-derived subsets.
Singles are untouched: none sit below the floor, so the WT adapter keeps its full range of
destabilisation. Raising the threshold flattens the slope of dddG against the additive
prediction from -0.43 toward **-0.23**, where it stops moving — that residual is genuine
diminishing-returns physics, reproducible across the two proteases, and should not be
filtered away. Use the slope plateau to pick the threshold, not the mean: the mean keeps
falling past the plateau only because a stricter cut also selects milder pairs.

**Do not use density capping for this.** The retired 2D cap over (additive ddG, dddG)
discarded 34% of doubles and left the bias *worse* than no filter (mean dddG +0.96 vs
+0.89; unreachable share 24% vs 21%), because clipped pairs sit in the sparse tail of that
histogram and per-bin capping spares them while thinning the well-measured core. It also
sampled against a correlation that is substantially real. For family balance use
`--subset_caps`; for emphasis use the per-subset weights.

### 3.3 Noise compounds in derived targets

Reliability here means test-retest: the fraction of observed variance that is real signal.
Estimated from genuine duplicate measurements of `ddG_ML` (4,534 pairs, assumption-free):
**reliability 0.93, noise sigma 0.24 kcal/mol**. Propagating that through a k-term
difference (noise variance scales as k):

| trained label | terms | sigma obs | sigma noise | reliability | attenuation ceiling |
|---|---|---|---|---|---|
| ddG, single | 1 | 1.02 | 0.24 | 0.945 | 0.972 |
| ddG, double | 1 | 1.08 | 0.24 | 0.950 | 0.975 |
| ddG(A\|B) | 2 | 0.92 | 0.34 | 0.864 | 0.930 |
| dddG | 3 | 0.92 | 0.42 | 0.798 | 0.893 |

The ceiling is the best correlation any model can reach against a label this noisy. With
validation Spearman around 0.82, **label noise is not the binding constraint** — the model
is. A per-protease confidence filter moves these ceilings by one or two points, so it is
worth little on its own; the dynamic-range bias in 3.2 is the defect that matters, and no
reliability check detects it, because two replicates of an unidentifiable fit agree with
each other while both are pinned to the same wrong value.

One trap: the *combined* `deltaG_95CI` column is anti-conservative. For the 2KVT singles it
reads 0.16 while each protease's own interval is 0.44-0.46 — averaging two independently
unconstrained fits narrows the interval without adding information. Only the per-protease
intervals are informative.

### 3.4 How much epistasis is real

On doubles passing a per-protease interval filter, dddG spread is 0.84 kcal/mol. A
stability-only model (additive prediction, dG_wt, both single effects; whole domains held
out) reaches R^2 = 0.54 *without knowing which residues are involved*. Its residual has
spread 0.57 and reproduces at r = 0.68 across proteases, leaving **0.47 kcal/mol of
reproducible residue-specific coupling** that no function of total stability can explain —
consistent with the authors' independently reported typical couplings of 0.5-1.0 kcal/mol.
So a structure-aware conditional head has real signal to learn, but a majority of epistasis
is global.

Caveat on the distance evidence: the double mutants were *chosen* as structural contacts,
so 83% of even long-range pairs sit within 5 A and there is almost no non-contact control
group. A better control exists and is unused — three domains (2K5H, 1H8K, 1OPS) were each
scanned in wild-type form and in two mutant backgrounds, giving 5,512 single mutations
measured in both contexts: directly measured conditional effects with a real distance axis.

## 3.5 Batch planning: one adapter per work unit

`training._plan_units` groups a batch into work units of `(kind, rows)`, and each unit
drives exactly one adapter, so the inner loop never alternates between them:

| kind | pass | rows |
|---|---|---|
| `wt` | WT only | WT-head items (and multi-mutants) with a finite additive target |
| `mt` | MT only | MT-head items, plus anchored singles when `mt_single_anchor_weight > 0` |
| `combined` | both | only the legacy `lambda_*_combined` objective, whose loss couples the passes |

Singles can appear in both a `wt` unit and an `mt` unit when the anchor is on. That is the
same two forward passes as before, split so each runs alone.

WT units are sized by how many *distinct* backbone inputs they hold, not by
`micro_batch_size`. Every item of a library shares one wild-type sequence, so with
`dedup_backbone` the whole WT block is one forward: 64 singles at `micro_batch_size 8`
become a single unit rather than eight, saving seven backbone passes. MT units stay at
`micro_batch_size`, since each mutant sequence is a distinct input.

Verified against the previous implementation: the legacy-combined and no-anchor
configurations are bit-identical, and the anchored configuration agrees to 8e-7 on
gradients — float summation order from the regrouped units.

## 4. Structure handling

Each item stores **one** structure, shared by both passes, so masking happens at forward
time rather than in the cache.

* The cache is built unmasked (`--mask_mutated_structure`, default off). Every item reuses
  its library's structure, which is also why generation takes seconds per library rather
  than hours.
* `--mask_structure` blanks structure at `struct_mut_pos` on the **MT pass only** — the WT
  pass's sequence and structure agree, so there is nothing to mask. Both channels are
  blanked: ESM3 builds its affine frames from `structure_coords` *and* embeds
  `structure_tokens` separately, so masking one leaves the other fully informative.
* `struct_mut_pos` is every position where the MT-pass sequence differs from the sequence
  the structure represents — the item's own mutations plus any background mutation the
  structure does not carry. For `cond` it is a superset of `mut_pos`.
* The setting is recorded in `hparams.yaml` and re-applied at inference, so training and
  inference cannot silently disagree.

Known limitation: masking blanks the masked position's own coordinates and token, but
neighbouring structure tokens were encoded from unmasked coordinates and still carry that
residue's geometry. Removing the leak requires re-encoding from masked coordinates in the
forward pass.

**The structure encoder is not adaptable as things stand, and LoRA on it was removed.**
`ESM3.forward` never calls it — it consumes precomputed structure tokens, and the encoder
is only invoked from `ESM3.encode()`. Adapters placed there could never receive gradient.
Adapting it requires moving encoding into the forward pass, which the unmasked cache now
makes possible.

## 5. Sequence masking (`--mask_strategy`)

Off by default. This masks the *scored sequence position*, which is a different question
from structure masking. Measured zero-shot on three domains, 160 items each, against the
true target:

| readout | forwards per k-mutant | rho vs ddG(A\|B) | rho vs ddG_X (singles) |
|---|---|---|---|
| background-only input (leave-one-out) | k | 0.338 | **0.703** |
| **full mutant (default)** | **1** | **0.334** | 0.605 |
| full mutant, scored site masked | k | 0.338 | 0.693 |

On conditionals all three tie; on singles the unmasked full-mutant readout is ~0.10 worse,
because there the mutant residue at the scored position is pure leakage with no
compensating information. On conditionals the full-mutant input is the only readout where
both mutant residues are simultaneously visible, and that gain offsets the leak. Since it
also costs one forward instead of k, it is the default — and `--mt_single_anchor_weight`
exists partly to teach the MT adapter to discount the leak where clean single-mutant labels
are plentiful.

Note `marginal` masks every mutated position at once, which on a real multi-mutant would
destroy the conditioning that makes the MT pass informative.

## 6. Validation metrics

Four, each scoring a head only on what it is responsible for, logged per dataloader plus
`_avg` (mean over libraries; what the checkpoint monitor reads) and `_pooled` (all items
pooled, weighting libraries by size).

| metric | computed on |
|---|---|
| `val_rho_wt` | WT head, single mutations |
| `val_rho_combined` | the reported prediction, all measured items |
| `val_rho_mt` | MT head, conditional targets only (`cond`, `native_cond`) |
| `val_rmse_combined` | calibration of the reported prediction, kcal/mol |

Conditional targets are a different physical quantity from a wild-type-context ddG and are
never pooled with the others. `rho_mt` is deliberately scoped to them: `cond` targets are
mostly positive, so mixing them with singles inflates a per-library Spearman.
