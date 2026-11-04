# ESM-MSR Architecture & Modeling Features

*Authoritative consolidated reference on the dual-adapter architecture, readout link functions, teacher forcing, training paradigms, and experimental modeling features in ESM-MSR.*

---

## 1. Dual-Adapter Architecture & Two-Pass Logit Formulation

ESM-MSR predicts mutant stability ($\Delta\Delta G$) and epistatic couplings ($\Delta\Delta\Delta G$) by attaching low-rank adaptation (LoRA) modules to the frozen 1.4B `esm3_sm_open_v1` transformer backbone. To decouple single-mutation main effects from multi-site interactions, the model employs a **dual-adapter architecture** ([`src/esm_msr/models.py`](file:///home/sareeves/playground/esm-msr-devel/repo/src/esm_msr/models.py#L248-L630), [`routing.py`](file:///home/sareeves/playground/esm-msr-devel/repo/src/esm_msr/routing.py#L1-L72)).

### 1.1 The Two Passes
Both passes score mutations as summed log-likelihood ratios (LLRs) over the mutated positions, normalized via `log_softmax` over the 20 canonical amino acid token IDs:
$$\operatorname{LLR} = \sum_{i=1}^{n_{\text{mut}}} \bigl[ \operatorname{logit}(m_i) - \operatorname{logit}(w_i) \bigr]$$

| Pass | Input Sequence | Active Adapter | What It Evaluates |
| :--- | :--- | :--- | :--- |
| **WT pass** | `wt_sequence_tokens` (wild-type sequence) | `peft_wt` ($r=2, \alpha=4$ rsLoRA standard; or $r=1, \alpha=4$) | Effect of each mutation in the unmutated wild-type context ($\text{wt-marginal}$). |
| **MT pass** | `mt_sequence_tokens` (sequence with all mutations present) | `peft_mt` ($r=16, \alpha=16$ rsLoRA standard; or $r=8, \alpha=11$) | Effect of each mutation in the context of all partner mutations ($\text{mt-marginal}$). |

Sign convention follows Tsuboyama: $\Delta\Delta G = \Delta G(\text{after}) - \Delta G(\text{before})$, where $\Delta G$ is unfolding free energy ($\Delta\Delta G > 0$ indicates stabilization).

### 1.2 Thermodynamic Decomposition of Double Mutants
For a double mutant $AB$, the WT pass sees wild type at both sites, producing an additive expectation:
$$y_{\text{WT}} \sim \Delta\Delta G_A + \Delta\Delta G_B \quad (\text{structurally additive})$$
The MT pass evaluates each mutated site with its partner present:
$$y_{\text{MT}} \sim \Delta\Delta G(A \mid B) + \Delta\Delta G(B \mid A) = 2 \Delta\Delta G_{AB} - (\Delta\Delta G_A + \Delta\Delta G_B)$$
Combining the two halves gives:
$$y_{\text{comb}} = \tfrac{1}{2} y_{\text{WT}} + \tfrac{1}{2} y_{\text{MT}} = \Delta\Delta G_{AB} \quad (\text{exact})$$
The half-and-half ensemble is the exact average of the two thermodynamic integration paths ($A \to AB$ and $B \to AB$). Under pairwise epistasis, this identity is exact.

### 1.3 Calibration & Per-Mutation Bias
Logit scores are mapped to kcal/mol via affine calibration heads ([`CalibrationHead`](file:///home/sareeves/playground/esm-msr-devel/repo/src/esm_msr/models.py#L102-L150)):
$$y_{\text{cal}} = \bigl(\operatorname{softplus}(s_{\text{raw}}) + 10^{-6}\bigr) y_{\text{raw}} + b \cdot n_{\text{mut}}$$
Scaling the bias term $b$ by the number of mutations $n_{\text{mut}}$ (`per_mutation_bias: true`) guarantees that the WT adapter is strictly additive:
$$y_{\text{WT}}(AB) - y_{\text{WT}}(A) - y_{\text{WT}}(B) \equiv 0$$
Without this scaling, a scalar bias shifts double mutants by $b$ relative to their constituent singles ($0.1065$ kcal/mol error).

### 1.4 Single-Adapter Antisymmetric Cycle Readout
In single-adapter mode (`adapter_mode: fused`), a single shared adapter `'default'` scores both passes. To enforce thermodynamic consistency and eliminate calibration bias, the model supports the antisymmetric cycle readout ([`models.py`](file:///home/sareeves/playground/esm-msr-devel/repo/src/esm_msr/models.py#L912-L914)):
$$y_{\text{comb}} = \tfrac{1}{2}\bigl[ f_\theta(\text{WT}, w \to m) - f_\theta(\text{MT}, m \to w) \bigr]$$

---

## 2. Readout Link Function & Two-Source Censoring

### 2.1 The Experimental Unfolding Floor
High-throughput cDNA display proteolysis measures unfolding free energy ($\Delta G$) through the concentration of protease required for 50% digestion ($K_{50}$). When a protein is completely unfolded, $K_{50}$ collapses onto the unfolded baseline ($K_{50,U}$), rendering stability changes unidentifiable. Bounded maximum-likelihood fits report nominal values between $-1.0$ and $+5.0$ kcal/mol, but variance analysis demonstrates that below $\Delta G \approx 0.0\text{--}0.5$ kcal/mol, measurements represent an unfolding floor rather than true thermodynamic stability.

### 2.2 SaturatingLink Formulation
To prevent the model from learning assay truncation as true stabilizing epistasis, the latent thermodynamic prediction is projected onto the observed assay scale via [`SaturatingLink`](file:///home/sareeves/playground/esm-msr-devel/repo/src/esm_msr/link.py#L75-L167):
$$\Delta G_{\text{lat}} = \Delta G_{\text{wt}} + \Delta\Delta G_{\text{lat}}$$
$$h(x) = f + \tau \operatorname{softplus}\left(\frac{x - f}{\tau}\right)$$
$$\Delta\Delta G_{\text{obs}} = h(\Delta G_{\text{wt}} + \Delta\Delta G_{\text{lat}}) - \Delta G_{\text{wt}}$$
- **Floor $f$**: Fixed at $0.0$ kcal/mol (the physical unfolded state).
- **Knee width $\tau$**: Fixed at $RT \approx 0.59$ kcal/mol.
When unlinked predictions are requested (`--no-link`), [`IdentityLink`](file:///home/sareeves/playground/esm-msr-devel/repo/src/esm_msr/link.py#L168-L190) bypasses this projection.

### 2.3 Two-Source Censoring
Variants are classified into three censoring states (`cens` $\in \{-1, 0, +1\}$) in [`src/esm_msr/censoring.py`](file:///home/sareeves/playground/esm-msr-devel/repo/src/esm_msr/censoring.py#L44-L120):
1. **`SRC_RANGE` (1)**: String-tagged bounds from experimental fitting (`<-1` assigned `cens = -1` with bound $-1 - \Delta G_{\text{wt}}$; `>5` assigned `cens = +1` with bound $5 - \Delta G_{\text{wt}}$).
2. **`SRC_FLOOR` (2)**: Numeric measurements where $\Delta G_{\text{meas}} \le \text{censor\_threshold}$ (default $0.5$ kcal/mol), assigned `cens = -1` with bound $\text{threshold} - \Delta G_{\text{wt}}$.
3. **`SRC_GAP` (3)**: Permanently removed. Protease disagreement filtering introduced selection bias without improving model generalizability.

### 2.4 Loss Objectives
- **Censored ListMLE Ranking** ([`ListMLELoss.forward_censored`](file:///home/sareeves/playground/esm-msr-devel/repo/src/esm_msr/losses.py#L87-L150)): Evaluated on raw/unlinked predictions in chunks of 16. Lower-censored items (`cens <= 0`) are retained in the risk-set denominator but skipped in the outer log-likelihood sum. Ablating rank loss ($\lambda_{\text{rank}}=0$) causes multi-mutant correlation to collapse (`ptmul` $0.58 \to 0.525$).
- **Censored Hinge Regression** ([`hinge_regression_loss`](file:///home/sareeves/playground/esm-msr-devel/repo/src/esm_msr/losses.py#L158-L200)): Evaluated on linked $\Delta\Delta G_{\text{obs}}$. Uncensored variants receive exact MSE or Huber ($\delta=1.0$) loss; censored variants receive a one-sided hinge penalty clamped at the link knee:
  $$\text{eff\_bound} = \max(\text{bound}, f - \Delta G_{\text{wt}})$$

---

## 3. The Frozen-WT Training Paradigm

### 3.1 Motivation & Rationale
Joint training of both adapters can lead to training instability: the WT adapter may take full-rate optimizer steps on single-mutation metrics after convergence, destabilizing the additive baseline that the MT adapter uses for teacher forcing.

The **frozen-WT paradigm** (`configs/mt_frozen.yaml`) addresses this:
1. Pretrains `peft_wt` ($r=2, \alpha=4$) on single mutations to convergence (`wt_phase.yaml`).
2. Initializes the model via `--init_wt_from_checkpoint <path>`.
3. Activates `--frozen_wt_mt_only`, which freezes `peft_wt` and the link parameters, training only `peft_mt` ($r=16, \alpha=16$) on single + double mutant streams.

### 3.2 Verification & Health Checks
To confirm the WT weights and link buffers loaded correctly:
1. Log output confirms: `WT path warm-started from ...: N tensors loaded` and `Link warm-started from ...: 4/4 tensor(s)`.
2. Zero-shot validation at step 0 reports `val_rho_singles_wt_avg ≈ 0.84--0.85`.
3. During MT training, `self.model.peft_wt.eval()` is enforced ([`training.py`](file:///home/sareeves/playground/esm-msr-devel/repo/src/esm_msr/training.py#L724-L726)) to prevent stochastic LoRA dropout from injecting noise into anchor evaluations.
4. Adapter compatibility: `--init_wt_from_checkpoint` strictly validates parameter shapes, requiring the model to match the saved WT adapter configuration (`lora_rank_wt`, `lora_alpha_wt`, `per_mutation_bias`).

---

## 4. De-Linked Teacher Forcing & Mixture (`tf_mix`)

### 4.1 Bias-Variance Tradeoff in Additive Expectations
When training the MT head on double mutants, the additive reference $\Delta\Delta G_A + \Delta\Delta G_B$ can be provided by:
- **The measured singles label**: Accurate on average but carries independent measurement noise from two separate experimental assays and saturation truncation.
- **The WT adapter prediction**: Smooth, continuous, and non-saturated, but carries systematic model bias.

### 4.2 De-Linking Formulation
Because the measured label is reported on the saturated assay scale, simply adding two saturated singles distorts the underlying energy. Under `--tf_delink` (default), constituent singles are de-linked through the inverse saturating link before summation ([`src/esm_msr/training.py`](file:///home/sareeves/playground/esm-msr-devel/repo/src/esm_msr/training.py#L520-L555)):
$$\Delta\Delta G_{\text{tf,lat}} = h^{-1}(\Delta G_{\text{wt}} + \Delta\Delta G_A) + h^{-1}(\Delta G_{\text{wt}} + \Delta\Delta G_B) - 2\Delta G_{\text{wt}}$$
If either constituent single is at or below the floor, the item is flagged as unusable (`teacher_forcing_unusable`).

### 4.2b The Label of One Single (`--tf_label`, `--tf_no_label`, block training only)
- **`inverse`** (default): the inverse link, with a cut (`--tf_cut`, else `--censor_threshold`). A single below the cut, tagged or missing is read from the WT head on its own, so its partner keeps its measurement: one measurement and one guess (`--tf_no_label wt`). With `--tf_no_label mt` a double missing a measurement is regressed through the MT head alone instead, on half the epistasis scale of a labelled double. The per-item path has no read of each single's tag and still replaces the double's whole WT half. The leaky inverse diverges on the way to its clamp (a reading of -0.5 gives -5.36, -0.9 gives -11.77), which puts the combined prediction in the flat part of the Tobit loss whatever MT says.
- **`posterior`**: $E[z \mid y]$ with the prior $N(\text{WT read}, 0.75^2)$ under the Tobit likelihood, in closed form (`losses.tobit_posterior_mean`), per single and detached. No cut and no fallback: a reading above the floor is averaged with the prior (88% reading), a reading at the floor keeps the part of the prior below it, and a `<-1` single (which has no `ddG_A` / `ddG_B`) gets the survival term's truncated mean.
- **`clamped`**: the inverse link with the reading raised to 0.25, the same continuity without the model.

### 4.3 Mixture Tuning (`--tf_mix`)
`--tf_mix m` blends the de-linked measured label with the WT adapter prediction:
$$y_{\text{forced\_wt}} = m \cdot y_{\text{measured\_additive}} + (1 - m) \cdot y_{\text{WT\_pred}}$$
- **$m = 1.0$ (standard)**: Uses the measured additive label wherever usable, falling back to the WT adapter otherwise.
- **$m = 0.5$**: Arithmetic midpoint blending empirical measurement with smooth model predictions.
- **$m = 0.0$**: Pure model-based training with no teacher forcing.

### 4.3b Detaching the Teacher-Forcing Half (`--tf_detach_wt`)
`--tf_detach_wt` (default **on**, what every run before it did): the WT half of the combined prediction that teacher-forces the MT head carries no gradient to the WT head. The WT head learns from its own losses (singles), the MT head from the combined one. `--no-tf_detach_wt` lets the combined loss train the WT head on every item as well: its graph then lives until the MT pass has run (one backward through both, so two graphs of memory), it needs a trainable MT path, and it is refused with `--joint_doubles alternate` (which frees the WT graph first). `--joint_doubles` is the narrower switch (live WT half on the doubles only). It applies to the per-item loop and to `--block_training`.

### 4.4 Joint Training on the Real Doubles (`--joint_doubles`)
By default a double's WT half is a label or a *detached* WT prediction, so only the MT adapter learns from a double and the WT adapter sees singles alone. `--joint_doubles` makes the two adapters fit the real double together:
$$\mathcal{L}_{\text{joint}} = \ell\big(h(\,\tfrac12[\,\text{WT}(AB)+\text{MT}(AB)\,]\,),\; y_{AB}\big)$$
with the WT half carrying the **live** WT prediction (never an additive stand-in for the double itself) and the gradient reaching both adapters. The weight is `--lambda_reg_mt`. `--tf_mix` applies to the doubles as it does to singles: the WT half is $m\cdot y_{\text{measured\_additive}} + (1-m)\cdot \text{WT}(AB)$ where a label is usable (and $\text{WT}(AB)$ alone where none is), so the WT head gets the $(1-m)$ share of the gradient. At the default $m=1$ a labelled double trains MT alone, exactly as under `off`; use $m<1$ (0.5 mixes the label and the live prediction evenly, and 0 is a pure joint prediction). Doubles with no measured $dG_{wt}$ stay out of it (as they do of every linked regression).
- **`off`** (default): unchanged.
- **`both`**: one backward through the WT *and* the MT graph. Both graphs are alive together, so `--micro_batch_size` has to come down.
- **`alternate`**: each double keeps **one** adapter's graph. Down the micro-batch's doubles the owner alternates (and which side starts flips with every micro-batch); the half a double does not own is read under `no_grad`, so no pass ever builds a graph over rows it does not train and peak memory is one graph. The owner's loss is doubled, so the expectation over owners equals `both`'s update (tested: the mean of the two owner assignments' gradients is `both`'s gradient).
- The rank term is unchanged: still an MT-side list loss, with the WT half detached (a WT-owned double sits in its list as a constant).
- Masking is independent of this flag and composes with it: `--mask_strategy_wt marginal` lets the WT pass see *which* sites are mutated (and removes the wild-type residue it would otherwise read, so no reversion prior) while the MT pass, unmasked or `independent`, sees the mutant residues.
- Unmasked, the WT pass reads one sequence for a single and its double alike, so $\text{WT}(AB)=\text{WT}(A)+\text{WT}(B)$ exactly and the joint loss can only move a double through the parameters that also fit the singles. With `marginal` WT masking a double has its own context (both sites masked), and the WT half is no longer exactly additive.
- Needs both adapters trainable: refused with `--frozen_wt_mt_only`, `--mt_direct_pred`, `--freeze_mt_adapter`, `--adapter_mode fused`, `--lambda_reg_mt 0` or without doubles in `--subsets`. If either path is frozen during a run (the plateau schedule), it falls back to `off` from then on.

### 4.5 Random Masks and the Validation Regime
- Training masks come in two forms. The per-item path takes `--mask_strategy_wt/_mt none|marginal|independent`. Block training reads sites by `--block_mask_wt/_mt p` (the probability a head reads a position masked, §4.7). The per-item `random` strategy with `--mask_prob` / `--mask_dist_*` was removed: `--block_mask_*` is its per-position form and the pair block draws both sites together; the one thing it lost is the "exactly one of the two sites masked" pattern for a double.
- **Validation reads one regime (`--val_masking independent`, the default).** Every mutated site is masked on its own for **both** adapters, with the partner's mutant residue visible, so a single is read with its site masked and a double as the thermodynamic cycle (mutations $A$ at site $a$, $B$ at site $b$):
$$\underbrace{\text{WT}(a)+\text{MT}(B|A)}_{\text{path 1}}\quad\text{and}\quad\underbrace{\text{WT}(b)+\text{MT}(A|B)}_{\text{path 2}}$$
  where $\text{WT}(a)$ is the WT pass with site $a$ masked (partner wild type) and $\text{MT}(B|A)$ the MT pass on the sequence carrying $A$ with site $b$ masked. The mean of the two paths is exactly the reported $\tfrac12(\text{WT}+\text{MT})$; their difference is the cycle's closing error, logged as `val_cycle_gap` (mean $|{\rm path\,1}-{\rm path\,2}|$). Every `val_*` metric, the plateau schedule, early stopping and the checkpoint monitor read this regime. It is the input block training and the `random` masks train on (a block-trained adapter has never seen an unmasked site, and read unmasked it scores far below itself: `blk_b0` epoch 2, `val_rho_combined_avg` 0.820 unmasked against 0.853 masked), and it comes free from the singles' masked sequences. There are no separate unmasked, marginal-WT or ensemble numbers.
- **`--val_unmasked`** (off by default) adds the regime with neither adapter masked, logged as `val_unmasked_*` (per-library metrics, `_avg`, and the epistasis hierarchy as `val_unmasked_epi_*`) (with `val_unmasked_epi_composite`). Nothing watches it unless pointed at it: the plateau, early stopping and the checkpoint monitor read the default regime, and a run trained on visible sites sets `--monitor_metric val_unmasked_epi_composite --mt_convergence_metric unmasked_epi_composite`. It is the matched regime of a run trained on visible sites (a run with one head masked and one visible has no single matched regime: its WT numbers are matched in `val_unmasked_*` and its MT numbers in `val_*`). The MT side is one forward per variant, so it adds roughly what validation cost before it was consolidated.
- **`--val_masking train`** reads the passes with the run's own `--mask_strategy_wt/_mt` instead (unmasked when unset), which is what validation was before and what an unmasked-trained run is matched to. It has no cycle gap.
- **Epistasis log set (Nov 3).** The `_epi` head of the epistasis hierarchy (`val_epi_<metric>_epi`) is no longer logged. It is the MT head at twice the reported scale ($\text{MT}-\text{WT}$, where the reported double is $\tfrac12(\text{WT}+\text{MT})$): `beyond_rho`, `beyond_lib`, `beyond_rho_pairavg`, `pair_rho` and `line_rho` are bit-identical to their `_comb` twin (checked on all 62 earlier runs and in `tests/test_epi_log_names.py`), and `cell_mag`, `cell_flipacc`, `class3_*`, `global_err` / `global_bias` and `naive_rho` differ only by that scale or the observed-scale link, so `_comb`, the reported prediction, is the one kept. `cell_rank_epi` equals `val_diag_epi_cell_rank_mt_legacy` exactly (the MT head on its own, kept, and logged as `val_unmasked_diag_epi_cell_rank_mt_legacy` for the unmasked regime), and `val_diag_epi_clean_{err,bias}_epi` were written equal to their `_comb` twins. `--monitor_metric` / `--mt_convergence_metric` naming a retired metric are rewritten to the kept one at parse time (a monitor with no exact twin is refused). Earlier Comet runs keep their old series: Comet cannot delete a single metric.
- **Cost.** Every unique masked sequence is computed once per library through the model's logit cache: a WT pass masks one site of the wild-type sequence, so a library of $N$ sites needs $N$ forwards for all its singles and doubles; an MT pass masks one site of a mutant sequence, so the doubles of a position pair share their (mutation, partner site) sequences (a $2\times2$ grid of mutations costs 6 forwards, not 16). The cycle gap is a lookup.

### 4.6 Site-Grouped Throughput (removed)
`--group_by_site` (site-grouped batches and one forward per distinct sequence in the per-item path) was removed once block training made it redundant: a block is one site by construction, a masked site is read once for the whole alphabet, and `forward_site_llr` runs identical rows once. Measured on the real cache at batch 256 it had taken the per-item WT-marginal + MT-independent path from 0.29 to 1.7-2.1 it/s; block training at B = 20 runs a whole epoch of singles in about 520 steps. The commits that have it are `claude/joint-double-heads` @ 36617ef and earlier pins.

### 4.7 Block Training (`--block_training`, `--block_positions B`)
A training step is a **block** of one library (`esm_msr/blocks.py`), not a list of variants. A masked site reads out the whole alphabet at once, $\text{logit}[x]-\text{logit}[wt]$ for all 20 residues with the wild type reading exactly 0.
- **Singles block**: $B$ positions give a $[B,20]$ tensor from $B$ masked-site passes (the WT and the MT adapter see the same input, the wild type with the site masked). The observed singles are regressed on the observed scale and ranked as **one** ListMLE list of $B\times20$ cells (the unobserved cells are masked out).
- **Pair block**: one position pair $(i,j)$ gives every $19\times19$ double from **2 WT passes** (wild type with $i$ masked; with $j$ masked) and **38 MT passes** (19 row passes: substitution $x$ at $i$, $j$ masked, which reads every $y$ at $j$; 19 column passes the other way). WT cell $=\text{llr}_i[x]+\text{llr}_j[y]$, MT cell $=\text{Row}[x][y]+\text{Col}[y][x]$, the cycle of §4.5, and $\tfrac12(\text{WT}+\text{MT})$ is the prediction. Ranking uses lists of $B$ rows or $B$ columns of the cell matrix (the unit is drawn at random for each block). A test pins every one of the 361 cells to what the per-item `independent` strategy reads for that double.
- An epoch is every library's singles blocks, then every pair block (both orders random, seeded by epoch). Labels are rebuilt as dense $[P,20]$ / $[Q,20,20]$ tensors from the cache's items when the datasets are built; **the cache is not rebuilt** (the whole training cache loads in 17 s).
- **Identity in the lists (`--block_identity`, default on)**: the wild-type residue is a cell of every ListMLE list with label 0 and prediction 0 (no mutation, so no bias either; it carries no regression). A singles block is then the full $B\times20$; a pair block is the $20\times20$ matrix whose identity row and column are the singles (the 40 mutant passes include the two that read a single through the MT adapter), 400 cells ranked together; an MT conditional list holds the partner single as its identity cell. The scale the lists are scored on is `--rank_calibrated_stopgrad`'s: on, the stop-gradient calibrated prediction (divided by the WT scale), where "no mutation" is 0 and a mutation carries its bias; off, the raw LLR. Either way the rank loss sends no gradient to the calibration heads (the stop-gradient is on their scale and bias; the raw LLR does not pass through them). A conditional leg's list is always calibrated.
- **Direct conditional training (`--lambda_cond_mt`, with `--cond_partner_min_dg`)**: each real row or column pass is a conditional leg, $\text{lat}(\text{partner single}) + \text{cal}(\text{leg})$ regressed through the link against the *double's* label (never a difference of two saturated readings), plus ListMLE over the same legs (rank weight $\lambda_{cond}\lambda_{rank}^{MT}/\lambda_{reg}^{MT}$). Only cells whose partner single is observed, above `--cond_partner_min_dg` in absolute dG and above the link's floor are used: an unfolded partner leaves nothing to condition on. In the conditional frame the wild type at the masked site is exactly 0 because it *is* the partner single.
- **Memory (`--block_row_chunk N`)**: the passes run in checkpointed chunks of $N$ rows, exact gradients (0 = all at once). Measured on a 19.7 GB card at the longest training library (74 tokens): a singles block peaks at $3.0+0.33B$ GB (live WT, `--no-tf_detach_wt`: $3.0+0.62B$), and the 40-pass pair block peaks at 16.3 GB un-checkpointed (16.9 GB with `--lambda_cond_mt`, live WT and `--joint_doubles both`), 9.7 GB at $N=20$, 6.4 GB at $N=10$. Checkpointing adds a recompute and smaller chunks use the GPU worse: the pair block runs 0.76 s at $N=0$, 1.11 s at $N=20$, 1.34 s at $N=10$. Singles throughput saturates at $B\approx16$–24 (`analysis_scripts/block_memory_probe.py`).
- **Visible sites (`--block_mask_wt p`, `--block_mask_mt p`)**: the probability that a head reads a position with its site MASKED; 1 (default) is everything above, 0 reads every position with its residue visible, anything between draws per position. The two heads differ in cost. *WT unmasked* is the plain wild-type sequence read at every position, so a block is **one forward** (identical rows are deduplicated in `forward_site_llr`); a pair block's WT half is one forward for both sites. *MT unmasked* sees the mutation: each of a position's 19 substitutions is its own sequence (`logit[x]-logit[wt]` at the visible mutation), 19 passes instead of 1, drawn per position of a singles block; a pair block is drawn once per block and, unmasked, is one sequence per cell read at both sites (20 x 20 sequences, 800 reads sharing 400 forwards, instead of 40 passes), with no row / column passes, so that block carries no `--lambda_cond_mt` legs. Use `--block_row_chunk` (an even number, so a sequence's two reads share a chunk) for the MT cases. Validate such runs with `--val_unmasked`.
- **Cycle consistency (`--lambda_cycle`, `--cycle_detach_wt`)**: the thermodynamic cycle of §4.5 as a loss. For every real double of a pair block (all $19\times19$, labelled or not) the two paths round the cycle, $\text{WT}(x)+\text{MT}(Y|X)$ and $\text{WT}(y)+\text{MT}(X|Y)$, are put on the stop-gradient calibrated scale (one mutation per leg) and their mean squared difference is the loss, logged as `L_cycle` (validation's `val_cycle_gap` is the same quantity as a mean absolute value). It is label-free, so it constrains the many doubles that were never measured; it moves one path onto the other, not their mean, so it cannot by itself improve the reported prediction. By default the WT readouts are constants (only the MT conditionals move toward the fitted singles); `--no-cycle_detach_wt` trains the WT head with it too. A visible-site pair block has no row / column passes and carries no cycle loss. `--lambda_cond_mt` is the labelled counterpart: each leg is regressed on the double's own label, which also closes the gap.
- **Gradient accumulation (`--grad_accum_blocks K`)**: a block is one protein's positions, so a small B (the unmasked-MT arms use B = 4 for memory) means many, noisy optimizer steps: B = 4 takes 2,356 steps an epoch against 697 at B = 20. K >= 2 adds up K consecutive blocks' gradients in `.grad` (the size of the trainable parameters, about 0.1 GB, and no activation memory, which is what limits B), divides by K, clips and steps once. Warm-up (`--lr_warmup_steps`) and the cosine length count optimizer steps (the cosine length is divided by K); logged `train/*` losses are the mean over the K blocks; blocks left at the end of an epoch carry into the next, and a resume starts a fresh window. It composes with every block option and is refused outside `--block_training`.
- **Tobit (`--reg_nll tobit`)** works in block training: the regression is the noisy-floor likelihood on the latent absolute dG ($\mathrm{dG}_{wt}$ + the cell's latent ddG), the link only reports the observed scale, and `--censor_threshold` only ties the floor cells' ranks (only the assay's `<-1` / `>5` range tags are censored for the regression; the library carries each cell's `cens_src`). It composes with `--joint_doubles` and `--lambda_cond_mt` here, which the per-item path refuses.
- **List unit and scale (`--block_list_rows R`, `--block_rank_norm`)**: by default a singles block is ONE list of $B\times20$ cells and a pair block's lists hold $B$ rows or columns. ListMLE **sums** its terms over a list (the regression is a mean), so the rank term grows with the list: at $B=8$ the logged list loss is about 500 against a regression loss of about 0.1, roughly 20 times more rank-heavy than the per-item path's 16-member lists at the same $\lambda$. `--block_list_rows 1` ranks each position (20 cells with the wild type) on its own, so a list is about the per-item path's size. `--block_rank_norm ref` rescales every list to `--rank_list_size` members' worth (loss $\times$ `rank_list_size` / valid members) so the balance of `--lambda_rank_*` : `--lambda_reg_*` no longer depends on $B$ or the list unit, which leaves the list unit a pure question of which cells compete.
- The objective is the per-item one on tensors: WT head on the singles, MT head through $\tfrac12(\text{forced WT half}+\text{MT})$ with `--tf_mix`; with `--joint_doubles` other than `off` a pair's WT half is live (its graph is only 2 rows, so `both` costs nothing). `--batch_size`, `--micro_batch_size`, `--rank_list_size` (but as the reference length of `--block_rank_norm ref`) and `--mask_strategy_*` do not apply; validation is unchanged. Refused with `--mt_direct_pred`, fused adapters and `--rank_group_by partner`; `native_cond` items are left out.

---

## 5. Nonspecific Epistasis Modeling (`--nonspecific_term fx`)

### 5.1 Additive vs Epistatic Variance in MegaScale
ANOVA variance-partition analyses demonstrate that ~90% of the variance in raw double-mutant non-additivity ($\Delta\Delta\Delta G$) is driven by global assay saturation, per-position-pair offsets, and single-mutant marginal scales. Only ~2--5% of variance represents true residue-specific chemical coupling.

### 5.2 Formulation
The nonspecific term models identity-independent non-additivity as an explicit function of additive expectation:
$$y_{\text{pred}} = y_{\text{comb}} + f(y_{\text{WT}})$$
Where $f(x)$ is parameterized as a smooth, monotonic neural spline or piecewise polynomial. In initial single-seed evaluations, adding $f(x)$ improved cell-level rankings; however, canonical training relies directly on `SaturatingLink` to model saturation non-additivity without introducing auxiliary unidentifiable parameters.
