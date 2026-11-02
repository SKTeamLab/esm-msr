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

### 4.5 Random Masks and the Four-Pass Cycle
- **`--mask_strategy_wt/_mt random --mask_prob p`** (default $p=0.5$): on a *training* pass every mutated site is masked independently with probability $p$ (a different draw on each pass), and the readout is the plain $\text{logit}[mt]-\text{logit}[wt]$ at each mutated site on that partly masked sequence. A double then meets all four combinations of masked and visible sites, so one model covers the masked and the unmasked setup (a visible partner on the MT pass lets an interaction show; a masked one hides it). $p=1$ is `marginal`, $p=0$ is unmasked. An eval-mode pass (validation, scoring) runs unmasked, and a checkpoint trained this way is scored unmasked unless `--mask_strategy_wt/_mt marginal|independent` is passed to the testing script.
- **`--val_four_pass`** adds the thermodynamic cycle of a double (mutations $A$ at site $a$, $B$ at site $b$) to validation, on the `independent` readout of both adapters:
$$\underbrace{\text{WT}(a)+\text{MT}(B|A)}_{\text{path 1}}\quad\text{and}\quad\underbrace{\text{WT}(b)+\text{MT}(A|B)}_{\text{path 2}}$$
  where $\text{WT}(a)$ is the WT pass with site $a$ masked (partner wild type) and $\text{MT}(B|A)$ the MT pass on the sequence carrying $A$ with site $b$ masked. The mean of the two paths is exactly the reported $\tfrac12(\text{WT}+\text{MT})$ of this regime; their difference is the cycle's closing error. Logged as `val_c4_*` (per-library metrics and their `_avg`, the epistasis hierarchy as `val_c4_epi_*`, and `val_c4_cycle_gap`, the mean $|{\rm path\,1}-{\rm path\,2}|$), written to the validation dump as `wt4/mt4/comb4/gap4`, and never watched by a plateau or the checkpoint monitor.
- **Cost.** Every unique masked sequence is computed once per library through the model's logit cache: a WT pass masks one site of the wild-type sequence, so a library of $N$ sites needs $N$ forwards for all its singles and doubles; an MT pass masks one site of a mutant sequence, so the doubles of a position pair share their (mutation, partner site) sequences (a $2\times2$ grid of mutations costs 6 forwards, not 16). A second visit to the library, or the standard regime when it is already `independent`, is a lookup.
- **Validation regimes (`--val_four_pass`, on by default).** Multiples are always scored with the `independent` readout of both heads (`val_c4_*`; the thermodynamic cycle above), because it comes free from the singles' masked sequences and reads a double the way it was derived. The WT head is also scored `val_wt_unmasked_*` (one forward per library) and `val_wt_marginal_*` (every mutated site masked at once), and `val_ens_*` is the unmasked WT score averaged with the cycle's prediction, which an inference run can always afford. The configured-regime numbers (`val_*`, no prefix) are unchanged and still drive the plateau and the checkpoint monitor unless pointed at `c4_*`.

### 4.6 Site-Grouped Throughput (`--group_by_site`, `--mask_dist`)
Under masking, the input of a pass depends on the *site*, not on what the site was mutated to: every single at a masked position is the same sequence, and so is every double of a masked position pair. `--group_by_site` exploits that:
- the batch sampler keeps whole sites together (the singles at one position, the doubles of one position pair; the groups come in random order and a group may be cut at a batch boundary);
- `forward_partitioned` reads each distinct sequence of a library once and indexes the logits per row: an unmasked WT pass is one forward for the whole group, a masked site is one forward whatever it was mutated to (the masked `marginal` / `independent` strategies already deduplicated; this extends it to the `random` strategy and the plain unmasked pass), so memory scales with the distinct sequences and `--micro_batch_size` can be raised a lot;
- ListMLE lists are drawn from a random permutation of the micro-batch, since consecutive rows are now neighbours of one site.
It changes how rows are drawn and computed, not the losses. With LoRA dropout on, rows of one sequence now share one draw.

Measured on the real training cache (joint recipe, batch 256, validation skipped, one 32 GB card; training steps per second):

| setting | it/s | backbone rows per step | reserved GB |
|---|---:|---:|---:|
| unmasked, micro-batch 64 (before) | 0.43 | | 24.5 |
| WT `marginal` + MT `independent`, micro-batch 32 (before) | 0.29 | about 450 | 20.9 |
| the same with `--group_by_site`, micro-batch 256 | 1.7-2.1 | about 35 | 28.9 |
| `--group_by_site`, both passes `random` (1 2 1), micro-batch 128 | 0.57 | | 29.1 |
| `--group_by_site`, unmasked | 0.50 | | 25.1 |

The gain is in the passes whose input is the same for a whole group: an MT pass that masks its own site (`independent`, or a `random` pattern that always masks one site) dedups, while an MT pass that leaves the mutation visible is one sequence per variant whatever the batching. What is left of a fast step is not data: assembling a batch costs 0.025 s against a 0.5 s step (the whole training cache loads in 17 s from the per-item pickles), and about 35 distinct sequences leave the GPU launch-bound (about 36% utilised) with the backward pass at 0.18 s, the LoRA forwards at 0.09 s and the ListMLE list loop at 0.04 s of it.

`--mask_dist_wt` / `--mask_dist_mt NONE ONE BOTH` set the pattern a `random` pass is masked with (one setting per pass), as weights of: no mutated site, exactly one (uniformly chosen), all of them (the same event for a single). `1 2 1` is the independent $p=0.5$ draw; `0 1 0` always masks exactly one site (on the MT pass the mutated site of a single is always masked, so it dedups like `independent`); `1 1 2` masks more. Each pass draws its own pattern.

### 4.7 Block Training (`--block_training`, `--block_positions B`)
A training step is a **block** of one library (`esm_msr/blocks.py`), not a list of variants. A masked site reads out the whole alphabet at once, $\text{logit}[x]-\text{logit}[wt]$ for all 20 residues with the wild type reading exactly 0.
- **Singles block**: $B$ positions give a $[B,20]$ tensor from $B$ masked-site passes (the WT and the MT adapter see the same input, the wild type with the site masked). The observed singles are regressed on the observed scale and ranked as **one** ListMLE list of $B\times20$ cells (the unobserved cells are masked out).
- **Pair block**: one position pair $(i,j)$ gives every $19\times19$ double from **2 WT passes** (wild type with $i$ masked; with $j$ masked) and **38 MT passes** (19 row passes: substitution $x$ at $i$, $j$ masked, which reads every $y$ at $j$; 19 column passes the other way). WT cell $=\text{llr}_i[x]+\text{llr}_j[y]$, MT cell $=\text{Row}[x][y]+\text{Col}[y][x]$, the cycle of §4.5, and $\tfrac12(\text{WT}+\text{MT})$ is the prediction. Ranking uses lists of $B$ rows or $B$ columns of the cell matrix (the unit is drawn at random for each block). A test pins every one of the 361 cells to what the per-item `independent` strategy reads for that double.
- An epoch is every library's singles blocks, then every pair block (both orders random, seeded by epoch). Labels are rebuilt as dense $[P,20]$ / $[Q,20,20]$ tensors from the cache's items when the datasets are built; **the cache is not rebuilt** (the whole training cache loads in 17 s).
- The objective is the per-item one on tensors: WT head on the singles, MT head through $\tfrac12(\text{forced WT half}+\text{MT})$ with `--tf_mix`; with `--joint_doubles` other than `off` a pair's WT half is live (its graph is only 2 rows, so `both` costs nothing). `--batch_size`, `--micro_batch_size`, `--rank_list_size`, `--group_by_site` and the training masks do not apply; validation is unchanged. Refused with `--mt_direct_pred`, fused adapters and `--rank_group_by partner`; `native_cond` items are left out.

---

## 5. Nonspecific Epistasis Modeling (`--nonspecific_term fx`)

### 5.1 Additive vs Epistatic Variance in MegaScale
ANOVA variance-partition analyses demonstrate that ~90% of the variance in raw double-mutant non-additivity ($\Delta\Delta\Delta G$) is driven by global assay saturation, per-position-pair offsets, and single-mutant marginal scales. Only ~2--5% of variance represents true residue-specific chemical coupling.

### 5.2 Formulation
The nonspecific term models identity-independent non-additivity as an explicit function of additive expectation:
$$y_{\text{pred}} = y_{\text{comb}} + f(y_{\text{WT}})$$
Where $f(x)$ is parameterized as a smooth, monotonic neural spline or piecewise polynomial. In initial single-seed evaluations, adding $f(x)$ improved cell-level rankings; however, canonical training relies directly on `SaturatingLink` to model saturation non-additivity without introducing auxiliary unidentifiable parameters.
