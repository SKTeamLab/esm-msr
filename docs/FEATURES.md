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

---

## 5. Nonspecific Epistasis Modeling (`--nonspecific_term fx`)

### 5.1 Additive vs Epistatic Variance in MegaScale
ANOVA variance-partition analyses demonstrate that ~90% of the variance in raw double-mutant non-additivity ($\Delta\Delta\Delta G$) is driven by global assay saturation, per-position-pair offsets, and single-mutant marginal scales. Only ~2--5% of variance represents true residue-specific chemical coupling.

### 5.2 Formulation
The nonspecific term models identity-independent non-additivity as an explicit function of additive expectation:
$$y_{\text{pred}} = y_{\text{comb}} + f(y_{\text{WT}})$$
Where $f(x)$ is parameterized as a smooth, monotonic neural spline or piecewise polynomial. In initial single-seed evaluations, adding $f(x)$ improved cell-level rankings; however, canonical training relies directly on `SaturatingLink` to model saturation non-additivity without introducing auxiliary unidentifiable parameters.

---

## 6. Confidence-Interval & Label Variance Weighting (`--ci_weight`)

### 6.1 Assay Disagreement vs Tabulated Intervals
Experimental MegaScale tables report a nominal 95% confidence interval (`deltaG_95CI`). Analysis showed that tabulated intervals correlate weakly with held-out prediction error, whereas the between-protease disagreement ($|\Delta G_{\text{trypsin}} - \Delta G_{\text{chymotrypsin}}|$) provides a reliable measurement error estimate.

### 6.2 Weighting Scheme & Findings
Under `--ci_weight`, items were weighted inversely by estimated label variance $\sigma_i^2$:
$$w_i = \frac{1}{\sigma_i^2 + \tau^2}$$
While mathematically sound for pure regression, weighting efficiency on in-range variants measured only $1.065$ at $\tau=0.5$. In ranking losses (Plackett-Luce), item-level weighting distorted risk-set normalization. Because the observed gain was within replicate noise, explicit variance weighting was permanently retired in favor of robust hinge regression and Plackett-Luce list ranking.

---

## 7. Stacking Rules, Decision Gates & Capacity Ladders

### 7.1 Stacking Wins
The foundational rule of ESM-MSR ablation campaigns is **strict stacking**:
- Any configuration change (e.g. loss formulation, link parameters, capacity) must demonstrate a statistically significant improvement over replicate noise on cell-level metrics (`val_epi_cell_rank_epi`, `val_epi_beyond_rho_epi`).
- Once confirmed, the winning change is adopted into the canonical baseline for all subsequent runs.

### 7.2 Architecture Matching in Resumed / Frozen Checkpoints
A PyTorch LoRA checkpoint cannot load into an adapter of different rank or scaling. `--init_wt_from_checkpoint` strictly validates parameter shapes. When freezing a WT head, the exact hyperparameters (`lora_rank_wt`, `lora_alpha_wt`, `lora_dropout_wt`, `per_mutation_bias`) are preserved.

### 7.3 Capacity Ladder Protocol
To determine minimal required capacity without underfitting, LoRA rank is evaluated via a descending ladder:
$$\text{Rank } 16 \longrightarrow 8 \longrightarrow 4 \longrightarrow 2 \longrightarrow 1$$
Scaling $\alpha = 4 \sqrt{r}$ maintains constant rsLoRA effective scale ($\alpha / \sqrt{r} = 4.0$):
- $r=16, \alpha=16$ (scale 4.0)
- $r=8, \alpha=11$ (scale 3.89)
- $r=4, \alpha=8$ (scale 4.0)
- $r=2, \alpha=4$ (scale 2.83 standard; $r=2, \alpha=6$ scale 4.24)
- $r=1, \alpha=4$ (scale 4.0)

If a lower-capacity arm degrades `val_rho_wt_valid_avg` by $\le 0.002$, the parsimonious lower-capacity model is accepted. This ladder directly motivated the current half-rank ablation arm (WT $r=1, \alpha=4$; MT $r=8, \alpha=11$).
