# Triplicate Seed & Joint Resampling Audit (`analysis/docs/SEED_AUDIT.md`)

## Canonical Rule

Every retrained model evaluated in `analysis/` is trained in triplicate (`seeds = (1, 2, 3)`):
- **ESM-MSR** (`FINAL_p0`, key `msrp0`) and all **`FINAL_*` ablation families** (`p1`, `p1_marg`, `pw0`, `norank`, `rcthr0`, `norange`, `mse`, `noseqhwm`, `r2a4`, `mega`),
- **MutateEverything** (`stage 1` singles `me1`, `stage 2` doubles `me2`),
- **ThermoMPNN** (`tmpnn`) and **ThermoMPNN-D** (`tmpnnd`),
- **SPURS** (`spurs`) and **SPURS-multi** (`spursm`, trained-adapter configuration),
- **ProteinMPNN** (our 3 independent decoding-order runs on ProteinGym and Activity screening).

Across every pipeline (`analysis/pipelines/p01..p12`) and mechanistic study (`analysis/studies/*`):
1. **Point estimates**: Compute the target metric independently for each seed $s \in \{1, \dots, n_s\}$, then average over seeds. (Scatter and density plots explicitly plot the seed-averaged per-variant prediction and label it as such.)
2. **Across-seed dispersion**: Always use the unbiased sample standard deviation (`ddof=1`) via `msr_analysis.stats.seed_sd()`.
3. **Joint `[cluster × seed]` bootstrap resampling (`B = 1000` default)**: Every bootstrap draw resamples **both** the biological sampling unit (protein, MegaScale domain, residue site, or category-stratified DMS assay) **and** the training seeds via multinomial seed weights shared across paired comparators, capturing biological sampling variance and training-seed variance simultaneously without double-counting.

---

## Single-Run & Partial-Seed Exceptions (Data Boundaries Only)

| Model / Run | Seeds Available | Reason |
| :--- | :---: | :--- |
| `FINAL_p1` combined (`comb`) on ProteinGym | `2` (`s1`, `s3`) | `s2` finished 126 of 219 assays before queue cutoff; `msr_analysis.boot` and `msr_analysis.stats.make_domain_seed_draws` draw exact 2-seed multinomial weights `multinomial(2, [0.5, 0.5])`. |
| `ESM3-small-open` (`msrrel@0`), `ProSST`, `GeoDDG-3D`, `Rosetta`, `ESM3ΔG`, `ESMtherm`, `IFUM`, `DDGemb`, `DDMut`, published `SPURS` | `1` | Deterministic zero-shot evaluations or single published third-party checkpoints. |
| Fixed-slot hardware profiling (`scoring/resources.py` -> `10_inference_resources/`) | `1` per model class | Measures wall-clock GPU hours and peak RSS/VRAM on a fixed RTX A4500 slot (replicates agree within ~1%). |

---

## Phase 2 Statistical & Bootstrap Unification (Nov 19 Refactor)

During the `analysis/` architecture refactor, five subtle seed-handling bugs uncovered by the Phase 1 audit were eliminated in `msr_analysis/`:

| Issue Found in Pre-Refactor Scripts | Fix in Refactored `msr_analysis/` & Pipelines |
| :--- | :--- |
| **`msr_analysis/boot.py` hardcoded 3-seed multinomial (`11.1% NaN` bug)**: `bootstrap_rho` drew `multinomial(3, [1/3, 1/3, 1/3])` and sliced `[:len(rs)]`. For 2-seed arms (`FINAL_p1:comb`), `[0, 0, 3]` occurred on $1/9 = 11.1\%$ of draws, yielding zero weight `[0, 0]` (`NaN`) and distorting variance on the remaining draws; for $>3$-member ensembles, seed resampling silently turned off. | `msr_analysis/boot.py::bootstrap_rho` and `msr_analysis/stats.py::make_domain_seed_draws` draw `rng.multinomial(ns, np.full(ns, 1.0 / ns))` keyed by each model's actual seed count `ns` (`1, 2, 3, ...`), verified by unit tests in `analysis/tests/test_boot.py`. |
| **Pre-bootstrap seed averaging in sub-studies & ablation tables**: `98_final_ablation_table.py` (`pg_boot`), `ablation_vs_p0/02`, `combined_vs_wt/04`, `proteingym/p0_sigma/{02,03}`, `pg_activity/02`, `11j` (pooled $\rho$), `17_gain_per_variant_fit`, `faure_epistasis/05`, and `mega_splits/03` averaged seeds *before* resampling. In `98`, this shrank `se_boot` on `FINAL_p0_r2a4` to `0.00020` and falsely flagged pure WT seed drift (`-0.001`) as significant. | Replaced everywhere by `proteingym.boot_pgym_categories()`, `stats.boot_domain_seeds()`, `epistasis.boot_epistasis_models()`, `screening.per_protein_curve()`, and `SpearmanEngine` seed-weighted replicates across `p02`, `p03`, `p04`, `p08`, `p09`, `p11`, and `studies/*`. |
| **Population SD (`ddof=0`) in MGnify scripts**: `scoring/mgnify_{baselines_compare, epistasis_beyond, rdenovo_stats}.py` used NumPy default `np.std(..., ddof=0)`, underestimating 3-seed SD by $18.4\%$ ($\sqrt{2/3} \approx 0.8165$). | Folded into `pipelines/p04_mgnify_rocklin.py` (`05_mgnify/`) using `msr_analysis.stats.seed_sd()` (`ddof=1`) throughout. |
| **Single-seed Oct-27 baseline in `proteingym/sigma_tradeoff/`**: Mechanistic $\sigma$-scaling plots used a single-seed Oct-27 baseline run over 214 unfiltered assays. | Replaced by `analysis/studies/sigma_mechanisms/` (`99_misc/sigma_mechanisms/`), running on **3-seed `FINAL_p0`** (`deployed` readout) across the **173 canonical held-out ProteinGym assays**. |
| **Draft `B = 100` CSVs & silent `B // 2` halving**: `01_core_rho.py` halved `B` on mutation bootstraps (`B // 2`), and committed CSVs in `ablation_vs_p0`, `combined_vs_wt`, and `faure_epistasis` were saved at `B = 100` (`min p = 0.0099`) with hardcoded `"100 resamples"` captions. | Unified `--B 1000` default across `analysis/run.py` and all 12 pipelines + 2 studies, with dynamic `B` interpolation in all figure footnotes and $p \ge 1/(B+1)$ floors via `stats.boot_pvalue()`. |

---

## Historical Seed-1 $\rightarrow$ Seeds 1–3 Audit Log (Nov 18)

| Component (Canonical Output Path) | Pre-Audit State | Triplicate State | Quantitative Impact |
| :--- | :--- | :--- | :--- |
| `msr_analysis/data.py` (`msrp0:*`) & `external.py` (`seed_frame`) | Loaded `FINAL_p0` seed 1 only; `_seed1_frame` sliced `preds[k][:1]` for ME, ThermoMPNN, and SPURS | Loads `FINAL_p0` seeds 1–3 across all 22 `final_bench/predictions` folders; `load_mutation_boot` resamples all 3 seeds | Thermostability sets move by $\le 0.006$ for ESM-MSR (`Ssym` $0.783 \to 0.778$, `PTMUL` $0.586 \to 0.592$); ME stage 2 on `PTMUL`/`PTMUL-D` $-0.010$ to $-0.027$ (seed 1 was high). |
| `p03_proteingym.py` (`03_proteingym/`, $\sigma$-sweep tables) | Seed 1 on `FINAL_p0`, ME, ThermoMPNN, SPURS, and $\sigma=0.5$ Domainome; ThermoMPNN-D was 1 seed | 3 seeds for all retrained models (including `final_pgym_comb_sigma0.5` s1–s3 and gpu4 ThermoMPNN-D s1–s3) + our 3 ProteinMPNN runs | ProteinGym average moves $\le 0.003$; Domainome SPURS $+0.016$, ThermoMPNN $-0.006$, ESM-MSR $-0.003$; Stability $\sigma=1$ median $\Delta\rho$: $+0.184 \to +0.191$. |
| `p09_screening.py` (`01_megascale/{screening,gain_fit}`, `02_thermostability/screening`, `03_proteingym/screening`, `04_domainome/screening`, `07_screening_overview`) | Screening curves & Activity screening used seed 1 for `FINAL_p0` (and seed 1 in logarithmic gain fit) | Per-protein screening tables computed per seed, averaged, and jointly resampled (`seeds 1–3`) | Curves smooth single-seed step noise; `g(K) = a - b \ln K` fit CIs reflect full domain + seed uncertainty. |
| `p11_ablations_and_heads.py` (`09_ablation/`, `99_misc/dual_view/`) | Paired deltas at seed 1 (`delta_s1.csv`) or seed-averaged before resampling | Joint `[unit × seed]` paired bootstrap across all 11 `FINAL_*` families (`delta_paired.csv`) | Median 95% CI widens (`0.020 -> 0.023`), filtering out single-seed false positives (`rank censor 0` on Stability no longer significant, Holm $p = 0.16$). |
| `p02_tsuboyama.py` & `p06_epistasis_hierarchy.py` (`01_megascale/`, `01_megascale/epistasis/`) | Scatter plots indexed `preds[k][0]` | Scatters plot seed-averaged predictions; `boot_epistasis_models` resamples domains $\times$ 3 seeds | Consistent triplicate treatment across all 22 epistasis metrics and validation/test hierarchies. |
