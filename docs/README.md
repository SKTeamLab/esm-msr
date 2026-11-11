# Documentation Index

Status key:
- **CURRENT**: Authoritative, active documentation reflecting the current repository state and conventions.
- **REFERENCE**: Deep-dive technical specifications and resource profiling.
- **ANALYSIS**: Empirical measurements, ANOVA variance decompositions, and statistical proofs.

---

## 1. Primary Operational Documentation

| Document | Status | Description |
| :--- | :--- | :--- |
| [`STATE.md`](STATE.md) | **CURRENT** | Active workspace state, queue runner status, running and queued arms on GPU 0, pitfalls, and design facts. **Read first upon context reset.** |
| [`codebase_spec_and_audit.md`](codebase_spec_and_audit.md) | **CURRENT** | Full architectural and infrastructure specification, prioritized bug resolutions/mitigations, efficiency shortcuts in training and inference, and active modeling directions. |
| [`FEATURES.md`](FEATURES.md) | **CURRENT** | Comprehensive consolidated guide to model features: dual-adapter design, readout links, two-source censoring, frozen-WT paradigm, and de-linked teacher forcing (`tf_mix`). |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | **CURRENT** | Chronological experiment history, measured metric progressions across lineages, triage records, and open questions. |
| [`OUTSTANDING_EXPERIMENTS.md`](OUTSTANDING_EXPERIMENTS.md) | **CURRENT** | Tracker for in-flight, queued, and blocked benchmark evaluations and ablations. |

---

## 2. Benchmarks, Metrics & Audits

| Document | Status | Description |
| :--- | :--- | :--- |
| [`BENCHMARKS_OCT2026.md`](BENCHMARKS_OCT2026.md) | **CURRENT** | Canonical evaluation benchmark comparing ESM-MSR against baseline and third-party models across 12 external $\Delta\Delta G$ sets, Tsuboyama, 7 DMS sets, Domainome, and ProteinGym (213 assays), with full training-data leak annotations. |
| [`LEAKAGE_AUDIT_OCT2026.md`](LEAKAGE_AUDIT_OCT2026.md) | **CURRENT** | Systematic audit of sequence and structural overlap between external benchmark sets and training data across all evaluated models (using retrained Homology-Restricted MegaScale inventories for SPURS and ThermoMPNN-D). |
| [`SPLITTING_AND_HOMOLOGY_AUDIT.md`](SPLITTING_AND_HOMOLOGY_AUDIT.md) | **CURRENT** | Technical audit of the `splits_oct06_capped` train/val/test splitting procedure and the cross-set homology/leakage pipeline: reproducibility, robustness, quarantine design, resolved pipeline bugs, and assumptions. |
| [`validation_metrics.md`](validation_metrics.md) | **CURRENT** | Formal definitions of all validation and epistasis metrics, covering the authoritative $\sim\text{MT}(AB) - \text{WT}(AB)$ formulation and the `_epi` metric hierarchy. |
| [`INFERENCE_RESOURCES.md`](INFERENCE_RESOURCES.md) | **REFERENCE** | Fixed-slot runtime, CPU, RAM, and VRAM profiling across models and datasets. |

---

## 3. Technical Reference & Operational Guides

| Document | Status | Description |
| :--- | :--- | :--- |
| [`known_issues.md`](known_issues.md) | **REFERENCE** | Practical workarounds for long structures, WSL driver memory behavior, and allocator configurations. |
| [`vram_and_batch_sizes.md`](vram_and_batch_sizes.md) | **REFERENCE** | Quadratic VRAM model and batch size bounds on RTX 5090 (32 GB). |
| [`trim_audit.md`](trim_audit.md) | **REFERENCE** | Census of the 160 legacy flags, run evidence, and rationale behind the consolidation into modern YAML recipes. |

---

## 4. Empirical & Statistical Analyses

| Document | Status | Description |
| :--- | :--- | :--- |
| [`anova_epistasis_report.md`](anova_epistasis_report.md) | **ANALYSIS** | ANOVA variance partition of $\Delta\Delta\Delta G$ across 127k MegaScale double mutants: global saturation vs pair offsets vs residue couplings. |
| [`conditional_ordering_test.md`](conditional_ordering_test.md) | **ANALYSIS** | Non-parametric test for identity-dependent interaction epistasis across near-complete 20×20 double-mutant matrices. |
| [`epistasis_reliability_audit.md`](epistasis_reliability_audit.md) | **ANALYSIS** | Analysis of measurement noise, $K_{50}$ gap closure, and systematic floor artifacts in derived coupling values. |

---

## Conventions & Standards

* **Metric Naming**: All interaction metrics are authoritatively named `*_epi` (e.g. `val_epi_cell_rank_epi`, `val_epi_beyond_rho_epi`). Legacy `*_mt` naming is retired.
* **Epistasis Definition**: Defined as $\sim\text{MT}(AB) - \text{WT}(AB)$ (the MT head evaluated relative to the additive WT baseline).
* **Execution Reproducibility**: Training runs operate from detached, locked checkouts in `dc_runs/pin_*` and log their tree ref and commit on startup.
