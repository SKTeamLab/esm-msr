# ProteinGym Multi-Property Pareto Frontier Audit (`pgym_pareto`)

Tests the claim that **ESM-MSR at both $\sigma = 1.0$ and $\sigma = 0.5$ sits on the Pareto frontier for multi-property prediction in ProteinGym**, comparing strictly on the unified set of **176 unseen assays** (`217 - 41` homologous; zero `PG_DROP` filtering) against **all 95 public ProteinGym zero-shot models** and all **7 stability-supervised baselines** (`SPURS`, `ThermoMPNN`, `MutateEverything`, `GeoDDG-3D`, `Rosetta`, `ESM3ΔG`, `ESMtherm`; no `SPURS published*`).

## Holdout Paradigm
Applies the repository's canonical four-part sequence + structure homology filter (`analysis/reference/tsuboyama_rule_flags.csv`):
- MMseqs2 $E \le 10^{-3}$
- BLOSUM62 shorter-chain local identity $> 0.40$
- Foldseek $E \le 10^{-3}$
- Foldseek $\text{TM}_{\max} > 0.70$

against the 172 capped MegaScale training libraries, leaving **176 unseen ProteinGym assays** (`Stability=37, Expression=18, Activity=38, Binding=13, OrganismalFitness=70`; 41 homologous assays excluded).

## Usage
```bash
PYTHONPATH=src:analysis /home/sareeves/miniconda3/envs/msr_venv/bin/python -m studies.pgym_pareto.run --B 10000
```

## Outputs (`analysis/results/99_misc/pgym_pareto/`)
- `pgym_pareto_main.{png,pdf}`, `pgym_pareto_pairwise.{png,pdf}`, `pgym_pareto_caveats.{png,pdf}`
- `pgym_176_all98_summary.csv`, `pgym_176_all105_summary.csv`, `pgym_176_bootstrap_cis.csv`, `pgym_176_pareto_bootstrap_probs.csv`, `pgym_176_paired_tests.csv`, `pgym_pareto_stress_tests.csv`
