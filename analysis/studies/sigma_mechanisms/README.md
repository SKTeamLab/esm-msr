# Mechanistic Decomposition of Thermodynamic LoRA Scaling (`σ ∈ {0, 0.5, 1}`)

Replaces the Oct-27 single-seed baseline exploration (`proteingym/sigma_tradeoff/`) with a reproducible, 3-seed `FINAL_p0` study across the **173 canonical held-out ProteinGym assays** (`Activity=38, Binding=13, Expression=18, OrganismalFitness=67, Stability=37`).

## Four Mechanistic Results (`sigma_mechanisms_composite.{png,pdf}`)
1. **Between-Site vs Within-Site Ranking (`site_vs_sub_decomposition.csv`, Panel a)**:
   - Turning on the stability LoRA (`σ = 0 -> 1`) strongly improves **between-site sensitivity ranking** across all 5 categories (identifying buried/structural scaffold positions intolerant to mutation), while leaving **within-site substitution preferences** nearly flat on non-stability functional categories.
2. **20-Amino-Acid Rank Shift vs Empirical Category Preferences (`aa_rank_shifts.csv`, `aa_category_heatmaps.{png,pdf}`, Panel b)**:
   - `σ = 1` promotes bulky/aliphatic hydrophobic residues (`W, C, F, I, L, V, Y, M`) and demotes polar/charged residues, while leaving Proline (`P`) uniquely unpenalized — matching empirical `Stability`/`Expression` tolerances.
3. **Bottom-10% Damage-Filtering vs Top-10% Forward Selection Decoupling (`tail_selection_vs_filtering.csv`, Panel c)**:
   - Resolves why bulk Spearman `ρ` rises across `Activity`, `Binding`, and `OrganismalFitness`: `σ > 0` substantially improves **rejection of misfolded/destabilized bottom-10% variants** across all categories, whereas top-10% forward selection gains are concentrated in `Stability` and `Expression`.
4. **Multi-Property Pareto Trajectories (`pareto_multi_property.csv`, Panel d)**:
   - Across 13 held-out proteins with paired abundance/expression and activity/binding assays (`KRAS`, `PTEN`, `KCNE1`, `Kir2.1`, `CYP2C9`, `GCK`, `OCT1`, `VKOR1`, `DAAO`, `HLA-A*02`, `Spike RBD`, `PSD95-PDZ3`, `GRB2-SH3`), intermediate scaling **`σ = 0.5` captures >80% of the `σ = 1.0` abundance gain with near-zero functional trade-off**.

## Usage
```bash
PYTHONPATH=src:analysis python -m studies.sigma_mechanisms.run --B 1000
```
