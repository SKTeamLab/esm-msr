#!/usr/bin/env python3
"""Generate per-panel annotation text files for all main-text figures (Figures 1–4).

Writes one standalone `.txt` caption/annotation file per panel under:
    analysis/results/00_main_text_figs/annotations/figure{1,2,3,4}_panel_{a..e}.txt

Usage:
    PYTHONPATH=src:analysis python -m analysis.main_figures.write_annotations
"""
from __future__ import annotations

import sys
from pathlib import Path

_ANALYSIS_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _ANALYSIS_DIR.parent
for _p in (str(_REPO_ROOT / 'src'), str(_ANALYSIS_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from msr_analysis.paths import atomic_write_text, results_dir


ANNOTATIONS: dict[str, str] = {
    # -------------------------------------------------------------------------
    # Main-Text Figure 1 (a, b, c)
    # -------------------------------------------------------------------------
    'figure1_panel_a.txt': (
        "Figure 1, Panel a — ESM-MSR dual-view architecture, calibration, and supervision\n"
        "================================================================================\n\n"
        "Schematic of the ESM-MSR architecture and training/inference pipeline built on the frozen 1.4B-parameter "
        "ESM3-small-open backbone with wild-type (WT) backbone coordinates encoded via the frozen ESM3 VQ-VAE "
        "structure tokenizer. Two asymmetric LoRA adapters attach to the shared transformer backbone: a low-rank WT "
        "adapter (rank r = 2, alpha = 4; single forward pass per protein with cached L x 20 logits) and a higher-rank "
        "mutant (MT) adapter (rank r = 16, alpha = 16; one forward pass per multi-mutant variant). Site log-odds sums "
        "(x = sum_{i in mut} [l_i(mut) - l_i(wt)]) are mapped to stability changes via per-mutation linear calibration "
        "(s * x + n * b, with s > 0), producing exact additive single-mutant predictions (ddG_WT) and dual-view "
        "ensemble multi-mutant predictions (ddG_AB = 0.5 * (WT + MT)), from which predicted epistasis is "
        "dddG_AB = ddG_AB - (ddG_A + ddG_B). Both views are supervised with assay-floor-aware Tobit NLL + ListMLE "
        "ranking losses using posterior teacher forcing on multi-mutants.\n"
    ),
    'figure1_panel_b.txt': (
        "Figure 1, Panel b — Homology-Restricted MegaScale test set: stabilizing vs destabilizing variants\n"
        "=================================================================================================\n\n"
        "Mean per-domain Spearman's rho across the 33 Homology-Restricted MegaScale test domains (n = 48,740 total "
        "variants), split into stabilizing (ddG > 0; n = 7,417, 33 domains; hatched bars) and destabilizing "
        "(ddG < 0; n = 41,323, 33 domains; solid bars) subsets. Horizontal black ticks denote overall per-domain mean "
        "Spearman's rho across all test variants (n = 48,740). Models are grouped into Zero-shot (ProSST, ProteinMPNN, "
        "ESM3-small-open), Retrained on Homology-Restricted MegaScale (ESM-MSR, MutateEverything, ThermoMPNN, SPURS), "
        "and Supervised, not trained on MegaScale (GeoDDG-3D, Rosetta). Retrained transfer models average 3 training "
        "seeds; error bars show 95% paired domain x seed bootstrap confidence intervals. Singles models (WT-view / "
        "Stage 1 / single) score single mutants and doubles-trained models (Dual-view / Stage 2 / -D / -multi) score "
        "double mutants.\n"
    ),
    'figure1_panel_c.txt': (
        "Figure 1, Panel c — Homology-Restricted MegaScale double mutants: additive vs epistatic scoring\n"
        "===============================================================================================\n\n"
        "Mean per-domain Spearman's rho on double mutants from the Homology-Restricted MegaScale test set "
        "(n = 13,319 double mutants across 13 test domains), comparing additive scoring (sum of single-mutant model "
        "predictions, hatched bars) against epistatic scoring (direct double-mutant model prediction, solid bars). "
        "ProSST is additive-only. Retrained models average 3 training seeds; error bars show 95% domain x seed "
        "bootstrap confidence intervals. Brackets report paired one-sided domain x seed bootstrap p-values "
        "(B = 10,000) testing H1: rho_epistatic > rho_additive.\n"
    ),

    # -------------------------------------------------------------------------
    # Main-Text Figure 2 (a, b, c, d, e)
    # -------------------------------------------------------------------------
    'figure2_panel_a.txt': (
        "Figure 2, Panel a — Thermodynamic-cycle schematic of pairwise epistasis types\n"
        "=============================================================================\n\n"
        "Thermodynamic-cycle schematic illustrating the six pairwise epistasis regimes "
        "(dddG = ddG_AB - ddG_A - ddG_B). Top row: weak positive (A), additive (B), and strong negative / clash (C) "
        "cycles. Bottom row: strong positive / compensation (D), additive (E), and weak negative (F) cycles.\n"
    ),
    'figure2_panel_b.txt': (
        "Figure 2, Panel b — Assay unfolded-state floor induces apparent negative epistasis\n"
        "==================================================================================\n\n"
        "Density scatter of measured pairwise epistasis (dddG, kcal/mol) versus the label additive stability "
        "expectation (dG_add = dG_wt + ddG_A + ddG_B, kcal/mol) for Homology-Restricted MegaScale test doubles with "
        "both constituent singles measured (n = 12,455 doubles across 13 test domains). The unfolded-state floor "
        "(y = -x, dG_AB = 0; dashed red) and MegaScale reporting cutoff (y = -x - 1, dG_AB = -1; dotted red) induce a "
        "strong negative correlation (Spearman rho = -0.64).\n"
    ),
    'figure2_panel_c.txt': (
        "Figure 2, Panel c — Distribution of measured epistasis types across additive stability bins\n"
        "===========================================================================================\n\n"
        "Top: share of measured pairwise epistasis types across bins of label additive stability expectation "
        "dG_add = dG_wt + ddG_A + ddG_B (kcal/mol; n = 12,455 doubles). Hatched segments indicate floor-dominated "
        "doubles (36% overall), where the unfolded-state assay floor determines the observed epistasis. Bottom: "
        "log-scale variant counts per bin split by the sign of the summed single-mutant effects "
        "(singles' sum ddG < 0 vs > 0).\n"
    ),
    'figure2_panel_d.txt': (
        "Figure 2, Panel d — Five-level epistasis ANOVA and residual decomposition schematic\n"
        "===================================================================================\n\n"
        "Schematic of the five-level epistasis residual decomposition aligned column-for-column above Panel e: "
        "(1) naive / raw dddG, (2) saturation-subtracted dddG removing the global dG_add floor trend, (3) pair-level "
        "mean epistasis across position-pair matrices, (4) line-level pair-centred row/column marginals, and "
        "(5) cell-level double-centred site-specific interaction residuals.\n"
    ),
    'figure2_panel_e.txt': (
        "Figure 2, Panel e — Five-level pooled epistasis residual hierarchy benchmarks\n"
        "=============================================================================\n\n"
        "Pooled Spearman's rho across all five epistasis residual hierarchy levels on the Homology-Restricted "
        "MegaScale test set (13 domains; B = 1,000 domain x seed bootstrap; 3 training seeds averaged for retrained "
        "models). Levels are evaluated in pooled form: naive / raw rho, saturation-subtracted rho, pair rho across "
        "position-pair matrices with >= 20 doubles, line rho across pair-centred row/column lines with >= 3 cells, "
        "and cell rho pooled over all double-centred residual cells. Black overlay bars evaluate the unsaturated "
        "subset (dG_add > 0); the red dotted line marks the -dG_add floor ceiling. Includes the ESM-MSR context "
        "control (ESM-MSR ctx ctrl).\n"
    ),

    # -------------------------------------------------------------------------
    # Main-Text Figure 3 (a, b, c, d)
    # -------------------------------------------------------------------------
    'figure3_panel_a.txt': (
        "Figure 3, Panel a — External thermostability benchmarks\n"
        "=======================================================\n\n"
        "Pooled Spearman's rho across 8 external thermostability datasets (Ssym, S461, K2369, S783, S2648, Q3421, "
        "S8754, and PTMUL), ordered from highest to lowest ESM-MSR performance (mean of seeds 1-3). Whiskers show "
        "95% mutation-level bootstrap intervals (seeds resampled jointly with bootstrap draws for triplicate retrained "
        "models). Singles models score the single-mutant benchmarks (hollow markers: ESM-MSR WT-view, "
        "MutateEverything Stage 1, ThermoMPNN, SPURS single); doubles-trained models score PTMUL (filled markers: "
        "ESM-MSR Dual-view, MutateEverything Stage 2, SPURS-multi; ThermoMPNN additive; Rosetta Cartesian ddG scaled "
        "by 1/2.94 throughout). SPURS uses the retrained model with trained adapter. Red open rings mark documented "
        "training overlap (GeoDDG-3D, ESM3dG; leak audit), with text indicating the most severe overlap tier and "
        "affected share (M = exact mutations, m = constituent singles, P = same protein, H = close homolog).\n"
        "NOTE (DRAFT): Currently rendered with pre-update ESM3-small-open (sigma = 0) and FINAL_p0 predictions; "
        "update once per_res_plddt = 1.0 zero-shot and 3-seed retrained runs complete.\n"
    ),
    'figure3_panel_b.txt': (
        "Figure 3, Panel b — Cross-benchmark generalization (ProteinGym, Domainome, MGnify, Rocklin Designs)\n"
        "====================================================================================================\n\n"
        "Spearman's rho across held-out ProteinGym average (173 held-out assays), held-out Domainome average "
        "(436 held-out domains), and pooled inferred-ddG benchmarks (MGnify and Rocklin Designs). Held-out excludes "
        "every target homologous to the Homology-Restricted MegaScale training libraries under the four-part "
        "sequence + structure homology filter (any of: MMseqs2 E-value <= 1e-3, BLOSUM62 identity > 0.40 over the "
        "shorter sequence, Foldseek E-value <= 1e-3, or Foldseek TM-score > 0.70; excluding 40 homologous ProteinGym "
        "assays, 86 Domainome domains, and homologous MGnify/Rocklin scaffolds). Seeds 1-3 are averaged for retrained "
        "models. Singles versions score singles assays and Domainome (hollow markers on Domainome for two-version "
        "models); doubles/multi-mutant versions score ProteinGym multi-mutant assays and MGnify/Rocklin sets "
        "(ESM-MSR Dual-view combined head, SPURS-multi, ThermoMPNN-D, MutateEverything Stage 2; ProteinMPNN and "
        "ProSST direct; Rosetta covers ProteinGym singles only, not Domainome). ESM-MSR sigma = 1, sigma = 0.5, "
        "and ESM3-small-open (sigma = 0, adapters off) use deployed readouts (combined on multi-mutant assays, WT "
        "head on singles). On MGnify and Rocklin Designs, one pooled Spearman rho is computed across all measured "
        "variants (whiskers resample scaffolds x seeds), reading ESM-MSR on the observed Tobit-linked scale "
        "h(dG_wt + latent) - dG_wt and other models raw. Red open rings indicate training overlap (ESM3dG was "
        "fine-tuned on MGnify stability data; 'train src').\n"
        "NOTE (DRAFT): Currently rendered with pre-update ESM3-small-open (sigma = 0) and FINAL_p0 predictions; "
        "update once per_res_plddt = 1.0 zero-shot and 3-seed retrained runs complete.\n"
    ),
    'figure3_panel_c.txt': (
        "Figure 3, Panel c — ProteinGym per-assay delta-rho by functional property across sigma = 0.5 and 1.0\n"
        "=====================================================================================================\n\n"
        "Per-assay Spearman's rho change (delta-rho = rho(sigma) - rho(sigma = 0)) across the five ProteinGym "
        "coarse functional categories (Activity, Binding, Expression, Organismal fitness, Stability) comparing "
        "ESM-MSR at sigma = 0.5 and sigma = 1.0 against ESM3-small-open (sigma = 0, adapters off), using the "
        "deployed readout (Dual-view combined head on multi-mutant assays, WT head on single-mutant assays). Each "
        "dot represents one assay; thin grey 'x' markers show assays homologous to the training set (plotted for "
        "completeness but excluded from boxes, medians, means, header counts, and p-values). Boxes show the "
        "interquartile range (IQR), thick horizontal lines show medians (with numerical values), and thin horizontal "
        "lines show means. P-values below the x-ticks are two-sided joint assay x seed bootstrap tests of "
        "H0: mean delta-rho = 0, Holm-adjusted across the 10 category x sigma comparisons.\n"
        "NOTE (DRAFT): Currently rendered with pre-update ESM3-small-open (sigma = 0) and FINAL_p0 predictions; "
        "update once per_res_plddt = 1.0 zero-shot and 3-seed retrained runs complete.\n"
    ),
    'figure3_panel_d.txt': (
        "Figure 3, Panel d — ProteinGym multi-property Pareto frontier (Biophysical vs Functional, 176 unseen assays)\n"
        "============================================================================================================\n\n"
        "Biophysical macro-average Spearman's rho (Stability [n = 37] and Expression [n = 18]) versus functional "
        "macro-average Spearman's rho (Activity [n = 38], Binding [n = 13], and Organismal fitness [n = 70]) across "
        "all 105 evaluated models on the 176 unseen ProteinGym assays (41 of 217 assays excluded under the four-part "
        "sequence + structure homology filter; no large-protein drop). Public ProteinGym zero-shot models (n = 95) "
        "are colored by inference-time modality: Sequence-only (46; top method annotated: VespaG), Structure + "
        "Sequence without MSA (26; top method annotated: ProSST-2048), and MSA / Retrieval (23; VenusREM and S3F_MSA "
        "annotated). All 7 stability baselines with full ProteinGym coverage are explicitly labeled directly on the "
        "plot using their canonical manuscript colors and markers (and omitted from the legend): SPURS (176/176), "
        "MutateEverything (174/176), ThermoMPNN (176/176), GeoDDG-3D (174/176), Rosetta (174/176), ESM3dG (173/176), "
        "and ESMtherm* (173/176); red open rings denote documented training overlap (ESM3dG, ESMtherm*). ESM-MSR "
        "variants are plotted in canonical manuscript greyscale connected by the dotted sigma trajectory "
        "(ESM3-small-open [sigma = 0] -> ESM-MSR sigma = 0.5 -> ESM-MSR sigma = 1.0), with 95% joint assay x "
        "triplicate-seed bootstrap whiskers (B = 10,000). Across all 105 models, only three are Pareto non-dominated "
        "(black staircase): ESM-MSR sigma = 1.0 (Pareto bootstrap probability 100.0%), ESM-MSR sigma = 0.5 (99.9%), "
        "and VenusREM (MSA-based).\n"
        "NOTE (DRAFT): Watermarked 'OLD DATA' as a reminder that ESM3-small-open (sigma = 0) and corresponding "
        "3-seed ESM-MSR predictions must be updated with per_res_plddt = 1.0 before publication.\n"
    ),

    # -------------------------------------------------------------------------
    # Main-Text Figure 4 (a, b, c, d)
    # -------------------------------------------------------------------------
    'figure4_panel_a.txt': (
        "Figure 4, Panel a — Per-property rectified stability gain at k = 96 (ASA x hydrophobicity)\n"
        "=========================================================================================\n\n"
        "Rectified stability gain at top k = 96 single-mutant picks per Homology-Restricted MegaScale test domain "
        "(33 domains with structural/hydrophobicity features; 35,295 single mutants), stratified into four quadrants "
        "by hydrophobicity change (delta-hydrophobicity <= 0 vs > 0) and relative solvent-accessible surface area "
        "(relative ASA <= 0.5 vs > 0.5). Rectified gain is defined as sum max(ddG, 0) over every pick regardless of "
        "predicted sign. Boxes show the interquartile range with darker colored median lines; dots show individual "
        "domains; black horizontal bars with vertical 95% bootstrap CI whiskers (B = 1,000) and numerical labels "
        "show means over domains; dashed horizontal lines mark oracle means. The bottom strip ('Picks (%)') shows "
        "the percentage of each model's top-96 picks falling into each biophysical quadrant.\n"
    ),
    'figure4_panel_b.txt': (
        "Figure 4, Panel b — Homology-Restricted MegaScale single-mutant screening curves\n"
        "=================================================================================\n\n"
        "Single-mutant screening efficiency on the Homology-Restricted MegaScale test set excluding mutations to "
        "cysteine (33 proteins; single-mutant models only), evaluated across screening budgets K from 1 to 100 "
        "variants per protein. Top: stabilizing hit rate (fraction of top-K picks with measured ddG > 0.5 kcal/mol). "
        "Bottom: P(at least one stabilizing hit in top K). Shaded grey band indicates the 95% bootstrap confidence "
        "interval (B = 1,000) for ESM-MSR WT-view only; dotted black and dashed red lines indicate oracle and random "
        "selection.\n"
    ),
    'figure4_panel_c.txt': (
        "Figure 4, Panel c — Held-out Domainome single-mutant screening curves\n"
        "======================================================================\n\n"
        "Single-mutant screening efficiency on held-out Domainome proteins (436 proteins; single-mutant models only), "
        "evaluated across screening budgets K from 1 to 100 variants per protein. Top: hit rate (fraction of top-K "
        "picks with measured abundance greater than wild type). Bottom: P(at least one hit in top K). Shaded grey "
        "band indicates the 95% bootstrap confidence interval (B = 1,000) for ESM-MSR WT-view only; dotted black and "
        "dashed red lines indicate oracle and random selection.\n"
    ),
    'figure4_panel_d.txt': (
        "Figure 4, Panel d — ProteinGym Activity preservation during stability-guided screening\n"
        "=======================================================================================\n\n"
        "Functional-damage curves on held-out ProteinGym Activity single-mutant assays (top: all 38 held-out "
        "Activity assays; bottom: 16 held-out enzyme assays; single-mutant models only), evaluated across screening "
        "budgets K from 1 to 100 variants per assay. Vertical axis shows the fraction of the top-K stability-ranked "
        "picks that hurt activity (defined as measured activity more than 0.5 assay standard deviations below the "
        "measured wild type; lower is better). Shaded grey band indicates the 95% bootstrap confidence interval "
        "(B = 1,000) for ESM-MSR WT-view only; dashed red line indicates random selection.\n"
    ),
}


def write_all_annotations() -> Path:
    ann_dir = results_dir('main_text_figs') / 'annotations'
    ann_dir.mkdir(parents=True, exist_ok=True)
    for fname, text in ANNOTATIONS.items():
        atomic_write_text(ann_dir / fname, text)
    return ann_dir


def main() -> int:
    ann_dir = write_all_annotations()
    print(f'[main_figures:write_annotations] Wrote {len(ANNOTATIONS)} panel annotation files -> {ann_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
