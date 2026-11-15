"""Shared paths, annotations and helpers for the sigma trade-off analysis.

Everything is driven by ProteinGym/DMS_substitutions.csv and by whatever per-assay CSVs
exist in the sigma*_wt_only folders, so re-running after the missing (OOM) assays are
added needs no code change.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("MSR_ROOT", "/home/sareeves/playground/esm-msr-devel")
# inputs: the per-assay wt_only predictions of the baseline model at sigma 0 .. 1 (the cluster results of Oct 27, kept in the snapshot below)
BASE = f"{ROOT}/msr_predictions/_not_in_benchmarks/results_cluster_snapshot_2026-10-27/pgym/esm-msr-baseline-w2a4-m16a16-capped"
OUT = os.path.dirname(HERE)                      # this folder (analysis/proteingym/sigma_tradeoff): figures/ and tables/ are written here
FIG, TAB = f"{OUT}/figures", f"{OUT}/tables"
REF = f"{ROOT}/ProteinGym/DMS_substitutions.csv"
SIGMAS = [0.0, 0.25, 0.5, 0.75, 1.0]
SCOL = {s: f"s{s}" for s in SIGMAS}
SCRATCH = os.environ.get("SIGMA_SCRATCH", "/tmp/sigma_scratch")

# ---- property annotation for proteins with >1 assay -----------------------------------------
# axis: ABUND (abundance / surface expression / folding readout), ACT (enzymatic or channel
# function), BIND (binding to a partner), FIT (growth / organismal fitness), COND (same property
# measured under a different condition/cell line), SPEC (assay-specific selection, see note).
# group: A = different properties, B = same property in different conditions, C = replicates of one property.
ANNOT = {
 # ---- group A: different properties
 "CP2C9_HUMAN_Amorosi_2021_abundance": ("CP2C9_HUMAN", "A", "ABUND", "VAMP-seq abundance"),
 "CP2C9_HUMAN_Amorosi_2021_activity": ("CP2C9_HUMAN", "A", "ACT", "probe-binding activity"),
 "HXK4_HUMAN_Gersing_2023_abundance": ("HXK4_HUMAN", "A", "ABUND", "DHFR-PCA abundance"),
 "HXK4_HUMAN_Gersing_2022_activity": ("HXK4_HUMAN", "A", "ACT", "complementation activity"),
 "KCNE1_HUMAN_Muhammad_2023_expression": ("KCNE1_HUMAN", "A", "ABUND", "surface expression"),
 "KCNE1_HUMAN_Muhammad_2023_function": ("KCNE1_HUMAN", "A", "ACT", "channel function"),
 "KCNJ2_MOUSE_Coyote-Maestas_2022_surface": ("KCNJ2_MOUSE", "A", "ABUND", "surface trafficking"),
 "KCNJ2_MOUSE_Coyote-Maestas_2022_function": ("KCNJ2_MOUSE", "A", "ACT", "ion conduction"),
 "OXDA_RHOTO_Vanella_2023_expression": ("OXDA_RHOTO", "A", "ABUND", "surface expression"),
 "OXDA_RHOTO_Vanella_2023_activity": ("OXDA_RHOTO", "A", "ACT", "enzyme activity"),
 "PTEN_HUMAN_Matreyek_2021": ("PTEN_HUMAN", "A", "ABUND", "VAMP-seq abundance"),
 "PTEN_HUMAN_Mighell_2018": ("PTEN_HUMAN", "A", "ACT", "lipid phosphatase (yeast growth)"),
 "Q53Z42_HUMAN_McShan_2019_expression": ("Q53Z42_HUMAN", "A", "ABUND", "surface expression"),
 "Q53Z42_HUMAN_McShan_2019_binding-TAPBPR": ("Q53Z42_HUMAN", "A", "BIND", "TAPBPR binding"),
 "RASK_HUMAN_Weng_2022_abundance": ("RASK_HUMAN", "A", "ABUND", "AbundancePCA"),
 "RASK_HUMAN_Weng_2022_binding-DARPin_K55": ("RASK_HUMAN", "A", "BIND", "DARPin K55 binding"),
 "S22A1_HUMAN_Yee_2023_abundance": ("S22A1_HUMAN", "A", "ABUND", "split-GFP abundance"),
 "S22A1_HUMAN_Yee_2023_activity": ("S22A1_HUMAN", "A", "ACT", "substrate uptake (survival)"),
 "SPIKE_SARS2_Starr_2020_expression": ("SPIKE_SARS2", "A", "ABUND", "RBD expression"),
 "SPIKE_SARS2_Starr_2020_binding": ("SPIKE_SARS2", "A", "BIND", "ACE2 binding"),
 "VKOR1_HUMAN_Chiasson_2020_abundance": ("VKOR1_HUMAN", "A", "ABUND", "eGFP abundance"),
 "VKOR1_HUMAN_Chiasson_2020_activity": ("VKOR1_HUMAN", "A", "ACT", "carboxylation activity"),
 "RL40A_YEAST_Roscoe_2013": ("RL40A_YEAST", "A", "FIT", "yeast growth"),
 "RL40A_YEAST_Mavor_2016": ("RL40A_YEAST", "A", "FIT", "yeast growth (stress DMSO)"),
 "RL40A_YEAST_Roscoe_2014": ("RL40A_YEAST", "A", "ACT", "E1 reactivity"),
 "SRC_HUMAN_Ahler_2019": ("SRC_HUMAN", "A", "ACT", "kinase activity (yeast toxicity)"),
 "SRC_HUMAN_Chakraborty_2023_binding-DAS_25uM": ("SRC_HUMAN", "A", "SPEC", "dasatinib resistance"),
 "SRC_HUMAN_Nguyen_2022": ("SRC_HUMAN", "A", "SPEC", "Hsp90 dependence"),
 # ---- group B: same property, different condition
 "CAR11_HUMAN_Meitlis_2020_gof": ("CAR11_HUMAN", "B", "COND", "signaling + ibrutinib (GOF)"),
 "CAR11_HUMAN_Meitlis_2020_lof": ("CAR11_HUMAN", "B", "COND", "signaling (LOF)"),
 "P53_HUMAN_Giacomelli_2018_Null_Etoposide": ("P53_HUMAN", "B", "COND", "null + etoposide"),
 "P53_HUMAN_Giacomelli_2018_Null_Nutlin": ("P53_HUMAN", "B", "COND", "null + nutlin"),
 "P53_HUMAN_Giacomelli_2018_WT_Nutlin": ("P53_HUMAN", "B", "COND", "WT + nutlin (dominant-negative)"),
 "P53_HUMAN_Kotler_2018": ("P53_HUMAN", "B", "COND", "H1299 growth (DBD)"),
 "HSP82_YEAST_Mishra_2016": ("HSP82_YEAST", "B", "COND", "growth, ATPase domain"),
 "HSP82_YEAST_Flynn_2019": ("HSP82_YEAST", "B", "COND", "growth, 37C, full length"),
 "HSP82_YEAST_Cote-Hammarlof_2020_growth-H2O2": ("HSP82_YEAST", "B", "COND", "growth, H2O2, middle domain"),
 "RPC1_LAMBD_Li_2019_low-expression": ("RPC1_LAMBD", "B", "COND", "repressor, low expression"),
 "RPC1_LAMBD_Li_2019_high-expression": ("RPC1_LAMBD", "B", "COND", "repressor, high expression"),
 "NPC1_HUMAN_Erwood_2022_HEK293T": ("NPC1_HUMAN", "B", "COND", "HEK293T"),
 "NPC1_HUMAN_Erwood_2022_RPE1": ("NPC1_HUMAN", "B", "COND", "RPE1"),
 # ---- group C: replicate assays of one property
 "BLAT_ECOLX_Deng_2012": ("BLAT_ECOLX", "C", "FIT", "Amp MIC"),
 "BLAT_ECOLX_Firnberg_2014": ("BLAT_ECOLX", "C", "FIT", "Amp growth"),
 "BLAT_ECOLX_Jacquier_2013": ("BLAT_ECOLX", "C", "FIT", "Amox MIC"),
 "BLAT_ECOLX_Stiffler_2015": ("BLAT_ECOLX", "C", "FIT", "Amp growth"),
 "CCDB_ECOLI_Adkar_2012": ("CCDB_ECOLI", "C", "FIT", "toxicity"),
 "CCDB_ECOLI_Tripathi_2016": ("CCDB_ECOLI", "C", "FIT", "toxicity"),
 "DYR_ECOLI_Nguyen_2023": ("DYR_ECOLI", "C", "FIT", "growth"),
 "DYR_ECOLI_Thompson_2019": ("DYR_ECOLI", "C", "FIT", "growth"),
 "F7YBW8_MESOW_Ding_2023": ("F7YBW8_MESOW", "C", "FIT", "growth"),
 "F7YBW8_MESOW_Aakre_2015": ("F7YBW8_MESOW", "C", "FIT", "growth"),
 "SPG1_STRSG_Olson_2014": ("SPG1_STRSG", "C", "BIND", "IgG binding"),
 "SPG1_STRSG_Wu_2016": ("SPG1_STRSG", "C", "BIND", "IgG binding"),
 "A0A2Z5U3Z0_9INFA_Doud_2016": ("A0A2Z5U3Z0_9INFA", "C", "FIT", "viral growth"),
 "A0A2Z5U3Z0_9INFA_Wu_2014": ("A0A2Z5U3Z0_9INFA", "C", "FIT", "viral growth"),
}
AXIS_LABEL = {"ABUND": "abundance / expression", "ACT": "activity / function", "BIND": "binding",
              "FIT": "growth / fitness", "SPEC": "drug resistance / Hsp90 dependence", "COND": "condition"}
AXIS_COLOR = {"ABUND": "#1b6ca8", "ACT": "#d1495b", "BIND": "#edae49", "FIT": "#66a182",
              "SPEC": "#8d6fb5", "COND": "#7f7f7f"}


def short(dms):
    return dms.replace("_Tsuboyama_2023", "")


def ref_table():
    r = pd.read_csv(REF, usecols=["DMS_id", "UniProt_ID", "coarse_selection_type", "selection_type",
                                  "seq_len", "includes_multiple_mutants", "DMS_total_number_mutants",
                                  "DMS_number_single_mutants", "taxon", "title", "first_author", "year"])
    r["category"] = r["coarse_selection_type"]
    r["is_tsuboyama"] = r["DMS_id"].str.contains("Tsuboyama")
    r["n_assays_protein"] = r.groupby("UniProt_ID")["DMS_id"].transform("size")
    a = pd.DataFrame.from_dict(ANNOT, orient="index", columns=["protein", "group", "axis", "assay_note"])
    a.index.name = "DMS_id"
    return r.merge(a.reset_index(), on="DMS_id", how="left")
