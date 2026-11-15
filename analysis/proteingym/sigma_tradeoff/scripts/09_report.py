"""Assemble REPORT.md from ../report_template.md by filling {{TABLE:...}} placeholders from tables/*.csv.

Prose numbers in the template are for the 214-assay run; the tables are regenerated from whatever is in tables/.
"""
import re
import numpy as np, pandas as pd
from common import *

res = pd.read_csv(f"{TAB}/per_assay_rho.csv", index_col=0)
cl = pd.read_csv(f"{TAB}/assay_classes.csv", index_col=0)
pr = pd.read_csv(f"{TAB}/abundance_vs_other_pairs.csv")
sh = pd.read_csv(f"{TAB}/abundance_vs_other_shared_mutants.csv").set_index(["abund", "other"])
sm = pd.read_csv(f"{TAB}/summary_category.csv")
CLS = {"RISE": "rises/plateaus", "HUMP": "peak then falls", "HUMP_LOSS": "peak then net loss", "FLAT": "flat", "FALL": "falls"}
rc = [f"rho_{s}" for s in SIGMAS]


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def curve(d):
    return " / ".join(f"{res.loc[d, c]:.2f}" for c in rc)


def short(d):
    return d.split("_", 2)[2] if d.count("_") >= 2 else d


T = {}
# category summary
rows = []
for _, r in sm.iterrows():
    nm = {"OrganismalFitness": "Organismal fitness", "ALL": "All assays", "ALL_nonTsuboyama": "All excl. Tsuboyama"}.get(r.group, r.group)
    rows.append({"category": nm, "n": int(r.n), "mean rho at sigma 0 / .25 / .5 / .75 / 1": " / ".join(f"{r[f'mean_{s}']:.3f}" for s in SIGMAS),
                 "delta at 1.0 [95% CI over assays]": f"{r['delta_1.0']:+.3f} [{r['delta_1.0_lo']:+.3f}, {r['delta_1.0_hi']:+.3f}]",
                 "assays improved / harmed by >=0.02 at 1.0": f"{r['frac_up_1.0']:.0%} / {r['frac_down_1.0']:.0%}",
                 "best sigma of the mean": f"{r.best_sigma_of_mean:g}"})
T["CATEGORY"] = md(pd.DataFrame(rows))

# group A pairs
rows = []
for _, r in pr.iterrows():
    s = sh.loc[(r.abund, r.other)]
    rows.append({"protein": r.protein.split("_")[0], "abundance assay: rho at sigma 0/.25/.5/.75/1": curve(r.abund) + f" ({CLS[r.cls_abund]})",
                 "other assay": f"{r.other_axis.lower()}: {short(r.other)}", "other: rho at sigma 0/.25/.5/.75/1": curve(r.other) + f" ({CLS[r.cls_other]})",
                 "best joint sigma (shared mutants)": f"{s.best_joint_sigma:g}", "verdict": r.verdict})
T["PAIRS"] = md(pd.DataFrame(rows))

# group A without abundance + B + C
def grp_table(g, skip=()):
    sub = cl[(cl.group == g) & ~cl.index.isin(skip)].sort_values(["protein", "axis"])
    rows = []
    for d, r in sub.iterrows():
        rows.append({"protein": r.protein.split("_")[0], "assay": short(d) if g != "A" else short(d), "what it measures": ANNOT[d][3],
                     "rho at sigma 0/.25/.5/.75/1": curve(d), "delta at 1.0": f"{r.d_full:+.3f}", "shape": f"{CLS[r.cls]} (sigma*={r.sigma_star:g})"})
    return md(pd.DataFrame(rows))

skipA = set(pr.abund) | set(pr.other)
T["A_NOABUND"] = grp_table("A", skip=skipA)
T["B"] = grp_table("B")
T["C"] = grp_table("C")

# multi-assay curve-class counts
m = cl[cl.group.notna()]
ct = pd.crosstab(m.group.map({"A": "A: different properties", "B": "B: conditions", "C": "C: replicates"}), m.cls).reindex(columns=list(CLS), fill_value=0)
ct.columns = [CLS[c] for c in ct.columns]
ct.insert(0, "group", ct.index)
T["MULTI_CLASSES"] = md(ct.reset_index(drop=True))

tpl = open(f"{OUT}/report_template.md").read()
out = re.sub(r"\{\{TABLE:(\w+)\}\}", lambda mo: T[mo.group(1)], tpl)
open(f"{OUT}/REPORT.md", "w").write(out)
print("wrote", f"{OUT}/REPORT.md", len(out), "chars")
