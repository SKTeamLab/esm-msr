"""Summary tables used by the report (re-generated on every run)."""
import numpy as np, pandas as pd
from common import *

res = pd.read_csv(f"{TAB}/per_assay_rho.csv", index_col=0)
cl = pd.read_csv(f"{TAB}/assay_classes.csv", index_col=0)
rc = [f"rho_{s}" for s in SIGMAS]
rng = np.random.default_rng(0)
CATS = ["Stability", "Expression", "Activity", "Binding", "OrganismalFitness"]


def ci(v, nb=4000):
    b = v[rng.integers(0, len(v), (nb, len(v)))].mean(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


rows = []
for c in CATS + ["ALL", "ALL_nonTsuboyama"]:
    m = (res.category == c) if c in CATS else (np.ones(len(res), bool) if c == "ALL" else ~res.is_tsuboyama.values)
    R = res.loc[m, rc].values
    row = dict(group=c, n=int(m.sum()))
    for s, k in zip(SIGMAS, range(5)):
        row[f"mean_{s}"] = R[:, k].mean()
    for s, k in zip(SIGMAS[1:], range(1, 5)):
        d = R[:, k] - R[:, 0]
        lo, hi = ci(d)
        row[f"delta_{s}"], row[f"delta_{s}_lo"], row[f"delta_{s}_hi"] = d.mean(), lo, hi
        row[f"frac_up_{s}"] = (d >= 0.02).mean(); row[f"frac_down_{s}"] = (d <= -0.02).mean()
    row["best_sigma_of_mean"] = SIGMAS[int(np.argmax(R.mean(0)))]
    rows.append(row)
sm = pd.DataFrame(rows)
sm.to_csv(f"{TAB}/summary_category.csv", index=False)
cls_ct = pd.crosstab(cl.category, cl.cls).reindex(CATS)
cls_ct.to_csv(f"{TAB}/summary_class_counts.csv")

ntb = res[~res.is_tsuboyama].copy()
ntb["delta_1"] = ntb["rho_1.0"] - ntb["rho_0.0"]
ntb["delta_best"] = ntb[rc].max(axis=1) - ntb["rho_0.0"]
ntb[["delta_1", "delta_best", "category", "protein", "axis"]].sort_values("delta_1").to_csv(f"{TAB}/nonTsuboyama_ranked_by_delta_full.csv")
print(sm[["group", "n"] + [f"mean_{s}" for s in SIGMAS]].round(3).to_string())
print(sm[["group", "delta_1.0", "delta_1.0_lo", "delta_1.0_hi", "frac_up_1.0", "frac_down_1.0", "best_sigma_of_mean"]].round(3).to_string())
print(cls_ct)
print("\nmost harmed at sigma 1.0 (non-Tsuboyama):"); print(ntb.sort_values("delta_1").head(12)[["delta_1", "category"]].round(3).to_string())
print("\nmost helped (non-Tsuboyama):"); print(ntb.sort_values("delta_1").tail(12)[["delta_1", "category"]].round(3).to_string())
print("\nsigma* distribution non-Tsuboyama:", cl[~cl.is_tsuboyama].sigma_star.value_counts().sort_index().to_dict())
print("peak at 0 (never helped) frac non-Tsub:", (cl[~cl.is_tsuboyama].sigma_star == 0).mean().round(3))
