"""Shared Tsuboyama / MegaScale column descriptors, per-library evaluation, and paired bootstrap helpers.

Used internally by ``p02_tsuboyama`` and ``p06_epistasis_hierarchy`` so neither pipeline duplicates
column-building or per-library bootstrap code and neither imports a sibling pipeline script.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from msr_analysis import data as D
from msr_analysis import models as M
from msr_analysis import style
from msr_analysis.boot import bootstrap_rho, ci
from msr_analysis.common import subframe
from msr_analysis.paths import atomic_to_csv
from msr_analysis.registry import tsu_datasets
from msr_analysis.stats import boot_pvalue, holm, make_domain_seed_draws, ndcg_at_k

MIN_N = 10
K_NDCG = 96
BEST_LABEL = {'geoddg': 'GeoDDG-3D', 'ddgemb': 'DDGemb'}
F_COMPOSITE = (
    'Every variant is scored by the version of its model that fits it: singles by the first, multi-mutants by the second '
    '(ESM-MSR WT-view / Dual-view, MutateEverything Stage 1 / Stage 2, ThermoMPNN / ThermoMPNN-D, SPURS / SPURS-multi, '
    'ESM3-small-open WT-view / Dual-view).'
)
IN_PER_UNIT = 12.5 / 11.85
XPAD = (0.7, 0.75)
SAME = object()


def split_title(split: str) -> str:
    return 'Homology-Restricted MegaScale' if split == 'capped' else 'Full MegaScale (mega_splits)'


def test_ds_for_split(split: str) -> str:
    return tsu_datasets(split)[1]


def col_spec(name: str, cat: str, color: str, label: str, single, double=SAME, **kw) -> dict:
    return dict(name=name, cat=cat, color=color, label=label, single=single, double=single if double is SAME else double, **kw)


def fig_width(cols: list[dict]) -> float:
    drawn = [c for c in cols if c.get('draw', True)]
    xs, _ = style.layout_grouped_columns(drawn, gap=0.7)
    return IN_PER_UNIT * (xs[-1] - xs[0] + sum(XPAD)) if xs else 8.0


def load_tsu_test_frame(split: str = 'capped') -> tuple[D.Frame, list[str]]:
    ds = test_ds_for_split(split)
    fr = D.load_frame(ds)
    # Work on a shallow copy of preds so callers can attach composite/masked keys cleanly.
    fr = D.Frame(fr.name, fr.spine, dict(fr.preds), dict(fr.notes))
    added = D.add_esm3_masked(fr, ds) if split == 'capped' else []
    if 'esm3avg:epi' in fr.preds:
        fr.preds['esm3:epi'] = fr.preds['esm3avg:epi']
    return fr, added


def add_esm3_marginal_additive(fr: D.Frame, split: str = 'capped') -> None:
    if split != 'capped':
        return
    p = D._esm3_file('marginal', test_ds_for_split(split))
    if p is not None and p.exists():
        d = pd.read_csv(p, usecols=['id', 'wt_lora_pred_additive'])
        fr.preds['esm3mm+'] = D._align_id(
            fr.spine['id'].to_numpy(dtype=object),
            d['id'].astype(str).to_numpy(),
            d['wt_lora_pred_additive'].to_numpy(),
        )[None, :]


def rho_fn(p: np.ndarray, y: np.ndarray) -> float:
    return float(spearmanr(p, y)[0]) if len(y) >= 3 and np.ptp(p) > 0 else np.nan


def ndcg96_rectified(p: np.ndarray, y: np.ndarray) -> float:
    return ndcg_at_k(p, y, k=K_NDCG, mode='rectified')


def ndcg96_shifted(p: np.ndarray, y: np.ndarray) -> float:
    return ndcg_at_k(p, y, k=K_NDCG, mode='shifted')


def per_library(fr: D.Frame, key: str, mask: np.ndarray, fn, min_n: int, y: np.ndarray | None = None) -> pd.DataFrame:
    sp = fr.spine
    yy = sp['y'].to_numpy() if y is None else y
    g = sp['group'].to_numpy()
    out = {}
    for lib in np.unique(g[mask]):
        m = mask & (g == lib) & np.isfinite(fr.preds[key]).all(axis=0)
        if m.sum() < min_n:
            continue
        vals = [fn(pr[m], yy[m]) for pr in fr.preds[key]]
        if np.isfinite(vals).any():
            out[lib] = vals
    if not out:
        return pd.DataFrame(dtype=float)
    return pd.DataFrame.from_dict(out, orient='index', dtype=float)


def mean_val(values: pd.Series | pd.DataFrame) -> float:
    if isinstance(values, pd.DataFrame):
        return float(values.mean(axis=1).mean())
    return float(values.mean())


def diff_df(va: pd.Series | pd.DataFrame, vb: pd.Series | pd.DataFrame, both=None) -> pd.DataFrame:
    da = va.to_frame() if isinstance(va, pd.Series) else va
    db = vb.to_frame() if isinstance(vb, pd.Series) else vb
    if both is not None:
        da, db = da.reindex(both), db.reindex(both)
    return pd.DataFrame(da.to_numpy(dtype=float) - db.to_numpy(dtype=float), index=da.index)


def boot_reps(values: pd.Series | pd.DataFrame, libs: list[str], counts: np.ndarray) -> np.ndarray:
    df = values.to_frame() if isinstance(values, pd.Series) else values
    v = df.reindex(libs)
    ok = v.notna().any(axis=1).to_numpy()
    c = counts[:, ok]
    V = v.loc[ok].to_numpy(dtype=float)
    n_seeds = V.shape[1]
    fin = np.isfinite(V).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        dom_per_seed = np.where((c @ fin) > 0, (c @ np.where(np.isfinite(V), V, 0.0)) / (c @ fin), np.nan)
    if n_seeds == 1:
        return dom_per_seed[:, 0]
    seed_w = np.random.default_rng(17).multinomial(n_seeds, np.full(n_seeds, 1.0 / n_seeds), size=len(counts)).astype(float) / n_seeds
    return np.nansum(dom_per_seed * seed_w, axis=1)


def boot_mean(values: pd.Series | pd.DataFrame, libs: list[str], counts: np.ndarray) -> tuple[float, float, float, int]:
    df = values.to_frame() if isinstance(values, pd.Series) else values
    v = df.reindex(libs)
    ok = v.notna().any(axis=1)
    lo, hi = ci(boot_reps(values, libs, counts))
    return float(v.loc[ok].mean(axis=1).mean()), lo, hi, int(ok.sum())


def best_other(fr: D.Frame, out_csv_path=None) -> tuple[str | None, dict[str, float]]:
    sp = fr.spine
    single = sp['use'].to_numpy() & (sp['n_mut'] == 1).to_numpy()
    sc = {k: mean_val(per_library(fr, k, single, rho_fn, MIN_N)) for k in BEST_LABEL if k in fr.preds}
    if not sc:
        return None, sc
    best = max(sc, key=sc.get)
    if out_csv_path is not None:
        s_df = pd.Series(sc, name='mean per-library singles rho').rename_axis('model').reset_index()
        atomic_to_csv(s_df, out_csv_path, index=False, float_format='%.4f')
    return best, sc


def esm3_choice(fr: D.Frame, want: str = 'unmasked'):
    p = fr.preds
    names = {
        'unmasked': ('msrrel@0:comb', 'ESM3-small-open\nunmasked'),
        'average': ('esm3avg', 'ESM3-small-open\nindependent masking\n(WT, MT average)'),
        'marginal': ('esm3mm', 'ESM3-small-open\nmasked marginal'),
        'independent': ('esm3ind', 'ESM3-small-open\nindependent masking\n(mutant pass)'),
    }
    have = [s for s in [want] + [k for k in names if k != want] if names[s][0] in p]
    if not have:
        return None, None, None
    return names[have[0]][0], names[have[0]][1], (names[have[1]] if len(have) > 1 else None)


def roster_columns(fr: D.Frame, which: str, best: str | None, esm3_want: str = 'unmasked') -> list[dict]:
    p = fr.preds
    cols = []
    for name in M.ZEROSHOT + M.TRANSFER + M.SUPERVISED:
        is_best = name == 'GeoDDG-3D'
        if is_best and best is None:
            continue
        c = M.column(BEST_LABEL[best] if is_best else name, which)
        if c is None:
            continue
        if is_best:
            c = dict(
                c,
                name='best other ddG',
                cat='supervised',
                single=(best + '+' if which == 'pair' else c['single']),
                double=best + (':epi' if which == 'epi' else ''),
            )
        if name == 'ESM3-small-open' and which == 'all':
            ek, el, _ = esm3_choice(fr, esm3_want)
            if ek is not None and ek != 'msrrel@0:comb':
                c = dict(c, label=el, single=ek, double=ek)
        if name == 'ESM3-small-open' and which == 'epi':
            c = dict(c, label=c['label'] + '\n(unmasked)')
        cols.append(c)
    return [c for c in cols if all(c[k] is None or c[k] in p for k in ('single', 'double'))]


def main_columns(fr: D.Frame, best: str | None, esm3_want: str = 'unmasked') -> list[dict]:
    cols = roster_columns(fr, 'all', best, esm3_want)
    _, _, alt = esm3_choice(fr, esm3_want)
    if alt:
        cols.append(dict(M.column('ESM3-small-open', 'all'), name='ESM3-small-open (alt)', label=alt[1], single=alt[0], double=alt[0], draw=False))
    n1 = (fr.spine['n_mut'] == 1).to_numpy()
    for c in cols:
        if c['single'] != c['double']:
            c['all'] = f"{c['single']}|{c['double']}"
            fr.preds[c['all']] = np.where(n1[None, :], fr.preds[c['single']], fr.preds[c['double']])
    return cols


def pair_columns(fr: D.Frame, best: str | None) -> list[dict]:
    return roster_columns(fr, 'pair', best)


def epi_columns(fr: D.Frame, best: str | None) -> list[dict]:
    p = fr.preds
    cols = roster_columns(fr, 'epi', best)
    es = M.column('ESM3-small-open', 'epi')
    if 'esm3:epi' in p:
        cols.append(dict(es, name='ESM3-small-open (independent masking)', label='ESM3-small-open', double='esm3:epi', draw=False))
    if 'esm3mm:epi' in p:
        cols.append(dict(es, name='ESM3-small-open (masked marginal)', label='ESM3-small-open', double='esm3mm:epi', draw=False))
    return cols


def key_for(c: dict, which: str) -> str | None:
    if which == 'all':
        return c.get('all', c['single'])
    return c['single'] if which == 'singles' else c['double']


def table_grouped(
    fr: D.Frame,
    cols: list[dict],
    cells: dict,
    fn,
    min_n: int,
    counts: np.ndarray,
    libs: list[str],
    y: np.ndarray | None = None,
    kind: str = 'domain',
    B_mut: int = 1000,
    seed: int = 11,
    vals: dict | None = None,
) -> pd.DataFrame:
    ds_name = fr.name
    g = fr.spine['group'].to_numpy()
    yy = fr.spine['y'].to_numpy() if y is None else y
    rows = []
    for c in cols:
        for sub, (m, which) in cells.items():
            key = key_for(c, which)
            if key is None or key not in fr.preds:
                continue
            v = per_library(fr, key, m, fn, min_n, y)
            if len(v) < 2:
                continue
            pt, lo, hi, n = boot_mean(v, libs, counts)
            if vals is not None:
                vals[(c['name'], sub)] = (key, v, m, yy)
            if kind == 'mutation':
                m2 = m & np.isin(g, v.index) & np.isfinite(yy) & np.isfinite(fr.preds[key]).all(axis=0)
                _, reps = bootstrap_rho({ds_name: subframe(fr, m2, y=yy, keys=[key])}, [key], kind='mutation', B=B_mut, seed=seed, per_protein=True)
                lo, hi = ci(reps[ds_name][key])
            rows.append(dict(column=c['name'], key=key, subset=sub, value=pt, lo=lo, hi=hi, n_libraries=n, n_variants=int((m & np.isin(g, v.index)).sum())))
    return pd.DataFrame(rows)


def table_pooled(
    fr: D.Frame,
    cols: list[dict],
    cells: dict,
    y: np.ndarray,
    B_: int = 1000,
    seed: int = 7,
    kind: str = 'domain',
    reps_out: dict | None = None,
) -> pd.DataFrame:
    ds_name = fr.name
    rows = []
    for sub, (m, which) in cells.items():
        names = {key_for(c, which): c['name'] for c in cols if key_for(c, which) in fr.preds}
        keys = list(names)
        if not keys:
            continue
        sf = subframe(fr, m & np.isfinite(y), y=y, keys=keys)
        point, reps = bootstrap_rho({ds_name: sf}, keys, kind='protein' if kind == 'domain' else 'mutation', B=B_, seed=seed, per_protein=False)
        for k, (r, _per_seed, n, ng) in point[ds_name].items():
            lo, hi = ci(reps[ds_name][k])
            if reps_out is not None:
                reps_out[(sub, names[k])] = reps[ds_name][k]
            rows.append(dict(column=names[k], key=k, subset=sub, value=r, lo=lo, hi=hi, n_variants=n, n_libraries=ng))
    return pd.DataFrame(rows)


def _pair_names(names: list[str]):
    for a in names:
        if a.startswith('ESM-MSR'):
            for b in names:
                if b != a and not (b.startswith('ESM-MSR') and names.index(b) < names.index(a)):
                    yield a, b


def row_of(sub: str, a: str, b: str, va: float, vb: float, d: float, reps: np.ndarray, n: int, bootstrap: str, what: str, name_map: dict[str, str] | None = None) -> dict:
    nm = name_map or {}
    lo, hi = ci(reps)
    return dict(
        comparison=sub,
        model_a=nm.get(a, a),
        model_b=nm.get(b, b),
        value_a=va,
        value_b=vb,
        delta=d,
        lo=lo,
        hi=hi,
        p=boot_pvalue(reps, null=0.0, alternative='two-sided'),
        n_libraries=n,
        bootstrap=bootstrap,
        statistic=what,
    )


def add_holm_by_group(df: pd.DataFrame, by=('comparison', 'model_a')) -> pd.DataFrame:
    if df.empty:
        return df
    group_cols = [b for b in (*by, 'panel') if b in df.columns]
    df = df.reset_index(drop=True).copy()
    df['p_holm'] = np.nan
    for _, idx in df.groupby(group_cols).groups.items():
        df.loc[idx, 'p_holm'] = holm(df.loc[idx, 'p'].to_numpy())
    return df


def compare_grouped(
    fr: D.Frame,
    vals: dict,
    subsets: list[str],
    counts: np.ndarray,
    libs: list[str],
    kind: str = 'domain',
    B_mut: int = 1000,
    seed: int = 13,
    what: str = 'mean over domains',
    name_map: dict[str, str] | None = None,
) -> list[dict]:
    ds_name = fr.name
    rows = []
    for sub in subsets:
        names = [n for (n, s_) in vals if s_ == sub]
        for a, b in _pair_names(names):
            ka, va, ma, ya = vals[(a, sub)]
            kb, vb, _mb, _yb = vals[(b, sub)]
            both = va.index.intersection(vb.index)
            if len(both) < 2:
                continue
            d = diff_df(va, vb, both)
            if kind == 'domain':
                reps = boot_reps(d, libs, counts)
            else:
                g = fr.spine['group'].to_numpy()
                m2 = ma & np.isin(g, both) & np.isfinite(ya) & np.isfinite(fr.preds[ka]).all(axis=0) & np.isfinite(fr.preds[kb]).all(axis=0)
                _, r = bootstrap_rho({ds_name: subframe(fr, m2, y=ya, keys=[ka, kb])}, [ka, kb], kind='mutation', B=B_mut, seed=seed, per_protein=True)
                reps = r[ds_name][ka] - r[ds_name][kb]
            rows.append(row_of(sub, a, b, mean_val(va.loc[both]), mean_val(vb.loc[both]), mean_val(d), reps, len(both), 'domains' if kind == 'domain' else 'mutations', what, name_map))
    return rows


def compare_pooled(T: pd.DataFrame, reps: dict, subsets: list[str], kind: str = 'domain', name_map: dict[str, str] | None = None) -> list[dict]:
    rows = []
    for sub in subsets:
        t = T[T.subset == sub].set_index('column')
        for a, b in _pair_names(list(t.index)):
            rows.append(row_of(
                sub, a, b,
                float(t.value[a]), float(t.value[b]), float(t.value[a] - t.value[b]),
                reps[(sub, a)] - reps[(sub, b)],
                int(min(t.n_libraries[a], t.n_libraries[b])),
                'domains' if kind == 'domain' else 'mutations',
                'pooled over domains',
                name_map,
            ))
    return rows


def count_text(T: pd.DataFrame, sub: str, text: str) -> str:
    r = T[T.subset == sub]
    if r.empty:
        return text
    return f'{text} (n = {int(r.n_variants.max()):,}, {int(r.n_libraries.max())} domains)'


def finish_bar_fig(fig, title: str, foot: list[str], top: float = 0.955) -> None:
    fig.suptitle(title, x=0.012, ha='left', fontsize=10, y=0.995)
    if foot:
        fig.text(0.012, 0.008, '\n'.join(foot), fontsize=7.2, va='bottom', ha='left', color='#444444')
    fig.tight_layout(rect=(0, 0.014 + 0.021 * len(foot), 1, top))


def make_domain_counts(libs: list[str], B: int, seed: int = 5) -> np.ndarray:
    counts, _ = make_domain_seed_draws(len(libs), B=B, seed=seed)
    return counts
