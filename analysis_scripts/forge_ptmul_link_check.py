# NOTE: cluster paths. Run on subhak-gpu under msr_venv; see docs/BENCHMARKS_OCT2026.md section 1.1.
"""Link the ESM3 Forge PTMUL rows to the preprocessed source by VARIANT IDENTITY, not by any ID string.
For each Forge row: apply its mutation string (read as sequence-position numbering) to the wild-type sequence of the
same PDB code and look the resulting sequence up among the source's mut_seq. A hit proves the string denotes that
variant, whatever the id looks like."""
import ast
import numpy as np, pandas as pd
F = '/home/sreeves/esm-msr-devel/analysis_notebooks/predictions/PTMUL/esm3_forge/zero_shot/esm3-large-2024-03/predictions.csv'
SRC = '/home/sreeves/esm-msr-devel/data/preprocessed/ptmul_mapped.csv'
f = pd.read_csv(F, low_memory=False)
s = pd.read_csv(SRC, low_memory=False)
f['code2'] = f['code'].fillna(f['id'].astype(str).str[:4]).astype(str)
f['nm'] = f.mut_type.astype(str).str.count(':') + 1
print('orig_to_new example:', str(s.orig_to_new.iloc[0])[:120])

def apply(seq, muts):
    q = list(seq)
    for m in muts.split(':'):
        try:
            p = int(m[1:-1]) - 1
        except ValueError:
            return None
        if p < 0 or p >= len(q) or q[p] != m[0]:
            return None
        q[p] = m[-1]
    return ''.join(q)

# source identity index: (code, mutated sequence) -> source row
s['code'] = s['code'].astype(str)
ident = {}
for i, (c, ms) in enumerate(zip(s.code, s.mut_seq)):
    ident.setdefault((c, ms), []).append(i)
wts = s.groupby('code').wt_seq.apply(lambda x: sorted(set(x)))
print('source: codes', s.code.nunique(), '| codes with >1 distinct wt_seq:', int((wts.map(len) > 1).sum()),
      '| duplicate (code, mut_seq):', sum(len(v) > 1 for v in ident.values()))

hit, src_row, how = [], [], []
for c, m in zip(f.code2, f.mut_type.astype(str)):
    got = None
    for w in wts.get(c, []):
        r = apply(w, m)
        if r is not None and (c, r) in ident:
            got = ident[(c, r)][0]
            break
    hit.append(got is not None); src_row.append(got)
f['hit'] = hit
f['src'] = src_row
print('\nForge rows linked to a source variant by sequence identity: %d of %d' % (int(f.hit.sum()), len(f)))
print('  by #mutations -> linked/total:', {int(n): f"{int(g.hit.sum())}/{len(g)}" for n, g in f.groupby('nm')})
ok = f[f.hit].copy()
ok['label'] = s.loc[ok.src.astype(int), 'ddG'].to_numpy()
ok['nmut_src'] = s.loc[ok.src.astype(int), 'NMUT'].to_numpy()
print('  linked source rows are distinct:', ok.src.nunique(), '| source rows NOT reached by any Forge row:', len(s) - ok.src.nunique())
def r(d): return d[['esm3_score', 'label']].corr('spearman').iloc[0, 1]
print('\nspearman(esm3_score, source label) via identity link: all %.4f (n=%d) | doubles %.4f (n=%d) | 3+ %.4f (n=%d)' % (
    r(ok), len(ok), r(ok[ok.nm == 2]), int((ok.nm == 2).sum()), r(ok[ok.nm >= 3]), int((ok.nm >= 3).sum())))
print('  by N:', {int(n): round(r(g), 3) for n, g in ok.groupby('nm') if len(g) > 20})
print('  does the Forge mutation count agree with the source NMUT on linked rows:', bool((ok.nm == ok.nmut_src).all()))
bad = f[~f.hit]
print('\nunlinked examples:'); print(bad[['id', 'mut_type', 'nm']].head(8).to_string(index=False))
