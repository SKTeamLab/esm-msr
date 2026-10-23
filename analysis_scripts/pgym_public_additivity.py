"""Is a published ProteinGym column additive? For every DMS with >= 50 multi-mutants, compares each multi-mutant's
published score with the sum of its component singles' published scores in the same file (ProSST-2048, ESM3, ProteinMPNN).
Result 2026-10-23: ProSST and ESM3 additive to <= 2e-14 over 1,499,350 variants in 69 DMS; ProteinMPNN is not (up to 52).
Run on the cluster under msr_venv."""
import zipfile, io, numpy as np, pandas as pd
z = zipfile.ZipFile('/home/sreeves/stability-baselines/pgym_public/zero_shot_substitutions_scores.zip')
tot = ok = 0; worst = 0.0; dms_used = 0; rows = []
for name in z.namelist():
    if not name.endswith('.csv'): continue
    d = pd.read_csv(io.TextIOWrapper(z.open(name)), usecols=['mutant', 'ProSST-2048', 'ESM3', 'ProteinMPNN'])
    multi = d[d.mutant.str.contains(':')]
    if len(multi) < 50: continue
    s = d[~d.mutant.str.contains(':')].drop_duplicates('mutant').set_index('mutant')
    row = {'dms': name[:-4], 'n_multi': len(multi)}
    for col in ['ProSST-2048', 'ESM3', 'ProteinMPNN']:
        parts = multi.mutant.str.split(':')
        add = parts.map(lambda ms: sum(s[col].get(m, np.nan) for m in ms))
        have = add.notna() & multi[col].notna()
        row[col + ' n'] = int(have.sum())
        row[col + ' maxabs'] = float(np.abs(multi[col][have] - add[have]).max()) if have.any() else np.nan
    rows.append(row)
R = pd.DataFrame(rows)
print(len(R), 'DMS with >=50 multi-mutants')
print(R.to_string(float_format=lambda x: f'{x:.2g}', max_rows=20))
for col in ['ProSST-2048', 'ESM3', 'ProteinMPNN']:
    print(col, 'variants checked', int(R[col + ' n'].sum()), '| worst |multi - sum of singles| over all DMS:', f'{R[col + " maxabs"].max():.3g}')
