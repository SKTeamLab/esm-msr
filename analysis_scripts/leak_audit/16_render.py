"""Render the annotated BENCHMARKS tables from the ORIGINAL doc's numbers + the leak matrix."""
import os, re, subprocess, json
import pandas as pd, numpy as np
ROOT=os.environ.get('LEAK_ROOT','/home/sareeves/playground/esm-msr-devel'); OUT=os.path.join(ROOT,'leak_audit','out')
W=os.path.join(ROOT,'repo/.claude/worktrees/devel-main-conservative-merge-5e554f')
DOC=os.path.join(W,'docs/BENCHMARKS_OCT2026.md')
mx=pd.read_csv(os.path.join(OUT,'leak_matrix.csv'))
CODE={(r.model,r.dataset):(r.code, r.sup if isinstance(r.sup,str) else '') for r in mx.itertuples()}
def code_for(model,ds):
    if (model,ds) in CODE: return CODE[(model,ds)]
    if model=='DDMut' and ds in ('s669','s461'): return CODE.get(('DDMut [partial]',ds))
    return None
EXCL={'ᴹ','ᵐ','ᴾ','ᴴ','ᴰ','ᵁ','ⱽ','#'}
FLAGS=dict.fromkeys(map(ord,'†‡‽'),None)
def norm(x):
    return re.sub(r'\s+',' ',x.translate(FLAGS).replace('*','').replace('_','').replace('**','')).strip()
# parse original tables
old=open(DOC).read()
TABS=[]
for m in re.finditer(r'(?m)((?:^\|.*\|\n)+)', old):
    lines=m.group(0).strip().split('\n')
    hdr=[c.strip() for c in lines[0].split('|')[1:-1]]
    rows=[]
    for l in lines[2:]:
        cells=[c.strip() for c in l.split('|')[1:-1]]
        rows.append((cells[0], cells[1:]))
    TABS.append((hdr,rows))
def raw_value(ti,label,header):
    hdr,rows=TABS[ti]
    try: j=hdr.index(header)
    except ValueError: return None
    for lab,cells in rows:
        if norm(lab)==norm(label): return cells[j-1] if 0 <= j-1 < len(cells) else None
    return None
def to_float(x):
    x=re.sub(r'[^0-9.\-]','',x)
    try: return float(x)
    except Exception: return None

def render(ti, header, rows, datasets, italic_rows=(), partial_cells=()):
    """rows: list of (new_label, model_key); datasets: list of (display col, old header, code ds)."""
    out=['| '+' | '.join(header)+' |','|'+'|'.join(['---']*len(header))+'|']
    cod=['| '+' | '.join([header[0]]+[f'{c} (leak code)' for c in header[1:]])+' |','|'+'|'.join(['---']*len(header))+'|']
    vals={}
    for i,(label,model) in enumerate(rows):
        for j,(col,oldh,ds) in enumerate(datasets):
            rv=raw_value(ti,label,oldh)
            vals[(i,j)]=rv
    best=[None]*len(datasets)
    for j in range(len(datasets)):
        for i,(label,model) in enumerate(rows):
            rv=vals[(i,j)]; c=code_for(model,datasets[j][2])
            if rv is None or c is None or c[1] in EXCL: continue
            if datasets[j][2]=='MEAN': continue
            if model in italic_rows or (i,j) in partial_cells: continue
            f=to_float(rv)
            if f is None: continue
            if best[j] is None or f>best[j][0]: best[j]=(f,i)
    for i,(label,model) in enumerate(rows):
        cells=[label]; cr=[label]
        for j,(col,oldh,ds) in enumerate(datasets):
            rv=vals[(i,j)]; c=code_for(model,ds)
            if rv is None: cells.append('–'); cr.append('–'); continue
            sup=(c[1] if c else '') or ''
            # strip old flags, keep * and _ markers
            rv2=rv.replace('**','').translate(FLAGS)
            if model in italic_rows: rv2=rv2.replace('*','')
            txt=rv2+sup
            if best[j] and best[j][1]==i and model not in italic_rows and (i,j) not in partial_cells:
                txt=f'**{txt}**'
            cells.append(txt); cr.append(c[0] if c else '–')
        out.append('| '+' | '.join(cells)+' |'); cod.append('| '+' | '.join(cr)+' |')
    return '\n'.join(out), '\n'.join(cod)

D11=[('s571','s571','s571'),('s783','s783','s783'),('s2648','s2648','s2648'),('s8754','s8754','s8754'),
     ('s669','s669','s669'),('s461','s461','s461'),('ssym','ssym','ssym'),('q3421','q3421','q3421'),
     ('k3822','k3822','k3822'),('k2369','k2369','k2369'),('ptmul','ptmul','ptmul'),('ptmuld','ptmuld','ptmuld'),
     ('mean of 12','mean of 12','MEAN')]
rows11=[('ESM-MSR (released), combined','ESM-MSR (released)'),('ESM-MSR (released), WT head','ESM-MSR (released) WT'),
 ('ESM-MSR (devel), combined (3 seeds)','ESM-MSR (devel)'),('ESM-MSR (devel), WT head','ESM-MSR (devel) WT'),
 ('MutateEverything st1 (3 seeds)','ME stage1'),('MutateEverything st2 (3 seeds)','ME stage2'),
 ('ThermoMPNN (additive, 3 seeds)','TMPNN'),('ThermoMPNN + ThermoMPNN-D (3 seeds)','TMPNN+D'),
 ('SPURS (single model)','SPURS'),('SPURS multi (multi-mutant model)','SPURS multi'),('ESMtherm','ESMtherm'),
 ('DDGemb (server)','DDGemb'),('DDMut (server)','DDMut'),('IFUM, WT backbone','IFUM (WT backbone)'),
 ('IFUM, ESMFold per variant','IFUM (native, ESMFold)'),('GeoDDG-3D (FASPR)','GeoDDG-3D'),('ESM3ΔG','ESM3dG'),
 ('ProSST (zero-shot)','ProSST'),('ProteinMPNN (zero-shot)','ProteinMPNN'),
 ('ESM3-small-open, ε = 0 (zero-shot)','ESM3-small-open (ε=0, MSR readout)'),
 ('ESM3-large (Forge, zero-shot)','ESM3-large zero-shot'),('ESM3-medium (Forge, zero-shot)','ESM3-medium zero-shot'),
 ('ESM3-small (Forge, zero-shot)','ESM3-small zero-shot'),('ESM-MSR-small (March reference)','ESM-MSR-small (March, crystal)')]
t11,c11=render(0,['model']+[c[0] for c in D11],rows11,D11)
# fix DDMut mean and partial cells: old doc already has the values in the DDMut [partial] row? no; keep old '–'
D12=[('capped val','capped val','tsu_capped_val'),('capped test','capped test','tsu_capped_test')]
rows12=rows11[:21]
t12,c12=render(1,['model','capped val','capped test'],rows12,D12)
D13=[('DLG4 abund.','DLG4 abund.','DLG4 abund.'),('DLG4 bind.','DLG4 bind.','DLG4 bind.'),
     ('GRB2 abund.','GRB2 abund.','GRB2 abund.'),('GRB2 bind.','GRB2 bind.','GRB2 bind.'),
     ('MYO','MYO','MYO'),('ESTA dTm','ESTA dTm','ESTA'),('GB1','GB1','GB1'),('Domainome','Domainome','domainome')]
rows13=[('ESM-MSR (released), combined','ESM-MSR (released)'),('ESM-MSR (released), WT head','ESM-MSR (released) WT'),
 ('ESM-MSR (devel), combined (3 seeds)','ESM-MSR (devel)'),('ESM-MSR (devel), WT head','ESM-MSR (devel) WT'),
 ('MutateEverything st1 (3 seeds)','ME stage1'),('MutateEverything st1, additive','ME stage1+'),
 ('MutateEverything st2 (3 seeds)','ME stage2'),('MutateEverything st2, additive','ME stage2+'),
 ('ThermoMPNN (additive, 3 seeds)','TMPNN'),('ThermoMPNN + ThermoMPNN-D (3 seeds)','TMPNN+D'),
 ('SPURS (single model)','SPURS'),('SPURS multi (multi-mutant model)','SPURS multi'),('ESMtherm','ESMtherm'),
 ('DDGemb (server)','DDGemb'),('IFUM, WT backbone','IFUM (WT backbone)'),('IFUM, ESMFold per variant','IFUM (native, ESMFold)'),
 ('GeoDDG-3D (FASPR)','GeoDDG-3D'),('ESM3ΔG','ESM3dG'),('ProSST (zero-shot)','ProSST'),('ProteinMPNN (zero-shot)','ProteinMPNN'),
 ('ESM3-small-open, ε = 0 (zero-shot)','ESM3-small-open (ε=0, MSR readout)'),
 ('ESM3-large (Forge, zero-shot)','ESM3-large zero-shot'),('ESM3-medium (Forge, zero-shot)','ESM3-medium zero-shot'),
 ('ESM3-small (Forge, zero-shot)','ESM3-small zero-shot')]
t13,c13=render(2,['model']+[c[0] for c in D13],rows13,D13)
D14=[('PG avg','PG avg','PG avg'),('Stability','Stability','PG Stability'),('DMS mean','DMS mean','PG DMS mean')]
rows14=[('ESM-MSR (released), combined','ESM-MSR (released)'),('ESM-MSR (released), WT head','ESM-MSR (released) WT'),
 ('ESM-MSR (devel), combined (3 seeds)','ESM-MSR (devel)'),('MutateEverything st1 (3 seeds)','ME stage1'),
 ('MutateEverything st2 (3 seeds)','ME stage2'),('ThermoMPNN (additive, 3 seeds)','TMPNN'),('SPURS (single model)','SPURS'),
 ('SPURS multi (multi-mutant model)','SPURS multi'),('ESMtherm','ESMtherm'),('GeoDDG-3D (FASPR)','GeoDDG-3D'),
 ('ESM3ΔG','ESM3dG'),('ProSST (zero-shot)','ProSST'),('Rosetta Cartesian ΔΔG','Rosetta'),('ProteinMPNN (zero-shot)','ProteinMPNN'),
 ('ProteinMPNN, ProteinGym published recipe','corpus'),('ESM3-small-open, ε = 0 (zero-shot)','ESM3-small-open (ε=0, MSR readout)'),
 ('ESM3-open, ProteinGym published recipe','corpus')]
t14,c14=render(4,['model','PG avg','Stability','DMS mean'],rows14,D14,italic_rows=('Rosetta',))
open(os.path.join(OUT,'tables.md'),'w').write('\n\n'.join([t11,c11,t12,c12,t13,c13,t14,c14]))
print(t11); print(t14)
