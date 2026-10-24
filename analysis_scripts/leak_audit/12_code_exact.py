"""Exact-mutation overlaps for inventories distributed as PDB-code/position tables:
DDMut S9028 + SM1242, GeoDDG S8754, DDGemb S2450 (via S2648)."""
import os, re, json, pickle
import pandas as pd, numpy as np
ROOT=os.environ.get('LEAK_ROOT','/home/sareeves/playground/esm-msr-devel'); D=os.path.join(ROOT,'leak_audit','data'); OUT=os.path.join(ROOT,'leak_audit','out')
REPO=os.path.join(ROOT,'repo','data','preprocessed')
res=[]
def add(inv,ds,n,exact,note=''):
    res.append(dict(inventory=inv,dataset=ds,n=n,exact=exact,frac=exact/max(n,1),note=note)); print(f'{inv:16s} {ds:18s} exact {exact:6d}/{n:6d} = {exact/max(n,1):.3f} {note}')

# ---- DDMut S9028
S=pd.read_csv(os.path.join(D,'S9028.csv'))
s9028=set(); gb=set()
for x in S['pdb_mut_chain'].astype(str):
    m=re.match(r'^([0-9A-Za-z]{4})_([A-Z])(\d+)([A-Z])_([A-Za-z0-9]+)$', x)
    if not m: continue
    code=m.group(1).upper(); wt=m.group(2); pos=int(m.group(3)); mt=m.group(4); ch=m.group(5)
    s9028.add((code,ch,pos,wt,mt))
    if code=='1PGA': gb.add((pos,wt,mt))
ext={'s571':'s571_mapped.csv','s783':'s783_mapped.csv','s2648':'s2648_mapped.csv','s8754':'s8754_mapped.csv',
     's669':'s669_mapped.csv','s461':'s461_mapped.csv','ssym':'ssym_mapped.csv','q3421':'q3421_mapped.csv',
     'k3822':'k3822_mapped.csv','k2369':'k2369_mapped.csv'}
for name,f in ext.items():
    d=pd.read_csv(os.path.join(REPO,f), low_memory=False); n=0; hit=0
    for _,r in d.iterrows():
        try: pos=int(r['position'])
        except Exception: continue
        n+=1
        if (str(r['code']).upper(), str(r['chain']), pos, str(r['wild_type']), str(r['mutation'])) in s9028: hit+=1
    add('ddmut_s9028', name, n, hit)
# GB1
gbt=pd.read_pickle(os.path.join(D,'testsets','GB1.pkl'))
hit=0
for mut in gbt['mutation']:
    m=re.match(r'^([A-Z])(\d+)([A-Z])$', mut)
    if m and (int(m.group(2)), m.group(1), m.group(3)) in gb: hit+=1
# constituent: every mutation of a GB1 row is present in some trained 1PGA single
def constituent_hits(testdf, single_set):
    c=0
    for mut in testdf['mutation']:
        parts=[x for x in str(mut).split(';') if x]
        ok=True
        for x in parts:
            m=re.match(r'^([A-Z])(\d+)([A-Z])$',x)
            if not m or (int(m.group(2)),m.group(1),m.group(3)) not in single_set: ok=False; break
        if ok and len(parts)>1: c+=1
    return c
add('ddmut_s9028','GB1',len(gbt),hit,'via 1PGA')
res[-1]['constituent']=constituent_hits(gbt,gb)
# SM1242 vs PTMUL/PTMULD
sm=pd.read_csv(os.path.join(D,'raw','DD_SM1242.csv'))
def parse_set(x):
    parts=re.split(r'[_:]', str(x)); return tuple(sorted(p for p in parts if re.match(r'^[A-Z]\d+[A-Z]$',p)))
smk=set(zip(sm['struc'].astype(str).str.upper(), [parse_set(x) for x in sm['mm_label']]))
for ds,f in [('ptmul','ptmul_mapped.csv'),('ptmuld','ptmuld_mapped.csv')]:
    d=pd.read_csv(os.path.join(REPO,f), low_memory=False); n=0;hit=0
    for _,r in d.iterrows():
        muts=tuple(sorted(x for x in re.split(r'[:;]',str(r.get('mut_info',''))) if re.match(r'^[A-Z]\d+[A-Z]$',x)))
        n+=1
        if (str(r['code']).upper()+'_'+str(r['chain']), muts) in smk: hit+=1
    add('ddmut_sm1242', ds, n, hit)

# ---- GeoDDG S8754 (training file == our S8754 dataset)
GS=pd.read_csv(os.path.join(D,'raw','GS_S8754.csv'))
gs=set(); gsgb=set()
for x in GS['name'].astype(str):
    m=re.match(r'^rcsb_([0-9A-Za-z]{4})_([A-Za-z0-9]+)_([A-Z])(\d+)([A-Z])', x)
    if not m: continue
    code=m.group(1).upper(); ch=m.group(2); wt=m.group(3); pos=int(m.group(4)); mt=m.group(5)
    gs.add((code,ch,pos,wt,mt))
    if code=='1PGA': gsgb.add((pos,wt,mt))
print('GS_S8754 unique keys',len(gs), 'rows',len(GS))
for name,f in ext.items():
    d=pd.read_csv(os.path.join(REPO,f), low_memory=False); n=0; hit=0
    for _,r in d.iterrows():
        try: pos=int(r['position'])
        except Exception: continue
        n+=1
        if (str(r['code']).upper(), str(r['chain']), pos, str(r['wild_type']), str(r['mutation'])) in gs: hit+=1
    add('geoddg_s8754', name, n, hit)
hit=0
for mut in gbt['mutation']:
    m=re.match(r'^([A-Z])(\d+)([A-Z])$', mut)
    if m and (int(m.group(2)), m.group(1), m.group(3)) in gsgb: hit+=1
add('geoddg_s8754','GB1',len(gbt),hit,'via 1PGA')
res[-1]['constituent']=constituent_hits(gbt,gsgb)

# ---- DDGemb S2450: published relationship to S2648
s2648=pd.read_pickle(os.path.join(D,'testsets','s2648.pkl'))
add('ddgemb_s2450','s2648',2648,2450,'S2450 = S2648 - 198 mutations (documented)')
# inherited upper bounds via the measured cross-set intersections with S2648
cross={'s8754':2587,'q3421':1549,'k3822':979,'k2369':566,'ssym':194,'s571':52,'s783':10}
ns={'s8754':8754,'q3421':3421,'k3822':3822,'k2369':2369,'ssym':684,'s571':571,'s783':783}
for k,v in cross.items(): add('ddgemb_s2450',k,ns[k],v,'upper bound via S2648')
add('ddgemb_s2450','s669',669,0,'by construction')
add('ddgemb_s2450','s461',461,0,'S461 subset of S669')
add('ddgemb_s2450','ptmul',914,0,'single-mutant training only')
add('ddgemb_s2450','ptmuld',536,0,'single-mutant training only')
pd.DataFrame(res).to_csv(os.path.join(OUT,'code_inventory_exact.csv'), index=False)
print('saved', len(res))
