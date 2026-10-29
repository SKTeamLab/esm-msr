"""Core leak-overlap engine.

For every training inventory (a model or family of model checkpoints) and every benchmark
dataset, counts the test rows whose
  exact        : identical variant (same protein family + identical mutation set) was trained
  constituent  : every mutation in the row was trained somewhere in the same protein, but not
                 this exact combination (e.g. a test double whose two singles were trained)
  sameprot     : the protein is in training, none of its mutations were
  homolog      : a homolog (MMseqs E<=1e-3 or identity>=30% cov>=60%) of the protein is in training
  weak         : only a weak/ambiguous homology (identity 20-30% or E 1e-3..0.1)
  none         : no detected relation
"""
import os, sys, pickle, collections, json
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from lib import parse_mut_str

ROOT=os.environ.get('LEAK_ROOT','/home/sareeves/playground/esm-msr-devel'); D=os.path.join(ROOT,'leak_audit','data')
OUT=os.path.join(ROOT,'leak_audit','out'); os.makedirs(OUT, exist_ok=True)
fm=pickle.load(open(os.path.join(D,'family_maps.pkl'),'rb'))
maps, seqs, reps = fm['maps'], fm['seqs'], fm['reps']
# family of every pid + representative sequence
pid_fam={pid:v[0] for pid,v in maps.items()}
fam_repseq={}
for pid,(fam,mp,rs) in maps.items(): fam_repseq[fam]=rs

_pos_cache={}
def canon_one(pid, wt, pos, mt):
    """canonical (family, rep_pos, wt, mt) or None."""
    key=(pid,pos)
    if key in _pos_cache:
        rp=_pos_cache[key]
    else:
        fam, mp, rs = maps[pid]
        rp=mp.get(pos)
        if rp is not None and (rp<1 or rp>len(rs) or rs[rp-1]!=wt):
            alt=None
            for d in (1,-1,2,-2,3,-3):
                q=rp+d
                if 1<=q<=len(rs) and rs[q-1]==wt: alt=q; break
            rp=alt
        _pos_cache[key]=rp
    if rp is None: return None
    return (maps[pid][0], rp, wt, mt)

_parsed={}
def parse_muts(s):
    if s in _parsed: return _parsed[s]
    m=parse_mut_str(s) or []
    _parsed[s]=m; return m

class Inventory:
    def __init__(self, iid, description, provenance, variants=None, protein_pids=None, exact_status='verified'):
        self.iid=iid; self.description=description; self.provenance=provenance
        self.exact_status=exact_status
        self.variants=collections.defaultdict(set)      # family -> set(frozenset canonical keys)
        self.singles=collections.defaultdict(set)       # family -> set(single canonical keys)
        self.protein_pids=set(protein_pids or [])
        self.families=set()
        for pid in self.protein_pids:
            if pid in pid_fam: self.families.add(pid_fam[pid])
        if variants:
            for pid, mutstr in variants:
                if pid not in maps: continue
                muts=parse_muts(mutstr)
                if not muts: continue
                keys=set(); ok=True
                for wt,pos,mt in muts:
                    c=canon_one(pid,wt,pos,mt)
                    if c is None: ok=False; break
                    keys.add(c)
                if not ok: continue
                fam=maps[pid][0]; fs=frozenset(keys)
                self.variants[fam].add(fs)
                self.singles[fam].update(keys)
                self.families.add(fam)
        self.singles_fams=set(self.singles)

# ---------------------------------------------------------------- inventories
msv=pd.read_pickle(os.path.join(D,'megascale_variants.pkl'))
msv=msv[msv['mutation'].notna()].copy()
msv['n']=msv['mutation'].str.count(';')+1
sp=pickle.load(open(os.path.join(D,'capped_split_sets.pkl'),'rb'))
capped=sp['train']
lib=pd.read_pickle(os.path.join(D,'megascale_libraries.pkl'))

INV=[]
def ms_inv(iid, desc, codes, keep=None, prov='', exact_status='verified'):
    sub=msv[msv['code'].isin(codes)]
    if keep is not None: sub=sub[keep(sub)]
    variants=[('lib:'+c, m) for c,m in zip(sub['code'], sub['mutation'])]
    prot=[ 'lib:'+c for c in set(sub['code']) ]
    return Inventory(iid, desc, prov, variants=variants, protein_pids=prot, exact_status=exact_status)

INV.append(ms_inv('esm_msr_capped','ESM-MSR (released & devel), all substitutions of the 172 capped-train libraries',
                  capped, prov='splits_oct06_capped.pkl train + Tsuboyama2023 Dataset2+3'))
INV.append(ms_inv('me_stage1','MutateEverything stage 1 (singles), capped-train libraries', capped, keep=lambda d: d.n==1,
                  prov='capped train; ME recipe stage1 = singles'))
INV.append(ms_inv('me_stage2','MutateEverything stage 2 (singles then doubles), capped-train libraries', capped,
                  prov='capped train; stage2 initialised from stage1'))
INV.append(ms_inv('tmpnn','ThermoMPNN additive (singles), capped-train libraries', capped, keep=lambda d: d.n==1,
                  prov='capped train; train_single.yaml'))
INV.append(ms_inv('tmpnnd','ThermoMPNN-D (doubles), capped-train libraries', capped, keep=lambda d: d.n>=2,
                  prov='capped train; train_epistatic.yaml'))
INV.append(ms_inv('spurs_retrain_single','SPURS retrained (single model: single + native_cond rows), capped-train libraries', capped, keep=lambda d: d.n==1,
                  prov='capped train; datamodule items, singles'))
INV.append(ms_inv('spurs_retrain_multi','SPURS-multi retrained (initialised from the same seed\'s single model, then doubles), capped-train libraries', capped,
                  prov='capped train; datamodule items, doubles after the single-model initialisation'))
INV.append(ms_inv('esm3dg_released','ESM3dG (released augmented ensemble): documented K50dG+DMSv4/v5/v7',
                  list(lib.index), prov='HF/GitHub README: fine-tuned on K50dG+DMSv4/v5/v7; row-level mapping unverified',
                  exact_status='likely-unverified'))

# SPURS
spm=pd.read_csv(os.path.join(D,'spurs_single_train_proteins.csv'))
for iid,desc in [('spurs_single','SPURS single-mutant model (train+val libraries)'),
                 ('spurs_multi','SPURS multi-mutant model (train+val doubles)')]:
    sm=pd.read_csv(os.path.join(D,'spurs_multi_train_proteins.csv')) if iid=='spurs_multi' else spm
    use=sm[sm['retained']==True]
    muts=pd.read_csv(os.path.join(D,'spurs_train_mutations.csv.gz'))
    muts=muts[(muts.model==iid)]
    variants=[('lib:'+c, m) for c,m in zip(muts['code_wt'], muts['mutation'])]
    prot=['lib:'+c for c in use['code_wt'].unique()]
    INV.append(Inventory(iid, desc, 'SPURS_mod data (same Tsuboyama CSV, ThermoMPNN 8:1:1 split, m8 de-leak)',
                         variants=variants, protein_pids=prot))

# ESMtherm (protein-level only)
et=pd.read_csv(os.path.join(D,'esmtherm_train_proteins.csv'))
et_codes=set(et.loc[et['split'].isin(['train','val']),'code'].astype(str).str.replace('.pdb','',regex=False).str.replace('|','_',regex=False))
INV.append(Inventory('esmtherm','EsmTherm (K50_dG Dataset1+2; mutant-level 80/10/10, train+val domains)',
                     'datasets/split/wildtype_split.csv; mutation-level rows unavailable',
                     protein_pids=['lib:'+c for c in et_codes], exact_status='protein-only'))

# IFUM
ifum_csv=os.path.join(D,'ifum_train_proteins.csv')
if os.path.exists(ifum_csv):
    fi=pd.read_csv(ifum_csv)
    codes=set(fi.loc[fi['split'].isin(['train','val']),'code'].astype(str).str.replace('.pdb','',regex=False).str.replace('|','_',regex=False))
    INV.append(Inventory('ifum','IFUM (Megascale Dataset2; own clustered split)', 'IFUM published split',
                         protein_pids=['lib:'+c for c in codes], exact_status='protein-only'))
else:
    print('NOTE: ifum_train_proteins.csv not available yet')

# ESM-MSR-small (hyperopt) -- reported on external sets only, out of Oct-split scope
hyp=pd.read_pickle(os.path.join(ROOT,'repo','data','hyperopt_splits.pkl'))
hcodes=[x.replace('.pdb','').replace('|','_') for x in hyp['train']]
INV.append(ms_inv('esm_msr_small_hyperopt','ESM-MSR-small (March reference), hyperopt-train libraries', hcodes,
                  prov='hyperopt_splits.pkl train (legacy split; out of Oct-split audit scope)'))

# DDMut / GeoDDG / DDGemb are code-position inventories handled separately (see 11_code_inventories.py)
with open(os.path.join(D,'inventories_megascale.pkl'),'wb') as f: pickle.dump(INV,f)
print('inventories', [(i.iid, len(i.families), len(i.variants)) for i in INV])

# ---------------------------------------------------------------- family homology graph
edges=pd.read_pickle(os.path.join(D,'protein_homologs.pkl'))
fam_h=collections.defaultdict(set); fam_w=collections.defaultdict(set)
for a,b,f,e,cv,h,w in zip(edges.a,edges.b,edges['fident'],edges['evalue'],edges['cov'],edges['homolog'],edges['weak']):
    fa,fb=pid_fam.get(a),pid_fam.get(b)
    if fa is None or fb is None or fa==fb: continue
    if h: fam_h[fa].add(fb); fam_h[fb].add(fa)
    if w: fam_w[fa].add(fb); fam_w[fb].add(fa)

# ---------------------------------------------------------------- test sets (canonical, lazy)
all_inv_fams=set()
for inv in INV: all_inv_fams |= inv.families
_inv_hom=set(); _inv_weak=set()
for fam in all_inv_fams:
    _inv_hom |= fam_h.get(fam,set()); _inv_weak |= fam_w.get(fam,set())

def canon_testsets():
    """Only canonicalise mutations for proteins whose family is any inventory family;
    the rest can only be homolog/weak/none and stay unevaluated."""
    out={}
    for f in sorted(os.listdir(os.path.join(D,'testsets'))):
        ds=f[:-4]
        if ds.startswith('tsu_hyperopt'): continue
        d=pd.read_pickle(os.path.join(D,'testsets',f))
        fams=[pid_fam.get(pid) for pid in d['pid']]
        keys=[None]*len(d)
        for i,(pid,mut,fam) in enumerate(zip(d['pid'], d['mutation'], fams)):
            if fam is None or fam not in all_inv_fams: continue
            muts=parse_muts(mut); acc=set(); ok=True
            for wt,pos,mt in muts:
                c=canon_one(pid,wt,pos,mt)
                if c is None: ok=False; break
                acc.add(c)
            keys[i]=frozenset(acc) if ok else None
        d['family']=fams; d['keys']=keys
        out[ds]=d
        print('  canonicalised', ds, len(d), 'rows;', sum(1 for x in keys if x is not None), 'mutation-checked')
    return out
tests=canon_testsets()

def eval_inventory(inv, ds):
    d=tests[ds]; counts=collections.Counter()
    hom=fam_h_union(inv); wk=fam_w_union(inv)
    for fam,keys in zip(d['family'], d['keys']):
        if fam is None: counts['unmapped']+=1; continue
        if fam in inv.families:
            if keys is not None and keys:
                if keys in inv.variants.get(fam,()): counts['exact']+=1
                elif keys <= inv.singles.get(fam,set()): counts['constituent']+=1
                else: counts['sameprot']+=1
            else: counts['sameprot']+=1
        elif fam in hom: counts['homolog']+=1
        elif fam in wk: counts['weak']+=1
        else: counts['none']+=1
    return dict(n=len(d), **counts)

_hc={}; _wc={}
def fam_h_union(inv):
    if inv.iid not in _hc:
        s=set()
        for fam in inv.families: s |= fam_h.get(fam,set())
        _hc[inv.iid]=s
    return _hc[inv.iid]
def fam_w_union(inv):
    if inv.iid not in _wc:
        s=set()
        for fam in inv.families: s |= fam_w.get(fam,set())
        _wc[inv.iid]=s
    return _wc[inv.iid]

rows=[]
for inv in INV:
    for ds in tests:
        c=eval_inventory(inv, ds)
        rows.append(dict(inventory=inv.iid, dataset=ds, **c))
ov=pd.DataFrame(rows).fillna(0)
for c in ['exact','constituent','sameprot','homolog','weak','none','unmapped']:
    if c not in ov: ov[c]=0
ov['frac_exact']=ov['exact']/ov['n']; ov['frac_any_mut']=(ov['exact']+ov['constituent'])/ov['n']
ov.to_csv(os.path.join(OUT,'inventory_overlaps.csv'), index=False)
print(ov[ov.inventory.isin(['esm_msr_capped','spurs_single','spurs_multi','esm3dg_released','esmtherm'])]
      [['inventory','dataset','n','exact','constituent','sameprot','homolog','weak','none']]
      .to_string(index=False))
