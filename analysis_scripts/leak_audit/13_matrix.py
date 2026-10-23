"""Assemble the per-(model,dataset) leak matrix with the audit notation."""
import os, sys, json, pickle, re
import pandas as pd, numpy as np
ROOT=os.environ.get('LEAK_ROOT','/home/sareeves/playground/esm-msr-devel'); D=os.path.join(ROOT,'leak_audit','data'); OUT=os.path.join(ROOT,'leak_audit','out')
long=pd.read_csv(os.path.join(ROOT,'repo/.claude/worktrees/devel-main-conservative-merge-5e554f/docs/benchmarks/all_by_all_long.csv'))
ov=pd.read_csv(os.path.join(OUT,'inventory_overlaps.csv'))
hom=pd.read_csv(os.path.join(OUT,'homology_classes.csv'))
cex=pd.read_csv(os.path.join(OUT,'code_inventory_exact.csv'))
MODEL_INV={
 'ESM-MSR (released)':'esm_msr_capped','ESM-MSR (released) WT':'esm_msr_capped','ESM-MSR (released) MT':'esm_msr_capped',
 'ESM-MSR (released) ε=0.25':'esm_msr_capped','ESM-MSR (released) ε=0.5':'esm_msr_capped','ESM-MSR (released) ε=0.75':'esm_msr_capped',
 'ESM-MSR (devel)':'esm_msr_capped','ESM-MSR (devel) WT':'esm_msr_capped','ESM-MSR (devel) MT':'esm_msr_capped',
 'ME stage1':'me_stage1','ME stage1+':'me_stage1','ME stage2':'me_stage2','ME stage2+':'me_stage2',
 'TMPNN':'tmpnn','TMPNN+D':'tmpnnd','SPURS':'spurs_single','SPURS multi':'spurs_multi',
 'ESMtherm':'esmtherm','DDGemb':'ddgemb_s2450','DDMut':'ddmut_s9028','DDMut [partial]':'ddmut_s9028',
 'GeoDDG-3D':'geoddg_s8754','IFUM (WT backbone)':'ifum','IFUM (native, ESMFold)':'ifum',
 'ESM3dG':'esm3dg_mgnify','ESM-MSR-small (March, crystal)':'esm_msr_small_hyperopt',
 'ProSST':'corpus','ProteinMPNN':'corpus','Rosetta':'none',
 'ESM3-small zero-shot':'corpus','ESM3-medium zero-shot':'corpus','ESM3-large zero-shot':'corpus',
 'ESM3-small-open (ε=0, MSR readout)':'corpus','ESM3-small-open (ε=0, MSR readout) WT':'corpus',
 'ESM3-small-open (ε=0, MSR readout) MT':'corpus',
}
DS={'s571':'s571','s783':'s783','s2648':'s2648','s8754':'s8754','s669':'s669','s461':'s461','ssym':'ssym',
 'q3421':'q3421','k3822':'k3822','k2369':'k2369','ptmul':'ptmul','ptmuld':'ptmuld',
 'tsu_capped_val':'tsu_capped_val','tsu_capped_test':'tsu_capped_test',
 'DLG4_HUMAN_Faure_2021_abundance_domain':'DLG4 abund.','DLG4_HUMAN_Faure_2021_binding_domain':'DLG4 bind.',
 'GRB2_HUMAN_Faure_2021_abundance_domain':'GRB2 abund.','GRB2_HUMAN_Faure_2021_binding_domain':'GRB2 bind.',
 'MYO_HUMAN_Kung_2025_display':'MYO','ESTA_BACSU_Nutschel_2020_dTm':'ESTA','GB1_Wu_2016_binding_domain':'GB1',
 'domainome':'domainome','PG avg':'proteingym','PG Stability':'proteingym','PG DMS mean':'proteingym',
 'tsu_hyperopt_val':'HYPEROPT','tsu_hyperopt_test':'HYPEROPT'}
def eng(inv,ds):
    r=ov[(ov.inventory==inv)&(ov.dataset==ds)]
    if len(r): return r.iloc[0]
def hm(inv,ds):
    r=hom[(hom.inventory==inv)&(hom.dataset==ds)]
    if len(r): return r.iloc[0]
def cx(inv,ds):
    r=cex[(cex.inventory==inv)&(cex.dataset==ds)]
    if len(r): return r.iloc[0]
def nrows(ds):
    p=os.path.join(D,'testsets',ds+'.pkl')
    return len(pd.read_pickle(p)) if os.path.exists(p) else None
NAT={}
def natural_fraction(ds):
    if ds not in ('tsu_capped_test','tsu_capped_val'): return 1.0
    if ds in NAT: return NAT[ds]
    d=pd.read_pickle(os.path.join(D,'testsets',ds+'.pkl'))
    code=d['pid'].str.replace('tsu:','',regex=False)
    nat=code.map(lambda c: bool(re.match(r'^[0-9][A-Za-z0-9]{3}$',str(c))) or str(c).startswith('v2_'))
    NAT[ds]=float(nat.mean()); return NAT[ds]

rows=[]
for _,r in long.iterrows():
    model=r['model']; dn=r['dataset']; inv=MODEL_INV.get(model); ds=DS.get(dn)
    if not inv or not ds: continue
    rec=dict(model=model,dataset=dn,rho=r['rho'],inventory=inv,n=None,exact=0,constituent=0,P=0,H=0,h=0,
             C=0.0,D=0,U=0,V=0,bad=0,note='')
    if ds=='HYPEROPT':
        rec['note']='legacy hyperopt split — out of scope'; rec['code']='—'; rows.append(rec); continue
    n=nrows(ds); rec['n']=n
    if inv in ('esm_msr_capped','me_stage1','me_stage2','tmpnn','tmpnnd','spurs_single','spurs_multi','esm3dg_released','esm_msr_small_hyperopt'):
        e=eng(inv,ds); hh=hm(inv,ds)
        if e is not None:
            rec['exact']=int(e['exact']); rec['constituent']=int(e['constituent']); rec['P']=int(e['sameprot'])
        if hh is not None:
            rec['H']=int(hh['n_H']); rec['h']=int(hh['n_h'])
            rec['P']=max(rec['P'], int(hh['n_sameprot'])-rec['exact']-rec['constituent'])
    elif inv in ('esmtherm','ifum','esm3dg_mgnify'):
        hh=hm(inv,ds)
        if hh is not None:
            rec['P']=int(hh['n_sameprot']); rec['H']=int(hh['n_H']); rec['h']=int(hh['n_h'])
    elif inv in ('ddgemb_s2450','ddmut_s9028','geoddg_s8754'):
        hh=hm(inv,ds)
        if hh is not None:
            rec['P']=int(hh['n_sameprot']); rec['H']=int(hh['n_H']); rec['h']=int(hh['n_h'])
        c=cx(inv,ds)
        if c is not None:
            rec['exact']=int(c['exact']); rec['constituent']=int(c['constituent']) if pd.notna(c.get('constituent')) else 0
            rec['P']=max(0, rec['P']-rec['exact']-rec['constituent'])
        if inv=='ddmut_s9028' and ds in ('ptmul','ptmuld'):
            c2=cx('ddmut_sm1242',ds)
            if c2 is not None:
                rec['exact']=int(c2['exact']); rec['P']=max(0, rec['P']-rec['exact'])
        if inv=='geoddg_s8754' and ds=='s8754':
            rec['exact']=n; rec['P']=0; rec['bad']=1; rec['note']='the test set IS the training set'
        if inv=='ddgemb_s2450' and ds=='s2648': rec['D']=1
        if inv=='ddmut_s9028' and ds in ('s2648','s8754'): rec['D']=1
    elif inv=='corpus':
        rec['C']=natural_fraction(ds)
    if ds=='tsu_capped_val' and inv not in ('corpus','none'): rec['V']=1
    rows.append(rec)
mx=pd.DataFrame(rows)

def f(x,n):
    if not n: return ''
    v=x/n
    if v==0: return '.000'
    if v<0.001: return '<.001'
    if v<0.01: return f'{v:.3f}'.lstrip('0')
    if v>=0.995: return '1'
    return f'{v:.2f}'.lstrip('0')
def code_for(r):
    n=r['n'] or 0
    parts=[]
    if r['inventory'] in ('esmtherm','ifum') and r['P']>0:
        m_est=int(r['P']* (0.8 if r['inventory']=='esmtherm' else 1.0))
        parts.append('P'+f(r['P'],n)); parts.append('M?'+f(m_est,n))
    else:
        if r['exact']>0: parts.append('M'+f(r['exact'],n))
        if r['constituent']>0: parts.append('m'+f(r['constituent'],n))
        if r['P']>0: parts.append('P'+f(r['P'],n))
    if r['H']>0: parts.append('H'+f(r['H'],n))
    if r['h']>0: parts.append('h'+f(r['h'],n))
    if r['V']: parts.append('V')
    if r['D']: parts.append('D')
    if r['C']: parts.append('C'+('' if r['C']>=0.995 else f(r['C'],1)))
    if r['bad']: parts.append('#')
    return ' '.join(parts) if parts else '–'
mx['code']=mx.apply(code_for,axis=1)
def worst(r):
    n=r['n'] or 1
    if r['bad']: return '#'
    if r['inventory'] in ('esmtherm','ifum') and r['P']>0: return 'ᴾ'
    for k,v in [('M',r['exact']),('m',r['constituent'])]:
        if v>0: return {'M':'ᴹ','m':'ᵐ'}[k]
    for k,v in [('P',r['P']),('H',r['H']),('h',r['h'])]:
        if v/n>=0.005: return {'P':'ᴾ','H':'ᴴ','h':'ʰ'}[k]
    if r['D']: return 'ᴰ'
    if r['C']: return 'ᶜ'
    return ''
mx['sup']=mx.apply(worst,axis=1)
mx.to_csv(os.path.join(OUT,'leak_matrix.csv'), index=False)
print(mx[['model','dataset','rho','n','exact','constituent','P','H','h','code','sup']].to_string(index=False))
