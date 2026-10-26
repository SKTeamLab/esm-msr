import os, re, json
import pandas as pd, numpy as np
ROOT=os.environ.get('LEAK_ROOT','/home/sareeves/playground/esm-msr-devel'); W=os.path.join(ROOT,'repo/.claude/worktrees/devel-main-conservative-merge-5e554f')
DOC=os.path.join(W,'docs/BENCHMARKS_OCT2026.md')
OUT=os.path.join(ROOT,'leak_audit','out')
s=open(DOC).read()
tabs=open(os.path.join(OUT,'tables.md')).read().split('\n\n')
t11,c11,t12,c12,t13,c13,t14,c14=tabs

# ---------------------------------------------------------------- legend
old_legend_start=s.index('**Legend.**')
old_legend_end=s.index('\n\n', old_legend_start)
new_legend=("**Leak notation (§6).** Every number carries a superscript for the worst training-data overlap it has with its test set, "
 "and the full code sits in the matrix directly below each table. **ᴹ** identical mutation(s) in training; **ᵐ** the test variant's "
 "mutations were trained but not this exact combination (additive models, doubles made of trained singles); **ᴾ** same protein, other "
 "mutations; **ᴴ** strong homolog in training (MMseqs2 E ≤ 1e-3 over ≥ 50 % of the test protein); **ʰ** weak/ambiguous similarity "
 "(E ≤ 1e-2 or ≥ 30 % identity over ≥ 30 %); **ᴰ** documented training on the same source dataset, rows not reconstructable; "
 "**ᶜ** pretraining-corpus exposure only (WT sequence/structure, no mutation supervision); **#** the number is not a test result "
 "(the set is in training). A code such as `M.59 P.35 H.02` gives the fraction of dataset rows in each class, `M?` marks a "
 "probable but unverified identical-mutation overlap, and `V` marks a dataset used for validation or model selection. "
 "Superscripts are omitted below 0.5 % of rows; the companion matrix keeps them. **\\*** partial coverage (DDMut: S669 98 %, "
 "S461 97 %); **_italic_** a different variant set (Rosetta on ProteinGym: single mutants only, shown but never compared); "
 "**q** a run in progress or queued; **–** not run. The audit, the inventories and the assumptions are in §6; the hyperopt-split "
 "rows of §1.2 and §1.5 predate the Oct06 split and are out of its scope.")
s=s[:old_legend_start]+new_legend+s[old_legend_end:]

# ---------------------------------------------------------------- replace tables
blocks=[m for m in re.finditer(r'(?m)((?:^\|.*\|\n)+)', s)]
assert len(blocks)==14, len(blocks)
def wrap_ds(block):
    return '| | |\n|---|---|\n| **Leak code per cell** (same row/column order as above; see §6.2) |\n\n'+block
# add the two published-recipe rows that are not in the long table
t14=t14.replace('| ESM3-small-open, σ = 0 (zero-shot) | 0.448ᶜ | 0.621ᶜ | 0.470ᶜ |',
 '| ESM3-small-open, σ = 0 (zero-shot) | 0.448ᶜ | 0.621ᶜ | 0.470ᶜ |\n| ProteinMPNN, ProteinGym published recipe | 0.257ᶜ | 0.565ᶜ | 0.298ᶜ |\n| ESM3-open, ProteinGym published recipe | 0.466ᶜ | 0.641ᶜ | 0.486ᶜ |')
c14=c14.replace('| ESM3-small-open, σ = 0 (zero-shot) | C | C | C |',
 '| ESM3-small-open, σ = 0 (zero-shot) | C | C | C |\n| ProteinMPNN, ProteinGym published recipe | C | C | C |\n| ESM3-open, ProteinGym published recipe | C | C | C |')
note12='\n\n*The legacy hyperopt val/test columns are omitted here: those rows predate the Oct06 split and are out of this audit\'s scope (their values are in §1.5).*\n'
new_tables={0:t11+'\n\n'+c11, 1:t12+note12+c12, 2:t13+'\n\n'+c13, 4:t14+'\n\n'+c14}
# rebuild document table by table (reverse order to keep offsets)
for idx in sorted(new_tables, reverse=True):
    m=blocks[idx]; s=s[:m.start()]+new_tables[idx]+s[m.end():]

# ---------------------------------------------------------------- bullets
def para(start):
    i=s.index(start)
    cands=[x for x in (s.find('\n- **', i+1), s.find('\n\n', i)) if x!=-1]
    j=min(cands)
    return i,j
def repl_para(start, new):
    global s
    try:
        i,j=para(start)
    except ValueError:
        print('MISSING START:',start[:80]); return
    s=s[:i]+new+s[j:]

repl_para('- **On all 12 sets GeoDDG (0.767)', """- **The 12-set ranking is dominated by training overlap.** GeoDDG (0.767) and DDGemb (0.697) lead only on cells that are partly or wholly their training data: S8754 *is* GeoDDG's training set (`M 1.00`, `#`), S2648 is 92.5 % of DDGemb's S2450, and every one of GeoDDG's twelve cells carries an identical-mutation, same-protein or strong-homolog flag (§6.5). Over the cells with no such flag the 12-set mean is ESM-MSR (devel) 0.62, ME stage 1 0.56, ThermoMPNN 0.53, ProteinMPNN 0.46, ProSST 0.42, against 0.65 for ESM-MSR in the unannotated table. On the two sets that are same-protein-clean for every supervised model (S669, S461) ESM-MSR and DDGemb are level on S669 (0.565 vs 0.570) and ESM-MSR leads S461 (0.756 vs 0.654).""")
repl_para('- **On the six sets with no documented overlap for any model**', """- **There are no six sets with no overlap for any model.** The overlap lattice is measured, not assumed: S669 and S461 are the only sets with zero identical mutations for every model, yet DDMut still shares 11 %/16 % of them at the same-protein level (its S9028 includes S2648+FireProtDB proteins), and GeoDDG has a same-protein or strong-homolog trace on both (1 % of S669, 3 % of S461). S571 is 39 % identical-mutation overlap with GeoDDG and 20 % with DDMut; Ssym is 37 % with GeoDDG, 30 % with DDMut and 28 % with DDGemb; PTMUL is 59 % with DDMut's multi-mutant training file SM1242 and 97 % same-protein with GeoDDG/DDGemb. The old six-set ranking therefore compared different models on different subsets. The corrected per-model means over their own clean subsets are in §6.5; only ESM-MSR, ME, ThermoMPNN, ProSST, ProteinMPNN and the zero-shot control are clean on all six, and ESM-MSR (0.621) leads them.""")
repl_para('- **Among the models trained on Tsuboyama/Megascale, ESM-MSR leads 9 of 12 sets.**', """- **Among the models trained on Tsuboyama/Megascale, ESM-MSR leads 9 of 12 sets, but only two of those models are clean comparators.** SPURS trained on 24 of the 33 capped test libraries (21,315 of 36,526 singles, 58.4 % exact), ESMtherm on 27 of 33 (82 % of test rows same-protein, mutant-level split) and IFUM on 21 of 33 (45 % same-protein); ESM3ΔG is not Megascale-trained at all (see below). Only ESM-MSR, MutateEverything and ThermoMPNN have zero exact-mutation and zero same-protein overlap with all 33 test libraries (§6.5).""")
repl_para('- **SPURS (single model) is above ESM-MSR (devel) on capped test', """- **SPURS is not held out, and that explains the comparison.** SPURS's Megascale fit uses the ThermoMPNN 8:1:1 split, not ours: 24 of our 33 capped-test libraries and 21,315 of 36,526 singles are in its fit (58.4 % exact; 41.2 % of the doubles for the multi model), and its val is in the fit too. Its 0.866/0.844 is therefore not a held-out result. Under the audit it is `M.41 m.13 P.35` on test; the identical-mutation part alone is larger than any difference between the models.""")
repl_para('- **IFUM scored on a per-variant ESMFold structure', """- **IFUM is not held out either.** Its published split puts 21 of the 33 capped-test libraries in train and the other 12 in its unseparated val/test block; 23,518 test rows (45 %) are on proteins it trained on, and the mutations are the Megascale rows themselves. Both IFUM modes are `P`-flagged on the capped split, and its GB1 row is same-protein (IFUM trained on the Megascale 1PGA/GB1 library). Its per-variant ESMFold mode still beats its WT-backbone mode, but neither is a clean test.""")
repl_para('- **Mean over the 7 DMS**', """- **Mean over the 7 DMS** (models with all seven): ESM-MSR (devel) WT head 0.618, (devel) combined 0.609, SPURS 0.605, ESM-MSR (released) 0.601, ProSST 0.591, ME st1 additive 0.589, ESM3ΔG 0.589, ESM3-small-open σ = 0 0.580, ThermoMPNN 0.533, ProteinMPNN 0.514, SPURS multi 0.504, GeoDDG 0.386, DDGemb 0.281, ESMtherm 0.232. **The GB1 column is not clean for four of these:** SPURS (same-protein via a near-identical Megascale design), ESMtherm and IFUM (trained on the Megascale 1PGA/GB1 library) and GeoDDG (1,538 1PGA/GB1 single mutants in S8754; 61 of the Wu rows exact and 61,880 constituent); ESM-MSR/ME/ThermoMPNN are clean on it because GB1 was quarantined from the capped split. GB1 is the lowest of the seven for every model, so dropping it moves the means up by 0.00–0.07 and leaves the top three unchanged (ESM-MSR devel WT 0.660, devel 0.654, SPURS 0.651); ProSST moves from fifth to fourth, ahead of ESM-MSR (released).""")
repl_para('- **Domainome:**', """- **Domainome is not clean either.** Domainome's domain `O95155_PF04564_1229` (UBE4B) is the same protein as Megascale library 3L1X, which is in the capped train split: 1,134 of its 1,206 rows (0.21 % of Domainome) are identical mutations in ESM-MSR's, ME's and ThermoMPNN's training data, and 3,177 more rows are same-protein. SPURS adds 0.1 % exact. ESMtherm (15,335 rows), IFUM (16,991), DDGemb (P 3 %) and GeoDDG (P 5 %) are same-protein on other domains, and ESM3ΔG has strong homologs for 19.8 % of rows in its MGnify fine-tune. The 0.548 leaderboard is therefore an upper bound for several models; the fully memorised domain is 1 of 522 in a per-domain mean, so its direct effect on that mean is at most about 0.001 ρ, but its own score should be discarded.""")
repl_para('- **ProSST ties ESM-MSR on the average and the tie hides a split.**', """- **ProSST ties ESM-MSR on the average only because of 23 fully memorised DMS.** 23 of the 213 ProteinGym assays are Tsuboyama DMS whose every row is in the capped train split used by ESM-MSR, ME and ThermoMPNN: 45,623 of 2,446,730 rows (1.9 % of the benchmark, 10.8 % of the assays). They are all Stability assays and ESM-MSR scores 0.897 on them. Removing them (local devel-seed reruns, 190 DMS) moves ESM-MSR's DMS mean from 0.561 to 0.520, its Stability category mean from 0.880 to 0.871, and its PG average from 0.516 to 0.514. ProSST, zero-shot on those assays, is 0.528 on the all-213 DMS mean and 0.514 on the all-213 PG average; after removing the 23 assays ESM-MSR is at 0.520 and the two PG averages are no longer directly comparable (§6.5). The same 23 DMS are in the training data of ME stage 1/2 (24,252 exact + 21,331 constituent) and ThermoMPNN (24,252 exact singles), and SPURS is worse: 51 of 213 DMS have ≥ 50 % of their rows in its fit.""")
repl_para('- **The ‡ models are not a held-out comparison, and two of them look strong.**', """- **The Megascale-trained models are not a held-out comparison, and one of them should not have been flagged at all.** On test SPURS multi has 41.2 % of doubles exact, ESMtherm 82 % of rows same-protein (mutant-level split) and IFUM 45 % same-protein; all three are trained on the Megascale libraries this split uses. ESM3ΔG was labelled ‡ but its released weights are fine-tuned on the authors' own 960,216-sequence MGnify Stability set, not on Megascale: 0 of the 33 test libraries are in training, 8 have a strong homolog (40.7 % of test rows) and 12 a weak one. Its 0.811/0.811 is a genuine zero-shot result with a homology caveat, not a training leak.""")
repl_para('- **ProSST wins four of five categories outright**', """- **ProSST wins four of five categories outright** (Activity, Binding, Expression, Organismal fitness) against every ESM-MSR setting, not just σ = 1; the 0.22 it concedes on Stability is what brings the averages level (§1.4). **Under the audit the Stability comparison is the contaminated one:** all 23 fully trained DMS are Stability assays, so the category where ESM-MSR wins by 0.22 is the only one where it has direct training data. Removing them puts ESM-MSR's Stability at 0.871 and ProSST's 0.653 unchanged; the four-category lead is untouched and the global tie is restored without ESM-MSR's training advantage.""")

# ---------------------------------------------------------------- add audit notes before §2 and §3
s=s.replace('## 2. Epistasis: ESM-MSR vs the retrained baselines and the third-party models',
"""## 2. Epistasis: ESM-MSR vs the retrained baselines and the third-party models

**Leak note (§6).** The capped val/test rows used here are clean for ESM-MSR, MutateEverything and ThermoMPNN (identical mutations 0, same protein 0). SPURS multi is the only third-party model that predicts doubles directly *and* was trained on these libraries: 6,400 of its 15,545 test doubles are exact (41.2 %) and 1,717 more are constituent. ESMtherm is same-protein on 82 % of test rows and IFUM on 45 %; ESM3ΔG is zero-shot with 40.7 % strong/24.6 % weak homologs. Read the ‡ rows (documented Megascale training; §6.5 quantifies it) accordingly; the "Among models that did not train on Tsuboyama" comparison is the only like-for-like interaction comparison in this section.""")
s=s.replace('## 3. ProteinGym in detail',
"""## 3. ProteinGym in detail

**Leak note (§6).** 23 of the 213 assays are fully inside the capped train split (45,623 rows; all Stability); they are listed in `docs/benchmarks/leak_audit_pg_per_dms.csv` and marked in the per-model `M` fractions of §6.5. The σ sweep below is unaffected in ranking (a lower σ is better on exactly these Stability assays, so removing them strengthens rather than weakens the σ = 0.75 conclusion) but its absolute values are upper bounds for ESM-MSR. SPURS's 51/213 contaminated assays make its 0.483 worse than ESM-MSR's 0.510, not better.""")

# ---------------------------------------------------------------- new section 6
i=s.index('## 6. Where the code and results live')
# local §2 flag corrections: ‡ now means documented Megascale training, and ESM3dG is not one
s=s.replace('Models marked ‡ trained on Megascale,','Models marked ‡ (EsmTherm, IFUM, SPURS multi) trained on Megascale,')
s=s.replace('| ESM3ΔG ‡ |','| ESM3ΔG |')
section6=open(os.path.join(OUT,'section6.md')).read()
s=s[:i]+section6+'\n\n'+s[i:].replace('## 6. Where the code and results live','## 7. Where the code and results live',1)
open(DOC,'w').write(s)
print('written',len(s))
