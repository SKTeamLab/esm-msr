# Tsuboyama analyses on the mega_splits retrains

Every Tsuboyama (MegaScale) analysis of `analysis/`, rerun on the models retrained on the full MegaScale split (`mega_splits`, the split
the baselines' own papers use) instead of the Homology-Restricted one (`splits_oct06_capped`). Results: `analysis/results/03_tsuboyama/mega_splits/`,
one folder per shared script, named as in `analysis/results/`.

```bash
bash analysis/tsuboyama/mega_splits/run_all.sh        # PYTHON=... to pick the interpreter; rebuilds results/03_tsuboyama/mega_splits/ from scratch
```

## Models (three seeds each)

| Key | Model | Source |
|---|---|---|
| `msrp0:wt`, `msrp0:comb`, `msrp0:epi` | ESM-MSR = FINAL_p0_mega (the FINAL_p0 recipe trained on mega_splits) | testing queue, `final_bench/predictions/mega_splits-{val,test}/final_checkpoints/FINAL_p0_mega/s{1,2,3}/` |
| `me1`, `me2`, `me2:epi` | MutateEverything stage 1 / stage 2 | cluster `outputs/tracked/me_mega_st{1,2}_s{k}/results_mega_splits_allbl_{val,test,testcomp}.csv` |
| `tmpnn`, `tmpnnd`, `tmpnnd:epi` | ThermoMPNN / ThermoMPNN + ThermoMPNN-D | cluster `outputs/tracked/thermompnn_mega_s{k}/predictions_{val,test}.csv` (`pred_single_additive`, `pred_combined`) |
| `spurs`, `spursm`, `spursm:epi` | SPURS / SPURS multi | cluster `outputs/tracked/spurs_mega_s{k}/tsuboyama_{val,test}.csv` |

The cluster runs are collected to `<MSR_ROOT>/baselines/retrained/mega_splits/tracked/<run>/` (`registry.MEGA_RAW`). Epistasis
definitions match the capped split: ESM-MSR comb(AB) − WT(A) − WT(B); ME st2 minus the same seed's st1 singles (the test run's
`testcomp` file supplies the constituent singles that are not measured); ThermoMPNN-D and SPURS multi minus their own single-additive reading.
No zero-shot or public supervised model was scored on mega_splits, so those rows and panels are empty here.

## How it works

* `00_build_tasks.py` writes `_tasks/tsuboyama_mega_{val,test}.csv.gz`, the label tables in the capped tables' layout, from FINAL_p0_mega's
  prediction file (the seeds are checked to agree on rows and labels; `ref_seq` is rebuilt per library and reproduces the capped table's on 100% of rows).
  val: 39,252 rows (37,610 measured), 31 libraries; test: 53,981 rows (49,048 measured), 28 libraries (30 backbones).
* `MSR_TSU_SPLIT=mega` (set by `run_all.sh`): `registry.TSU` becomes `['tsuboyama_mega_val', 'tsuboyama_mega_test']`, `RESULTS` becomes
  `results/03_tsuboyama/mega_splits/`, `data.load_frame` reads FINAL_p0_mega under the `msrp0` keys (three seeds) and the mega baselines
  (`data._load_mega`), and `style.save` rewrites the capped split's name and "ESM-MSR (p0)" in figure text. Scripts drop what has no mega
  counterpart: the document's tables (06: 2.0–2.2, 2.5), the devel link and ctx control (06, 11d, 13), Rosetta checks (13), the
  external and Domainome screening (08).
* `01_rho_overview.py`: the `03_tsuboyama.py` figures for the mega models, plus `tsuboyama_capped_vs_mega` (each model on its own split's test set).
* Then the shared scripts: `06_epistasis`, `11d` + `11b` (the `epi_metrics` hierarchy, bootstrapped over domains), `11c`, `11f`, `12`, `13`, `08`, `15`, `17`.

## The mega_splits sets, and how they relate to the originals

`02_split_provenance.py` reproduces every number in this section (CSVs in `provenance/`; it needs the tables named in its docstring, two of which, the retraining
item tables and ME's cDNA files, are cluster copies). Inputs measured 2026-11-16.

### Where the split comes from

`data/mega_splits.pkl` is **byte-identical (md5 `acb76f92…`) to the file ThermoMPNN ships (`dataset_splits/mega_splits.pkl`) and to the copy in SPURS's data
directory**: 239 train / 31 validation / 28 test wild-type libraries, assigned by MMseqs2 sequence clustering ("splits produced by mmseqs clustering" in ThermoMPNN's loader).
It is the one split behind three of the four published comparisons:

| Work | Training / evaluation data | Relation to mega_splits |
|---|---|---|
| **ThermoMPNN** (singles; PNAS 2024) | `Tsuboyama2023_Dataset2_Dataset3_20230416.csv`, `ddG_ML`, insertions/deletions/`wt` rows dropped; singles of the split's libraries | the split's origin; test = 28 libraries. A row is kept only if its `WT_name` is exactly a name in the split |
| **ThermoMPNN-D** (doubles; bioRxiv 2024) | same file, doubles; "the splitting procedure previously described for ThermoMPNN"; train/val/test = 85,253 / 10,282 / 18,574 doubles over 90 / 17 / 20 proteins after removing homologs (> 25 % similarity) of PTMUL | same libraries; **its validation (10,282) and test (18,574) doubles equal our mega val and test double counts exactly**; its training set is 24,027 doubles smaller than ours (109,280), because of the PTMUL filter |
| **SPURS / SPURS multi** (Nat. Commun. 2026) | the same file and split ("the training, validation, and test splits (8:1:1 ratio) established in the ThermoMPNN study"); training sequences with > 25 % identity to the test or other benchmark sets are removed | same split file (md5 above). Upstream's removal is a hard-coded list of 64 libraries applied to train and val (see below) |
| **Mutate Everything** (NeurIPS 2023) | **its own** cDNA split: `cdna_train` (116 proteins, 213,798 rows: 116,688 singles + 97,110 doubles), `cdna1_test` (19 proteins, 19,645 singles) and `cdna2_test` (18 proteins, 22,913 doubles); "we follow the train-val split introduced in" Stability Oracle and filter training proteins similar to the evaluation sets | **not mega_splits** (next paragraph) |

**Mutate Everything does not use the ThermoMPNN split.** Mapping its cDNA proteins onto mega_splits: `cdna_train` holds 75 mega train, 14 mega val, **13 mega test** and 14 unlisted
proteins, and both its evaluation sets (`cdna1_test`, `cdna2_test`) lie **entirely inside the mega train set** (19 / 19 and 18 / 18 proteins). Its published numbers are therefore not on
the ThermoMPNN test set and cannot be compared with ours. On the retraining question: the ThermoMPNN-D paper says ME was "obtained from their respective GitHub repositories" (no
retraining stated) and that **ThermoMPNN was retrained on ME's cDNA2 split**, the reverse direction; it also compares ME on a 5-fold cross-validation of its own Megascale-D
(127,476 doubles, 153 proteins) "on matching splits" (SCC 0.55; additive ThermoMPNN 0.59) without saying ME was retrained there. I found no source for ME being retrained on mega_splits,
so our ME mega retrain is our own experiment, not a replication.

### The test set, against the capped split's

| | mega_splits | capped (Homology-Restricted) |
|---|---|---|
| Libraries train / val / test | 239 / 31 / 28 | 172 / 23 / 33 |
| Test backbones (wild type + mutant background) | 30 | 37 |
| Test rows (uncensored) | 53,981 (49,048) | 52,071 (48,740) |
| Singles / doubles | 31,614 / 22,367 | 36,526 / 15,545 |
| Natural domains | 21 | 19 |
| De novo, Rocklin designs (`HHH_`, `HEEH_`) | 4 | 12 |
| De novo, trRosetta hallucinations | 3 | 1 |
| Redesign of a natural domain (`v2_2LC2`) | 0 | 1 |
| Mutant-background (destabilized) backbones | 2 | 7 |

Only six libraries are in both test sets (1GYZ, 2K5H, 2KVS, 3DKM, HEEH_KT_rd6_0793, HHH_rd1_0244). **13 of the 28 mega test libraries are in the capped *training* set** (1PSE, 1QP2,
2K1B, 2K28, 2L33, 2LQK, 4C26, 6EWS, 6EWT, 6EWU and the three hallucinations), so a capped-split model has seen them; likewise the capped test libraries were mega training data.
("Natural" only means a PDB-style code; whether any such entry is itself a deposited design was not checked.)

**De novo:** seven test libraries (HEEH_KT_rd6_0746, HEEH_KT_rd6_0793, HHH_rd1_0142, HHH_rd1_0244, r11_1081, r12_757 and r18_3 of the trRosetta set). **Destabilized backbones:**
two, both mutant backgrounds of 2K5H: G11S (ΔG 1.81, 1,149 uncensored rows) and G23A (ΔG 1.60, 1,153) against 4.74 kcal/mol for the 2K5H wild type. Wild-type ΔG runs 1.29 to 4.74 in the
mega test (1.01 to 4.74 in the capped one), so there is no unstable wild type.

### Did our retraining add proteins or backbones the originals did not use?

**Libraries: no.** Train, validation and test hold exactly the split's 239 / 31 / 28 wild-type libraries; none is missing and none is extra.

**Mutant-background libraries: yes, and this is the one real difference.** The raw Tsuboyama file has 75 libraries named `<pdb>.pdb_<mutation>` (a destabilized variant of a wild-type
structure; `provenance/mutant_background_libraries.csv`). The ThermoMPNN and SPURS loaders keep a row only if `WT_name` is exactly a name in the split, and the split lists none of them, so
**the originals trained and tested on none**. Our retraining tables follow the capped-split pipeline, which includes a mutant-background library when its parent is in the split:

* **train: +3,538 singles** from four libraries of two train proteins (1H8K_R44S, 1H8K_W36A, 1OPS_T53S, 1OPS_V44A, trained on the parent's AlphaFold backbone with the background residue substituted);
  the six `pross` libraries of 2B88 and 2HBB have parents in train but contributed no rows;
* **val: none;**
* **test: +2,302 singles** (2K5H_G11S, 2K5H_G23A), which the original 28-library test did not contain.

Row reconciliation (`provenance/row_reconciliation.csv`; "upstream" = the raw file filtered as the upstream loaders do, before their further filters):

| Part | upstream singles | our singles | upstream doubles | our doubles |
|---|---|---|---|---|
| train | 215,731 | 219,269 (= upstream + 3,538 mutant background) | 125,511 | 109,280 |
| val | 27,328 | 27,328 | 11,459 | 10,282 |
| test | 28,172 | 30,474 (= upstream + 2,302 mutant background) | 22,081 | 18,574 |

Singles match upstream exactly apart from the mutant backgrounds. The doubles differ because our tables keep only doubles whose two singles are measured and uncensored (and, for training, that
ESM-MSR would regress on), the ESM-MSR scoring rule; the 18,574 test and 10,282 validation doubles are the same counts ThermoMPNN-D reports. Our training doubles (109,280) sit between
ThermoMPNN-D's 85,253 and SPURS multi's reported 122,278.

**Upstream SPURS trained on less than our SPURS.** Upstream drops a hard-coded list of 64 libraries (`remove_homologs_from_scaffold`; the paper's purpose is to drop > 25 % identity homologs of the test
and independent sets) from train and val: 40 of the 239 train libraries (36,962 of our 219,269 singles, 17 %; 14,879 of 109,280 doubles) and 6 of 31 validation libraries. Our SPURS retrain reads the prepared item tables
(`ItemsTableDataset` in `spurs_retrain.patch`) instead of `mega_splits.pkl`, so it **did not apply this list** and trained on all 239. The same holds for our ThermoMPNN, ThermoMPNN-D and ME mega
retrains, which share those tables (ThermoMPNN-D additionally lacks the PTMUL filter).

### Is the test set held out from train and val under our (sequence + structure) rule?

**No.** With the capped split's rule (MMseqs2 E ≤ 1e-3, Foldseek E ≤ 1e-3, identity > 0.40 or TM-score > 0.70 by the shorter chain; the all-vs-all library table the capped split was built from):

| Split | test–train homologous pairs | test libraries affected | test–val pairs |
|---|---|---|---|
| capped | 0 | 0 / 33 | 0 |
| mega_splits | **70** (48 sequence, 22 structure, 30 identity > 0.4, 37 TM > 0.7) | **16 / 28** | 5 (4 libraries) |

The worst cases: 1TUC and 2KXD (identity 0.75 and 0.73 to train library 2CDT), 2K1B and 2K28 (TM 0.91 and 0.85 to 2N88) and the hallucinated designs r18_3 and r12_757 (0.57, 0.55 to r11_951).
Twelve libraries are clean (`provenance/mega_test_library_homology.csv`). **Upstream SPURS's removal list does not fix this**: 24 of the 75 test–(train + val) pairs involve a library on the
list, and after removing those, 51 pairs remain, linking **13 of 28** test libraries, with identities up to 0.71 (1TUC, 2KXD) and TM up to 0.91 (2K1B). The SPURS paper describes the filter
as removing train/val sequences with > 25 % identity to the test split or other independent sets; the hard-coded list the code applies does not achieve that against this test set (the
shipped `mmseq_mut_search_0.25.m8` file may be used elsewhere; I did not establish that). **mega_splits results are the baselines' own-split comparison, not a held-out generalization test.**

### Do our results agree with the published ones?

| Quantity | Published | Ours (mega retrain) |
|---|---|---|
| SPURS, median per-protein Spearman, 28 test proteins (28,312 mutants) | 0.83 | **0.817** (28,172 WT-backbone singles; mean 0.795) |
| ThermoMPNN, same statistic | 0.77 (SPURS paper) | **0.773** (mean 0.768) |
| ThermoMPNN-D, Spearman pooled over the 18,574 test doubles | 0.53 ± 0.02 (PCC 0.52 ± 0.02) | **0.529** |
| ME | no result on this split | st1 0.779 / st2 0.769 median on singles; 0.545 pooled on doubles (st2) |
| SPURS and ThermoMPNN test rows | 28,312 mutants (third-party summary of the SPURS paper) | 28,172 (−140, unexplained) |

SPURS is 0.013 below its published median, ThermoMPNN and ThermoMPNN-D match. Both of the SPURS-side differences run against a simple "more data helps" reading (we trained on 46 more libraries than upstream and
lost 0.013), so a smaller deviation in the SPURS retrain recipe is more likely than a data effect: ours restores the ProteinMPNN `v_48_020` initialisation the public code no longer loads
(`../../../baselines/retrained/spurs/README.md`) and uses early stopping with patience 30. The three medians use wild-type-backbone singles only; adding the 2,302 mutant-background singles
moves them by at most 0.006 (SPURS 0.817 to 0.817, ThermoMPNN 0.773 to 0.779, ESM-MSR WT head 0.846 to 0.848), so the mutant backgrounds do not drive any comparison.

### How SPURS was retrained: the adapter question (frozen random vs trained)

No training code for the published SPURS is available, so the retrain follows the public repository plus two findings from the retraining work (hand-off notes in `baselines/retrained/spurs/docs/`; cluster
workspace `~/spurs_retrain`, logs in `SPURS/logs/{oct06,mega}`):

* **Pretrained ProteinMPNN initialisation.** The public code stopped loading `v_48_020.pt`; the released checkpoint's ProteinMPNN is cos 0.9996 to it. Without it a retrain peaks at validation 0.74 against the
  released 0.809 (epoch-0 validation 0.54–0.60 against 0.76–0.77). All our retrains restore it.
* **The structural adapter (ESM layer 30, `adapter_layer_indices: [-3]`) of the released checkpoint was never trained.** Before the August 2025 code change (`cc854c1`) the forward pass ran the plain ESM layers under
  `no_grad`, which also cut the gradient path to an adapter placed before them: the released run's optimizer state has 126 tensors against 142 for a run of the current code, the 16 missing being the adapter, so the adapter keeps its random
  initialisation (inference from the optimizer state; the multi checkpoint's adapter equals the single's, cos 1.0000). `model.legacy_esm_forward=true` reproduces that forward (adapter keeps its random init,
  adapter layer in `train()` mode, other layers in `eval()` under `no_grad`); `false` is the current public forward, which trains the adapter. The released checkpoint scores the same under both forwards at inference
  (validation differs by < 0.001).
* **The frozen-adapter choice** (`legacy_esm_forward=true`, the `leg_*` runs) is therefore the retrain that matches how the published model was trained, and it is the recipe of every SPURS number in the BENCHMARKS
  tables and in this folder. The trained-adapter variant was tried as a control (`pub_*` and `ta_*` runs; `ta_idx1_seed1` moves the adapter to ESM layer −1).

What the two variants did where they were compared:

| Split | Run | Adapter | Best validation (pooled Spearman) | Notes |
|---|---|---|---|---|
| capped | `leg_seed1/2/3` | frozen | 0.790 (seed 1) | the headline triplicate: Tsuboyama val 0.803 ± 0.002, test 0.801 ± 0.004 |
| capped | `pub_seed1/2/3` | trained | 0.786 / 0.792 / 0.789 | `pub_seed2/3` were cancelled mid-training; only `pub_seed1` was scored (val 0.798, test 0.799; 12-set mean 0.586, DMS 0.583, Domainome 0.494 against 0.588, 0.578, 0.489) |
| capped | `ta_seed2/3`, `ta_multi_seed1/2/3` | trained | 0.792 / 0.795 (single) | trained by 2026-10-29; no scores in BENCHMARKS or the hand-off (I found none), so treat as **unscored** |
| capped | `s45_pretrained_legacy` / `_public` | frozen / trained | 0.791 / 0.790 | same seed 45, otherwise identical: no difference |
| **mega_splits** | `leg_seed1/2/3` (our retrains) | frozen | 0.804 / 0.813 / 0.811 (multi 0.669–0.697) | every `spurs_mega_*` number in this folder |
| **mega_splits** | `mega/r1_pretrained_public` | trained | 0.803 at epoch 15, still rising when cancelled at epoch 16 | upstream data loader on `mega_splits.pkl`, pretrained MPNN, seed 45; **never tested** |
| mega_splits | `mega/r2_pretrained_legacy` | frozen | 0.785 after epoch 1 | cancelled after one epoch, as were `r0` and `r3` (random ProteinMPNN) |

So the trained adapter has **not** been tested properly on mega_splits: no completed run, no test evaluation, no multi model. On the capped split it was indistinguishable from the frozen adapter within seed noise, one scored
seed. The released SPURS multi reaches validation 0.749 (116 steps per epoch for 22 epochs, doubles only), against 0.669–0.697 for our frozen-adapter multi retrain, which is the largest gap to the published model.
**Queued 2026-11-16** (gpu4, three paired seeds, same recipe as `leg_*` with `model.legacy_esm_forward=false`): trained-adapter single `spurs_ta_s{1,2,3}` (jobs 389770, 389773, 389776), multi `spurs_tam_s*`
(389771, 389774, 389777) and the val/test-only tracked test `spurs_mega_ta_s*` (389787, 389788, 389789; outputs `outputs/tracked/spurs_mega_ta_s{1,2,3}`).

**Which SPURS to present (manuscript).** The released weights were trained by code that no longer exists in the public repository, so the public repository's current forward (trained adapter) and the
released checkpoint are different models. Our frozen-adapter retrain already matches the released validation score (0.804 / 0.813 / 0.811 against 0.809), which points to the frozen recipe as the faithful one;
whether it also reproduces the published *test* result (median per-protein Spearman 0.83 against our 0.817) is exactly what the trained-adapter runs and the public-checkpoint run are for. Do not choose a version
before they land. Decision rule once they do: present the recipe whose single- and multi-mutant predictions are closest to the public checkpoint's on the mega_splits test set (per-library ρ, and the correlation of
the predictions variant by variant), and state the adapter setting in the methods.

**Standing rule: the public SPURS checkpoint is always included in mega_splits analyses.** It appears to have been trained on exactly this split (the released run is 212 steps per epoch for 25 epochs, which is the
mega_splits training set of 239 libraries minus 27 without a wild-type row, selected on the split's validation set; the checkpoint reproduces its stored validation score of 0.8088 exactly, with pooled train / val / test
Spearman 0.908 / 0.809 / 0.741 and per-protein means 0.898 / 0.827 / 0.797). Its test libraries are therefore held out by the split, though not homology-clean (see above), and its training set is the 64-library-filtered
one. **Queued 2026-11-16:** `spurs_mega_public` (job 389783, `outputs/tracked/spurs_mega_public`; single and multi, tracked slot, same harness as `spurs_mega_s*`). Still to do when it lands: collect it into
`baselines/retrained/mega_splits/tracked/`, give it keys in `msr_analysis.data._load_mega` (one checkpoint, so one seed) and the registry, add it to `01_rho_overview.py` and to the epistasis, screening and gain-fit
panels, and add the per-variant agreement of the public checkpoint with each retrain (frozen, trained adapter) to this README.

### Published checkpoints trained on these splits (searched 2026-11-16)

**Scope.** These comparisons only need the Tsuboyama **validation and test** sets of mega_splits: they exist to show the effect of our altered retraining procedure on the baselines' own split. Whether a model generalizes is
tested on the capped splits, not here. The queued runs therefore score val and test only (`TSUBOYAMA_ONLY=1` in `inference_scripts/baselines/slurm_inference.sh`; no external sets, DMS, Domainome or ProteinGym).

| Model | Released checkpoints | Trained on | Usable as the "original" for our retrains? |
|---|---|---|---|
| **SPURS** | HF checkpoint `luo-group/SPURS` (single and multi) | mega_splits train, selected on val (see above) | **Yes**: queued (`spurs_mega_public`, job 389783) |
| **ThermoMPNN** (PNAS 2024) | `models/thermoMPNN_default.pt` in `Kuhlman-Lab/ThermoMPNN` ("best ThermoMPNN model trained on Megascale training dataset"); the split files are in its `dataset_splits/` | mega_splits | The original PNAS model; not downloaded here |
| **ThermoMPNN / ThermoMPNN-D v2** | `Kuhlman-Lab/ThermoMPNN-D` `model_weights/`: `ThermoMPNN-ens{1,2,3}.ckpt` (single) and `ThermoMPNN-D-ens{1,2,3}.ckpt` (doubles), committed 2024-08-01; on the cluster in `~/software/ThermoMPNN-D/model_weights` | the repo's `train_single.yaml` / `train_epistatic.yaml`: `megascale`, `mega_splits.pkl`, `splits: [train, val]` (singles; doubles plus augmented doubles), 100 epochs, lr 1e-5. The checkpoints themselves embed no hyper-parameters, so this rests on the configs and the paper | **Yes**: queued as three ensemble members (see below) |
| **Mutate Everything** | four checkpoints on Box (ESM2 and AlphaFold, each full and "single only"); `mutate_everything_{af,esm,af_old}.pth` are on the cluster in `~/software/MutateEverything/models` | **its own cDNA split** (`cdna_train`); the README mentions no Megascale, ThermoMPNN or mega_splits data; the repository has 7 commits and no releases | **No.** No ME checkpoint trained on mega_splits exists in any release I could find |

The ThermoMPNN-D Zenodo record (13345274, "Double mutant stability data and single mutant structures") holds only data archives (`Megascale.tar.gz`, `DMS`, `PTMUL-D`, `Megascale-Rosetta-Sweeps`), no checkpoints or split
files (archive contents not opened). ThermoMPNN-D's own code does carry hooks for ME's cDNA split (`train_cdna`, `test_cdna`, `cdna2_test_ThermoMPNN.csv`), which matches the paper: the ThermoMPNN group retrained **ThermoMPNN on ME's
split**, not ME on theirs. ME's training set contains 13 of the 28 mega test proteins and its evaluation sets lie inside the mega train set, so even the public ME checkpoints cannot be scored on mega_splits as a clean comparison
(running them would give a labelled cDNA-trained reference only; not queued).

**What the published ThermoMPNN checkpoints isolate.** They were trained by the authors with the same code as our `tmpnn` / `tmpnnd` retrains, but on mega_splits **train + val** with no validation selection, no mutant-background
libraries, and the paper's doubles set, whereas ours train on train only, select on val, add the 3,538 mutant-background singles and use 109,280 doubles. Scoring the three ensemble members on our test tables (single and epistatic
pairs `ens1-3` as three "seeds", `outputs/tracked/thermompnn_mega_public_ens{1,2,3}`, jobs 389784–389786, val and test only) shows how much those data differences move the numbers; compare per-library
ρ, doubles ρ and the 2,302 mutant-background test rows separately.

### What I could not verify

* The ThermoMPNN paper's own text (PNAS 2024 / bioRxiv) was not retrievable here (403); ThermoMPNN's split procedure rests on its repository, on its loader and on the SPURS and ThermoMPNN-D papers' descriptions. The 0.77 for
  ThermoMPNN is SPURS's, not ThermoMPNN's own figure.
* The SPURS median comes from Tamarind's summary page of the paper; the paper itself was not opened.
* The 140-row gap in single-mutant test rows (28,172 against 28,312) is not explained.
* Whether Mutate Everything was ever retrained on mega_splits or on another split by the ThermoMPNN group (the recollection that prompted this section) is not supported by anything I could read.

Sources: ThermoMPNN split file https://github.com/Kuhlman-Lab/ThermoMPNN/tree/main/dataset_splits; SPURS code and data layout https://github.com/luo-group/SPURS; Mutate Everything (Ouyang-Zhang et al., NeurIPS 2023, Appendix A.1) https://arxiv.org/abs/2310.12979 and
its data https://github.com/jozhang97/MutateEverything; ThermoMPNN-D (Dieckhaus and Kuhlman 2024) https://www.biorxiv.org/content/10.1101/2024.08.20.608844 (read through Europe PMC, PMC11370451); SPURS summary https://www.tamarind.bio/tools/spurs.

## Results (test set; seed means; bootstrap B = 200 for the ρ overview, 300 for the epistasis hierarchy)

**These are not held-out numbers**: 16 of the 28 test libraries have a training homolog under the capped split's rule (see above), and the test set includes two destabilized
backgrounds the original ThermoMPNN/SPURS test did not.

**ddG ranking** (mean per-library Spearman ρ, `03_tsuboyama/table_mega_split.md`):

| Model | val | test | test singles | test doubles | same model, Homology-Restricted test |
|---|---|---|---|---|---|
| ESM-MSR combined | 0.868 | 0.827 | 0.830 | 0.662 | 0.845 |
| ESM-MSR WT head | 0.845 | 0.812 | 0.821 | 0.569 | 0.827 |
| SPURS | 0.813 | 0.799 | 0.796 | 0.567 | 0.801 |
| ThermoMPNN + D | 0.788 | 0.779 | 0.769 | 0.615 | 0.784 |
| ThermoMPNN | 0.782 | 0.777 | 0.769 | 0.557 | 0.794 |
| MutateEverything st1 | 0.790 | 0.774 | 0.775 | 0.509 | 0.771 |
| SPURS multi | 0.768 | 0.771 | 0.738 | 0.598 | 0.772 |
| MutateEverything st2 | 0.781 | 0.767 | 0.758 | 0.595 | 0.760 |

ESM-MSR combined is above every baseline in every paired library bootstrap draw and higher on 24–27 of the 28 test libraries. Training on
all of MegaScale moves no model by more than 0.02 against its Homology-Restricted score (different test sets, 28 vs 33 libraries).
Pooled over libraries (not the benchmark metric), ESM-MSR ranks doubles worst (ρ 0.42 vs 0.52–0.55, `06_epistasis/doubles_pooled_vs_per_library.csv`):
its per-library offsets are less consistent than the baselines'.

**Epistasis hierarchy** (`epi_metrics`, 16,923 test doubles with both singles measured in 20 domains; [95% domain bootstrap], `06_epistasis/fig5_*`):

| Model | naive ρ | beyond ρ | pair ρ | line ρ | cell rank | cell mag |
|---|---|---|---|---|---|---|
| ESM-MSR | 0.507 [0.41, 0.65] | 0.455 [0.27, 0.60] | 0.560 | 0.308 [0.25, 0.39] | 0.108 [0.07, 0.16] | 0.158 [0.10, 0.22] |
| ME stage 2 | 0.480 [0.40, 0.57] | 0.362 [0.21, 0.52] | 0.417 | 0.248 [0.21, 0.30] | 0.076 [0.05, 0.10] | 0.121 [0.07, 0.15] |
| SPURS multi | 0.397 [0.30, 0.50] | 0.309 [0.15, 0.45] | 0.394 | 0.289 [0.22, 0.35] | 0.049 [0.02, 0.08] | 0.039 [−0.02, 0.09] |
| ThermoMPNN-D | 0.354 [0.21, 0.52] | 0.305 [0.14, 0.48] | 0.418 | 0.215 [0.14, 0.28] | 0.016 [0.00, 0.03] | 0.003 [−0.03, 0.05] |

ESM-MSR has the highest point value at every level on test, and at every level but pair ρ on validation (val pair ρ 0.219 vs 0.299
for ME st2, both intervals spanning ~0 to 0.6; val cell rank 0.206 vs 0.090). The intervals overlap at the coarse levels; the finest
(line, cell) levels separate ESM-MSR from SPURS multi and ThermoMPNN-D. On doubles clear of
the assay floor (additive dG > 1, `06_epistasis/dddG_scatters/stats.csv`) ESM-MSR keeps ρ 0.571 while ME st2 falls to 0.267,
ThermoMPNN-D to 0.197 and SPURS multi to −0.067: the baselines' epistasis signal on mega_splits is mostly the floor. The measured −dG_add
control alone reaches naive ρ 0.618, above every model.

**Screening** (Tsuboyama test, Cys excluded, K = 10 per protein, `08_screening/screening_per_protein_*.csv`): SPURS leads on hit rate
(0.42) and rectified gain (5.6 kcal/mol singles; ESM-MSR 5.3, ME 4.5, ThermoMPNN 4.5); ESM-MSR ties ME on P(≥ 1 hit) (0.885).
The gain per variant still follows a − b ln K, but less tightly than on the capped split (rmse 0.018 vs 0.004 for ESM-MSR, `17_gain_fit/`).
