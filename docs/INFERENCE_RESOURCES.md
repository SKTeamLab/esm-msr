# Inference cost of every model on the full benchmark

One tracked run per model type, each scoring every benchmark set: Tsuboyama validation and test, the 12 external sets, the
7 DMS sets, Domainome and all 217 ProteinGym assays. Each run used the same fixed slot on one RTX A4500 (20 GB).

## Protocol

- **Slot:** 1 GPU (NVIDIA RTX A4500, 20 GB), 6 CPUs and 60 GB host RAM on the `gpu4` node. This is one quarter of the node,
  set by `inference_scripts/sbatch_fixed_slot.sh` with `STRICT` on. Every run passed the start-of-job slot check: visible GPUs,
  CPU affinity and cgroup memory limit, recorded in `resources_run.json`. Other jobs shared the node (three at the start and
  end of each run). They held their own GPUs, CPUs and memory, so they share only the node's I/O and bus.
- **Job:** `inference_scripts/baselines/slurm_inference.sh`, one model per job, from the cluster checkout of `devel-revision`
  at the time. Tracked runs ran 2026-10-28 to 2026-10-31; the two pending rows were submitted 2026-11-13.
- **What is measured:** `inference_scripts/resource_tracker.py` records one scope per protein. *Inference h* is the sum of
  the protein scopes, so model loading and data handling are excluded. Model-load time is in the CSV column `model_load_s`.
  Peak RAM is the host RSS high-water mark; peak GPU memory is the device high-water mark over the run's proteins.
- **Batching:** calibrated per-dataset sizes from `inference_scripts/baselines/batch_sizes_gpu4.json`, with ProteinGym sized
  per assay (`--batch_mode table`). MutateEverything has no variant batching (one protein per forward pass).

<!-- BEGIN resource_summary.py --docs -->
| Model | Run | Job | Inference h | Tsuboyama h | External h | DMS h | Domainome h | ProteinGym h | Proteins (failed) | Peak RAM GB | Peak GPU GB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ESM-MSR, WT + MT passes | dc8 (devel) seed 1 | 387814 | 34.82 | 0.13 | 0.17 | 0.41 | 0.93 | 33.18 | 2538 (4) | 7.1 | 19.5 |
| ESM-MSR, WT head only | fl0 (PG WT-only) (until esm_msr_final_p0_wt_s1 lands) | 387960 | 1.67 | 0.13 | 0.17 | 0.41 | 0.93 | 0.03 | 2538 (4) | 7.1 | 19.5 |
| MutateEverything stage 2 | seed 1 | 387860 | 7.53 | 0.04 | 3.24 | 0.01 | 0.23 | 4.00 | 2312 (2) | 11.7 | 19.7 |
| MutateEverything stage 1 | pending | gpu4 job 388811 | — | — | — | — | — | — | — | — | — |
| ThermoMPNN + ThermoMPNN-D | seed 1 | 387802 | 6.56 | 0.09 | 0.03 | 0.45 | 0.30 | 5.69 | 2875 (0) | 15.2 | 18.2 |
| SPURS + SPURS-multi | retrained, seed 1 | 387798 | 0.29 | 0.00 | 0.06 | 0.00 | 0.02 | 0.20 | 2542 (0) | 9.6 | 10.0 |
| ProteinMPNN | zero-shot | 387855 | 1.81 | 0.00 | 0.02 | 0.01 | 0.01 | 1.77 | 2542 (0) | 4.5 | 17.7 |
<!-- END resource_summary.py --docs -->

## Reading the table

- **ESM-MSR rows.** *WT + MT passes* is the deployed predictor: an MT pass for every variant, plus the WT pass. On ProteinGym
  each variant needs its own forward pass over sequences up to thousands of residues, which is most of the run (about 33 of
  35 h, ~20 variants/s). The *WT head only* row reads the WT adapter alone (`--skip_reverse`), which scores every single
  mutant of a protein from one forward pass. Until FINAL_p0's all-WT run (gpu4 job 388812) lands, this row is fl0 seed 1:
  WT head only on ProteinGym (0.03 h for all 217 assays), both passes on the other sets.
- **Rows differ between models.** Each model scores the rows its own scripts produce: ThermoMPNN counts extra DMS passes;
  MutateEverything stage 1 scores singles only and has no K2369 or PTMUL-D. Compare speed per dataset with
  `rows_per_s` in `analysis/results/inference_resources/per_group.csv`, not the row totals.
- **Peak GPU memory is not each model's requirement** for ESM-MSR, ThermoMPNN and ProteinMPNN. Their batch sizes were
  calibrated to fill the card to 90 %, so the peak reflects the batching policy. The MutateEverything and SPURS peaks are
  close to what those models need.
- **Failures** are proteins whose inference raised.
  - ESM-MSR: four ProteinGym assays longer than the card allows (BRCA1_HUMAN_Findlay_2018, BRCA2_HUMAN_Erwood_2022_HEK293T,
    POLG_CXB3N_Mattenberger_2021, SCN5A_HUMAN_Glazer_2019). The ProteinGym tables drop these four for every model.
  - MutateEverything stage 2: BRCA1_HUMAN and POLG_CXB3N. BRCA2 never loaded ("muts too far away"), so it is not counted.
- **Replicates agree.** Seeds and checkpoints of one architecture differ by about 1 % in inference time (ESM-MSR ×6,
  SPURS ×4, MutateEverything stage 2 ×2 on this card). That is why one run per type is listed. Every run is in
  `analysis/results/inference_resources/per_run.csv`.
- **Excluded.** Runs on other hardware are not in this table. That covers MutateEverything stage 1 and stage-2 seed 3 on an
  RTX 6000 (24 GB), and the RTX 5090 runs in `inference_scripts/baselines/README.md`. A one-of-each RTX 5090 set with a
  32 GB batch table is queued locally.

## Reproduce

```bash
# mirror every tracked run from the cluster into <MSR_ROOT>/resources_tracked/<label>/<run>/
MSR_CLUSTER=<user>@<gpu4 host> MSR_CLUSTER_HOME=<cluster home> MSR_NOTEBOOK_OUT_DIR=<cluster home>/esm-msr-devel/analysis_notebooks \
  python analysis/scoring/collect_predictions.py run --groups resources
# summarise every run, and refresh the table above and docs/benchmarks/inference_resources_a4500.csv
python analysis/scoring/resource_summary.py --docs
```

A new tracked run needs a label in `LABELS` of `analysis/scoring/resource_summary.py`. To make it the run a table row
uses, put it first in that row's candidate list in `DOC_ROWS`.
