# Sigma trade-off of the WT adapter on ProteinGym (baseline model, Oct 27)

Copied here from the cluster-results snapshot `msr_predictions/_not_in_benchmarks/results_cluster_snapshot_2026-10-27/pgym/esm-msr-baseline-w2a4-m16a16-capped/sigma_tradeoff_analysis/` (the snapshot is left untouched).
`REPORT.md` is the report, `scripts/` the pipeline (`bash scripts/run_all.sh`; it reads the per-assay `wt_only` predictions of the baseline model at sigma 0, 0.25, 0.5, 0.75, 1 from the snapshot
and writes `figures/` and `tables/` here; `common.py` holds the paths and the hand annotation of proteins with more than one assay: group A = different properties, B = conditions, C = replicates),
`tables/assay_classes.csv` the per-assay classes and `figures/` F01-F31 plus one dossier per multi-assay protein. The scripts were copied as they were and their output paths pointed at this folder;
they were not re-run after the move.

Model: `esm-msr-baseline-w2a4-m16a16-capped` (the released code retrained on the capped split), not FINAL_p0. The p0 version of the per-protein property comparison is in `../multi_property/`.
