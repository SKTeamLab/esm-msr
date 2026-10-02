#!/usr/bin/env python3
"""Build per-DMS esm-msr input CSVs for all runnable ProteinGym DMS.

Reads the official ProteinGym substitution benchmark release (Zenodo record
15293562, DOI 10.5281/zenodo.15293562 -- ProteinGym v1.3, 217 DMS) and maps
every DMS mutation onto the coordinates of its AlphaFold2 structure,
producing one input CSV per DMS plus a ``manifest.csv`` consumed by
``inference_scripts/esm_msr_testing.py --protein_gym``.

Expected layout under ``--proteingym_dir`` (as extracted from the release
zips; both flat and one-level-nested extraction layouts are auto-detected):

    DMS_substitutions.csv            metadata: one row per DMS
    ProteinGym_AF2_structures/       AF2 PDB structures (199 files)
    DMS_ProteinGym_substitutions/    one CSV per DMS (mutant, DMS_score, ...)

Mapping (empirically validated on the full 217-DMS release):
  - ProteinGym ``mutant`` positions are 1-based in ``target_seq``.
  - Structure sequence = ProteinChain.from_pdb(af2, 'A', is_predicted=True).sequence
  - identical:        struct_pos = target_pos
  - af2_subset (af2 in target, 0-based offset o in target): struct_pos = target_pos - o
  - target_subset (target in af2, 0-based offset o in af2):  struct_pos = target_pos + o
  - positional:       same length, >=90% identity (rare sequence revisions)

Usage:
    python preprocessing/pgym_preprocess.py \
        --proteingym_dir /path/to/ProteinGym --out pgym_inputs

The output directory then contains ``<DMS_id>.csv`` (columns: pdb_file,
code, chain, mut_type_renumbered, mutant, DMS_score) and ``manifest.csv``
(one row per runnable DMS with mapping mode, offsets, and mapped/unmapped
counts). ``pdb_file`` in the per-DMS CSVs is written *relative to the
ProteinGym root* (e.g. ``ProteinGym_AF2_structures/1A0F.pdb``); the
``manifest.csv`` records that root in a ``proteingym_dir`` column so the
output directory is portable across machines. DMS whose structure or
substitution file is missing, or whose sequence cannot be mapped, are
skipped and listed in the console summary.
"""
import argparse
import os
import re
import sys

import pandas as pd

from esm.utils.structure.protein_chain import ProteinChain

TOK_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def _detect_dir(candidates):
    """Return the first candidate directory that exists, or None."""
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proteingym_dir", required=True,
                    help="Root of the ProteinGym release (contains "
                         "DMS_substitutions.csv).")
    ap.add_argument("--out", default="pgym_inputs",
                    help="Output directory for per-DMS CSVs + manifest.csv "
                         "(default: pgym_inputs).")
    ap.add_argument("--subs_dir", default=None,
                    help="Directory of per-DMS substitution CSVs. Default: "
                         "auto-detected under --proteingym_dir "
                         "(DMS_ProteinGym_substitutions, flat or nested).")
    ap.add_argument("--af2_dir", default=None,
                    help="Directory of AF2 PDB files. Default: "
                         "<proteingym_dir>/ProteinGym_AF2_structures "
                         "(flat or nested).")
    args = ap.parse_args()

    pg = os.path.abspath(args.proteingym_dir)
    meta_path = os.path.join(pg, "DMS_substitutions.csv")
    if not os.path.exists(meta_path):
        sys.exit(f"error: {meta_path} not found (is --proteingym_dir correct?)")

    subs_dir = args.subs_dir or _detect_dir([
        os.path.join(pg, "DMS_ProteinGym_substitutions",
                     "DMS_ProteinGym_substitutions"),
        os.path.join(pg, "DMS_ProteinGym_substitutions"),
    ])
    if subs_dir is None or not os.path.isdir(subs_dir):
        sys.exit("error: per-DMS substitution CSVs not found; pass --subs_dir")
    af2_dir = args.af2_dir or _detect_dir([
        os.path.join(pg, "ProteinGym_AF2_structures",
                     "ProteinGym_AF2_structures"),
        os.path.join(pg, "ProteinGym_AF2_structures"),
    ])
    if af2_dir is None or not os.path.isdir(af2_dir):
        sys.exit("error: AF2 structure directory not found; pass --af2_dir")
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    meta = pd.read_csv(meta_path)
    seq_cache = {}

    def struct_seq(pdb_file):
        if pdb_file not in seq_cache:
            p = os.path.join(af2_dir, pdb_file)
            if not os.path.exists(p):
                seq_cache[pdb_file] = None
            else:
                try:
                    seq_cache[pdb_file] = ProteinChain.from_pdb(
                        p, "A", is_predicted=True).sequence
                except Exception as e:
                    print(f"  !! failed to load {pdb_file}: {e}", file=sys.stderr)
                    seq_cache[pdb_file] = None
        return seq_cache[pdb_file]

    manifest = []
    total_mapped = 0
    total_in = 0
    runnable = 0
    skipped = []

    for _, m in meta.iterrows():
        did = m["DMS_id"]; ts = str(m["target_seq"]); pdb = m["pdb_file"]
        ssub = os.path.join(subs_dir, did + ".csv")
        spath = os.path.join(af2_dir, pdb)
        pdb_rel = os.path.relpath(spath, pg)   # portable, resolved by the runner
        if not os.path.exists(ssub) or not os.path.exists(spath):
            skipped.append((did, "missing file", None, 0, 0)); continue
        sseq = struct_seq(pdb)
        if sseq is None:
            skipped.append((did, "no structure seq", pdb, 0, 0)); continue

        # determine mapping
        if sseq == ts:
            mode, off = "identical", 0
        elif len(sseq) == len(ts):
            match = sum(a == b for a, b in zip(sseq, ts)) / len(ts)
            if match >= 0.9:
                mode, off = "positional", 0   # same length, few subs (e.g. P53 pos72)
            else:
                skipped.append((did, f"fuzzy/unmapped match={match:.2f}", pdb, 0, 0)); continue
        elif sseq in ts:
            mode, off = "af2_subset", ts.find(sseq)
        elif ts in sseq:
            mode, off = "target_subset", sseq.find(ts)
        else:
            skipped.append((did, "fuzzy/unmapped", pdb, 0, 0)); continue

        df = pd.read_csv(ssub)
        n_in = len(df)
        out_rows = []
        n_unmapped = 0; n_wtmis = 0
        Ls = len(sseq); Lt = len(ts)
        for _, r in df.iterrows():
            mut = str(r["mutant"])
            toks = mut.split(":")
            built = []
            ok = True
            for t in toks:
                mm = TOK_RE.match(t)
                if not mm:
                    ok = False; break
                wt, pos, mt = mm.group(1), int(mm.group(2)), mm.group(3)
                if mode in ("identical", "positional"):
                    spos = pos
                    valid = 1 <= spos <= Ls
                elif mode == "af2_subset":
                    spos = pos - off
                    valid = (off + 1) <= pos <= (off + Ls)
                else:  # target_subset
                    spos = pos + off
                    valid = (1 <= pos <= Lt) and (spos <= Ls)
                if not valid:
                    ok = False; break
                cw = sseq[spos - 1]
                if cw != wt:
                    n_wtmis += 1
                built.append(f"{cw}{spos}{mt}")
            if not ok:
                n_unmapped += 1
                continue
            out_rows.append({
                "pdb_file": pdb_rel, "code": did, "chain": "A",
                "mut_type_renumbered": ":".join(built),
                "mutant": mut, "DMS_score": r["DMS_score"],
            })
        n_mapped = len(out_rows)
        total_mapped += n_mapped; total_in += n_in; runnable += 1
        if n_mapped > 0:
            out = pd.DataFrame(out_rows)
            out.to_csv(os.path.join(out_dir, did + ".csv"), index=False)
        manifest.append({
            "DMS_id": did, "pdb_file": pdb, "map_mode": mode, "offset": off,
            "seq_len_target": Lt, "seq_len_struct": Ls,
            "n_mutants_in": n_in, "n_mapped": n_mapped,
            "n_unmapped": n_unmapped, "n_wt_mismatch": n_wtmis,
            "proteingym_dir": pg,
        })

    man = pd.DataFrame(manifest)
    man.to_csv(os.path.join(out_dir, "manifest.csv"), index=False)
    print(f"subs_dir: {subs_dir}")
    print(f"af2_dir:  {af2_dir}")
    print(f"out_dir:  {out_dir}")
    print(f"Runnable DMS: {runnable}")
    print(f"Total mutants in: {total_in}")
    print(f"Total mapped:     {total_mapped}")
    print(f"Skipped DMS: {len(skipped)}")
    for d in skipped:
        print(f"   SKIP {d[0][:40]:42s} {d[1]}")
    print(f"\nMap mode counts:")
    print(man["map_mode"].value_counts().to_string())
    print(f"\nwt_mismatch total: {man['n_wt_mismatch'].sum()}")
    print(f"unmapped total:    {man['n_unmapped'].sum()}")
    print(f"Top DMS by n_mapped:")
    print(man.sort_values('n_mapped', ascending=False)
          [['DMS_id', 'n_mapped', 'map_mode', 'seq_len_struct']].head(10)
          .to_string(index=False))


if __name__ == "__main__":
    main()
