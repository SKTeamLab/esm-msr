# vim: set expandtab shiftwidth=4 softtabstop=4:
"""Pure-Python/pandas parsing, path resolution, and validation helpers for ChimeraX-ESM-MSR."""

from collections import defaultdict
from dataclasses import dataclass, field
import os
import re
import shutil
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Anchored PDB mutation token: WT amino acid, optional negative sign + digits, optional insertion code, MT amino acid.
MUT_TOKEN_RE = re.compile(r"^([A-Za-z])(-?\d+)([A-Za-z]?)([A-Za-z])$")
TQDM_PCT_RE = re.compile(r"(\d{1,3})%\|")

SCORE_COLUMNS = ("combined_pred", "wt_lora_pred", "mt_lora_pred")
EPISTASIS_COLUMNS = ("epi_pred", "combined_dddg_pred")

PosKey = Tuple[int, str]
ResKey = Tuple[str, PosKey]


def normalize_icode(icode: Optional[str]) -> str:
    """Normalize a PDB insertion code to '' or uppercase single letter."""
    if not icode:
        return ""
    return str(icode).strip().upper()


def format_res_pos(pos_key) -> str:
    """Format a residue position key (int, str) or int for ChimeraX specs (e.g. -2, 15, 15A)."""
    if isinstance(pos_key, tuple):
        num, icode = pos_key
        ic = normalize_icode(icode)
        return f"{int(num)}{ic}"
    return str(pos_key)


def parse_mut_string(m_str) -> List[dict]:
    """Parse a ':'-delimited PDB mutation string into structured token dicts.

    Supports negative residue numbers (e.g. 'G-2A') and insertion codes (e.g. 'A15AC').
    Returns an empty list if the string is empty/NaN or if ANY token is malformed so
    a corrupted double mutant is never silently downgraded into a single mutant.
    """
    if m_str is None or (isinstance(m_str, float) and np.isnan(m_str)):
        return []
    raw = str(m_str).strip()
    if not raw or raw.lower() == "nan":
        return []

    tokens = raw.split(":")
    muts: List[dict] = []
    for token in tokens:
        t = token.strip()
        if not t:
            return []
        match = MUT_TOKEN_RE.match(t)
        if not match:
            return []
        wt, num_str, icode, mt = match.groups()
        num = int(num_str)
        ic = normalize_icode(icode)
        muts.append({
            "wt": wt.upper(),
            "pos": num,
            "icode": ic,
            "pos_key": (num, ic),
            "mut": mt.upper(),
        })
    return muts


def deduplicate_columns_case_insensitive(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Safely normalize DataFrame column names to lowercase stripped strings.

    If case-duplicate columns exist (e.g. both 'CHAIN' and 'chain'), keeps the exact
    lowercase match when present (otherwise the first occurrence) so df['chain'] is
    always a 1D Series instead of a 2D DataFrame.
    """
    chosen_indices: Dict[str, int] = {}
    chosen_is_exact: Dict[str, bool] = {}
    duplicate_names: List[str] = []

    for idx, raw_col in enumerate(df.columns):
        raw_str = str(raw_col)
        stripped = raw_str.strip()
        norm = stripped.lower()
        is_exact = (stripped == norm)

        if norm not in chosen_indices:
            chosen_indices[norm] = idx
            chosen_is_exact[norm] = is_exact
        else:
            if norm not in duplicate_names:
                duplicate_names.append(norm)
            if is_exact and not chosen_is_exact[norm]:
                chosen_indices[norm] = idx
                chosen_is_exact[norm] = True

    ordered_pairs = sorted(chosen_indices.items(), key=lambda kv: kv[1])
    keep_idx = [idx for _, idx in ordered_pairs]
    new_names = [name for name, _ in ordered_pairs]

    out = df.iloc[:, keep_idx].copy()
    out.columns = new_names
    return out, duplicate_names


def resolve_user_path(path_str: str, base_repo_path: str = "") -> str:
    """Resolve a user-supplied path; relative paths resolve against base_repo_path when set."""
    cleaned = (path_str or "").strip()
    if not cleaned:
        return ""
    expanded = os.path.expanduser(cleaned)
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    if base_repo_path and base_repo_path.strip():
        base = os.path.expanduser(base_repo_path.strip())
        return os.path.normpath(os.path.abspath(os.path.join(base, expanded)))
    return os.path.normpath(os.path.abspath(expanded))


def resolve_python_command(env_spec: str, base_repo_path: str = "") -> Tuple[Optional[str], List[str], Optional[str]]:
    """Resolve Python Env setting into (program, base_args, error_message).

    Accepts:
      1. Direct path to a python executable file (checked first).
      2. Path to a venv or conda environment directory.
      3. Conda environment name (invoked via 'conda run --no-capture-output -n <name> python -u').
      4. Empty string -> system 'python' (or 'python3') with '-u'.
    Always preserves '-u' for unbuffered stdout streaming.
    """
    cleaned = (env_spec or "").strip()
    if not cleaned:
        py = shutil.which("python") or shutil.which("python3")
        if not py:
            return None, [], "No Python Env specified and 'python'/'python3' was not found on PATH."
        return py, ["-u"], None

    expanded = os.path.expanduser(cleaned)
    candidate_paths = [expanded]
    if not os.path.isabs(expanded) and base_repo_path and base_repo_path.strip():
        candidate_paths.append(os.path.join(os.path.expanduser(base_repo_path.strip()), expanded))

    # 1. Direct path to executable file
    for cand in candidate_paths:
        if os.path.isfile(cand):
            return os.path.normpath(os.path.abspath(cand)), ["-u"], None

    # 2. Directory containing a Python environment
    for cand in candidate_paths:
        if os.path.isdir(cand):
            exe_candidates = [
                os.path.join(cand, "python.exe"),
                os.path.join(cand, "Scripts", "python.exe"),
                os.path.join(cand, "bin", "python"),
                os.path.join(cand, "bin", "python3"),
            ]
            pyexe = next((p for p in exe_candidates if os.path.isfile(p)), None)
            if pyexe:
                return os.path.normpath(os.path.abspath(pyexe)), ["-u"], None
            return (
                None,
                [],
                f"Python Env directory '{cand}' exists, but no python executable was found inside it "
                f"(checked bin/python, bin/python3, Scripts/python.exe, python.exe).",
            )

    # 3. Looks like a filesystem path that does not exist
    if os.sep in cleaned or (os.altsep and os.altsep in cleaned) or cleaned.startswith((".", "~")):
        return None, [], f"Python Env path does not exist: '{cleaned}'."

    # 4. Conda environment name
    conda_exe = shutil.which("conda")
    if not conda_exe:
        return (
            None,
            [],
            f"Python Env '{cleaned}' is not a local path and 'conda' was not found on PATH.",
        )
    return conda_exe, ["run", "--no-capture-output", "-n", cleaned, "python", "-u"], None


def parse_tqdm_progress(line: str) -> Optional[int]:
    """Extract integer percentage (0..100) from a tqdm status line, or None."""
    if not line:
        return None
    m = TQDM_PCT_RE.search(line)
    if m:
        pct = int(m.group(1))
        if 0 <= pct <= 100:
            return pct
    return None


@dataclass
class ParsedCSVData:
    filepath: str
    ok: bool = False
    error: str = ""
    warnings: List[str] = field(default_factory=list)
    available_score_cols: Set[str] = field(default_factory=set)
    available_epi_cols: List[str] = field(default_factory=list)
    effective_score_col: str = ""
    auto_fallback_score_col: Optional[str] = None
    auto_fallback_mode: Optional[str] = None
    effective_epi_col: Optional[str] = None
    chains_in_csv: List[str] = field(default_factory=list)
    selected_chain: str = ""
    chain_matched: bool = True
    n_singles_total: int = 0
    n_doubles_total: int = 0
    n_singles: int = 0
    n_doubles: int = 0
    n_positions: int = 0
    score_min: Optional[float] = None
    score_max: Optional[float] = None
    residue_scores_data: Dict[ResKey, Tuple[float, str]] = field(default_factory=dict)
    all_singles_scores: Dict[Tuple[str, PosKey, str], float] = field(default_factory=dict)
    epistasis_df: Optional[pd.DataFrame] = None


def parse_esm_msr_csv(
    filepath: str,
    requested_score_col: str = "combined_pred",
    requested_epi_col: str = "epi_pred",
    mode: str = "Singles",
    target_chain: str = "",
    exclude_wt_aa: Optional[Set[str]] = None,
    exclude_mt_aa: Optional[Set[str]] = None,
    allow_mode_fallback: bool = True,
) -> ParsedCSVData:
    """Parse an ESM-MSR prediction CSV safely across pandas 2.x and 3.x.

    Handles:
      - Case-duplicate columns
      - Negative residue indices and PDB insertion codes
      - Missing combined_pred/mt_lora_pred (--skip_reverse auto-fallback to wt_lora_pred)
      - Both epi_pred and combined_dddg_pred epistasis columns
      - Non-finite (NaN/Inf) filtering while preserving valid 0.0 scores
      - Target chain filtering with informative reporting
    """
    result = ParsedCSVData(filepath=filepath, selected_chain=(target_chain or "").strip())

    if not filepath or not os.path.isfile(filepath):
        result.error = f"CSV file not found: '{filepath}'."
        return result

    try:
        raw_df = pd.read_csv(filepath)
    except Exception as e:
        result.error = f"Failed to read CSV '{os.path.basename(filepath)}': {e}"
        return result

    if raw_df.empty:
        result.error = f"CSV '{os.path.basename(filepath)}' is empty."
        return result

    df, dup_cols = deduplicate_columns_case_insensitive(raw_df)
    if dup_cols:
        result.warnings.append(
            f"Resolved case-duplicate CSV column(s): {', '.join(dup_cols)}."
        )

    if "mut_type_pdb" not in df.columns:
        result.error = (
            "Missing required column 'mut_type_pdb'. Ensure your CSV was generated with ESM-MSR inference.py."
        )
        return result

    if "chain" not in df.columns:
        result.error = "Missing required column 'chain' in CSV."
        return result

    # Inspect which score and epistasis columns exist with at least one finite numeric value.
    numeric_cols: Dict[str, pd.Series] = {}
    for col in SCORE_COLUMNS:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            if bool(np.isfinite(vals.to_numpy(dtype=float, na_value=np.nan)).any()):
                result.available_score_cols.add(col)
                numeric_cols[col] = vals

    for col in EPISTASIS_COLUMNS:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            if bool(np.isfinite(vals.to_numpy(dtype=float, na_value=np.nan)).any()):
                result.available_epi_cols.append(col)
                numeric_cols[col] = vals

    if not result.available_score_cols:
        result.error = (
            "No usable numeric score columns found in CSV "
            f"(expected at least one of {', '.join(SCORE_COLUMNS)})."
        )
        return result

    # Parse mutation strings & chain IDs cleanly (pandas 3.0 safe, no chained assignment).
    parsed_list = [parse_mut_string(v) for v in df["mut_type_pdb"].tolist()]
    mut_lens = np.array([len(m) for m in parsed_list], dtype=np.int64)
    chain_list = [str(c).strip() if pd.notna(c) else "" for c in df["chain"].tolist()]

    n_invalid_muts = int((mut_lens == 0).sum())
    if n_invalid_muts > 0:
        result.warnings.append(
            f"Skipped {n_invalid_muts} row(s) with unparseable 'mut_type_pdb' values."
        )

    valid_mut_mask = mut_lens > 0
    if not valid_mut_mask.any():
        result.error = "No parseable mutations found in 'mut_type_pdb'."
        return result

    # Record available chains across all valid rows.
    chains_seen = sorted({chain_list[i] for i in range(len(chain_list)) if valid_mut_mask[i] and chain_list[i]})
    result.chains_in_csv = chains_seen
    result.n_singles_total = int((mut_lens == 1).sum())
    result.n_doubles_total = int((mut_lens == 2).sum())

    # Resolve effective score column (auto-fallback for --skip_reverse CSVs).
    if requested_score_col in result.available_score_cols:
        eff_score_col = requested_score_col
    elif "wt_lora_pred" in result.available_score_cols:
        eff_score_col = "wt_lora_pred"
        result.auto_fallback_score_col = eff_score_col
        result.warnings.append(
            f"'{requested_score_col}' not present in CSV; automatically using 'wt_lora_pred'."
        )
    else:
        eff_score_col = next(c for c in SCORE_COLUMNS if c in result.available_score_cols)
        result.auto_fallback_score_col = eff_score_col
        result.warnings.append(
            f"'{requested_score_col}' not present in CSV; automatically using '{eff_score_col}'."
        )
    result.effective_score_col = eff_score_col

    # Resolve effective epistasis column (prefer requested if present, else epi_pred, else combined_dddg_pred).
    if requested_epi_col in result.available_epi_cols:
        result.effective_epi_col = requested_epi_col
    elif "epi_pred" in result.available_epi_cols:
        result.effective_epi_col = "epi_pred"
    elif "combined_dddg_pred" in result.available_epi_cols:
        result.effective_epi_col = "combined_dddg_pred"
    else:
        result.effective_epi_col = None

    # Target chain filtering.
    sel_chain = result.selected_chain
    if sel_chain:
        if sel_chain not in chains_seen:
            result.chain_matched = False
            chains_str = ", ".join(chains_seen) if chains_seen else "(none)"
            result.error = (
                f"Selected chain '{sel_chain}' has no matching rows in CSV "
                f"(available chain(s): {chains_str})."
            )
            return result
        chain_mask = np.array([c == sel_chain for c in chain_list], dtype=bool)
    else:
        chain_mask = np.ones(len(chain_list), dtype=bool)

    # Amino-acid exclusion filters.
    ex_wt = {a.upper() for a in (exclude_wt_aa or set())}
    ex_mt = {a.upper() for a in (exclude_mt_aa or set())}

    def _keep_aa(muts: List[dict]) -> bool:
        if not ex_wt and not ex_mt:
            return True
        for m in muts:
            if m["wt"] in ex_wt or m["mut"] in ex_mt:
                return False
        return True

    aa_mask = np.array([valid_mut_mask[i] and _keep_aa(parsed_list[i]) for i in range(len(parsed_list))], dtype=bool)
    row_mask = valid_mut_mask & chain_mask & aa_mask
    if not row_mask.any():
        result.error = "Global amino-acid filters removed all mutations for the selected chain."
        return result

    score_arr = numeric_cols[eff_score_col].to_numpy(dtype=float, na_value=np.nan)

    # Aggregate singles (preserving valid 0.0 scores, filtering non-finite values, averaging duplicate entries).
    single_accum: Dict[Tuple[str, PosKey, str], List[float]] = defaultdict(list)
    singles_indices = np.flatnonzero(row_mask & (mut_lens == 1))
    nonfinite_singles = 0
    for idx in singles_indices:
        val = float(score_arr[idx])
        if not np.isfinite(val):
            nonfinite_singles += 1
            continue
        m0 = parsed_list[idx][0]
        key = (chain_list[idx], m0["pos_key"], m0["mut"])
        single_accum[key].append(val)

    if nonfinite_singles > 0:
        result.warnings.append(f"Ignored {nonfinite_singles} single-mutant row(s) with NaN/Inf scores.")

    if not single_accum:
        result.error = (
            "No valid single mutations found in the CSV for the selected chain. "
            "Single mutation data is required in all modes to color individual mutant sidechains. "
            "Ensure you ran inference with mode 'singles+doubles' or 'singles'."
        )
        return result

    all_singles_scores: Dict[Tuple[str, PosKey, str], float] = {}
    by_pos: Dict[ResKey, Tuple[float, str]] = {}
    all_vals: List[float] = []

    for (c, pos_key, mut_aa), vals in single_accum.items():
        mean_val = float(np.mean(vals))
        all_singles_scores[(c, pos_key, mut_aa)] = mean_val
        all_vals.append(mean_val)
        res_key = (c, pos_key)
        prev = by_pos.get(res_key)
        if prev is None or mean_val > prev[0]:
            by_pos[res_key] = (mean_val, mut_aa)

    result.all_singles_scores = all_singles_scores
    result.residue_scores_data = by_pos
    result.n_singles = len(all_singles_scores)
    result.n_positions = len(by_pos)
    result.score_min = float(min(all_vals))
    result.score_max = float(max(all_vals))

    # Build doubles/epistasis DataFrame whenever doubles exist (so counts and epistasis modes are ready).
    doubles_indices = np.flatnonzero(row_mask & (mut_lens == 2))
    epi_col = result.effective_epi_col

    if len(doubles_indices) > 0 and epi_col is not None:
        epi_arr = numeric_cols[epi_col].to_numpy(dtype=float, na_value=np.nan)
        comb_arr = (
            numeric_cols["combined_pred"].to_numpy(dtype=float, na_value=np.nan)
            if "combined_pred" in numeric_cols
            else score_arr
        )
        records = []
        nonfinite_doubles = 0
        for idx in doubles_indices:
            epi_val = float(epi_arr[idx])
            comb_val = float(comb_arr[idx])
            if not np.isfinite(epi_val) or not np.isfinite(comb_val):
                nonfinite_doubles += 1
                continue
            m0, m1 = parsed_list[idx][0], parsed_list[idx][1]
            records.append({
                "chain_id": chain_list[idx],
                "pos1_key": m0["pos_key"],
                "pos1_pdb": m0["pos"],
                "wt1": m0["wt"],
                "mut1": m0["mut"],
                "pos2_key": m1["pos_key"],
                "pos2_pdb": m1["pos"],
                "wt2": m1["wt"],
                "mut2": m1["mut"],
                "dddg_pred": epi_val,
                "combined_pred": comb_val,
            })

        if nonfinite_doubles > 0:
            result.warnings.append(f"Ignored {nonfinite_doubles} double-mutant row(s) with NaN/Inf scores.")

        if records:
            result.epistasis_df = pd.DataFrame(records)
            result.n_doubles = len(records)

    # Validate epistasis mode requirements if user explicitly requested epistasis visualization.
    if "epistasis" in (mode or "").lower():
        if epi_col is None or result.epistasis_df is None or result.epistasis_df.empty:
            reason = (
                "an epistasis column ('epi_pred' or 'combined_dddg_pred') is not present in this CSV"
                if epi_col is None
                else "no valid double mutations were found in this CSV for the selected chain/filters"
            )
            if allow_mode_fallback:
                result.auto_fallback_mode = "Singles"
                result.warnings.append(
                    f"{mode} mode unavailable ({reason}); automatically switching to Singles mode."
                )
            else:
                result.error = f"{mode} mode requested, but {reason}."
                return result

    result.ok = True
    return result
