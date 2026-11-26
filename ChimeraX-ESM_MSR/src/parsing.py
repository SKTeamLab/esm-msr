# vim: set expandtab shiftwidth=4 softtabstop=4:
"""Pure-Python/pandas parsing, palette, environment discovery, and ranking helpers for ChimeraX-ESM-MSR."""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import glob
import os
import re
import shutil
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

# Anchored PDB mutation token: WT amino acid, optional negative sign + digits, optional insertion code, MT amino acid.
MUT_TOKEN_RE = re.compile(r"^([A-Za-z])(-?\d+)([A-Za-z]?)([A-Za-z])$")
TQDM_PCT_RE = re.compile(r"(\d{1,3})%\|")

SCORE_COLUMNS = ("combined_pred", "wt_lora_pred", "mt_lora_pred")
EPISTASIS_COLUMNS = ("epi_pred", "combined_dddg_pred")

PosKey = Tuple[int, str]
ResKey = Tuple[str, PosKey]

# ---------------------------------------------------------------------------
# Color palettes (verified zero-network in ChimeraX 1.11.1)
# Convention: low/negative (destabilizing) -> left stop; high/positive (stabilizing) -> right stop.
# ---------------------------------------------------------------------------

DEFAULT_PALETTE_KEY = "blue_white_red"

COLOR_PALETTES: Dict[str, Dict[str, Any]] = {
    "blue_white_red": {
        "key": "blue_white_red",
        "label": "Blue\u2013White\u2013Red (Classic)",
        "chimerax": "blue:white:red",
        "mpl": "RdBu_r",
        "diverging": True,
        "stops": ("blue", "white", "red"),
    },
    "purple_white_green": {
        "key": "purple_white_green",
        "label": "Purple\u2013White\u2013Green (PRGn, Colorblind-Safe)",
        "chimerax": "#762a83:#f7f7f7:#1b7837",
        "mpl": "PRGn",
        "diverging": True,
        "stops": ("#762a83", "#f7f7f7", "#1b7837"),
    },
    "brown_white_teal": {
        "key": "brown_white_teal",
        "label": "Brown\u2013White\u2013Teal (BrBG, Colorblind-Safe)",
        "chimerax": "#8c510a:#f5f5f5:#01665e",
        "mpl": "BrBG",
        "diverging": True,
        "stops": ("#8c510a", "#f5f5f5", "#01665e"),
    },
    "viridis": {
        "key": "viridis",
        "label": "Viridis (Perceptually Uniform)",
        "chimerax": "#440154:#3b528b:#21918c:#5ec962:#fde725",
        "mpl": "viridis",
        "diverging": False,
        "stops": ("#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"),
    },
    "cividis": {
        "key": "cividis",
        "label": "Cividis (Colorblind-Safe Sequential)",
        "chimerax": "#00204c:#414d6b:#7c7b78:#bcaf6f:#ffea46",
        "mpl": "cividis",
        "diverging": False,
        "stops": ("#00204c", "#414d6b", "#7c7b78", "#bcaf6f", "#ffea46"),
    },
}

# 12 high-contrast colorblind-friendly categorical colors starting with Okabe-Ito.
CATEGORICAL_PAIR_PALETTE: Tuple[str, ...] = (
    "#0072B2",  # Blue (Okabe-Ito)
    "#D55E00",  # Vermilion (Okabe-Ito)
    "#009E73",  # Bluish green (Okabe-Ito)
    "#CC79A7",  # Reddish purple (Okabe-Ito)
    "#E69F00",  # Orange (Okabe-Ito)
    "#56B4E9",  # Sky blue (Okabe-Ito)
    "#332288",  # Indigo (Tol)
    "#882255",  # Wine (Tol)
    "#44AA99",  # Teal (Tol)
    "#AA4499",  # Purple (Tol)
    "#117733",  # Green (Tol)
    "#999933",  # Olive (Tol)
)


def get_color_palette(palette_key: Optional[str] = None) -> Dict[str, Any]:
    """Return a palette spec dict from COLOR_PALETTES, falling back to blue_white_red."""
    if not palette_key:
        return COLOR_PALETTES[DEFAULT_PALETTE_KEY]
    key = str(palette_key).strip().lower()
    if key in COLOR_PALETTES:
        return COLOR_PALETTES[key]
    for spec in COLOR_PALETTES.values():
        if spec["label"].lower() == key or spec["mpl"].lower() == key:
            return spec
    return COLOR_PALETTES[DEFAULT_PALETTE_KEY]


def get_pair_color(rank_idx: int) -> str:
    """Return a high-contrast categorical hex color for a 0-based pair index."""
    return CATEGORICAL_PAIR_PALETTE[int(rank_idx) % len(CATEGORICAL_PAIR_PALETTE)]


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


# ---------------------------------------------------------------------------
# Fast disk-only Python environment discovery (zero subprocesses)
# ---------------------------------------------------------------------------

@dataclass
class DiscoveredEnv:
    name: str
    python_path: str
    env_dir: str
    has_esm_msr: bool
    has_esm: bool
    has_torch: bool
    label: str


def _find_python_in_env_dir(env_dir: str) -> Optional[str]:
    for rel in (
        os.path.join("bin", "python"),
        os.path.join("bin", "python3"),
        os.path.join("Scripts", "python.exe"),
        "python.exe",
    ):
        cand = os.path.join(env_dir, rel)
        if os.path.isfile(cand):
            return os.path.normpath(os.path.abspath(cand))
    return None


def _inspect_env_packages(env_dir: str) -> Tuple[bool, bool, bool]:
    """Detect (has_esm_msr, has_esm, has_torch) via fast site-packages directory scan."""
    sp_dirs: List[str] = []
    lib_dir = os.path.join(env_dir, "lib")
    if os.path.isdir(lib_dir):
        for sp in glob.glob(os.path.join(lib_dir, "python*", "site-packages")):
            if os.path.isdir(sp):
                sp_dirs.append(sp)
    win_sp = os.path.join(env_dir, "Lib", "site-packages")
    if os.path.isdir(win_sp):
        sp_dirs.append(win_sp)

    has_esm_msr = False
    has_esm = False
    has_torch = False

    for sp in sp_dirs:
        if os.path.isdir(os.path.join(sp, "torch")):
            has_torch = True
        if os.path.isdir(os.path.join(sp, "esm")):
            has_esm = True
        if os.path.isdir(os.path.join(sp, "esm_msr")):
            has_esm_msr = True
        if not has_esm_msr:
            try:
                entries = os.listdir(sp)
            except OSError:
                entries = []
            for fname in entries:
                low = fname.lower()
                if ("esm_msr" in low or "esm-msr" in low) and low.endswith(
                    (".pth", ".egg-link", ".dist-info", ".egg-info")
                ):
                    has_esm_msr = True
                    break
        if has_esm_msr and has_esm and has_torch:
            break

    return has_esm_msr, has_esm, has_torch


def discover_python_environments(base_repo_path: Optional[str] = None) -> List[DiscoveredEnv]:
    """Scan local conda/mamba/venv directories on disk (zero subprocesses, <300 ms).

    Returns environments ranked with ESM-MSR + ESM + PyTorch installs first.
    """
    candidate_dirs: List[str] = []

    # 1. Active environment variables
    for env_var in ("CONDA_PREFIX", "VIRTUAL_ENV"):
        val = (os.environ.get(env_var) or "").strip()
        if val and os.path.isdir(val):
            candidate_dirs.append(val)

    # 2. ~/.conda/environments.txt
    home = os.path.expanduser("~")
    envs_txt = os.path.join(home, ".conda", "environments.txt")
    if os.path.isfile(envs_txt):
        try:
            with open(envs_txt, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    p = line.strip()
                    if p and os.path.isdir(p):
                        candidate_dirs.append(p)
        except OSError:
            pass

    # 3. Common conda / mamba / miniforge roots and their envs/* children
    conda_roots = [
        os.path.join(home, "miniconda3"),
        os.path.join(home, "anaconda3"),
        os.path.join(home, "mambaforge"),
        os.path.join(home, "miniforge3"),
        os.path.join(home, "micromamba"),
        os.path.join(home, ".conda"),
        "/opt/conda",
    ]
    for root in conda_roots:
        if os.path.isdir(root):
            candidate_dirs.append(root)
            envs_sub = os.path.join(root, "envs")
            if os.path.isdir(envs_sub):
                try:
                    for entry in sorted(os.listdir(envs_sub)):
                        full = os.path.join(envs_sub, entry)
                        if os.path.isdir(full):
                            candidate_dirs.append(full)
                except OSError:
                    pass

    # 4. Repo-local virtualenvs
    if base_repo_path and str(base_repo_path).strip():
        repo_dir = os.path.expanduser(str(base_repo_path).strip())
        if os.path.isdir(repo_dir):
            for venv_name in (".venv", "venv", "env"):
                cand = os.path.join(repo_dir, venv_name)
                if os.path.isdir(cand):
                    candidate_dirs.append(cand)

    discovered: List[DiscoveredEnv] = []
    seen_real_pythons: Set[str] = set()

    for raw_dir in candidate_dirs:
        env_dir = os.path.normpath(os.path.abspath(raw_dir))
        py_path = _find_python_in_env_dir(env_dir)
        if not py_path:
            continue
        real_py = os.path.realpath(py_path)
        if real_py in seen_real_pythons:
            continue
        seen_real_pythons.add(real_py)

        name = os.path.basename(env_dir.rstrip(os.sep)) or env_dir
        has_esm_msr, has_esm, has_torch = _inspect_env_packages(env_dir)
        tags: List[str] = []
        if has_esm_msr:
            tags.append("esm_msr")
        if has_esm:
            tags.append("esm")
        if has_torch:
            tags.append("torch")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        label = f"{name}{tag_str} \u2014 {py_path}"

        discovered.append(
            DiscoveredEnv(
                name=name,
                python_path=py_path,
                env_dir=env_dir,
                has_esm_msr=has_esm_msr,
                has_esm=has_esm,
                has_torch=has_torch,
                label=label,
            )
        )

    discovered.sort(
        key=lambda e: (
            not e.has_esm_msr,
            not (e.has_torch and e.has_esm),
            not e.has_torch,
            not e.has_esm,
            e.name.lower(),
            e.python_path,
        )
    )
    return discovered


# ---------------------------------------------------------------------------
# Parsed CSV container & RankedPair selection
# ---------------------------------------------------------------------------

def _finite_float_or_none(arr: Optional[np.ndarray], idx: int) -> Optional[float]:
    if arr is None:
        return None
    val = float(arr[idx])
    return val if np.isfinite(val) else None


def _finite_int_or_none(arr: Optional[np.ndarray], idx: int) -> Optional[int]:
    if arr is None:
        return None
    val = float(arr[idx])
    return int(round(val)) if np.isfinite(val) else None


@dataclass
class ParsedCSVData:
    filepath: str
    ok: bool = False
    error: str = ""
    warnings: List[str] = field(default_factory=list)
    available_score_cols: Set[str] = field(default_factory=set)
    available_epi_cols: List[str] = field(default_factory=list)
    effective_score_col: str = ""
    singles_score_col: str = ""
    singles_score_fallback_note: Optional[str] = None
    is_two_stage_csv: bool = False
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
    wt_residues: Dict[ResKey, str] = field(default_factory=dict)
    single_records: List[Dict[str, Any]] = field(default_factory=list)
    double_records: List[Dict[str, Any]] = field(default_factory=list)
    epistasis_df: Optional[pd.DataFrame] = None


# Convenience alias
ParsedCSV = ParsedCSVData


@dataclass
class RankedPair:
    rank: int
    color_hex: str
    chain: str
    mut_str: str
    mut1_str: str
    mut2_str: str
    pos1: PosKey
    pos2: PosKey
    wt1: str
    mut1: str
    wt2: str
    mut2: str
    score: float
    combined_pred: Optional[float] = None
    epi_score: Optional[float] = None
    wt_pred: Optional[float] = None
    mt_pred: Optional[float] = None
    stage1_additive_pred: Optional[float] = None
    additive_rank: Optional[int] = None
    full_rank: Optional[int] = None
    rank_change: Optional[int] = None
    dist: Optional[float] = None
    shared_positions: Tuple[PosKey, ...] = ()


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
      - Unified two-stage fast-doubles CSVs (Stage-1 singles with wt_lora_pred + Stage-2 doubles with combined_pred)
      - Both epi_pred and combined_dddg_pred epistasis columns
      - Non-finite (NaN/Inf) filtering while preserving valid 0.0 scores
      - Target chain filtering with informative reporting
      - Structured single_records and double_records for tables and Top-M categorical visualization
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
    numeric_arrays: Dict[str, np.ndarray] = {}
    for col in SCORE_COLUMNS:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            arr = vals.to_numpy(dtype=float, na_value=np.nan)
            if bool(np.isfinite(arr).any()):
                result.available_score_cols.add(col)
                numeric_cols[col] = vals
                numeric_arrays[col] = arr

    for col in EPISTASIS_COLUMNS:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            arr = vals.to_numpy(dtype=float, na_value=np.nan)
            if bool(np.isfinite(arr).any()):
                result.available_epi_cols.append(col)
                numeric_cols[col] = vals
                numeric_arrays[col] = arr

    # Optional enrichment columns (two-stage metadata + heavy-atom distance).
    optional_numeric_cols = (
        "stage1_additive_pred",
        "additive_rank",
        "full_rank",
        "rank_change",
        "dist",
    )
    for col in optional_numeric_cols:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            arr = vals.to_numpy(dtype=float, na_value=np.nan)
            if bool(np.isfinite(arr).any()):
                numeric_arrays[col] = arr

    if any(c in numeric_arrays for c in ("stage1_additive_pred", "additive_rank", "full_rank", "rank_change")):
        result.is_two_stage_csv = True

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

    score_arr = numeric_arrays[eff_score_col]
    wt_arr = numeric_arrays.get("wt_lora_pred")
    mt_arr = numeric_arrays.get("mt_lora_pred")
    comb_arr_opt = numeric_arrays.get("combined_pred")

    # Determine which score array to use for singles:
    # In two-stage fast-doubles CSVs, Stage-1 singles have finite wt_lora_pred while combined_pred/mt_lora_pred are NaN.
    singles_indices = np.flatnonzero(row_mask & (mut_lens == 1))
    singles_score_arr = score_arr
    result.singles_score_col = eff_score_col

    if len(singles_indices) > 0 and eff_score_col != "wt_lora_pred" and wt_arr is not None:
        primary_finite_singles = int(np.isfinite(score_arr[singles_indices]).sum())
        wt_finite_singles = int(np.isfinite(wt_arr[singles_indices]).sum())
        if primary_finite_singles == 0 and wt_finite_singles > 0:
            singles_score_arr = wt_arr
            result.singles_score_col = "wt_lora_pred"
            result.is_two_stage_csv = True
            note = (
                f"Single-mutant rows use Stage-1 'wt_lora_pred' because '{eff_score_col}' is only "
                f"populated on Stage-2 doubles."
            )
            result.singles_score_fallback_note = note
            result.warnings.append(note)

    # Aggregate singles (preserving valid 0.0 scores, filtering non-finite values, averaging duplicate entries).
    single_accum: Dict[Tuple[str, PosKey, str], List[float]] = defaultdict(list)
    single_meta: Dict[Tuple[str, PosKey, str], Dict[str, Any]] = {}
    wt_residues: Dict[ResKey, str] = {}
    nonfinite_singles = 0

    for idx in singles_indices:
        val = float(singles_score_arr[idx])
        if not np.isfinite(val):
            nonfinite_singles += 1
            continue
        m0 = parsed_list[idx][0]
        c = chain_list[idx]
        key = (c, m0["pos_key"], m0["mut"])
        single_accum[key].append(val)
        wt_residues[(c, m0["pos_key"])] = m0["wt"]
        if key not in single_meta:
            single_meta[key] = {
                "chain": c,
                "pos_key": m0["pos_key"],
                "pos": m0["pos"],
                "icode": m0["icode"],
                "wt": m0["wt"],
                "mut": m0["mut"],
                "mut_str": f"{m0['wt']}{format_res_pos(m0['pos_key'])}{m0['mut']}",
                "wt_pred": _finite_float_or_none(wt_arr, idx),
                "mt_pred": _finite_float_or_none(mt_arr, idx),
                "combined_pred": _finite_float_or_none(comb_arr_opt, idx),
            }

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
    single_records: List[Dict[str, Any]] = []

    for (c, pos_key, mut_aa), vals in single_accum.items():
        mean_val = float(np.mean(vals))
        all_singles_scores[(c, pos_key, mut_aa)] = mean_val
        all_vals.append(mean_val)
        res_key = (c, pos_key)
        prev = by_pos.get(res_key)
        if prev is None or mean_val > prev[0]:
            by_pos[res_key] = (mean_val, mut_aa)
        rec = dict(single_meta[(c, pos_key, mut_aa)])
        rec["score"] = mean_val
        single_records.append(rec)

    single_records.sort(key=lambda r: (-r["score"], r["chain"], r["pos"], r["icode"], r["mut"]))

    result.all_singles_scores = all_singles_scores
    result.residue_scores_data = by_pos
    result.wt_residues = wt_residues
    result.single_records = single_records
    result.n_singles = len(all_singles_scores)
    result.n_positions = len(by_pos)
    result.score_min = float(min(all_vals))
    result.score_max = float(max(all_vals))

    # Build doubles/epistasis DataFrame & structured double_records whenever doubles exist.
    doubles_indices = np.flatnonzero(row_mask & (mut_lens == 2))
    epi_col = result.effective_epi_col

    if len(doubles_indices) > 0:
        epi_arr = numeric_arrays.get(epi_col) if epi_col is not None else None
        comb_arr = comb_arr_opt if comb_arr_opt is not None else score_arr
        s1_arr = numeric_arrays.get("stage1_additive_pred")
        add_rank_arr = numeric_arrays.get("additive_rank")
        full_rank_arr = numeric_arrays.get("full_rank")
        rank_chg_arr = numeric_arrays.get("rank_change")
        dist_arr = numeric_arrays.get("dist")

        epi_rows: List[Dict[str, Any]] = []
        double_records: List[Dict[str, Any]] = []
        nonfinite_doubles = 0

        for idx in doubles_indices:
            comb_val = float(comb_arr[idx])
            primary_val = float(score_arr[idx])
            eff_double_score = primary_val if np.isfinite(primary_val) else comb_val
            epi_val = float(epi_arr[idx]) if epi_arr is not None else float("nan")

            if epi_arr is not None:
                if not np.isfinite(epi_val) or not np.isfinite(comb_val):
                    nonfinite_doubles += 1
                    continue
            else:
                if not np.isfinite(eff_double_score):
                    nonfinite_doubles += 1
                    continue

            m0, m1 = parsed_list[idx][0], parsed_list[idx][1]
            c = chain_list[idx]
            mut1_str = f"{m0['wt']}{format_res_pos(m0['pos_key'])}{m0['mut']}"
            mut2_str = f"{m1['wt']}{format_res_pos(m1['pos_key'])}{m1['mut']}"
            mut_str = f"{mut1_str}:{mut2_str}"

            wt_v = _finite_float_or_none(wt_arr, idx)
            mt_v = _finite_float_or_none(mt_arr, idx)
            comb_v = comb_val if np.isfinite(comb_val) else None
            epi_v = epi_val if np.isfinite(epi_val) else None
            s1_v = _finite_float_or_none(s1_arr, idx)
            add_r = _finite_int_or_none(add_rank_arr, idx)
            full_r = _finite_int_or_none(full_rank_arr, idx)
            r_chg = _finite_int_or_none(rank_chg_arr, idx)
            dist_v = _finite_float_or_none(dist_arr, idx)

            if r_chg is None and add_r is not None and full_r is not None:
                r_chg = add_r - full_r

            drec = {
                "chain": c,
                "mut_str": mut_str,
                "mut1_str": mut1_str,
                "mut2_str": mut2_str,
                "pos1": m0["pos_key"],
                "pos2": m1["pos_key"],
                "pos1_pdb": m0["pos"],
                "pos2_pdb": m1["pos"],
                "wt1": m0["wt"],
                "mut1": m0["mut"],
                "wt2": m1["wt"],
                "mut2": m1["mut"],
                "score": float(eff_double_score),
                "combined_pred": comb_v,
                "epi_score": epi_v,
                "wt_pred": wt_v,
                "mt_pred": mt_v,
                "stage1_additive_pred": s1_v,
                "additive_rank": add_r,
                "full_rank": full_r,
                "rank_change": r_chg,
                "dist": dist_v,
            }
            double_records.append(drec)

            if epi_arr is not None:
                erow: Dict[str, Any] = {
                    "chain_id": c,
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
                    "mut_str": mut_str,
                }
                if wt_v is not None:
                    erow["wt_lora_pred"] = wt_v
                if mt_v is not None:
                    erow["mt_lora_pred"] = mt_v
                if s1_v is not None:
                    erow["stage1_additive_pred"] = s1_v
                if add_r is not None:
                    erow["additive_rank"] = add_r
                if full_r is not None:
                    erow["full_rank"] = full_r
                if r_chg is not None:
                    erow["rank_change"] = r_chg
                if dist_v is not None:
                    erow["dist"] = dist_v
                epi_rows.append(erow)

        if nonfinite_doubles > 0:
            result.warnings.append(f"Ignored {nonfinite_doubles} double-mutant row(s) with NaN/Inf scores.")

        if double_records:
            double_records.sort(
                key=lambda r: (
                    r["full_rank"] if r["full_rank"] is not None else 10**12,
                    -r["score"],
                    r["mut_str"],
                )
            )
            result.double_records = double_records
            result.n_doubles = len(double_records)

        if epi_rows:
            result.epistasis_df = pd.DataFrame(epi_rows)

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


def _extract_pair_sort_value(rec: Dict[str, Any], sort_by: str) -> Optional[float]:
    key_norm = (sort_by or "combined_pred").strip().lower()
    mapping = {
        "combined_pred": "combined_pred",
        "stability": "combined_pred",
        "score": "score",
        "epi_pred": "epi_score",
        "epi_score": "epi_score",
        "epistasis": "epi_score",
        "dddg_pred": "epi_score",
        "combined_dddg_pred": "epi_score",
        "wt_lora_pred": "wt_pred",
        "wt_pred": "wt_pred",
        "mt_lora_pred": "mt_pred",
        "mt_pred": "mt_pred",
        "stage1_additive_pred": "stage1_additive_pred",
        "rank_change": "rank_change",
        "dist": "dist",
    }
    field_name = mapping.get(key_norm, key_norm)
    val = rec.get(field_name)
    if val is None and field_name == "combined_pred":
        val = rec.get("score")
    if val is None:
        return None
    fval = float(val)
    return fval if np.isfinite(fval) else None


def select_top_m_pairs(
    parsed: ParsedCSVData,
    top_m: int = 5,
    sort_by: str = "combined_pred",
    priority: str = "High score",
    diverse_positions: bool = True,
    max_pairs_per_pos: int = 1,
    min_dist: Optional[float] = None,
) -> List[RankedPair]:
    """Select up to top_m double-mutant pairs with distinct categorical colors.

    When diverse_positions=True, greedily selects highest-ranked pairs while capping each
    residue position to at most max_pairs_per_pos appearances (default 1 = disjoint pairs)
    so a single hub mutation cannot monopolize all Top-M slots.
    """
    if parsed is None or not parsed.double_records or top_m <= 0:
        return []

    max_per_pos = max(1, int(max_pairs_per_pos))
    prio_low = (priority or "High score").strip().lower()

    candidates: List[Tuple[float, Dict[str, Any]]] = []
    for rec in parsed.double_records:
        if min_dist is not None and min_dist > 0:
            d = rec.get("dist")
            if d is None or not np.isfinite(d) or d < min_dist:
                continue
        sval = _extract_pair_sort_value(rec, sort_by)
        if sval is None:
            continue
        candidates.append((sval, rec))

    if not candidates:
        return []

    if "low" in prio_low:
        candidates.sort(key=lambda item: (item[0], item[1]["mut_str"]))
    elif "abs" in prio_low or "mag" in prio_low:
        candidates.sort(key=lambda item: (-abs(item[0]), -item[0], item[1]["mut_str"]))
    else:
        candidates.sort(key=lambda item: (-item[0], item[1]["mut_str"]))

    chosen_recs: List[Dict[str, Any]] = []
    pos_counts: Counter = Counter()

    for _, rec in candidates:
        if len(chosen_recs) >= top_m:
            break
        k1 = (rec["chain"], rec["pos1"])
        k2 = (rec["chain"], rec["pos2"])
        if diverse_positions:
            if pos_counts[k1] >= max_per_pos or pos_counts[k2] >= max_per_pos:
                continue
        chosen_recs.append(rec)
        pos_counts[k1] += 1
        pos_counts[k2] += 1

    # Identify any positions shared across >1 selected pair (useful when diverse_positions=False or max_pairs_per_pos>1).
    shared_keys = {k for k, cnt in pos_counts.items() if cnt > 1}

    ranked: List[RankedPair] = []
    for idx, rec in enumerate(chosen_recs):
        shared: List[PosKey] = []
        if (rec["chain"], rec["pos1"]) in shared_keys:
            shared.append(rec["pos1"])
        if (rec["chain"], rec["pos2"]) in shared_keys and rec["pos2"] != rec["pos1"]:
            shared.append(rec["pos2"])
        ranked.append(
            RankedPair(
                rank=idx + 1,
                color_hex=get_pair_color(idx),
                chain=rec["chain"],
                mut_str=rec["mut_str"],
                mut1_str=rec["mut1_str"],
                mut2_str=rec["mut2_str"],
                pos1=rec["pos1"],
                pos2=rec["pos2"],
                wt1=rec["wt1"],
                mut1=rec["mut1"],
                wt2=rec["wt2"],
                mut2=rec["mut2"],
                score=float(rec["score"]),
                combined_pred=rec.get("combined_pred"),
                epi_score=rec.get("epi_score"),
                wt_pred=rec.get("wt_pred"),
                mt_pred=rec.get("mt_pred"),
                stage1_additive_pred=rec.get("stage1_additive_pred"),
                additive_rank=rec.get("additive_rank"),
                full_rank=rec.get("full_rank"),
                rank_change=rec.get("rank_change"),
                dist=rec.get("dist"),
                shared_positions=tuple(shared),
            )
        )

    return ranked
