# vim: set expandtab shiftwidth=4 softtabstop=4:
"""Shared headless-safe ChimeraX visualization engine and native CLI commands for ChimeraX-ESM-MSR."""

from __future__ import annotations

from collections import defaultdict
import os
import tempfile
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from chimerax.atomic import PseudobondGroup, Residue, Structure
from chimerax.core.commands import (
    BoolArg,
    CmdDesc,
    FloatArg,
    IntArg,
    ModelsArg,
    OpenFileNameArg,
    SaveFileNameArg,
    StringArg,
    register,
    run,
)

from .heatmap import export_mutational_heatmap
from .parsing import (
    COLOR_PALETTES,
    DEFAULT_PALETTE_KEY,
    ParsedCSVData,
    RankedPair,
    format_res_pos,
    get_color_palette,
    normalize_icode,
    parse_esm_msr_csv,
    select_top_m_pairs,
)

SCORE_ATTRIBUTE_NAME = "esm_msr_score"
EPISTASIS_PBG_NAME = "ESM-MSR Epistasis"
RANKED_PAIRS_PBG_NAME = "ESM-MSR Ranked Pairs"
LEGEND_LABEL_PREFIX = "esm_msr_leg_"

ONE_TO_THREE_LETTER_AA: Dict[str, str] = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL",
}

_SORT_BY_LABELS: Dict[str, str] = {
    "combined_pred": "Combined \u0394\u0394G",
    "epi_score": "Epistasis",
    "stage1_additive_pred": "Stage-1 Additive",
    "rank_change": "Rank Gain (\u0394Rank)",
}


# ---------------------------------------------------------------------------
# Session-level state (shared across CLI commands & GUI tool)
# ---------------------------------------------------------------------------

def get_session_state(session) -> Dict[str, Any]:
    state = getattr(session, "_esm_msr_state", None)
    if state is None:
        state = {
            "clone_ids": [],
            "pair_clone_map": {},
            "wt_model_id": "",
            "target_chain": "",
            "last_csv_path": "",
            "last_parsed": None,
            "last_mode": "Singles",
            "last_ranked_pairs": [],
            "attr_registered": False,
        }
        setattr(session, "_esm_msr_state", state)
    return state


def _get_open_tool(session):
    """Return an active ESM_MSR_Tool instance if one is open in this session."""
    tools_mgr = getattr(session, "tools", None)
    if tools_mgr is None:
        return None
    for t in tools_mgr.list():
        if type(t).__name__ == "ESM_MSR_Tool" and not getattr(t, "_closing", False):
            return t
    return None


# ---------------------------------------------------------------------------
# Color, palette, and range helpers
# ---------------------------------------------------------------------------

def hex_to_rgb(color_str: str) -> Tuple[int, int, int]:
    """Convert '#RRGGBB' or common CSS color names to (r, g, b) ints in 0..255."""
    s = (color_str or "").strip().lower()
    named = {
        "blue": (0, 0, 255),
        "red": (255, 0, 0),
        "white": (255, 255, 255),
        "green": (0, 128, 0),
        "orange": (255, 165, 0),
        "purple": (128, 0, 128),
        "teal": (0, 128, 128),
        "brown": (140, 81, 10),
        "gray": (128, 128, 128),
        "grey": (128, 128, 128),
    }
    if s in named:
        return named[s]
    if s.startswith("#"):
        h = s[1:]
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        if len(h) >= 6:
            try:
                return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            except ValueError:
                pass
    return (0, 114, 178)


def hex_to_rgba(color_str: str, alpha: int = 255) -> Tuple[int, int, int, int]:
    r, g, b = hex_to_rgb(color_str)
    a = max(0, min(255, int(alpha)))
    return (r, g, b, a)


def interpolate_palette_rgba(palette_spec: Dict[str, Any], norm_0_to_1: float, alpha: int = 255) -> Tuple[int, int, int, int]:
    """Interpolate a color along palette_spec['stops'] for t in [0.0, 1.0]."""
    stops = palette_spec.get("stops") or ("blue", "white", "red")
    t = max(0.0, min(1.0, float(norm_0_to_1)))
    if len(stops) == 1:
        return hex_to_rgba(stops[0], alpha)
    pos = t * (len(stops) - 1)
    idx0 = min(len(stops) - 2, int(pos))
    frac = pos - idx0
    r0, g0, b0 = hex_to_rgb(stops[idx0])
    r1, g1, b1 = hex_to_rgb(stops[idx0 + 1])
    r = int(round(r0 + (r1 - r0) * frac))
    g = int(round(g0 + (g1 - g0) * frac))
    b = int(round(b0 + (b1 - b0) * frac))
    return (r, g, b, max(0, min(255, int(alpha))))


def compute_score_range(
    values: Sequence[float],
    palette_spec: Dict[str, Any],
    auto_range: bool = True,
    manual_min: Optional[float] = None,
    manual_max: Optional[float] = None,
) -> Tuple[float, float]:
    """Compute (vmin, vmax) for ChimeraX color byattribute and key commands."""
    if not auto_range and manual_min is not None and manual_max is not None:
        lo, hi = float(manual_min), float(manual_max)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            return lo, hi

    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return (-1.0, 1.0) if palette_spec.get("diverging", True) else (0.0, 1.0)

    arr = np.asarray(finite, dtype=float)
    if palette_spec.get("diverging", True):
        bound = float(np.percentile(np.abs(arr), 98))
        if bound <= 0.05:
            bound = float(np.max(np.abs(arr)))
        if bound <= 0.0:
            bound = 0.01
        return (-bound, bound)

    p_lo = float(np.percentile(arr, 2))
    p_hi = float(np.percentile(arr, 98))
    if p_hi <= p_lo:
        p_lo = float(np.min(arr))
        p_hi = float(np.max(arr))
    if p_hi <= p_lo:
        p_lo -= 0.5
        p_hi += 0.5
    return (p_lo, p_hi)


def build_key_command(palette_spec: Dict[str, Any], vmin: float, vmax: float) -> str:
    """Construct a verified zero-network ChimeraX `key` command from palette stops."""
    stops = list(palette_spec.get("stops") or ("blue", "white", "red"))
    n = len(stops)
    if n == 3:
        mid = 0.0 if (vmin < 0.0 < vmax) else 0.5 * (vmin + vmax)
        return f"key {stops[0]}:{vmin:.2f} {stops[1]}:{mid:.2f} {stops[2]}:{vmax:.2f} pos 0.05,0.05"
    if n == 5:
        mid = 0.5 * (vmin + vmax)
        return (
            f"key {stops[0]}:{vmin:.2f} {stops[1]}: {stops[2]}:{mid:.2f} "
            f"{stops[3]}: {stops[4]}:{vmax:.2f} pos 0.05,0.05"
        )
    if n >= 2:
        parts = [f"{stops[0]}:{vmin:.2f}"]
        for s in stops[1:-1]:
            parts.append(f"{s}:")
        parts.append(f"{stops[-1]}:{vmax:.2f}")
        return f"key {' '.join(parts)} pos 0.05,0.05"
    return f"key blue:{vmin:.2f} white:0 red:{vmax:.2f} pos 0.05,0.05"


# ---------------------------------------------------------------------------
# Residue / Model / Cleanup primitives
# ---------------------------------------------------------------------------

def get_spec(model_id_string: str, res_keys: Iterable[Tuple[str, Any]]) -> str:
    if not res_keys:
        return "None"
    by_chain: Dict[str, List[str]] = defaultdict(list)
    for c, pos_key in res_keys:
        by_chain[c].append(format_res_pos(pos_key))
    specs = []
    for c, positions in by_chain.items():
        specs.append(f"#{model_id_string}/{c}:{','.join(positions)}")
    return " | ".join(specs)


def build_residue_lookup(model) -> Dict[Tuple[str, Tuple[int, str]], Any]:
    return {
        (r.chain_id, (r.number, normalize_icode(getattr(r, "insertion_code", "")))): r
        for r in model.residues
    }


def set_residue_scores(session, model, score_map: Dict[Tuple[str, Tuple[int, str]], float]) -> None:
    """Assign numeric SCORE_ATTRIBUTE_NAME on residues via ChimeraX Python API."""
    if not score_map:
        return
    state = get_session_state(session)
    if not state.get("attr_registered"):
        try:
            Residue.register_attr(session, SCORE_ATTRIBUTE_NAME, "ESM_MSR", attr_type=float)
            state["attr_registered"] = True
        except Exception as e:
            session.logger.warning(f"Attribute registration note: {e}")

    res_lookup = build_residue_lookup(model)
    for key, val in score_map.items():
        if not np.isfinite(val):
            continue
        res = res_lookup.get(key)
        if res is not None:
            setattr(res, SCORE_ATTRIBUTE_NAME, float(val))


def clear_esm_msr_pbonds(
    session,
    names: Sequence[str] = (EPISTASIS_PBG_NAME, RANKED_PAIRS_PBG_NAME),
) -> None:
    """Delete ONLY ESM-MSR dedicated pseudobond groups (never user pseudobonds)."""
    name_set = set(names)
    for group in list(session.models.list(type=PseudobondGroup)):
        if getattr(group, "name", "") in name_set:
            try:
                session.models.close([group])
            except Exception as e:
                session.logger.warning(f"Failed closing pseudobond group '{group.name}': {e}")


def clear_esm_msr_legend_labels(session, prefix: str = LEGEND_LABEL_PREFIX) -> None:
    """Close ONLY ESM-MSR 2D legend LabelModels via session.models.close (ChimeraX 1.11 safe)."""
    matches = [
        m for m in list(session.models.list())
        if str(getattr(m, "name", "")).startswith(prefix)
    ]
    if matches:
        try:
            session.models.close(matches)
        except Exception as e:
            session.logger.warning(f"Failed closing ESM-MSR 2D legend labels: {e}")


def clear_esm_msr_clones(session, clone_ids: Optional[Sequence[str]] = None) -> None:
    """Close all tracked ESM-MSR mutant clone models."""
    state = get_session_state(session)
    to_close: List[str] = []
    for mid in list(clone_ids or []) + list(state.get("clone_ids", [])):
        if mid and mid not in to_close:
            to_close.append(mid)

    if to_close:
        existing_ids = {m.id_string for m in session.models.list()}
        for mid in to_close:
            if mid in existing_ids:
                try:
                    run(session, f"close #{mid}")
                except Exception as e:
                    session.logger.warning(f"Could not close generated model #{mid}: {e}")

    state["clone_ids"] = []
    state["pair_clone_map"] = {}


def clear_esm_msr_visuals(
    session,
    wt_model=None,
    clone_ids: Optional[Sequence[str]] = None,
    reset_wt: bool = True,
) -> None:
    """Clear ESM-MSR clones, dedicated pseudobond groups, 2D legend labels, and optionally reset WT model."""
    clear_esm_msr_pbonds(session)
    clear_esm_msr_legend_labels(session)
    clear_esm_msr_clones(session, clone_ids=clone_ids)

    if reset_wt and wt_model is not None:
        run(session, f"show #{wt_model.id_string} models")
        run(session, f"color #{wt_model.id_string} white")
        run(session, f"transparency #{wt_model.id_string} 0 target ac")
        run(session, f"ribbon style #{wt_model.id_string}")
        run(session, f"show #{wt_model.id_string} ribbons")
        run(session, f"hide #{wt_model.id_string} atoms")


def create_mutated_clone(
    session,
    wt_model,
    suffix: str = "viz",
    on_before_open: Optional[Callable[[], None]] = None,
    on_registered: Optional[Callable[[str], None]] = None,
    on_after_open: Optional[Callable[[], None]] = None,
):
    """Create a clean PDB clone of wt_model without leaking it into model comboboxes."""
    safe_id = wt_model.id_string.replace(":", "_").replace("/", "_")
    fd, temp_pdb = tempfile.mkstemp(prefix=f"chimerax_esm_msr_clone_{safe_id}_", suffix=".pdb")
    os.close(fd)

    if on_before_open is not None:
        on_before_open()
    try:
        try:
            run(session, f'save "{temp_pdb}" models #{wt_model.id_string} format pdb')
            safe_name = (wt_model.name or "model").replace('"', "")
            opened = run(session, f'open "{temp_pdb}" name "{safe_name}_{suffix}"')
        finally:
            if os.path.exists(temp_pdb):
                try:
                    os.remove(temp_pdb)
                except OSError:
                    pass

        mut_model = None
        if isinstance(opened, (list, tuple)):
            mut_model = next((m for m in opened if isinstance(m, Structure)), None)
        elif isinstance(opened, Structure):
            mut_model = opened

        if mut_model is None:
            atomic_models = session.models.list(type=Structure)
            mut_model = atomic_models[-1] if atomic_models else None
        if mut_model is None:
            return None

        mut_id = mut_model.id_string.split(".")[0]
        state = get_session_state(session)
        if mut_id not in state["clone_ids"]:
            state["clone_ids"].append(mut_id)
        if on_registered is not None:
            on_registered(mut_id)
    finally:
        if on_after_open is not None:
            on_after_open()

    return mut_model


def apply_swapaa(session, mutated_model_id: str, mutated_model, mutation_plan: Dict[Tuple[str, Tuple[int, str]], str]) -> None:
    res_lookup = build_residue_lookup(mutated_model)
    for (chain_val, pos_key), tgt_aa in mutation_plan.items():
        res_wt = res_lookup.get((chain_val, pos_key))
        three_letter = ONE_TO_THREE_LETTER_AA.get(tgt_aa, "")
        if res_wt is not None and three_letter and three_letter != res_wt.name:
            spec = f"#{mutated_model_id}/{chain_val}:{format_res_pos(pos_key)}"
            run(session, f"swapaa {spec} {three_letter.lower()} log false")


def _normalize_sort_by(sort_by: Optional[str]) -> str:
    raw = (sort_by or "combined_pred").strip().lower()
    aliases = {
        "combined": "combined_pred",
        "combined_pred": "combined_pred",
        "stability": "combined_pred",
        "ddg": "combined_pred",
        "combined_ddg": "combined_pred",
        "epi": "epi_score",
        "epi_pred": "epi_score",
        "epi_score": "epi_score",
        "epistasis": "epi_score",
        "stage1": "stage1_additive_pred",
        "additive": "stage1_additive_pred",
        "stage1_additive": "stage1_additive_pred",
        "stage1_additive_pred": "stage1_additive_pred",
        "rank_change": "rank_change",
        "rank_gain": "rank_change",
        "drank": "rank_change",
    }
    return aliases.get(raw, raw)


def _normalize_connectors(connectors: Optional[str]) -> str:
    raw = (connectors or "dashed").strip().lower()
    if raw.startswith("none") or raw == "off":
        return "none"
    if raw.startswith("solid"):
        return "solid"
    return "dashed"


def _normalize_mode(mode: Optional[str]) -> str:
    raw = (mode or "").strip().lower()
    if not raw:
        return ""
    if "pair" in raw or "top" in raw:
        return "Top-M Ranked Pairs"
    if "wt" in raw and "epi" in raw:
        return "WT epistasis"
    if "mt" in raw and "epi" in raw or raw == "epistasis":
        return "MT epistasis"
    if "single" in raw:
        return "Singles"
    return mode or ""


def _resolve_target_structure(session, models_arg=None, preferred_id: str = ""):
    state = get_session_state(session)
    clone_set = set(state.get("clone_ids", []))

    if models_arg:
        for m in models_arg:
            if isinstance(m, Structure) and m.id_string.split(".")[0] not in clone_set:
                return m
        for m in models_arg:
            if isinstance(m, Structure):
                return m

    candidates = [
        m for m in session.models.list(type=Structure)
        if m.id_string.split(".")[0] not in clone_set
    ]
    if preferred_id:
        match = next((m for m in candidates if m.id_string == preferred_id), None)
        if match is not None:
            return match
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Shared Publication-Grade Top-M Ranked Pairs Renderer (GUI + CLI identical)
# ---------------------------------------------------------------------------

def render_top_m_pairs(
    session,
    wt_model,
    parsed: ParsedCSVData,
    top_m: int = 5,
    sort_by: str = "combined_pred",
    priority: str = "High score",
    diverse_positions: bool = True,
    connectors: str = "dashed",
    show_labels_3d: bool = True,
    show_legend_2d: bool = True,
    pub_lighting: bool = True,
    target_chain: str = "",
    non_target_alpha: int = 90,
    hide_non_target: bool = False,
    isolate_rank: Optional[int] = None,
    existing_clone_ids: Optional[Sequence[str]] = None,
    on_before_open: Optional[Callable[[], None]] = None,
    on_registered: Optional[Callable[[str], None]] = None,
    on_after_open: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """Render Top-M double-mutant pairs with Okabe-Ito categorical colors, subtle connectors, and 2D legend.

    Used identically by both ESM_MSR_Tool._apply_top_m_pairs and CLI `esm_msr pairs`.
    """
    clear_esm_msr_pbonds(session)
    clear_esm_msr_legend_labels(session)
    clear_esm_msr_clones(session, clone_ids=existing_clone_ids)

    if wt_model is None:
        return {"ok": False, "error": "No atomic structure is open in ChimeraX.", "pairs": [], "clone_ids": []}

    norm_sort_by = _normalize_sort_by(sort_by)
    norm_conn = _normalize_connectors(connectors)
    eff_top_m = max(1, min(12, int(top_m)))

    ranked_pairs = select_top_m_pairs(
        parsed,
        top_m=eff_top_m,
        sort_by=norm_sort_by,
        priority=priority,
        diverse_positions=bool(diverse_positions),
    )
    if not ranked_pairs:
        return {
            "ok": False,
            "error": "No double-mutant pairs available in CSV for Top-M Ranked Pairs visualization.",
            "pairs": [],
            "clone_ids": [],
        }

    wt_id = wt_model.id_string
    eff_chain = (target_chain or parsed.selected_chain or ranked_pairs[0].chain or "").strip()

    # 1. Neutral matte backdrop
    if pub_lighting:
        run(session, "set bgColor white")
        run(session, "lighting soft")
        run(session, "graphics silhouettes true width 1.5")

    run(session, f"show #{wt_id} models")
    run(session, f"ribbon style #{wt_id}")
    run(session, f"show #{wt_id} ribbons")
    run(session, f"hide #{wt_id} atoms")

    if eff_chain:
        run(session, f"color #{wt_id}/{eff_chain} #D8DCE2 target c")
        run(session, f"transparency #{wt_id}/{eff_chain} 35 target c")
        has_non_target = any(c.chain_id != eff_chain for c in wt_model.chains)
        if has_non_target:
            nt_spec = f"#{wt_id} & ~#{wt_id}/{eff_chain}"
            if hide_non_target or non_target_alpha >= 99:
                run(session, f"hide {nt_spec} atoms")
                run(session, f"hide {nt_spec} ribbons")
            else:
                run(session, f"show {nt_spec} ribbons")
                run(session, f"color {nt_spec} #E5E7EB target c")
                run(session, f"transparency {nt_spec} {non_target_alpha} target ac")
    else:
        run(session, f"color #{wt_id} #D8DCE2 target c")
        run(session, f"transparency #{wt_id} 35 target c")

    # Prepare dedicated Ranked Pairs pseudobond group when connectors enabled
    pbg = None
    if norm_conn != "none":
        pbg = session.pb_manager.get_group(RANKED_PAIRS_PBG_NAME)
        if getattr(pbg, "id", None) is None:
            session.models.add([pbg])
        pbg.dashes = 6 if norm_conn == "dashed" else 0

    pair_clone_map: Dict[int, Dict[str, Any]] = {}
    clone_ids: List[str] = []

    # 2. Per-pair clones, sidechains, anchor ribbon highlights, 3D labels, and connectors
    for p in ranked_pairs:
        clone = create_mutated_clone(
            session,
            wt_model,
            suffix=f"pair_{p.rank}",
            on_before_open=on_before_open,
            on_registered=on_registered,
            on_after_open=on_after_open,
        )
        if clone is None:
            continue

        cid = clone.id_string.split(".")[0]
        clone_ids.append(cid)

        # Hide clone ribbon & all atoms to guarantee zero ribbon z-fighting
        run(session, f"hide #{cid} ribbons")
        run(session, f"hide #{cid} atoms")

        # Mutate ONLY this pair's two residues
        plan = {
            (p.chain, p.pos1): p.mut1,
            (p.chain, p.pos2): p.mut2,
        }
        apply_swapaa(session, cid, clone, plan)

        pos1_s = format_res_pos(p.pos1)
        pos2_s = format_res_pos(p.pos2)
        pair_spec = f"#{cid}/{p.chain}:{pos1_s},{pos2_s}"

        # Show ONLY mutated sidechains (~@N,C,O,OXT includes CA + sidechain, so Gly CA shows too)
        run(session, f"show ({pair_spec}) & ~@N,C,O,OXT & ~H atoms")
        run(session, f"style ({pair_spec}) stick")
        run(session, f"size ({pair_spec}) stickRadius 0.25")
        if p.mut1 == "G":
            run(session, f"style #{cid}/{p.chain}:{pos1_s} & @CA sphere")
            run(session, f"size #{cid}/{p.chain}:{pos1_s} & @CA atomRadius 0.45")
        if p.mut2 == "G":
            run(session, f"style #{cid}/{p.chain}:{pos2_s} & @CA sphere")
            run(session, f"size #{cid}/{p.chain}:{pos2_s} & @CA atomRadius 0.45")

        # Color sidechain carbons with pair categorical hue, keep heteroatoms (N, O, S) legible
        run(session, f"color ({pair_spec}) {p.color_hex} target a")
        run(session, f"color ({pair_spec}) byhetero target a")

        # Highlight anchor ribbon segments on the neutral WT backbone
        wt_pair_spec = f"#{wt_id}/{p.chain}:{pos1_s},{pos2_s}"
        run(session, f"color ({wt_pair_spec}) {p.color_hex} target c")
        run(session, f"transparency ({wt_pair_spec}) 0 target c")

        # 3D residue labels attached to clone residues (auto-hide on isolate / auto-close on reset)
        if show_labels_3d:
            t1 = f"#{p.rank} {p.mut1_str}".replace('"', "'")
            t2 = f"#{p.rank} {p.mut2_str}".replace('"', "'")
            try:
                run(
                    session,
                    f'label #{cid}/{p.chain}:{pos1_s} residues text "{t1}" '
                    f"color {p.color_hex} height 0.85 bgColor white",
                )
                run(
                    session,
                    f'label #{cid}/{p.chain}:{pos2_s} residues text "{t2}" '
                    f"color {p.color_hex} height 0.85 bgColor white",
                )
            except Exception as e:
                session.logger.warning(f"Could not place 3D label for pair #{p.rank}: {e}")

        # Subtle C-alpha to C-alpha pseudobond in dedicated group
        pb_obj = None
        if pbg is not None:
            res_lookup = build_residue_lookup(clone)
            r1 = res_lookup.get((p.chain, p.pos1))
            r2 = res_lookup.get((p.chain, p.pos2))
            if r1 is not None and r2 is not None:
                a1 = next((a for a in r1.atoms if a.name == "CA"), None) or (r1.atoms[0] if len(r1.atoms) else None)
                a2 = next((a for a in r2.atoms if a.name == "CA"), None) or (r2.atoms[0] if len(r2.atoms) else None)
                if a1 is not None and a2 is not None:
                    pb_obj = pbg.new_pseudobond(a1, a2)
                    pb_obj.color = hex_to_rgba(p.color_hex, alpha=215 if norm_conn == "dashed" else 235)
                    pb_obj.radius = 0.15 if norm_conn == "dashed" else 0.16
                    pb_obj.shown_when_atoms_hidden = True

        pair_clone_map[p.rank] = {
            "model_id": cid,
            "pair": p,
            "pseudobond": pb_obj,
        }

    # Hide hydrogens on WT + all pair clones
    all_specs = [f"#{wt_id}"] + [f"#{cid}" for cid in clone_ids]
    run(session, f"hide ({' | '.join(all_specs)}) & H")

    # 3. Stacked 2D Legend in top-left
    if show_legend_2d:
        # In offscreen/headless sessions, ChimeraX defaults to a 100x100 viewport which scales
        # 2dlabels ~10x on high-res save; ensure a crisp default canvas size before creating labels.
        is_gui = bool(getattr(getattr(session, "ui", None), "is_gui", False))
        if not is_gui:
            ws = getattr(getattr(session, "main_view", None), "window_size", None)
            if not ws or min(ws) < 400:
                try:
                    run(session, "windowsize 1400 950")
                except Exception:
                    pass

        sort_lbl = _SORT_BY_LABELS.get(norm_sort_by, norm_sort_by)
        hdr_text = f"Top-{len(ranked_pairs)} Pairs ({sort_lbl})"
        try:
            run(
                session,
                f'2dlabels create {LEGEND_LABEL_PREFIX}hdr text "{hdr_text}" '
                f"xpos 0.02 ypos 0.94 size 16 bold true color #111111 bgColor #ffffffdc margin 3",
            )
            step_y = 0.035 if len(ranked_pairs) <= 8 else 0.030
            for idx, p in enumerate(ranked_pairs, start=1):
                ypos = max(0.04, 0.94 - step_y * idx)
                sc_val = p.combined_pred if p.combined_pred is not None else p.score
                parts = [f"#{p.rank} {p.mut1_str} + {p.mut2_str}  \u0394\u0394G {sc_val:+.2f}"]
                if p.epi_score is not None and np.isfinite(p.epi_score):
                    parts.append(f"(epi {p.epi_score:+.2f})")
                if p.dist is not None and np.isfinite(p.dist):
                    parts.append(f"[{p.dist:.1f}\u00c5]")
                leg_text = " ".join(parts).replace('"', "'")
                run(
                    session,
                    f'2dlabels create {LEGEND_LABEL_PREFIX}{p.rank} text "{leg_text}" '
                    f"xpos 0.02 ypos {ypos:.3f} size 15 bold true color {p.color_hex} bgColor #ffffffdc margin 3",
                )
        except Exception as e:
            session.logger.warning(f"Could not draw 2D pair legend: {e}")

    # Persist in session state for isolate_ranked_pair / show_all_ranked_pairs / reset
    state = get_session_state(session)
    state["wt_model_id"] = wt_id
    state["target_chain"] = eff_chain
    state["pair_clone_map"] = pair_clone_map
    state["last_ranked_pairs"] = list(ranked_pairs)
    state["last_parsed"] = parsed
    state["last_csv_path"] = parsed.filepath
    state["last_mode"] = "Top-M Ranked Pairs"

    if isolate_rank is not None:
        isolate_ranked_pair(session, int(isolate_rank))

    return {
        "ok": True,
        "pairs": ranked_pairs,
        "clone_ids": clone_ids,
        "pair_clone_map": pair_clone_map,
    }


def isolate_ranked_pair(session, rank: int) -> bool:
    """Spotlight a single ranked pair: hide other pair clones/pbonds/legend rows and zoom to it."""
    state = get_session_state(session)
    pair_map: Dict[int, Dict[str, Any]] = state.get("pair_clone_map") or {}
    if int(rank) not in pair_map:
        return False

    target_entry = pair_map[int(rank)]
    target_pair: RankedPair = target_entry["pair"]
    wt_id = state.get("wt_model_id", "")
    eff_chain = state.get("target_chain") or target_pair.chain

    # Show only target clone; hide all other pair clones
    for r, entry in pair_map.items():
        cid = entry["model_id"]
        is_target = (r == int(rank))
        try:
            run(session, f"{'show' if is_target else 'hide'} #{cid} models")
        except Exception:
            pass
        pb = entry.get("pseudobond")
        if pb is not None:
            try:
                pb.display = is_target
            except Exception:
                pass

    # Filter 2D legend to header + target rank
    for m in list(session.models.list()):
        mname = str(getattr(m, "name", ""))
        if mname.startswith(LEGEND_LABEL_PREFIX):
            keep = (mname == f"{LEGEND_LABEL_PREFIX}hdr") or (mname == f"{LEGEND_LABEL_PREFIX}{int(rank)}")
            try:
                m.display = keep
            except Exception:
                pass

    # Re-apply neutral WT ribbon and highlight ONLY this pair's anchor residues
    if wt_id:
        pos1_s = format_res_pos(target_pair.pos1)
        pos2_s = format_res_pos(target_pair.pos2)
        if eff_chain:
            run(session, f"color #{wt_id}/{eff_chain} #D8DCE2 target c")
            run(session, f"transparency #{wt_id}/{eff_chain} 35 target c")
        wt_pair_spec = f"#{wt_id}/{target_pair.chain}:{pos1_s},{pos2_s}"
        run(session, f"color ({wt_pair_spec}) {target_pair.color_hex} target c")
        run(session, f"transparency ({wt_pair_spec}) 0 target c")
        try:
            run(session, f"view #{wt_id}/{target_pair.chain}:{pos1_s},{pos2_s} pad 0.35")
        except Exception:
            pass

    return True


def show_all_ranked_pairs(session) -> bool:
    """Restore visibility of all rendered Top-M pairs, connectors, legend rows, and anchor highlights."""
    state = get_session_state(session)
    pair_map: Dict[int, Dict[str, Any]] = state.get("pair_clone_map") or {}
    if not pair_map:
        return False

    wt_id = state.get("wt_model_id", "")
    eff_chain = state.get("target_chain", "")

    if wt_id and eff_chain:
        run(session, f"color #{wt_id}/{eff_chain} #D8DCE2 target c")
        run(session, f"transparency #{wt_id}/{eff_chain} 35 target c")

    for r, entry in pair_map.items():
        cid = entry["model_id"]
        p: RankedPair = entry["pair"]
        try:
            run(session, f"show #{cid} models")
        except Exception:
            pass
        pb = entry.get("pseudobond")
        if pb is not None:
            try:
                pb.display = True
            except Exception:
                pass
        if wt_id:
            pos1_s = format_res_pos(p.pos1)
            pos2_s = format_res_pos(p.pos2)
            wt_pair_spec = f"#{wt_id}/{p.chain}:{pos1_s},{pos2_s}"
            run(session, f"color ({wt_pair_spec}) {p.color_hex} target c")
            run(session, f"transparency ({wt_pair_spec}) 0 target c")

    for m in list(session.models.list()):
        mname = str(getattr(m, "name", ""))
        if mname.startswith(LEGEND_LABEL_PREFIX):
            try:
                m.display = True
            except Exception:
                pass

    if wt_id:
        try:
            run(session, f"view #{wt_id}")
        except Exception:
            pass
    return True


# ---------------------------------------------------------------------------
# Headless Singles & Epistasis renderers (used by CLI when GUI tool is closed)
# ---------------------------------------------------------------------------

def _render_singles_headless(
    session,
    wt_model,
    parsed: ParsedCSVData,
    palette_key: str = DEFAULT_PALETTE_KEY,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    pos_threshold: float = 0.0,
    neg_threshold: float = 0.0,
) -> Dict[str, Any]:
    clear_esm_msr_pbonds(session)
    clear_esm_msr_legend_labels(session)
    clear_esm_msr_clones(session)

    wt_id = wt_model.id_string
    run(session, f"show #{wt_id} models")
    run(session, f"color #{wt_id} white")
    run(session, f"transparency #{wt_id} 0 target ac")
    run(session, f"ribbon style #{wt_id}")
    run(session, f"show #{wt_id} ribbons")
    run(session, f"hide #{wt_id} atoms")

    pal = get_color_palette(palette_key)
    scores = [s for s, _ in parsed.residue_scores_data.values() if np.isfinite(s)]
    auto = (min_score is None or max_score is None)
    vmin, vmax = compute_score_range(scores, pal, auto_range=auto, manual_min=min_score, manual_max=max_score)
    color_range = f"{vmin:.3f},{vmax:.3f}"

    wt_score_map = {k: score for k, (score, _) in parsed.residue_scores_data.items()}
    set_residue_scores(session, wt_model, wt_score_map)

    chains_present = sorted({c for c, _ in parsed.residue_scores_data.keys()})
    if chains_present:
        chain_spec = ",".join(chains_present)
        run(
            session,
            f"color byattribute r:{SCORE_ATTRIBUTE_NAME} #{wt_id}/{chain_spec} & backbone "
            f"palette {pal['chimerax']} range {color_range} key false",
        )

    mutation_plan = {
        k: tgt_aa
        for k, (score, tgt_aa) in parsed.residue_scores_data.items()
        if score >= pos_threshold or score <= neg_threshold
    }
    if mutation_plan:
        clone = create_mutated_clone(session, wt_model, suffix="singles_viz")
        if clone is not None:
            cid = clone.id_string.split(".")[0]
            run(session, f"hide #{cid} ribbons")
            run(session, f"hide #{cid} atoms")
            apply_swapaa(session, cid, clone, mutation_plan)
            mut_score_map = {
                k: parsed.all_singles_scores.get((k[0], k[1], aa), 0.0)
                for k, aa in mutation_plan.items()
            }
            set_residue_scores(session, clone, mut_score_map)
            spec = get_spec(cid, mutation_plan.keys())
            if spec != "None":
                run(session, f"show ({spec}) & ~@N,C,O,OXT & ~H atoms")
                run(session, f"style ({spec}) stick")
                run(session, f"size ({spec}) stickRadius 0.2")
                run(
                    session,
                    f"color byattribute r:{SCORE_ATTRIBUTE_NAME} ({spec}) "
                    f"palette {pal['chimerax']} range {color_range} key false target a",
                )
                run(session, f"color ({spec}) byhetero target a")

    try:
        run(session, build_key_command(pal, vmin, vmax))
    except Exception as e:
        session.logger.warning(f"Failed to draw colorbar key: {e}")

    return {"ok": True, "n_mutations": len(mutation_plan)}


def _render_epistasis_headless(
    session,
    wt_model,
    parsed: ParsedCSVData,
    palette_key: str = DEFAULT_PALETTE_KEY,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    pos_threshold: float = 0.0,
    neg_threshold: float = 0.0,
) -> Dict[str, Any]:
    clear_esm_msr_pbonds(session)
    clear_esm_msr_legend_labels(session)
    clear_esm_msr_clones(session)

    df = parsed.epistasis_df
    if df is None or df.empty:
        return {"ok": False, "error": "No valid epistasis double-mutant rows in CSV."}

    wt_id = wt_model.id_string
    run(session, f"show #{wt_id} models")
    run(session, f"color #{wt_id} white")
    run(session, f"transparency #{wt_id} 0 target ac")
    run(session, f"ribbon style #{wt_id}")
    run(session, f"show #{wt_id} ribbons")
    run(session, f"hide #{wt_id} atoms")

    pal = get_color_palette(palette_key)
    filt = df[(df["dddg_pred"] >= pos_threshold) | (df["dddg_pred"] <= neg_threshold)].copy()
    if filt.empty:
        return {"ok": False, "error": "No epistasis pairs passed the specified thresholds."}

    epi_vals = filt["dddg_pred"].to_numpy(dtype=float)
    auto = (min_score is None or max_score is None)
    vmin, vmax = compute_score_range(epi_vals, pal, auto_range=auto, manual_min=min_score, manual_max=max_score)
    max_abs = max(abs(vmin), abs(vmax), 0.01)

    pbg = session.pb_manager.get_group(EPISTASIS_PBG_NAME)
    if getattr(pbg, "id", None) is None:
        session.models.add([pbg])
    pbg.dashes = 0

    wt_lookup = build_residue_lookup(wt_model)
    drawn = 0
    seen_pairs = set()
    for row in filt.sort_values(by="dddg_pred", key=lambda s: s.abs(), ascending=False).itertuples(index=False):
        c = str(row.chain_id).strip()
        p1, p2 = row.pos1_key, row.pos2_key
        pk = tuple(sorted([(c, p1), (c, p2)]))
        if pk in seen_pairs:
            continue
        r1 = wt_lookup.get((c, p1))
        r2 = wt_lookup.get((c, p2))
        if r1 is None or r2 is None:
            continue
        a1 = next((a for a in r1.atoms if a.name == "CA"), None)
        a2 = next((a for a in r2.atoms if a.name == "CA"), None)
        if a1 is None or a2 is None:
            continue
        sc = float(row.dddg_pred)
        norm = max(0.0, min(1.0, (sc - vmin) / max(vmax - vmin, 1e-6)))
        mag = min(1.0, abs(sc) / max_abs)
        pb = pbg.new_pseudobond(a1, a2)
        pb.color = interpolate_palette_rgba(pal, norm, alpha=int(round(255 * (0.35 + 0.65 * mag))))
        pb.radius = 0.08 + 0.28 * mag
        pb.shown_when_atoms_hidden = True
        seen_pairs.add(pk)
        drawn += 1

    try:
        run(session, build_key_command(pal, vmin, vmax))
    except Exception:
        pass

    return {"ok": True, "n_pairs": drawn}


# ---------------------------------------------------------------------------
# Native ChimeraX CLI Command Handlers
# ---------------------------------------------------------------------------

def cmd_esm_msr_open(session):
    """Open or raise the ESM-MSR Predictor & Visualizer GUI tool panel."""
    tool = _get_open_tool(session)
    if tool is not None:
        try:
            tool.tool_window.shown = True
        except Exception:
            pass
        session.logger.info("ESM-MSR tool panel is already open.")
        return tool

    from .tool import ESM_MSR_Tool
    tool = ESM_MSR_Tool(session, "ESM_MSR")
    session.logger.info("Opened ESM-MSR Predictor & Visualizer tool.")
    return tool


def cmd_esm_msr_load(session, csv_path, model=None, chain=None, mode=None):
    """Load an ESM-MSR prediction CSV and visualize it (works in GUI and headless modes)."""
    norm_mode = _normalize_mode(mode)
    tool = _get_open_tool(session)

    if tool is not None:
        if model:
            wt = _resolve_target_structure(session, models_arg=model)
            if wt is not None:
                idx = tool.viz_model_combobox.findData(wt.id_string)
                if idx >= 0:
                    tool.viz_model_combobox.setCurrentIndex(idx)
        if chain:
            tool.viz_chain_id_combobox.setCurrentText(str(chain).strip())
        if norm_mode:
            idx = tool.display_mode_combo.findText(norm_mode)
            if idx >= 0:
                tool.display_mode_combo.setCurrentIndex(idx)
        ok = tool._load_csv_file(csv_path)
        if not ok:
            session.logger.error(f"esm_msr load failed for '{csv_path}'.")
        return ok

    # Headless path (no Qt GUI tool open)
    wt_model = _resolve_target_structure(session, models_arg=model)
    target_chain = (chain or "").strip()
    eff_mode = norm_mode or "Singles"

    parsed = parse_esm_msr_csv(
        filepath=csv_path,
        mode=eff_mode,
        target_chain=target_chain,
    )
    if not parsed.ok:
        session.logger.error(f"esm_msr load error: {parsed.error}")
        return False

    state = get_session_state(session)
    state["last_csv_path"] = csv_path
    state["last_parsed"] = parsed
    state["last_mode"] = eff_mode
    state["target_chain"] = target_chain or parsed.selected_chain
    if wt_model is not None:
        state["wt_model_id"] = wt_model.id_string

    session.logger.info(
        f"Loaded '{os.path.basename(csv_path)}': {parsed.n_singles} singles ({parsed.n_positions} pos), "
        f"{parsed.n_doubles} doubles."
    )

    if wt_model is not None:
        if eff_mode == "Top-M Ranked Pairs":
            render_top_m_pairs(session, wt_model, parsed, target_chain=target_chain)
        elif "epistasis" in eff_mode.lower():
            _render_epistasis_headless(session, wt_model, parsed)
        else:
            _render_singles_headless(session, wt_model, parsed)
    return True


def cmd_esm_msr_visualize(
    session,
    mode=None,
    palette=None,
    minScore=None,
    maxScore=None,
    posThreshold=None,
    negThreshold=None,
):
    """Run Singles / Epistasis / Top-M visualization with custom palette & range."""
    norm_mode = _normalize_mode(mode)
    tool = _get_open_tool(session)

    if tool is not None:
        if norm_mode:
            idx = tool.display_mode_combo.findText(norm_mode)
            if idx >= 0:
                tool.display_mode_combo.setCurrentIndex(idx)
        if palette:
            pal = get_color_palette(palette)
            pidx = tool.palette_combobox.findData(pal["key"])
            if pidx >= 0:
                tool.palette_combobox.setCurrentIndex(pidx)
        if minScore is not None or maxScore is not None:
            tool.auto_range_checkbox.setChecked(False)
            if minScore is not None:
                tool.range_min_spinbox.setValue(float(minScore))
            if maxScore is not None:
                tool.range_max_spinbox.setValue(float(maxScore))
        if posThreshold is not None:
            tool.pos_threshold_spinbox.setValue(float(posThreshold))
        if negThreshold is not None:
            tool.neg_threshold_spinbox.setValue(float(negThreshold))
        tool._update_visualization()
        return True

    state = get_session_state(session)
    parsed = state.get("last_parsed")
    if parsed is None:
        session.logger.error("esm_msr visualize: no CSV loaded. Run `esm_msr load <csv_path>` first.")
        return False

    wt_model = _resolve_target_structure(session, preferred_id=state.get("wt_model_id", ""))
    if wt_model is None:
        session.logger.error("esm_msr visualize: no atomic structure open in ChimeraX.")
        return False

    eff_mode = norm_mode or state.get("last_mode", "Singles")
    pal_key = get_color_palette(palette)["key"] if palette else DEFAULT_PALETTE_KEY
    pos_t = float(posThreshold) if posThreshold is not None else 0.0
    neg_t = float(negThreshold) if negThreshold is not None else 0.0

    if eff_mode == "Top-M Ranked Pairs":
        res = render_top_m_pairs(session, wt_model, parsed, target_chain=state.get("target_chain", ""))
    elif "epistasis" in eff_mode.lower():
        res = _render_epistasis_headless(
            session, wt_model, parsed,
            palette_key=pal_key, min_score=minScore, max_score=maxScore,
            pos_threshold=pos_t, neg_threshold=neg_t,
        )
    else:
        res = _render_singles_headless(
            session, wt_model, parsed,
            palette_key=pal_key, min_score=minScore, max_score=maxScore,
            pos_threshold=pos_t, neg_threshold=neg_t,
        )

    if not res.get("ok"):
        session.logger.error(f"esm_msr visualize error: {res.get('error', 'unknown error')}")
        return False
    return True


def cmd_esm_msr_pairs(
    session,
    csv_path,
    model=None,
    chain=None,
    topM=5,
    sortBy="combined_pred",
    diverse=True,
    connectors="dashed",
    labels=True,
    legend=True,
    pubLighting=True,
    isolate=None,
):
    """Scriptable publication-grade categorical Top-M double-mutant pair visualizer (GUI + offscreen)."""
    wt_model = _resolve_target_structure(session, models_arg=model)
    if wt_model is None:
        session.logger.error("esm_msr pairs: no open atomic structure found in ChimeraX.")
        return []

    target_chain = (chain or "").strip()
    parsed = parse_esm_msr_csv(
        filepath=csv_path,
        mode="Top-M Ranked Pairs",
        target_chain=target_chain,
    )
    if not parsed.ok:
        session.logger.error(f"esm_msr pairs: failed to parse '{csv_path}': {parsed.error}")
        return []

    tool = _get_open_tool(session)
    on_before = None
    on_reg = None
    on_after = None
    existing_clones = None
    if tool is not None:
        existing_clones = list(tool.mutated_model_id_strings)
        tool.mutated_model_id_strings = []

        def _before():
            tool._suppress_model_refresh = True

        def _reg(mid):
            if mid not in tool.mutated_model_id_strings:
                tool.mutated_model_id_strings.append(mid)

        def _after():
            tool._suppress_model_refresh = False

        on_before, on_reg, on_after = _before, _reg, _after

    res = render_top_m_pairs(
        session=session,
        wt_model=wt_model,
        parsed=parsed,
        top_m=topM,
        sort_by=sortBy,
        diverse_positions=diverse,
        connectors=connectors,
        show_labels_3d=labels,
        show_legend_2d=legend,
        pub_lighting=pubLighting,
        target_chain=target_chain,
        isolate_rank=isolate,
        existing_clone_ids=existing_clones,
        on_before_open=on_before,
        on_registered=on_reg,
        on_after_open=on_after,
    )

    if not res.get("ok"):
        session.logger.error(f"esm_msr pairs: {res.get('error')}")
        return []

    pairs: List[RankedPair] = res["pairs"]
    if tool is not None:
        tool.loaded_csv_path = csv_path
        tool.last_parsed_csv = parsed
        tool.current_ranked_pairs = list(pairs)
        tool.update_viz_button.setEnabled(True)
        tool.csv_label.setText(tool._format_csv_summary(parsed))
        tool._populate_results_table()
        tool._refresh_export_buttons()

    summary_lines = [f"Rendered Top-{len(pairs)} ESM-MSR pairs from '{os.path.basename(csv_path)}':"]
    for p in pairs:
        sc_val = p.combined_pred if p.combined_pred is not None else p.score
        epi_s = f", epi={p.epi_score:+.2f}" if p.epi_score is not None else ""
        dist_s = f", dist={p.dist:.1f}A" if p.dist is not None else ""
        summary_lines.append(f"  #{p.rank} [{p.color_hex}] {p.mut1_str} + {p.mut2_str}: ddG={sc_val:+.2f}{epi_s}{dist_s}")
    session.logger.info("\n".join(summary_lines))
    return pairs


def cmd_esm_msr_heatmap(
    session,
    csv_path,
    output_path,
    chain=None,
    palette=None,
    minScore=None,
    maxScore=None,
    title=None,
):
    """Export a 2D single-mutant landscape heatmap image directly from ChimeraX CLI."""
    pal_key = get_color_palette(palette)["key"] if palette else DEFAULT_PALETTE_KEY
    out_abs = export_mutational_heatmap(
        parsed_csv=csv_path,
        output_path=output_path,
        title=title,
        palette_key=pal_key,
        vmin=minScore,
        vmax=maxScore,
        chain=chain,
    )
    session.logger.info(f"Exported ESM-MSR 2D mutational heatmap to '{out_abs}'.")
    return out_abs


def cmd_esm_msr_reset(session):
    """Clear ESM-MSR clones, dedicated pseudobond groups, and 2D legend labels."""
    tool = _get_open_tool(session)
    if tool is not None:
        tool._reset_visualization()
        return

    state = get_session_state(session)
    wt_model = _resolve_target_structure(session, preferred_id=state.get("wt_model_id", ""))
    clear_esm_msr_visuals(session, wt_model=wt_model, reset_wt=True)
    session.logger.info("Cleared ESM-MSR clones, pseudobonds, and 2D legend labels.")


# ---------------------------------------------------------------------------
# Command Registration
# ---------------------------------------------------------------------------

def register_command(command_name: str, logger) -> None:
    """Register individual ChimeraX CLI commands declared in bundle_info.xml."""
    if command_name == "esm_msr open":
        desc = CmdDesc(
            synopsis="Open or raise the ESM-MSR Predictor & Visualizer GUI tool",
        )
        register(command_name, desc, cmd_esm_msr_open, logger=logger)
    elif command_name == "esm_msr load":
        desc = CmdDesc(
            required=[("csv_path", OpenFileNameArg)],
            keyword=[
                ("model", ModelsArg),
                ("chain", StringArg),
                ("mode", StringArg),
            ],
            synopsis="Load an ESM-MSR prediction CSV and visualize it on the target structure",
        )
        register(command_name, desc, cmd_esm_msr_load, logger=logger)
    elif command_name == "esm_msr visualize":
        desc = CmdDesc(
            keyword=[
                ("mode", StringArg),
                ("palette", StringArg),
                ("minScore", FloatArg),
                ("maxScore", FloatArg),
                ("posThreshold", FloatArg),
                ("negThreshold", FloatArg),
            ],
            synopsis="Visualize loaded ESM-MSR scores with custom mode, palette, and range",
        )
        register(command_name, desc, cmd_esm_msr_visualize, logger=logger)
    elif command_name == "esm_msr pairs":
        desc = CmdDesc(
            required=[("csv_path", OpenFileNameArg)],
            keyword=[
                ("model", ModelsArg),
                ("chain", StringArg),
                ("topM", IntArg),
                ("sortBy", StringArg),
                ("diverse", BoolArg),
                ("connectors", StringArg),
                ("labels", BoolArg),
                ("legend", BoolArg),
                ("pubLighting", BoolArg),
                ("isolate", IntArg),
            ],
            synopsis="Render Top-M ranked double-mutant pairs with categorical colors, connectors, and 2D legend",
        )
        register(command_name, desc, cmd_esm_msr_pairs, logger=logger)
    elif command_name == "esm_msr heatmap":
        desc = CmdDesc(
            required=[
                ("csv_path", OpenFileNameArg),
                ("output_path", SaveFileNameArg),
            ],
            keyword=[
                ("chain", StringArg),
                ("palette", StringArg),
                ("minScore", FloatArg),
                ("maxScore", FloatArg),
                ("title", StringArg),
            ],
            synopsis="Export a publication-quality 2D single-mutant landscape heatmap image",
        )
        register(command_name, desc, cmd_esm_msr_heatmap, logger=logger)
    elif command_name == "esm_msr reset":
        desc = CmdDesc(
            synopsis="Clear ESM-MSR mutant clones, pseudobond groups, and 2D legend labels",
        )
        register(command_name, desc, cmd_esm_msr_reset, logger=logger)
