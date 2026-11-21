#!/usr/bin/env python3
"""Generate publication-ready ESM-MSR schematics (Vertical primary + Horizontal companion).

Incorporates all latest user refinements on the Vertical figure (and synced Horizontal):
  1. Nature-style subplot label 'a' (bold lowercase sans-serif in upper-left corner).
  2. Real 3D PDB domain render (GB1 domain, SPG1_STRSG residues 228-282, projected from
     atomic N/CA/C/O coordinates via smooth cubic B-spline peptide-plane ribbons with
     3D Lambertian lighting and painter's-algorithm depth sorting in pure vector SVG).
  3. Canonical two-tongue ML trainable flame icon with bright inner core.
  4. 'WT Structure' updated to 'WT BB Coords'.
  5. 4-layer z-ordering (bg_panels -> under_header_wires -> headers -> body) so inter-panel
     lines pass cleanly BEHIND the dark section header bars instead of through header text.
  6. 'Cached' badge placed on the left side of the WT Logits (L x 20) matrix.
  7. Strict 100% orthogonal (straight or 90-degree rounded-bend) routing enforced by a
     programmatic assertion on every segment — including redrawn LoRA emission lines,
     orthogonal site-diff summation, stacked ΔΔG_A / ΔΔG_B in the upper-left of Panel 4,
     and strictly parallel horizontal [TF] emission lines.

Outputs written to <repo>/_assets/ and copied as PNG previews to <artifact_dir>/:
  - esm_msr_vertical.{svg,png}
  - esm_msr_horizontal.{svg,png}
"""

from __future__ import annotations

import ctypes
import html
import math
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFont
from scipy.interpolate import CubicSpline

OUT_DIR = Path(__file__).resolve().parent

FONT_SANS = "'Liberation Sans', 'DejaVu Sans', Helvetica, Arial, sans-serif"
FONT_MATH = "'DejaVu Serif', 'Liberation Serif', Georgia, serif"
FONT_MONO = "'Liberation Mono', 'DejaVu Sans Mono', monospace"

PIL_FONT_PATHS = {
    ("sans", False): "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ("sans", True): "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ("mono", False): "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ("mono", True): "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
}
_FONT_CACHE: Dict[Tuple[str, bool, int], ImageFont.FreeTypeFont] = {}
MIN_FONT_PX = 11.5
FORBIDDEN_SUBSTRINGS = ("350", "faster", "6 min", "6m", "35 h", "34.8", "0.10")

_BARE_AMP_RE = re.compile(r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)")


def _get_font(family: str, bold: bool, size: float) -> ImageFont.FreeTypeFont:
    key = (family, bold, max(8, int(round(size))))
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(PIL_FONT_PATHS[(family, bold)], key[2])
    return _FONT_CACHE[key]


def measure_text_px(text: str, size: float, family: str = "sans", bold: bool = False) -> float:
    f = _get_font(family, bold, size)
    bbox = f.getbbox(text)
    return float(bbox[2] - bbox[0]) if bbox else 0.0


def _parse_markup(markup: str) -> ET.Element:
    safe = _BARE_AMP_RE.sub("&amp;", markup)
    return ET.fromstring(f"<r>{safe}</r>")


def _emit_rich_node(node: ET.Element, base_size: float, out: List[str]) -> None:
    sub_sz = max(9.8, round(base_size * 0.76, 1))
    tag = node.tag
    attrs: List[str] = []
    if tag == "b":
        attrs.append('font-weight="700"')
    elif tag == "i":
        attrs.append(f'font-family="{FONT_MATH}" font-style="italic"')
    elif tag == "mono":
        attrs.append(f'font-family="{FONT_MONO}"')
    elif tag == "sub":
        attrs.append(f'font-size="{sub_sz}" baseline-shift="sub"')
    elif tag == "sup":
        attrs.append(f'font-size="{sub_sz}" baseline-shift="super"')

    if tag != "r":
        out.append(f'<tspan {" ".join(attrs)}>')
    if node.text:
        out.append(html.escape(node.text))
    for child in node:
        _emit_rich_node(child, base_size, out)
        if child.tail:
            out.append(html.escape(child.tail))
    if tag != "r":
        out.append("</tspan>")


def render_rich_runs(markup: str, base_size: float) -> str:
    root = _parse_markup(markup)
    out: List[str] = []
    _emit_rich_node(root, base_size, out)
    return "".join(out)


def strip_markup(markup: str) -> str:
    return "".join(_parse_markup(markup).itertext())


# ---------------------------------------------------------------------------
# Journal Palette (Warm Stone + Forest-Sage WT + Gold/Amber MT + Purple Epistasis)
# ---------------------------------------------------------------------------
PAL = {
    "ink": "#1C1917",
    "ink_mid": "#44403C",
    "ink_muted": "#57534E",
    "hdr_bg": "#292524",
    "hdr_fg": "#FAFAF9",
    "p1_bg": "#FAF9F6", "p1_bdr": "#D6D3D1",
    "p2_bg": "#F5F3EE", "p2_bdr": "#D6D3D1",
    "p3_bg": "#FAF8F5", "p3_bdr": "#D6D3D1",
    "p4_bg": "#F5F6F4", "p4_bdr": "#CBD5CB",
    "frozen_bg": "#E7E5E4", "frozen_bdr": "#57534E", "ice": "#0369A1",
    "wt_dark": "#166534",
    "wt_mid": "#15803D",
    "wt_soft": "#DCFCE7",
    "wt_card": "#F0FDF4",
    "wt_card_bdr": "#86EFAC",
    "wt_ink": "#14532D",
    "mt_lora": "#92400E",
    "mt_dark": "#B45309",
    "mt_mid": "#D97706",
    "mt_soft": "#FEF3C7",
    "mt_pred": "#FDE68A",
    "mt_card": "#FFFBEB",
    "mt_card_bdr": "#FCD34D",
    "mt_ink": "#78350F",
    "epi_soft": "#EDE9FE",
    "epi_pred": "#DDD6FE",
    "epi_bdr": "#6D28D9",
    "epi_ink": "#4C1D95",
    "gt_bg": "#E7E5E4",
    "gt_bdr": "#57534E",
    "loss_bg": "#FCE7F3",
    "loss_bdr": "#9D174D",
    "loss_ink": "#831843",
    "seq_bg": "#F5F5F4",
    "seq_bdr": "#A8A29E",
}


# ---------------------------------------------------------------------------
# Real 3D PDB Domain Ribbon Renderer (GB1 Domain: SPG1_STRSG residues 228-282)
# ---------------------------------------------------------------------------
# Embedded PCA-aligned backbone (CA, C, O) coordinates of GB1 domain (SPG1_STRSG residues 228-282)
_GB1_EMBEDDED = np.array([
    (-10.6453, 0.0743, 5.9507, -9.7715, -0.8126, 5.0543, -9.9325, -2.0322, 5.0161),
    (-7.8233, -0.8661, 3.5354, -6.5025, -1.0701, 4.2826, -6.1518, -0.3097, 5.1915),
    (-4.3999, -2.4102, 4.3715, -3.3425, -2.1960, 3.2862, -3.6121, -2.4434, 2.1145),
    (-0.9262, -1.8164, 2.8641, 0.1431, -2.6714, 3.5473, 0.5060, -2.4153, 4.6942),
    (1.8767, -4.4049, 3.1688, 3.0605, -3.8572, 2.3657, 3.0303, -3.8429, 1.1334),
    (5.3715, -2.9982, 2.4464, 6.3935, -4.1297, 2.5464, 6.8460, -4.4824, 3.6358),
    (7.8423, -5.6442, 1.2127, 9.0714, -4.9688, 0.5807, 9.5333, -5.3676, -0.4885),
    (10.7918, -3.2228, 0.8121, 12.0743, -3.9400, 1.2363, 12.0828, -4.8123, 2.1084),
    (14.5623, -3.8346, 1.0726, 14.9487, -2.9122, 2.2304, 15.5982, -3.3607, 3.1696),
    (14.7722, -0.5793, 3.1326, 13.7219, -0.5130, 4.2422, 14.0786, -0.3815, 5.4117),
    (11.3137, -0.5349, 4.8271, 10.3498, -1.7144, 4.6681, 9.8842, -2.0213, 3.5711),
    (9.0451, -3.4491, 5.8694, 8.0304, -3.1613, 6.9619, 8.4137, -2.7111, 8.0399),
    (5.7067, -3.2683, 7.7008, 4.2970, -3.3342, 7.1287, 4.0977, -3.5597, 5.9350),
    (1.9108, -2.9696, 7.6796, 1.4366, -1.5615, 8.0679, 1.8667, -1.0169, 9.0844),
    (-0.2340, 0.2111, 7.6587, -1.6560, 0.1282, 7.0961, -1.9476, -0.7250, 6.2564),
    (-3.9843, 0.9721, 7.1692, -4.4923, 2.3794, 6.8645, -3.9410, 3.3765, 7.3270),
    (-6.2530, 3.7345, 5.7843, -7.7419, 3.4816, 5.5791, -8.1250, 2.4280, 5.0756),
    (-10.0058, 4.4348, 5.6017, -10.1963, 5.2075, 4.2921, -9.6588, 6.3055, 4.1282),
    (-11.2066, 5.2919, 2.0548, -12.5740, 4.9002, 1.4834, -13.0786, 3.8112, 1.7350),
    (-14.4834, 5.5729, 0.0436, -14.4520, 4.5522, -1.0990, -15.4913, 4.0324, -1.4936),
    (-13.0178, 3.2879, -2.6796, -11.5306, 2.8895, -2.7218, -10.6460, 3.5883, -2.2085),
    (-9.8973, 1.2299, -3.5221, -8.9649, 2.1738, -4.3050, -7.7915, 2.2811, -3.9721),
    (-8.6717, 3.8600, -6.0834, -8.2734, 5.1215, -5.2888, -7.3960, 5.8824, -5.7042),
    (-8.6191, 6.4381, -3.2050, -7.6070, 5.9326, -2.1817, -6.5965, 6.6040, -1.9689),
    (-6.7744, 4.0368, -0.8415, -5.4379, 3.8715, -1.5910, -4.3865, 4.2194, -1.0606),
    (-4.2890, 3.2584, -3.7065, -3.4707, 4.5534, -3.8411, -2.2445, 4.5153, -3.7756),
    (-3.4455, 7.0213, -4.0603, -2.7612, 7.3822, -2.7450, -1.6174, 7.8266, -2.7709),
    (-2.8812, 7.4185, -0.2734, -1.6756, 6.5094, -0.0346, -0.6184, 6.9815, 0.3751),
    (-0.7143, 4.2556, -0.2083, 0.4602, 4.4927, -1.1609, 1.6039, 4.3936, -0.7218),
    (1.2961, 5.2546, -3.3536, 2.0220, 6.5178, -2.9036, 3.2458, 6.5473, -2.9664),
    (1.9451, 8.6913, -1.7876, 2.7939, 8.2924, -0.5717, 3.9677, 8.6363, -0.5203),
    (2.9674, 7.0003, 1.5231, 4.2288, 6.1973, 1.1642, 5.2729, 6.3821, 1.7855),
    (5.3155, 4.5596, -0.3086, 6.3945, 5.4733, -0.9216, 7.5576, 5.3790, -0.5294),
    (6.8940, 7.4504, -2.3316, 7.5668, 8.2990, -1.2358, 8.7771, 8.5103, -1.2837),
    (7.3120, 9.5411, 0.9025, 8.3249, 8.7644, 1.7684, 9.1896, 9.3655, 2.4069),
    (9.1907, 6.5220, 2.4206, 10.2929, 5.9867, 1.4781, 11.1323, 5.2016, 1.9157),
    (11.3060, 5.9453, -0.7826, 11.1531, 4.4916, -1.2512, 12.1300, 3.9093, -1.7153),
    (9.6474, 2.5418, -1.5968, 8.8804, 2.6202, -2.9189, 7.6838, 2.9025, -2.9242),
    (8.9178, 2.0684, -5.3301, 9.0537, 0.5825, -5.6991, 9.9710, -0.1077, -5.2489),
    (8.0600, -1.3908, -6.7248, 6.7907, -1.8943, -7.3978, 5.9767, -1.1442, -7.9350),
    (5.4876, -3.8908, -7.9817, 4.2938, -3.8767, -7.0176, 4.3787, -4.3591, -5.8848),
    (1.9509, -3.1116, -6.7241, 0.9474, -4.2116, -7.0642, 0.5758, -4.3792, -8.2260),
    (-0.7140, -5.7972, -6.1484, -1.8309, -5.3538, -5.2039, -1.5943, -4.6485, -4.2215),
    (-4.2484, -5.4758, -4.7042, -5.1277, -6.7254, -4.6465, -5.5068, -7.2700, -5.6844),
    (-6.3794, -8.2480, -3.1524, -7.6969, -7.6551, -2.6269, -7.7613, -7.0966, -1.5290),
    (-10.0633, -7.1721, -3.0923, -10.8351, -7.9582, -2.0186, -11.6715, -7.3833, -1.3182),
    (-11.1844, -10.0876, -0.8289, -10.7104, -9.7383, 0.5937, -11.4674, -9.8713, 1.5532),
    (-8.8258, -8.8391, 1.9697, -8.6045, -7.3247, 2.0609, -7.9845, -6.8642, 3.0207),
    (-8.8595, -5.0896, 0.9228, -7.3942, -4.6673, 1.1299, -7.1031, -3.6083, 1.6926),
    (-5.0317, -5.3935, 0.9924, -4.2286, -5.0766, -0.2636, -4.1750, -5.8606, -1.2080),
    (-2.4870, -3.5962, -1.1790, -1.1712, -4.2028, -0.6961, -0.8796, -4.2039, 0.4991),
    (1.0569, -5.0157, -1.3410, 1.9462, -4.2624, -2.3158, 1.7294, -4.3368, -3.5234),
    (4.0667, -3.0449, -2.6150, 5.3162, -3.8612, -2.3140, 5.7557, -3.9524, -1.1676),
    (7.1255, -5.2477, -3.2406, 8.1937, -4.4749, -3.9881, 8.0855, -4.3153, -5.2026),
    (10.3327, -3.2739, -3.8836, 11.0045, -4.2372, -4.8591, 11.5598, -5.2711, -4.4707),
], dtype=float)

def _load_gb1_coords() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _GB1_EMBEDDED[:, 0:3], _GB1_EMBEDDED[:, 3:6], _GB1_EMBEDDED[:, 6:9]
    atoms: Dict[int, Dict[str, np.ndarray]] = {}
    with open(GB1_PDB_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            resi = int(line[22:26])
            if 228 <= resi <= 282:
                aname = line[12:16].strip()
                if aname in ("CA", "C", "O"):
                    xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    atoms.setdefault(resi, {})[aname] = xyz
    resis = sorted(atoms.keys())
    ca = np.array([atoms[r]["CA"] for r in resis])
    c_at = np.array([atoms[r]["C"] for r in resis])
    o_at = np.array([atoms[r]["O"] for r in resis])
    ctr = ca.mean(axis=0)
    ca -= ctr
    c_at -= ctr
    o_at -= ctr
    _, _, vt = np.linalg.svd(ca, full_matrices=False)
    r0 = vt.T
    if np.linalg.det(r0) < 0:
        r0[:, 2] *= -1
    _GB1_CACHE = (ca @ r0, c_at @ r0, o_at @ r0)
    return _GB1_CACHE


def _hex_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)], dtype=float)


def _rgb_hex(rgb: np.ndarray) -> str:
    c = np.clip(np.round(rgb), 0, 255).astype(int)
    return f"#{c[0]:02X}{c[1]:02X}{c[2]:02X}"


def render_gb1_pdb_ribbon(cx: float, cy: float, scale: float = 3.55, angles: Tuple[float, float, float] = (1.15, 0.35, -0.82)) -> str:
    """Project real GB1 domain atomic backbone into a silky-smooth 3D ChimeraX-style vector ribbon."""
    ca, c_at, o_at = _load_gb1_coords()
    ax, ay, az = angles
    rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    rot = rz @ ry @ rx

    ca_r = ca @ rot.T
    o_vec = (o_at - c_at) @ rot.T
    n_res = len(ca_r)

    strands = [(0, 8), (11, 19), (41, 46), (49, 54)]
    helices = [(22, 36)]

    ca_sm = ca_r.copy()
    for a, b in strands:
        for _ in range(3):
            tmp = ca_sm.copy()
            for i in range(a + 1, b):
                tmp[i] = 0.25 * ca_sm[i - 1] + 0.5 * ca_sm[i] + 0.25 * ca_sm[i + 1]
            ca_sm = tmp

    b_vec = np.zeros_like(ca_sm)
    for i in range(n_res):
        t_v = ca_sm[min(n_res - 1, i + 1)] - ca_sm[max(0, i - 1)]
        t_v /= np.linalg.norm(t_v) + 1e-8
        v = o_vec[i] - np.dot(o_vec[i], t_v) * t_v
        v /= np.linalg.norm(v) + 1e-8
        if i > 0 and np.dot(v, b_vec[i - 1]) < 0:
            v = -v
        b_vec[i] = v
    for _ in range(4):
        bn = b_vec.copy()
        for i in range(1, n_res - 1):
            bn[i] = 0.25 * b_vec[i - 1] + 0.5 * b_vec[i] + 0.25 * b_vec[i + 1]
            bn[i] /= np.linalg.norm(bn[i]) + 1e-8
        b_vec = bn

    sub = 5
    ts = np.linspace(0, n_res - 1, (n_res - 1) * sub + 1)
    p_sp = CubicSpline(np.arange(n_res), ca_sm)(ts)
    b_sp = CubicSpline(np.arange(n_res), b_vec)(ts)

    m_pts = len(ts)
    tang = np.gradient(p_sp, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-8
    u_dir = b_sp - np.sum(b_sp * tang, axis=1, keepdims=True) * tang
    u_dir /= np.linalg.norm(u_dir, axis=1, keepdims=True) + 1e-8
    v_dir = np.cross(tang, u_dir)

    w_prof = np.zeros(m_pts)
    th_prof = np.zeros(m_pts)
    kind: List[str] = []
    for k, t in enumerate(ts):
        in_str = next(((a, b) for a, b in strands if a - 0.25 <= t <= b + 0.35), None)
        in_hel = next(((a, b) for a, b in helices if a - 0.3 <= t <= b + 0.3), None)
        if in_str is not None:
            a, b = in_str
            rem = (b + 0.35) - t
            arrow_len = 1.5
            w_prof[k] = (2.15 * max(0.06, rem / arrow_len)) if rem < arrow_len else 1.24
            th_prof[k] = 0.23
            kind.append("E")
        elif in_hel is not None:
            a, b = in_hel
            edge = min(t - (a - 0.3), (b + 0.3) - t)
            tap = min(1.0, max(0.22, edge / 0.75))
            w_prof[k] = 0.32 + 1.08 * tap
            th_prof[k] = 0.24
            kind.append("H")
        else:
            w_prof[k] = 0.33
            th_prof[k] = 0.33
            kind.append("C")

    light = np.array([-0.40, -0.50, 0.77])
    light /= np.linalg.norm(light)
    stops = [_hex_rgb("#0F766E"), _hex_rgb("#15803D"), _hex_rgb("#D97706"), _hex_rgb("#9A3412")]

    def base_col(k: int) -> np.ndarray:
        uf = (k / max(1, m_pts - 1)) * (len(stops) - 1)
        idx = min(len(stops) - 2, int(uf))
        fr = uf - idx
        return (1 - fr) * stops[idx] + fr * stops[idx + 1]

    items: List[Tuple[float, str]] = []
    # Group contiguous coil segments into smooth polylines (eliminating segment-cap rings!)
    k = 0
    while k < m_pts - 1:
        if kind[k] == "C" and kind[k + 1] == "C":
            j = k
            while j < m_pts - 1 and kind[j] == "C" and kind[j + 1] == "C":
                j += 1
            seg_pts = p_sp[k : j + 1]
            mid_k = (k + j) // 2
            c0 = base_col(mid_k)
            r_px = w_prof[mid_k] * scale
            z_m = float(seg_pts[:, 2].mean())
            col_main = _rgb_hex(c0 * 0.86)
            col_hi = _rgb_hex(np.minimum(255, c0 * 1.08 + 32))
            col_bdr = _rgb_hex(c0 * 0.28)
            d_main = "M " + " L ".join(f"{pt[0]*scale:.2f},{-pt[1]*scale:.2f}" for pt in seg_pts)
            d_hi = "M " + " L ".join(f"{pt[0]*scale - 0.18*r_px:.2f},{-pt[1]*scale - 0.18*r_px:.2f}" for pt in seg_pts)
            el = (
                f'<path d="{d_main}" fill="none" stroke="{col_bdr}" stroke-width="{2*r_px+1.5:.2f}" stroke-linecap="round" stroke-linejoin="round"/>'
                f'<path d="{d_main}" fill="none" stroke="{col_main}" stroke-width="{2*r_px:.2f}" stroke-linecap="round" stroke-linejoin="round"/>'
                f'<path d="{d_hi}" fill="none" stroke="{col_hi}" stroke-width="{0.62*r_px:.2f}" stroke-linecap="round" stroke-linejoin="round"/>'
            )
            items.append((z_m, el))
            k = j
            continue

        c0 = base_col(k)
        w0, h0 = w_prof[k], th_prof[k]
        w1, h1 = w_prof[k + 1], th_prof[k + 1]
        c00 = p_sp[k] - w0 * u_dir[k] - h0 * v_dir[k]
        c01 = p_sp[k] + w0 * u_dir[k] - h0 * v_dir[k]
        c02 = p_sp[k] + w0 * u_dir[k] + h0 * v_dir[k]
        c03 = p_sp[k] - w0 * u_dir[k] + h0 * v_dir[k]

        c10 = p_sp[k + 1] - w1 * u_dir[k + 1] - h1 * v_dir[k + 1]
        c11 = p_sp[k + 1] + w1 * u_dir[k + 1] - h1 * v_dir[k + 1]
        c12 = p_sp[k + 1] + w1 * u_dir[k + 1] + h1 * v_dir[k + 1]
        c13 = p_sp[k + 1] - w1 * u_dir[k + 1] + h1 * v_dir[k + 1]

        faces = [
            (np.array([c03, c02, c12, c13]), 0.5 * (v_dir[k] + v_dir[k + 1]), True),
            (np.array([c00, c01, c11, c10]), -0.5 * (v_dir[k] + v_dir[k + 1]), True),
            (np.array([c01, c02, c12, c11]), 0.5 * (u_dir[k] + u_dir[k + 1]), False),
            (np.array([c00, c03, c13, c10]), -0.5 * (u_dir[k] + u_dir[k + 1]), False),
        ]
        bdr_col = _rgb_hex(c0 * 0.26)
        for q, fn, is_broad in faces:
            fn_u = fn / (np.linalg.norm(fn) + 1e-8)
            z_face = fn_u[2]
            eff_n = fn_u if z_face >= 0 else -fn_u
            diff = max(0.0, float(np.dot(eff_n, light)))
            spec = (diff ** 10) * (0.55 if kind[k] == "H" else 0.30)
            base = (0.50 * c0 + 0.50 * np.array([238.0, 244.0, 242.0])) if (kind[k] == "H" and z_face < 0 and is_broad) else c0
            shaded = base * (0.48 + 0.56 * diff) + 38.0 * spec
            fc = _rgb_hex(shaded)
            pts = " ".join(f"{pt[0]*scale:.2f},{-pt[1]*scale:.2f}" for pt in q)
            el = f'<polygon points="{pts}" fill="{fc}" stroke="{fc}" stroke-width="0.55" stroke-linejoin="round"/>'
            if is_broad and z_face >= -0.15:
                el += (
                    f'<line x1="{q[0,0]*scale:.2f}" y1="{-q[0,1]*scale:.2f}" x2="{q[3,0]*scale:.2f}" y2="{-q[3,1]*scale:.2f}" stroke="{bdr_col}" stroke-width="0.85" stroke-linecap="round"/>'
                    f'<line x1="{q[1,0]*scale:.2f}" y1="{-q[1,1]*scale:.2f}" x2="{q[2,0]*scale:.2f}" y2="{-q[2,1]*scale:.2f}" stroke="{bdr_col}" stroke-width="0.85" stroke-linecap="round"/>'
                )
            items.append((float(q[:, 2].mean()), el))
        k += 1

    items.sort(key=lambda it: it[0])
    return f'<g transform="translate({cx:.1f},{cy:.1f})">' + "".join(el for _, el in items) + "</g>"


# ---------------------------------------------------------------------------
# 4-Layer Publication SVG Builder (Strict Orthogonality + Header Layering)
# ---------------------------------------------------------------------------
class PublicationSVG:
    def __init__(self, width: int, height: int, title: str):
        self.width = width
        self.height = height
        self.title = title
        self.defs: List[str] = []
        # 4 z-ordered layers: bg_panels -> under_wires -> headers -> body
        self.bg_panels: List[str] = []
        self.under_wires: List[str] = []
        self.headers: List[str] = []
        self.body: List[str] = []
        self.assertions: List[str] = []
        self.panel_boxes: List[Tuple[float, float, float, float]] = []
        self.legend_ys: List[float] = []
        self._init_defs()

    def _init_defs(self) -> None:
        for mid, col in [
            ("arr", PAL["ink_mid"]),
            ("arr-dark", PAL["ink"]),
            ("arr-wt", PAL["wt_mid"]),
            ("arr-mt", PAL["mt_dark"]),
            ("arr-epi", PAL["epi_bdr"]),
            ("arr-loss", PAL["loss_bdr"]),
        ]:
            self.defs.append(
                f'<marker id="{mid}" markerWidth="9.5" markerHeight="8.5" refX="8.0" refY="4.25" orient="auto" markerUnits="userSpaceOnUse">'
                f'<path d="M0,0.6 L8.6,4.25 L0,7.9 L1.7,4.25 Z" fill="{col}"/>'
                f"</marker>"
            )
        self.defs.append(
            f'<marker id="arr-grad-lg" markerWidth="15" markerHeight="14" refX="12.0" refY="7" orient="auto" markerUnits="userSpaceOnUse">'
            f'<path d="M1.0,1.2 L13.6,7.0 L1.0,12.8 Z" fill="#FFFFFF" stroke="{PAL["ink"]}" stroke-width="1.9" stroke-linejoin="round"/>'
            f"</marker>"
        )
        self.defs.append(
            f'<marker id="arr-grad-sm" markerWidth="11.5" markerHeight="10.5" refX="9.2" refY="5.25" orient="auto" markerUnits="userSpaceOnUse">'
            f'<path d="M0.8,1.0 L10.2,5.25 L0.8,9.5 Z" fill="#FFFFFF" stroke="{PAL["ink"]}" stroke-width="1.5" stroke-linejoin="round"/>'
            f"</marker>"
        )
        self.defs.append(
            '<linearGradient id="pred-tri-grad" x1="0%" y1="0%" x2="100%" y2="0%">'
            '<stop offset="0%" stop-color="#86EFAC"/>'
            '<stop offset="33%" stop-color="#BBF7D0"/>'
            '<stop offset="50%" stop-color="#FDE047"/>'
            '<stop offset="67%" stop-color="#FDE68A"/>'
            '<stop offset="100%" stop-color="#C4B5FD"/>'
            "</linearGradient>"
        )
        self.defs.append(
            '<linearGradient id="pred-tri-stroke" x1="0%" y1="0%" x2="100%" y2="0%">'
            f'<stop offset="0%" stop-color="{PAL["wt_mid"]}"/>'
            f'<stop offset="50%" stop-color="{PAL["mt_dark"]}"/>'
            f'<stop offset="100%" stop-color="{PAL["epi_bdr"]}"/>'
            "</linearGradient>"
        )

    def add(self, s: str, *, layer: str = "body") -> None:
        if layer == "bg":
            self.bg_panels.append(s)
        elif layer == "under":
            self.under_wires.append(s)
        elif layer == "hdr":
            self.headers.append(s)
        else:
            self.body.append(s)

    def nature_subplot_label(self, x: float, y: float, letter: str = "a") -> None:
        """Nature house style: bold lowercase sans-serif letter, no punctuation or box."""
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT_SANS}" font-size="24.0" '
            f'font-weight="700" fill="#000000" text-anchor="middle">{html.escape(letter)}</text>'
        )

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        rx: float = 8,
        fill: str = "#FFFFFF",
        stroke: Optional[str] = None,
        sw: float = 1.4,
        dash: Optional[str] = None,
        opacity: float = 1.0,
        layer: str = "body",
    ) -> None:
        attrs = [
            f'x="{x:.1f}"', f'y="{y:.1f}"', f'width="{w:.1f}"', f'height="{h:.1f}"',
            f'rx="{rx:.1f}"', f'fill="{fill}"',
        ]
        attrs.append(f'stroke="{stroke}" stroke-width="{sw:.2f}"' if stroke else 'stroke="none"')
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        if opacity < 1.0:
            attrs.append(f'opacity="{opacity:.2f}"')
        self.add(f'<rect {" ".join(attrs)}/>', layer=layer)

    def panel(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        *,
        fill: str,
        stroke: str,
        hdr_bg: Optional[str] = None,
    ) -> None:
        """Emit background on 'bg' layer and opaque header bar + title on 'hdr' layer,
        allowing 'under' wires to pass cleanly behind the whole header panel!
        """
        self.panel_boxes.append((x, y, x + w, y + h))
        r = 10.0
        hb = hdr_bg or PAL["hdr_bg"]
        self.rect(x, y, w, h, rx=r, fill=fill, stroke=stroke, sw=1.4, layer="bg")
        self.add(
            f'<path d="M {x:.1f},{y+34:.1f} L {x:.1f},{y+r:.1f} Q {x:.1f},{y:.1f} {x+r:.1f},{y:.1f} '
            f'L {x+w-r:.1f},{y:.1f} Q {x+w:.1f},{y:.1f} {x+w:.1f},{y+r:.1f} L {x+w:.1f},{y+34:.1f} Z" '
            f'fill="{hb}"/>',
            layer="hdr",
        )
        self.text(x + w / 2, y + 23.0, title, size=15.0, fill=PAL["hdr_fg"], bold=True, max_w=w - 28, layer="hdr")

    def draw_h_columns(
        self,
        x0: float,
        y0: float,
        h: float,
        cols: Sequence[Tuple[float, str, str, str]],
    ) -> None:
        hdr_h = 34.0
        r = 10.0
        xc = x0
        for idx, (cw, _, cfill, cbdr) in enumerate(cols):
            self.panel_boxes.append((xc, y0, xc + cw, y0 + h))
            yb0 = y0 + hdr_h
            yb1 = y0 + h
            if idx == 0:
                d = (
                    f"M {xc:.1f},{yb0:.1f} L {xc+cw:.1f},{yb0:.1f} L {xc+cw:.1f},{yb1:.1f} "
                    f"L {xc+r:.1f},{yb1:.1f} Q {xc:.1f},{yb1:.1f} {xc:.1f},{yb1-r:.1f} Z"
                )
            elif idx == len(cols) - 1:
                d = (
                    f"M {xc:.1f},{yb0:.1f} L {xc+cw:.1f},{yb0:.1f} L {xc+cw:.1f},{yb1-r:.1f} "
                    f"Q {xc+cw:.1f},{yb1:.1f} {xc+cw-r:.1f},{yb1:.1f} L {xc:.1f},{yb1:.1f} Z"
                )
            else:
                d = f"M {xc:.1f},{yb0:.1f} L {xc+cw:.1f},{yb0:.1f} L {xc+cw:.1f},{yb1:.1f} L {xc:.1f},{yb1:.1f} Z"
            self.add(f'<path d="{d}" fill="{cfill}" stroke="{cbdr}" stroke-width="1.3"/>', layer="bg")
            xc += cw

        tot_w = sum(c[0] for c in cols)
        self.add(
            f'<path d="M {x0:.1f},{y0+hdr_h:.1f} L {x0:.1f},{y0+r:.1f} Q {x0:.1f},{y0:.1f} {x0+r:.1f},{y0:.1f} '
            f'L {x0+tot_w-r:.1f},{y0:.1f} Q {x0+tot_w:.1f},{y0:.1f} {x0+tot_w:.1f},{y0+r:.1f} L {x0+tot_w:.1f},{y0+hdr_h:.1f} Z" '
            f'fill="{PAL["hdr_bg"]}"/>',
            layer="hdr",
        )
        xc = x0
        for idx, (cw, ctitle, _, _) in enumerate(cols):
            if idx > 0:
                self.add(f'<line x1="{xc:.1f}" y1="{y0:.1f}" x2="{xc:.1f}" y2="{y0+hdr_h:.1f}" stroke="#0F172A" stroke-width="1.8"/>', layer="hdr")
            self.text(xc + cw / 2, y0 + 23.0, ctitle, size=14.5, fill=PAL["hdr_fg"], bold=True, max_w=cw - 18, layer="hdr")
            xc += cw

    def text(
        self,
        x: float,
        y: float,
        markup: str,
        *,
        size: float = 13.5,
        fill: Optional[str] = None,
        anchor: str = "middle",
        bold: bool = False,
        family: str = FONT_SANS,
        max_w: Optional[float] = None,
        pil_family: str = "sans",
        layer: str = "body",
    ) -> None:
        plain = strip_markup(markup)
        for bad in FORBIDDEN_SUBSTRINGS:
            if bad.lower() in plain.lower():
                self.assertions.append(f"Forbidden speed claim {bad!r} found in text: {plain!r}")
        if size < MIN_FONT_PX:
            self.assertions.append(f"Font too small ({size:.1f}px < {MIN_FONT_PX}px): {plain!r}")
        if max_w is not None and plain:
            est = measure_text_px(plain, size, family=pil_family, bold=(bold or "<b>" in markup))
            if est > max_w + 6:
                self.assertions.append(
                    f"Text overflow ({est:.1f}px > {max_w:.1f}px) at ({x:.0f},{y:.0f}): {plain!r}"
                )
        runs = render_rich_runs(markup, size)
        fw = ' font-weight="700"' if bold else ""
        col = fill or PAL["ink"]
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size:.1f}" '
            f'fill="{col}" text-anchor="{anchor}"{fw}>{runs}</text>',
            layer=layer,
        )

    def _assert_orthogonal(self, pts: Sequence[Tuple[float, float]]) -> None:
        for i in range(len(pts) - 1):
            dx = abs(pts[i + 1][0] - pts[i][0])
            dy = abs(pts[i + 1][1] - pts[i][1])
            if dx > 1e-3 and dy > 1e-3:
                self.assertions.append(
                    f"Non-orthogonal diagonal segment {pts[i]} -> {pts[i+1]} (dx={dx:.1f}, dy={dy:.1f})"
                )

    @staticmethod
    def _rounded_poly(pts: Sequence[Tuple[float, float]], radius: float) -> str:
        if len(pts) == 2 or radius <= 0:
            return f"M {pts[0][0]:.1f},{pts[0][1]:.1f} " + " ".join(f"L {px:.1f},{py:.1f}" for px, py in pts[1:])
        cmds = [f"M {pts[0][0]:.1f},{pts[0][1]:.1f}"]
        for i in range(1, len(pts) - 1):
            p0, p1, p2 = pts[i - 1], pts[i], pts[i + 1]
            v1 = (p1[0] - p0[0], p1[1] - p0[1])
            v2 = (p2[0] - p1[0], p2[1] - p1[1])
            l1, l2 = math.hypot(*v1), math.hypot(*v2)
            if l1 < 1e-3 or l2 < 1e-3:
                cmds.append(f"L {p1[0]:.1f},{p1[1]:.1f}")
                continue
            r = min(radius, l1 * 0.45, l2 * 0.45)
            a = (p1[0] - v1[0] / l1 * r, p1[1] - v1[1] / l1 * r)
            b = (p1[0] + v2[0] / l2 * r, p1[1] + v2[1] / l2 * r)
            cmds.append(f"L {a[0]:.1f},{a[1]:.1f} Q {p1[0]:.1f},{p1[1]:.1f} {b[0]:.1f},{b[1]:.1f}")
        cmds.append(f"L {pts[-1][0]:.1f},{pts[-1][1]:.1f}")
        return " ".join(cmds)

    def arrow(
        self,
        pts: Sequence[Tuple[float, float]],
        *,
        color: Optional[str] = None,
        marker: str = "arr",
        sw: float = 2.2,
        dash: Optional[str] = None,
        radius: float = 7.5,
        arrow_end: bool = True,
        under_header: bool = False,
    ) -> None:
        self._assert_orthogonal(pts)
        d = self._rounded_poly(pts, radius)
        col = color or PAL["ink_mid"]
        attrs = [
            f'd="{d}"', 'fill="none"', f'stroke="{col}"', f'stroke-width="{sw:.2f}"',
            'stroke-linecap="round"', 'stroke-linejoin="round"',
        ]
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        if arrow_end:
            attrs.append(f'marker-end="url(#{marker})"')
        self.add(f'<path {" ".join(attrs)}/>', layer="under" if under_header else "body")

    def gradient_arrow(
        self,
        pts: Sequence[Tuple[float, float]],
        *,
        radius: float = 10.0,
        arrow_end: bool = True,
        t_junction: bool = False,
        large: bool = True,
        under_header: bool = False,
    ) -> None:
        self._assert_orthogonal(pts)
        d = self._rounded_poly(pts, radius)
        outer_w = 7.8 if large else 5.6
        inner_w = 4.0 if large else 2.8
        mk = "arr-grad-lg" if large else "arr-grad-sm"
        lyr = "under" if under_header else "body"
        self.add(f'<path d="{d}" fill="none" stroke="{PAL["ink"]}" stroke-width="{outer_w:.1f}" stroke-linecap="round" stroke-linejoin="round"/>', layer=lyr)
        end_attr = f' marker-end="url(#{mk})"' if arrow_end else ""
        self.add(f'<path d="{d}" fill="none" stroke="#FFFFFF" stroke-width="{inner_w:.1f}" stroke-linecap="round" stroke-linejoin="round"{end_attr}/>', layer=lyr)
        if t_junction:
            jx, jy = pts[0]
            self.add(f'<circle cx="{jx:.1f}" cy="{jy:.1f}" r="{inner_w * 0.55:.2f}" fill="#FFFFFF"/>', layer=lyr)

    def dot(self, cx: float, cy: float, r: float = 3.1, fill: Optional[str] = None) -> None:
        self.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill or PAL["ink_mid"]}"/>')

    def hexagon(
        self,
        cx: float,
        cy: float,
        w: float,
        h: float,
        label: str,
        *,
        sublabel: Optional[str] = None,
        fill: str = "#FFFFFF",
        stroke: str = "#334155",
        ink: Optional[str] = None,
        sw: float = 1.8,
        size: float = 14.0,
    ) -> None:
        col = ink or PAL["ink"]
        tip = min(14.0, h * 0.36)
        hw, hh = w / 2, h / 2
        pts = [
            (cx - hw, cy), (cx - hw + tip, cy - hh), (cx + hw - tip, cy - hh),
            (cx + hw, cy), (cx + hw - tip, cy + hh), (cx - hw + tip, cy + hh),
        ]
        pstr = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        self.add(f'<polygon points="{pstr}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.2f}" stroke-linejoin="round"/>')
        avail = w - 2 * tip + 10
        if sublabel:
            self.text(cx, cy - 2.0, label, size=size, fill=col, bold=True, max_w=avail)
            self.text(cx, cy + 14.5, sublabel, size=11.5, fill=PAL["ink_mid"], max_w=avail + 6)
        else:
            self.text(cx, cy + size * 0.35, label, size=size, fill=col, bold=True, max_w=avail)

    @staticmethod
    def hex_edge_offset(h: float, dy: float) -> float:
        """Horizontal x-inset from left/right tip of a hexagon at vertical offset dy from center."""
        tip = min(14.0, h * 0.36)
        hh = h / 2.0
        return tip * min(1.0, abs(dy) / max(1e-6, hh))

    def circle_op(self, cx: float, cy: float, r: float, label: str, *, fill: str = "#FFFFFF", stroke: Optional[str] = None, size: float = 13.5) -> None:
        self.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke or PAL["ink"]}" stroke-width="1.6"/>')
        self.text(cx, cy + size * 0.35, label, size=size, bold=True)

    def stop_grad(self, cx: float, cy: float, *, stroke: Optional[str] = None) -> None:
        self.rect(cx - 17, cy - 11, 34, 22, rx=5, fill="#FFFFFF", stroke=stroke or PAL["wt_mid"], sw=1.5)
        self.text(cx, cy + 4.2, "sg", size=12.0, fill=PAL["wt_ink"] if stroke else PAL["ink"], bold=True)

    def floating_portal_box(self, cx: float, cy: float, w: float = 34.0, h: float = 46.0, label: str = "TF") -> None:
        x, y = cx - w / 2, cy - h / 2
        self.rect(x + 2.8, y + 2.8, w, h, rx=7, fill="#D6D3D1", stroke=None)
        self.rect(x, y, w, h, rx=7, fill="#FFFFFF", stroke=PAL["ink"], sw=1.6)
        self.rect(x + 3.0, y + 3.0, w - 6.0, h - 6.0, rx=4.5, fill="#FAF9F6", stroke=PAL["p1_bdr"], sw=1.0, dash="2.5,2")
        self.text(cx, cy + 4.5, f"<b>{label}</b>", size=12.0, fill=PAL["ink"], max_w=w - 6)

    def icon_snowflake(self, cx: float, cy: float, r: float = 7.2) -> None:
        cmds = []
        for k in range(3):
            a = k * math.pi / 3
            dx, dy = r * math.cos(a), r * math.sin(a)
            cmds.append(f"M {cx-dx:.1f},{cy-dy:.1f} L {cx+dx:.1f},{cy+dy:.1f}")
        for k in range(6):
            a = k * math.pi / 3
            px, py = cx + r * 0.58 * math.cos(a), cy + r * 0.58 * math.sin(a)
            for s_dir in (-1, 1):
                a2 = a + s_dir * 0.65
                cmds.append(f"M {px:.1f},{py:.1f} L {px+2.8*math.cos(a2):.1f},{py+2.8*math.sin(a2):.1f}")
        self.add(f'<path d="{" ".join(cmds)}" stroke="{PAL["ice"]}" stroke-width="1.5" stroke-linecap="round" fill="none"/>')

    def icon_trainable(self, cx: float, cy: float, scale: float = 0.80, *, on_dark: bool = False) -> None:
        """Canonical two-tongue ML trainable fire icon with bright golden inner core."""
        ring = ' stroke="#FFFFFF" stroke-width="0.95"' if on_dark else ' stroke="#9A3412" stroke-width="0.65"'
        outer = (
            "M 0.8,-11.5 "
            "C 1.4,-7.4 5.8,-5.2 6.8,-0.4 "
            "C 7.5,3.2 6.6,7.0 4.0,9.2 "
            "C 2.2,10.7 -0.2,11.2 -2.4,10.7 "
            "C -6.0,9.8 -8.2,6.4 -7.8,2.4 "
            "C -7.5,-0.5 -5.8,-3.0 -4.2,-4.6 "
            "C -4.4,-2.4 -3.5,-0.9 -2.3,-0.6 "
            "C -3.2,-4.2 -1.8,-8.4 0.8,-11.5 Z"
        )
        inner = (
            "M 0.5,-3.6 "
            "C 1.6,-1.2 3.6,0.4 3.6,3.4 "
            "C 3.6,6.0 1.8,8.2 -0.6,8.4 "
            "C -3.0,8.5 -4.6,6.6 -4.4,4.2 "
            "C -4.2,2.2 -2.8,0.8 -1.8,-0.2 "
            "C -1.7,1.2 -0.9,2.0 -0.2,2.0 "
            "C -0.5,0.0 -0.2,-1.9 0.5,-3.6 Z"
        )
        self.add(
            f'<g transform="translate({cx:.1f},{cy:.1f}) scale({scale:.3f})">'
            f'<path d="{outer}" fill="#F97316"{ring} stroke-linejoin="round"/>'
            f'<path d="{inner}" fill="#FDE047"/>'
            f"</g>"
        )

    def draw_gb1_domain(self, cx: float, cy: float, scale: float = 3.55) -> None:
        self.add(render_gb1_pdb_ribbon(cx, cy, scale=scale))

    def draw_seq_strip(
        self,
        cx: float,
        y: float,
        tag: str,
        residues: Sequence[str],
        *,
        is_mt: bool,
        stacked: int = 0,
    ) -> Tuple[float, float, float]:
        cw, ch = 26.0, 28.0
        tag_w, gap = 34.0, 6.0
        total_w = tag_w + gap + len(residues) * cw
        x0 = cx - total_w / 2

        for k in range(stacked, 0, -1):
            ox, oy = k * 4.0, -k * 4.0
            self.rect(
                x0 + tag_w + gap + ox, y + oy, len(residues) * cw, ch, rx=3,
                fill=PAL["mt_soft"] if is_mt else PAL["wt_soft"],
                stroke=PAL["mt_mid"] if is_mt else PAL["wt_mid"], sw=1.0, opacity=0.78,
            )

        self.rect(x0, y, tag_w, ch, rx=5, fill="#FFFFFF", stroke=PAL["mt_dark"] if is_mt else PAL["wt_mid"], sw=1.5)
        self.text(x0 + tag_w / 2, y + 19.0, tag, size=13.5, fill=PAL["mt_ink"] if is_mt else PAL["wt_ink"], bold=True)

        sx = x0 + tag_w + gap
        for idx, aa in enumerate(residues):
            rx_i = sx + idx * cw
            mut_site = idx in (2, 4)
            if mut_site and is_mt:
                fc, sc, tc = PAL["mt_pred"], PAL["mt_dark"], PAL["mt_ink"]
            elif mut_site and not is_mt:
                fc, sc, tc = PAL["wt_soft"], PAL["wt_mid"], PAL["wt_ink"]
            else:
                fc, sc, tc = PAL["seq_bg"], PAL["seq_bdr"], PAL["ink"]
            self.rect(rx_i, y, cw, ch, rx=2.5, fill=fc, stroke=sc, sw=1.6 if mut_site else 1.0)
            self.text(rx_i + cw / 2, y + 19.0, aa, size=13.0, fill=tc, bold=mut_site, family=FONT_MONO, pil_family="mono")
            if mut_site:
                pos_lbl = "3" if idx == 2 else "5"
                self.text(rx_i + cw / 2, y + ch + 13.5, pos_lbl, size=11.5, fill=PAL["ink_muted"], bold=True)

        return x0, x0 + total_w, y + ch + 17.0

    def draw_logit_matrix_and_diffs(
        self,
        mx: float,
        my: float,
        *,
        stacked: int = 0,
        cached_side: bool = False,
    ) -> Tuple[float, float, float]:
        """Draw [Logits (Lx20)] -> [D-E / G-S] -> orthogonal collector -> Sigma starting at mx, my.
        Returns (matrix_cx, sigma_cx, cy_mid).
        """
        mw, mh = 104.0, 74.0
        cy_mid = my + mh / 2

        if cached_side:
            # Side badge on the left of the WT Logits (Lx20) matrix!
            bw, bh = 68.0, 26.0
            bx, by = mx - bw - 8.0, cy_mid - bh / 2
            self.add(f'<line x1="{bx+bw:.1f}" y1="{cy_mid:.1f}" x2="{mx:.1f}" y2="{cy_mid:.1f}" stroke="{PAL["wt_mid"]}" stroke-width="1.6"/>')
            self.rect(bx, by, bw, bh, rx=13, fill=PAL["wt_soft"], stroke=PAL["wt_mid"], sw=1.3)
            self.text(bx + bw / 2, by + 17.5, "Cached", size=12.0, fill=PAL["wt_ink"], bold=True, max_w=bw - 8)

        for k in range(stacked, 0, -1):
            self.rect(mx + k * 4.0, my - k * 4.0, mw, mh, rx=5, fill=PAL["mt_soft"], stroke=PAL["mt_mid"], sw=1.0, opacity=0.78)

        self.rect(mx, my, mw, mh, rx=5, fill="#FFFFFF", stroke=PAL["ink_mid"], sw=1.4)
        self.text(mx + mw / 2, my + mh + 16.5, "Logits (<i>L</i>×20)", size=12.2, fill=PAL["ink"], bold=True)

        cols, rows = 5, 4
        gx, gy = mx + 23, my + 18
        cw, ch = (mw - 27) / cols, (mh - 22) / rows
        self.text(mx + 11.5, my + mh / 2 + 6, "20", size=11.5, fill=PAL["ink_muted"], bold=True)
        for c_i, clbl in enumerate(["1", "3", "4", "5", "…"]):
            is_m = c_i in (1, 3)
            self.text(gx + (c_i + 0.5) * cw, my + 13.5, clbl, size=11.5, fill=PAL["mt_dark"] if is_m else PAL["ink_muted"], bold=is_m)

        for r_i in range(rows):
            for c_i in range(cols):
                rx_c, ry_c = gx + c_i * cw, gy + r_i * ch
                cell_lbl = None
                if c_i == 1 and r_i == 0:
                    fc, sc, tc, cell_lbl = PAL["mt_pred"], PAL["mt_dark"], PAL["mt_ink"], "D"
                elif c_i == 3 and r_i == 1:
                    fc, sc, tc, cell_lbl = PAL["mt_pred"], PAL["mt_dark"], PAL["mt_ink"], "G"
                elif c_i == 1 and r_i == 2:
                    fc, sc, tc, cell_lbl = PAL["wt_soft"], PAL["wt_mid"], PAL["wt_ink"], "E"
                elif c_i == 3 and r_i == 3:
                    fc, sc, tc, cell_lbl = PAL["wt_soft"], PAL["wt_mid"], PAL["wt_ink"], "S"
                else:
                    fc, sc, tc = "#FAF9F6", "#CBD5E1", PAL["ink"]
                self.add(f'<rect x="{rx_c:.1f}" y="{ry_c:.1f}" width="{cw-1.2:.1f}" height="{ch-1.2:.1f}" fill="{fc}" stroke="{sc}" stroke-width="0.8"/>')
                if cell_lbl:
                    self.text(rx_c + (cw - 1.2) / 2, ry_c + ch - 1.8, cell_lbl, size=11.5, fill=tc, bold=True, family=FONT_MONO, pil_family="mono")

        # Site log-odds [D - E] and [G - S] with 100% orthogonal arrows!
        dx = mx + mw + 16
        r1_y, r2_y = cy_mid - 28, cy_mid + 4
        r1_cy, r2_cy = r1_y + 12, r2_y + 12
        self.arrow([(mx + mw, r1_cy), (dx - 2, r1_cy)], sw=1.8)
        self.arrow([(mx + mw, r2_cy), (dx - 2, r2_cy)], sw=1.8)

        for ry, m_aa, w_aa in [(r1_y, "D", "E"), (r2_y, "G", "S")]:
            self.rect(dx, ry, 24, 24, rx=4, fill=PAL["mt_pred"], stroke=PAL["mt_dark"], sw=1.4)
            self.text(dx + 12, ry + 17.0, m_aa, size=12.5, fill=PAL["mt_ink"], bold=True, family=FONT_MONO, pil_family="mono")
            self.text(dx + 33, ry + 17.2, "−", size=14.0, bold=True)
            self.rect(dx + 42, ry, 24, 24, rx=4, fill=PAL["wt_soft"], stroke=PAL["wt_mid"], sw=1.4)
            self.text(dx + 54, ry + 17.0, w_aa, size=12.5, fill=PAL["wt_ink"], bold=True, family=FONT_MONO, pil_family="mono")

        # 90-degree rounded collector into Sigma (zero diagonal segments!)
        col_x = dx + 66 + 10
        sig_cx = col_x + 28
        self.arrow([(dx + 66, r1_cy), (col_x, r1_cy), (col_x, r2_cy), (dx + 66, r2_cy)], sw=1.8, radius=5.0, arrow_end=False)
        self.dot(col_x, cy_mid)
        self.arrow([(col_x, cy_mid), (sig_cx - 15.5, cy_mid)], sw=1.9)
        self.circle_op(sig_cx, cy_mid, 14.5, "∑", fill="#FFFFFF", size=14.0)
        return mx + mw / 2, sig_cx, cy_mid

    def draw_outside_legend(self, y: float) -> None:
        self.legend_ys.append(y)
        cx = self.width / 2
        xs = [cx - 438, cx - 308, cx - 140, cx + 14, cx + 166, cx + 312]

        self.icon_snowflake(xs[0], y - 4, r=6.8)
        self.text(xs[0] + 14, y, "Frozen ESM3", size=12.5, anchor="start", bold=True)

        self.icon_trainable(xs[1], y - 4, scale=0.78, on_dark=False)
        self.text(xs[1] + 14, y, "Trainable LoRA / Calib", size=12.5, anchor="start", bold=True)

        self.hexagon(xs[2] + 26, y - 4, 54, 24, "ΔΔ<i>Ĝ</i>", fill="url(#pred-tri-grad)", stroke="url(#pred-tri-stroke)", sw=1.8, size=11.5)
        self.text(xs[2] + 60, y, "Prediction", size=12.5, anchor="start", bold=True)

        self.hexagon(xs[3] + 26, y - 4, 54, 24, "ΔΔ<i>G</i>", fill=PAL["gt_bg"], stroke=PAL["gt_bdr"], sw=1.7, size=11.5)
        self.text(xs[3] + 60, y, "Label / TF", size=12.5, anchor="start")

        self.stop_grad(xs[4] + 18, y - 4)
        self.text(xs[4] + 42, y, "Stop-grad", size=12.5, anchor="start")

        self.gradient_arrow([(xs[5], y - 4), (xs[5] + 42, y - 4)], large=False)
        self.text(xs[5] + 52, y, "Gradient update", size=12.5, anchor="start", bold=True)

    def to_svg(self) -> str:
        for ly in self.legend_ys:
            for (x0, y0, x1, y1) in self.panel_boxes:
                if (y0 - 12) <= ly <= (y1 + 12):
                    self.assertions.append(f"Legend at y={ly:.1f} overlaps colored panel [{y0:.1f}, {y1:.1f}]")
        if self.assertions:
            raise AssertionError(f"SVG validation failed in {self.title!r}:\n  " + "\n  ".join(self.assertions))
        defs_s = "\n  ".join(self.defs)
        all_layers = self.bg_panels + self.under_wires + self.headers + self.body
        body_s = "\n  ".join(all_layers)
        return (
            f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" '
            f'width="{self.width}" height="{self.height}">\n'
            f'  <title>{html.escape(self.title)}</title>\n'
            f'  <defs>\n  {defs_s}\n  </defs>\n'
            f'  <rect width="{self.width}" height="{self.height}" fill="#FFFFFF"/>\n'
            f'  {body_s}\n'
            f'</svg>\n'
        )


# ===========================================================================
# 1. VERTICAL UNIFIED SCHEMATIC (1020 x 1246)
# ===========================================================================
def build_vertical_svg(include_label: bool = False) -> str:
    title = (
        "ESM-MSR Architecture (Vertical Unified Schematic, Subplot a)"
        if include_label
        else "ESM-MSR Architecture (Vertical Unified Schematic)"
    )
    s = PublicationSVG(1020, 1246, title)

    if include_label:
        # Nature-style bold lowercase subplot label 'a' in upper-left margin
        s.nature_subplot_label(28.0, 38.0, "a")

    px, pw = 60.0, 900.0
    wx, cx, mx = 224.0, 510.0, 796.0
    grad_l, grad_r = 28.0, 992.0

    # --- PANEL 1: SEQUENCE & WT STRUCTURE INPUTS (y = 16..208) ---
    p1_y, p1_h = 16.0, 192.0
    s.panel(px, p1_y, pw, p1_h, "1. SEQUENCE & WT STRUCTURE INPUTS", fill=PAL["p1_bg"], stroke=PAL["p1_bdr"])

    s.rect(wx - 86, p1_y + 44, 172, 24, rx=12, fill=PAL["wt_soft"], stroke=PAL["wt_mid"], sw=1.3)
    s.text(wx, p1_y + 60.5, "1 pass / protein", size=12.2, fill=PAL["wt_ink"], bold=True, max_w=156)

    s.rect(mx - 86, p1_y + 44, 172, 24, rx=12, fill=PAL["mt_soft"], stroke=PAL["mt_dark"], sw=1.3)
    s.text(mx, p1_y + 60.5, "1 pass / variant", size=12.2, fill=PAL["mt_ink"], bold=True, max_w=156)

    _, _, wt_seq_bot = s.draw_seq_strip(wx, p1_y + 78, "ab", ["M", "N", "E", "F", "S", "…"], is_mt=False, stacked=0)
    _, _, mt_seq_bot = s.draw_seq_strip(mx, p1_y + 78, "AB", ["M", "N", "D", "F", "G", "…"], is_mt=True, stacked=2)

    tok_y = p1_y + 142
    for col_x, seq_bot in [(wx, wt_seq_bot), (mx, mt_seq_bot)]:
        s.arrow([(col_x, seq_bot + 2), (col_x, tok_y - 2)], sw=2.1)
        s.rect(col_x - 80, tok_y, 160, 34, rx=6, fill=PAL["frozen_bg"], stroke=PAL["frozen_bdr"], sw=1.3)
        s.text(col_x, tok_y + 22.0, "Seq Tokenizer", size=13.0, bold=True, max_w=146)

    # Center: Real 3D GB1 PDB Domain Render + 'WT BB Coords' + 'ESM3 VQ-VAE'
    bb_cx = cx - 64.0
    s.draw_gb1_domain(bb_cx, p1_y + 92, scale=3.35)
    s.rect(bb_cx - 66, tok_y, 132, 34, rx=6, fill="#FFFFFF", stroke=PAL["frozen_bdr"], sw=1.3)
    s.text(bb_cx, tok_y + 22.0, "WT BB Coords", size=12.6, bold=True, max_w=122)

    vq_x, vq_y, vq_w, vq_h = cx + 18, p1_y + 66, 126, 110
    s.arrow([(bb_cx + 46, p1_y + 100), (vq_x - 2, p1_y + 100)], sw=2.1)
    s.rect(vq_x, vq_y, vq_w, vq_h, rx=7, fill=PAL["frozen_bg"], stroke=PAL["frozen_bdr"], sw=1.4)
    s.icon_snowflake(vq_x + vq_w - 15, vq_y + 15, r=6.8)
    s.text(vq_x + vq_w / 2, vq_y + 44, "<b>ESM3</b>", size=14.0)
    s.text(vq_x + vq_w / 2, vq_y + 64, "<b>VQ-VAE</b>", size=14.0)
    s.text(vq_x + vq_w / 2, vq_y + 89, "Struct Tokens", size=11.8, fill=PAL["ink_mid"], max_w=114)

    # --- PANEL 2: SHARED ESM3 BACKBONE & ASYMMETRIC LoRA ADAPTERS (y = 222..382) ---
    p2_y, p2_h = 222.0, 160.0
    s.panel(px, p2_y, pw, p2_h, "2. SHARED ESM3 BACKBONE & ASYMMETRIC LoRA ADAPTERS", fill=PAL["p2_bg"], stroke=PAL["p2_bdr"])

    lora_y, bw, bh = p2_y + 52, 224.0, 92.0
    ew = 232.0

    s.rect(wx + bw / 2 - 4, lora_y + 26, (cx - ew / 2) - (wx + bw / 2) + 8, 16, rx=0, fill=PAL["frozen_bg"], stroke=PAL["frozen_bdr"], sw=1.2)
    s.rect(wx + bw / 2 - 4, lora_y + 48, (cx - ew / 2) - (wx + bw / 2) + 8, 16, rx=0, fill=PAL["wt_dark"], stroke=PAL["wt_dark"], sw=1.2)

    s.rect(cx + ew / 2 - 4, lora_y + 26, (mx - bw / 2) - (cx + ew / 2) + 8, 16, rx=0, fill=PAL["frozen_bg"], stroke=PAL["frozen_bdr"], sw=1.2)
    s.rect(cx + ew / 2 - 4, lora_y + 48, (mx - bw / 2) - (cx + ew / 2) + 8, 16, rx=0, fill=PAL["mt_lora"], stroke=PAL["mt_lora"], sw=1.2)

    # Left: ESM-MSR WT LoRA
    s.rect(wx - bw / 2, lora_y, bw, bh, rx=12, fill=PAL["wt_dark"], stroke=PAL["ink"], sw=1.6)
    s.icon_trainable(wx + bw / 2 - 18, lora_y + 19, scale=0.84, on_dark=True)
    s.text(wx, lora_y + 32, "<b>ESM-MSR WT LoRA</b>", size=14.5, fill="#FFFFFF", max_w=bw - 34)
    s.rect(wx - 82, lora_y + 47, 164, 24, rx=12, fill=PAL["wt_soft"], stroke=None)
    s.text(wx, lora_y + 63.5, "rank <i>r</i> = 2 · QKV/FFN", size=12.0, fill=PAL["wt_ink"], bold=True, max_w=154)

    # Center: Frozen ESM3 Transformer
    s.rect(cx - ew / 2, lora_y, ew, bh, rx=9, fill=PAL["frozen_bg"], stroke=PAL["frozen_bdr"], sw=1.5)
    s.icon_snowflake(cx + ew / 2 - 16, lora_y + 16, r=7.0)
    s.text(cx, lora_y + 32, "<b>ESM3 Transformer</b>", size=14.5, max_w=ew - 28)
    s.text(cx, lora_y + 54, "1.4B shared backbone", size=12.5, fill=PAL["ink_mid"], bold=True, max_w=ew - 18)
    s.text(cx, lora_y + 75, "Geom. + Token Attention", size=11.8, fill=PAL["ink_muted"], max_w=ew - 16)

    # Right: ESM-MSR MT LoRA
    for k in (2, 1):
        s.rect(mx - bw / 2 + k * 4.5, lora_y - k * 4.5, bw, bh, rx=12, fill=PAL["mt_soft"], stroke=PAL["mt_mid"], sw=1.0)
    s.rect(mx - bw / 2, lora_y, bw, bh, rx=12, fill=PAL["mt_lora"], stroke=PAL["ink"], sw=1.6)
    s.icon_trainable(mx + bw / 2 - 18, lora_y + 19, scale=0.84, on_dark=True)
    s.text(mx, lora_y + 32, "<b>ESM-MSR MT LoRA</b>", size=14.5, fill="#FFFFFF", max_w=bw - 34)
    s.rect(mx - 82, lora_y + 47, 164, 24, rx=12, fill=PAL["mt_soft"], stroke=None)
    s.text(mx, lora_y + 63.5, "rank <i>r</i> = 16 · QKV/FFN", size=12.0, fill=PAL["mt_ink"], bold=True, max_w=154)

    # Panel 1 -> Panel 2 arrows pass BEHIND Panel 2's dark header bar (under_header=True)!
    s.arrow([(wx, tok_y + 34), (wx, lora_y - 2)], color=PAL["wt_mid"], marker="arr-wt", sw=2.3, under_header=True)
    s.arrow([(bb_cx, tok_y + 34), (bb_cx, lora_y - 2)], sw=2.1, under_header=True)
    s.arrow([(vq_x + vq_w / 2, vq_y + vq_h), (vq_x + vq_w / 2, lora_y - 2)], sw=2.1, under_header=True)
    s.arrow([(mx, tok_y + 34), (mx, lora_y - 10)], color=PAL["mt_dark"], marker="arr-mt", sw=2.3, under_header=True)

    # --- PANEL 3: SITE LOG-ODDS EXTRACTION & PER-MUTATION CALIBRATION (y = 396..706) ---
    p3_y, p3_h = 396.0, 310.0
    s.panel(px, p3_y, pw, p3_h, "3. SITE LOG-ODDS EXTRACTION & PER-MUTATION CALIBRATION", fill=PAL["p3_bg"], stroke=PAL["p3_bdr"])

    mat_y = p3_y + 66
    # Position WT matrix at mx=150 so 'Cached' badge fits cleanly on its LEFT side (x=74..142)!
    wt_mcx, wt_sig_x, wt_mcy = s.draw_logit_matrix_and_diffs(150.0, mat_y, stacked=0, cached_side=True)
    mt_mcx, mt_sig_x, mt_mcy = s.draw_logit_matrix_and_diffs(686.0, mat_y, stacked=2, cached_side=False)

    # Redrawn 90-degree orthogonal LoRA emission lines passing behind Panel 3 header bar!
    bend_y3 = p3_y + 47
    s.arrow([(wx, lora_y + bh), (wx, bend_y3), (wt_mcx, bend_y3), (wt_mcx, mat_y - 2)], color=PAL["wt_mid"], marker="arr-wt", sw=2.2, radius=6.5, under_header=True)
    s.arrow([(mx, lora_y + bh), (mx, bend_y3), (mt_mcx, bend_y3), (mt_mcx, mat_y - 10)], color=PAL["mt_dark"], marker="arr-mt", sw=2.2, radius=6.5, under_header=True)

    # Center Site Log-Odds Sum annotation card (x = 406..644, cleanly centered in gap [375..686]!)
    card_cx = 528.0
    s.rect(card_cx - 120, mat_y - 2, 240, 92, rx=8, fill="#FFFFFF", stroke=PAL["p3_bdr"], sw=1.4)
    s.text(card_cx, mat_y + 22, "<b>Site Log-Odds Sum</b>", size=13.5, fill=PAL["ink"])
    s.text(card_cx, mat_y + 49, "<i>x</i> = ∑<sub><i>i</i>∈mut</sub> [ℓ<sub><i>i</i></sub>(mut) − ℓ<sub><i>i</i></sub>(wt)]", size=12.5, fill=PAL["ink_mid"], max_w=228)
    s.text(card_cx, mat_y + 75, "<i>n</i> = # mutated sites", size=12.0, fill=PAL["ink_muted"], bold=True)

    # Calibration boxes (90-degree orthogonal from Sigma!)
    cal_y, cal_w, cal_h = p3_y + 176, 232.0, 52.0
    for col_x, sig_x, cal_sub, acc_c in [
        (wx, wt_sig_x, "WT calibration (exact add.)", PAL["wt_mid"]),
        (mx, mt_sig_x, "MT calibration (<i>s</i> &gt; 0)", PAL["mt_dark"]),
    ]:
        s.arrow([(sig_x, wt_mcy + 14.5), (sig_x, cal_y - 11), (col_x, cal_y - 11), (col_x, cal_y - 2)], sw=2.0, radius=6.5)
        s.rect(col_x - cal_w / 2, cal_y, cal_w, cal_h, rx=9, fill="#FFFFFF", stroke=acc_c, sw=1.8)
        s.icon_trainable(col_x + cal_w / 2 - 17, cal_y + 17, scale=0.72, on_dark=False)
        s.text(col_x - 4, cal_y + 22, "<b><i>s</i>·<i>x</i> + <i>n</i>·<i>b</i></b>", size=14.2, fill=PAL["ink"], max_w=cal_w - 38)
        s.text(col_x, cal_y + 41, cal_sub, size=12.0, fill=PAL["ink_mid"], max_w=cal_w - 16)

    pred_y = p3_y + 270
    s.arrow([(wx, cal_y + cal_h), (wx, pred_y - 22)], color=PAL["wt_mid"], marker="arr-wt", sw=2.3)
    s.arrow([(mx, cal_y + cal_h), (mx, pred_y - 22)], color=PAL["mt_dark"], marker="arr-mt", sw=2.3)
    s.hexagon(wx, pred_y, 148, 42, "ΔΔ<i>Ĝ</i><sub>WT</sub>", fill=PAL["wt_soft"], stroke=PAL["wt_mid"], ink=PAL["wt_ink"], size=14.5)
    s.hexagon(mx, pred_y, 148, 42, "ΔΔ<i>Ĝ</i><sub>MT</sub>", fill=PAL["mt_pred"], stroke=PAL["mt_dark"], ink=PAL["mt_ink"], size=14.5)

    # --- PANEL 4: UNIFIED OUTPUTS (INFERENCE) & SUPERVISION (TRAINING) (y = 720..1174) ---
    p4_y, p4_h = 720.0, 454.0
    s.panel(px, p4_y, pw, p4_h, "4. UNIFIED OUTPUTS (INFERENCE) & SUPERVISION (TRAINING)", fill=PAL["p4_bg"], stroke=PAL["p4_bdr"])

    # Panel 3 -> Panel 4 stems pass BEHIND Panel 4's dark header bar!
    y_bus = p4_y + 46
    s.arrow([(wx, pred_y + 21), (wx, y_bus)], color=PAL["wt_mid"], sw=2.3, arrow_end=False, under_header=True)
    s.arrow([(mx, pred_y + 21), (mx, y_bus)], color=PAL["mt_dark"], sw=2.3, arrow_end=False, under_header=True)

    # Center Inference Card (x = 334..686, w = 352, centered at cx = 510)
    inf_x, inf_y, inf_w, inf_h = 334.0, p4_y + 54, 352.0, 384.0
    s.rect(inf_x, inf_y, inf_w, inf_h, rx=10, fill="#FFFFFF", stroke=PAL["ink_muted"], sw=1.5)
    s.text(cx, inf_y + 22, "<b>INFERENCE OUTPUTS</b>", size=14.5, fill=PAL["ink"])

    s_card_x, s_card_y, s_card_w, s_card_h = inf_x + 12, inf_y + 48, 158.0, 184.0
    s.rect(s_card_x, s_card_y, s_card_w, s_card_h, rx=8, fill=PAL["wt_card"], stroke=PAL["wt_card_bdr"], sw=1.3)
    s.text(s_card_x + s_card_w / 2, s_card_y + 20, "<b>Singles (WT-only)</b>", size=12.5, fill=PAL["wt_ink"], max_w=148)
    s.text(s_card_x + s_card_w / 2, s_card_y + 38, "1 pass / protein", size=11.5, fill=PAL["ink_mid"], max_w=148)

    m_card_x, m_card_y, m_card_w, m_card_h = inf_x + 182, inf_y + 48, 158.0, 184.0
    s.rect(m_card_x, m_card_y, m_card_w, m_card_h, rx=8, fill=PAL["mt_card"], stroke=PAL["mt_card_bdr"], sw=1.3)
    s.text(m_card_x + m_card_w / 2, m_card_y + 20, "<b>Multimutants</b>", size=12.5, fill=PAL["mt_ink"], max_w=148)
    s.text(m_card_x + m_card_w / 2, m_card_y + 38, "Ensemble (WT + MT)", size=11.5, fill=PAL["ink_mid"], max_w=148)

    ga_cx, ga_cy = s_card_x + 73, s_card_y + 84
    gb_cx, gb_cy = s_card_x + 73, s_card_y + 144
    s.hexagon(ga_cx, ga_cy, 108, 40, "ΔΔ<i>Ĝ</i><sub><i>A</i></sub>", fill=PAL["wt_soft"], stroke=PAL["wt_mid"], ink=PAL["wt_ink"], size=14.0)
    s.hexagon(gb_cx, gb_cy, 108, 40, "ΔΔ<i>Ĝ</i><sub><i>B</i></sub>", fill=PAL["wt_soft"], stroke=PAL["wt_mid"], ink=PAL["wt_ink"], size=14.0)

    ens_cx = m_card_x + m_card_w / 2
    op_ens_y = m_card_y + 78
    comb_cy = m_card_y + 144
    s.circle_op(ens_cx, op_ens_y, 15.5, "½+", size=13.0)
    s.arrow([(ens_cx, op_ens_y + 15.5), (ens_cx, comb_cy - 22)], sw=2.1)
    s.hexagon(ens_cx, comb_cy, 138, 42, "ΔΔ<i>Ĝ</i><sub><i>AB</i></sub>", sublabel="½(WT + MT)", fill=PAL["mt_pred"], stroke=PAL["mt_dark"], ink=PAL["mt_ink"], size=14.0)

    sub_cx, sub_cy = cx, inf_y + 270
    epi_cy = inf_y + 342
    s.arrow([(ens_cx, comb_cy + 21), (ens_cx, inf_y + 246), (sub_cx, inf_y + 246), (sub_cx, sub_cy - 17)], sw=2.1)
    col_ab_x = ga_cx + 64
    s.arrow([(ga_cx + 54, ga_cy), (col_ab_x, ga_cy), (col_ab_x, sub_cy), (sub_cx - 17, sub_cy)], color=PAL["wt_mid"], marker="arr-wt", sw=2.0)
    s.dot(col_ab_x, gb_cy, fill=PAL["wt_mid"])
    s.arrow([(ga_cx + 54, gb_cy), (col_ab_x, gb_cy)], color=PAL["wt_mid"], sw=2.0, arrow_end=False)

    s.circle_op(sub_cx, sub_cy, 15.5, "−", size=14.5)
    s.arrow([(sub_cx, sub_cy + 15.5), (sub_cx, epi_cy - 24)], color=PAL["epi_bdr"], marker="arr-epi", sw=2.2)
    s.hexagon(sub_cx, epi_cy, 168, 46, "ΔΔΔ<i>Ĝ</i><sub><i>AB</i></sub>", sublabel="epistasis", fill=PAL["epi_pred"], stroke=PAL["epi_bdr"], ink=PAL["epi_ink"], size=14.2)

    # Wire Stage 3 Predictions into Inference
    s.dot(wx, y_bus, fill=PAL["wt_mid"])
    s.dot(mx, y_bus, fill=PAL["mt_dark"])

    s.arrow([(wx, y_bus), (322, y_bus), (322, gb_cy), (ga_cx - 55, gb_cy)], color=PAL["wt_mid"], marker="arr-wt", sw=2.1)
    s.dot(322, ga_cy, fill=PAL["wt_mid"])
    s.arrow([(322, ga_cy), (ga_cx - 55, ga_cy)], color=PAL["wt_mid"], marker="arr-wt", sw=2.1)
    s.dot(322, inf_y + 36, fill=PAL["wt_mid"])
    s.arrow([(322, inf_y + 36), (510, inf_y + 36), (510, op_ens_y), (ens_cx - 16.5, op_ens_y)], color=PAL["wt_mid"], marker="arr-wt", sw=2.0)

    s.arrow([(mx, y_bus), (698, y_bus), (698, op_ens_y), (ens_cx + 16.5, op_ens_y)], color=PAL["mt_dark"], marker="arr-mt", sw=2.1)

    # --- LEFT WING (x = 74..314): ΔΔG_A & ΔΔG_B STRADDLING (+) WITH SYMMETRIC 90° BENDS! ---
    port_wt_cy = p4_y + 194
    port_wt_top = port_wt_cy - 11   # p4_y + 183
    port_wt_bot = port_wt_cy + 11   # p4_y + 205
    port_wt_cx = 296.0

    # 1. Green-dashed WT arrow (via sg at x = 256) with 90° bend into top port of WT [TF]!
    sg_x = 256.0
    s.dot(sg_x, y_bus, fill=PAL["wt_mid"])
    s.arrow([(sg_x, y_bus), (sg_x, port_wt_top), (port_wt_cx - 18, port_wt_top)], color=PAL["wt_mid"], marker="arr-wt", sw=2.1, dash="5,3.5", radius=7.0)
    s.stop_grad(sg_x, p4_y + 112, stroke=PAL["wt_mid"])

    # 2. ΔΔG_A (above +) and ΔΔG_B (below +) straddling (+) at y = port_wt_bot (±47 px mirror symmetry!)
    lab_cx = 154.0
    add_op_cx = 222.0
    la_cy = port_wt_bot - 47   # p4_y + 158
    lb_cy = port_wt_bot + 47   # p4_y + 252
    s.text(lab_cx, la_cy - 31, "true singles", size=12.0, fill=PAL["ink_mid"], bold=True)
    s.hexagon(lab_cx, la_cy, 92, 38, "ΔΔ<i>G</i><sub><i>A</i></sub>", fill=PAL["gt_bg"], stroke=PAL["gt_bdr"], size=13.5)
    s.hexagon(lab_cx, lb_cy, 92, 38, "ΔΔ<i>G</i><sub><i>B</i></sub>", fill=PAL["gt_bg"], stroke=PAL["gt_bdr"], size=13.5)

    # Mirror-symmetric 90°-bended emissions from ΔΔG_A (above) and ΔΔG_B (below) into (+) -> [TF]!
    s.arrow([(lab_cx + 46, la_cy), (add_op_cx, la_cy), (add_op_cx, port_wt_bot - 14)], sw=1.9, dash="4,3", radius=7.5)
    s.arrow([(lab_cx + 46, lb_cy), (add_op_cx, lb_cy), (add_op_cx, port_wt_bot + 14)], sw=1.9, dash="4,3", radius=7.5)
    s.circle_op(add_op_cx, port_wt_bot, 12.5, "+", size=13.0)
    s.arrow([(add_op_cx + 12.5, port_wt_bot), (port_wt_cx - 18, port_wt_bot)], sw=1.9, dash="4,3")

    s.floating_portal_box(port_wt_cx, port_wt_cy, w=34, h=46, label="TF")

    # Downward 90°-bended emission from ΔΔG_A & ΔΔG_B into Singles-Only Loss!
    loss_wt_x, loss_wt_y, loss_w, loss_h = 78.0, p4_y + 316, 228.0, 122.0
    s.arrow([(wx, y_bus), (84, y_bus), (84, loss_wt_y - 2)], color=PAL["loss_bdr"], marker="arr-loss", sw=2.1, dash="5,3", radius=8.0)
    s.arrow([(lab_cx - 46, la_cy), (100, la_cy), (100, loss_wt_y - 2)], sw=1.9, dash="4,3", radius=6.0)
    s.arrow([(lab_cx, lb_cy + 19), (lab_cx, loss_wt_y - 2)], sw=1.9, dash="4,3")
    s.rect(loss_wt_x, loss_wt_y, loss_w, loss_h, rx=9, fill=PAL["loss_bg"], stroke=PAL["loss_bdr"], sw=1.7, dash="6,3")
    s.text(loss_wt_x + loss_w / 2, loss_wt_y + 32, "<b>Singles-Only Loss</b>", size=14.0, fill=PAL["loss_ink"], max_w=212)
    s.text(loss_wt_x + loss_w / 2, loss_wt_y + 64, "<b>Tobit NLL + ListMLE</b>", size=13.2, fill=PAL["loss_ink"], max_w=212)
    s.text(loss_wt_x + loss_w / 2, loss_wt_y + 96, "assay-floor aware (trains WT)", size=11.8, fill=PAL["wt_ink"], bold=True, max_w=214)

    # --- RIGHT WING (x = 706..946): POSTERIOR TF SHIFTED UP FOR BALANCED WHITESPACE ---
    port_mt_cx = 724.0
    port_mt_cy = p4_y + 138
    port_mt_top = port_mt_cy - 11
    port_mt_bot = port_mt_cy + 11
    s.floating_portal_box(port_mt_cx, port_mt_cy, w=34, h=46, label="TF")

    tf_cx, tf_cy, tf_w, tf_h = 858.0, port_mt_cy, 158.0, 50.0
    s.text(tf_cx - 12, tf_cy - 36, "<b>Posterior Teacher Forcing</b>", size=12.8, fill=PAL["ink"], max_w=216)
    tf_in_x = (tf_cx - tf_w / 2) + s.hex_edge_offset(tf_h, 11.0) - 1.5
    s.arrow([(port_mt_cx + 17, port_mt_top), (tf_in_x, port_mt_top)], color=PAL["wt_mid"], marker="arr-wt", sw=2.1, dash="5,3.5")
    s.arrow([(port_mt_cx + 17, port_mt_bot), (tf_in_x, port_mt_bot)], sw=1.9, dash="4,3")
    s.hexagon(tf_cx, tf_cy, tf_w, tf_h, "ΔΔ<i>G̃</i><sub>add</sub><sup>TF</sup>", sublabel="ΔΔG<sub>add</sub> + sg(WT)", fill=PAL["gt_bg"], stroke=PAL["gt_bdr"], size=14.0)

    op_tf_cx, op_tf_cy = 830.0, p4_y + 236
    s.arrow([(tf_cx, tf_cy + 25), (tf_cx, p4_y + 194), (op_tf_cx, p4_y + 194), (op_tf_cx, op_tf_cy - 17)], sw=2.0, dash="4,3", radius=7.0)
    s.arrow([(mx, y_bus), (944, y_bus), (944, op_tf_cy), (op_tf_cx + 17, op_tf_cy)], color=PAL["mt_dark"], marker="arr-mt", sw=2.1, dash="5,3", radius=8.0)
    s.circle_op(op_tf_cx, op_tf_cy, 15.5, "½+", size=13.0)

    loss_mt_x, loss_mt_y = 714.0, p4_y + 316
    s.arrow([(op_tf_cx, op_tf_cy + 15.5), (op_tf_cx, loss_mt_y - 2)], color=PAL["loss_bdr"], marker="arr-loss", sw=2.2, dash="5,3")
    s.rect(loss_mt_x, loss_mt_y, loss_w, loss_h, rx=9, fill=PAL["loss_bg"], stroke=PAL["loss_bdr"], sw=1.7, dash="6,3")
    s.hexagon(loss_mt_x + loss_w / 2, loss_mt_y + 32, 138, 36, "ΔΔ<i>G</i><sub><i>AB</i></sub>", fill="#FFFFFF", stroke=PAL["gt_bdr"], size=13.8)
    s.text(loss_mt_x + loss_w / 2, loss_mt_y + 74, "<b>Tobit NLL + ListMLE</b>", size=13.2, fill=PAL["loss_ink"], max_w=212)
    s.text(loss_mt_x + loss_w / 2, loss_mt_y + 102, "multi-mutant loss (trains MT)", size=11.8, fill=PAL["mt_ink"], bold=True, max_w=214)

    # --- LARGE PROMINENT GRADIENT UPDATE ARROWS IN WIDE OUTER MARGINS ---
    s.gradient_arrow([(loss_wt_x, loss_wt_y + 62), (grad_l, loss_wt_y + 62), (grad_l, lora_y + bh / 2), (wx - bw / 2 - 3, lora_y + bh / 2)], large=True)
    s.gradient_arrow([(grad_l, cal_y + cal_h / 2), (wx - cal_w / 2 - 3, cal_y + cal_h / 2)], large=True, t_junction=True)

    s.gradient_arrow([(loss_mt_x + loss_w, loss_mt_y + 62), (grad_r, loss_mt_y + 62), (grad_r, lora_y + bh / 2), (mx + bw / 2 + 3, lora_y + bh / 2)], large=True)
    s.gradient_arrow([(grad_r, cal_y + cal_h / 2), (mx + cal_w / 2 + 3, cal_y + cal_h / 2)], large=True, t_junction=True)

    # --- OUTSIDE LEGEND ON PURE #FFFFFF (y = 1214) ---
    s.draw_outside_legend(1214)
    return s.to_svg()


# ===========================================================================
# 2. HORIZONTAL UNIFIED SCHEMATIC (1660 x 836) — Synced Companion
# ===========================================================================
def build_horizontal_svg(include_label: bool = False) -> str:
    title = (
        "ESM-MSR Architecture (Horizontal Unified Schematic, Subplot a)"
        if include_label
        else "ESM-MSR Architecture (Horizontal Unified Schematic)"
    )
    s = PublicationSVG(1660, 836, title)

    if include_label:
        s.nature_subplot_label(18.0, 36.0, "a")

    x0, y0, ph = 36.0, 58.0, 668.0
    grad_top, grad_bot = 26.0, 758.0

    cols = [
        (316.0, "1. INPUTS", PAL["p1_bg"], PAL["p1_bdr"]),
        (264.0, "2. BACKBONE & LoRAs", PAL["p2_bg"], PAL["p2_bdr"]),
        (454.0, "3. LOGITS & CALIBRATION", PAL["p3_bg"], PAL["p3_bdr"]),
        (572.0, "4. OUTPUTS & SUPERVISION", PAL["p4_bg"], PAL["p4_bdr"]),
    ]
    s.draw_h_columns(x0, y0, ph, cols)

    wy, cy, my = 212.0, 406.0, 606.0

    # --- COLUMN 1: 1. INPUTS (x = 36..352) ---
    s.rect(52, y0 + 52, 168, 24, rx=12, fill=PAL["wt_soft"], stroke=PAL["wt_mid"], sw=1.3)
    s.text(136, y0 + 68.5, "1 pass / protein", size=12.0, fill=PAL["wt_ink"], bold=True, max_w=154)
    _, wt_s_r, _ = s.draw_seq_strip(140, wy - 14, "ab", ["M", "N", "E", "F", "S", "…"], is_mt=False, stacked=0)

    s.rect(244, wy - 18, 94, 36, rx=6, fill=PAL["frozen_bg"], stroke=PAL["frozen_bdr"], sw=1.3)
    s.text(291, wy + 5, "Tokenizer", size=12.8, bold=True, max_w=84)
    s.arrow([(wt_s_r + 4, wy), (242, wy)], sw=2.1)

    s.draw_gb1_domain(118, cy - 12, scale=3.30)
    s.rect(52, cy + 38, 132, 32, rx=6, fill="#FFFFFF", stroke=PAL["frozen_bdr"], sw=1.3)
    s.text(118, cy + 59, "WT BB Coords", size=12.5, bold=True, max_w=122)

    vq_x, vq_y, vq_w, vq_h = 216.0, cy - 54, 122.0, 108.0
    s.arrow([(168, cy - 6), (vq_x - 2, cy - 6)], sw=2.1)
    s.rect(vq_x, vq_y, vq_w, vq_h, rx=7, fill=PAL["frozen_bg"], stroke=PAL["frozen_bdr"], sw=1.4)
    s.icon_snowflake(vq_x + vq_w - 15, vq_y + 15, r=6.8)
    s.text(vq_x + vq_w / 2, vq_y + 44, "<b>ESM3</b>", size=14.0)
    s.text(vq_x + vq_w / 2, vq_y + 64, "<b>VQ-VAE</b>", size=14.0)
    s.text(vq_x + vq_w / 2, vq_y + 88, "Struct Tokens", size=11.8, fill=PAL["ink_mid"], max_w=114)

    _, mt_s_r, _ = s.draw_seq_strip(140, my - 14, "AB", ["M", "N", "D", "F", "G", "…"], is_mt=True, stacked=2)
    s.rect(52, my + 36, 168, 24, rx=12, fill=PAL["mt_soft"], stroke=PAL["mt_dark"], sw=1.3)
    s.text(136, my + 52.5, "1 pass / variant", size=12.0, fill=PAL["mt_ink"], bold=True, max_w=154)

    s.rect(244, my - 18, 94, 36, rx=6, fill=PAL["frozen_bg"], stroke=PAL["frozen_bdr"], sw=1.3)
    s.text(291, my + 5, "Tokenizer", size=12.8, bold=True, max_w=84)
    s.arrow([(mt_s_r + 12, my), (242, my)], sw=2.1)

    # --- COLUMN 2: 2. BACKBONE & LoRAs (x = 352..616) ---
    lx, lw, lh = 378.0, 212.0, 96.0
    wt_ly, esm_ly, mt_ly = wy - lh / 2, cy - lh / 2, my - lh / 2

    s.rect(lx + 90, wt_ly + lh - 2, 16, esm_ly - (wt_ly + lh) + 4, rx=0, fill=PAL["frozen_bg"], stroke=PAL["frozen_bdr"], sw=1.2)
    s.rect(lx + 108, wt_ly + lh - 2, 16, esm_ly - (wt_ly + lh) + 4, rx=0, fill=PAL["wt_dark"], stroke=PAL["wt_dark"], sw=1.2)
    s.rect(lx + 90, esm_ly + lh - 2, 16, mt_ly - (esm_ly + lh) + 4, rx=0, fill=PAL["frozen_bg"], stroke=PAL["frozen_bdr"], sw=1.2)
    s.rect(lx + 108, esm_ly + lh - 2, 16, mt_ly - (esm_ly + lh) + 4, rx=0, fill=PAL["mt_lora"], stroke=PAL["mt_lora"], sw=1.2)

    s.rect(lx, wt_ly, lw, lh, rx=12, fill=PAL["wt_dark"], stroke=PAL["ink"], sw=1.6)
    s.icon_trainable(lx + lw - 18, wt_ly + 19, scale=0.82, on_dark=True)
    s.text(lx + lw / 2, wt_ly + 34, "<b>ESM-MSR WT LoRA</b>", size=14.0, fill="#FFFFFF", max_w=lw - 32)
    s.rect(lx + 24, wt_ly + 50, lw - 48, 24, rx=12, fill=PAL["wt_soft"], stroke=None)
    s.text(lx + lw / 2, wt_ly + 66.5, "rank <i>r</i> = 2 · QKV/FFN", size=11.8, fill=PAL["wt_ink"], bold=True, max_w=lw - 56)

    s.rect(lx, esm_ly, lw, lh, rx=9, fill=PAL["frozen_bg"], stroke=PAL["frozen_bdr"], sw=1.5)
    s.icon_snowflake(lx + lw - 16, esm_ly + 16, r=7.0)
    s.text(lx + lw / 2, esm_ly + 34, "<b>ESM3 Transformer</b>", size=14.0, max_w=lw - 26)
    s.text(lx + lw / 2, esm_ly + 56, "1.4B shared backbone", size=12.0, fill=PAL["ink_mid"], bold=True, max_w=lw - 16)
    s.text(lx + lw / 2, esm_ly + 77, "Geom. + Token Attention", size=11.5, fill=PAL["ink_muted"], max_w=lw - 14)

    for k in (2, 1):
        s.rect(lx + k * 4.5, mt_ly - k * 4.5, lw, lh, rx=12, fill=PAL["mt_soft"], stroke=PAL["mt_mid"], sw=1.0)
    s.rect(lx, mt_ly, lw, lh, rx=12, fill=PAL["mt_lora"], stroke=PAL["ink"], sw=1.6)
    s.icon_trainable(lx + lw - 18, mt_ly + 19, scale=0.82, on_dark=True)
    s.text(lx + lw / 2, mt_ly + 34, "<b>ESM-MSR MT LoRA</b>", size=14.0, fill="#FFFFFF", max_w=lw - 32)
    s.rect(lx + 24, mt_ly + 50, lw - 48, 24, rx=12, fill=PAL["mt_soft"], stroke=None)
    s.text(lx + lw / 2, mt_ly + 66.5, "rank <i>r</i> = 16 · QKV/FFN", size=11.8, fill=PAL["mt_ink"], bold=True, max_w=lw - 56)

    s.arrow([(338, wy), (lx - 2, wy)], color=PAL["wt_mid"], marker="arr-wt", sw=2.2)
    s.arrow([(338, cy - 12), (lx - 2, cy - 12)], sw=2.1)
    s.arrow([(118, cy + 70), (118, cy + 84), (360, cy + 84), (360, cy + 16), (lx - 2, cy + 16)], sw=2.0)
    s.arrow([(338, my), (lx - 2, my)], color=PAL["mt_dark"], marker="arr-mt", sw=2.2)

    # --- COLUMN 3: 3. LOGITS & CALIBRATION (x = 616..1070) ---
    wt_mcx_h, wt_sig_x, _ = s.draw_logit_matrix_and_diffs(704.0, wy - 37, stacked=0, cached_side=True)
    _, mt_sig_x, _ = s.draw_logit_matrix_and_diffs(664.0, my - 37, stacked=2, cached_side=False)
    s.arrow([(lx + lw, wy - 42), (602, wy - 42), (602, wy - 54), (wt_mcx_h, wy - 54), (wt_mcx_h, wy - 39)], color=PAL["wt_mid"], marker="arr-wt", sw=2.2, radius=5.5)
    s.arrow([(lx + lw + 9, my), (662, my)], color=PAL["mt_dark"], marker="arr-mt", sw=2.2)

    cal_hx, cal_hw, cal_hh = 948.0, 110.0, 56.0
    for lane_y, sig_x, c_sub, acc_c in [
        (wy, wt_sig_x, "WT calib", PAL["wt_mid"]),
        (my, mt_sig_x, "MT calib", PAL["mt_dark"]),
    ]:
        s.arrow([(sig_x + 14.5, lane_y), (cal_hx - 2, lane_y)], sw=2.0)
        s.rect(cal_hx, lane_y - cal_hh / 2, cal_hw, cal_hh, rx=9, fill="#FFFFFF", stroke=acc_c, sw=1.8)
        s.icon_trainable(cal_hx + cal_hw - 14, lane_y - cal_hh / 2 + 16, scale=0.68, on_dark=False)
        s.text(cal_hx + cal_hw / 2 - 5, lane_y - 4, "<b><i>s</i>·<i>x</i>+<i>n</i>·<i>b</i></b>", size=13.0, max_w=cal_hw - 24)
        s.text(cal_hx + cal_hw / 2, lane_y + 17, c_sub, size=11.8, fill=PAL["ink_mid"], max_w=cal_hw - 10)

    s.rect(698, cy - 48, 290, 96, rx=8, fill="#FFFFFF", stroke=PAL["p3_bdr"], sw=1.4)
    s.text(843, cy - 22, "<b>Site Log-Odds Sum</b>", size=13.5, fill=PAL["ink"])
    s.text(843, cy + 6, "<i>x</i> = ∑<sub><i>i</i>∈mut</sub> [ℓ<sub><i>i</i></sub>(mut) − ℓ<sub><i>i</i></sub>(wt)]", size=12.5, fill=PAL["ink_mid"], max_w=274)
    s.text(843, cy + 32, "<i>n</i> = # mutated sites", size=12.0, fill=PAL["ink_muted"], bold=True)

    # --- COLUMN 4: 4. OUTPUTS & SUPERVISION (x = 1070..1642) — 100% Orthogonal! ---
    pred_hx = 1138.0
    s.arrow([(cal_hx + cal_hw, wy), (pred_hx - 58, wy)], color=PAL["wt_mid"], marker="arr-wt", sw=2.2)
    s.arrow([(cal_hx + cal_hw, my), (pred_hx - 58, my)], color=PAL["mt_dark"], marker="arr-mt", sw=2.2)
    s.hexagon(pred_hx, wy, 114, 40, "ΔΔ<i>Ĝ</i><sub>WT</sub>", fill=PAL["wt_soft"], stroke=PAL["wt_mid"], ink=PAL["wt_ink"], size=13.8)
    s.hexagon(pred_hx, my, 114, 40, "ΔΔ<i>Ĝ</i><sub>MT</sub>", fill=PAL["mt_pred"], stroke=PAL["mt_dark"], ink=PAL["mt_ink"], size=13.8)

    h_inf_x, h_inf_y, h_inf_w, h_inf_h = 1090.0, 290.0, 536.0, 224.0
    s.rect(h_inf_x, h_inf_y, h_inf_w, h_inf_h, rx=10, fill="#FFFFFF", stroke=PAL["ink_muted"], sw=1.5)
    s.text(h_inf_x + h_inf_w / 2, h_inf_y + 22, "<b>INFERENCE OUTPUTS</b>", size=13.8, fill=PAL["ink"])

    hs_x, hs_y = h_inf_x + 12, h_inf_y + 44
    s.rect(hs_x, hs_y, 166, 164, rx=8, fill=PAL["wt_card"], stroke=PAL["wt_card_bdr"], sw=1.3)
    s.text(hs_x + 83, hs_y + 20, "<b>Singles (WT-only)</b>", size=12.0, fill=PAL["wt_ink"], max_w=156)
    h_ga_cx, h_ga_cy = hs_x + 83, hs_y + 64
    h_gb_cx, h_gb_cy = hs_x + 83, hs_y + 116
    s.hexagon(h_ga_cx, h_ga_cy, 104, 36, "ΔΔ<i>Ĝ</i><sub><i>A</i></sub>", fill=PAL["wt_soft"], stroke=PAL["wt_mid"], ink=PAL["wt_ink"], size=13.5)
    s.hexagon(h_gb_cx, h_gb_cy, 104, 36, "ΔΔ<i>Ĝ</i><sub><i>B</i></sub>", fill=PAL["wt_soft"], stroke=PAL["wt_mid"], ink=PAL["wt_ink"], size=13.5)
    s.text(hs_x + 83, hs_y + 153, "1 pass / protein", size=11.5, fill=PAL["ink_mid"])

    hm_x, hm_y = hs_x + 186, hs_y
    s.rect(hm_x, hm_y, 168, 164, rx=8, fill=PAL["mt_card"], stroke=PAL["mt_card_bdr"], sw=1.3)
    s.text(hm_x + 84, hm_y + 20, "<b>Multimutants</b>", size=12.0, fill=PAL["mt_ink"], max_w=158)
    h_ens_op_cx, h_ens_op_cy = hm_x + 42, hm_y + 84
    h_comb_cx, h_comb_cy = hm_x + 116, hm_y + 84
    s.circle_op(h_ens_op_cx, h_ens_op_cy, 14.5, "½+", size=12.5)
    s.arrow([(h_ens_op_cx + 14.5, h_ens_op_cy), (h_comb_cx - 48, h_comb_cy)], sw=2.0)
    s.hexagon(h_comb_cx, h_comb_cy, 94, 40, "ΔΔ<i>Ĝ</i><sub><i>AB</i></sub>", fill=PAL["mt_pred"], stroke=PAL["mt_dark"], ink=PAL["mt_ink"], size=13.5)
    s.text(hm_x + 84, hm_y + 132, "Ensemble ½(WT+MT)", size=11.5, fill=PAL["mt_ink"], bold=True, max_w=158)
    s.text(hm_x + 84, hm_y + 153, "1 pass / variant", size=11.5, fill=PAL["ink_mid"])

    h_sub_cx, h_sub_cy = hm_x + 194, h_comb_cy
    h_epi_cx, h_epi_cy = hm_x + 266, h_comb_cy
    s.arrow([(h_comb_cx + 47, h_comb_cy), (h_sub_cx - 16, h_sub_cy)], sw=2.0)
    s.arrow([(h_ga_cx + 52, h_ga_cy), (hs_x + 150, h_ga_cy), (hs_x + 150, h_inf_y + 215), (h_sub_cx, h_inf_y + 215), (h_sub_cx, h_sub_cy + 16)], color=PAL["wt_mid"], marker="arr-wt", sw=1.9)
    s.dot(hs_x + 150, h_gb_cy, fill=PAL["wt_mid"])
    s.arrow([(h_gb_cx + 52, h_gb_cy), (hs_x + 150, h_gb_cy)], color=PAL["wt_mid"], sw=1.9, arrow_end=False)

    s.circle_op(h_sub_cx, h_sub_cy, 14.5, "−", size=14.0)
    s.arrow([(h_sub_cx + 14.5, h_sub_cy), (h_epi_cx - 52, h_epi_cy)], color=PAL["epi_bdr"], marker="arr-epi", sw=2.1)
    s.hexagon(h_epi_cx, h_epi_cy, 102, 42, "ΔΔΔ<i>Ĝ</i><sub><i>AB</i></sub>", fill=PAL["epi_pred"], stroke=PAL["epi_bdr"], ink=PAL["epi_ink"], size=13.5)
    s.text(h_epi_cx, h_epi_cy + 36, "epistasis", size=12.0, fill=PAL["epi_ink"], bold=True)

    alley_x = hs_x + 176
    s.arrow([(pred_hx, wy + 20), (pred_hx, 256), (1078, 256), (1078, h_gb_cy), (h_gb_cx - 53, h_gb_cy)], color=PAL["wt_mid"], marker="arr-wt", sw=2.0)
    s.dot(1078, h_ga_cy, fill=PAL["wt_mid"])
    s.arrow([(1078, h_ga_cy), (h_ga_cx - 53, h_ga_cy)], color=PAL["wt_mid"], marker="arr-wt", sw=2.0)
    s.dot(1078, h_inf_y + 34, fill=PAL["wt_mid"])
    s.arrow([(1078, h_inf_y + 34), (alley_x, h_inf_y + 34), (alley_x, h_ens_op_cy - 6), (h_ens_op_cx - 15, h_ens_op_cy - 6)], color=PAL["wt_mid"], marker="arr-wt", sw=2.0)

    s.arrow([(pred_hx, my - 20), (pred_hx, 526), (alley_x, 526), (alley_x, h_ens_op_cy + 6), (h_ens_op_cx - 15, h_ens_op_cy + 6)], color=PAL["mt_dark"], marker="arr-mt", sw=2.0)

    # Top band of Col 4: stacked/orthogonal ΔΔG_A & ΔΔG_B -> (+) -> [TF] & Singles-Only Loss
    h_la_cx, h_lb_cx, h_lab_y = 1248.0, 1344.0, y0 + 66
    s.hexagon(h_la_cx, h_lab_y, 80, 34, "ΔΔ<i>G</i><sub><i>A</i></sub>", fill=PAL["gt_bg"], stroke=PAL["gt_bdr"], size=13.0)
    s.hexagon(h_lb_cx, h_lab_y, 80, 34, "ΔΔ<i>G</i><sub><i>B</i></sub>", fill=PAL["gt_bg"], stroke=PAL["gt_bdr"], size=13.0)

    h_loss_wt_x, h_loss_wt_y = 1454.0, y0 + 50
    s.arrow([(h_la_cx, h_lab_y - 17), (h_la_cx, y0 + 42), (1426, y0 + 42), (1426, h_lab_y)], sw=1.8, dash="4,3", arrow_end=False)
    s.dot(1426, h_lab_y, fill=PAL["ink_mid"])
    s.arrow([(h_lb_cx + 40, h_lab_y), (h_loss_wt_x - 2, h_lab_y)], sw=1.9, dash="4,3")

    s.dot(1208, wy, fill=PAL["wt_mid"])
    s.arrow([(pred_hx + 57, wy), (1208, wy), (1208, 248), (1434, 248), (1434, h_loss_wt_y + 72), (h_loss_wt_x - 2, h_loss_wt_y + 72)], color=PAL["loss_bdr"], marker="arr-loss", sw=2.0, dash="5,3")
    s.rect(h_loss_wt_x, h_loss_wt_y, 174, 96, rx=8, fill=PAL["loss_bg"], stroke=PAL["loss_bdr"], sw=1.6, dash="6,3")
    s.text(h_loss_wt_x + 87, h_loss_wt_y + 28, "<b>Singles-Only Loss</b>", size=13.0, fill=PAL["loss_ink"], max_w=162)
    s.text(h_loss_wt_x + 87, h_loss_wt_y + 54, "<b>Tobit + ListMLE</b>", size=12.5, fill=PAL["loss_ink"], max_w=162)
    s.text(h_loss_wt_x + 87, h_loss_wt_y + 78, "trains WT head", size=11.8, fill=PAL["wt_ink"], bold=True)

    h_port_wt_cx, h_port_wt_cy = 1396.0, 200.0
    h_add_cx, h_add_cy = 1296.0, h_port_wt_cy - 11
    s.arrow([(h_la_cx, h_lab_y + 17), (h_la_cx, h_add_cy), (h_add_cx - 13, h_add_cy)], sw=1.8, dash="4,3", radius=6.0)
    s.arrow([(h_lb_cx, h_lab_y + 17), (h_lb_cx, h_add_cy - 16), (h_add_cx, h_add_cy - 16), (h_add_cx, h_add_cy - 13)], sw=1.8, dash="4,3", radius=5.0)
    s.circle_op(h_add_cx, h_add_cy, 11.5, "+", size=12.5)
    s.arrow([(h_add_cx + 11.5, h_add_cy), (h_port_wt_cx - 18, h_add_cy)], sw=1.9, dash="4,3")

    s.arrow([(1208, wy), (h_port_wt_cx - 18, wy)], color=PAL["wt_mid"], marker="arr-wt", sw=2.0, dash="5,3.5")
    s.stop_grad(1268, wy, stroke=PAL["wt_mid"])
    s.floating_portal_box(h_port_wt_cx, h_port_wt_cy, w=34, h=44, label="TF")

    # Bottom band of Col 4: Parallel horizontal [TF] emissions -> Posterior TF -> Multi Loss (100% orthogonal!)
    h_port_mt_cx, h_port_mt_cy = 1218.0, 560.0
    s.floating_portal_box(h_port_mt_cx, h_port_mt_cy, w=34, h=44, label="TF")

    h_tf_cx, h_tf_cy, h_tf_w, h_tf_h = 1342.0, 560.0, 144.0, 44.0
    h_tf_in_x = (h_tf_cx - h_tf_w / 2) + s.hex_edge_offset(h_tf_h, 10.0) - 1.5
    s.arrow([(h_port_mt_cx + 17, h_port_mt_cy - 10), (h_tf_in_x, h_port_mt_cy - 10)], sw=1.9, dash="4,3")
    s.arrow([(h_port_mt_cx + 17, h_port_mt_cy + 10), (h_tf_in_x, h_port_mt_cy + 10)], color=PAL["wt_mid"], marker="arr-wt", sw=2.0, dash="5,3.5")
    s.hexagon(h_tf_cx, h_tf_cy, h_tf_w, h_tf_h, "ΔΔ<i>G̃</i><sub>add</sub><sup>TF</sup>", sublabel="Posterior TF", fill=PAL["gt_bg"], stroke=PAL["gt_bdr"], size=13.5)

    h_tf_op_cx, h_tf_op_cy = 1412.0, 636.0
    s.arrow([(pred_hx + 57, my), (1226, my), (1226, h_tf_op_cy), (h_tf_op_cx - 16, h_tf_op_cy)], color=PAL["mt_dark"], marker="arr-mt", sw=2.0, dash="5,3")
    s.arrow([(h_tf_cx, h_tf_cy + 22), (h_tf_cx, 602), (h_tf_op_cx, 602), (h_tf_op_cx, h_tf_op_cy - 16)], sw=1.9, dash="4,3", radius=6.0)
    s.circle_op(h_tf_op_cx, h_tf_op_cy, 14.5, "½+", size=12.5)

    h_loss_mt_x, h_loss_mt_y = 1454.0, 594.0
    s.arrow([(h_tf_op_cx + 14.5, h_tf_op_cy), (h_loss_mt_x - 2, h_tf_op_cy)], color=PAL["loss_bdr"], marker="arr-loss", sw=2.1, dash="5,3")
    s.rect(h_loss_mt_x, h_loss_mt_y, 174, 96, rx=8, fill=PAL["loss_bg"], stroke=PAL["loss_bdr"], sw=1.6, dash="6,3")
    s.hexagon(h_loss_mt_x + 87, h_loss_mt_y + 28, 126, 32, "ΔΔ<i>G</i><sub><i>AB</i></sub>", fill="#FFFFFF", stroke=PAL["gt_bdr"], size=13.2)
    s.text(h_loss_mt_x + 87, h_loss_mt_y + 62, "<b>Tobit + ListMLE</b>", size=12.5, fill=PAL["loss_ink"], max_w=162)
    s.text(h_loss_mt_x + 87, h_loss_mt_y + 84, "trains MT head", size=11.8, fill=PAL["mt_ink"], bold=True)

    # Top & bottom outer gradient rails passing BEHIND the top header bar!
    s.gradient_arrow([(h_loss_wt_x + 87, h_loss_wt_y), (h_loss_wt_x + 87, grad_top), (lx + lw / 2, grad_top), (lx + lw / 2, wt_ly - 3)], large=True, under_header=True)
    s.gradient_arrow([(cal_hx + cal_hw / 2, grad_top), (cal_hx + cal_hw / 2, wy - cal_hh / 2 - 3)], large=True, t_junction=True, under_header=True)

    s.gradient_arrow([(h_loss_mt_x + 87, h_loss_mt_y + 96), (h_loss_mt_x + 87, grad_bot), (lx + lw / 2, grad_bot), (lx + lw / 2, mt_ly + lh + 3)], large=True)
    s.gradient_arrow([(cal_hx + cal_hw / 2, grad_bot), (cal_hx + cal_hw / 2, my + cal_hh / 2 + 3)], large=True, t_junction=True)

    s.draw_outside_legend(808)
    return s.to_svg()


# ---------------------------------------------------------------------------
# High-DPI PNG Rendering via librsvg-2 + libcairo
# ---------------------------------------------------------------------------
def _setup_rsvg_cairo():
    rsvg = ctypes.CDLL("librsvg-2.so.2")
    cairo = ctypes.CDLL("libcairo.so.2")
    rsvg.rsvg_handle_new_from_data.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
    rsvg.rsvg_handle_new_from_data.restype = ctypes.c_void_p
    rsvg.rsvg_handle_render_cairo.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    rsvg.rsvg_handle_render_cairo.restype = ctypes.c_int
    cairo.cairo_image_surface_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    cairo.cairo_image_surface_create.restype = ctypes.c_void_p
    cairo.cairo_create.argtypes = [ctypes.c_void_p]
    cairo.cairo_create.restype = ctypes.c_void_p
    cairo.cairo_scale.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
    cairo.cairo_surface_write_to_png.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    cairo.cairo_surface_write_to_png.restype = ctypes.c_int
    cairo.cairo_destroy.argtypes = [ctypes.c_void_p]
    cairo.cairo_surface_destroy.argtypes = [ctypes.c_void_p]
    return rsvg, cairo


def render_svg_to_png(svg_path: Path, png_path: Path, scale: float = 1.8) -> None:
    root = ET.parse(svg_path).getroot()
    w = int(float(root.attrib["width"]))
    h = int(float(root.attrib["height"]))
    for elem in root.iter():
        if elem.tag.endswith("image") or elem.tag.endswith("filter"):
            raise AssertionError(f"Disallowed element {elem.tag} in {svg_path}")

    svg_bytes = svg_path.read_bytes()
    rsvg, cairo = _setup_rsvg_cairo()
    err = ctypes.c_void_p()
    handle = rsvg.rsvg_handle_new_from_data(svg_bytes, len(svg_bytes), ctypes.byref(err))
    if not handle:
        raise RuntimeError(f"librsvg failed to parse {svg_path}")

    pw, ph = int(round(w * scale)), int(round(h * scale))
    surf = cairo.cairo_image_surface_create(0, pw, ph)
    cr = cairo.cairo_create(surf)
    cairo.cairo_scale(cr, ctypes.c_double(scale), ctypes.c_double(scale))
    ok = rsvg.rsvg_handle_render_cairo(handle, cr)
    if not ok:
        raise RuntimeError(f"librsvg failed to render {svg_path}")
    cairo.cairo_surface_write_to_png(surf, str(png_path).encode("utf-8"))
    cairo.cairo_destroy(cr)
    cairo.cairo_surface_destroy(surf)

    im = Image.open(png_path)
    assert im.size == (pw, ph), f"Expected {(pw, ph)}, got {im.size}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stem, builder in [
        ("esm_msr_vertical", build_vertical_svg),
        ("esm_msr_horizontal", build_horizontal_svg),
    ]:
        svg_str = builder()
        svg_path = OUT_DIR / f"{stem}.svg"
        png_path = OUT_DIR / f"{stem}.png"
        svg_path.write_text(svg_str, encoding="utf-8")
        render_svg_to_png(svg_path, png_path, scale=1.7)
        print(f"Generated {svg_path.name} ({svg_path.stat().st_size:,} B) & {png_path.name} ({png_path.stat().st_size:,} B)")


if __name__ == "__main__":
    main()
