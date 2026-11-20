from pathlib import Path
import math

# Exact 5-type palette from analysis/11f_epistasis_types.py (fig7):
PAL = {
    "comp":  {"plane": "#2166AC", "node": "#2166AC", "ui": "#2166AC", "txt": "#FFFFFF", "op": "0.62"},
    "wpos":  {"plane": "#92C5DE", "node": "#92C5DE", "ui": "#3682B8", "txt": "#1A1A1A", "op": "0.72"},
    "neut":  {"plane": "#D9D9D9", "node": "#D9D9D9", "ui": "#333333", "txt": "#1A1A1A", "op": "0.58"},
    "wneg":  {"plane": "#F4A582", "node": "#F4A582", "ui": "#D4633B", "txt": "#1A1A1A", "op": "0.72"},
    "clash": {"plane": "#B2182B", "node": "#B2182B", "ui": "#B2182B", "txt": "#FFFFFF", "op": "0.62"},
}
COL_SINGLE_POS = "#2166AC"
COL_SINGLE_NEG = "#D62828"

def fmt(v):
    return f"{int(v)}" if abs(v - round(v)) < 1e-6 else f"{v:.2f}".rstrip("0").rstrip(".")

def edge_path(p1, p2, t=0.5):
    x1, y1 = p1
    x2, y2 = p2
    xm = x1 + t * (x2 - x1)
    ym = y1 + t * (y2 - y1)
    return f"M {fmt(x1)},{fmt(y1)} L {fmt(xm)},{fmt(ym)} L {fmt(x2)},{fmt(y2)}"

def arrow_label_svg(p1, p2, text, t=0.5, side="above"):
    x1, y1 = p1
    x2, y2 = p2
    cx = x1 + t * (x2 - x1)
    cy = y1 + t * (y2 - y1)
    ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
    dy = -9.5 if side == "above" else 20.5
    return (
        f'<text transform="translate({fmt(cx)},{fmt(cy)}) rotate({ang:.2f})" '
        f'x="0" y="{fmt(dy)}" fill="#333333" text-anchor="middle">{text}</text>'
    )

def make_defs(prefix=""):
    p = f"{prefix}-" if prefix else ""
    return f'''  <defs>
    <marker id="{p}mid-arrow" markerWidth="16" markerHeight="13" refX="11.5" refY="6.5" orient="auto" markerUnits="userSpaceOnUse">
      <path d="M 0,0.5 L 16,6.5 L 0,12.5 L 4,6.5 Z" fill="#1A1A1A"/>
    </marker>
    <marker id="{p}wt-start" markerWidth="7" markerHeight="6" refX="0.8" refY="3" orient="auto" markerUnits="userSpaceOnUse">
      <path d="M 6.5,0.5 L 0.5,3 L 6.5,5.5 L 4.8,3 Z" fill="#CCCCCC"/>
    </marker>
    <marker id="{p}wt-end" markerWidth="7" markerHeight="6" refX="6.2" refY="3" orient="auto" markerUnits="userSpaceOnUse">
      <path d="M 0.5,0.5 L 6.5,3 L 0.5,5.5 L 2.2,3 Z" fill="#CCCCCC"/>
    </marker>
    <marker id="{p}arr-comp" markerWidth="7.5" markerHeight="7.5" refX="6.5" refY="3.75" orient="auto" markerUnits="userSpaceOnUse">
      <path d="M 0,0.3 L 7.2,3.75 L 0,7.2 L 1.9,3.75 Z" fill="{PAL['comp']['ui']}"/>
    </marker>
    <marker id="{p}arr-wpos" markerWidth="7.5" markerHeight="7.5" refX="6.5" refY="3.75" orient="auto" markerUnits="userSpaceOnUse">
      <path d="M 0,0.3 L 7.2,3.75 L 0,7.2 L 1.9,3.75 Z" fill="{PAL['wpos']['ui']}"/>
    </marker>
    <marker id="{p}arr-wneg" markerWidth="7.5" markerHeight="7.5" refX="6.5" refY="3.75" orient="auto" markerUnits="userSpaceOnUse">
      <path d="M 0,0.3 L 7.2,3.75 L 0,7.2 L 1.9,3.75 Z" fill="{PAL['wneg']['ui']}"/>
    </marker>
    <marker id="{p}arr-clash" markerWidth="7.5" markerHeight="7.5" refX="6.5" refY="3.75" orient="auto" markerUnits="userSpaceOnUse">
      <path d="M 0,0.3 L 7.2,3.75 L 0,7.2 L 1.9,3.75 Z" fill="{PAL['clash']['ui']}"/>
    </marker>
    <marker id="{p}arr-dark" markerWidth="7.5" markerHeight="7.5" refX="6.5" refY="3.75" orient="auto" markerUnits="userSpaceOnUse">
      <path d="M 0,0.3 L 7.2,3.75 L 0,7.2 L 1.9,3.75 Z" fill="#333333"/>
    </marker>
  </defs>'''

def make_draft1_mirror(prefix=""):
    p = f"{prefix}-" if prefix else ""
    W, H = 536, 502
    x_wt, x_Ab, x_aB, x_AB = 40, 204, 256, 420
    x_tick_end = 520
    S = 70  # 1u = 70px (~0.75 kcal/mol), 2u = 140px (~1.5 kcal/mol)

    t_wt = 112
    t_Ab = t_wt - S           # 42
    t_A  = t_wt - S // 2      # 77  (additive / near-neutral)
    t_aB = t_wt + S // 2      # 147
    t_B  = t_wt + S // 2      # 147 (weak negative: -1u)
    t_C  = t_wt + 3 * S // 2  # 217 (clash / negative sign: -2u)
    t_floor_back, t_floor_front = 244, 269.5

    b_wt = 318
    b_D  = b_wt - S // 2      # 283 (compensation / positive sign: +2u)
    b_Ab = b_wt + S // 2      # 353
    b_E  = b_wt + S // 2      # 353 (weak positive: +1u)
    b_aB = b_wt + S           # 388 (exact midpoint of E and F)
    b_F  = b_wt + 3 * S // 2  # 423 (additive / near-neutral)
    b_floor_back, b_floor_front = 450, 475.5

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
  <title>Thermodynamic Epistasis Cycles — 5 Main Types (Draft 1: 6-Plane Symmetric Mirror)</title>

{make_defs(prefix)}

  <!-- TOP PANEL: Net stabilizing singles sum (A=near-neutral, B=weak negative, C=clash/negative sign) -->
  <g id="{p}panel-top" font-family="'Liberation Sans', Arial, Helvetica, 'DejaVu Sans', sans-serif" font-size="14">
    <polygon points="{x_wt},{t_floor_front} {x_Ab},{t_floor_back} {x_AB},{t_floor_back} {x_aB},{t_floor_front}" fill="#333333" fill-opacity="0.5"/>

    <g>
      <line x1="{fmt(x_wt + 7.5)}" y1="{t_wt}" x2="{fmt(x_AB - 7.5)}" y2="{t_wt}" stroke="#CCCCCC" stroke-width="1.1" marker-start="url(#{p}wt-start)" marker-end="url(#{p}wt-end)"/>
      <line x1="{x_AB}" y1="{t_A}" x2="{x_tick_end}" y2="{t_A}" stroke="#333333" stroke-width="1.1"/>
      <line x1="{x_AB}" y1="{t_wt}" x2="{x_tick_end}" y2="{t_wt}" stroke="#333333" stroke-width="1.1"/>
      <line x1="{x_AB}" y1="{t_B}" x2="{x_tick_end}" y2="{t_B}" stroke="#333333" stroke-width="1.1"/>
      <line x1="{x_AB}" y1="{t_C}" x2="{x_tick_end}" y2="{t_C}" stroke="#333333" stroke-width="1.1"/>
    </g>

    <g id="{p}top-wedge-C">
      <polygon points="{x_Ab},{t_Ab} {x_aB},{t_aB} {x_AB},{t_C}" fill="{PAL['clash']['plane']}" fill-opacity="{PAL['clash']['op']}"/>
      <path d="{edge_path((x_Ab, t_Ab), (x_AB, t_C), 0.68)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
      <path d="{edge_path((x_aB, t_aB), (x_AB, t_C), 0.44)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
    </g>

    <g id="{p}top-wedge-B">
      <polygon points="{x_Ab},{t_Ab} {x_aB},{t_aB} {x_AB},{t_B}" fill="{PAL['wneg']['plane']}" fill-opacity="{PAL['wneg']['op']}"/>
      <path d="{edge_path((x_Ab, t_Ab), (x_AB, t_B), 0.50)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
      <path d="{edge_path((x_aB, t_aB), (x_AB, t_B), 0.62)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
    </g>

    <g id="{p}top-wedge-A">
      <polygon points="{x_Ab},{t_Ab} {x_aB},{t_aB} {x_AB},{t_A}" fill="{PAL['neut']['plane']}" fill-opacity="{PAL['neut']['op']}"/>
      <path d="{edge_path((x_Ab, t_Ab), (x_AB, t_A), 0.50)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
      <path d="{edge_path((x_aB, t_aB), (x_AB, t_A), 0.64)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
    </g>

    <g stroke="#333333" stroke-width="2" stroke-linecap="round" stroke-dasharray="0.1 7">
      <line x1="{x_wt}" y1="{t_wt}" x2="{x_wt}" y2="{t_floor_front}"/>
      <line x1="{x_Ab}" y1="{t_Ab}" x2="{x_Ab}" y2="{t_floor_back}"/>
      <line x1="{x_aB}" y1="{t_aB}" x2="{x_aB}" y2="{t_floor_front}"/>
      <line x1="{x_AB}" y1="{t_A}" x2="{x_AB}" y2="{t_floor_back}"/>
    </g>

    <g>
      <path d="{edge_path((x_wt, t_wt), (x_Ab, t_Ab), 0.50)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
      <path d="{edge_path((x_wt, t_wt), (x_aB, t_aB), 0.46)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
    </g>

    <g>
      <circle cx="{x_wt}" cy="{t_wt}" r="6.5" fill="#FFFFFF" stroke="#333333" stroke-width="1.3"/>
      <circle cx="{x_AB}" cy="{t_wt}" r="6.5" fill="#FFFFFF" stroke="#333333" stroke-width="1.3"/>
      <circle cx="{x_Ab}" cy="{t_Ab}" r="10" fill="{COL_SINGLE_POS}" stroke="#333333" stroke-width="1.5"/>
      <circle cx="{x_aB}" cy="{t_aB}" r="10" fill="{COL_SINGLE_NEG}" stroke="#333333" stroke-width="1.5"/>

      <circle cx="{x_AB}" cy="{t_A}" r="10" fill="{PAL['neut']['node']}" stroke="#333333" stroke-width="1.5"/>
      <text x="{x_AB}" y="{fmt(t_A + 4.5)}" fill="{PAL['neut']['txt']}" font-size="13" font-weight="700" text-anchor="middle">A</text>

      <circle cx="{x_AB}" cy="{t_B}" r="10" fill="{PAL['wneg']['node']}" stroke="#333333" stroke-width="1.5"/>
      <text x="{x_AB}" y="{fmt(t_B + 4.5)}" fill="{PAL['wneg']['txt']}" font-size="13" font-weight="700" text-anchor="middle">B</text>

      <circle cx="{x_AB}" cy="{t_C}" r="10" fill="{PAL['clash']['node']}" stroke="#333333" stroke-width="1.5"/>
      <text x="{x_AB}" y="{fmt(t_C + 4.5)}" fill="{PAL['clash']['txt']}" font-size="13" font-weight="700" text-anchor="middle">C</text>
    </g>

    <!-- Crimson −2u (433), Salmon −1u (441.5), Dark wt->A (450) -->
    <g stroke-width="1.4">
      <line x1="433" y1="{fmt(t_A + 3.5)}" x2="433" y2="{fmt(t_C - 4.5)}" stroke="{PAL['clash']['ui']}" marker-end="url(#{p}arr-clash)"/>
      <line x1="441.5" y1="{fmt(t_A + 3.5)}" x2="441.5" y2="{fmt(t_B - 4.5)}" stroke="{PAL['wneg']['ui']}" marker-end="url(#{p}arr-wneg)"/>
      <line x1="450" y1="{fmt(t_wt - 3.5)}" x2="450" y2="{fmt(t_A + 4.5)}" stroke="#333333" marker-end="url(#{p}arr-dark)"/>
    </g>

    <g>
      <text x="{x_wt}" y="{t_wt - 28}" fill="#333333" text-anchor="middle">ΔG ab</text>
      <text x="{x_wt}" y="{t_wt - 12}" fill="#333333" text-anchor="middle">(wild-type)</text>

      <text x="{x_Ab}" y="{t_Ab - 16}" fill="#333333" text-anchor="middle">ΔG Ab</text>
      <text x="230" y="{t_aB + 25}" fill="#333333" text-anchor="middle">ΔG aB</text>

      {arrow_label_svg((x_wt, t_wt), (x_Ab, t_Ab), "ΔΔG(A|wt)", t=0.55, side="above")}
      {arrow_label_svg((x_Ab, t_Ab), (x_AB, t_A),  "ΔΔG(B|A)",  t=0.52, side="above")}
      {arrow_label_svg((x_wt, t_wt), (x_aB, t_aB), "ΔΔG(B|wt)", t=0.45, side="below")}
      {arrow_label_svg((x_aB, t_aB), (x_AB, t_C),  "ΔΔG(A|B)",  t=0.52, side="below")}

      <text x="486" y="{t_A + 14}" fill="#333333" text-anchor="middle">ΔΔG</text>
      <text x="486" y="{t_A + 29}" fill="#333333" text-anchor="middle">additive</text>
      <text x="447" y="{fmt((t_wt + t_B) / 2 + 5)}" fill="{PAL['wneg']['ui']}" text-anchor="start">δΔΔG</text>
      <text x="439" y="{fmt((t_B + t_C) / 2 + 5)}" fill="{PAL['clash']['ui']}" text-anchor="start">δΔΔG</text>
    </g>
  </g>

  <!-- BOTTOM PANEL: Net destabilizing singles sum (D=compensation, E=weak positive, F=near-neutral) -->
  <g id="{p}panel-bottom" font-family="'Liberation Sans', Arial, Helvetica, 'DejaVu Sans', sans-serif" font-size="14">
    <polygon points="{x_wt},{b_floor_front} {x_Ab},{b_floor_back} {x_AB},{b_floor_back} {x_aB},{b_floor_front}" fill="#333333" fill-opacity="0.5"/>

    <g>
      <line x1="{fmt(x_wt + 7.5)}" y1="{b_wt}" x2="{fmt(x_AB - 7.5)}" y2="{b_wt}" stroke="#CCCCCC" stroke-width="1.1" marker-start="url(#{p}wt-start)" marker-end="url(#{p}wt-end)"/>
      <line x1="{x_AB}" y1="{b_D}" x2="{x_tick_end}" y2="{b_D}" stroke="#333333" stroke-width="1.1"/>
      <line x1="{x_AB}" y1="{b_wt}" x2="{x_tick_end}" y2="{b_wt}" stroke="#333333" stroke-width="1.1"/>
      <line x1="{x_AB}" y1="{b_E}" x2="{x_tick_end}" y2="{b_E}" stroke="#333333" stroke-width="1.1"/>
      <line x1="{x_AB}" y1="{b_F}" x2="{x_tick_end}" y2="{b_F}" stroke="#333333" stroke-width="1.1"/>
    </g>

    <g id="{p}bottom-wedge-F">
      <polygon points="{x_Ab},{b_Ab} {x_aB},{b_aB} {x_AB},{b_F}" fill="{PAL['neut']['plane']}" fill-opacity="{PAL['neut']['op']}"/>
      <path d="{edge_path((x_Ab, b_Ab), (x_AB, b_F), 0.68)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
      <path d="{edge_path((x_aB, b_aB), (x_AB, b_F), 0.40)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
    </g>

    <g id="{p}bottom-wedge-E">
      <polygon points="{x_Ab},{b_Ab} {x_aB},{b_aB} {x_AB},{b_E}" fill="{PAL['wpos']['plane']}" fill-opacity="{PAL['wpos']['op']}"/>
      <path d="{edge_path((x_Ab, b_Ab), (x_AB, b_E), 0.50)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
      <path d="{edge_path((x_aB, b_aB), (x_AB, b_E), 0.64)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
    </g>

    <g id="{p}bottom-wedge-D">
      <polygon points="{x_Ab},{b_Ab} {x_aB},{b_aB} {x_AB},{b_D}" fill="{PAL['comp']['plane']}" fill-opacity="{PAL['comp']['op']}"/>
      <path d="{edge_path((x_Ab, b_Ab), (x_AB, b_D), 0.50)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
      <path d="{edge_path((x_aB, b_aB), (x_AB, b_D), 0.66)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
    </g>

    <g stroke="#333333" stroke-width="2" stroke-linecap="round" stroke-dasharray="0.1 7">
      <line x1="{x_wt}" y1="{b_wt}" x2="{x_wt}" y2="{b_floor_front}"/>
      <line x1="{x_Ab}" y1="{b_Ab}" x2="{x_Ab}" y2="{b_floor_back}"/>
      <line x1="{x_aB}" y1="{b_aB}" x2="{x_aB}" y2="{b_floor_front}"/>
      <line x1="{x_AB}" y1="{b_D}" x2="{x_AB}" y2="{b_floor_back}"/>
    </g>

    <g>
      <path d="{edge_path((x_wt, b_wt), (x_Ab, b_Ab), 0.44)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
      <path d="{edge_path((x_wt, b_wt), (x_aB, b_aB), 0.58)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
    </g>

    <g>
      <circle cx="{x_wt}" cy="{b_wt}" r="6.5" fill="#FFFFFF" stroke="#333333" stroke-width="1.3"/>
      <circle cx="{x_AB}" cy="{b_wt}" r="6.5" fill="#FFFFFF" stroke="#333333" stroke-width="1.3"/>
      <circle cx="{x_Ab}" cy="{b_Ab}" r="10" fill="{COL_SINGLE_NEG}" stroke="#333333" stroke-width="1.5"/>
      <circle cx="{x_aB}" cy="{b_aB}" r="10" fill="{COL_SINGLE_NEG}" stroke="#333333" stroke-width="1.5"/>

      <circle cx="{x_AB}" cy="{b_D}" r="10" fill="{PAL['comp']['node']}" stroke="#333333" stroke-width="1.5"/>
      <text x="{x_AB}" y="{fmt(b_D + 4.5)}" fill="{PAL['comp']['txt']}" font-size="13" font-weight="700" text-anchor="middle">D</text>

      <circle cx="{x_AB}" cy="{b_E}" r="10" fill="{PAL['wpos']['node']}" stroke="#333333" stroke-width="1.5"/>
      <text x="{x_AB}" y="{fmt(b_E + 4.5)}" fill="{PAL['wpos']['txt']}" font-size="13" font-weight="700" text-anchor="middle">E</text>

      <circle cx="{x_AB}" cy="{b_F}" r="10" fill="{PAL['neut']['node']}" stroke="#333333" stroke-width="1.5"/>
      <text x="{x_AB}" y="{fmt(b_F + 4.5)}" fill="{PAL['neut']['txt']}" font-size="13" font-weight="700" text-anchor="middle">F</text>
    </g>

    <g stroke-width="1.4">
      <line x1="433" y1="{fmt(b_F - 3.5)}" x2="433" y2="{fmt(b_D + 4.5)}" stroke="{PAL['comp']['ui']}" marker-end="url(#{p}arr-comp)"/>
      <line x1="441.5" y1="{fmt(b_wt + 3.5)}" x2="441.5" y2="{fmt(b_F - 4.5)}" stroke="#333333" marker-end="url(#{p}arr-dark)"/>
      <line x1="450" y1="{fmt(b_F - 3.5)}" x2="450" y2="{fmt(b_E + 4.5)}" stroke="{PAL['wpos']['ui']}" marker-end="url(#{p}arr-wpos)"/>
    </g>

    <g>
      <text x="439" y="{fmt((b_D + b_wt) / 2 + 5)}" fill="{PAL['comp']['ui']}" text-anchor="start">δΔΔG</text>
      <text x="486" y="{b_wt + 14}" fill="#333333" text-anchor="middle">ΔΔG</text>
      <text x="486" y="{b_wt + 29}" fill="#333333" text-anchor="middle">additive</text>
      <text x="455" y="{fmt((b_E + b_F) / 2 + 5)}" fill="{PAL['wpos']['ui']}" text-anchor="start">δΔΔG</text>
    </g>
  </g>
</svg>
'''

def make_draft2_7plane(prefix=""):
    p = f"{prefix}-" if prefix else ""
    W, H = 536, 518
    x_wt, x_Ab, x_aB, x_AB = 40, 204, 256, 420
    x_tick_end = 520
    u = 60  # 1u = 60px, 2u = 120px

    t_wt = 88
    t_Ab = t_wt - u // 2      # 58
    t_A  = t_wt - u // 2      # 58  (weak positive: +1u)
    t_B  = t_wt + u // 2      # 118 (near-neutral / additive: 0)
    t_aB = t_wt + u           # 148
    t_C  = t_wt + 3 * u // 2  # 178 (weak negative: -1u)
    t_D  = t_wt + 5 * u // 2  # 238 (clash / negative sign: -2u)
    t_floor_back, t_floor_front = 264, 289.5

    b_wt = 336
    b_E  = b_wt - u // 2      # 306 (compensation / positive sign: +2u)
    b_Ab = b_wt + u // 2      # 366
    b_F  = b_wt + u // 2      # 366 (weak positive: +1u)
    b_aB = b_wt + u           # 396
    b_G  = b_wt + 3 * u // 2  # 426 (near-neutral / additive: 0)
    b_floor_back, b_floor_front = 464, 489.5

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
  <title>Thermodynamic Epistasis Cycles — 5 Main Types (Draft 2: 7-Plane Equal-Step Ladder)</title>

{make_defs(prefix)}

  <!-- TOP PANEL: 4 equally-spaced outputs (+1u, 0, −1u, −2u) -->
  <g id="{p}panel-top" font-family="'Liberation Sans', Arial, Helvetica, 'DejaVu Sans', sans-serif" font-size="14">
    <polygon points="{x_wt},{t_floor_front} {x_Ab},{t_floor_back} {x_AB},{t_floor_back} {x_aB},{t_floor_front}" fill="#333333" fill-opacity="0.5"/>

    <g>
      <line x1="{fmt(x_wt + 7.5)}" y1="{t_wt}" x2="{fmt(x_AB - 7.5)}" y2="{t_wt}" stroke="#CCCCCC" stroke-width="1.1" marker-start="url(#{p}wt-start)" marker-end="url(#{p}wt-end)"/>
      <line x1="{x_AB}" y1="{t_A}" x2="{x_tick_end}" y2="{t_A}" stroke="#333333" stroke-width="1.1"/>
      <line x1="{x_AB}" y1="{t_wt}" x2="{x_tick_end}" y2="{t_wt}" stroke="#333333" stroke-width="1.1"/>
      <line x1="{x_AB}" y1="{t_B}" x2="{x_tick_end}" y2="{t_B}" stroke="#333333" stroke-width="1.1"/>
      <line x1="{x_AB}" y1="{t_C}" x2="{x_tick_end}" y2="{t_C}" stroke="#333333" stroke-width="1.1"/>
      <line x1="{x_AB}" y1="{t_D}" x2="{x_tick_end}" y2="{t_D}" stroke="#333333" stroke-width="1.1"/>
    </g>

    <g id="{p}top-wedge-D">
      <polygon points="{x_Ab},{t_Ab} {x_aB},{t_aB} {x_AB},{t_D}" fill="{PAL['clash']['plane']}" fill-opacity="{PAL['clash']['op']}"/>
      <path d="{edge_path((x_Ab, t_Ab), (x_AB, t_D), 0.72)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
      <path d="{edge_path((x_aB, t_aB), (x_AB, t_D), 0.46)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
    </g>

    <g id="{p}top-wedge-C">
      <polygon points="{x_Ab},{t_Ab} {x_aB},{t_aB} {x_AB},{t_C}" fill="{PAL['wneg']['plane']}" fill-opacity="{PAL['wneg']['op']}"/>
      <path d="{edge_path((x_Ab, t_Ab), (x_AB, t_C), 0.45)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
      <path d="{edge_path((x_aB, t_aB), (x_AB, t_C), 0.62)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
    </g>

    <g id="{p}top-wedge-B">
      <polygon points="{x_Ab},{t_Ab} {x_aB},{t_aB} {x_AB},{t_B}" fill="{PAL['neut']['plane']}" fill-opacity="{PAL['neut']['op']}"/>
      <path d="{edge_path((x_Ab, t_Ab), (x_AB, t_B), 0.48)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
      <path d="{edge_path((x_aB, t_aB), (x_AB, t_B), 0.70)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
    </g>

    <g id="{p}top-wedge-A">
      <polygon points="{x_Ab},{t_Ab} {x_aB},{t_aB} {x_AB},{t_A}" fill="{PAL['wpos']['plane']}" fill-opacity="{PAL['wpos']['op']}"/>
      <path d="{edge_path((x_Ab, t_Ab), (x_AB, t_A), 0.50)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
      <path d="{edge_path((x_aB, t_aB), (x_AB, t_A), 0.66)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
    </g>

    <g stroke="#333333" stroke-width="2" stroke-linecap="round" stroke-dasharray="0.1 7">
      <line x1="{x_wt}" y1="{t_wt}" x2="{x_wt}" y2="{t_floor_front}"/>
      <line x1="{x_Ab}" y1="{t_Ab}" x2="{x_Ab}" y2="{t_floor_back}"/>
      <line x1="{x_aB}" y1="{t_aB}" x2="{x_aB}" y2="{t_floor_front}"/>
      <line x1="{x_AB}" y1="{t_A}" x2="{x_AB}" y2="{t_floor_back}"/>
    </g>

    <g>
      <path d="{edge_path((x_wt, t_wt), (x_Ab, t_Ab), 0.50)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
      <path d="{edge_path((x_wt, t_wt), (x_aB, t_aB), 0.50)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
    </g>

    <g>
      <circle cx="{x_wt}" cy="{t_wt}" r="6.5" fill="#FFFFFF" stroke="#333333" stroke-width="1.3"/>
      <circle cx="{x_AB}" cy="{t_wt}" r="6.5" fill="#FFFFFF" stroke="#333333" stroke-width="1.3"/>
      <circle cx="{x_Ab}" cy="{t_Ab}" r="10" fill="{COL_SINGLE_POS}" stroke="#333333" stroke-width="1.5"/>
      <circle cx="{x_aB}" cy="{t_aB}" r="10" fill="{COL_SINGLE_NEG}" stroke="#333333" stroke-width="1.5"/>

      <circle cx="{x_AB}" cy="{t_A}" r="10" fill="{PAL['wpos']['node']}" stroke="#333333" stroke-width="1.5"/>
      <text x="{x_AB}" y="{fmt(t_A + 4.5)}" fill="{PAL['wpos']['txt']}" font-size="13" font-weight="700" text-anchor="middle">A</text>

      <circle cx="{x_AB}" cy="{t_B}" r="10" fill="{PAL['neut']['node']}" stroke="#333333" stroke-width="1.5"/>
      <text x="{x_AB}" y="{fmt(t_B + 4.5)}" fill="{PAL['neut']['txt']}" font-size="13" font-weight="700" text-anchor="middle">B</text>

      <circle cx="{x_AB}" cy="{t_C}" r="10" fill="{PAL['wneg']['node']}" stroke="#333333" stroke-width="1.5"/>
      <text x="{x_AB}" y="{fmt(t_C + 4.5)}" fill="{PAL['wneg']['txt']}" font-size="13" font-weight="700" text-anchor="middle">C</text>

      <circle cx="{x_AB}" cy="{t_D}" r="10" fill="{PAL['clash']['node']}" stroke="#333333" stroke-width="1.5"/>
      <text x="{x_AB}" y="{fmt(t_D + 4.5)}" fill="{PAL['clash']['txt']}" font-size="13" font-weight="700" text-anchor="middle">D</text>
    </g>

    <g stroke-width="1.4">
      <line x1="433" y1="{fmt(t_B - 3.5)}" x2="433" y2="{fmt(t_A + 4.5)}" stroke="{PAL['wpos']['ui']}" marker-end="url(#{p}arr-wpos)"/>
      <line x1="442" y1="{fmt(t_wt + 3.5)}" x2="442" y2="{fmt(t_B - 4.5)}" stroke="#333333" marker-end="url(#{p}arr-dark)"/>
      <line x1="442" y1="{fmt(t_B + 3.5)}" x2="442" y2="{fmt(t_C - 4.5)}" stroke="{PAL['wneg']['ui']}" marker-end="url(#{p}arr-wneg)"/>
      <line x1="433" y1="{fmt(t_B + 3.5)}" x2="433" y2="{fmt(t_D - 4.5)}" stroke="{PAL['clash']['ui']}" marker-end="url(#{p}arr-clash)"/>
    </g>

    <g>
      <text x="{x_wt}" y="{t_wt - 28}" fill="#333333" text-anchor="middle">ΔG ab</text>
      <text x="{x_wt}" y="{t_wt - 12}" fill="#333333" text-anchor="middle">(wild-type)</text>

      <text x="{x_Ab}" y="{t_Ab - 16}" fill="#333333" text-anchor="middle">ΔG Ab</text>
      <text x="230" y="{t_aB + 25}" fill="#333333" text-anchor="middle">ΔG aB</text>

      {arrow_label_svg((x_wt, t_wt), (x_Ab, t_Ab), "ΔΔG(A|wt)", t=0.56, side="above")}
      {arrow_label_svg((x_Ab, t_Ab), (x_AB, t_A),  "ΔΔG(B|A)",  t=0.50, side="above")}
      {arrow_label_svg((x_wt, t_wt), (x_aB, t_aB), "ΔΔG(B|wt)", t=0.45, side="below")}
      {arrow_label_svg((x_aB, t_aB), (x_AB, t_D),  "ΔΔG(A|B)",  t=0.52, side="below")}

      <text x="439" y="{fmt((t_A + t_wt) / 2 + 5)}" fill="{PAL['wpos']['ui']}" text-anchor="start">δΔΔG</text>
      <text x="482" y="{t_wt + 13}" font-size="13" fill="#333333" text-anchor="middle">ΔΔG</text>
      <text x="482" y="{t_wt + 26}" font-size="13" fill="#333333" text-anchor="middle">additive</text>
      <text x="448" y="{fmt((t_B + t_C) / 2 + 5)}" fill="{PAL['wneg']['ui']}" text-anchor="start">δΔΔG</text>
      <text x="439" y="{fmt((t_C + t_D) / 2 + 5)}" fill="{PAL['clash']['ui']}" text-anchor="start">δΔΔG</text>
    </g>
  </g>

  <!-- BOTTOM PANEL: 3 equally-spaced outputs (+2u, +1u, 0) -->
  <g id="{p}panel-bottom" font-family="'Liberation Sans', Arial, Helvetica, 'DejaVu Sans', sans-serif" font-size="14">
    <polygon points="{x_wt},{b_floor_front} {x_Ab},{b_floor_back} {x_AB},{b_floor_back} {x_aB},{b_floor_front}" fill="#333333" fill-opacity="0.5"/>

    <g>
      <line x1="{fmt(x_wt + 7.5)}" y1="{b_wt}" x2="{fmt(x_AB - 7.5)}" y2="{b_wt}" stroke="#CCCCCC" stroke-width="1.1" marker-start="url(#{p}wt-start)" marker-end="url(#{p}wt-end)"/>
      <line x1="{x_AB}" y1="{b_E}" x2="{x_tick_end}" y2="{b_E}" stroke="#333333" stroke-width="1.1"/>
      <line x1="{x_AB}" y1="{b_wt}" x2="{x_tick_end}" y2="{b_wt}" stroke="#333333" stroke-width="1.1"/>
      <line x1="{x_AB}" y1="{b_F}" x2="{x_tick_end}" y2="{b_F}" stroke="#333333" stroke-width="1.1"/>
      <line x1="{x_AB}" y1="{b_G}" x2="{x_tick_end}" y2="{b_G}" stroke="#333333" stroke-width="1.1"/>
    </g>

    <g id="{p}bottom-wedge-G">
      <polygon points="{x_Ab},{b_Ab} {x_aB},{b_aB} {x_AB},{b_G}" fill="{PAL['neut']['plane']}" fill-opacity="{PAL['neut']['op']}"/>
      <path d="{edge_path((x_Ab, b_Ab), (x_AB, b_G), 0.68)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
      <path d="{edge_path((x_aB, b_aB), (x_AB, b_G), 0.40)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
    </g>

    <g id="{p}bottom-wedge-F">
      <polygon points="{x_Ab},{b_Ab} {x_aB},{b_aB} {x_AB},{b_F}" fill="{PAL['wpos']['plane']}" fill-opacity="{PAL['wpos']['op']}"/>
      <path d="{edge_path((x_Ab, b_Ab), (x_AB, b_F), 0.50)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
      <path d="{edge_path((x_aB, b_aB), (x_AB, b_F), 0.64)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
    </g>

    <g id="{p}bottom-wedge-E">
      <polygon points="{x_Ab},{b_Ab} {x_aB},{b_aB} {x_AB},{b_E}" fill="{PAL['comp']['plane']}" fill-opacity="{PAL['comp']['op']}"/>
      <path d="{edge_path((x_Ab, b_Ab), (x_AB, b_E), 0.50)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
      <path d="{edge_path((x_aB, b_aB), (x_AB, b_E), 0.66)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
    </g>

    <g stroke="#333333" stroke-width="2" stroke-linecap="round" stroke-dasharray="0.1 7">
      <line x1="{x_wt}" y1="{b_wt}" x2="{x_wt}" y2="{b_floor_front}"/>
      <line x1="{x_Ab}" y1="{b_Ab}" x2="{x_Ab}" y2="{b_floor_back}"/>
      <line x1="{x_aB}" y1="{b_aB}" x2="{x_aB}" y2="{b_floor_front}"/>
      <line x1="{x_AB}" y1="{b_E}" x2="{x_AB}" y2="{b_floor_back}"/>
    </g>

    <g>
      <path d="{edge_path((x_wt, b_wt), (x_Ab, b_Ab), 0.44)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
      <path d="{edge_path((x_wt, b_wt), (x_aB, b_aB), 0.58)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
    </g>

    <g>
      <circle cx="{x_wt}" cy="{b_wt}" r="6.5" fill="#FFFFFF" stroke="#333333" stroke-width="1.3"/>
      <circle cx="{x_AB}" cy="{b_wt}" r="6.5" fill="#FFFFFF" stroke="#333333" stroke-width="1.3"/>
      <circle cx="{x_Ab}" cy="{b_Ab}" r="10" fill="{COL_SINGLE_NEG}" stroke="#333333" stroke-width="1.5"/>
      <circle cx="{x_aB}" cy="{b_aB}" r="10" fill="{COL_SINGLE_NEG}" stroke="#333333" stroke-width="1.5"/>

      <circle cx="{x_AB}" cy="{b_E}" r="10" fill="{PAL['comp']['node']}" stroke="#333333" stroke-width="1.5"/>
      <text x="{x_AB}" y="{fmt(b_E + 4.5)}" fill="{PAL['comp']['txt']}" font-size="13" font-weight="700" text-anchor="middle">E</text>

      <circle cx="{x_AB}" cy="{b_F}" r="10" fill="{PAL['wpos']['node']}" stroke="#333333" stroke-width="1.5"/>
      <text x="{x_AB}" y="{fmt(b_F + 4.5)}" fill="{PAL['wpos']['txt']}" font-size="13" font-weight="700" text-anchor="middle">F</text>

      <circle cx="{x_AB}" cy="{b_G}" r="10" fill="{PAL['neut']['node']}" stroke="#333333" stroke-width="1.5"/>
      <text x="{x_AB}" y="{fmt(b_G + 4.5)}" fill="{PAL['neut']['txt']}" font-size="13" font-weight="700" text-anchor="middle">G</text>
    </g>

    <g stroke-width="1.4">
      <line x1="433" y1="{fmt(b_G - 3.5)}" x2="433" y2="{fmt(b_E + 4.5)}" stroke="{PAL['comp']['ui']}" marker-end="url(#{p}arr-comp)"/>
      <line x1="441.5" y1="{fmt(b_wt + 3.5)}" x2="441.5" y2="{fmt(b_G - 4.5)}" stroke="#333333" marker-end="url(#{p}arr-dark)"/>
      <line x1="450" y1="{fmt(b_G - 3.5)}" x2="450" y2="{fmt(b_F + 4.5)}" stroke="{PAL['wpos']['ui']}" marker-end="url(#{p}arr-wpos)"/>
    </g>

    <g>
      <text x="439" y="{fmt((b_E + b_wt) / 2 + 5)}" fill="{PAL['comp']['ui']}" text-anchor="start">δΔΔG</text>
      <text x="483" y="{b_wt + 13}" font-size="13" fill="#333333" text-anchor="middle">ΔΔG</text>
      <text x="483" y="{b_wt + 26}" font-size="13" fill="#333333" text-anchor="middle">additive</text>
      <text x="455" y="{fmt((b_F + b_G) / 2 + 5)}" fill="{PAL['wpos']['ui']}" text-anchor="start">δΔΔG</text>
    </g>
  </g>
</svg>
'''

def make_draft3_split(prefix=""):
    """Draft 3: 6-plane (3 Top + 3 Bottom) Balanced Split around additive B (Top) and E (Bottom),
    both panels with ΔΔG_additive < 0 and identical 3u = 156px right-side output range (u = 52px):
      - Top: A = weak positive (+1u = +52px), B = near-neutral/additive (0), C = clash/negative sign (-2u = -104px)
      - Bottom: D = compensation/positive sign (+2u = +104px), E = near-neutral/additive (0), F = weak negative (-1u = -52px)
    """
    p = f"{prefix}-" if prefix else ""
    W, H = 536, 510
    x_wt, x_Ab, x_aB, x_AB = 40, 204, 256, 420
    x_tick_end = 520
    u = 52  # 1u = 52px, 2u = 104px, total range = 3u = 156px in both panels

    # Top panel: Ab = -26 (56), wt = 82, B = +26 (108, additive: -26 + 52 = +26), aB = +52 (134), C = 212 (-2u below B)
    t_wt = 82
    t_Ab = 56   # -26
    t_A  = 56   # +1u = +52px above B=108 (level with Ab, wt halfway between A and B)
    t_B  = 108  # additive: 0
    t_aB = 134  # +52
    t_C  = 212  # -2u = -104px below B=108
    t_floor_back, t_floor_front = 240, 265.5

    # Bottom panel: D = 278 (+2u = +104px above E=382), wt = 308, Ab = 319 (+11), aB = 371 (+63) -> E = 382 (additive: +74), F = 434 (-1u = -52px below E)
    b_D  = 278
    b_wt = 308
    b_Ab = 319  # +11
    b_aB = 371  # +63
    b_E  = 382  # additive: +11 + 63 = +74
    b_F  = 434  # -1u = -52px below E=382
    b_floor_back, b_floor_front = 460, 485.5

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
  <title>Thermodynamic Epistasis Cycles — 5 Main Types (Draft 3: 6-Plane Balanced Split around B &amp; E)</title>

{make_defs(prefix)}

  <!-- TOP PANEL: A=weak positive (+1u), B=near-neutral (0), C=clash/negative sign (−2u) -->
  <g id="{p}panel-top" font-family="'Liberation Sans', Arial, Helvetica, 'DejaVu Sans', sans-serif" font-size="14">
    <polygon points="{x_wt},{t_floor_front} {x_Ab},{t_floor_back} {x_AB},{t_floor_back} {x_aB},{t_floor_front}" fill="#333333" fill-opacity="0.5"/>

    <g>
      <line x1="{fmt(x_wt + 7.5)}" y1="{t_wt}" x2="{fmt(x_AB - 7.5)}" y2="{t_wt}" stroke="#CCCCCC" stroke-width="1.1" marker-start="url(#{p}wt-start)" marker-end="url(#{p}wt-end)"/>
      <line x1="{x_AB}" y1="{t_A}" x2="{x_tick_end}" y2="{t_A}" stroke="#333333" stroke-width="1.1"/>
      <line x1="{x_AB}" y1="{t_wt}" x2="{x_tick_end}" y2="{t_wt}" stroke="#333333" stroke-width="1.1"/>
      <line x1="{x_AB}" y1="{t_B}" x2="{x_tick_end}" y2="{t_B}" stroke="#333333" stroke-width="1.1"/>
      <line x1="{x_AB}" y1="{t_C}" x2="{x_tick_end}" y2="{t_C}" stroke="#333333" stroke-width="1.1"/>
    </g>

    <g id="{p}top-wedge-C">
      <polygon points="{x_Ab},{t_Ab} {x_aB},{t_aB} {x_AB},{t_C}" fill="{PAL['clash']['plane']}" fill-opacity="{PAL['clash']['op']}"/>
      <path d="{edge_path((x_Ab, t_Ab), (x_AB, t_C), 0.68)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
      <path d="{edge_path((x_aB, t_aB), (x_AB, t_C), 0.45)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
    </g>

    <g id="{p}top-wedge-B">
      <polygon points="{x_Ab},{t_Ab} {x_aB},{t_aB} {x_AB},{t_B}" fill="{PAL['neut']['plane']}" fill-opacity="{PAL['neut']['op']}"/>
      <path d="{edge_path((x_Ab, t_Ab), (x_AB, t_B), 0.50)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
      <path d="{edge_path((x_aB, t_aB), (x_AB, t_B), 0.65)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
    </g>

    <g id="{p}top-wedge-A">
      <polygon points="{x_Ab},{t_Ab} {x_aB},{t_aB} {x_AB},{t_A}" fill="{PAL['wpos']['plane']}" fill-opacity="{PAL['wpos']['op']}"/>
      <path d="{edge_path((x_Ab, t_Ab), (x_AB, t_A), 0.50)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
      <path d="{edge_path((x_aB, t_aB), (x_AB, t_A), 0.65)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
    </g>

    <g stroke="#333333" stroke-width="2" stroke-linecap="round" stroke-dasharray="0.1 7">
      <line x1="{x_wt}" y1="{t_wt}" x2="{x_wt}" y2="{t_floor_front}"/>
      <line x1="{x_Ab}" y1="{t_Ab}" x2="{x_Ab}" y2="{t_floor_back}"/>
      <line x1="{x_aB}" y1="{t_aB}" x2="{x_aB}" y2="{t_floor_front}"/>
      <line x1="{x_AB}" y1="{t_A}" x2="{x_AB}" y2="{t_floor_back}"/>
    </g>

    <g>
      <path d="{edge_path((x_wt, t_wt), (x_Ab, t_Ab), 0.50)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
      <path d="{edge_path((x_wt, t_wt), (x_aB, t_aB), 0.50)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
    </g>

    <g>
      <circle cx="{x_wt}" cy="{t_wt}" r="6.5" fill="#FFFFFF" stroke="#333333" stroke-width="1.3"/>
      <circle cx="{x_AB}" cy="{t_wt}" r="6.5" fill="#FFFFFF" stroke="#333333" stroke-width="1.3"/>
      <circle cx="{x_Ab}" cy="{t_Ab}" r="10" fill="{COL_SINGLE_POS}" stroke="#333333" stroke-width="1.5"/>
      <circle cx="{x_aB}" cy="{t_aB}" r="10" fill="{COL_SINGLE_NEG}" stroke="#333333" stroke-width="1.5"/>

      <circle cx="{x_AB}" cy="{t_A}" r="10" fill="{PAL['wpos']['node']}" stroke="#333333" stroke-width="1.5"/>
      <text x="{x_AB}" y="{fmt(t_A + 4.5)}" fill="{PAL['wpos']['txt']}" font-size="13" font-weight="700" text-anchor="middle">A</text>

      <circle cx="{x_AB}" cy="{t_B}" r="10" fill="{PAL['neut']['node']}" stroke="#333333" stroke-width="1.5"/>
      <text x="{x_AB}" y="{fmt(t_B + 4.5)}" fill="{PAL['neut']['txt']}" font-size="13" font-weight="700" text-anchor="middle">B</text>

      <circle cx="{x_AB}" cy="{t_C}" r="10" fill="{PAL['clash']['node']}" stroke="#333333" stroke-width="1.5"/>
      <text x="{x_AB}" y="{fmt(t_C + 4.5)}" fill="{PAL['clash']['txt']}" font-size="13" font-weight="700" text-anchor="middle">C</text>
    </g>

    <g stroke-width="1.4">
      <line x1="434" y1="{fmt(t_B - 3.5)}" x2="434" y2="{fmt(t_A + 4.5)}" stroke="{PAL['wpos']['ui']}" marker-end="url(#{p}arr-wpos)"/>
      <line x1="443" y1="{fmt(t_wt + 3.5)}" x2="443" y2="{fmt(t_B - 4.5)}" stroke="#333333" marker-end="url(#{p}arr-dark)"/>
      <line x1="434" y1="{fmt(t_B + 3.5)}" x2="434" y2="{fmt(t_C - 4.5)}" stroke="{PAL['clash']['ui']}" marker-end="url(#{p}arr-clash)"/>
    </g>

    <g>
      <text x="{x_wt}" y="{t_wt - 28}" fill="#333333" text-anchor="middle">ΔG ab</text>
      <text x="{x_wt}" y="{t_wt - 12}" fill="#333333" text-anchor="middle">(wild-type)</text>

      <text x="{x_Ab}" y="{t_Ab - 16}" fill="#333333" text-anchor="middle">ΔG Ab</text>
      <text x="230" y="{t_aB + 25}" fill="#333333" text-anchor="middle">ΔG aB</text>

      {arrow_label_svg((x_wt, t_wt), (x_Ab, t_Ab), "ΔΔG(A|wt)", t=0.56, side="above")}
      {arrow_label_svg((x_Ab, t_Ab), (x_AB, t_A),  "ΔΔG(B|A)",  t=0.50, side="above")}
      {arrow_label_svg((x_wt, t_wt), (x_aB, t_aB), "ΔΔG(B|wt)", t=0.45, side="below")}
      {arrow_label_svg((x_aB, t_aB), (x_AB, t_C),  "ΔΔG(A|B)",  t=0.52, side="below")}

      <text x="440" y="{fmt((t_A + t_wt) / 2 + 5)}" fill="{PAL['wpos']['ui']}" text-anchor="start">δΔΔG</text>
      <text x="483" y="{t_wt + 12}" font-size="12.5" fill="#333333" text-anchor="middle">ΔΔG</text>
      <text x="483" y="{t_wt + 24}" font-size="12.5" fill="#333333" text-anchor="middle">additive</text>
      <text x="440" y="{fmt((t_B + t_C) / 2 + 5)}" fill="{PAL['clash']['ui']}" text-anchor="start">δΔΔG</text>
    </g>
  </g>

  <!-- BOTTOM PANEL: D=compensation (+2u), E=near-neutral (0), F=weak negative (−1u) -->
  <g id="{p}panel-bottom" font-family="'Liberation Sans', Arial, Helvetica, 'DejaVu Sans', sans-serif" font-size="14">
    <polygon points="{x_wt},{b_floor_front} {x_Ab},{b_floor_back} {x_AB},{b_floor_back} {x_aB},{b_floor_front}" fill="#333333" fill-opacity="0.5"/>

    <g>
      <line x1="{fmt(x_wt + 7.5)}" y1="{b_wt}" x2="{fmt(x_AB - 7.5)}" y2="{b_wt}" stroke="#CCCCCC" stroke-width="1.1" marker-start="url(#{p}wt-start)" marker-end="url(#{p}wt-end)"/>
      <line x1="{x_AB}" y1="{b_D}" x2="{x_tick_end}" y2="{b_D}" stroke="#333333" stroke-width="1.1"/>
      <line x1="{x_AB}" y1="{b_wt}" x2="{x_tick_end}" y2="{b_wt}" stroke="#333333" stroke-width="1.1"/>
      <line x1="{x_AB}" y1="{b_E}" x2="{x_tick_end}" y2="{b_E}" stroke="#333333" stroke-width="1.1"/>
      <line x1="{x_AB}" y1="{b_F}" x2="{x_tick_end}" y2="{b_F}" stroke="#333333" stroke-width="1.1"/>
    </g>

    <g id="{p}bottom-wedge-F">
      <polygon points="{x_Ab},{b_Ab} {x_aB},{b_aB} {x_AB},{b_F}" fill="{PAL['wneg']['plane']}" fill-opacity="{PAL['wneg']['op']}"/>
      <path d="{edge_path((x_Ab, b_Ab), (x_AB, b_F), 0.74)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
      <path d="{edge_path((x_aB, b_aB), (x_AB, b_F), 0.44)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
    </g>

    <g id="{p}bottom-wedge-E">
      <polygon points="{x_Ab},{b_Ab} {x_aB},{b_aB} {x_AB},{b_E}" fill="{PAL['neut']['plane']}" fill-opacity="{PAL['neut']['op']}"/>
      <path d="{edge_path((x_Ab, b_Ab), (x_AB, b_E), 0.66)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
      <path d="{edge_path((x_aB, b_aB), (x_AB, b_E), 0.70)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
    </g>

    <g id="{p}bottom-wedge-D">
      <polygon points="{x_Ab},{b_Ab} {x_aB},{b_aB} {x_AB},{b_D}" fill="{PAL['comp']['plane']}" fill-opacity="{PAL['comp']['op']}"/>
      <path d="{edge_path((x_Ab, b_Ab), (x_AB, b_D), 0.46)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
      <path d="{edge_path((x_aB, b_aB), (x_AB, b_D), 0.62)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
    </g>

    <g stroke="#333333" stroke-width="2" stroke-linecap="round" stroke-dasharray="0.1 7">
      <line x1="{x_wt}" y1="{b_wt}" x2="{x_wt}" y2="{b_floor_front}"/>
      <line x1="{x_Ab}" y1="{b_Ab}" x2="{x_Ab}" y2="{b_floor_back}"/>
      <line x1="{x_aB}" y1="{b_aB}" x2="{x_aB}" y2="{b_floor_front}"/>
      <line x1="{x_AB}" y1="{b_D}" x2="{x_AB}" y2="{b_floor_back}"/>
    </g>

    <g>
      <path d="{edge_path((x_wt, b_wt), (x_Ab, b_Ab), 0.45)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
      <path d="{edge_path((x_wt, b_wt), (x_aB, b_aB), 0.58)}" fill="none" stroke="#1A1A1A" stroke-width="2.9" marker-mid="url(#{p}mid-arrow)"/>
    </g>

    <g>
      <circle cx="{x_wt}" cy="{b_wt}" r="6.5" fill="#FFFFFF" stroke="#333333" stroke-width="1.3"/>
      <circle cx="{x_AB}" cy="{b_wt}" r="6.5" fill="#FFFFFF" stroke="#333333" stroke-width="1.3"/>
      <circle cx="{x_Ab}" cy="{b_Ab}" r="10" fill="{COL_SINGLE_NEG}" stroke="#333333" stroke-width="1.5"/>
      <circle cx="{x_aB}" cy="{b_aB}" r="10" fill="{COL_SINGLE_NEG}" stroke="#333333" stroke-width="1.5"/>

      <circle cx="{x_AB}" cy="{b_D}" r="10" fill="{PAL['comp']['node']}" stroke="#333333" stroke-width="1.5"/>
      <text x="{x_AB}" y="{fmt(b_D + 4.5)}" fill="{PAL['comp']['txt']}" font-size="13" font-weight="700" text-anchor="middle">D</text>

      <circle cx="{x_AB}" cy="{b_E}" r="10" fill="{PAL['neut']['node']}" stroke="#333333" stroke-width="1.5"/>
      <text x="{x_AB}" y="{fmt(b_E + 4.5)}" fill="{PAL['neut']['txt']}" font-size="13" font-weight="700" text-anchor="middle">E</text>

      <circle cx="{x_AB}" cy="{b_F}" r="10" fill="{PAL['wneg']['node']}" stroke="#333333" stroke-width="1.5"/>
      <text x="{x_AB}" y="{fmt(b_F + 4.5)}" fill="{PAL['wneg']['txt']}" font-size="13" font-weight="700" text-anchor="middle">F</text>
    </g>

    <g stroke-width="1.4">
      <line x1="434" y1="{fmt(b_E - 3.5)}" x2="434" y2="{fmt(b_D + 4.5)}" stroke="{PAL['comp']['ui']}" marker-end="url(#{p}arr-comp)"/>
      <line x1="443" y1="{fmt(b_wt + 3.5)}" x2="443" y2="{fmt(b_E - 4.5)}" stroke="#333333" marker-end="url(#{p}arr-dark)"/>
      <line x1="434" y1="{fmt(b_E + 3.5)}" x2="434" y2="{fmt(b_F - 4.5)}" stroke="{PAL['wneg']['ui']}" marker-end="url(#{p}arr-wneg)"/>
    </g>

    <g>
      <text x="440" y="{fmt((b_D + b_wt) / 2 + 5)}" fill="{PAL['comp']['ui']}" text-anchor="start">δΔΔG</text>
      <text x="483" y="{b_wt + 33}" fill="#333333" text-anchor="middle">ΔΔG</text>
      <text x="483" y="{b_wt + 49}" fill="#333333" text-anchor="middle">additive</text>
      <text x="440" y="{fmt((b_E + b_F) / 2 + 5)}" fill="{PAL['wneg']['ui']}" text-anchor="start">δΔΔG</text>
    </g>
  </g>
</svg>
'''

def main():
    out_dir = Path(__file__).resolve().parent
    (out_dir / "epistasis_cycles.svg").write_text(make_draft1_mirror(), encoding="utf-8")
    (out_dir / "epistasis_cycles_7plane.svg").write_text(make_draft2_7plane(), encoding="utf-8")
    (out_dir / "epistasis_cycles_split.svg").write_text(make_draft3_split(), encoding="utf-8")
    print(f"Generated all 3 epistasis cycle SVG drafts in {out_dir}")


if __name__ == "__main__":
    main()

