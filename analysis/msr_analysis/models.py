"""Canonical model roster and immutable visual styles for all figures and tables.

Every benchmarked model has one canonical name, group, order, colour, marker, line style, and prediction-key
mapping across all figures and tables:

    zero-shot            ProSST, ProteinMPNN, ESM3-small-open
    transfer             ESM-MSR (sigma = 0.5, then sigma = 1), MutateEverything, ThermoMPNN, SPURS
    supervised           GeoDDG-3D, Rosetta, ESM3ΔG
    megascale            ESMtherm, IFUM, DDGemb, DDMut

``ABLATION_STYLES`` provides non-conflicting styles for all ``FINAL_*`` ablation families (never using
``#c0392b``, which is reserved for training-overlap rings, and reserving the black diamond ``'D'`` for
``FINAL_p0`` / ESM-MSR).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from types import MappingProxyType

GROUP_ORDER = ['zeroshot', 'transfer', 'supervised', 'megascale']
GROUP_TITLE = {
    'zeroshot': 'Zero-shot',
    'transfer': 'Retrained on Homology-Restricted MegaScale',
    'supervised': 'Supervised, not trained\non Megascale',
    'megascale': 'Trained on Megascale*',
}
GROUP_ORDER_INDEX = {g: i for i, g in enumerate(GROUP_ORDER)}


@dataclass(frozen=True)
class StyleSpec(Mapping):
    """Immutable style entry supporting both attribute access and dict-like ``spec['color']`` lookup."""
    label: str
    color: str
    marker: str
    fill: bool
    group: str
    model: str
    ls: str = '-'
    lw: float = 2.0

    def __getitem__(self, key: str):
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def __iter__(self):
        for f in fields(self):
            yield f.name

    def __len__(self) -> int:
        return len(fields(self))

    def with_overrides(self, **overrides) -> 'StyleSpec':
        return replace(self, **overrides) if overrides else self

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class Model:
    name: str
    group: str
    color: str
    marker: str
    ls: str = '-'
    lw: float = 2.0
    single: str = ''
    multi: str = ''
    epi: str = ''
    add: str = ''
    views: tuple = ('', '')
    legend: str = ''
    label_all: str = ''
    label_pair: str = ''
    label_epi: str = ''
    sigma: bool = False
    additive_only: bool = False

    @property
    def short(self) -> str:
        return self.name

    @property
    def two_versions(self) -> bool:
        return bool(self.views[0] or self.views[1])

    @property
    def label(self) -> str:
        return self.legend or self.name


ROSTER = [
    # ---- zero-shot --------------------------------------------------------------------------------------------------------------------------------------------
    Model('ProSST', 'zeroshot', '#8e44ad', 'X', single='prosst', multi='prosst:comb', epi='prosst:epi', add='prosst',
          views=('WT-view', 'Dual-view'),
          label_all='ProSST\nWT-view (& Dual-view)', label_pair='ProSST\nWT-view → Dual-view', label_epi='ProSST\nDual-view −\nsum of singles'),
    Model('ProteinMPNN', 'zeroshot', '#34495e', 'd', ls='-.', single='pmpnn', multi='pmpnn', epi='pmpnn:epi', add='pmpnn+',
          label_all='ProteinMPNN', label_pair='ProteinMPNN\nOne-shot → Dual-path', label_epi='ProteinMPNN\nDual-path −\nsum of singles'),
    Model('ESM3-small-open', 'zeroshot', '#7f8c8d', '^', single='msrrel@0:wt', multi='msrrel@0:comb', epi='msrrel@0:epi', add='msrrel@0:wt+',
          views=('WT-view', 'Dual-view'), legend='ESM3-small-open (σ = 0)',
          label_all='ESM3-small-open\nWT-view (& Dual-view)', label_pair='ESM3-small-open\nWT-view → Dual-view', label_epi='ESM3-small-open\ncomb_AB −\nWT_A − WT_B'),
    # ---- transfer models, retrained on Homology-Restricted MegaScale ----------------------------------------------------------------------------------------
    Model('ESM-MSR σ=0.5', 'transfer', '#4d4d4d', 'D', lw=2.2, single='msrp0@0.5:wt', multi='msrp0@0.5:comb', epi='msrp0@0.5:epi', add='msrp0@0.5:wt+',
          views=('WT-view', 'Dual-view'), legend='ESM-MSR σ = 0.5', sigma=True,
          label_all='ESM-MSR σ = 0.5\nWT-view (& Dual-view)', label_pair='ESM-MSR σ = 0.5\nWT-view → Dual-view', label_epi='ESM-MSR σ = 0.5\ncomb_AB −\nWT_A − WT_B'),
    Model('ESM-MSR', 'transfer', '#000000', 'D', lw=2.6, single='msrp0:wt', multi='msrp0:comb', epi='msrp0:epi', add='msrp0:wt+',
          views=('WT-view', 'Dual-view'),
          label_all='ESM-MSR\nWT-view (& Dual-view)', label_pair='ESM-MSR\nWT-view → Dual-view', label_epi='ESM-MSR\ncomb_AB −\nWT_A − WT_B'),
    Model('MutateEverything', 'transfer', '#1b8a4b', 'h', single='me1', multi='me2', epi='me2:epi', views=('Stage 1', 'Stage 2'),
          label_all='MutateEverything\n(Stage 1 & Stage 2)', label_pair='MutateEverything\nStage 1 → Stage 2', label_epi='MutateEverything\nStage 2 − Stage 1'),
    Model('ThermoMPNN', 'transfer', '#1f5f8b', 'p', single='tmpnn', multi='tmpnnd', epi='tmpnnd:epi', views=('', '-D'),
          label_all='ThermoMPNN\n(& ThermoMPNN-D)', label_pair='ThermoMPNN\n→ ThermoMPNN-D', label_epi='ThermoMPNN\n-D − single'),
    Model('SPURS', 'transfer', '#d35400', 's', single='spurs', multi='spursm', epi='spursm:epi', views=('', '-multi'),
          label_all='SPURS\n(& SPURS-multi)', label_pair='SPURS\n→ SPURS-multi', label_epi='SPURS\nmulti − single'),
    # ---- supervised, not trained on Megascale -----------------------------------------------------------------------------------------------------------------
    Model('GeoDDG-3D', 'supervised', '#117a65', 'v', ls='--', single='geoddg', multi='geoddg', epi='geoddg:epi', add='geoddg+',
          label_all='GeoDDG-3D', label_pair='GeoDDG-3D\nAdd. singles → sim. double', label_epi='GeoDDG-3D\ndirect −\nsum of singles'),
    Model('Rosetta', 'supervised', '#8c564b', 'P', ls='--', single='rosetta', multi='rosetta', epi='rosetta:epi', add='rosetta+',
          label_all='Rosetta', label_pair='Rosetta\nAdd. singles → sim. double', label_epi='Rosetta\ndirect −\nsum of singles'),
    Model('ESM3ΔG', 'supervised', '#cb4335', '*', ls=':', single='esm3dg', multi='esm3dg', epi='esm3dg:epi', add='esm3dg+',
          label_all='ESM3ΔG', label_pair='ESM3ΔG\nAdd. singles → sim. double', label_epi='ESM3ΔG\ndirect −\nsum of singles'),
    # ---- trained on Megascale: supplement only ----------------------------------------------------------------------------------------------------------------
    Model('ESMtherm', 'megascale', '#7d6608', '<', single='esmtherm', multi='esmtherm', legend='ESMtherm*', label_all='ESMtherm*'),
    Model('IFUM', 'megascale', '#8d6e63', 'v', single='ifum_wt', multi='ifum_wt', legend='IFUM*', label_all='IFUM*'),
    Model('DDGemb', 'megascale', '#b7950b', '>', single='ddgemb', multi='ddgemb', epi='ddgemb:epi', add='ddgemb+',
          label_all='DDGemb', label_pair='DDGemb\nAdd. singles → sim. double', label_epi='DDGemb\ndirect −\nsum of singles'),
    Model('DDMut', 'megascale', '#d4ac0d', '>', single='ddmut', multi='ddmut', label_all='DDMut'),
]
BY_NAME = MappingProxyType({m.name: m for m in ROSTER})
ORDER = [m.name for m in ROSTER]
ORDER_INDEX = MappingProxyType({n: i for i, n in enumerate(ORDER)})

UNIFIED = [m.name for m in ROSTER if not m.sigma and m.group != 'megascale']
ZEROSHOT_MODELS = [m.name for m in ROSTER if m.group == 'zeroshot']
ZEROSHOT = ZEROSHOT_MODELS
TRANSFER = [m.name for m in ROSTER if m.group == 'transfer' and not m.sigma]
SUPERVISED = [m.name for m in ROSTER if m.group == 'supervised']
MEGASCALE_MODELS = [m.name for m in ROSTER if m.group == 'megascale']

KEY_MODEL = MappingProxyType({
    'prosst': 'ProSST', 'pmpnn': 'ProteinMPNN', 'msrrel@0': 'ESM3-small-open',
    'msrp0': 'ESM-MSR', 'msrp0@0.5': 'ESM-MSR σ=0.5',
    'me1': 'MutateEverything', 'me2': 'MutateEverything',
    'tmpnn': 'ThermoMPNN', 'tmpnnd': 'ThermoMPNN',
    'spurs': 'SPURS', 'spursm': 'SPURS',
    'spursu': 'SPURS', 'spursum': 'SPURS',
    'tmpnnpub': 'ThermoMPNN', 'tmpnndpub': 'ThermoMPNN',
    'geoddg': 'GeoDDG-3D', 'rosetta': 'Rosetta', 'esm3dg': 'ESM3ΔG',
    'esmtherm': 'ESMtherm', 'ifum_wt': 'IFUM', 'ifum_esmfold': 'IFUM',
    'ddgemb': 'DDGemb', 'ddmut': 'DDMut',
})
SINGLES_KEYS = frozenset({
    'prosst', 'msrp0:wt', 'msrp0@0.5:wt', 'msrrel@0:wt',
    'me1', 'me1+', 'tmpnn', 'spurs', 'spursu', 'tmpnnpub',
})

VARIANT_COLOR = MappingProxyType({
    'ESM-MSR ctx ctrl': '#b5b5b5',
    'ESM-MSR independent masking': '#2c5f8a',
    'ESM-MSR masked marginal': '#5b8db8',
})

CATEGORY_COLOR = MappingProxyType({
    'Stability': '#6250d6',
    'Expression': '#2a78d6',
    'Activity': '#eb6834',
    'Binding': '#1baf7a',
    'OrganismalFitness': '#e87ba4',
})

# Canonical non-conflicting styles for all 11 FINAL_* ablation families (+ 2 cross-mask arms).
# Never uses #c0392b (reserved for training-overlap rings) and reserves 'D' for FINAL_p0.
ABLATION_STYLES = MappingProxyType({
    'FINAL_p0': StyleSpec('ESM-MSR (reference)', '#000000', 'D', True, 'reference', 'FINAL_p0', lw=2.6),
    'FINAL_p1': StyleSpec('Independent single-site masking', '#2c5f8a', 's', True, 'masking', 'FINAL_p1', lw=2.0),
    'FINAL_p1_marg': StyleSpec('WT independent + MT marginal', '#5b8db8', '^', True, 'masking', 'FINAL_p1_marg', lw=2.0),
    'FINAL_p0_pw0': StyleSpec('No pair loss', '#117a65', 'o', True, 'loss', 'FINAL_p0_pw0', lw=2.0),
    'FINAL_p0_norank': StyleSpec('No rank loss', '#d68910', 'v', True, 'loss', 'FINAL_p0_norank', lw=2.0),
    'FINAL_p0_rcthr0': StyleSpec('Rank censor threshold 0', '#7d6608', '<', True, 'loss', 'FINAL_p0_rcthr0', lw=2.0),
    'FINAL_p0_norange': StyleSpec('Exclude range-tagged cells', '#1e8449', '>', True, 'loss', 'FINAL_p0_norange', lw=2.0),
    'FINAL_p0_mse': StyleSpec('Plain MSE (no Tobit NLL)', '#b9770e', 'p', True, 'loss', 'FINAL_p0_mse', lw=2.0),
    'FINAL_p0_noseqhwm': StyleSpec('No sequence head', '#6c3483', 'h', True, 'architecture', 'FINAL_p0_noseqhwm', lw=2.0),
    'FINAL_p0_r2a4': StyleSpec('MT LoRA rank 2 / alpha 4', '#8d6e63', 'P', True, 'architecture', 'FINAL_p0_r2a4', lw=2.0),
    'FINAL_p0_mega': StyleSpec('Full MegaScale split', '#34495e', 'X', True, 'data', 'FINAL_p0_mega', lw=2.0),
    'FINAL_p0@independent': StyleSpec('ESM-MSR, tested masked', '#666666', 'v', False, 'cross-mask', 'FINAL_p0@independent', ls='--', lw=1.8),
    'FINAL_p1@unmasked': StyleSpec('Masked model, tested unmasked', '#8c8c8c', '^', False, 'cross-mask', 'FINAL_p1@unmasked', ls='--', lw=1.8),
})


def tint(color_hex: str, f: float) -> str:
    """Mix ``color_hex`` with white by fraction ``f``."""
    c = color_hex.lstrip('#')
    r, g, b = (int(c[i:i + 2], 16) for i in (0, 2, 4))
    return '#%02x%02x%02x' % tuple(int(round(v + (255 - v) * f)) for v in (r, g, b))


def version_label(name: str, i: int) -> str:
    """Version ``i`` (0 singles, 1 doubles) display name of roster model ``name``."""
    m = BY_NAME[name]
    v = m.views[i]
    base = m.legend or m.name if m.sigma else m.name
    if not v:
        return base
    return base + v if v.startswith('-') else f'{base} {v}'


def tick_label(name: str, i: int) -> str:
    """Two-line x-tick label of version ``i`` of roster model ``name``."""
    m = BY_NAME[name]
    v = m.views[i]
    base = m.legend or m.name if m.sigma else m.name
    if not v:
        return base
    return base + v if v.startswith('-') else f'{base}\n{v}'


def ordered(names) -> list[str]:
    """Return ``names`` sorted in canonical ``ROSTER`` order (unlisted names preserved at end)."""
    names = list(names)
    return sorted(names, key=lambda n: (ORDER_INDEX.get(n, len(ORDER)), names.index(n)))


def grouped(names) -> list[tuple[str, list[str]]]:
    """Return ``[(group, [names])]`` in canonical group and model order."""
    out = []
    for g in GROUP_ORDER:
        sel = [n for n in ordered(names) if n in BY_NAME and BY_NAME[n].group == g]
        if sel:
            out.append((g, sel))
    return out


def color(name: str) -> str:
    if name in BY_NAME:
        return BY_NAME[name].color
    if name in ABLATION_STYLES:
        return ABLATION_STYLES[name].color
    raise KeyError(name)


def line_style(name: str) -> tuple[str, str, float, int]:
    """Return ``(colour, linestyle, linewidth, markersize)`` for curve plots."""
    if name in BY_NAME:
        m = BY_NAME[name]
        return m.color, m.ls, m.lw, 8 if m.name == 'ESM-MSR' else 6
    if name in ABLATION_STYLES:
        s = ABLATION_STYLES[name]
        return s.color, s.ls, s.lw, 8 if name == 'FINAL_p0' else 6
    raise KeyError(name)


def column(name: str, which: str = 'all', **kw) -> dict | None:
    """Build a bar-chart column descriptor for roster model ``name`` ('all', 'pair', or 'epi')."""
    m = BY_NAME[name]
    if which == 'all':
        sd = dict(label_s=tick_label(name, 0), label_d=tick_label(name, 1)) if m.two_versions else {}
        return dict(name=name, cat=m.group, color=m.color, label=m.label_all or m.name, single=m.single, double=m.multi, **sd, **kw)
    if which == 'pair':
        if not (m.add or m.two_versions or m.additive_only):
            return None  # single prediction per variant and no additive singles sum (ESMtherm, IFUM, DDMut)
        return dict(name=name, cat=m.group, color=m.color, label=m.label_pair or m.name,
                    single=m.add or m.single, double=None if m.additive_only else m.multi, **kw)
    if which == 'epi':
        if not m.epi:
            return None
        return dict(name=name, cat=m.group, color=m.color, label=m.label_epi or m.name, single=None, double=m.epi, **kw)
    raise ValueError(which)
