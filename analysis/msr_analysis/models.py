"""The one roster of models for every figure and table: name, order, colour, marker, line style, which prediction to read for singles and for multi-mutants, and how the versions are named.

A model has the same colour, marker, line style and position in every figure, bar chart, dot panel, line plot and table (an exception has to say so in its script).  The order is

    zero-shot            ProSST, ProteinMPNN, ESM3-small-open
    retrained on the Homology-Restricted MegaScale split (the transfer models)
                         ESM-MSR (sigma = 1, then sigma = 0.5), MutateEverything, ThermoMPNN, SPURS
    supervised, not trained on Megascale
                         GeoDDG-3D, Rosetta, ESM3ΔG
    trained on Megascale (supplement only, drawn last and marked with an asterisk)
                         ESMtherm, IFUM, SPURS (published checkpoint), DDGemb, DDMut

ESM-MSR is the final model (FINAL_p0, seeds 1-3) and is never labelled p0; no other version of it (devel, baseline, released) is drawn.  Every figure scores a variant with the model that fits it: the singles
version on singles (ESM-MSR and ESM3-small-open: the WT-view; MutateEverything stage 1, ThermoMPNN, SPURS single) and the doubles version on multi-mutants (ESM-MSR and ESM3-small-open: the Dual-view, i.e. the
combined head; MutateEverything stage 2, ThermoMPNN-D, SPURS multi).  A hue is a model, a version is a fill: singles versions are hollow markers, doubles versions and models with one version are filled
(dashed against solid in line plots).  Figures that compare the versions themselves (doubles-vs-singles tables, additive-vs-epistatic pairs) are the stated exception and keep one colour per model.
"""
from dataclasses import dataclass

GROUP_ORDER = ['zeroshot', 'transfer', 'supervised', 'megascale']
GROUP_TITLE = {'zeroshot': 'Zero-shot', 'transfer': 'Retrained on Homology-Restricted MegaScale', 'supervised': 'Supervised, not trained\non Megascale', 'megascale': 'Trained on Megascale*'}
GROUP_ORDER_INDEX = {g: i for i, g in enumerate(GROUP_ORDER)}


@dataclass(frozen=True)
class Model:
    name: str                                   # the name in every table and legend
    group: str
    color: str
    marker: str
    ls: str = '-'                               # line style of its curves
    lw: float = 2.0
    single: str = ''                            # prediction key (``msr_analysis.data`` frames) that scores singles
    multi: str = ''                             # ... and multi-mutants (same as ``single`` for a model with one version)
    epi: str = ''                               # key of the predicted epistasis (multi - additive singles), '' when it has none
    add: str = ''                               # key of the sum of the singles' predictions
    views: tuple = ('', '')                     # names of the singles and the doubles version
    legend: str = ''                            # the legend entry when it differs from the name
    label_all: str = ''                         # x tick label of the column that scores every variant (singles with the first version, multi with the second)
    label_pair: str = ''                        # x tick label of an additive -> epistatic pair
    label_epi: str = ''                         # x tick label of a predicted-epistasis column
    sigma: bool = False                         # a LoRA-scale variant of ESM-MSR, not part of the unified roster of the external sets
    additive_only: bool = False                 # scores every variant as a sum over its substitutions (no epistatic version): one bar in an additive -> epistatic pair

    @property
    def short(self):
        return self.name

    @property
    def two_versions(self):
        return bool(self.views[0] or self.views[1])

    @property
    def label(self):
        return self.legend or self.name


ROSTER = [
    # ---- zero-shot --------------------------------------------------------------------------------------------------------------------------------------------
    Model('ProSST', 'zeroshot', '#8e44ad', 'X', additive_only=True, single='prosst', multi='prosst', label_all='ProSST', label_pair='ProSST\n(additive only)', label_epi='ProSST'),
    Model('ProteinMPNN', 'zeroshot', '#34495e', 'd', ls='-.', single='pmpnn', multi='pmpnn', epi='pmpnn:epi', add='pmpnn+', label_all='ProteinMPNN',
          label_pair='ProteinMPNN\nOne-shot singles → AR doubles', label_epi='ProteinMPNN\ndirect −\nsum of singles'),
    Model('ESM3-small-open', 'zeroshot', '#7f8c8d', '^', single='msrrel@0:wt', multi='msrrel@0:comb', epi='msrrel@0:epi', add='msrrel@0:wt+', views=('WT-view', 'Dual-view'), legend='ESM3-small-open (σ = 0)',
          label_all='ESM3-small-open\nWT-view (& Dual-view)', label_pair='ESM3-small-open\nWT-view → Dual-view', label_epi='ESM3-small-open\ncomb_AB −\nWT_A − WT_B'),
    # ---- transfer models, retrained on Homology-Restricted MegaScale ----------------------------------------------------------------------------------------
    Model('ESM-MSR', 'transfer', '#000000', 'D', lw=2.6, single='msrp0:wt', multi='msrp0:comb', epi='msrp0:epi', add='msrp0:wt+', views=('WT-view', 'Dual-view'),
          label_all='ESM-MSR\nWT-view (& Dual-view)', label_pair='ESM-MSR\nWT-view → Dual-view', label_epi='ESM-MSR\ncomb_AB −\nWT_A − WT_B'),
    Model('ESM-MSR σ=0.5', 'transfer', '#4d4d4d', 'D', lw=2.2, views=('WT-view', 'Dual-view'), legend='ESM-MSR σ = 0.5', sigma=True,
          label_all='ESM-MSR σ = 0.5\nWT-view (& Dual-view)'),
    Model('MutateEverything', 'transfer', '#1b8a4b', 'h', single='me1', multi='me2', epi='me2:epi', views=('Stage 1', 'Stage 2'),
          label_all='MutateEverything\n(Stage 1 & Stage 2)', label_pair='MutateEverything\nStage 1 → Stage 2', label_epi='MutateEverything\nStage 2 − Stage 1'),
    Model('ThermoMPNN', 'transfer', '#1f5f8b', 'p', single='tmpnn', multi='tmpnnd', epi='tmpnnd:epi', views=('', '-D'),
          label_all='ThermoMPNN\n(& ThermoMPNN-D)', label_pair='ThermoMPNN\n→ ThermoMPNN-D', label_epi='ThermoMPNN\n-D − single'),
    Model('SPURS', 'transfer', '#d35400', 's', single='spurs', multi='spursm', epi='spursm:epi', views=('', '-multi'),
          label_all='SPURS\n(& SPURS-multi)', label_pair='SPURS\n→ SPURS-multi', label_epi='SPURS\nmulti − single'),
    # ---- supervised, not trained on Megascale -----------------------------------------------------------------------------------------------------------------
    Model('GeoDDG-3D', 'supervised', '#117a65', 'v', ls='--', single='geoddg', multi='geoddg', epi='geoddg:epi', add='geoddg+', label_all='GeoDDG-3D',
          label_pair='GeoDDG-3D\nAdd. singles → sim. double', label_epi='GeoDDG-3D\ndirect −\nsum of singles'),
    Model('Rosetta', 'supervised', '#8c564b', 'P', ls='--', single='rosetta', multi='rosetta', epi='rosetta:epi', add='rosetta+', label_all='Rosetta',
          label_pair='Rosetta\nAdd. singles → sim. double', label_epi='Rosetta\ndirect −\nsum of singles'),
    Model('ESM3ΔG', 'supervised', '#cb4335', '*', ls=':', single='esm3dg', multi='esm3dg', epi='esm3dg:epi', add='esm3dg+', label_all='ESM3ΔG',
          label_pair='ESM3ΔG\nAdd. singles → sim. double', label_epi='ESM3ΔG\ndirect −\nsum of singles'),
    # ---- trained on Megascale: supplement only ----------------------------------------------------------------------------------------------------------------
    Model('ESMtherm', 'megascale', '#7d6608', '<', single='esmtherm', multi='esmtherm', legend='ESMtherm*', label_all='ESMtherm*'),
    Model('IFUM', 'megascale', '#8d6e63', 'v', single='ifum_wt', multi='ifum_wt', legend='IFUM*', label_all='IFUM*'),
    Model('SPURS published*', 'megascale', '#e59866', 's', single='spurspub', multi='spurspubm', legend='SPURS (published checkpoint)*', label_all='SPURS (published)*'),
    Model('DDGemb', 'megascale', '#b7950b', '>', single='ddgemb', multi='ddgemb', epi='ddgemb:epi', add='ddgemb+', label_all='DDGemb', label_pair='DDGemb\nAdd. singles → sim. double', label_epi='DDGemb\ndirect −\nsum of singles'),
    Model('DDMut', 'megascale', '#d4ac0d', '>', single='ddmut', multi='ddmut', label_all='DDMut'),
]
BY_NAME = {m.name: m for m in ROSTER}
ORDER = [m.name for m in ROSTER]
ORDER_INDEX = {n: i for i, n in enumerate(ORDER)}

# the unified roster of the external benchmarks (``msr_analysis.external``): every model of the main figures, with one prediction per set (its singles version on a singles set, its doubles version on a set of multi-mutants)
UNIFIED = [m.name for m in ROSTER if not m.sigma and m.group != 'megascale']
ZEROSHOT = [m.name for m in ROSTER if m.group == 'zeroshot']
TRANSFER = [m.name for m in ROSTER if m.group == 'transfer' and not m.sigma]
SUPERVISED = [m.name for m in ROSTER if m.group == 'supervised']

# the raw prediction key (``msr_analysis.data``) -> the model it belongs to.  A key is a model plus a version: the version is the fill of the marker.
KEY_MODEL = {
    'prosst': 'ProSST', 'pmpnn': 'ProteinMPNN', 'msrrel@0': 'ESM3-small-open', 'msrp0': 'ESM-MSR', 'msrp0@0.5': 'ESM-MSR σ=0.5',
    'me1': 'MutateEverything', 'me2': 'MutateEverything', 'tmpnn': 'ThermoMPNN', 'tmpnnd': 'ThermoMPNN', 'spurs': 'SPURS', 'spursm': 'SPURS',
    'spursu': 'SPURS', 'spursum': 'SPURS', 'spurspub': 'SPURS published*', 'spurspubm': 'SPURS published*', 'tmpnnpub': 'ThermoMPNN', 'tmpnndpub': 'ThermoMPNN',
    'geoddg': 'GeoDDG-3D', 'rosetta': 'Rosetta', 'esm3dg': 'ESM3ΔG', 'esmtherm': 'ESMtherm', 'ifum_wt': 'IFUM', 'ifum_esmfold': 'IFUM', 'ddgemb': 'DDGemb', 'ddmut': 'DDMut',
}
SINGLES_KEYS = {'msrp0:wt', 'msrp0@0.5:wt', 'msrrel@0:wt', 'me1', 'me1+', 'tmpnn', 'spurs', 'spursu', 'spurspub', 'tmpnnpub'}       # the singles versions of the models that have two: hollow markers, solid lines
ADDITIVE_KEYS = {'me1+', 'me2+'}


VARIANT_COLOR = {       # shades of the model's own colour for the control and ablation arms drawn beside it (the one place where a model has more than one colour)
    'ESM-MSR ctx ctrl': '#b5b5b5', 'ESM-MSR independent masking': '#4d4d4d', 'ESM-MSR masked marginal': '#8c8c8c',
}


CATEGORY_COLOR = {'Stability': '#6250d6', 'Expression': '#2a78d6', 'Activity': '#eb6834', 'Binding': '#1baf7a', 'OrganismalFitness': '#e87ba4'}      # the ProteinGym categories: one colour each in every figure


def tint(color, f):
    """``color`` mixed with white by the fraction ``f`` (a lighter shade of the same hue, for a secondary version of a model)."""
    c = color.lstrip('#')
    r, g, b = (int(c[i:i + 2], 16) for i in (0, 2, 4))
    return '#%02x%02x%02x' % tuple(int(round(v + (255 - v) * f)) for v in (r, g, b))


def version_label(name, i):
    """'ESM-MSR WT-view', 'MutateEverything Stage 2', 'ThermoMPNN-D', 'SPURS-multi', 'ProSST': the name of version ``i`` (0 singles, 1 doubles) of the roster model ``name``."""
    m = BY_NAME[name]
    v = m.views[i]
    if not v:
        return m.name
    return m.name + v if v.startswith('-') else f'{m.name} {v}'


def tick_label(name, i):
    """x tick label of version ``i`` of a model in a figure that draws the versions separately: 'ESM-MSR\nWT-view', 'MutateEverything\nStage 2', 'ThermoMPNN-D'."""
    m = BY_NAME[name]
    v = m.views[i]
    if not v:
        return m.name
    return m.name + v if v.startswith('-') else f'{m.name}\n{v}'


def model_of_key(key):
    """The roster entry of a prediction key such as ``msrp0:comb``, ``me2:epi``, ``spursm``, ``rosetta+``; None when the key belongs to no roster model."""
    base = key.split(':')[0].rstrip('+')
    name = KEY_MODEL.get(base)
    return BY_NAME[name] if name else None


def ordered(names):
    """``names`` (roster names; names outside the roster last, in their given order) in the canonical order."""
    names = list(names)
    return sorted(names, key=lambda n: (ORDER_INDEX.get(n, len(ORDER)), names.index(n)))


def grouped(names):
    """[(group, [names])] of the roster names in ``names``, in the canonical order of groups and models."""
    out = []
    for g in GROUP_ORDER:
        sel = [n for n in ordered(names) if n in BY_NAME and BY_NAME[n].group == g]
        if sel:
            out.append((g, sel))
    return out


def color(name):
    return BY_NAME[name].color


def line_style(name):
    """(colour, linestyle, linewidth, markersize) of a model's curve in the line plots (screening curves)."""
    m = BY_NAME[name]
    return m.color, m.ls, m.lw, 8 if m.name == 'ESM-MSR' else 6


def column(name, which='all', **kw):
    """A column of a bar figure for the roster model ``name`` (the keys of ``11_publication_figures.col``), or None when the model has no such column.  ``which``: 'all' (singles with the singles version, multi-mutants
    with the doubles version), 'pair' (additive -> epistatic: the sum of the singles' predictions, or the singles version, against the model that scores the double) or 'epi' (the predicted epistasis)."""
    m = BY_NAME[name]
    if which == 'all':
        sd = dict(label_s=tick_label(name, 0), label_d=tick_label(name, 1)) if m.two_versions else {}
        return dict(name=name, cat=m.group, color=m.color, label=m.label_all or m.name, single=m.single, double=m.multi, **sd, **kw)
    if which == 'pair':
        if not (m.add or m.two_versions or m.additive_only):
            return None                                                              # one prediction per variant and nothing to add up (ESM3ΔG)
        return dict(name=name, cat=m.group, color=m.color, label=m.label_pair or m.name, single=m.add or m.single, double=None if m.additive_only else m.multi, **kw)
    if which == 'epi':
        if not m.epi:
            return None
        return dict(name=name, cat=m.group, color=m.color, label=m.label_epi or m.name, single=None, double=m.epi, **kw)
    raise ValueError(which)
