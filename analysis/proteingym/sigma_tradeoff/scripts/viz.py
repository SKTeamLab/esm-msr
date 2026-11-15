"""Plot style shared by the figure scripts (reference data-viz palette, light surface)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from common import *

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8982", "#e4e3de"
CAT_COLOR = {"Stability": "#6250d6", "Expression": "#2a78d6", "Activity": "#eb6834",
             "Binding": "#1baf7a", "OrganismalFitness": "#e87ba4"}
CAT_LABEL = {"Stability": "Stability (Tsuboyama)", "Expression": "Expression / abundance",
             "Activity": "Activity", "Binding": "Binding", "OrganismalFitness": "Organismal fitness"}
CATS = ["Stability", "Expression", "Activity", "Binding", "OrganismalFitness"]
AXIS_COLOR.update({"ABUND": "#2a78d6", "ACT": "#eb6834", "SPEC": "#8a6f3b", "COND": "#8a8982", "FIT": "#e87ba4", "BIND": "#1baf7a"})
AXIS_NAME = {"ABUND": "abundance / expression", "ACT": "activity / function", "BIND": "binding",
             "FIT": "growth / fitness", "SPEC": "drug resistance / Hsp90 dep.", "COND": "condition variant"}
CLASS_COLOR = {"RISE": "#2a78d6", "HUMP": "#86b6ef", "HUMP_LOSS": "#e8a14a", "FLAT": "#cfcec8", "FALL": "#d03b3b"}
CLASS_LABEL = {"RISE": "rises / plateaus", "HUMP": "interior optimum, net gain", "HUMP_LOSS": "interior optimum, net loss at 1.0",
               "FLAT": "flat", "FALL": "falls"}
CLASSES = ["RISE", "HUMP", "HUMP_LOSS", "FLAT", "FALL"]
DIV = LinearSegmentedColormap.from_list("div", ["#d03b3b", "#f0efec", "#2a78d6"])  # red(-) - gray - blue(+)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "font.size": 9.5, "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6,
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
})


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    p = f"{FIG}/{name}.png"
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p)


def sigma_axis(ax, label=True):
    ax.set_xticks(SIGMAS)
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    if label:
        ax.set_xlabel("sigma (LoRA alpha scale)")
