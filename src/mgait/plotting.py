"""House chart style, shared by the notebooks and the presentation figures.

Colour roles:
- SPEED: ordered conditions use one hue stepped light to dark (ordinal ramp).
- CAT: identity (modalities, models) uses a fixed categorical order, never cycled.
- Text and axes stay in neutral inks; colour is reserved for data.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

from . import config as C

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SPEED_COLORS = {"0.5": "#86b6ef", "0.75": "#2a78d6", "1": "#104281"}
SPEED_LABEL = {"0.5": "0.50 m/s", "0.75": "0.75 m/s", "1": "1.00 m/s"}
MODALITY_COLORS = {"Force plates": CAT[0], "IMU": CAT[1], "EMG": CAT[2], "EEG": CAT[3]}
SIDE_COLORS = {"L": CAT[0], "R": CAT[1]}
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")
HIGHLIGHT = CAT[1]
NEUTRAL = "#b9b8b2"


def use_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "axes.labelcolor": INK2, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "text.color": INK, "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2.0, "lines.markersize": 6, "patch.linewidth": 0,
        "axes.prop_cycle": mpl.cycler(color=CAT),
    })


def save(fig, name: str) -> None:
    fig.savefig(C.FIGURES / f"{name}.png")


def note(ax, text: str, loc: str = "lower right") -> None:
    """Small muted annotation inside an axis."""
    xy = {"lower right": (0.98, 0.03, "right", "bottom"), "upper right": (0.98, 0.97, "right", "top"),
          "upper left": (0.02, 0.97, "left", "top"), "lower left": (0.02, 0.03, "left", "bottom")}[loc]
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha=xy[2], va=xy[3], fontsize=8.5, color=INK2)


def band(ax, x, mean, sd, color, label=None, alpha=0.18):
    ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=alpha, linewidth=0)
    ax.plot(x, mean, color=color, label=label)
