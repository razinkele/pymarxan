"""Ocean palette — centralized color constants for the Marxan UI.

Every visualization module should import from here rather than
hardcoding hex values. The palette harmonizes with the CSS custom
properties defined in ``www/ocean_theme.css``.
"""
from __future__ import annotations

# ── Primary Ocean Palette ──────────────────────────────────────────
OCEAN_DEEP     = "#0b2545"   # deep navy
OCEAN_MID      = "#134074"   # mid-ocean blue
OCEAN_BRIGHT   = "#13507c"   # brighter blue
OCEAN_TEAL     = "#0fa3b1"   # tropical teal (primary accent)
OCEAN_AQUA     = "#48bfe3"   # lighter aqua
OCEAN_SEAFOAM  = "#b8e0d2"   # seafoam green highlight
OCEAN_MIST     = "#e8f4f8"   # very light blue
OCEAN_FOAM     = "#f0f7fa"   # near-white blue
OCEAN_WHITE    = "#fafeff"   # purest white hint of blue

# ── Warm Accents (Coral Reef) ─────────────────────────────────────
REEF_CORAL     = "#e07a5f"   # living coral
REEF_WARM      = "#f2cc8f"   # sandy warm
REEF_SAND      = "#e8dab2"   # beach sand

# ── Semantic Colors ───────────────────────────────────────────────
SUCCESS        = "#2d936c"   # kelp green
WARNING        = "#e6a23c"   # amber
DANGER         = "#c1440e"   # deep coral
INFO           = "#0fa3b1"   # same as teal

# ── Map Categorical Colors ────────────────────────────────────────
MAP_AVAILABLE  = "#8faec1"   # muted blue-gray (PU available)
MAP_LOCKED_IN  = "#2d936c"   # kelp green   (PU locked-in)
MAP_LOCKED_OUT = "#c1440e"   # deep coral   (PU locked-out)
MAP_SELECTED   = "#0fa3b1"   # teal         (PU selected in solution)
MAP_NOT_SEL    = "#8faec1"   # muted blue-gray (PU not selected)
MAP_FALLBACK   = "#bcc9d1"   # light steel gray

# ── Comparison Map ────────────────────────────────────────────────
CMP_BOTH       = "#0fa3b1"   # teal  — in both solutions
CMP_A_ONLY     = "#134074"   # mid-ocean — A only
CMP_B_ONLY     = "#e07a5f"   # coral  — B only
CMP_NEITHER    = "#bcc9d1"   # steel gray — neither

# ── Gradient Endpoints ────────────────────────────────────────────
# Cost heatmap: seafoam → deep coral
COST_LOW_RGB   = (184, 224, 210)  # seafoam #b8e0d2
COST_HIGH_RGB  = (193, 68, 14)    # deep coral #c1440e

# Frequency: white → deep navy
FREQ_LOW_RGB   = (250, 254, 255)  # ocean white
FREQ_HIGH_RGB  = (11, 37, 69)     # ocean deep #0b2545

# Network metric: aqua → deep navy
METRIC_LOW_RGB  = (72, 191, 227)  # aqua #48bfe3
METRIC_HIGH_RGB = (11, 37, 69)    # deep navy #0b2545

# Network edge
EDGE_COLOR      = "#48bfe3"       # aqua

# ── Chart/Plot Colors ────────────────────────────────────────────
PLOT_PRIMARY    = "#0fa3b1"   # teal
PLOT_SECONDARY  = "#e07a5f"   # coral
PLOT_TERTIARY   = "#134074"   # mid-ocean
PLOT_SUCCESS    = "#2d936c"   # kelp green
PLOT_ACCENT     = "#e6a23c"   # amber

# ── Inline text (Met / Not Met in summary_table) ─────────────────
TEXT_MET        = "#2d936c"   # kelp green
TEXT_NOT_MET    = "#c1440e"   # deep coral
