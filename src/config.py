"""Thresholds and display constants."""

# Figure settings
CI_ALPHA         = 0.15
DPI              = 300
FIG_DIR          = "figs"
FIG_DIR_BINS     = "figs_bins"

# Analysis constants
N_BRACKETS       = 5
N_HEIGHT_BRACKETS = 4
BP_DIFF_THRESHOLD = 25

# Panel labels
PANEL_LABELS     = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Column names
SYSTOLIC_COLS  = [
    "First SYSTOLIC reading",
    "Second SYSTOLIC reading",
    "Third SYSTOLIC reading",
]
DIASTOLIC_COLS = [
    "First DIASTOLIC reading",
    "Second DIASTOLIC reading",
    "Third DIASTOLIC reading",
]
ALL_BP_COLS    = SYSTOLIC_COLS + DIASTOLIC_COLS

READING_PAIRS  = [("First", "Second"), ("Second", "Third"), ("First", "Third")]

# Hypertension stages
HTN_ORDER = [
    "Normal",
    "Pre-Hypertensive",
    "Hypertensive 1",
    "Hypertensive 2",
    "Hypertensive Crisis",
]
