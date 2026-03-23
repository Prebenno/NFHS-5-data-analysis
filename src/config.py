"""Thresholds and display constants."""

#Figure settings
CI_ALPHA         = 0.15
DPI              = 300
FIG_DIR          = "figs"
FIG_DIR_BINS     = "figs_bins"

#Analysis constants
N_BRACKETS       = 5
N_HEIGHT_BRACKETS = 4
BP_DIFF_THRESHOLD = 25

#Panel labels
PANEL_LABELS     = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

#Column names
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

#Hypertension stages
HTN_ORDER = [
    "Normal",
    "Pre-Hypertensive",
    "Hypertensive 1",
    "Hypertensive 2",
    "Hypertensive Crisis",
]


from collections import OrderedDict

SBP_COL = "First SYSTOLIC reading"
DBP_COL = "First DIASTOLIC reading"
#BMI ranges
BMI_RANGES = {
    "14-": (1, 15), "15": (15, 16), "16": (16, 17), "17": (17, 18),
    "18": (18, 19), "19": (19, 20), "20": (20, 21), "21": (21, 22),
    "22": (22, 23), "23": (23, 24), "24": (24, 25), "25": (25, 26),
    "26": (26, 27), "27+": (27, 99),
}

#BMI and age course bins
COARSE_BMI = OrderedDict([('Low', (1, 18.5)), ('Middle', (18.5, 25)), ('High', (25, 99))])
COARSE_AGE = OrderedDict([('15-25', (15, 26)), ('26-37', (26, 38)), ('38-49', (38, 50))])
BMI_ORDER = ['High', 'Middle', 'Low']
AGE_ORDER = ['15-25', '26-37', '38-49']



# Phase 1 = states surveyed entirely pre-COVID (Jun 2019 – Mar 2020).
# Phase 2 = states surveyed entirely or predominantly post-COVID (Nov 2020 – Apr 2021).
# Overlap states 7, 8, 34 are classified by majority period (7→Phase1, 8→Phase1, 34→Phase1).
PHASE1_STATE_CODES = {
    1, 2, 7, 8, 10, 11, 13, 14, 15, 16, 17, 18, 19,
    24, 25, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37,
}
PHASE2_STATE_CODES = {3, 4, 5, 6, 9, 12, 20, 21, 22, 23, 33}
