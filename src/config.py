"""Project-wide configuration."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA = DATA_DIR / "processed" / "collections_dataset.csv"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"

FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 20260521

CUTOFF = 1000.0

CHANNELS = ["SMS", "Email", "Phone", "FieldVisit"]
AGENT_TIERS = ["Junior", "Senior", "Specialist"]

COST_MATRIX = {
    ("SMS", "Junior"): 0.50,
    ("SMS", "Senior"): 0.50,
    ("SMS", "Specialist"): 0.50,
    ("Email", "Junior"): 0.30,
    ("Email", "Senior"): 0.30,
    ("Email", "Specialist"): 0.30,
    ("Phone", "Junior"): 5.00,
    ("Phone", "Senior"): 10.00,
    ("Phone", "Specialist"): 15.00,
    ("FieldVisit", "Junior"): 30.00,
    ("FieldVisit", "Senior"): 60.00,
    ("FieldVisit", "Specialist"): 100.00,
}

TIME_MINUTES = {"SMS": 1, "Email": 2, "Phone": 15, "FieldVisit": 120}

LEVEL_0_COST = 1.5
LEVEL_1_BASE_COST = 50.0

AGENT_CAPACITY_HOURS = {
    "Junior": 180.0,
    "Senior": 140.0,
    "Specialist": 90.0,
}

RDD_BANDWIDTH_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
PLACEBO_CUTOFFS = [400.0, 600.0, 800.0, 900.0, 1100.0, 1200.0, 1400.0]

TOTAL_BUDGET = 25000.0

RDD_N_BINS = 40
PLACEBO_BANDWIDTH_FRACTION = 0.6
