"""Centralized configuration for A.E.C.O.

All hardcoded values should be read from here instead of
being scattered across modules. Environment variables override defaults.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Paths ---
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
LABELS_DIR = DATA_DIR / "labels"
PATCHES_DIR = DATA_DIR / "patches"
GSW_DIR = DATA_DIR / "gsw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
VALIDATION_DIR = OUTPUTS_DIR / "validation"

# --- Satellite Data ---
NTB_BBOX = [115.7, -9.1, 119.2, -8.1]
LOMBOK_BBOX = [115.7, -8.9, 116.6, -8.1]
SUMBAWA_BBOX = [116.6, -9.1, 119.2, -8.1]
SUMBAWA_CENTER_LAT = -8.5
SUMBAWA_CENTER_LON = 117.8

# --- SAR Thresholds ---
VV_THRESH = float(os.environ.get("VV_THRESH", "-18.0"))
VH_THRESH = float(os.environ.get("VH_THRESH", "-24.0"))
SAR_VV_THRESH = float(os.environ.get("SAR_VV_THRESH", "-15.0"))

# --- Optical Thresholds ---
NDWI_THRESH = float(os.environ.get("NDWI_THRESH", "0.1"))

# --- Ocean Masking ---
OCEAN_ELEV_THRESHOLD = float(os.environ.get("OCEAN_ELEV_THRESHOLD", "2.0"))
SLOPE_OCEAN_THRESHOLD = float(os.environ.get("SLOPE_OCEAN_THRESHOLD", "1.0"))

# --- ML Training ---
RANDOM_STATE = 42
FEATURE_NAMES = ["NDWI", "SAR_flood_mask", "Slope_deg", "VV_dB", "VH_dB", "HAND_m"]
SAMPLE_FRAC = 0.05

# --- U-Net ---
PATCH_SIZE = 256
STRIDE = 128
UNET_IN_CHANNELS = 6

# --- API ---
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))

# --- Data Retention (days) ---
RETENTION_RAW = int(os.environ.get("RETENTION_RAW", "7"))
RETENTION_PREDICTIONS = int(os.environ.get("RETENTION_PREDICTIONS", "30"))
RETENTION_REPORTS = int(os.environ.get("RETENTION_REPORTS", "14"))
RETENTION_PATCHES = int(os.environ.get("RETENTION_PATCHES", "90"))
RETENTION_HISTORY = int(os.environ.get("RETENTION_HISTORY", "365"))
