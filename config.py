"""Shared project defaults; paths can be overridden through YAML configs/CLI."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
R01_DIR = DATA_DIR / "r01"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
SEED = 42
IMAGE_SIZE = 224
CLIP_LENGTH = 16
CLIP_STRIDE = 4
