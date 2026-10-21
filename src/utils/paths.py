from pathlib import Path
from config import ROOT, DATA_DIR, R01_DIR, MODELS_DIR, OUTPUTS_DIR

def ensure_project_dirs():
    for path in (DATA_DIR, R01_DIR, MODELS_DIR, OUTPUTS_DIR):
        path.mkdir(parents=True, exist_ok=True)

def timestamp():
    from datetime import datetime
    return datetime.now().strftime('%Y%m%d_%H%M%S')
