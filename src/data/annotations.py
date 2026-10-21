"""Annotation helpers. Dataset label formats are discovered, never guessed."""
from pathlib import Path
import pandas as pd

def read_annotation(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == '.csv': return pd.read_csv(path)
    if path.suffix.lower() in {'.tsv', '.txt'}: return pd.read_csv(path, sep='\t')
    raise ValueError(f'Unsupported annotation format: {path.suffix}')
