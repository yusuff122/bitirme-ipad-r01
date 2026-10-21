from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

class R01FrameDataset(Dataset):
    """Generic frame reader. Labels/splits must come from inspected annotations."""
    def __init__(self, records, transform=None):
        self.records = records.reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.records)
    def __getitem__(self, index):
        row = self.records.iloc[index]
        image = Image.open(row['path']).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, row.to_dict()

def list_frames(root):
    return sorted(p for p in Path(root).rglob('*') if p.suffix.lower() in IMAGE_EXTS)
