import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import GroupShuffleSplit
from src.data.ipad_r01_dataset import list_frames

def main():
    p=argparse.ArgumentParser(description='Create the sequence-aware R01 manifest using official test labels.'); p.add_argument('--root',default='data/r01'); p.add_argument('--output',default='data/processed/r01_manifest.csv'); p.add_argument('--val-fraction',type=float,default=.2); p.add_argument('--seed',type=int,default=42); a=p.parse_args(); root=Path(a.root)
    if not (root/'training/frames').is_dir() or not (root/'testing/frames').is_dir(): raise SystemExit('Expected R01/training/frames and R01/testing/frames.')
    rows=[]
    train_sequences=sorted(p for p in (root/'training/frames').iterdir() if p.is_dir())
    if len(train_sequences)<2: raise SystemExit('Need at least two training sequences for sequence-level train/validation split.')
    splitter=GroupShuffleSplit(n_splits=1,test_size=a.val_fraction,random_state=a.seed)
    seq_names=[p.name for p in train_sequences]
    train_idx,val_idx=next(splitter.split(seq_names,groups=seq_names)); train_set={seq_names[i] for i in train_idx}; val_set={seq_names[i] for i in val_idx}
    for part,seq_root in [('training',root/'training/frames'),('testing',root/'testing/frames')]:
        for seq_dir in sorted(p for p in seq_root.iterdir() if p.is_dir()):
            seq=seq_dir.name
            if part=='training': split='train' if seq in train_set else 'val'; labels=None
            else:
                split='test'; label_path=root/'test_label'/f'{int(seq):03d}.npy'
                if not label_path.is_file(): raise SystemExit(f'Missing test annotation: {label_path}')
                labels=np.asarray(np.load(label_path,allow_pickle=False)).reshape(-1)
            image_files=sorted(seq_dir.glob('*.jpg'),key=lambda f:int(re.search(r'\d+',f.stem).group()) if re.search(r'\d+',f.stem) else f.stem)
            if labels is not None and len(labels)!=len(image_files): raise SystemExit(f'Label/frame count mismatch for test sequence {seq}: {len(labels)} labels vs {len(image_files)} frames')
            if labels is not None:
                print(f'Validated test seq {seq} -> test_label/{int(seq):03d}.npy: {len(image_files)} frames = {len(labels)} labels')
            for idx,f in enumerate(image_files):
                rows.append({'path':str(f.resolve()),'sequence':f'{part}/{seq}','frame_index':idx,'frame_name':f.name,'split':split,'label':int(labels[idx]) if labels is not None else (0 if part=='training' else None)})
    if not rows: raise SystemExit(f'No frame images found under {root}')
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out,index=False)
    df=pd.DataFrame(rows)
    print(f'Wrote {len(rows)} frame rows to {out}')
    print(df.groupby(['split','label'],dropna=False).size().to_string())
    print('Train/validation split is sequence-level; test labels come from matching test_label/NNN.npy arrays.')
if __name__=='__main__': main()
