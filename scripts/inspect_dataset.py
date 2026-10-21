import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from src.data.ipad_r01_dataset import list_frames

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',default='data/r01'); a=p.parse_args(); root=Path(a.root)
    if not root.exists(): raise SystemExit(f'Dataset path not found: {root.resolve()}')
    files=list(root.rglob('*')); frames=list_frames(root); annotations=[x for x in files if x.is_file() and x.suffix.lower() in {'.csv','.tsv','.json','.txt','.xml','.npy'}]
    print(f'Root: {root.resolve()}\nFiles: {sum(x.is_file() for x in files)}\nFrame images: {len(frames)}\nAnnotation candidates: {len(annotations)}')
    print('\nDirectories:');
    for x in sorted(y for y in root.rglob('*') if y.is_dir()): print('  ',x.relative_to(root))
    print('\nAnnotation candidates:');
    for x in annotations: print('  ',x.relative_to(root))
    print('\nFrame samples:');
    for x in frames[:8]: print('  ',x.relative_to(root))
    sizes={}
    for f in frames:
        im=cv2.imread(str(f))
        if im is not None: sizes[tuple(im.shape[1::-1])]=sizes.get(tuple(im.shape[1::-1]),0)+1
    print(f'Frame resolutions (width x height: count): {sizes}')
    for part in ('training','testing'):
        folder=root/part/'frames'
        if folder.is_dir():
            seqs=sorted(p for p in folder.iterdir() if p.is_dir())
            print(f'{part}: {len(seqs)} sequences, {sum(len(list(s.glob("*.jpg"))) for s in seqs)} jpg frames')
            for seq in seqs: print(f'  sequence {seq.name}: {len(list(seq.glob("*.jpg")))} frames')
    label_dir=root/'test_label'
    if label_dir.is_dir():
        total=normal=anomaly=0
        for label_file in sorted(label_dir.glob('*.npy')):
            arr=np.load(label_file,allow_pickle=False).reshape(-1)
            vals,counts=np.unique(arr,return_counts=True)
            print(f'label {label_file.name}: length={len(arr)}, values={dict(zip(vals.tolist(),counts.tolist()))}')
            total+=len(arr); normal+=int((arr==0).sum()); anomaly+=int((arr==1).sum())
        print(f'Test labels total={total}, normal={normal}, anomaly={anomaly}; convention in files: 0=normal, 1=anomaly')
    print('\nSequence identifiers are the numeric directory names; test_label 001.npy maps to testing/frames/01.')
if __name__=='__main__': main()
