import argparse, zipfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description='Extract only R01_conveyer_belt_real from the IPAD archive.'); p.add_argument('--zip',required=True); p.add_argument('--destination',default='data/r01'); a=p.parse_args()
    archive=Path(a.zip)
    if not archive.is_file(): raise SystemExit(f'ZIP not found: {archive}')
    target='R01_conveyer_belt_real/'; dest=Path(a.destination); dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        members=[m for m in z.infolist() if target in m.filename and not Path(m.filename).is_absolute() and '..' not in Path(m.filename).parts]
        if not members: raise SystemExit(f'{target} not found in {archive}')
        for m in members: z.extract(m,dest)
    print(f'Extracted {len(members)} entries to {dest}')
if __name__=='__main__': main()
