"""E1 scaffold. Dataset manifest and installed pretrained 3D Swin are required."""
import argparse

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/video_swin.yaml'); p.add_argument('--manifest',default='data/processed/r01_manifest.csv'); args=p.parse_args()
    raise SystemExit('E1 scaffold is ready; implement annotation-aware clip extraction after inspect_dataset.py confirms the R01 layout. No training was started.')
if __name__=='__main__': main()
