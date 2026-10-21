import argparse
def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/temporal.yaml'); p.parse_args(); raise SystemExit('E2 temporal training scaffold; E0/E1 features and sequence-aware labels are required.')
if __name__=='__main__': main()
