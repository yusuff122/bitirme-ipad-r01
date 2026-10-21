import argparse
def main():
    p=argparse.ArgumentParser(); p.add_argument('--run-dir',required=True); p.parse_args(); raise SystemExit('E1 evaluation will be enabled after the annotation-aware clip pipeline is implemented.')
if __name__=='__main__': main()
