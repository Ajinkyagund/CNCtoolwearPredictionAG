"""Run the whole 8-stage pipeline.

    python train.py                       # uses data/mill.mat or data/mill.csv
    python train.py --data path/to/file   # .mat or .csv
    python train.py --quick               # fast smoke test (numbers are NOT reportable)
"""
import argparse

from cncwear.config import load_config
from cncwear.pipeline import run


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="path to mill.mat or a CSV")
    ap.add_argument("--config", default=None, help="path to config.yaml (default: the one in this folder)")
    ap.add_argument("--out", default="outputs", help="folder for tables and figures")
    ap.add_argument("--models", default="models", help="folder for the saved models")
    ap.add_argument("--quick", action="store_true", help="fast smoke test with reduced settings")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, quick=args.quick)
    return run(cfg, args.data, args.out, args.models)


if __name__ == "__main__":
    main()
