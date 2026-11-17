"""FareScope pipeline: download -> clean -> cluster -> train -> evaluate.

    python run_pipeline.py            # reuse downloaded data if present
    python run_pipeline.py --refresh  # re-download from DOT and FRED
"""
import argparse

from src import clean, download, train

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refresh", action="store_true", help="re-download the raw data")
    args = p.parse_args()

    download.main(args.refresh)
    clean.main()
    train.main()  # runs clustering (Model 1), then trains and evaluates XGBoost (Model 2)
