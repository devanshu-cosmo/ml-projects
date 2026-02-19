"""
Entry point for intraday trading strategy execution.

Loads a pretrained model pipeline, applies it to a single
intraday CSV file, and outputs a per-day trade log.
"""

import argparse
import os

from data import load_day_csv
from execution import run_execution_for_day
from train_offline import load_trained_pipeline
from config import H, THR, TC_RATE


# Configuration


DEFAULT_PIPELINE_PATH = os.path.join(
    "artifacts", "trained_pipeline.joblib"
)

PRICE_COL = "P3"
#TC_RATE = 1e-4
#THRESHOLD = 0.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run intraday trading strategy on one day of data"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file (single trading day)",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to output trade log CSV",
    )

    parser.add_argument(
        "--pipeline",
        default=DEFAULT_PIPELINE_PATH,
        help="Path to pretrained pipeline artifact",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    
    # Load trained pipeline
    
    if not os.path.exists(args.pipeline):
        raise FileNotFoundError(
            f"Trained pipeline not found at {args.pipeline}. "
            "Run offline training first."
        )

    model, sanitizer, feature_engineer = load_trained_pipeline(
        args.pipeline
    )

    
    # Load input data
    
    df_day = load_day_csv(args.input)

    
    # Run execution
    
    trades_df = run_execution_for_day(
        df_day_raw=df_day,
        sanitizer=sanitizer,
        feature_engineer=feature_engineer,
        model=model,
        price_col=PRICE_COL,
        threshold=THR,
        tc_rate=TC_RATE,
        use_min_hold=True,
        use_flip_confirm=True,
        force_flat_before_flip=True,
    )

    
    # Save output
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    trades_df.to_csv(args.output, index=False)

    print(
        f"Execution completed. Final PnL = "
        f"{trades_df['cum_pnl'].iloc[-1]:.4f}"
    )


if __name__ == "__main__":
    main()
