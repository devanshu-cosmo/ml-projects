Intraday Predictive Modeling & Execution Framework

This repository contains a runnable intraday trading pipeline that loads a pretrained model, applies it to a single day of intraday data, and produces a per-day trade log with full PnL accounting.

The structure of the project is:

DevanshuSharma_project/
├── strategy.py          # Entry point (run this)
├── execution.py         # Intraday execution logic
├── model.py             # ML model wrapper
├── features.py          # Feature sanitization / engineering
├── data.py              # Data loading utilities
├── train_offline.py     # Offline training utilities (not used at runtime)
├── config.py            # Shared configuration constants
├── artifacts/
│   └── trained_pipeline.joblib
└── train/
    └── day.csv
    
Requirements:

- Python 3.9+
- Packages: Numpy, Pandas, scikit-learn, argparse, os, re, joblib
- A saved trained model at artifacts/trained_pipeline.joblib

How to run:
- From the root directory, run:
    python strategy.py \
  --input train/42.csv \
  --output outputs/trades_42.csv
  
where, 

--input: specifies the path to a single itraday file
--output: path where the offline model is trained

Output

Output is a per-day trade log CSV with the following columns for each time-stamp
- Position
- Entry Price
- Price (P3)
- Realized PnL
- MtM PnL 
- Transaction costs
- Cumulative realized PnL
- Number of trades
- Cumulative Transaction costs
- Cumulative PnL

Notes:
- Fully causal execution pipeline
- No training, forward looking targets or multi day logic at runtime
- Offline training part is for convinience but not a part of the runnable code
