"""
Global configuration for intraday training and execution.
"""


# Target construction

H = 90 # horizon for prediction
THR = 1e-6 # threshold for classification


# Offline training


TRAINING_MODE = "rolling"     # "rolling" or "expanding"
ROLLING_WINDOW_SIZE = 20
ADD_FAMILY_AGG = True


# Execution / trading


PRICE_COL = "P3"
TC_RATE = 1e-4


# Risk / execution defaults


MIN_HORIZON_BARS = 30

#EXCLUDED_COLUMNS = {"ts_ns", "P1", "P2", "P3", "P4", "target_reg", "target_cls"}
