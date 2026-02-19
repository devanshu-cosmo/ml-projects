#DATA_DIR = "train"  # adjust path if needed

#def load_day_csv(day_id, data_dir=DATA_DIR):
    #"""
    #Load one intraday file (one trading day), sort by ts_ns, and reset index.
    
    #Parameters
    #----------
    #day_id : int or str
        #File identifier, e.g. 33 for '33.csv'.
    #data_dir : str
        #Directory where the CSV files live.
        
    #Returns
    #-------
    #df : pd.DataFrame
        #Sorted DataFrame with index 0..n-1 and original columns.
    #"""
    #fname = f"{day_id}.csv"
    #path = os.path.join(data_dir, fname)
    #df = pd.read_csv(path)
    
    ## Sort strictly by ts_ns; if ts_ns is missing, this will raise a KeyError (good early check)
    #df = df.sort_values("ts_ns").reset_index(drop=True)
    #return df
    
import os
import pandas as pd
import re

def load_day_csv(path):
    """
    Load a single intraday CSV file.

    Parameters
    ----------
    path : str
        Path to CSV file for one trading day.

    Returns
    -------
    pd.DataFrame
        DataFrame sorted by timestamp with a clean index.
    """
    df = pd.read_csv(path)

    if "ts_ns" not in df.columns:
        raise ValueError("Required column 'ts_ns' not found in input CSV")

    df = df.sort_values("ts_ns").reset_index(drop=True)
    return df


def list_day_ids(data_dir, pattern=r"(\d+)\.csv"):
    """
    List integer day IDs available in a directory.

    Example:
        '33.csv' -> day_id = 33
    """
    day_ids = []
    for fname in os.listdir(data_dir):
        m = re.match(pattern, fname)
        if m:
            day_ids.append(int(m.group(1)))
    return sorted(day_ids) 
