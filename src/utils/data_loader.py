"""
Data loading utility for TransferIQ.
"""

import pandas as pd
import os


def load_data(path=None):
    """
    Load the master dataset. Tries common locations if path is None.
    """
    if path and os.path.exists(path):
        return pd.read_csv(path)

    candidates = [
        'data/master_all_seasons.csv',
        '../data/master_all_seasons.csv',
        'master_all_seasons.csv',
        'master_all_seasons__1_.csv',
    ]
    for c in candidates:
        if os.path.exists(c):
            return pd.read_csv(c)

    raise FileNotFoundError(
        "Could not find master_all_seasons.csv. Pass the path explicitly."
    )
