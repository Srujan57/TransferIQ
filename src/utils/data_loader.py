"""
Data loading utility for TransferIQ.

Two datasets are used:
  - FINALMASTER.csv         : one row per player (latest season). Used for
                              inference, display, and held-out evaluation.
  - master_all_seasons.csv  : multi-season history (11k rows). Used for
                              model training only.
"""

import pandas as pd
import os


def load_data(path=None, test_season_cutoff=2024):
    """
    Load FINALMASTER (one row per player, latest season).
    Used for inference and Streamlit display.

    Args:
        test_season_cutoff : for evaluation, only include rows with
                             Season_End_Year >= this value (default 2024).
                             Ensures test rows are temporally after the
                             training cutoff of 2023. Pass None to load all
                             rows (used by Streamlit for full display).
    """
    if path and os.path.exists(path):
        df = pd.read_csv(path)
    else:
        candidates = [
            'data/FINALMASTER.csv',
            '../data/FINALMASTER.csv',
            'FINALMASTER.csv',
        ]
        for c in candidates:
            if os.path.exists(c):
                df = pd.read_csv(c)
                break
        else:
            raise FileNotFoundError(
                "Could not find FINALMASTER.csv. Pass the path explicitly."
            )

    if test_season_cutoff is not None:
        df = df[df['Season_End_Year'] >= test_season_cutoff].copy()

    return df


def load_training_data(path=None, season_cutoff=2023):
    """
    Load master_all_seasons (multi-season history). Used for training only.

    Args:
        season_cutoff : only include seasons <= this year (default 2023).
                        This enforces a temporal train/test split so the model
                        is never trained on the same seasons it is tested on.
                        FINALMASTER test rows are Season_End_Year == 2024, so
                        training on <=2023 guarantees zero row overlap.
    """
    if path and os.path.exists(path):
        df = pd.read_csv(path)
    else:
        candidates = [
            'data/master_all_seasons.csv',
            '../data/master_all_seasons.csv',
            'master_all_seasons.csv',
        ]
        for c in candidates:
            if os.path.exists(c):
                df = pd.read_csv(c)
                break
        else:
            raise FileNotFoundError(
                "Could not find master_all_seasons.csv. Pass the path explicitly."
            )

    if season_cutoff is not None:
        df = df[df['season_year'] <= season_cutoff].copy()

    return df
