"""
Shared feature engineering for all TransferIQ models.
Every model imports from here to ensure consistent transformations.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def engineer_features(df):
    """
    Takes raw master_all_seasons CSV and returns df with all engineered columns.
    No lag features. No highest_market_value_in_eur (leaky).
    """
    df = df.copy()

    # ── Target ────────────────────────────────────────────────────────────
    df['log_market_value'] = np.log1p(df['market_value_in_eur'])

    # ── Age curve ─────────────────────────────────────────────────────────
    df['age_sq']     = df['age'] ** 2
    df['prime_age']  = ((df['age'] >= 23) & (df['age'] <= 28)).astype(int)
    df['past_prime'] = (df['age'] > 29).astype(int)

    # ── Per-90 rates (Transfermarkt minutes) ──────────────────────────────
    safe90_tm = (df['minutes_total'] / 90).clip(lower=1)
    df['goals_per90']   = df['goals_total']   / safe90_tm
    df['assists_per90'] = df['assists_total']  / safe90_tm
    df['ga_per90']      = (df['goals_total'] + df['assists_total']) / safe90_tm

    # ── Per-90 rates (FBRef minutes) ──────────────────────────────────────
    safe90_fb = (df['Min_Playing'] / 90).clip(lower=1)
    df['xG_per90']       = df['xG_Expected']            / safe90_fb
    df['xAG_per90']      = df['xAG_Expected']           / safe90_fb
    df['npxG_per90']     = df['npxG_Expected']           / safe90_fb
    df['npxG_xAG_per90'] = df['npxG_plus_xAG_Expected'] / safe90_fb

    # ── Interaction & derived ─────────────────────────────────────────────
    df['intl_goal_rate']   = df['international_goals'] / df['international_caps'].clip(lower=1)
    df['rating_x_caps']    = df['Rating'] * np.log1p(df['international_caps'])
    df['ga_x_rating']      = df['ga_per90'] * df['Rating']
    df['log_transfer_fee'] = np.log1p(df['transfer_fee'])
    df['start_rate']       = df['Starts_Playing'] / df['MP_Playing'].clip(lower=1)
    df['mins_per_app']     = df['minutes_total']  / df['appearances'].clip(lower=1)

    # ── Categorical encodings ─────────────────────────────────────────────
    label_encoders = {}
    for col, src in [('League_enc', 'League'), ('sub_position_enc', 'sub_position'),
                     ('position_enc', 'position'), ('foot_enc', 'foot')]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[src].fillna('Unknown').astype(str))
        label_encoders[src] = le

    return df, label_encoders
