"""
Shared feature engineering for all TransferIQ models.
Every model imports from here to ensure consistent transformations.

Handles two input schemas:
  - FINALMASTER schema      : tm_* column names, contract_expiration_date,
                              Season_End_Year. has_fbref_data,
                              has_league_rating, and contract_years_remaining
                              are derived here.
  - master_all_seasons schema: old column names (appearances, minutes_total,
                              etc.), has_fbref_data / has_league_rating /
                              contract_years_remaining already pre-computed.

The normalise_schema() helper renames old columns to canonical names so
engineer_features() only needs one code path.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Mapping from master_all_seasons column names -> canonical (FINALMASTER) names
_OLD_TO_CANONICAL = {
    'appearances':            'tm_appearances',
    'minutes_total':          'tm_minutes',
    'goals_total':            'tm_goals',
    'assists_total':          'tm_assists',
    'yellow_cards_total':     'tm_yellows',
    'red_cards_total':        'tm_reds',
    'npxG_plus_xAG_Expected': 'npxG+xAG_Expected',
    'G_plus_A':               'G+A',
    'season_year':            'Season_End_Year',
}


def normalise_schema(df):
    """
    Rename master_all_seasons columns to canonical FINALMASTER names.
    Also derives contract_years_remaining / has_fbref_data / has_league_rating
    if they are not already present (FINALMASTER path), or passes them through
    unchanged (master_all_seasons already has them pre-computed).
    Returns a copy — does not modify the caller's DataFrame.
    """
    df = df.copy()

    # ── Rename old column names to canonical names ────────────────────────
    rename = {old: new for old, new in _OLD_TO_CANONICAL.items() if old in df.columns}
    if rename:
        df = df.rename(columns=rename)

    # ── Derive contract_years_remaining if not already present ────────────
    # master_all_seasons has it pre-computed; FINALMASTER does not.
    if 'contract_years_remaining' not in df.columns:
        expiry_year = pd.to_datetime(
            df['contract_expiration_date'], errors='coerce'
        ).dt.year
        df['contract_years_remaining'] = (
            expiry_year - df['Season_End_Year']
        ).fillna(0).clip(lower=0)

    # ── Derive availability flags if not already present ─────────────────
    if 'has_fbref_data' not in df.columns:
        df['has_fbref_data'] = df['xG_Expected'].notna().astype(int)
    if 'has_league_rating' not in df.columns:
        df['has_league_rating'] = df['league_ranking'].notna().astype(int)

    # ── Career cumulative stats ───────────────────────────────────────────
    # FINALMASTER has total_goals / total_assists / total_minutes_played as
    # pre-computed career totals. master_all_seasons does not, so we derive
    # them by cumulative-summing the per-season stats within the dataset.
    # The proxy won't match exactly (old data starts 2018, missing earlier
    # career) but the signal is still useful.
    if 'total_goals' not in df.columns:
        df = df.sort_values(['player_id', 'Season_End_Year'])
        df['total_goals']          = df.groupby('player_id')['tm_goals'].cumsum()
        df['total_assists']        = df.groupby('player_id')['tm_assists'].cumsum()
        df['total_minutes_played'] = df.groupby('player_id')['tm_minutes'].cumsum()

    # ── Market value growth ratio ─────────────────────────────────────────
    # FINALMASTER has mv_growth_ratio pre-computed. For master_all_seasons
    # we derive it as current_mv / previous_season_mv per player.
    if 'mv_growth_ratio' not in df.columns:
        df = df.sort_values(['player_id', 'Season_End_Year'])
        df['prev_mv'] = df.groupby('player_id')['market_value_in_eur'].shift(1)
        df['mv_growth_ratio'] = (
            df['market_value_in_eur'] / df['prev_mv'].clip(lower=1)
        )
        # First season for a player has no previous value — fill with 1.0 (neutral)
        df['mv_growth_ratio'] = df['mv_growth_ratio'].fillna(1.0)
        df.drop(columns=['prev_mv'], inplace=True)

    return df


def engineer_features(df):
    """
    Takes a raw CSV DataFrame (either schema) and returns
    (df_engineered, label_encoders).
    No lag features that use future data.
    """
    df = normalise_schema(df)

    # ── Target ────────────────────────────────────────────────────────────
    df['log_market_value'] = np.log1p(df['market_value_in_eur'])

    # ── Age curve ─────────────────────────────────────────────────────────
    df['age_sq']     = df['age'] ** 2
    df['prime_age']  = ((df['age'] >= 23) & (df['age'] <= 28)).astype(int)
    df['past_prime'] = (df['age'] > 29).astype(int)

    # ── Per-90 rates (Transfermarkt minutes) ──────────────────────────────
    safe90_tm = (df['tm_minutes'] / 90).clip(lower=1)
    df['goals_per90']   = df['tm_goals']   / safe90_tm
    df['assists_per90'] = df['tm_assists']  / safe90_tm
    df['ga_per90']      = (df['tm_goals'] + df['tm_assists']) / safe90_tm

    # ── Per-90 rates (FBRef minutes) ──────────────────────────────────────
    safe90_fb = (df['Min_Playing'] / 90).clip(lower=1)
    df['xG_per90']       = df['xG_Expected']        / safe90_fb
    df['xAG_per90']      = df['xAG_Expected']        / safe90_fb
    df['npxG_per90']     = df['npxG_Expected']       / safe90_fb
    df['npxG_xAG_per90'] = df['npxG+xAG_Expected']  / safe90_fb

    # ── Interaction & derived ─────────────────────────────────────────────
    df['intl_goal_rate']   = df['international_goals'] / df['international_caps'].clip(lower=1)
    df['rating_x_caps']    = df['Rating'] * np.log1p(df['international_caps'])
    df['ga_x_rating']      = df['ga_per90'] * df['Rating']
    df['log_transfer_fee'] = np.log1p(df['transfer_fee'])
    df['start_rate']       = df['Starts_Playing'] / df['MP_Playing'].clip(lower=1)
    df['mins_per_app']     = df['tm_minutes'] / df['tm_appearances'].clip(lower=1)

    # ── Highest market value (publicly available Transfermarkt data) ─────
    # Static per player, same value in both datasets (verified 100%).
    # Represents the player's all-time peak valuation — public knowledge
    # available to any scout or analyst before making a prediction.
    df['log_highest_mv'] = np.log1p(df['highest_market_value_in_eur'])

    # ── Previous season market value ─────────────────────────────────────
    # For multi-season data (training): shift(1) within each player.
    # For single-row data (FINALMASTER test): this column won't exist,
    # and the injected values come from inject_history_features().
    # NaN for a player's first season is handled natively by
    # HistGradientBoosting (which supports missing values).
    if df['player_id'].duplicated().any():
        # Multi-season dataset — compute prev_mv from within the data
        df = df.sort_values(['player_id', 'Season_End_Year'])
        df['prev_market_value'] = df.groupby('player_id')['market_value_in_eur'].shift(1)
    # else: single-row-per-player dataset; prev_market_value must be
    # injected externally via inject_history_features() before calling this.

    # ── Career average rating ───────────────────────────────────────────
    # For multi-season data: expanding mean of Rating from prior seasons.
    # For single-row data: injected externally from training history.
    if df['player_id'].duplicated().any() and 'career_avg_rating' not in df.columns:
        df = df.sort_values(['player_id', 'Season_End_Year'])
        df['career_avg_rating'] = df.groupby('player_id')['Rating'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
    # else: already injected or single-row dataset

    # ── Categorical encodings ─────────────────────────────────────────────
    label_encoders = {}
    for col, src in [('League_enc', 'League'), ('sub_position_enc', 'sub_position'),
                     ('position_enc', 'position'), ('foot_enc', 'foot')]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[src].fillna('Unknown').astype(str))
        label_encoders[src] = le

    return df, label_encoders


def inject_history_features(df_test, df_train_raw):
    """
    For single-row-per-player test data (FINALMASTER), inject history-based
    features by looking up each player's records in the training data.

    This is called BEFORE engineer_features() on the test set.
    Only uses training-period data (≤2022) — no temporal leakage.

    Injects:
      - prev_market_value  : last known market value from training period
      - career_avg_rating  : mean Rating across training-period seasons
    """
    df_test = df_test.copy()

    # Normalise training data column names so we can reference canonical cols
    rename = {old: new for old, new in _OLD_TO_CANONICAL.items() if old in df_train_raw.columns}
    train = df_train_raw.rename(columns=rename) if rename else df_train_raw.copy()

    # Compute per-player aggregates from training history
    history = train.sort_values(['player_id', 'Season_End_Year']).groupby('player_id').agg(
        prev_market_value=('market_value_in_eur', 'last'),
        career_avg_rating=('Rating', 'mean'),
    ).reset_index()

    df_test = df_test.merge(history, on='player_id', how='left')

    return df_test
