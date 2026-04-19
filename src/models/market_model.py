"""
Model 2 — Market Perception (Prestige, History, Market Context)

Shared foundation features + market-specific extensions.
No on-pitch performance stats — those belong to Model 2b.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

MODEL2_FEATURES = [
    # ── Foundation (shared with all 3 models) ─────────────────────────────
    'age', 'age_sq', 'prime_age', 'past_prime',
    'sub_position_enc', 'position_enc',
    'log_highest_mv', 'prev_market_value', 'career_avg_rating',
    'Rating', 'contract_years_remaining',
    # ── Market extensions ─────────────────────────────────────────────────
    'height_in_cm', 'foot_enc',
    'international_caps', 'international_goals', 'intl_goal_rate', 'rating_x_caps',
    'league_ranking', 'League_enc', 'squad_size',
    'has_league_rating', 'has_fbref_data',
    'log_transfer_fee', 'mv_growth_ratio',
]

TARGET = 'log_market_value'


def build_model2():
    """Return an untrained Model 2 estimator."""
    return HistGradientBoostingRegressor(
        max_iter=1500, max_depth=6, learning_rate=0.03,
        min_samples_leaf=15, l2_regularization=0.5, max_bins=255,
        random_state=42, early_stopping=True,
        n_iter_no_change=50, validation_fraction=0.1,
    )


def train_model2(df):
    """
    Train Model 2 on the full dataset. Returns (model, features, target).
    df must already have engineered features from feature_engineering.py.
    """
    from sklearn.model_selection import train_test_split

    X = df[MODEL2_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_model2()
    model.fit(X_train, y_train)
    return model, MODEL2_FEATURES, TARGET, X_test, y_test


def predict_model2(model, df):
    """Return log-scale predictions."""
    return model.predict(df[MODEL2_FEATURES])
