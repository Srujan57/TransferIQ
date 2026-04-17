"""
Model 3 — Player Value to Parent Club (Club Utility)
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

MODEL3_FEATURES = [
    # Club utility: how AVAILABLE and RELIABLE is this player?
    # Playing time, discipline, squad context — no performance quality stats.
    'age', 'age_sq', 'prime_age', 'past_prime',
    'contract_years_remaining',
    'tm_appearances', 'tm_minutes', 'tm_goals', 'tm_assists',
    'MP_Playing', 'Starts_Playing', 'Min_Playing',
    'start_rate', 'mins_per_app',
    'tm_yellows', 'tm_reds', 'CrdY', 'CrdR',
    'squad_size',
    'league_ranking', 'League_enc',
    'sub_position_enc', 'position_enc',
    'log_highest_mv', 'prev_market_value',
]

TARGET = 'log_market_value'


def build_model3():
    """Return an untrained Model 3 estimator."""
    return HistGradientBoostingRegressor(
        max_iter=800, max_depth=7, learning_rate=0.04,
        min_samples_leaf=15, l2_regularization=0.5, max_bins=255,
        random_state=42, early_stopping=True,
        n_iter_no_change=50, validation_fraction=0.1,
    )


def train_model3(df):
    """
    Train Model 3 on the full dataset. Returns (model, features, target, X_test, y_test).
    """
    from sklearn.model_selection import train_test_split

    X = df[MODEL3_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_model3()
    model.fit(X_train, y_train)
    return model, MODEL3_FEATURES, TARGET, X_test, y_test


def predict_model3(model, df):
    """Return log-scale predictions."""
    return model.predict(df[MODEL3_FEATURES])
