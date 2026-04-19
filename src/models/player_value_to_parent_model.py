"""
Model 3 — Club Utility (Availability, Playing Time, Discipline)

Shared foundation features + availability/reliability extensions.
No performance quality stats — those belong to Model 2b.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

MODEL3_FEATURES = [
    # ── Foundation (shared with all 3 models) ─────────────────────────────
    'age', 'age_sq', 'prime_age', 'past_prime',
    'sub_position_enc', 'position_enc',
    'log_highest_mv', 'prev_market_value', 'career_avg_rating',
    'Rating', 'contract_years_remaining',
    # ── Utility extensions ────────────────────────────────────────────────
    'tm_appearances', 'tm_minutes', 'tm_goals', 'tm_assists',
    'MP_Playing', 'Starts_Playing', 'Min_Playing',
    'start_rate', 'mins_per_app',
    'tm_yellows', 'tm_reds', 'CrdY', 'CrdR',
    'squad_size', 'league_ranking', 'League_enc',
]

TARGET = 'log_market_value'


def build_model3():
    """Return an untrained Model 3 estimator."""
    return HistGradientBoostingRegressor(
        max_iter=1500, max_depth=6, learning_rate=0.03,
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
