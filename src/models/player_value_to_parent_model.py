"""
Model 3 — Player Value to Parent Club (Club Utility)
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

MODEL3_FEATURES = [
    'age', 'age_sq', 'prime_age', 'past_prime',
    'contract_years_remaining',
    'appearances', 'minutes_total',
    'MP_Playing', 'Starts_Playing',
    'start_rate', 'mins_per_app',
    'goals_total', 'assists_total',
    'goals_per90', 'assists_per90', 'ga_per90',
    'xG_per90', 'xAG_per90', 'npxG_per90', 'npxG_xAG_per90',
    'xG_Expected', 'xAG_Expected', 'npxG_plus_xAG_Expected',
    'Successful_Dribbles', 'Big_Chances_Created', 'Possession_Lost',
    'PrgC_Progression', 'PrgP_Progression',
    'Rating', 'ga_x_rating',
    'yellow_cards_total', 'red_cards_total',
    'squad_size',
    'league_ranking', 'League_enc',
    'sub_position_enc', 'position_enc',
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
