"""
Model 2 — Player Value on Market (Prestige & Market Perception)
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

MODEL2_FEATURES = [
    'age', 'age_sq', 'prime_age', 'past_prime',
    'height_in_cm', 'foot_enc',
    'contract_years_remaining',
    'international_caps', 'international_goals', 'intl_goal_rate',
    'rating_x_caps',
    'league_ranking', 'League_enc', 'sub_position_enc', 'position_enc',
    'squad_size', 'has_league_rating', 'has_fbref_data',
    'log_transfer_fee', 'Rating',
    'appearances', 'minutes_total', 'goals_total', 'assists_total',
    'goals_per90', 'assists_per90', 'ga_per90',
    'xG_per90', 'xAG_per90',
    'Successful_Dribbles', 'Big_Chances_Created',
    'PrgC_Progression', 'PrgP_Progression', 'Possession_Lost',
    'ga_x_rating', 'start_rate', 'mins_per_app',
    'yellow_cards_total', 'npxG_plus_xAG_Expected', 'G_plus_A',
]

TARGET = 'log_market_value'


def build_model2():
    """Return an untrained Model 2 estimator."""
    return HistGradientBoostingRegressor(
        max_iter=1000, max_depth=8, learning_rate=0.04,
        min_samples_leaf=12, l2_regularization=0.4, max_bins=255,
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
