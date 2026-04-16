"""
Model 1 — Position-Split Value Model (Inherent Ability by Position)
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

MODEL2B_FEATURES = [
    'age', 'age_sq', 'prime_age', 'past_prime',
    'height_in_cm', 'foot_enc',
    'contract_years_remaining',
    'international_caps', 'international_goals', 'intl_goal_rate',
    'rating_x_caps',
    'league_ranking', 'League_enc', 'sub_position_enc',
    'has_league_rating', 'has_fbref_data', 'squad_size',
    'log_transfer_fee', 'Rating',
    'appearances', 'minutes_total', 'goals_total', 'assists_total',
    'goals_per90', 'assists_per90', 'ga_per90',
    'xG_per90', 'xAG_per90', 'npxG_per90',
    'xG_Expected', 'xAG_Expected', 'npxG_plus_xAG_Expected',
    'Successful_Dribbles', 'Big_Chances_Created', 'Possession_Lost',
    'PrgC_Progression', 'PrgP_Progression',
    'MP_Playing', 'Starts_Playing',
    'ga_x_rating', 'start_rate', 'mins_per_app',
    'yellow_cards_total',
]


def build_position_model():
    """Return an untrained per-position estimator."""
    return HistGradientBoostingRegressor(
        max_iter=800, max_depth=9, learning_rate=0.04,
        min_samples_leaf=12, l2_regularization=0.3, max_bins=255,
        random_state=42, early_stopping=True,
        n_iter_no_change=50, validation_fraction=0.1,
    )


def train_model2b(df):
    """
    Train per-position models with player-level splits.
    Returns (dict of models, features, results_dict).
    """
    df_m = df.copy()
    df_m[MODEL2B_FEATURES] = df_m[MODEL2B_FEATURES].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(-1)

    position_models = {}
    results = {}
    all_y_test, all_y_pred = [], []

    for pos in sorted(df_m['position'].unique()):
        pos_players = df_m[df_m['position'] == pos]['player_id'].unique()
        if len(pos_players) < 20:
            continue

        df_pos = df_m[df_m['player_id'].isin(pos_players)].copy()

        np.random.seed(42)
        test_players = np.random.choice(
            pos_players, size=int(len(pos_players) * 0.2), replace=False
        )
        test_mask = df_pos['player_id'].isin(test_players)

        X_train = df_pos.loc[~test_mask, MODEL2B_FEATURES]
        X_test  = df_pos.loc[test_mask,  MODEL2B_FEATURES]
        y_train = np.log(df_pos.loc[~test_mask, 'market_value_in_eur'].clip(lower=1))
        y_test  = np.log(df_pos.loc[test_mask,  'market_value_in_eur'].clip(lower=1))

        model = build_position_model()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)

        position_models[pos] = model
        results[pos] = {'r2': r2, 'n_test': len(X_test)}

        all_y_test.extend(y_test.tolist())
        all_y_pred.extend(preds.tolist())

    results['AGGREGATE'] = {'r2': r2_score(all_y_test, all_y_pred)}
    return position_models, MODEL2B_FEATURES, results


def predict_model2b(position_models, df):
    """
    Predict using the correct position model per row.
    Returns log-scale predictions (natural log, not log1p).
    """
    df_m = df.copy()
    df_m[MODEL2B_FEATURES] = df_m[MODEL2B_FEATURES].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(-1)

    preds = np.zeros(len(df_m))
    for pos, model in position_models.items():
        mask = df_m['position'] == pos
        if mask.any():
            preds[mask] = model.predict(df_m.loc[mask, MODEL2B_FEATURES])
    return preds
