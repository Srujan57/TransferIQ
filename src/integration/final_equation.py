"""
TransferIQ — Final Valuation Equation

Combines three sub-models into a single predicted market value:
  Model 2:  Market Perception (prestige, reputation, league context)
  Model 2b: Inherent Ability  (position-specific talent assessment)
  Model 3:  Club Utility      (playing time, tactical contribution)

The final prediction is a learned weighted average via Ridge regression
on the three sub-model outputs, trained on held-out data to find optimal
weights without overfitting.

Run this file directly to train everything, evaluate, and save artifacts.
"""

import sys
import os
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error

# ── Add project root to path ─────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.feature_engineering import engineer_features
from src.utils.data_loader import load_data
from src.utils.metrics import compute_metrics, tier_analysis, print_metrics
from src.models.market_model import (
    MODEL2_FEATURES, build_model2, predict_model2,
)
from src.models.inherent_ability_model import (
    MODEL2B_FEATURES, build_position_model, predict_model2b,
)
from src.models.player_value_to_parent_model import (
    MODEL3_FEATURES, build_model3, predict_model3,
)


# ═════════════════════════════════════════════════════════════════════════════
# COMBINED EQUATION
# ═════════════════════════════════════════════════════════════════════════════

class TransferIQValuation:
    """
    Three sub-models → Ridge meta-learner → final EUR prediction.

    The meta-learner learns optimal weights for each sub-model's prediction.
    This is better than hand-tuning weights because it adapts to how each
    model's errors correlate.
    """

    def __init__(self):
        self.model2 = None           # Market perception
        self.position_models = None  # Inherent ability (per-position)
        self.model3 = None           # Club utility
        self.meta_learner = None     # Ridge combining the three
        self.label_encoders = None
        self.metrics = {}

    def train(self, df_raw, data_path=None):
        """
        Full training pipeline:
        1. Engineer features
        2. Split into train/test at the PLAYER level
        3. Train each sub-model on training players
        4. Generate sub-model predictions on test players
        5. Train Ridge meta-learner on sub-model outputs
        6. Evaluate
        """
        print("=" * 60)
        print("TransferIQ — Training Full Valuation Pipeline")
        print("=" * 60)

        # ── Feature engineering ───────────────────────────────────────────
        print("\n[1/5] Engineering features...")
        df, self.label_encoders = engineer_features(df_raw)
        print(f"  Dataset: {len(df):,} rows, {df.shape[1]} columns")

        # ── Player-level train/test split ─────────────────────────────────
        print("\n[2/5] Player-level train/test split...")
        all_players = df['player_id'].unique()
        np.random.seed(42)
        test_players = np.random.choice(
            all_players, size=int(len(all_players) * 0.2), replace=False
        )
        test_mask = df['player_id'].isin(test_players)
        df_train = df[~test_mask].copy()
        df_test  = df[test_mask].copy()
        print(f"  Train: {len(df_train):,} rows ({(~test_mask).sum()} rows, "
              f"{len(all_players) - len(test_players)} players)")
        print(f"  Test:  {len(df_test):,} rows ({test_mask.sum()} rows, "
              f"{len(test_players)} players)")

        # ── Train sub-models ──────────────────────────────────────────────
        print("\n[3/5] Training sub-models...")

        # Model 2: Market Perception
        print("  Training Model 2 (Market Perception)...")
        self.model2 = build_model2()
        self.model2.fit(df_train[MODEL2_FEATURES], df_train['log_market_value'])
        m2_train_pred = predict_model2(self.model2, df_train)
        m2_test_pred  = predict_model2(self.model2, df_test)
        m2_r2 = r2_score(df_test['log_market_value'], m2_test_pred)
        print(f"    R² = {m2_r2:.4f}")

        # Model 2b: Inherent Ability (per-position)
        print("  Training Model 2b (Inherent Ability, per-position)...")
        df_train_clean = df_train.copy()
        df_train_clean[MODEL2B_FEATURES] = df_train_clean[MODEL2B_FEATURES].replace(
            [np.inf, -np.inf], np.nan
        ).fillna(-1)
        df_test_clean = df_test.copy()
        df_test_clean[MODEL2B_FEATURES] = df_test_clean[MODEL2B_FEATURES].replace(
            [np.inf, -np.inf], np.nan
        ).fillna(-1)

        self.position_models = {}
        for pos in sorted(df_train['position'].unique()):
            pos_mask_tr = df_train_clean['position'] == pos
            pos_mask_te = df_test_clean['position'] == pos
            if pos_mask_tr.sum() < 20:
                continue
            m = build_position_model()
            y_tr = np.log(df_train_clean.loc[pos_mask_tr, 'market_value_in_eur'].clip(lower=1))
            m.fit(df_train_clean.loc[pos_mask_tr, MODEL2B_FEATURES], y_tr)
            self.position_models[pos] = m

        m2b_test_pred_log = predict_model2b(self.position_models, df_test)
        # Convert Model 2b from ln to log1p scale for consistency
        m2b_test_pred = np.log1p(np.exp(m2b_test_pred_log))
        m2b_train_pred_log = predict_model2b(self.position_models, df_train)
        m2b_train_pred = np.log1p(np.exp(m2b_train_pred_log))
        m2b_r2 = r2_score(df_test['log_market_value'], m2b_test_pred)
        print(f"    Aggregate R² = {m2b_r2:.4f}")

        # Model 3: Club Utility
        print("  Training Model 3 (Club Utility)...")
        self.model3 = build_model3()
        self.model3.fit(df_train[MODEL3_FEATURES], df_train['log_market_value'])
        m3_train_pred = predict_model3(self.model3, df_train)
        m3_test_pred  = predict_model3(self.model3, df_test)
        m3_r2 = r2_score(df_test['log_market_value'], m3_test_pred)
        print(f"    R² = {m3_r2:.4f}")

        # ── Train meta-learner ────────────────────────────────────────────
        print("\n[4/5] Training meta-learner (Ridge)...")
        X_meta_train = np.column_stack([m2_train_pred, m2b_train_pred, m3_train_pred])
        X_meta_test  = np.column_stack([m2_test_pred,  m2b_test_pred,  m3_test_pred])
        y_meta_train = df_train['log_market_value'].values
        y_meta_test  = df_test['log_market_value'].values

        self.meta_learner = Ridge(alpha=1.0)
        self.meta_learner.fit(X_meta_train, y_meta_train)

        weights = self.meta_learner.coef_
        print(f"  Learned weights:")
        print(f"    Model 2  (Market):    {weights[0]:.4f}")
        print(f"    Model 2b (Ability):   {weights[1]:.4f}")
        print(f"    Model 3  (Utility):   {weights[2]:.4f}")
        print(f"    Intercept:            {self.meta_learner.intercept_:.4f}")

        # ── Evaluate combined model ───────────────────────────────────────
        print("\n[5/5] Evaluating combined model...")
        final_pred_log = self.meta_learner.predict(X_meta_test)
        final_pred_eur = np.expm1(final_pred_log)
        actual_eur     = np.expm1(y_meta_test)

        self.metrics = compute_metrics(y_meta_test, final_pred_log)
        self.metrics['sub_model_r2'] = {
            'model2_market': m2_r2,
            'model2b_ability': m2b_r2,
            'model3_utility': m3_r2,
        }
        self.metrics['meta_weights'] = {
            'model2': float(weights[0]),
            'model2b': float(weights[1]),
            'model3': float(weights[2]),
            'intercept': float(self.meta_learner.intercept_),
        }

        print_metrics(self.metrics, "Combined Model — Test Set")

        # Tier analysis
        tiers = tier_analysis(actual_eur, final_pred_eur)
        print("\n  Error by Value Tier:")
        print(tiers.to_string())

        # Comparison table
        print(f"\n  {'─' * 55}")
        print(f"  {'Model':<30} {'R²':>8}")
        print(f"  {'─' * 55}")
        print(f"  {'Model 2  (Market Perception)':<30} {m2_r2:>8.4f}")
        print(f"  {'Model 2b (Inherent Ability)':<30} {m2b_r2:>8.4f}")
        print(f"  {'Model 3  (Club Utility)':<30} {m3_r2:>8.4f}")
        print(f"  {'─' * 55}")
        print(f"  {'COMBINED (Ridge Meta)':<30} {self.metrics['r2']:>8.4f}")
        print(f"  {'─' * 55}")

        # Store test data for later use
        self._df_test = df_test
        self._final_pred_log = final_pred_log
        self._final_pred_eur = final_pred_eur
        self._actual_eur = actual_eur

        return self.metrics

    def predict(self, df_raw):
        """
        Predict market values for new data.
        Input: raw DataFrame (pre-feature-engineering).
        Output: array of EUR predictions.
        """
        df, _ = engineer_features(df_raw)

        m2_pred  = predict_model2(self.model2, df)
        m2b_pred_log = predict_model2b(self.position_models, df)
        m2b_pred = np.log1p(np.exp(m2b_pred_log))
        m3_pred  = predict_model3(self.model3, df)

        X_meta = np.column_stack([m2_pred, m2b_pred, m3_pred])
        final_log = self.meta_learner.predict(X_meta)
        return np.expm1(final_log)

    def predict_decomposed(self, df_raw):
        """
        Return a DataFrame with each sub-model's prediction + final.
        Useful for the Streamlit dashboard.
        """
        df, _ = engineer_features(df_raw)

        m2_pred  = predict_model2(self.model2, df)
        m2b_pred_log = predict_model2b(self.position_models, df)
        m2b_pred = np.log1p(np.exp(m2b_pred_log))
        m3_pred  = predict_model3(self.model3, df)

        X_meta = np.column_stack([m2_pred, m2b_pred, m3_pred])
        final_log = self.meta_learner.predict(X_meta)

        pass_cols = ['player_name', 'market_value_in_eur', 'age', 'position',
                    'sub_position', 'League', 'Rating', 'current_club_name']
        if 'season_year' in df.columns:
            pass_cols.append('season_year')
        result = df[pass_cols].copy()
        result['pred_market_perception'] = np.expm1(m2_pred)
        result['pred_inherent_ability']  = np.exp(m2b_pred_log)  # Model 2b uses ln
        result['pred_club_utility']      = np.expm1(m3_pred)
        result['pred_combined']          = np.expm1(final_log)
        result['actual_value']           = df['market_value_in_eur'].values
        result['error_eur']              = result['pred_combined'] - result['actual_value']
        result['error_pct']              = (
            np.abs(result['error_eur']) / result['actual_value'].clip(lower=1) * 100
        ).round(1)
        # Positive = model thinks player is UNDERVALUED by market
        result['value_gap']              = result['pred_combined'] - result['actual_value']
        result['value_gap_pct']          = (
            result['value_gap'] / result['actual_value'].clip(lower=1) * 100
        ).round(1)

        return result

    def get_undervalued(self, df_raw, min_value=1_000_000, top_n=20):
        """
        Find players the model thinks are most undervalued.
        Returns top_n players where pred >> actual.
        """
        result = self.predict_decomposed(df_raw)
        filtered = result[result['actual_value'] >= min_value]
        return filtered.nlargest(top_n, 'value_gap_pct')

    def get_overvalued(self, df_raw, min_value=1_000_000, top_n=20):
        """
        Find players the model thinks are most overvalued.
        """
        result = self.predict_decomposed(df_raw)
        filtered = result[result['actual_value'] >= min_value]
        return filtered.nsmallest(top_n, 'value_gap_pct')

    def save(self, path='models/transferiq_model.pkl'):
        """Save the full trained pipeline."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        package = {
            'model2': self.model2,
            'position_models': self.position_models,
            'model3': self.model3,
            'meta_learner': self.meta_learner,
            'label_encoders': self.label_encoders,
            'metrics': self.metrics,
        }
        with open(path, 'wb') as f:
            pickle.dump(package, f)
        print(f"  Saved to {path}")

    @classmethod
    def load(cls, path='models/transferiq_model.pkl'):
        """Load a trained pipeline."""
        with open(path, 'rb') as f:
            package = pickle.load(f)
        obj = cls()
        obj.model2 = package['model2']
        obj.position_models = package['position_models']
        obj.model3 = package['model3']
        obj.meta_learner = package['meta_learner']
        obj.label_encoders = package['label_encoders']
        obj.metrics = package['metrics']
        return obj


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT — run this file directly to train and save
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))

    print("Loading data...")
    df_raw = load_data()
    print(f"Loaded: {len(df_raw):,} rows\n")

    engine = TransferIQValuation()
    metrics = engine.train(df_raw)

    # Save
    print("\n" + "=" * 60)
    print("SAVING")
    print("=" * 60)
    engine.save('models/transferiq_model.pkl')

    # Show undervalued / overvalued players
    print("\n" + "=" * 60)
    print("TOP 15 UNDERVALUED PLAYERS (model says worth MORE than market)")
    print("=" * 60)
    undervalued = engine.get_undervalued(df_raw, min_value=5_000_000, top_n=15)
    for _, row in undervalued.iterrows():
        print(f"  {row.player_name:<25s} {row.position:<10s} "
              f"Actual: EUR {row.actual_value:>12,.0f}  "
              f"Pred: EUR {row.pred_combined:>12,.0f}  "
              f"Gap: {row.value_gap_pct:>+6.1f}%")

    print("\n" + "=" * 60)
    print("TOP 15 OVERVALUED PLAYERS (model says worth LESS than market)")
    print("=" * 60)
    overvalued = engine.get_overvalued(df_raw, min_value=5_000_000, top_n=15)
    for _, row in overvalued.iterrows():
        print(f"  {row.player_name:<25s} {row.position:<10s} "
              f"Actual: EUR {row.actual_value:>12,.0f}  "
              f"Pred: EUR {row.pred_combined:>12,.0f}  "
              f"Gap: {row.value_gap_pct:>+6.1f}%")

    print("\n" + "=" * 60)
    print("DONE — Pipeline ready for Streamlit deployment")
    print("=" * 60)
    print(f"\n  Combined R²: {metrics['r2']:.4f}")
    print(f"  Combined MAE: EUR {metrics['mae']:,.0f}")
    print(f"  Model saved to: models/transferiq_model.pkl")
    print(f"\n  Next: run `streamlit run app/main.py`")
