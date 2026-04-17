"""
TransferIQ — Final Valuation Equation

Combines three sub-models into a single predicted market value:
  Model 2:  Market Perception (prestige, reputation, league context)
  Model 2b: Inherent Ability  (position-specific talent assessment)
  Model 3:  Club Utility      (playing time, tactical contribution)

The final prediction is a learned weighted average via Ridge regression
on the three sub-model outputs, trained on held-out data to find optimal
weights without overfitting.

Uses cross-validated stacking (GroupKFold by player_id) to generate
out-of-fold predictions for the meta-learner, preventing it from
overfitting on in-sample sub-model predictions.

Run this file directly to train everything, evaluate, and save artifacts.
"""

import sys
import os
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error

# ── Add project root to path ─────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.feature_engineering import engineer_features, inject_history_features
from src.utils.data_loader import load_data, load_training_data
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
    Cross-validated stacking ensures the meta-learner trains on honest
    out-of-fold predictions rather than inflated in-sample ones.
    """

    def __init__(self):
        self.model2 = None           # Market perception
        self.position_models = None  # Inherent ability (per-position)
        self.model3 = None           # Club utility
        self.meta_learner = None     # Ridge combining the three
        self.label_encoders = None
        self.metrics = {}
        self._train_raw = None       # Stored for inject_history_features at predict time

    def train(self, df_train_raw, df_test_raw=None):
        """
        Full training pipeline with cross-validated stacking.

        Uses GroupKFold (grouped by player_id) to generate out-of-fold
        predictions for the meta-learner, preventing it from overfitting
        on in-sample sub-model predictions.
        """
        print("=" * 60)
        print("TransferIQ — Training Full Valuation Pipeline")
        print("=" * 60)

        # ── Feature engineering ───────────────────────────────────────────
        print("\n[1/6] Engineering features...")
        self._train_raw = df_train_raw
        df_train, self.label_encoders = engineer_features(df_train_raw)
        print(f"  Train dataset: {len(df_train):,} rows, {df_train.shape[1]} columns")

        if df_test_raw is not None:
            df_test_raw_enriched = inject_history_features(df_test_raw, df_train_raw)
            df_test, _ = engineer_features(df_test_raw_enriched)
            print(f"  Test dataset:  {len(df_test):,} rows")
        else:
            print("  No test set provided — using 20% player-level split")
            all_players = df_train['player_id'].unique()
            np.random.seed(42)
            test_players = np.random.choice(
                all_players, size=int(len(all_players) * 0.2), replace=False
            )
            test_mask = df_train['player_id'].isin(test_players)
            df_test   = df_train[test_mask].copy()
            df_train  = df_train[~test_mask].copy()
            print(f"  Train: {len(df_train):,} rows | Test: {len(df_test):,} rows")

        # ── Print split summary ───────────────────────────────────────────
        print("\n[2/6] Dataset summary...")
        print(f"  Train: {len(df_train):,} rows, "
              f"{df_train['player_id'].nunique():,} unique players")
        print(f"  Test:  {len(df_test):,} rows, "
              f"{df_test['player_id'].nunique():,} unique players")
        overlap = set(df_train['player_id'].unique()) & set(df_test['player_id'].unique())
        print(f"  Player overlap (train ∩ test): {len(overlap):,}")

        # ── Generate out-of-fold predictions for meta-learner ─────────────
        print("\n[3/6] Cross-validated stacking (5-fold GroupKFold)...")
        gkf = GroupKFold(n_splits=5)
        groups = df_train['player_id'].values
        y_train_all = df_train['log_market_value'].values

        oof_m2  = np.zeros(len(df_train))
        oof_m2b = np.zeros(len(df_train))
        oof_m3  = np.zeros(len(df_train))

        for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(df_train, groups=groups), 1):
            df_tr  = df_train.iloc[tr_idx]
            df_val = df_train.iloc[val_idx]

            # Model 2
            m2_fold = build_model2()
            m2_fold.fit(df_tr[MODEL2_FEATURES], df_tr['log_market_value'])
            oof_m2[val_idx] = m2_fold.predict(df_val[MODEL2_FEATURES])

            # Model 2b (per-position)
            df_tr_clean  = df_tr.copy()
            df_val_clean = df_val.copy()
            df_tr_clean[MODEL2B_FEATURES] = df_tr_clean[MODEL2B_FEATURES].replace(
                [np.inf, -np.inf], np.nan
            ).fillna(-1)
            df_val_clean[MODEL2B_FEATURES] = df_val_clean[MODEL2B_FEATURES].replace(
                [np.inf, -np.inf], np.nan
            ).fillna(-1)

            fold_pos_models = {}
            for pos in sorted(df_tr['position'].unique()):
                pos_mask = df_tr_clean['position'] == pos
                if pos_mask.sum() < 20:
                    continue
                m = build_position_model()
                y_tr_pos = np.log(df_tr_clean.loc[pos_mask, 'market_value_in_eur'].clip(lower=1))
                m.fit(df_tr_clean.loc[pos_mask, MODEL2B_FEATURES], y_tr_pos)
                fold_pos_models[pos] = m
            m2b_pred_log = predict_model2b(fold_pos_models, df_val)
            oof_m2b[val_idx] = np.log1p(np.exp(m2b_pred_log))

            # Model 3
            m3_fold = build_model3()
            m3_fold.fit(df_tr[MODEL3_FEATURES], df_tr['log_market_value'])
            oof_m3[val_idx] = m3_fold.predict(df_val[MODEL3_FEATURES])

            print(f"    Fold {fold_i}: M2={r2_score(y_train_all[val_idx], oof_m2[val_idx]):.4f}  "
                  f"M2b={r2_score(y_train_all[val_idx], oof_m2b[val_idx]):.4f}  "
                  f"M3={r2_score(y_train_all[val_idx], oof_m3[val_idx]):.4f}")

        print(f"  OOF R²: M2={r2_score(y_train_all, oof_m2):.4f}  "
              f"M2b={r2_score(y_train_all, oof_m2b):.4f}  "
              f"M3={r2_score(y_train_all, oof_m3):.4f}")

        # ── Train final sub-models on ALL training data ───────────────────
        print("\n[4/6] Training final sub-models on full training data...")

        self.model2 = build_model2()
        self.model2.fit(df_train[MODEL2_FEATURES], df_train['log_market_value'])
        m2_test_pred = predict_model2(self.model2, df_test)
        m2_r2 = r2_score(df_test['log_market_value'], m2_test_pred)
        print(f"    Model 2  (Market):   R² = {m2_r2:.4f}")

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
            if pos_mask_tr.sum() < 20:
                continue
            m = build_position_model()
            y_tr = np.log(df_train_clean.loc[pos_mask_tr, 'market_value_in_eur'].clip(lower=1))
            m.fit(df_train_clean.loc[pos_mask_tr, MODEL2B_FEATURES], y_tr)
            self.position_models[pos] = m

        m2b_test_pred_log = predict_model2b(self.position_models, df_test)
        m2b_test_pred = np.log1p(np.exp(m2b_test_pred_log))
        m2b_r2 = r2_score(df_test['log_market_value'], m2b_test_pred)
        print(f"    Model 2b (Ability):  R² = {m2b_r2:.4f}")

        self.model3 = build_model3()
        self.model3.fit(df_train[MODEL3_FEATURES], df_train['log_market_value'])
        m3_test_pred = predict_model3(self.model3, df_test)
        m3_r2 = r2_score(df_test['log_market_value'], m3_test_pred)
        print(f"    Model 3  (Utility):  R² = {m3_r2:.4f}")

        # ── Train meta-learner on OOF predictions ─────────────────────────
        print("\n[5/6] Training meta-learner on out-of-fold predictions...")
        X_meta_oof  = np.column_stack([oof_m2, oof_m2b, oof_m3])
        X_meta_test = np.column_stack([m2_test_pred, m2b_test_pred, m3_test_pred])
        y_meta_test = df_test['log_market_value'].values

        self.meta_learner = Ridge(alpha=1.0)
        self.meta_learner.fit(X_meta_oof, y_train_all)

        weights = self.meta_learner.coef_
        print(f"  Learned weights:")
        print(f"    Model 2  (Market):    {weights[0]:.4f}")
        print(f"    Model 2b (Ability):   {weights[1]:.4f}")
        print(f"    Model 3  (Utility):   {weights[2]:.4f}")
        print(f"    Intercept:            {self.meta_learner.intercept_:.4f}")

        # ── Evaluate combined model ───────────────────────────────────────
        print("\n[6/6] Evaluating combined model...")
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

        tiers = tier_analysis(actual_eur, final_pred_eur)
        print("\n  Error by Value Tier:")
        print(tiers.to_string())

        print(f"\n  {'─' * 55}")
        print(f"  {'Model':<30} {'R²':>8}")
        print(f"  {'─' * 55}")
        print(f"  {'Model 2  (Market Perception)':<30} {m2_r2:>8.4f}")
        print(f"  {'Model 2b (Inherent Ability)':<30} {m2b_r2:>8.4f}")
        print(f"  {'Model 3  (Club Utility)':<30} {m3_r2:>8.4f}")
        print(f"  {'─' * 55}")
        print(f"  {'COMBINED (Stacked Ridge)':<30} {self.metrics['r2']:>8.4f}")
        print(f"  {'─' * 55}")

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
        if self._train_raw is not None:
            df_raw = inject_history_features(df_raw, self._train_raw)
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
        if self._train_raw is not None:
            df_raw = inject_history_features(df_raw, self._train_raw)
        df, _ = engineer_features(df_raw)

        m2_pred  = predict_model2(self.model2, df)
        m2b_pred_log = predict_model2b(self.position_models, df)
        m2b_pred = np.log1p(np.exp(m2b_pred_log))
        m3_pred  = predict_model3(self.model3, df)

        X_meta = np.column_stack([m2_pred, m2b_pred, m3_pred])
        final_log = self.meta_learner.predict(X_meta)

        pass_cols = ['player_name', 'market_value_in_eur', 'age', 'position',
                    'sub_position', 'League', 'Rating', 'current_club_name']
        if 'Season' in df.columns:
            pass_cols.append('Season')
        if 'last_season' in df.columns:
            pass_cols.append('last_season')
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
            'train_raw': self._train_raw,
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
        obj._train_raw = package.get('train_raw')
        return obj


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT — run this file directly to train and save
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))

    # ── Temporal split: train on <=2023, test on 2024 ─────────────────────
    # Zero row overlap. master_all_seasons <=2023 for training,
    # FINALMASTER Season_End_Year == 2024 for evaluation.
    TRAIN_CUTOFF = 2023
    TEST_CUTOFF  = 2024

    print(f"Temporal split: train on seasons <={TRAIN_CUTOFF}, "
          f"test on seasons >={TEST_CUTOFF}")
    print()

    print("Loading training data (master_all_seasons, seasons <= 2023)...")
    df_train_raw = load_training_data(season_cutoff=TRAIN_CUTOFF)
    print(f"  Loaded: {len(df_train_raw):,} rows, "
          f"seasons {sorted(df_train_raw['season_year'].unique())}\n")

    print("Loading test data (FINALMASTER, Season_End_Year >= 2024)...")
    df_test_raw = load_data(test_season_cutoff=TEST_CUTOFF)
    print(f"  Loaded: {len(df_test_raw):,} rows, "
          f"seasons {sorted(df_test_raw['Season_End_Year'].dropna().unique())}\n")

    engine = TransferIQValuation()
    metrics = engine.train(df_train_raw, df_test_raw)

    # Save
    print("\n" + "=" * 60)
    print("SAVING")
    print("=" * 60)
    engine.save('models/transferiq_model.pkl')

    # Show undervalued / overvalued players
    print("\n" + "=" * 60)
    print("TOP 15 UNDERVALUED PLAYERS (model says worth MORE than market)")
    print("=" * 60)
    undervalued = engine.get_undervalued(df_test_raw, min_value=5_000_000, top_n=15)
    for _, row in undervalued.iterrows():
        print(f"  {row.player_name:<25s} {row.position:<10s} "
              f"Actual: EUR {row.actual_value:>12,.0f}  "
              f"Pred: EUR {row.pred_combined:>12,.0f}  "
              f"Gap: {row.value_gap_pct:>+6.1f}%")

    print("\n" + "=" * 60)
    print("TOP 15 OVERVALUED PLAYERS (model says worth LESS than market)")
    print("=" * 60)
    overvalued = engine.get_overvalued(df_test_raw, min_value=5_000_000, top_n=15)
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
