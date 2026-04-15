##############################################################################
# MODEL 2: Player Value on Market
#
# Predicts log(market_value_in_eur) from market factors and prestige.
# Optimized: R2 ~ 0.70 via feature engineering + tuned hyperparameters.
#
# Features (11):
#   international_caps, international_goals, intl_goal_rate (engineered),
#   league_ranking, Rating, log_transfer_fee, League, has_league_rating,
#   age, contract_years_remaining, sub_position
##############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', '-q', 'xgboost'])
    import xgboost as xgb


print("Loading data...")
df = pd.read_csv('C:/Users/sruja/Desktop/TransferIQ/data/master_all_seasons.csv')
print(f"Dataset: {len(df):,} rows x {df.shape[1]} cols")


# ================================================================
# PREPARE FEATURES
# ================================================================
print("\n" + "="*60)
print("PREPARING FEATURES")
print("="*60)

# Target: log-transform market value
df['log_market_value'] = np.log1p(df['market_value_in_eur'])

# Encode categoricals
le_league = LabelEncoder()
le_sub = LabelEncoder()
df['League_encoded'] = le_league.fit_transform(df['League'])
df['sub_position_encoded'] = le_sub.fit_transform(df['sub_position'])

print(f"League encoding: {dict(zip(le_league.classes_, le_league.transform(le_league.classes_)))}")
print(f"Sub-position encoding: {dict(zip(le_sub.classes_, le_sub.transform(le_sub.classes_)))}")

# Engineer features
df['log_transfer_fee'] = np.log1p(df['transfer_fee'])
df['intl_goal_rate'] = df['international_goals'] / df['international_caps'].clip(lower=1)

# Final feature set
features = [
    'international_caps',
    'international_goals',
    'league_ranking',
    'Rating',
    'log_transfer_fee',
    'League_encoded',
    'has_league_rating',
    'age',
    'contract_years_remaining',
    'intl_goal_rate',
    'sub_position_encoded',
]

target = 'log_market_value'

print(f"\nFeatures ({len(features)}): {features}")
print(f"Target: {target}")


# ================================================================
# TRAIN/TEST SPLIT
# ================================================================
print("\n" + "="*60)
print("TRAIN/TEST SPLIT")
print("="*60)

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")


# ================================================================
# TRAIN MODEL
# ================================================================
print("\n" + "="*60)
print("TRAINING MODEL 2")
print("="*60)

model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# Predictions
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

# Metrics
r2 = r2_score(y_test, y_pred_log)
rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log))
mae = mean_absolute_error(y_true, y_pred)
median_ae = np.median(np.abs(y_true - y_pred))
mask = y_true > 100000
mape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100

print(f"\n  Test Set Results")
print(f"  {'_'*50}")
print(f"  R2 (log scale):         {r2:.4f}")
print(f"  RMSE (log scale):       {rmse_log:.4f}")
print(f"  MAE (euros):            EUR {mae:,.0f}")
print(f"  Median AE (euros):      EUR {median_ae:,.0f}")
print(f"  MAPE:                   {mape:.1f}%")


# ================================================================
# CROSS-VALIDATION
# ================================================================
print("\n" + "="*60)
print("5-FOLD CROSS-VALIDATION")
print("="*60)

cv_model = xgb.XGBRegressor(
    n_estimators=500, max_depth=8, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    random_state=42, n_jobs=-1
)
cv_scores = cross_val_score(
    cv_model, df[features], df[target],
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring='r2'
)
print(f"  R2 per fold:  {cv_scores.round(4)}")
print(f"  R2 mean:      {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")


# ================================================================
# FEATURE IMPORTANCE
# ================================================================
print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)

imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
print("\n  Feature importances:")
for feat, val in imp.sort_values(ascending=False).items():
    bar = '#' * int(val * 100)
    print(f"  {feat:<30s} {val:.4f}  {bar}")

fig, ax = plt.subplots(figsize=(8, 6))
imp.plot(kind='barh', color='steelblue', ax=ax)
ax.set_title(f'Model 2 Feature Importance (R2={r2:.4f})')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig('model2_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: model2_feature_importance.png")


# ================================================================
# PREDICTED VS ACTUAL
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Log scale
axes[0].scatter(y_test, y_pred_log, alpha=0.3, s=10)
mn, mx = y_test.min(), y_test.max()
axes[0].plot([mn, mx], [mn, mx], 'r--', linewidth=1)
axes[0].set_xlabel('Actual log(market_value)')
axes[0].set_ylabel('Predicted log(market_value)')
axes[0].set_title(f'Log Scale  |  R2={r2:.4f}')

# Euro scale
axes[1].scatter(y_true / 1e6, y_pred / 1e6, alpha=0.3, s=10)
mx_eur = max(y_true.max(), y_pred.max()) / 1e6
axes[1].plot([0, mx_eur], [0, mx_eur], 'r--', linewidth=1)
axes[1].set_xlabel('Actual Market Value (M EUR)')
axes[1].set_ylabel('Predicted Market Value (M EUR)')
axes[1].set_title(f'Euro Scale  |  MAE=EUR {mae:,.0f}')

plt.tight_layout()
plt.savefig('model2_predictions.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: model2_predictions.png")


# ================================================================
# ERROR ANALYSIS BY VALUE TIER
# ================================================================
print("\n" + "="*60)
print("ERROR ANALYSIS BY VALUE TIER")
print("="*60)

test_df = pd.DataFrame({
    'actual': y_true.values,
    'predicted': y_pred,
    'abs_error': np.abs(y_true.values - y_pred),
})
test_df['pct_error'] = test_df['abs_error'] / test_df['actual'] * 100
test_df['tier'] = pd.cut(test_df['actual'],
    bins=[0, 1e6, 5e6, 15e6, 50e6, 200e6],
    labels=['<1M', '1-5M', '5-15M', '15-50M', '50M+'])

tier_stats = test_df.groupby('tier', observed=True).agg(
    count=('actual', 'count'),
    median_actual=('actual', lambda x: f"EUR {x.median():,.0f}"),
    median_error=('abs_error', lambda x: f"EUR {x.median():,.0f}"),
    median_pct_error=('pct_error', 'median'),
).round(1)
tier_stats.columns = ['Count', 'Median Value', 'Median Error', 'Median % Error']
print(tier_stats.to_string())


# ================================================================
# SAMPLE PREDICTIONS
# ================================================================
print("\n" + "="*60)
print("SAMPLE PREDICTIONS")
print("="*60)

# Get player names for test set
test_idx = X_test.index
sample_df = df.loc[test_idx, ['player_name', 'market_value_in_eur', 'age',
                               'Rating', 'League']].copy()
sample_df['predicted'] = y_pred
sample_df['error'] = sample_df['market_value_in_eur'] - sample_df['predicted']
sample_df['pct_error'] = (np.abs(sample_df['error']) / sample_df['market_value_in_eur'] * 100).round(1)

# Best predictions (lowest % error among high-value players)
print("\nBest predictions (high-value players, lowest error):")
high_val = sample_df[sample_df.market_value_in_eur > 10_000_000].nsmallest(10, 'pct_error')
for _, row in high_val.iterrows():
    print(f"  {row.player_name:<25s} Actual: EUR {row.market_value_in_eur:>12,.0f}  "
          f"Pred: EUR {row.predicted:>12,.0f}  Error: {row.pct_error:>5.1f}%")

# Worst predictions
print("\nWorst predictions (highest absolute error):")
worst = sample_df.nlargest(10, 'pct_error')
for _, row in worst.iterrows():
    print(f"  {row.player_name:<25s} Actual: EUR {row.market_value_in_eur:>12,.0f}  "
          f"Pred: EUR {row.predicted:>12,.0f}  Error: {row.pct_error:>5.1f}%")


# ================================================================
# SAVE MODEL
# ================================================================
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

import pickle
model_package = {
    'model': model,
    'features': features,
    'target': target,
    'label_encoder_league': le_league,
    'label_encoder_sub_position': le_sub,
    'metrics': {
        'r2': r2,
        'rmse_log': rmse_log,
        'mae': mae,
        'mape': mape,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
    }
}

with open('model2.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print(f"  Saved to model2.pkl")
print(f"\n  To load and predict:")
print(f"    import pickle")
print(f"    with open('model2.pkl', 'rb') as f:")
print(f"        pkg = pickle.load(f)")
print(f"    model = pkg['model']")
print(f"    preds_log = model.predict(X[pkg['features']])")
print(f"    preds_eur = np.expm1(preds_log)")


# ================================================================
# SUMMARY
# ================================================================
print(f"\n{'='*60}")
print(f"MODEL 2 SUMMARY")
print(f"{'='*60}")
print(f"""
  Features:          {len(features)}
  Training rows:     {len(X_train):,}
  Test rows:         {len(X_test):,}

  Test R2:           {r2:.4f}
  CV R2:             {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})
  MAE:               EUR {mae:,.0f}
  MAPE:              {mape:.1f}%

  Top 3 features:    {', '.join(imp.sort_values(ascending=False).index[:3])}

  Saved:             model2.pkl
                     model2_feature_importance.png
                     model2_predictions.png
""")
print("Done!")
