"""
Metrics utilities for TransferIQ model evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_test_log, y_pred_log):
    """
    Compute all evaluation metrics from log-scale predictions.
    Returns a dict with r2, rmse_log, mae, median_ae, mape.
    """
    y_true = np.expm1(np.asarray(y_test_log))
    y_pred = np.expm1(np.asarray(y_pred_log))

    r2       = r2_score(y_test_log, y_pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_test_log, y_pred_log))
    mae      = mean_absolute_error(y_true, y_pred)
    median_ae = float(np.median(np.abs(y_true - y_pred)))
    mask     = y_true > 100_000
    mape     = float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100)

    return {
        'r2': r2,
        'rmse_log': rmse_log,
        'mae': mae,
        'median_ae': median_ae,
        'mape': mape,
    }


def tier_analysis(y_true_eur, y_pred_eur):
    """
    Break down error by value tier. Returns a DataFrame.
    """
    test_df = pd.DataFrame({
        'actual': np.asarray(y_true_eur),
        'predicted': np.asarray(y_pred_eur),
        'abs_error': np.abs(np.asarray(y_true_eur) - np.asarray(y_pred_eur)),
    })
    test_df['pct_error'] = test_df['abs_error'] / test_df['actual'].clip(lower=1) * 100
    test_df['tier'] = pd.cut(
        test_df['actual'],
        bins=[0, 1e6, 5e6, 15e6, 50e6, 200e6],
        labels=['<1M', '1-5M', '5-15M', '15-50M', '50M+'],
    )

    tier_stats = test_df.groupby('tier', observed=True).agg(
        count=('actual', 'count'),
        median_actual=('actual', 'median'),
        median_error=('abs_error', 'median'),
        median_pct_error=('pct_error', 'median'),
    ).round(1)
    tier_stats.columns = ['Count', 'Median Value', 'Median Error', 'Median % Error']
    return tier_stats


def print_metrics(metrics_dict, label="Test Set"):
    """Pretty-print a metrics dict."""
    m = metrics_dict
    print(f"\n  {label}")
    print(f"  {'─' * 50}")
    print(f"  R² (log scale):      {m['r2']:.4f}")
    print(f"  RMSE (log scale):    {m['rmse_log']:.4f}")
    print(f"  MAE:                 EUR {m['mae']:,.0f}")
    print(f"  Median AE:           EUR {m['median_ae']:,.0f}")
    print(f"  MAPE:                {m['mape']:.1f}%")
