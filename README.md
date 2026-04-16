# TransferIQ — Soccer Player Valuation Engine

## Overview
TransferIQ uses three machine learning models to predict soccer player market values across Europe's top 5 leagues, each capturing a different dimension of value:

| Model | Concept | R² |
|---|---|---|
| **Model 2** — Market Perception | Prestige, international profile, league context, transfer history | ~0.76 |
| **Model 2b** — Inherent Ability | Position-specific talent assessment (separate model per position) | ~0.69 |
| **Model 3** — Club Utility | Playing time, tactical contribution, reliability, contractual leverage | ~0.71 |
| **Combined** (Ridge meta-learner) | Learned weighted average of all three | ~0.77 |

The models are combined via a Ridge regression meta-learner that finds optimal weights on held-out player data.

## Key Design Decisions
- **No lag features**: We don't use prior-season market values. Every prediction is based on current-season performance and profile data only.
- **No `highest_market_value_in_eur`**: This column includes the current season's value and would be leaky.
- **Player-level splits**: Model 2b uses player-level train/test splits to prevent cross-season data leakage.
- **Honest metrics**: All R² numbers are on held-out test data, confirmed by 5-fold cross-validation.

## Project Structure
```
TransferIQ/
├── app/
│   └── main.py                  # Streamlit dashboard
├── data/
│   └── master_all_seasons.csv   # Dataset
├── models/
│   └── transferiq_model.pkl     # Trained model (generated)
├── src/
│   ├── models/
│   │   ├── market_model.py          # Model 2
│   │   ├── inherent_ability_model.py # Model 2b
│   │   └── player_value_to_parent_model.py  # Model 3
│   ├── integration/
│   │   └── final_equation.py        # Combined equation + training
│   ├── utils/
│   │   ├── data_loader.py
│   │   ├── feature_engineering.py
│   │   └── metrics.py
│   └── pipeline.py
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the models
```bash
python src/integration/final_equation.py
```
This trains all three sub-models, the meta-learner, evaluates performance, and saves the combined model to `models/transferiq_model.pkl`.

### 3. Launch the dashboard
```bash
streamlit run app/main.py
```

## Methodology
Each sub-model uses **HistGradientBoostingRegressor** (scikit-learn's LightGBM equivalent). The target is `log1p(market_value_in_eur)` to handle the skewed value distribution. Predictions are converted back to EUR via `expm1()`.

The final valuation equation is:
```
predicted_value = expm1(w1 * model2_pred + w2 * model2b_pred + w3 * model3_pred + intercept)
```
Where weights `w1, w2, w3` are learned by Ridge regression on held-out data.
