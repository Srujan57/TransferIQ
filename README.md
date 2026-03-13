# Soccer Player Valuation Engine 2026

## 🎯 Overview
This project leverages machine learning to identify undervalued soccer players in a curated 240-player dataset. We move beyond reputation-based pricing to identify "Market Inefficiencies" using performance, physical, and historical metrics.

## 🛠 Project Architecture
- `src/models/`: Expert-level sub-model scripts.
- `src/integration/`: The final valuation equation logic.
- `app/`: Streamlit dashboard for investment simulation.
- `notebooks/`: Exploratory work and model prototyping.

## 🚀 Getting Started
1. **Clone the repo:** `git clone <REPO_URL>`
2. **Setup virtual environment:** `python -m venv venv`
3. **Install dependencies:** `pip install -r requirements.txt`
4. **Data Setup:** Place your `players_data.csv` in `data/`.

## 📈 Methodology
We utilize a **MAPE (Mean Absolute Percentage Error)** loss function to minimize valuation variance. Our model is built to detect players whose predicted value significantly exceeds their current market value, identifying high-ROI investment targets.