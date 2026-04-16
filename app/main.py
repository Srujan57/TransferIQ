"""
TransferIQ — Streamlit Dashboard

Run: streamlit run app/main.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.data_loader import load_data
from src.utils.feature_engineering import engineer_features
from src.integration.final_equation import TransferIQValuation

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TransferIQ",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Load / train model ───────────────────────────────────────────────────
@st.cache_resource
def get_engine():
    model_path = 'models/transferiq_model.pkl'
    if os.path.exists(model_path):
        return TransferIQValuation.load(model_path)
    else:
        df = load_data()
        engine = TransferIQValuation()
        engine.train(df)
        engine.save(model_path)
        return engine


@st.cache_data
def get_predictions(_engine):
    df = load_data()
    return _engine.predict_decomposed(df)


# ── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.title("⚽ TransferIQ")
st.sidebar.markdown("**Soccer Player Valuation Engine**")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Player Explorer", "Undervalued / Overvalued", "Model Performance"],
)

engine = get_engine()
results = get_predictions(engine)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ═════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("TransferIQ — Player Valuation Engine")
    st.markdown("""
    TransferIQ uses **three machine learning models** to predict soccer player market values,
    each capturing a different dimension of value:
    """)

    col1, col2, col3, col4 = st.columns(4)
    m = engine.metrics
    col1.metric("Combined R²", f"{m['r2']:.4f}")
    col2.metric("MAE", f"€{m['mae']:,.0f}")
    col3.metric("Median Error", f"€{m['median_ae']:,.0f}")
    col4.metric("MAPE", f"{m['mape']:.1f}%")

    st.markdown("---")

    # Sub-model comparison
    st.subheader("Sub-Model Performance")
    sub = m.get('sub_model_r2', {})
    weights = m.get('meta_weights', {})

    model_data = pd.DataFrame({
        'Model': ['Market Perception', 'Inherent Ability', 'Club Utility'],
        'Concept': [
            'Prestige, international profile, league context',
            'Position-specific talent (per-position models)',
            'Playing time, tactical contribution, reliability',
        ],
        'R²': [
            sub.get('model2_market', 0),
            sub.get('model2b_ability', 0),
            sub.get('model3_utility', 0),
        ],
        'Weight': [
            weights.get('model2', 0),
            weights.get('model2b', 0),
            weights.get('model3', 0),
        ],
    })

    st.dataframe(
        model_data.style.format({'R²': '{:.4f}', 'Weight': '{:.4f}'}),
        use_container_width=True, hide_index=True,
    )

    # Predicted vs Actual scatter
    st.subheader("Predicted vs Actual Market Value")
    fig = px.scatter(
        results, x='actual_value', y='pred_combined',
        hover_data=['player_name', 'position', 'age'],
        labels={'actual_value': 'Actual (EUR)', 'pred_combined': 'Predicted (EUR)'},
        opacity=0.4,
    )
    max_val = max(results['actual_value'].max(), results['pred_combined'].max())
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode='lines', line=dict(dash='dash', color='red'),
        name='Perfect prediction',
    ))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Player Explorer
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Player Explorer":
    st.title("Player Explorer")

    # Deduplicate: keep only each player's most recent season
    explorer_df = results.copy()
    if 'season_year' in explorer_df.columns:
        explorer_df = explorer_df.sort_values('season_year').drop_duplicates(
            'player_name', keep='last'
        )
    else:
        explorer_df = explorer_df.drop_duplicates('player_name', keep='last')

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        positions = ['All'] + sorted(explorer_df['position'].unique().tolist())
        sel_position = st.selectbox("Position", positions)
    with col2:
        search = st.text_input("Search Player", "")

    filtered = explorer_df.copy()
    if sel_position != 'All':
        filtered = filtered[filtered['position'] == sel_position]
    if search:
        filtered = filtered[filtered['player_name'].str.contains(search, case=False, na=False)]

    st.markdown(f"**{len(filtered):,} players**")

    # Display table — no League, no Club
    display_cols = [
        'player_name', 'age', 'position', 'sub_position', 'Rating',
        'actual_value', 'pred_combined', 'error_pct', 'value_gap_pct',
    ]
    display_df = filtered[display_cols].sort_values('actual_value', ascending=False)
    display_df.columns = [
        'Player', 'Age', 'Pos', 'Sub-Pos', 'Rating',
        'Actual (EUR)', 'Predicted (EUR)', 'Error %', 'Value Gap %',
    ]

    st.dataframe(
        display_df.style.format({
            'Actual (EUR)': '€{:,.0f}',
            'Predicted (EUR)': '€{:,.0f}',
            'Error %': '{:.1f}%',
            'Value Gap %': '{:+.1f}%',
            'Rating': '{:.2f}',
        }),
        use_container_width=True, hide_index=True, height=600,
    )

    # Player detail card
    if search and len(filtered) > 0:
        player = filtered.iloc[0]
        st.markdown("---")
        st.subheader(f"📋 {player['player_name']}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Actual Value", f"€{player['actual_value']:,.0f}")
        c2.metric("Predicted", f"€{player['pred_combined']:,.0f}")
        c3.metric("Value Gap", f"{player['value_gap_pct']:+.1f}%")
        c4.metric("Age", f"{player['age']:.0f}")

        # Sub-model breakdown
        st.markdown("**Sub-Model Predictions:**")
        breakdown = pd.DataFrame({
            'Model': ['Market Perception', 'Inherent Ability', 'Club Utility', 'Combined'],
            'Prediction (EUR)': [
                player['pred_market_perception'],
                player['pred_inherent_ability'],
                player['pred_club_utility'],
                player['pred_combined'],
            ],
        })
        st.dataframe(
            breakdown.style.format({'Prediction (EUR)': '€{:,.0f}'}),
            use_container_width=True, hide_index=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Undervalued / Overvalued
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Undervalued / Overvalued":
    st.title("Market Inefficiencies")
    st.markdown("""
    Players where the model's prediction **significantly differs** from their
    current Transfermarkt value — potential scouting opportunities.
    """)

    min_val = st.slider(
        "Minimum market value (EUR)", 1_000_000, 50_000_000, 5_000_000,
        step=1_000_000, format="€%d",
    )
    top_n = st.slider("Number of players", 5, 50, 20)

    # Deduplicate for this page too
    deduped = results.copy()
    if 'season_year' in deduped.columns:
        deduped = deduped.sort_values('season_year').drop_duplicates(
            'player_name', keep='last'
        )
    else:
        deduped = deduped.drop_duplicates('player_name', keep='last')

    filtered = deduped[deduped['actual_value'] >= min_val].copy()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🟢 Most Undervalued")
        st.markdown("*Model thinks they're worth MORE than the market says*")
        undervalued = filtered.nlargest(top_n, 'value_gap_pct')
        st.dataframe(
            undervalued[['player_name', 'position', 'age',
                         'actual_value', 'pred_combined', 'value_gap_pct']].rename(columns={
                'player_name': 'Player', 'position': 'Pos', 'age': 'Age',
                'actual_value': 'Actual', 'pred_combined': 'Predicted',
                'value_gap_pct': 'Gap %',
            }).style.format({
                'Actual': '€{:,.0f}', 'Predicted': '€{:,.0f}', 'Gap %': '{:+.1f}%',
            }),
            use_container_width=True, hide_index=True,
        )

    with col2:
        st.subheader("🔴 Most Overvalued")
        st.markdown("*Model thinks they're worth LESS than the market says*")
        overvalued = filtered.nsmallest(top_n, 'value_gap_pct')
        st.dataframe(
            overvalued[['player_name', 'position', 'age',
                        'actual_value', 'pred_combined', 'value_gap_pct']].rename(columns={
                'player_name': 'Player', 'position': 'Pos', 'age': 'Age',
                'actual_value': 'Actual', 'pred_combined': 'Predicted',
                'value_gap_pct': 'Gap %',
            }).style.format({
                'Actual': '€{:,.0f}', 'Predicted': '€{:,.0f}', 'Gap %': '{:+.1f}%',
            }),
            use_container_width=True, hide_index=True,
        )

    # Value gap distribution
    st.markdown("---")
    st.subheader("Value Gap Distribution")
    fig = px.histogram(
        filtered, x='value_gap_pct', nbins=60,
        labels={'value_gap_pct': 'Value Gap %'},
        title='Distribution of Model vs Market Disagreement',
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Model Performance
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.title("Model Performance Analysis")

    m = engine.metrics

    # Key metrics
    st.subheader("Combined Model Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("R²", f"{m['r2']:.4f}")
    c2.metric("RMSE (log)", f"{m['rmse_log']:.4f}")
    c3.metric("MAE", f"€{m['mae']:,.0f}")
    c4.metric("Median AE", f"€{m['median_ae']:,.0f}")
    c5.metric("MAPE", f"{m['mape']:.1f}%")

    # Sub-model R² comparison
    st.subheader("Sub-Model Comparison")
    sub = m.get('sub_model_r2', {})
    bar_df = pd.DataFrame({
        'Model': ['Market\nPerception', 'Inherent\nAbility', 'Club\nUtility', 'Combined'],
        'R²': [
            sub.get('model2_market', 0),
            sub.get('model2b_ability', 0),
            sub.get('model3_utility', 0),
            m['r2'],
        ],
    })
    fig = px.bar(
        bar_df, x='Model', y='R²', text='R²',
        title='R² by Sub-Model vs Combined',
        color='Model',
        color_discrete_sequence=['#4C78A8', '#72B7B2', '#54A24B', '#E45756'],
    )
    fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig.update_layout(height=400, showlegend=False, yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    # Error by tier
    st.subheader("Error by Value Tier")
    from src.utils.metrics import tier_analysis
    tiers = tier_analysis(results['actual_value'], results['pred_combined'])
    st.dataframe(
        tiers.style.format({
            'Median Value': '€{:,.0f}', 'Median Error': '€{:,.0f}',
            'Median % Error': '{:.1f}%',
        }),
        use_container_width=True,
    )

    # Error distribution
    st.subheader("Prediction Error Distribution")
    fig = px.histogram(
        results[results['actual_value'] > 1_000_000],
        x='error_pct', nbins=80,
        labels={'error_pct': 'Absolute Error %'},
        title='Error Distribution (players valued >€1M)',
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Meta-learner weights
    st.subheader("Meta-Learner Weights")
    weights = m.get('meta_weights', {})
    st.markdown(f"""
    The combined prediction is a learned weighted average:

    **Predicted Value = {weights.get('model2', 0):.3f} × Market Perception
    + {weights.get('model2b', 0):.3f} × Inherent Ability
    + {weights.get('model3', 0):.3f} × Club Utility
    + {weights.get('intercept', 0):.3f}**

    These weights were learned by a Ridge regression on held-out player data,
    ensuring the combination is optimized without overfitting.
    """)


# ── Footer ────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built with [Streamlit](https://streamlit.io) • "
    "Models: HistGradientBoosting + Ridge"
)
