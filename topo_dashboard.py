"""
TopoOmega v2.0 Dashboard: Topology Visualization
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import time
import os
import numpy as np

st.set_page_config(
    page_title="TOPOOMEGA v2.0 WAR ROOM",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e2028 0%, #2d303e 100%);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #00ff88;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.2);
    }
    .big-font {
        font-size: 28px !important;
        font-weight: bold;
        color: #00ff88;
    }
    .topology-metric {
        font-size: 18px;
        color: #88ccff;
        font-family: 'Courier New', monospace;
    }
    </style>
    """, unsafe_allow_html=True)

MONITOR_FILE = "src/data/monitor.json"

def load_state():
    if not os.path.exists(MONITOR_FILE):
        return None
    try:
        with open(MONITOR_FILE, 'r') as f:
            return json.load(f)
    except:
        return None

def plot_persistence_diagram(h0_data, h1_data):
    """Plot H0 and H1 persistence diagrams side by side"""
    # Mock data for now (will be populated from monitor)
    if h0_data is None:
        return None
        
    fig = go.Figure()
    
    # H0 points (connected components)
    if len(h0_data) > 0:
        fig.add_trace(go.Scatter(
            x=h0_data[:, 0],
            y=h0_data[:, 1],
            mode='markers',
            name='H0 (Components)',
            marker=dict(size=8, color='cyan', symbol='circle')
        ))
    
    # H1 points (loops)
    if len(h1_data) > 0:
        fig.add_trace(go.Scatter(
            x=h1_data[:, 0],
            y=h1_data[:, 1],
            mode='markers',
            name='H1 (Loops)',
            marker=dict(size=10, color='magenta', symbol='diamond')
        ))
    
    # Diagonal line (birth = death)
    max_val = max(
        h0_data[:, 1].max() if len(h0_data) > 0 else 0,
        h1_data[:, 1].max() if len(h1_data) > 0 else 0
    )
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        name='Birth=Death',
        line=dict(color='white', dash='dash', width=1)
    ))
    
    fig.update_layout(
        xaxis_title="Birth",
        yaxis_title="Death",
        height=300,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,32,40,0.8)',
        font=dict(color='white'),
        showlegend=True
    )
    
    return fig

def main():
    st.markdown("# 🦅 TOPOOMEGA v2.0 WAR ROOM")
    st.markdown("**Persistent Homology × Bifiltrated Persistence × Wasserstein-PPO**")
    
    if st.button("⚡ REFRESH"):
        st.rerun()
    
    state = load_state()
    
    if state is None:
        st.warning("⏳ WAITING FOR ENGINE DATA...")
        time.sleep(2)
        st.rerun()
        return

    # Header Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💰 EQUITY", f"${state.get('equity', 0):,.2f}")
        
    with col2:
        price = state.get('price')
        st.metric("📊 PRICE", f"${price:,.2f}" if price else "N/A")
        
    with col3:
        signal = state.get('signal', 0)
        leverage = state.get('leverage', 1)
        if signal == 1:
            sig_str = f"🟢 LONG {leverage:.1f}x"
            color = "#00ff88"
        elif signal == -1:
            sig_str = f"🔴 SHORT {leverage:.1f}x"
            color = "#ff4444"
        else:
            sig_str = "⚪ NEUTRAL"
            color = "white"
        st.markdown(f"<div class='big-font' style='color:{color}'>{sig_str}</div>", unsafe_allow_html=True)
        
    with col4:
        coin = state.get('coin', 'N/A')
        st.metric("🪙 COIN", coin)

    st.markdown("---")

    # Topology Metrics Row
    st.markdown("### 🔬 TOPOLOGICAL ANALYSIS")
    
    tc1, tc2, tc3, tc4, tc5 = st.columns(5)
    
    with tc1:
        loop_score = state.get('loop_score', 0)
        st.markdown(f"<div class='topology-metric'>Loop Score<br><b>{loop_score:.4f}</b></div>", unsafe_allow_html=True)
        st.caption("H1 Persistence Strength")
        
    with tc2:
        tti = state.get('tti', 0)
        tti_color = "#ff4444" if tti > 3.0 else "#00ff88"
        st.markdown(f"<div class='topology-metric' style='color:{tti_color}'>TTI<br><b>{tti:.2f}</b></div>", unsafe_allow_html=True)
        st.caption("Turbulence Index")
        
    with tc3:
        regime = state.get('regime', 'N/A')
        regime_icon = "📈" if regime == "trending" else ("🔄" if regime == "mean_reverting" else "⚠️")
        st.markdown(f"<div class='topology-metric'>{regime_icon} {regime.upper()}</div>", unsafe_allow_html=True)
        st.caption("Market Regime")
        
    with tc4:
        confidence = state.get('confidence', 0)
        conf_color = "#00ff88" if confidence > 0.7 else ("#ffaa00" if confidence > 0.5 else "#ff4444")
        st.markdown(f"<div class='topology-metric' style='color:{conf_color}'>Confidence<br><b>{confidence:.2f}</b></div>", unsafe_allow_html=True)
        st.caption("Model Certainty")
        
    with tc5:
        dissolution_time = state.get('dissolution_time')
        if dissolution_time is not None:
            st.markdown(f"<div class='topology-metric'>Loop Dissolution<br><b>{dissolution_time:.0f}h</b></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='topology-metric'>Loop Dissolution<br><b>N/A</b></div>", unsafe_allow_html=True)
        st.caption("Predicted Time")

    st.markdown("---")

    # Charts Row
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("### 📊 PRICE ACTION")
        history = state.get('history', [])
        if history:
            df = pd.DataFrame(history)
            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['o'],
                high=df['h'],
                low=df['l'],
                close=df['c'],
                name='Price'
            )])
            
            fig.update_layout(
                xaxis_rangeslider_visible=False,
                height=350,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(30,32,40,0.8)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("WAITING FOR HISTORY DATA...")
    
    with col_chart2:
        st.markdown("### 🌀 PERSISTENCE DIAGRAM")
        st.caption("Birth-Death pairs of topological features")
        # Mock persistence diagram (would come from state in production)
        st.info("Persistence diagram visualization coming in next update")

    # Performance Metrics
    st.markdown("---")
    st.markdown("### 📈 PERFORMANCE METRICS")
    
    perf1, perf2, perf3, perf4 = st.columns(4)
    
    with perf1:
        st.metric("Target CAGR", "450-650%", help="Annualized Return Target")
    with perf2:
        st.metric("Target Sharpe", "15-19", help="Risk-Adjusted Return")
    with perf3:
        st.metric("Max DD Limit", "<7%", help="Maximum Drawdown Threshold")
    with perf4:
        st.metric("Target Calmar", ">70", help="CAGR / Max DD")

    # Raw State (Debug)
    with st.expander("🔧 RAW STATE DATA"):
        st.json(state)
        
    # Auto-refresh
    time.sleep(5)
    st.rerun()

if __name__ == "__main__":
    main()
