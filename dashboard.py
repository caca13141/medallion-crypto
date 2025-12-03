import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import time
import os

# Set page config
st.set_page_config(
    page_title="MEDALLION WAR ROOM",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for "War Room" aesthetic
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #4f4f4f;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .profit { color: #00ff00; }
    .loss { color: #ff0000; }
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

def main():
    st.title("🦅 MEDALLION CRYPTO WAR ROOM")
    
    # Auto-refresh logic
    if st.button("REFRESH DATA"):
        st.rerun()
        
    state = load_state()
    
    if state is None:
        st.warning("WAITING FOR ENGINE DATA...")
        time.sleep(2)
        st.rerun()
        return

    # Header Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("EQUITY", f"${state.get('equity', 0):,.2f}")
        
    with col2:
        price = state.get('price')
        st.metric("LAST PRICE", f"${price:,.2f}" if price else "N/A")
        
    with col3:
        signal = state.get('signal', 0)
        sig_str = "NEUTRAL"
        color = "white"
        if signal == 1: 
            sig_str = "LONG 🟢"
            color = "#00ff00"
        elif signal == -1: 
            sig_str = "SHORT 🔴"
            color = "#ff0000"
        st.markdown(f"### SIGNAL: <span style='color:{color}'>{sig_str}</span>", unsafe_allow_html=True)
        
    with col4:
        coin = state.get('coin', 'N/A')
        st.metric("ACTIVE COIN", coin)

    st.markdown("---")

    # Alpha Metrics
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        hurst = state.get('hurst')
        val = f"{hurst:.2f}" if hurst else "N/A"
        st.metric("HURST EXPONENT", val, help="> 0.55 = Trending")
        
    with c2:
        fusion = state.get('fusion')
        val = f"{fusion}" if fusion is not None else "N/A"
        st.metric("FUSION SCORE", val, help="> 0 = Crowded Shorts, < 0 = Crowded Longs")
        
    with c3:
        narrative = state.get('narrative')
        st.metric("NARRATIVE ID", narrative)
        
    with c4:
        atr = state.get('atr')
        val = f"{atr:.2f}" if atr else "N/A"
        st.metric("ATR (VOLATILITY)", val)

    # Charts
    st.markdown("### 📊 MARKET STRUCTURE")
    
    history = state.get('history', [])
    if history:
        df = pd.DataFrame(history)
        # Create Candlestick Chart
        fig = go.Figure(data=[go.Candlestick(
            x=df.index, # We don't have timestamps in MVP feed yet, using index
            open=df['o'],
            high=df['h'],
            low=df['l'],
            close=df['c'],
            name='Price'
        )])
        
        # Add EMA if available (we need to calculate or pass them)
        # For now just Price
        
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=400,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("WAITING FOR HISTORY DATA...")
    
    # Raw JSON for debug
    with st.expander("RAW STATE DATA"):
        st.json(state)
        
    # Auto-refresh loop using sleep (simple polling)
    time.sleep(5)
    st.rerun()

if __name__ == "__main__":
    main()
