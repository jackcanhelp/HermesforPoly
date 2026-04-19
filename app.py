import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Hermes V4 | Real Bankroll", layout="wide")

st.title("🏛️ Hermes Polymarket 實盤演練室 (Bankroll V4)")
st.markdown("啟動 **自動深度研究** x **資金水位模擬** 的真實戰場。")

def load_data():
    conn = sqlite3.connect("paper_trading.db")
    try:
        trades_df = pd.read_sql_query("SELECT * FROM paper_trades", conn)
        lessons_df = pd.read_sql_query("SELECT * FROM lessons_learned", conn)
        pf_df = pd.read_sql_query("SELECT * FROM portfolio", conn)
    except Exception:
        trades_df = pd.DataFrame()
        lessons_df = pd.DataFrame()
        pf_df = pd.DataFrame()
    finally:
        conn.close()
    return trades_df, lessons_df, pf_df

trades_df, lessons_df, pf_df = load_data()

# 防呆：確保新的欄位存在
if not trades_df.empty:
    for col in ['kelly_fraction', 'trade_size', 'realized_pnl']:
        if col not in trades_df.columns:
            trades_df[col] = 0.0

if pf_df.empty:
    st.info("目前尚無建立資金庫。請重啟 `main.py` 開始收集數據與建倉。")
else:
    # 頂部 KPI 面板
    col1, col2, col3, col4 = st.columns(4)
    
    current_balance = pf_df.iloc[-1]['balance']
    current_equity = pf_df.iloc[-1].get('total_equity', 10000.0)
    net_profit = current_equity - 10000.0
    
    open_trades = len(trades_df[trades_df['status'] == 'OPEN']) if not trades_df.empty else 0
    closed_trades = trades_df[trades_df['status'] == 'CLOSED'] if not trades_df.empty else pd.DataFrame()
    num_closed = len(closed_trades)
    
    win_rate = 0.0
    if num_closed > 0:
        win_trades = closed_trades[closed_trades['realized_pnl'] > 0]
        win_rate = (len(win_trades) / num_closed) * 100
    
    col1.metric("真實總資產 (Total Equity)", f"${current_equity:,.2f} USD", f"{net_profit:,.2f} USD")
    col2.metric("真實開獎勝率 (Win Rate)", f"{win_rate:.1f}%", f"已開獎 {num_closed} 局")
    col3.metric("在途風險部位 (Open Bets)", f"{open_trades} 筆合約")
    
    # 計算在途資金
    if not trades_df.empty:
        open_risk = trades_df[trades_df['status'] == 'OPEN']['trade_size'].sum()
    else:
        open_risk = 0.0
    col4.metric("已占用保證金 (Locked)", f"${open_risk:,.2f} USD")
    
    st.divider()

    # 圖表：真實資金成長線
    st.subheader("📈 真實資金水位曲線 (Bankroll & Equity)")
    if len(pf_df) > 1:
        # 重塑 DataFrame 畫多條線
        pf_melted = pf_df.melt(id_vars=['timestamp'], value_vars=['balance', 'total_equity'], var_name='Type', value_name='Amount')
        # 幫線段重新命名以利閱讀
        pf_melted['Type'] = pf_melted['Type'].replace({'balance': '手中現金 (Available Cash)', 'total_equity': '總資產淨值 (Total Equity)'})
        
        fig = px.line(pf_melted, x='timestamp', y='Amount', color='Type', markers=True, title="Portfolio Progression")
        fig.add_hline(y=10000, line_dash="dash", line_color="gray", annotation_text="Initial $10,000")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("目前尚無足夠的交易流動，尚未產生曲線。")

    st.divider()
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("🛒 在途未平倉 (Open Trades)")
        if not trades_df.empty:
            open_df = trades_df[trades_df['status'] == 'OPEN'][['timestamp', 'question', 'action', 'market_price', 'trade_size']].sort_values(by='timestamp', ascending=False)
            st.dataframe(open_df, use_container_width=True)
        else:
            st.write("目前空手中。")
            
    with col_right:
        st.subheader("💰 已結算戰果 (Closed PnL)")
        if not closed_trades.empty:
            closed_show = closed_trades[['question', 'action', 'trade_size', 'realized_pnl']].sort_values(by='realized_pnl', ascending=False)
            st.dataframe(closed_show, use_container_width=True)
        else:
            st.write("尚未有任何 14 天內的短線事件滿足結算條件。")

    # 隱藏的推論過程檢視
    st.divider()
    st.subheader("🔍 大腦深度研究報告 (Auto-Research Evidence)")
    if not trades_df.empty:
        selected_trade = st.selectbox("選擇要檢視的交易邏輯", trades_df['question'].unique())
        if selected_trade:
            trade_details = trades_df[trades_df['question'] == selected_trade].iloc[-1]
            st.markdown(f"**建議行動:** {trade_details['action']} | **下注 USD:** ${trade_details.get('trade_size', 0):.2f}")
            with st.expander("詳細調查報告與法官判決"):
                st.write("--- 🧐 情報員匯總的萬字爬蟲總結 (Digest) ---")
                st.text(trade_details.get('context_at_time', 'N/A'))
                st.write("--- 👨‍⚖️ 法官最終交叉辯論 (Reasoning) ---")
                st.markdown(trade_details.get('reasoning', 'N/A'))
