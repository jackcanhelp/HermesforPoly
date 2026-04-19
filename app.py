import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Hermes Polymarket Dashboard", layout="wide")

st.title("🏛️ Hermes Polymarket 預測儀表板")
st.markdown("基於 **多代理人辯論系統 (V2)** 驅動的 Paper Trading 即時戰情室")

# 讀取 DB
def load_data():
    conn = sqlite3.connect("paper_trading.db")
    try:
        trades_df = pd.read_sql_query("SELECT * FROM paper_trades", conn)
        lessons_df = pd.read_sql_query("SELECT * FROM lessons_learned", conn)
    except Exception:
        trades_df = pd.DataFrame()
        lessons_df = pd.DataFrame()
    finally:
        conn.close()
    return trades_df, lessons_df

trades_df, lessons_df = load_data()

# 確保 V2 新增的欄位存在 (以免使用者還沒重啟主程式時看網頁會崩潰)
if not trades_df.empty and 'kelly_fraction' not in trades_df.columns:
    trades_df['kelly_fraction'] = 0.0

if trades_df.empty:
    st.info("目前尚無任何交易紀錄。請執行 `main.py` 開始收集數據。")
else:
    # KPI
    col1, col2, col3, col4 = st.columns(4)
    total_trades = len(trades_df)
    open_trades = len(trades_df[trades_df['status'] == 'OPEN'])
    closed_trades = total_trades - open_trades
    avg_ev = trades_df['ev'].mean() * 100 if total_trades > 0 else 0
    
    col1.metric("總監控發現訊號數", total_trades)
    col2.metric("活躍的未平倉交易", open_trades)
    col3.metric("已結算的交易數", closed_trades)
    col4.metric("平均期望值 (EV) 優勢", f"{avg_ev:.2f}%")
    
    st.divider()

    # 圖表：期望值累計捕捉量
    st.subheader("📈 總期望值累積曲線")
    if total_trades > 0:
        trades_df['Cumulative EV'] = trades_df['ev'].cumsum()
        fig = px.line(trades_df, x=trades_df.index, y='Cumulative EV', markers=True, title="Cumulative Edge Captured (單位: EV)")
        st.plotly_chart(fig, use_container_width=True)

    # 兩大表格展示
    st.divider()
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("🛒 最新活躍預測 (Open Trades)")
        open_df = trades_df[trades_df['status'] == 'OPEN'][['timestamp', 'question', 'action', 'ev', 'kelly_fraction']].sort_values(by='timestamp', ascending=False)
        st.dataframe(open_df, use_container_width=True)
        
    with col_right:
        st.subheader("🧠 大腦反思紀錄庫 (Lessons Learned)")
        if not lessons_df.empty:
            lessons_show = lessons_df[['timestamp', 'lesson']].sort_values(by='timestamp', ascending=False)
            st.dataframe(lessons_show, use_container_width=True)
        else:
            st.write("目前尚無結算單的反思紀錄。")

    # 隱藏的推論過程檢視 (細節)
    st.divider()
    st.subheader("🔍 當下推論邏輯檢視 (避免 Look-Ahead Bias)")
    selected_trade = st.selectbox("選擇要檢視的事件", trades_df['question'].unique())
    if selected_trade:
        trade_details = trades_df[trades_df['question'] == selected_trade].iloc[0]
        st.markdown(f"**建議行動:** {trade_details['action']} | **預估勝率:** {trade_details['predicted_prob']*100:.1f}% | **防爆倉下注:** 總資產 {trade_details.get('kelly_fraction', 0)*100:.1f}%")
        st.write("--- 當下掌握的情報 (Context) ---")
        st.text(trade_details.get('context_at_time', 'N/A'))
        st.write("--- 辯論最終觀點 (Reasoning) ---")
        st.markdown(trade_details.get('reasoning', 'N/A'))
