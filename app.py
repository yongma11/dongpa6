# 파일명: app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ---------------------------------------------------------
# 1. 설정 및 파라미터 (최종 확정값)
# ---------------------------------------------------------
st.set_page_config(page_title="동파법 매매 비서", page_icon="📈", layout="wide")

PARAMS = {
    'Safe':    {'buy': 3.0, 'sell': 0.5, 'time': 35, 'desc': '방어 모드'},
    'Offense': {'buy': 5.0, 'sell': 3.0, 'time': 7,  'desc': '공세 모드'}
}

MAX_SLOTS = 7
RESET_CYCLE = 10

# ---------------------------------------------------------
# 2. 데이터 처리 함수 (캐싱 적용)
# ---------------------------------------------------------
@st.cache_data(ttl=3600) # 1시간마다 데이터 갱신
def get_data():
    try:
        df = yf.download(['QQQ', 'SOXL'], period='2y', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs('Close', level='Price', axis=1)
        return df
    except Exception as e:
        st.error(f"데이터 다운로드 실패: {e}")
        return None

def calc_mode_excel_logic(df_qqq):
    # 엑셀 로직 (유지 기능 포함)
    qqq_weekly = df_qqq.resample('W-FRI').last()
    
    delta = qqq_weekly.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    rsi_series = 100 - (100 / (1 + rs))
    
    modes = []
    current_mode = 'Safe'
    
    for i in range(len(rsi_series)):
        if i < 2:
            modes.append(current_mode)
            continue
        
        rsi_t1 = rsi_series.iloc[i-1] # 지난주
        rsi_t2 = rsi_series.iloc[i-2] # 지지난주
        
        if np.isnan(rsi_t1) or np.isnan(rsi_t2):
            modes.append(current_mode)
            continue
            
        safe_cond = (
            (rsi_t2 > 65 and rsi_t2 > rsi_t1) or
            (40 < rsi_t2 < 50 and rsi_t2 > rsi_t1) or
            (rsi_t1 < 50 and rsi_t2 > 50)
        )
        offense_cond = (
            (rsi_t2 < 35 and rsi_t2 < rsi_t1) or
            (50 < rsi_t2 < 60 and rsi_t2 < rsi_t1) or
            (rsi_t1 > 50 and rsi_t2 < 50)
        )
        
        if safe_cond: current_mode = 'Safe'
        elif offense_cond: current_mode = 'Offense'
        # else: pass (유지)
        
        modes.append(current_mode)
        
    return modes[-1], rsi_series.iloc[-1]

# ---------------------------------------------------------
# 3. UI 구성
# ---------------------------------------------------------
def main():
    st.title("🤖 동파법 실전 트레이딩 센터")
    st.markdown("---")

    # [사이드바] 사용자 입력
    with st.sidebar:
        st.header("⚙️ 내 자산 설정")
        current_capital = st.number_input("현재 총 평가금 ($)", value=10000.0, step=100.0, format="%.2f")
        start_date = st.date_input("매매 시작일", value=datetime(2026, 1, 1))
        
        # 사이클 계산
        today = datetime.now().date()
        days_passed = (today - start_date).days
        cycle_day = (days_passed % RESET_CYCLE) + 1 # 1일차 ~ 10일차
        
        st.info(f"""
        **🗓️ 사이클 상태**
        - 진행: {days_passed}일째
        - 현재: **{cycle_day}일차** / 10일
        """)
        
        if cycle_day == 1 or cycle_day == 10:
            st.warning("🔔 자금 리셋 주기입니다! 총 평가금을 업데이트하세요.")

    # [데이터 로드]
    df = get_data()
    if df is None: return

    mode, rsi_val = calc_mode_excel_logic(df['QQQ'])
    soxl_price = df['SOXL'].iloc[-1]
    soxl_prev_close = df['SOXL'].iloc[-2]
    
    # 파라미터 로드
    rule = PARAMS[mode]

    # [상단 정보창]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("시장 모드", f"{rule['desc']} ({mode})", delta=f"RSI {rsi_val:.2f}", delta_color="inverse")
    with col2:
        st.metric("SOXL 현재가", f"${soxl_price:.2f}", delta=f"{((soxl_price-soxl_prev_close)/soxl_prev_close)*100:.2f}%")
    with col3:
        slot_size = current_capital / MAX_SLOTS
        st.metric("1슬롯 투자금 (7분할)", f"${slot_size:,.0f}")
    with col4:
        st.metric("매매 사이클", f"{cycle_day}일차")

    st.markdown("---")

    # [탭 구성] 매수 / 매도 관리
    tab1, tab2 = st.tabs(["🛒 신규 매수 (Buy)", "💰 보유 매도 (Sell)"])

    # -----------------------------------------------------
    # TAB 1: 매수 가이드
    # -----------------------------------------------------
    with tab1:
        st.subheader("오늘의 매수 주문표")
        
        limit_price = soxl_prev_close * (1 + rule['buy']/100)
        buy_qty = int(slot_size / soxl_price)
        
        # 디자인 박스
        buy_col1, buy_col2 = st.columns([1, 2])
        
        with buy_col1:
            st.success(f"""
            ### **${limit_price:.2f}**
            **매수 상한가 (LOC)**
            """)
        
        with buy_col2:
            st.markdown(f"""
            * **전일 종가:** ${soxl_prev_close:.2f}
            * **조건:** 전일비 **+{rule['buy']}%** 이하 상승 시
            * **주문 수량:** 약 **{buy_qty}주** (${slot_size:,.0f} 기준)
            """)
            
        st.markdown("💡 **Tip:** 장 시작 전 `LOC 매수`로 주문을 걸어두면 자동 체결됩니다.")
        
        # 보너스 매수 로직 설명
        with st.expander("ℹ️ 보너스 매수 조건 확인"):
            st.write("슬롯이 꽉 찼더라도, **현재 예수금이 1슬롯($"+f"{slot_size:,.0f}"+") 이상** 남아있다면 추가 매수가 가능합니다.")

    # -----------------------------------------------------
    # TAB 2: 매도 가이드 (계산기)
    # -----------------------------------------------------
    with tab2:
        st.subheader("보유 종목 매도 관리 (Sticky Mode)")
        st.markdown("⚠️ 매도는 **'매수했을 당시의 모드'**를 따라야 합니다. 매수 기록을 확인하세요.")
        
        # 인터랙티브 계산기
        with st.container(border=True):
            col_in1, col_in2, col_in3 = st.columns(3)
            with col_in1:
                my_buy_price = st.number_input("내 평단가 (매수가)", value=soxl_price)
            with col_in2:
                my_buy_date = st.date_input("매수 체결일", value=datetime.now())
            with col_in3:
                origin_mode = st.selectbox("매수 당시 모드", ["Safe", "Offense"])
            
            # 계산 로직
            sell_rule = PARAMS[origin_mode]
            target_price = my_buy_price * (1 + sell_rule['sell']/100)
            cut_date = my_buy_date + timedelta(days=sell_rule['time']*1.5) # 영업일 대략 계산 (여유있게)
            
            st.markdown("#### 👇 당신의 매도 목표")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric("익절 목표가 (LOC 매도)", f"${target_price:.2f}", f"+{sell_rule['sell']}%")
            with res_col2:
                st.metric("손절 기한 (Time Cut)", f"{sell_rule['time']} 거래일 뒤", f"약 {cut_date.strftime('%Y-%m-%d')} 까지")

if __name__ == "__main__":
    main()