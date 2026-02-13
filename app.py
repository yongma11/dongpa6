# 파일명: app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import requests # [패치] 요청 헤더 조작용

# ---------------------------------------------------------
# 1. 페이지 설정 & 스타일
# ---------------------------------------------------------
st.set_page_config(page_title="동파법 마스터 v5.6", page_icon="💎", layout="wide")

PARAMS = {
    'Safe':    {'buy': 3.0, 'sell': 0.5, 'time': 35, 'desc': '🛡️ 방어 (Safe)'},
    'Offense': {'buy': 5.0, 'sell': 3.0, 'time': 7,  'desc': '⚔️ 공세 (Offense)'}
}
MAX_SLOTS = 7
RESET_CYCLE = 10

# GitHub 설정
try:
    GH_TOKEN = st.secrets["general"]["GH_TOKEN"]
except:
    st.error("🚨 GitHub 토큰 오류: Streamlit Secrets에 GH_TOKEN을 설정해주세요.")
    st.stop()

REPO_KEY = "yongma11/dongpa6" 
HOLDINGS_FILE = "my_holdings.csv"
JOURNAL_FILE = "trading_journal.csv"
EQUITY_FILE = "equity_history.csv"
SETTINGS_FILE = "settings.json"

# ---------------------------------------------------------
# 2. 데이터 & 엔진 함수 (Connection Fix)
# ---------------------------------------------------------
@st.cache_data(ttl=300)
def get_data_final(period='max'):
    try:
        start_date = '2010-01-01'
        
        # [패치] 세션 헤더 조작 (봇 차단 우회)
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })

        def fetch_ticker(symbol):
            # yf.download 대신 Ticker.history 사용 (더 안정적)
            ticker = yf.Ticker(symbol, session=session)
            df = ticker.history(start=start_date, auto_adjust=False)
            return df['Close']

        qqq_close = fetch_ticker("QQQ")
        soxl_close = fetch_ticker("SOXL")
        
        if qqq_close.empty or soxl_close.empty: return None

        # 데이터 병합
        df = pd.DataFrame({'QQQ': qqq_close, 'SOXL': soxl_close})
        df = df.ffill().bfill().dropna()
        df.index = df.index.tz_localize(None)
        
        return df

    except Exception as e:
        print(f"Data Load Error: {e}")
        return None

def calc_mode_series(df_qqq):
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
        rsi_t1 = rsi_series.iloc[i-1]
        rsi_t2 = rsi_series.iloc[i-2]
        if np.isnan(rsi_t1) or np.isnan(rsi_t2):
            modes.append(current_mode)
            continue
        safe = ((rsi_t2 > 65) and (rsi_t2 > rsi_t1)) or ((40 < rsi_t2 < 50) and (rsi_t2 > rsi_t1)) or ((rsi_t1 < 50) and (rsi_t2 > 50))
        offense = ((rsi_t2 < 35) and (rsi_t2 < rsi_t1)) or ((50 < rsi_t2 < 60) and (rsi_t2 < rsi_t1)) or ((rsi_t1 > 50) and (rsi_t2 < 50))
        if safe: current_mode = 'Safe'
        elif offense: current_mode = 'Offense'
        modes.append(current_mode)
    
    weekly_mode = pd.Series(modes, index=qqq_weekly.index)
    return weekly_mode.resample('D').ffill(), rsi_series.resample('D').ffill()

def get_repo():
    g = Github(GH_TOKEN)
    try:
        return g.get_repo(REPO_KEY)
    except:
        return None

def load_settings():
    try:
        repo = get_repo()
        if repo:
            contents = repo.get_contents(SETTINGS_FILE)
            return json.loads(contents.decoded_content.decode("utf-8"))
    except:
        pass
    return {"start_date": "2025-01-01", "init_cap": 100000.0}

def save_settings(settings_dict):
    try:
        repo = get_repo()
        if repo:
            json_str = json.dumps(settings_dict)
            try:
                contents = repo.get_contents(SETTINGS_FILE)
                repo.update_file(contents.path, "Update settings", json_str, contents.sha)
            except:
                repo.create_file(SETTINGS_FILE, "Create settings", json_str)
    except Exception as e:
        print(f"설정 저장 실패: {e}")

def load_csv(filename, columns):
    try:
        repo = get_repo()
        if repo:
            try:
                contents = repo.get_contents(filename)
                csv_string = contents.decoded_content.decode("utf-8")
                return pd.read_csv(StringIO(csv_string))
            except:
                pass
    except:
        pass
    return pd.DataFrame(columns=columns)

def save_csv(df, filename):
    try:
        repo = get_repo()
        if repo:
            csv_string = df.to_csv(index=False)
            try:
                contents = repo.get_contents(filename)
                repo.update_file(contents.path, f"Update {filename}", csv_string, contents.sha)
            except:
                repo.create_file(filename, f"Create {filename}", csv_string)
    except Exception as e:
        st.error(f"GitHub 저장 실패: {e}")

def auto_sync_engine(df, start_date, init_cap):
    mode_daily, _ = calc_mode_series(df['QQQ'])
    sim_df = pd.concat([df['SOXL'], mode_daily], axis=1).dropna()
    sim_df.columns = ['Price', 'Mode']
    end_date = datetime.now() - timedelta(days=1)
    mask = (sim_df.index >= pd.to_datetime(start_date)) & (sim_df.index <= pd.to_datetime(end_date))
    sim_df = sim_df[mask]
    if sim_df.empty: return None, None, None

    sim_df['Prev_Price'] = sim_df['Price'].shift(1)
    sim_df = sim_df.dropna()

    real_cash = init_cap
    cum_profit = 0.0
    cum_loss = 0.0
    slots = []
    journal = []
    daily_equity = []
    
    cycle_days = 0
    local_params = {'Safe': {'buy': 0.03, 'sell': 1.005, 'time': 35}, 'Offense': {'buy': 0.05, 'sell': 1.03, 'time': 7}}

    for date, row in sim_df.iterrows():
        price = row['Price']
        mode = row['Mode']
        
        cycle_days += 1
        if cycle_days >= 10:
            virtual = init_cap + (cum_profit * 0.7) - (cum_loss * 0.6)
            if virtual < 1000: virtual = 1000
            current_slot_size = virtual / 7
            cycle_days = 0
        else:
            if 'current_slot_size' not in locals(): current_slot_size = init_cap / 7

        sold_idx = []
        for i in range(len(slots)-1, -1, -1):
            s = slots[i]
            s['days'] += 1
            rule = local_params.get(s['birth_mode'], local_params['Safe'])
            if (price >= s['buy_price'] * rule['sell']) or (s['days'] >= rule['time']):
                rev = s['shares'] * price
                prof = rev - (s['shares'] * s['buy_price'])
                current_holdings_val = sum(slots[k]['shares'] * price for k in range(len(slots)) if k != i)
                equity_at_sell = real_cash + rev + current_holdings_val
                journal.append({
                    "날짜": date.date(), "총자산": equity_at_sell, "수익금": prof,
                    "수익률": (prof / (equity_at_sell - prof)) * 100 if (equity_at_sell - prof) > 0 else 0
                })
                real_cash += rev
                if prof > 0: cum_profit += prof
                else: cum_loss += abs(prof)
                sold_idx.append(i)
        for i in sold_idx: del slots[i]
        
        chg = (price - row['Prev_Price']) / row['Prev_Price']
        curr_rule = local_params.get(mode, local_params['Safe'])
        if chg <= curr_rule['buy']:
            if (len(slots) < 7) or (real_cash >= current_slot_size * 0.98):
                amt = min(real_cash, current_slot_size)
                if amt > 10:
                    shares = amt / price
                    real_cash -= amt
                    tr = PARAMS[mode]
                    tg = price * (1 + tr['sell']/100)
                    cd = date + timedelta(days=tr['time']*1.45)
                    slots.append({
                        '매수일': date.date(), '모드': mode, '매수가': price, '수량': int(shares),
                        '목표가': tg, '손절기한': cd.date(), 'buy_price': price, 'shares': int(shares), 'days': 0, 'birth_mode': mode
                    })
        
        total_holdings_value = sum(s['shares'] * price for s in slots)
        daily_total_equity = real_cash + total_holdings_value
        daily_equity.append({"날짜": date.date(), "총자산": daily_total_equity})
    
    final_holdings = []
    for s in slots:
        final_holdings.append({
            "매수일": s['매수일'], "모드": s['모드'], "매수가": s['매수가'], 
            "수량": s['수량'], "목표가": s['목표가'], "손절기한": s['손절기한']
        })
    
    return pd.DataFrame(final_holdings), pd.DataFrame(journal), pd.DataFrame(daily_equity)

def run_backtest_fixed(df, start_date, end_date, init_cap):
    mode_daily, rsi_daily = calc_mode_series(df['QQQ'])
    sim_df = pd.concat([df['SOXL'], mode_daily, rsi_daily], axis=1).dropna()
    sim_df.columns = ['Price', 'Mode', 'RSI']
    mask = (sim_df.index >= pd.to_datetime(start_date)) & (sim_df.index <= pd.to_datetime(end_date))
    sim_df = sim_df[mask]
    if sim_df.empty: return None, None, None, None
    sim_df['Prev_Price'] = sim_df['Price'].shift(1)
    sim_df = sim_df.dropna()
    
    real_cash = init_cap
    cum_profit = 0.0
    cum_loss = 0.0
    slots = []
    equity_curve = []
    debug_logs = []
    cycle_days = 0
    gross_profit = 0.0
    gross_loss = 0.0
    local_params = {'Safe': {'buy': 0.03, 'sell': 1.005, 'time': 35}, 'Offense': {'buy': 0.05, 'sell': 1.03, 'time': 7}}
    
    for date, row in sim_df.iterrows():
        price = row['Price']
        mode = row['Mode']
        rsi_val = row['RSI']
        cycle_days += 1
        if cycle_days >= 10:
            virtual = init_cap + (cum_profit * 0.7) - (cum_loss * 0.6)
            if virtual < 1000: virtual = 1000
            current_slot_size = virtual / 7
            cycle_days = 0
        else:
            if 'current_slot_size' not in locals(): current_slot_size = init_cap / 7
        action_today = "관망"
        sold_idx = []
        for i in range(len(slots)-1, -1, -1):
            s = slots[i]
            s['days'] += 1
            rule = local_params.get(s['birth_mode'], local_params['Safe'])
            if (price >= s['buy_price'] * rule['sell']) or (s['days'] >= rule['time']):
                rev = s['shares'] * price
                prof = rev - (s['shares'] * s['buy_price'])
                real_cash += rev
                if prof > 0: 
                    cum_profit += prof
                    gross_profit += prof
                else: 
                    cum_loss += abs(prof)
                    gross_loss += abs(prof)
                sold_idx.append(i)
                action_today = "매도 (익절/손절)"
        for i in sold_idx: del slots[i]
        chg = (price - row['Prev_Price']) / row['Prev_Price']
        curr_rule = local_params.get(mode, local_params['Safe'])
        if chg <= curr_rule['buy']:
            if (len(slots) < 7) or (real_cash >= current_slot_size * 0.98):
                amt = min(real_cash, current_slot_size)
                if amt > 10:
                    shares = amt / price
                    real_cash -= amt
                    slots.append({'buy_price': price, 'shares': shares, 'days': 0, 'birth_mode': mode})
                    action_today = "매수 (LOC)"
        current_equity = real_cash + sum(s['shares']*price for s in slots)
        equity_curve.append({'Date': date, 'Equity': current_equity})
        debug_logs.append({"날짜": date.date(), "RSI (주봉)": f"{rsi_val:.2f}", "적용 모드": mode, "SOXL 종가": f"${price:.2f}", "매매 행동": action_today, "총 자산": f"${current_equity:,.0f}"})
    
    res_df = pd.DataFrame(equity_curve).set_index('Date')
    df_debug = pd.DataFrame(debug_logs).set_index("날짜")
    
    if not res_df.empty:
        res_df['Returns'] = res_df['Equity'].pct_change()
        downside_returns = res_df.loc[res_df['Returns'] < 0, 'Returns']
        downside_std = downside_returns.std() * np.sqrt(252)
        total_ret = (res_df['Equity'].iloc[-1] / init_cap) - 1
        days = (res_df.index[-1] - res_df.index[0]).days
        cagr = (1 + total_ret) ** (365 / days) - 1 if days > 0 else 0
        sortino = cagr / downside_std if downside_std > 0 else 0
        metrics = {'profit_factor': gross_profit / gross_loss if gross_loss > 0 else 99.9, 'sortino': sortino}
    else:
        metrics = {'profit_factor': 0, 'sortino': 0}

    yearly_stats = []
    years = res_df.index.year.unique()
    def calc_mdd(series):
        peak = series.cummax()
        dd = (series - peak) / peak
        return dd.min()
    prev_equity = init_cap
    for yr in years:
        df_yr = res_df[res_df.index.year == yr]
        end_equity = df_yr['Equity'].iloc[-1]
        yr_return = (end_equity - prev_equity) / prev_equity
        yr_mdd = calc_mdd(df_yr['Equity'])
        yearly_stats.append({"연도": yr, "수익률": yr_return, "MDD": yr_mdd, "기말자산": end_equity})
        prev_equity = end_equity
    return res_df, metrics, pd.DataFrame(yearly_stats).set_index("연도"), df_debug

# ---------------------------------------------------------
# 3. 메인 UI
# ---------------------------------------------------------
def main():
    st.title("💎 동파법 마스터 v5.6 (Connection Fix)")
    
    tab_trade, tab_backtest, tab_logic = st.tabs(["💎 실전 트레이딩", "🧪 백테스트", "📚 전략 로직"])

    with st.spinner("데이터 로딩 중..."):
        df = get_data_final()
    
    if df is None:
        st.error("📉 주식 데이터 연결 실패 (Yahoo Finance Blocking)")
        st.warning("👉 잠시 후(1분 뒤) F5를 눌러 다시 시도하거나, GitHub의 requirements.txt 버전을 확인하세요.")
        st.stop()
    
    mode_s, rsi_s = calc_mode_series(df['QQQ'])
    curr_mode = mode_s.iloc[-1]
    curr_rsi = rsi_s.iloc[-1]
    soxl_price = df['SOXL'].iloc[-1]
    prev_close = df['SOXL'].iloc[-2]

    settings = load_settings()
    if 'auto_run_done' not in st.session_state: st.session_state['auto_run_done'] = False

    try:
        saved_start_date = datetime.strptime(settings.get("start_date", "2025-01-01"), "%Y-%m-%d").date()
        saved_init_cap = float(settings.get("init_cap", 100000.0))
    except:
        saved_start_date = datetime(2025, 1, 1).date()
        saved_init_cap = 100000.0

    if 'holdings' not in st.session_state or not st.session_state['auto_run_done']:
        h_auto, j_auto, eq_auto = auto_sync_engine(df, saved_start_date, saved_init_cap)
        if h_auto is not None:
            old_h = load_csv(HOLDINGS_FILE, h_auto.columns)
            if len(h_auto) != len(old_h) or (not old_h.empty and str(h_auto.iloc[-1].values) != str(old_h.iloc[-1].values)):
                save_csv(h_auto, HOLDINGS_FILE)
                save_csv(j_auto, JOURNAL_FILE)
                save_csv(eq_auto, EQUITY_FILE)
            st.session_state['holdings'] = h_auto
            st.session_state['journal'] = j_auto
            st.session_state['equity_history'] = eq_auto
            st.session_state['auto_run_done'] = True
    
    with tab_trade:
        with st.sidebar:
            st.header("🤖 설정 및 초기화")
            auto_start_date = st.date_input("전략 시작일", value=saved_start_date)
            auto_init_cap = st.number_input("시작 원금 ($)", value=saved_init_cap, step=100.0)
            if st.button("🔄 설정 변경 및 재동기화", type="primary"):
                new_settings = {"start_date": auto_start_date.strftime("%Y-%m-%d"), "init_cap": auto_init_cap}
                save_settings(new_settings)
                st.session_state['auto_run_done'] = False
                st.rerun()
            st.markdown("---")
            if st.button("🗑️ 데이터 초기화"):
                empty_df = pd.DataFrame(columns=["매수일", "모드", "매수가", "수량", "목표가", "손절기한"])
                empty_j = pd.DataFrame(columns=["날짜", "총자산", "수익금", "수익률"])
                empty_eq = pd.DataFrame(columns=["날짜", "총자산"])
                save_csv(empty_df, HOLDINGS_FILE)
                save_csv(empty_j, JOURNAL_FILE)
                save_csv(empty_eq, EQUITY_FILE)
                st.session_state['holdings'] = empty_df
                st.session_state['journal'] = empty_j
                st.session_state['equity_history'] = empty_eq
                st.rerun()

            today = datetime.now().date()
            cycle = ((today - saved_start_date).days % RESET_CYCLE) + 1
            st.info(f"🔄 사이클: **{cycle}일차** / 10일")

        r = PARAMS[curr_mode]
        slot_sz = saved_init_cap / MAX_SLOTS
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("시장 모드", f"{r['desc']}", f"RSI {curr_rsi:.2f}", delta_color="inverse")
        c2.metric("SOXL 현재가", f"${soxl_price:.2f}", f"{((soxl_price-prev_close)/prev_close)*100:.2f}%")
        c3.metric("1슬롯 할당금", f"${slot_sz:,.0f}")
        c4.metric("매매 사이클", f"{cycle}일차")
        st.markdown("---")

        order_date_str = today.strftime("%Y-%m-%d")
        st.subheader(f"📋 오늘의 주문 (Today's Orders - {order_date_str})")
        
        df_h = st.session_state['holdings']
        sell_orders = []
        buy_orders = []
        
        if not df_h.empty:
            df_h['손절기한'] = pd.to_datetime(df_h['손절기한']).dt.date
            for idx, row in df_h.iterrows():
                if row['손절기한'] <= today:
                    sell_orders.append(f"**[매도]** 티어{idx+1}: **{row['수량']}주** (시장가) - **MOC (기간만료)**")
                else:
                    sell_orders.append(f"**[매도]** 티어{idx+1}: **{row['수량']}주** (${row['목표가']:.2f}) - **LOC (익절)**")
        
        if soxl_price > 0:
            b_lim = prev_close * (1 + r['buy']/100)
            b_qty = int(slot_sz / soxl_price)
            buy_orders.append(f"**[매수]** 신규: **{b_qty}주 (예상)** (${b_lim:.2f}) - **LOC (진입)**")
            
        if not sell_orders and not buy_orders:
            st.info("오늘 예정된 주문이 없습니다. (No Orders)")
        else:
            if sell_orders:
                for order in sell_orders: st.error(order)
            if buy_orders:
                for order in buy_orders: st.success(order)

        st.markdown("---")

        st.subheader("📊 나의 티어 현황 (Cloud 저장)")
        if not df_h.empty:
            df_h['매수일'] = pd.to_datetime(df_h['매수일']).dt.date
            df_h.index = range(1, len(df_h) + 1)
            df_h.index.name = "티어"
            current_yields = ((soxl_price - df_h['매수가']) / df_h['매수가'] * 100)
            yield_display = [f"{'🔺' if y > 0 else '🔻'} {y:.2f} %" for y in current_yields]
            df_h['수익률'] = yield_display
            status_list = ["🚨 MOC 매도" if row['손절기한'] <= today else "🔵 LOC 대기" for _, row in df_h.iterrows()]
            df_h['상태'] = status_list

            total_qty = edited_h['수량'].sum() if 'edited_h' in locals() else df_h['수량'].sum()
            total_invested = (df_h['매수가'] * df_h['수량']).sum()
            avg_price = total_invested / total_qty if total_qty > 0 else 0
            current_val = total_qty * soxl_price
            total_profit = current_val - total_invested
            total_yield_pct = (total_profit / total_invested * 100) if total_invested > 0 else 0
            
            st.markdown("#### 📌 전체 계좌 요약")
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("총 보유수량", f"{total_qty} 주")
            sc2.metric("통합 평단가", f"${avg_price:,.2f}")
            sc3.metric("총 평가손익", f"${total_profit:,.2f}", delta_color="normal")
            sc4.metric("평균 수익률", f"{total_yield_pct:,.2f}%", delta_color="normal")
            
            st.markdown("👇 **보유 티어 상세 내역 (편집 가능)**")
            edited_h = st.data_editor(
                df_h, num_rows="dynamic", use_container_width=True, key="h_edit",
                column_config={"수익률": st.column_config.TextColumn("수익률", disabled=True), "매수가": st.column_config.NumberColumn(format="$%.2f"), "목표가": st.column_config.NumberColumn(format="$%.1f"), "상태": st.column_config.TextColumn(disabled=True)}
            )
            if st.button("💾 티어 수정 저장 (GitHub)"):
                save_cols = ["매수일", "모드", "매수가", "수량", "목표가", "손절기한"]
                save_csv(edited_h[save_cols], HOLDINGS_FILE)
                st.session_state['holdings'] = edited_h[save_cols]
                st.success("저장되었습니다!")
                st.rerun()
        else: st.info("현재 보유 중인 티어가 없습니다.")
        
        st.markdown("---")
        
        st.subheader("📝 매매 수익 기록장 (Cloud 저장)")
        df_j = st.session_state['journal']
        df_eq = st.session_state['equity_history']
        init_prin = saved_init_cap
        
        if not df_j.empty:
            df_j['날짜'] = pd.to_datetime(df_j['날짜']).dt.date
            df_j = df_j.sort_values(by="날짜", ascending=True).reset_index(drop=True)
            total_prof_j = df_j['수익금'].sum()
            total_yield_j = (total_prof_j / init_prin * 100)
            
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("🏁 시작 원금", f"${init_prin:,.0f}")
            mc2.metric("💰 누적 수익금", f"${total_prof_j:,.2f}", delta_color="normal")
            mc3.metric("📈 총 수익률", f"{total_yield_j:.1f}%", delta_color="normal")
            
            st.markdown("")
            start_date_display = saved_start_date.strftime("%Y-%m-%d")
            # [UI] 과거 매매 기록은 접이식으로 제공
            with st.expander(f"📜 전략 시작일({start_date_display}) 이후 매매 기록 보기", expanded=False):
                df_history = df_j[df_j['날짜'] >= saved_start_date].sort_values(by="날짜", ascending=False).reset_index(drop=True)
                st.dataframe(
                    df_history, 
                    use_container_width=True,
                    column_config={
                        "수익금": st.column_config.NumberColumn(format="$%.2f"),
                        "수익률": st.column_config.NumberColumn(format="%.2f %%"),
                        "총자산": st.column_config.NumberColumn(format="$%.0f"),
                    }
                )

            st.markdown("### 📈 내 자산 성장 그래프 (Equity Curve)")
            if not df_eq.empty:
                df_eq['날짜'] = pd.to_datetime(df_eq['날짜'])
                df_eq = df_eq.sort_values(by="날짜")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df_eq['날짜'], df_eq['총자산'], color='#4CAF50', linewidth=2)
                ax.fill_between(df_eq['날짜'], df_eq['총자산'], init_prin, where=(df_eq['총자산'] >= init_prin), color='#4CAF50', alpha=0.1)
                ax.fill_between(df_eq['날짜'], df_eq['총자산'], init_prin, where=(df_eq['총자산'] < init_prin), color='red', alpha=0.1)
                ax.axhline(y=init_prin, color='gray', linestyle='--', alpha=0.5, label='원금')
                ax.set_title("Total Equity Growth", fontweight='bold')
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
                st.pyplot(fig)
            else: st.info("그래프 데이터가 없습니다.")
        else: st.info("실현된 수익 없음.")

    with tab_backtest:
        st.header("🧪 백테스트 성과분석")
        # 백테스트 코드는 그대로 유지 (분량상 생략 없이 포함됨)
        # ... (생략 없이 v5.3의 백테스트 코드 그대로 사용)
        # (전체 코드를 복사하실 때 이 부분도 자동으로 포함되어 있으니 걱정 마세요!)
        bt_init_cap = st.number_input("백테스트 초기 자본 ($)", value=10000.0, step=1000.0)
        bc1, bc2 = st.columns(2)
        start_d = bc1.date_input("검증 시작일", value=datetime(2010, 1, 1), min_value=datetime(2000, 1, 1))
        end_d = bc2.date_input("검증 종료일", value=today, min_value=datetime(2000, 1, 1))
        
        if st.button("🚀 분석 실행"):
            with st.spinner("분석 중..."):
                res, metrics, df_yearly, df_debug = run_backtest_fixed(df, start_d, end_d, bt_init_cap)
                if res is not None:
                    final = res['Equity'].iloc[-1]
                    ret = (final/bt_init_cap) - 1
                    days = (res.index[-1] - res.index[0]).days
                    cagr = (1+ret)**(365/days) - 1 if days > 0 else 0
                    res['Peak'] = res['Equity'].cummax()
                    res['Drawdown'] = (res['Equity'] - res['Peak']) / res['Peak']
                    mdd = res['Drawdown'].min()
                    calmar = cagr / abs(mdd) if mdd != 0 else 0
                    
                    m1, m2, m3, m4, m5, m6 = st.columns(6)
                    m1.metric("최종 수익금", f"${final:,.0f}", f"{ret*100:,.1f}%")
                    m2.metric("CAGR", f"{cagr*100:.2f}%")
                    m3.metric("MDD", f"{mdd*100:.2f}%", delta_color="inverse")
                    m4.metric("Calmar", f"{calmar:.2f}")
                    m5.metric("Sortino", f"{metrics['sortino']:.2f}")
                    m6.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
                    
                    st.markdown("#### 📊 통합 성과 차트")
                    plt.style.use('default')
                    fig, ax1 = plt.subplots(figsize=(12, 6))
                    color = 'tab:blue'
                    ax1.set_xlabel('Date')
                    ax1.set_ylabel('Total Equity ($)', color=color, fontweight='bold')
                    ax1.plot(res.index, res['Equity'], color=color, linewidth=1.5, label='Equity')
                    ax1.tick_params(axis='y', labelcolor=color)
                    ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
                    ax1.grid(True, linestyle='--', alpha=0.3)
                    ax2 = ax1.twinx()
                    color = 'tab:red'
                    ax2.set_ylabel('Drawdown (%)', color=color, fontweight='bold')
                    ax2.fill_between(res.index, res['Drawdown']*100, 0, color=color, alpha=0.2, label='Drawdown')
                    ax2.tick_params(axis='y', labelcolor=color)
                    ax2.set_ylim(-100, 5)
                    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
                    plt.title(f"Portfolio Performance vs Risk", fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.markdown("#### 📅 연도별 성과표")
                    df_yearly_fmt = df_yearly.copy()
                    df_yearly_fmt['수익률'] = df_yearly_fmt['수익률'].apply(lambda x: f"{x*100:.1f}%")
                    df_yearly_fmt['MDD'] = df_yearly_fmt['MDD'].apply(lambda x: f"{x*100:.1f}%")
                    df_yearly_fmt['기말자산'] = df_yearly_fmt['기말자산'].apply(lambda x: f"${x:,.0f}")
                    st.dataframe(df_yearly_fmt.T, use_container_width=True)
                    
                    st.markdown("#### 🔍 상세 매매 및 지표 로그 (Debug Log)")
                    st.dataframe(df_debug.sort_index(ascending=False), use_container_width=True)
                    
                else: st.error("데이터 부족")

    with tab_logic:
        st.header("📚 동파법(Dongpa) 전략 매뉴얼 (상세)")
        st.markdown("""
        ### 1. 전략 개요 (Philosophy)
        * **핵심:** "시장의 계절(Mode)을 먼저 파악하고, 그에 맞는 옷(Rule)을 입는다."
        * **대상:** SOXL (3배 레버리지) / **지표:** QQQ (나스닥100)
        * **특징:** 예측보다는 **대응**에 초점을 맞춘 변동성 돌파 & 추세 추종 하이브리드 전략.

        ---

        ### 2. 시장 모드 판단 (Market Modes)
        매주 금요일 종가 기준으로 **QQQ 주봉 RSI(14)**를 분석하여 다음 주의 모드를 결정합니다.

        | 모드 | 조건 (Condition) | 시장 상황 해석 |
        | :--- | :--- | :--- |
        | **🛡️ Safe** | `RSI > 65` & `하락` | 고점 과열 후 꺾임 (조정 임박) |
        | **🛡️ Safe** | `40 < RSI < 50` & `하락` | 약세장에서의 지속 하락 |
        | **🛡️ Safe** | `50선 하향 돌파` | 추세가 꺾이는 데드크로스 |
        | **⚔️ Offense** | `RSI < 35` & `상승` | 과매도권에서의 바닥 반등 |
        | **⚔️ Offense** | `50 < RSI < 60` & `상승` | 전형적인 상승 추세 |
        | **⚔️ Offense** | `50선 상향 돌파` | 추세가 살아나는 골든크로스 |
        
        * **유지(Hold):** 위 조건에 해당하지 않으면 **직전 주의 모드를 그대로 유지**합니다.

        ---

        ### 3. 실전 매매 규칙 (Action Rules)
        **중요:** 매수 체결 당시의 모드 규칙을 매도 시까지 유지합니다 (Sticky Rule).

        | 구분 | 🛡️ 방어 (Safe) | ⚔️ 공세 (Offense) |
        | :--- | :--- | :--- |
        | **매수 타점** | 전일 종가 대비 **-3.0%** | 전일 종가 대비 **-5.0%** |
        | **익절 목표** | 매수가 대비 **+0.5%** | 매수가 대비 **+3.0%** |
        | **손절 기한** | **35 거래일** | **7 거래일** |
        
        #### 🛒 주문 방식 (Order Types)
        * **매수:** **LOC (Limit On Close)** - 장 마감 종가가 타점 이하일 때만 체결.
        * **익절 매도:** **LOC (Limit On Close)** - 장 마감 종가가 목표가 이상일 때만 체결 (장중 휩소 방지).
        * **기간 만료 매도:** **MOC (Market On Close)** - 손절 기한 도래 시 장 마감 시장가로 무조건 청산.

        ---

        ### 4. 자금 관리 (Money Management)
        * **7분할:** 총 자금을 7개 슬롯으로 분할 투입하여 리스크를 분산합니다.
        * **10일 리셋:** 2주(10거래일)마다 총 자산 기준으로 슬롯 크기를 재산정하여 복리 효과를 극대화합니다.
        """)

if __name__ == "__main__":
    main()
