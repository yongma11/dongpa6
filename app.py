# 파일명: app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from github import Github
from io import StringIO
import json

# ---------------------------------------------------------
# 1. 페이지 설정 & 스타일
# ---------------------------------------------------------
st.set_page_config(page_title="동파법 마스터 v4.4", page_icon="💎", layout="wide")

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

# 👇 사용자 설정
REPO_KEY = "yongma11/dongpa6" 

HOLDINGS_FILE = "my_holdings.csv"
JOURNAL_FILE = "trading_journal.csv"
SETTINGS_FILE = "settings.json"

# ---------------------------------------------------------
# 2. 데이터 & 엔진 함수
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def get_data_final(period='max'):
    try:
        start_date = '2010-01-01'
        qqq = yf.download("QQQ", start=start_date, progress=False, auto_adjust=False)
        soxl = yf.download("SOXL", start=start_date, progress=False, auto_adjust=False)
        
        if qqq.empty or soxl.empty: return None

        if isinstance(qqq.columns, pd.MultiIndex): qqq = qqq.xs('Close', level=0, axis=1)
        else: qqq = qqq['Close']
        
        if isinstance(soxl.columns, pd.MultiIndex): soxl = soxl.xs('Close', level=0, axis=1)
        else: soxl = soxl['Close']
        
        if 'QQQ' in qqq.columns: qqq = qqq['QQQ']
        elif len(qqq.columns) > 0: qqq = qqq.iloc[:, 0]
            
        if 'SOXL' in soxl.columns: soxl = soxl['SOXL']
        elif len(soxl.columns) > 0: soxl = soxl.iloc[:, 0]

        df = pd.DataFrame({'QQQ': qqq, 'SOXL': soxl})
        df = df.ffill().bfill().dropna()
        df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        st.error(f"데이터 로딩 중 문제 발생: {e}")
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
    return weekly_mode.resample('D').ffill(), rsi_series

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
                df = pd.read_csv(StringIO(csv_string))
                return df
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
    if sim_df.empty: return None, None

    sim_df['Prev_Price'] = sim_df['Price'].shift(1)
    sim_df = sim_df.dropna()

    real_cash = init_cap
    cum_profit = 0.0
    cum_loss = 0.0
    slots = []
    journal = []
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
                
                remaining_shares_val = sum(slots[k]['shares'] * price for k in range(len(slots)) if k != i)
                equity_at_sell = real_cash + rev + remaining_shares_val
                
                journal_entry = {
                    "날짜": date.date(),
                    "총자산": equity_at_sell,
                    "수익금": prof,
                    "수익률": (prof / (equity_at_sell - prof)) * 100 if (equity_at_sell - prof) > 0 else 0
                }
                journal.append(journal_entry)
                
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
                        '매수일': date.date(),
                        '모드': mode,
                        '매수가': price,
                        '수량': int(shares),
                        '목표가': tg,
                        '손절기한': cd.date(),
                        'buy_price': price, 'shares': int(shares), 'days': 0, 'birth_mode': mode
                    })
    
    final_holdings = []
    for s in slots:
        final_holdings.append({
            "매수일": s['매수일'], "모드": s['모드'], "매수가": s['매수가'], 
            "수량": s['수량'], "목표가": s['목표가'], "손절기한": s['손절기한']
        })
    df_holdings = pd.DataFrame(final_holdings)
    df_journal = pd.DataFrame(journal)
    
    return df_holdings, df_journal

def run_backtest_fixed(df, start_date, end_date, init_cap):
    mode_daily, _ = calc_mode_series(df['QQQ'])
    sim_df = pd.concat([df['SOXL'], mode_daily], axis=1).dropna()
    sim_df.columns = ['Price', 'Mode']
    mask = (sim_df.index >= pd.to_datetime(start_date)) & (sim_df.index <= pd.to_datetime(end_date))
    sim_df = sim_df[mask]
    if sim_df.empty: return None, None, None
    sim_df['Prev_Price'] = sim_df['Price'].shift(1)
    sim_df = sim_df.dropna()
    real_cash = init_cap
    cum_profit = 0.0
    cum_loss = 0.0
    slots = []
    equity_curve = []
    cycle_days = 0
    gross_profit = 0.0
    gross_loss = 0.0
    
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
                real_cash += rev
                if prof > 0: 
                    cum_profit += prof
                    gross_profit += prof
                else: 
                    cum_loss += abs(prof)
                    gross_loss += abs(prof)
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
                    slots.append({'buy_price': price, 'shares': shares, 'days': 0, 'birth_mode': mode})
        equity_curve.append({'Date': date, 'Equity': real_cash + sum(s['shares']*price for s in slots)})
    
    res_df = pd.DataFrame(equity_curve).set_index('Date')
    
    metrics = {
        'profit_factor': gross_profit / gross_loss if gross_loss > 0 else 99.9,
    }
    
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

    df_yearly = pd.DataFrame(yearly_stats).set_index("연도")
    return res_df, metrics, df_yearly

# ---------------------------------------------------------
# 3. 메인 UI
# ---------------------------------------------------------
def main():
    st.title("💎 동파법 마스터 v4.4 (Bug Fix)")
    
    tab_trade, tab_backtest, tab_logic = st.tabs(["💎 실전 트레이딩", "🧪 백테스트", "📚 전략 로직"])

    df = get_data_final()
    if df is None:
        st.error("📉 주식 데이터 로드 실패. 잠시 후 다시 시도해주세요.")
        return
    
    mode_s, rsi_s = calc_mode_series(df['QQQ'])
    curr_mode = mode_s.iloc[-1]
    curr_rsi = rsi_s.iloc[-1]
    soxl_price = df['SOXL'].iloc[-1]
    prev_close = df['SOXL'].iloc[-2]

    # [수정된 부분] 데이터 로드 시 컬럼명 호환성 체크
    if 'holdings' not in st.session_state:
        st.session_state['holdings'] = load_csv(HOLDINGS_FILE, ["매수일", "모드", "매수가", "수량", "목표가", "손절기한"])
    
    if 'journal' not in st.session_state:
        # 일단 파일 로드
        loaded_j = load_csv(JOURNAL_FILE, ["날짜", "총자산", "수익금", "수익률"])
        # 호환성 패치: '원금' 컬럼이 있으면 '총자산'으로 이름 변경
        if '원금' in loaded_j.columns:
            loaded_j.rename(columns={'원금': '총자산'}, inplace=True)
        st.session_state['journal'] = loaded_j
    
    settings = load_settings()

    with tab_trade:
        with st.sidebar:
            st.header("🤖 자동 동기화 설정")
            default_date = datetime.strptime(settings.get("start_date", "2025-01-01"), "%Y-%m-%d").date()
            default_cap = float(settings.get("init_cap", 100000.0))
            
            auto_start_date = st.date_input("전략 시작일", value=default_date)
            auto_init_cap = st.number_input("시작 원금 ($)", value=default_cap, step=100.0)
            
            if st.button("🔄 전략대로 자동 동기화 (Sync)", type="primary"):
                new_settings = {"start_date": auto_start_date.strftime("%Y-%m-%d"), "init_cap": auto_init_cap}
                save_settings(new_settings)
                
                with st.spinner("과거 데이터 시뮬레이션 및 클라우드 동기화 중..."):
                    h_new, j_new = auto_sync_engine(df, auto_start_date, auto_init_cap)
                    
                    if h_new is not None:
                        save_csv(h_new, HOLDINGS_FILE)
                        save_csv(j_new, JOURNAL_FILE)
                        st.session_state['holdings'] = h_new
                        st.session_state['journal'] = j_new
                        st.success("완료! 데이터가 저장되었습니다.")
                        st.rerun()
                    else: st.error("동기화 실패")
            
            st.markdown("---")
            if st.button("🗑️ 데이터 초기화"):
                empty_df = pd.DataFrame(columns=["매수일", "모드", "매수가", "수량", "목표가", "손절기한"])
                empty_j = pd.DataFrame(columns=["날짜", "총자산", "수익금", "수익률"])
                save_csv(empty_df, HOLDINGS_FILE)
                save_csv(empty_j, JOURNAL_FILE)
                st.session_state['holdings'] = empty_df
                st.session_state['journal'] = empty_j
                st.rerun()

            today = datetime.now().date()
            cycle = ((today - auto_start_date).days % RESET_CYCLE) + 1
            st.info(f"🔄 사이클: **{cycle}일차** / 10일")

        r = PARAMS[curr_mode]
        slot_sz = auto_init_cap / MAX_SLOTS
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("시장 모드", f"{r['desc']}", f"RSI {curr_rsi:.1f}", delta_color="inverse")
        c2.metric("SOXL 현재가", f"${soxl_price:.2f}", f"{((soxl_price-prev_close)/prev_close)*100:.2f}%")
        c3.metric("1슬롯 할당금", f"${slot_sz:,.0f}")
        c4.metric("매매 사이클", f"{cycle}일차")
        st.markdown("---")

        # 1. 통합 주문표
        st.subheader("⚖️ 오늘의 통합 주문표")
        df_h = st.session_state['holdings']
        
        if soxl_price > 0:
            b_lim = prev_close * (1 + r['buy']/100)
            b_qty = int(slot_sz / soxl_price)
        else: b_lim, b_qty = 0, 0
        
        moc_sell = 0
        loc_list = []
        if not df_h.empty:
            df_h['손절기한'] = pd.to_datetime(df_h['손절기한']).dt.date
            for idx, row in df_h.iterrows():
                if row['손절기한'] <= today: moc_sell += row['수량']
                else: loc_list.append(f"티어{idx+1} ({row['수량']}주 @ ${row['목표가']:.1f})")

        oc1, oc2 = st.columns(2)
        oc1.info(f"**🛒 매수 (LOC):** **{b_qty} 주** (@ ${b_lim:.2f} 이하)")
        if moc_sell > 0: oc2.error(f"**🚨 매도 (MOC):** **{moc_sell} 주** (기한 만료)")
        else: oc2.write("**✅ MOC 매도 없음**")
        if loc_list:
            with st.expander(f"🔵 익절 대기 ({len(loc_list)}건)"):
                for l in loc_list: st.write(f"- {l}")
        if moc_sell > 0: st.warning(f"**🧮 퉁치기:** 순매수 **{b_qty - moc_sell} 주**")

        st.markdown("---")

        # 2. 티어 현황
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

            st.caption("👇 GitHub 데이터")
            edited_h = st.data_editor(
                df_h, num_rows="dynamic", use_container_width=True, key="h_edit",
                column_config={"수익률": st.column_config.TextColumn("수익률", disabled=True), "매수가": st.column_config.NumberColumn(format="$%.2f"), "목표가": st.column_config.NumberColumn(format="$%.1f"), "상태": st.column_config.TextColumn(disabled=True)}
            )
            
            total_qty = edited_h['수량'].sum()
            total_invested = (edited_h['매수가'] * edited_h['수량']).sum()
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
            
            if st.button("💾 티어 수정 저장 (GitHub)"):
                save_cols = ["매수일", "모드", "매수가", "수량", "목표가", "손절기한"]
                save_csv(edited_h[save_cols], HOLDINGS_FILE)
                st.session_state['holdings'] = edited_h[save_cols]
                st.success("저장되었습니다!")
                st.rerun()
        else: st.info("보유 티어 없음")
        
        st.markdown("---")
        
        # 3. 매매일지
        st.subheader("📝 매매 수익 기록장 (Cloud 저장)")
        df_j = st.session_state['journal']
        init_prin = auto_init_cap
        
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
            
            with st.expander("📂 상세 수익 기록표 보기/접기 (편집 가능)", expanded=False):
                st.caption("👇 GitHub 기록 (최신순 / 스크롤 가능)")
                df_display = df_j.sort_values(by="날짜", ascending=False).reset_index(drop=True)
                
                edited_j = st.data_editor(
                    df_display, num_rows="dynamic", use_container_width=True, height=400, key="j_editor",
                    column_config={
                        "수익금": st.column_config.NumberColumn(format="$%.2f"),
                        "수익률": st.column_config.NumberColumn(label="수익률(%)", format="%.2f %%"),
                        "총자산": st.column_config.NumberColumn(label="당시 총자산($)", format="$%.0f"),
                    }
                )
                if st.button("💾 일지 수정 저장 (GitHub)"):
                    if not edited_j.empty:
                        save_csv(edited_j, JOURNAL_FILE)
                        st.session_state['journal'] = edited_j
                        st.success("저장되었습니다!")
                        st.rerun()
            
            st.markdown("### 📈 내 자산 성장 그래프 (Equity Curve)")
            df_chart = df_j.sort_values(by="날짜", ascending=True).copy()
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df_chart['날짜'], df_chart['총자산'], color='#4CAF50', linewidth=2, marker='o', markersize=3)
            ax.fill_between(df_chart['날짜'], df_chart['총자산'], init_prin, where=(df_chart['총자산'] >= init_prin), color='#4CAF50', alpha=0.1)
            ax.fill_between(df_chart['날짜'], df_chart['총자산'], init_prin, where=(df_chart['총자산'] < init_prin), color='red', alpha=0.1)
            ax.axhline(y=init_prin, color='gray', linestyle='--', alpha=0.5, label='원금')
            ax.set_title("Total Equity Growth", fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
            st.pyplot(fig)
        else:
            st.info("실현된 수익 없음.")

        with st.expander("✍️ 수동 기록 추가"):
            with st.form("journal_manual"):
                jc1, jc2, jc3 = st.columns(3)
                j_d = jc1.date_input("정산일", value=today)
                j_p = jc2.number_input("당시 총자산($)", value=float(auto_init_cap))
                j_r = jc3.number_input("손익($)")
                if st.form_submit_button("추가"):
                    nj = {"날짜": j_d, "총자산": j_p, "수익금": j_r, "수익률": (j_r/(j_p-j_r))*100 if (j_p-j_r)>0 else 0}
                    df_j = pd.concat([df_j, pd.DataFrame([nj])], ignore_index=True)
                    save_csv(df_j, JOURNAL_FILE)
                    st.session_state['journal'] = df_j
                    st.rerun()

    with tab_backtest:
        st.header("🧪 백테스트 성과분석")
        bt_init_cap = st.number_input("백테스트 초기 자본 ($)", value=10000.0, step=1000.0)
        bc1, bc2 = st.columns(2)
        start_d = bc1.date_input("검증 시작일", value=datetime(2010, 1, 1), min_value=datetime(2000, 1, 1))
        end_d = bc2.date_input("검증 종료일", value=today, min_value=datetime(2000, 1, 1))
        
        if st.button("🚀 분석 실행"):
            with st.spinner("분석 중..."):
                res, metrics, df_yearly = run_backtest_fixed(df, start_d, end_d, bt_init_cap)
                if res is not None:
                    final = res['Equity'].iloc[-1]
                    ret = (final/bt_init_cap) - 1
                    days = (res.index[-1] - res.index[0]).days
                    cagr = (1+ret)**(365/days) - 1 if days > 0 else 0
                    res['Peak'] = res['Equity'].cummax()
                    res['Drawdown'] = (res['Equity'] - res['Peak']) / res['Peak']
                    mdd = res['Drawdown'].min()
                    calmar = cagr / abs(mdd) if mdd != 0 else 0
                    
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("최종 수익금", f"${final:,.0f}", f"{ret*100:,.1f}%")
                    m2.metric("CAGR", f"{cagr*100:.2f}%")
                    m3.metric("MDD", f"{mdd*100:.2f}%", delta_color="inverse")
                    m4.metric("Calmar", f"{calmar:.2f}")
                    m5.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
                    
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
                else: st.error("데이터 부족")

    with tab_logic:
        st.header("📚 동파법(Dongpa) 전략 매뉴얼 (상세)")
        st.markdown("""
        ### 1. 전략 개요
        * **핵심:** "시장의 계절(Mode)을 먼저 파악하고, 그에 맞는 옷(Rule)을 입는다."
        * **대상:** SOXL (3배 레버리지) / **지표:** QQQ (나스닥100)
        
        ### 2. 시장 모드 판단
        매주 금요일 종가 기준으로 **QQQ 주봉 RSI(14)** 분석
        
        | 모드 | 조건 (Condition) |
        | :--- | :--- |
        | **🛡️ Safe** | `RSI > 65` & `하락` / `40 < RSI < 50` & `하락` / `50선 하향 돌파` |
        | **⚔️ Offense** | `RSI < 35` & `상승` / `50 < RSI < 60` & `상승` / `50선 상향 돌파` |
        
        ### 3. 실전 매매 규칙
        **Sticky Rule:** 매수 당시의 모드를 매도 시까지 유지
        
        | 구분 | 🛡️ 방어 (Safe) | ⚔️ 공세 (Offense) |
        | :--- | :--- | :--- |
        | **매수 타점** | -3.0% 이하 | -5.0% 이하 |
        | **익절 목표** | +0.5% | +3.0% |
        | **손절 기한** | 35일 | 7일 |
        """)

if __name__ == "__main__":
    main()
