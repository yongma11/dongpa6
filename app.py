# 기존의 @st.cache_data 부분을 찾아서 이 내용으로 통째로 바꾸세요.

@st.cache_data(ttl=3600)
def get_data_final(period='max'):
    try:
        # 1. 2010년부터 데이터 시도
        start_date = '2010-01-01'
        
        # 2. 티커별로 따로 받아서 합치기 (이 방식이 에러가 훨씬 적습니다)
        qqq = yf.download("QQQ", start=start_date, progress=False, auto_adjust=False)
        soxl = yf.download("SOXL", start=start_date, progress=False, auto_adjust=False)
        
        # 3. 데이터가 비었는지 확인
        if qqq.empty or soxl.empty:
            return None

        # 4. 종가(Close)만 추출 (버전 호환성 처리)
        if isinstance(qqq.columns, pd.MultiIndex): qqq = qqq.xs('Close', level=0, axis=1)
        elif 'Close' in qqq.columns: qqq = qqq['Close']
        else: qqq = qqq.iloc[:, 0] # 강제 추출
        
        if isinstance(soxl.columns, pd.MultiIndex): soxl = soxl.xs('Close', level=0, axis=1)
        elif 'Close' in soxl.columns: soxl = soxl['Close']
        else: soxl = soxl.iloc[:, 0] # 강제 추출
        
        # 이름 통일
        if 'QQQ' in qqq.columns: qqq = qqq['QQQ']
        if 'SOXL' in soxl.columns: soxl = soxl['SOXL']

        # 데이터프레임 합치기
        df = pd.DataFrame({'QQQ': qqq, 'SOXL': soxl})
        
        # 5. 결측치 제거 및 시간대 제거
        df = df.ffill().bfill().dropna()
        df.index = df.index.tz_localize(None)
        
        return df

    except Exception as e:
        # 에러가 나면 화면에 작게 이유를 표시해줍니다.
        st.error(f"데이터 처리 중 오류 발생: {e}")
        return None
