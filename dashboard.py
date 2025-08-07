import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import os
import json
import base64
from dateutil import parser
from datetime import datetime

# Gemini API integration
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

# ========================
# Page Configuration
# ========================
st.set_page_config(
    page_title="EV Market Intelligence Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================
# CSS Styling
# ========================
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 1.5rem 0;
    background: linear-gradient(90deg, #3b5998 0%, #4a6fa5 100%);
    color: white;
    font-weight: bold;
    font-size: 1.8rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}
.sub-header {
    font-size: 0.9rem;
    color: #dce3eb;
    margin-top: 0.3rem;
}
.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 0.5rem 0;
}
.filter-section {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ========================
# Main Header
# ========================
st.markdown("""
<div class="main-header">
    🚗 EV Market Intelligence Dashboard
    <div class="sub-header">SKO 전기차 시장 분석 대시보드</div>
</div>
""", unsafe_allow_html=True)

# ========================
# API Key Setup (Sidebar)
# ========================
st.sidebar.header("🔑 API Key 설정")
gemini_api_key = st.sidebar.text_input(
    "Google Gemini API Key", 
    value="", 
    type="password",
    help="Gemini AI 분석을 위한 API 키를 입력하세요"
)
serp_api_key = st.sidebar.text_input(
    "SerpAPI Key", 
    value="", 
    type="password",
    help="뉴스 검색을 위한 SerpAPI 키를 입력하세요"
)

# Initialize Gemini client if API key is provided
gemini_client = None
if gemini_api_key and GEMINI_AVAILABLE:
    try:
        genai.configure(api_key=gemini_api_key)
        gemini_client = genai.GenerativeModel("gemini-1.5-flash")
        st.sidebar.success("✅ Gemini API 연결 성공")
    except Exception as e:
        st.sidebar.error(f"❌ Gemini API 연결 실패: {str(e)}")
elif gemini_api_key and not GEMINI_AVAILABLE:
    st.sidebar.error("❌ Gemini 라이브러리가 설치되지 않았습니다")

# ========================
# Data Loading Functions
# ========================
@st.cache_data
def load_default_data():
    df = None
    file_name = "AI Camp 교육 과제용 DB.csv"
    if os.path.exists(file_name):
        for enc in ["utf-8-sig", "utf-8", "cp949", "latin1"]:
            try:
                df = pd.read_csv(file_name, encoding=enc)
                st.sidebar.success(f"✅ 기본 CSV 로드 성공: {file_name} ({enc})")
                break
            except (UnicodeDecodeError, FileNotFoundError):
                continue
    else:
        st.sidebar.warning(f"⚠️ {file_name} 파일을 찾을 수 없습니다. 파일을 업로드하거나, 같은 폴더에 넣어주세요.")
    return df if df is not None else pd.DataFrame()

def load_uploaded_data():
    uploaded_file = st.sidebar.file_uploader(
        "📂 CSV 데이터 업로드 (관리자 모드)", 
        type=['csv']
    )
    if uploaded_file:
        for enc in ["utf-8-sig", "utf-8", "cp949", "latin1"]:
            try:
                return pd.read_csv(uploaded_file, encoding=enc)
            except:
                continue
        st.sidebar.error("❌ 업로드 실패")
    return None

uploaded_df = load_uploaded_data()
df = uploaded_df if uploaded_df is not None else load_default_data()
if df.empty:
    st.error("⚠️ 데이터가 로드되지 않았습니다.")
    st.stop()

# ========================
# Monthly Columns + Data Cleaning
# ========================
def get_monthly_columns(df):
    cols = [c for c in df.columns if "(EV)" in c and "Q" not in c and "Y" not in c]
    def parse_date(c):
        try:
            date_str = c.replace("(EV)\n","" ).replace("(EV)", "").strip()
            return pd.to_datetime(date_str, format="%b-%y", errors="coerce")
        except:
            return pd.NaT
    return sorted([c for c in cols if parse_date(c) is not pd.NaT], key=parse_date)

monthly_cols = get_monthly_columns(df)
df_months = [pd.to_datetime(c.replace("(EV)", "").replace("\n", "").strip(), format="%b-%y") for c in monthly_cols]
month_map = dict(zip(df_months, monthly_cols))

def clean_numeric_data(df, monthly_cols):
    df_clean = df.copy()
    for col in monthly_cols:
        if col in df_clean.columns:
            df_clean[col] = (
                df_clean[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("-", "0", regex=False)
                .replace("", "0")
            )
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(0)
            df_clean[col] = df_clean[col].clip(lower=0)
    return df_clean

df = clean_numeric_data(df, monthly_cols)

# 기준월: Jun-25 고정 (EVdashboard 방식)
df_months = [pd.to_datetime(c.replace("(EV)", "").replace("\n", "").strip(), format="%b-%y") for c in monthly_cols]
month_map = dict(zip(df_months, monthly_cols))
fixed_month = pd.to_datetime("Jun-25", format="%b-%y")
latest_month_dt = fixed_month if fixed_month in df_months else max(df_months)
sorted_months = sorted(df_months)
latest_idx = sorted_months.index(latest_month_dt)
start_idx = max(0, latest_idx - 12)
recent_13_months = [month_map[m] for m in sorted_months[start_idx:latest_idx + 1]]
latest_month = month_map[latest_month_dt]
# ========================
# Sidebar: Base Month Selection
# ========================
st.sidebar.header("📅 기준월 설정")

# Create list of available months
available_months = []
for col in monthly_cols:
    try:
        date_str = col.replace("(EV)", "").replace("\n", "").strip()
        month_dt = pd.to_datetime(date_str, format="%b-%y")
        available_months.append((date_str, month_dt))
    except:
        continue

available_months = sorted(available_months, key=lambda x: x[1])
month_options = [month[0] for month in available_months]

# 기준월 선택 (Jun-25 기본값)
default_month = "Jun-25"
if default_month not in month_options and month_options:
    default_month = month_options[-1]

selected_base_month = st.sidebar.selectbox(
    "기준월 선택",
    options=month_options,
    index=month_options.index(default_month) if default_month in month_options else 0,
    help="분석의 기준이 되는 월을 선택하세요"
)

def get_m_history_data(monthly_cols, base_month_str, num_months=12):
    df_months = [pd.to_datetime(c.replace("(EV)", "").replace("\n", "").strip(), format="%b-%y") for c in monthly_cols]
    month_map = dict(zip(df_months, monthly_cols))

    fixed_month = pd.to_datetime(base_month_str, format="%b-%y")
    latest_month_dt = fixed_month if fixed_month in df_months else max(df_months)

    sorted_months = sorted(df_months)
    latest_idx = sorted_months.index(latest_month_dt)
    start_idx = max(0, latest_idx - num_months)

    recent_months = [month_map[m] for m in sorted_months[start_idx:latest_idx + 1]]
    latest_month = month_map[latest_month_dt]

    return recent_months, latest_month, latest_month_dt

# 선택된 기준월로 M-13 데이터 재계산
recent_13_months, latest_month, latest_month_dt = get_m_history_data(monthly_cols, selected_base_month, num_months=12)

st.sidebar.info(f"선택된 기준월: **{selected_base_month}**")
st.sidebar.info(f"분석 기간: **{len(recent_13_months)}개월**")

# ========================
# Session State Management
# ========================
if "search_performed" not in st.session_state:
    st.session_state.search_performed = False

if not st.session_state.search_performed:
    # ========================
    # Initial Search Screen
    # ========================
    st.header("🔍 검색 조건 설정")

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            selected_oems = st.multiselect(
                "🏢 OEM (자동차 제조사)",
                ["전체"] + sorted(df["AutoGroup"].dropna().unique()),
                default=["전체"],
                key="oems_select",
                help="분석할 자동차 제조사를 선택하세요"
            )

            selected_regions = st.multiselect(
                "🌍 주요 시장",
                ["전체"] + sorted(df["Region"].dropna().unique()),
                default=["전체"],
                key="regions_select",
                help="분석할 지역을 선택하세요"
            )

        with col2:
            selected_models = st.multiselect(
                "🚗 차종",
                ["전체"] + sorted(df["Model"].dropna().unique()),
                default=["전체"],
                key="models_select",
                help="분석할 차량 모델을 선택하세요"
            )

            selected_countries = st.multiselect(
                "🏳️ 국가",
                ["전체"] + sorted(df["Country"].dropna().unique()),
                default=["전체"],
                key="countries_select",
                help="분석할 국가를 선택하세요"
            )

    selected_xev_types = st.multiselect(
        "🔋 xEV Type (전기차 유형)",
        ["전체"] + sorted(df["Type_2"].dropna().unique()),
        default=["전체"],
        key="xev_select",
        help="분석할 전기차 유형을 선택하세요 (BEV, PHEV, FHEV, MHEV)"
    )

    st.markdown("---")

    if st.button("🔍 검색하기", type="primary", key="search_button"):
        # Store search conditions in session state
        st.session_state.update({
            "selected_oems": selected_oems,
            "selected_regions": selected_regions,
            "selected_models": selected_models,
            "selected_countries": selected_countries,
            "selected_xev_types": selected_xev_types,
            "selected_base_month": selected_base_month,
            "search_performed": True
        })
        st.rerun()

else:
    # ========================
    # Search Results Screen
    # ========================

    # Display current search conditions in sidebar
    st.sidebar.header("🔍 현재 검색 조건")
    st.sidebar.write(f"**기준월**: {st.session_state.get('selected_base_month', selected_base_month)}")
    st.sidebar.write(f"**OEM**: {', '.join(st.session_state['selected_oems'])}")
    st.sidebar.write(f"**지역**: {', '.join(st.session_state['selected_regions'])}")
    st.sidebar.write(f"**모델**: {', '.join(st.session_state['selected_models'])}")
    st.sidebar.write(f"**국가**: {', '.join(st.session_state['selected_countries'])}")
    st.sidebar.write(f"**xEV타입**: {', '.join(st.session_state['selected_xev_types'])}")

    if st.sidebar.button("🔄 새로운 검색", key="reset_button"):
        st.session_state.search_performed = False
        st.rerun()

    # Apply filters to dataframe
    filtered_df = df.copy()

    if "전체" not in st.session_state["selected_oems"]:
        filtered_df = filtered_df[filtered_df["AutoGroup"].isin(st.session_state["selected_oems"])]

    if "전체" not in st.session_state["selected_regions"]:
        filtered_df = filtered_df[filtered_df["Region"].isin(st.session_state["selected_regions"])]

    if "전체" not in st.session_state["selected_models"]:
        filtered_df = filtered_df[filtered_df["Model"].isin(st.session_state["selected_models"])]

    if "전체" not in st.session_state["selected_countries"]:
        filtered_df = filtered_df[filtered_df["Country"].isin(st.session_state["selected_countries"])]

    if "전체" not in st.session_state["selected_xev_types"]:
        filtered_df = filtered_df[filtered_df["Type_2"].isin(st.session_state["selected_xev_types"])]

    # Generate keywords for news search
    selected_keywords = []
    for group in [
        st.session_state["selected_oems"],
        st.session_state["selected_regions"],
        st.session_state["selected_models"],
        st.session_state["selected_countries"],
        st.session_state["selected_xev_types"]
    ]:
        for item in group:
            if item != "전체":
                selected_keywords.append(item)

    if not selected_keywords or "전체" in str(selected_keywords):
        selected_keywords = ["EV battery", "electric vehicle", "전기차"]

    # ========================
    # Tab Navigation
    # ========================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 기준월 데이터 분석",
        "📈 M-13 히스토리",
        "🤖 AI Assistant (Gemini)",
        "📰 관련 뉴스 (Google News)"
    ])

    # ========================
    # Tab 1: Base Month Data Analysis
    # ========================
    with tab1:
        st.header(f"📊 기준월 데이터 분석 ({selected_base_month})")

        if latest_month and latest_month in filtered_df.columns:
            # 주요 지표 계산
            total_sales = filtered_df[latest_month].sum()
            total_models = filtered_df.loc[filtered_df[latest_month] > 0, "Model"].nunique()
            total_oems = filtered_df.loc[filtered_df[latest_month] > 0, "AutoGroup"].nunique()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("총 판매량", f"{int(total_sales):,}")
            with col2:
                st.metric("판매 모델 수", f"{total_models:,}")
            with col3:
                st.metric("참여 OEM 수", f"{total_oems:,}")

            # 🔋 xEV Type별 판매량
            st.subheader("🔋 xEV Type별 판매량")
            xev_type_data = filtered_df.groupby("Type_2")[latest_month].sum().reset_index()
            xev_type_data = xev_type_data.sort_values(latest_month, ascending=False)
            fig_xev_type = px.bar(xev_type_data, x="Type_2", y=latest_month, text=latest_month)
            fig_xev_type.update_traces(texttemplate="%{text:,.0f}", hovertemplate="%{x}: %{y:,}")
            st.plotly_chart(fig_xev_type, use_container_width=True)

            # 🏢 OEM Market Share
            st.subheader("🏢 OEM Market Share (Top 10 + Others)")
            oem_data = filtered_df.groupby("AutoGroup")[latest_month].sum().reset_index()
            oem_data = oem_data.sort_values(latest_month, ascending=False)
            top10_oem = oem_data.head(10)
            others_oem_sum = oem_data.iloc[10:][latest_month].sum()
            if others_oem_sum > 0:
                top10_oem = pd.concat([top10_oem, pd.DataFrame({"AutoGroup": ["Others"], latest_month: [others_oem_sum]})])
            fig_oem = px.pie(top10_oem, values=latest_month, names="AutoGroup")
            fig_oem.update_traces(textinfo="percent+label", hovertemplate="%{label}: %{value:,}")
            st.plotly_chart(fig_oem, use_container_width=True)

            # 🌍 지역별 판매량
            st.subheader("🌍 지역별 판매량")
            region_data = filtered_df.groupby("Region")[latest_month].sum().reset_index()
            region_data = region_data.sort_values(latest_month, ascending=False).head(10)
            fig_region = px.bar(region_data, x=latest_month, y="Region", orientation="h")
            fig_region.update_traces(hovertemplate="%{y}: %{x:,}")
            fig_region.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_region, use_container_width=True)

            # 🔋 Battery Supplier 비중
            st.subheader("🔋 Battery Supplier 비중 (Top 10 + Others)")
            supplier_data = filtered_df.groupby("Battery Supplier")[latest_month].sum().reset_index()
            supplier_data = supplier_data.sort_values(latest_month, ascending=False)
            top10_supplier = supplier_data.head(10)
            others_supplier_sum = supplier_data.iloc[10:][latest_month].sum()
            if others_supplier_sum > 0:
                top10_supplier = pd.concat([top10_supplier, pd.DataFrame({"Battery Supplier": ["Others"], latest_month: [others_supplier_sum]})])
            fig_supplier = px.pie(top10_supplier, values=latest_month, names="Battery Supplier")
            fig_supplier.update_traces(textinfo="percent+label", hovertemplate="%{label}: %{value:,}")
            st.plotly_chart(fig_supplier, use_container_width=True)

            # 🚘 Top Selling Model
            st.subheader("🚘 Top Selling Model (Top 10)")
            model_data = filtered_df.groupby("Model")[latest_month].sum().reset_index()
            model_data = model_data.sort_values(latest_month, ascending=False).head(10)
            fig_model = px.bar(model_data, x=latest_month, y="Model", orientation="h")
            fig_model.update_traces(hovertemplate="%{y}: %{x:,}")
            fig_model.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_model, use_container_width=True)
        else:
            st.warning("선택된 기준월에 데이터가 없습니다.")
    # ========================
    # Tab 2: M-13 History
    # ========================
    with tab2:
        st.header("📈 M-13 히스토리 분석")

        def create_month_labels(columns):
            return [col.replace("(EV)", "").replace("\n", "").strip() for col in columns]

        month_labels = create_month_labels(recent_13_months)

        # 🚗 2-1 xEV 판매량 추이 (누적막대)
        st.subheader("🚗 xEV 판매량 추이 (Top 10 + Others)")
        xev_type_hist_full = filtered_df.groupby("Type_2")[recent_13_months].sum()
        totals_by_type = xev_type_hist_full.sum(axis=1)
        top10_types = totals_by_type.nlargest(10).index
        xev_type_hist = xev_type_hist_full.loc[top10_types]
        others_sum = xev_type_hist_full.loc[~xev_type_hist_full.index.isin(top10_types)].sum()
        if others_sum.sum() > 0:
            xev_type_hist.loc["Others"] = others_sum
        xev_type_hist = xev_type_hist.T
        xev_type_hist.index = month_labels
        fig_xev_hist = px.bar(xev_type_hist, x=xev_type_hist.index, y=xev_type_hist.columns, barmode="stack")
        fig_xev_hist.update_traces(hovertemplate="%{y:,}")
        st.plotly_chart(fig_xev_hist, use_container_width=True)

        # 🏢 2-2 OEM별 판매량 추이 (꺾은선)
        st.subheader("🏢 OEM별 판매량 추이 (Top 10 + Others)")
        oem_hist_full = filtered_df.groupby("AutoGroup")[recent_13_months].sum()
        totals_by_oem = oem_hist_full.sum(axis=1)
        top10_oems = totals_by_oem.nlargest(10).index
        oem_hist = oem_hist_full.loc[top10_oems]
        others_sum = oem_hist_full.loc[~oem_hist_full.index.isin(top10_oems)].sum()
        if others_sum.sum() > 0:
            oem_hist.loc["Others"] = others_sum
        oem_hist = oem_hist.T
        oem_hist.index = month_labels
        fig_oem_hist = px.line(oem_hist, x=oem_hist.index, y=oem_hist.columns, markers=True)
        fig_oem_hist.update_traces(hovertemplate="%{y:,}")
        st.plotly_chart(fig_oem_hist, use_container_width=True)

        # 🌍 2-3 지역별 판매량 추이 (꺾은선)
        st.subheader("🌍 지역별 판매량 추이 (Top 10 + Others)")
        region_hist_full = filtered_df.groupby("Region")[recent_13_months].sum()
        totals_by_region = region_hist_full.sum(axis=1)
        top10_regions = totals_by_region.nlargest(10).index
        region_hist = region_hist_full.loc[top10_regions]
        others_sum = region_hist_full.loc[~region_hist_full.index.isin(top10_regions)].sum()
        if others_sum.sum() > 0:
            region_hist.loc["Others"] = others_sum
        region_hist = region_hist.T
        region_hist.index = month_labels
        fig_region_hist = px.line(region_hist, x=region_hist.index, y=region_hist.columns, markers=True)
        fig_region_hist.update_traces(hovertemplate="%{y:,}")
        st.plotly_chart(fig_region_hist, use_container_width=True)

        # 🔋 2-4 Battery Supplier 판매량 추이 (꺾은선)
        st.subheader("🔋 Battery Supplier 판매량 추이 (Top 10)")
        supplier_hist_full = filtered_df.groupby("Battery Supplier")[recent_13_months].sum()
        totals_by_supplier = supplier_hist_full.sum(axis=1)
        top10_suppliers = totals_by_supplier.nlargest(10).index
        supplier_hist = supplier_hist_full.loc[top10_suppliers]
        supplier_hist = supplier_hist.T
        supplier_hist.index = month_labels
        fig_supplier_hist = px.line(supplier_hist, x=supplier_hist.index, y=supplier_hist.columns, markers=True)
        fig_supplier_hist.update_traces(hovertemplate="%{y:,}")
        st.plotly_chart(fig_supplier_hist, use_container_width=True)

        # 🚘 2-5 Top Selling Model 판매량 추이 (꺾은선)
        st.subheader("🚘 Top Selling Model 판매량 추이 (Top 10)")
        model_hist_full = filtered_df.groupby("Model")[recent_13_months].sum()
        totals_by_model = model_hist_full.sum(axis=1)
        top10_models = totals_by_model.nlargest(10).index
        model_hist = model_hist_full.loc[top10_models]
        model_hist = model_hist.T
        model_hist.index = month_labels
        fig_model_hist = px.line(model_hist, x=model_hist.index, y=model_hist.columns, markers=True)
        fig_model_hist.update_traces(hovertemplate="%{y:,}")
        st.plotly_chart(fig_model_hist, use_container_width=True)
    # ========================
    # Tab 3: Gemini 분석
    # ========================
    with tab3:
        st.header("🤖 AI Assistant (Gemini)")

        if gemini_client and monthly_cols:
            st.info(f"기준월 **{selected_base_month}** 기준 최근 6개월 및 전년 동월 데이터를 Gemini AI가 분석합니다.")

            # Gemini 분석을 위한 데이터 범위 재설정 (최근 6개월 + 전년 동월)
            base_month_dt = pd.to_datetime(selected_base_month, format="%b-%y")

            # 최근 6개월
            end_idx = sorted_months.index(base_month_dt)
            start_idx_6m = max(0, end_idx - 5)
            recent_6_months_dt = sorted_months[start_idx_6m:end_idx + 1]

            # 전년 동월
            last_year_dt = base_month_dt - pd.DateOffset(years=1)
            last_year_col = month_map.get(last_year_dt)

            analysis_columns = []
            analysis_columns.extend([month_map[m] for m in recent_6_months_dt])
            if last_year_col and last_year_col not in analysis_columns:
                analysis_columns.append(last_year_col)

            # EVdashboard 방식: M-13 데이터 추출
            analysis_df = filtered_df[['AutoGroup', 'Model', 'Battery Supplier', 'Type_2'] + analysis_columns]
            analysis_df = analysis_df.copy()

            # 수치 데이터 정합화 (NaN → 0, 음수 제거)
            for col in analysis_columns:
                analysis_df[col] = pd.to_numeric(analysis_df[col], errors='coerce').fillna(0)
                analysis_df[col] = analysis_df[col].clip(lower=0)

            if not analysis_df.empty and len(analysis_columns) > 1:
                st.subheader("📋 분석 데이터 미리보기")
                st.write(f"레코드: **{len(analysis_df)}개**, 기간: **{len(analysis_columns)}개월**")
                st.write("분석 기간 열:", ', '.join(analysis_columns))
                with st.expander("데이터 보기", expanded=False):
                    st.dataframe(analysis_df.head(10))

                if st.button("🔍 Gemini AI 분석 실행", type="primary"):
                    try:
                        with st.spinner("Gemini AI 분석 중..."):
                            csv_data = analysis_df.to_csv(index=False)

                            gemini_prompt = f"""
    당신은 EV(전기차)/배터리 부문 10년차 Market Intelligence 전문가입니다.
    다음 CSV 데이터는 기준월 {selected_base_month} 포함 최근 6개월 및 전년 동월 판매 데이터입니다.

    CSV 데이터:
    {csv_data}

    다음 내용을 분석하세요:
    1. 시장 트렌드 분석 (YoY, MoM)
    2. OEM별 점유율 및 경쟁 구도
    3. xEV 타입별 성장세와 배터리 공급 현황
    4. 지역별 시장 특성과 성장 요인
    5. 전략적 시사점 및 리스크
    """
                            response = gemini_client.generate_content(gemini_prompt)

                            if response and response.text:
                                st.subheader("🤖 Gemini AI 분석 결과")
                                st.markdown(response.text)
                                st.download_button(
                                    label="📥 분석 결과 다운로드",
                                    data=response.text,
                                    file_name=f"gemini_analysis_{selected_base_month}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                                    mime="text/plain"
                                )
                            else:
                                st.error("Gemini AI 응답이 없습니다.")
                    except Exception as e:
                        st.error(f"Gemini 분석 오류: {str(e)}")
            else:
                st.warning("분석할 데이터가 충분하지 않습니다. 검색 조건을 다시 설정해주세요.")
        else:
            st.warning("⚠️ Gemini API가 연결되지 않았습니다. 사이드바에서 API 키를 입력하세요.")

    # ========================
    # Tab 4: News Search (제목만 최신순 출력)
    # ========================
    with tab4:
        st.header("📰 관련 뉴스 (Google News)")

        def fetch_google_news(query, api_key):
            """SerpAPI를 사용한 Google News 검색"""
            if not api_key:
                return []

            try:
                url = "https://serpapi.com/search.json"
                params = {
                    "engine": "google_news",
                    "q": query,
                    "api_key": api_key,
                    "gl": "us",
                    "hl": "en",
                    "num": 10
                }

                response = requests.get(url, params=params, timeout=10)

                if response.status_code == 200:
                    results = response.json().get("news_results", [])
                    # 날짜 최신순 정렬
                    try:
                        return sorted(
                            results,
                            key=lambda x: parser.parse(x.get("date", "1970-01-01")),
                            reverse=True
                        )[:10]
                    except:
                        return results[:10]
                else:
                    st.error(f"뉴스 검색 API 오류: {response.status_code}")
                    return []

            except requests.exceptions.RequestException as e:
                st.error(f"뉴스 검색 중 네트워크 오류: {str(e)}")
                return []
            except Exception as e:
                st.error(f"뉴스 검색 중 오류: {str(e)}")
                return []

        if serp_api_key:
            st.info("선택된 검색 조건과 관련된 최신 뉴스를 검색합니다.")

            # 선택된 키워드별 뉴스 표시 (제목만 출력)
            for keyword in selected_keywords[:5]:  # 처음 5개 키워드로 제한
                with st.expander(f"🔍 '{keyword}' 관련 뉴스", expanded=False):
                    news_results = fetch_google_news(keyword, serp_api_key)

                    if news_results:
                        for i, news in enumerate(news_results, 1):
                            title = news.get("title", "제목 없음")
                            link = news.get("link", "#")
                            st.markdown(f"**{i}. [{title}]({link})**")
                    else:
                        st.info(f"'{keyword}' 관련 뉴스를 찾을 수 없습니다.")

        else:
            st.warning("⚠️ SerpAPI 키가 설정되지 않았습니다.")
            st.info("뉴스 검색 기능을 사용하려면 사이드바에서 SerpAPI 키를 입력해주세요.")

            with st.expander("SerpAPI 키 설정 방법", expanded=False):
                st.markdown("""
                1. https://serpapi.com/ 방문
                2. 계정 생성 후 로그인
                3. Dashboard에서 API 키 확인
                4. 생성된 키를 사이드바에 입력

                ⚠️ 참고: SerpAPI는 유료 서비스이며, 제한된 무료 사용량을 제공합니다.
                """)
# ========================
# Footer
# ========================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    🚗 EV Market Intelligence Dashboard | Powered by Streamlit & Gemini AI
</div>
""", unsafe_allow_html=True)
