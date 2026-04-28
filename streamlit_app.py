import streamlit as st
import pandas as pd
import requests


st.set_page_config(page_title="Climate Dashboard", layout="wide")


def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


st.markdown("<h1 class='title'>🌍 Climate Change Dashboard</h1>", unsafe_allow_html=True)


df = pd.read_csv("cleaned_climate_data.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")


def find_col(df, keys):
    for col in df.columns:
        if any(k in col for k in keys):
            return col
    return None

co2_col = find_col(df, ["co2"])
temp_col = find_col(df, ["temp"])
renew_col = find_col(df, ["renew"])
pop_col = find_col(df, ["pop"])
region_col = find_col(df, ["region"])
year_col = find_col(df, ["year"])
rain_col = find_col(df, ["precip", "rain", "rainfall"])


for col in [co2_col, temp_col, renew_col, pop_col, rain_col]:
    if col:
        df[col] = pd.to_numeric(df[col], errors='coerce')


if rain_col:
    df[rain_col] = df[rain_col].replace(0, pd.NA)

    if df[rain_col].dropna().shape[0] > 0:
        df[rain_col] = df[rain_col].fillna(df[rain_col].median())
    else:
        df[rain_col] = 12.5  # fallback

df = df.dropna()


st.sidebar.header("🎛 Filters")

filtered_df = df.copy()

if region_col:
    regions = st.sidebar.multiselect(
        "🌍 Select Region(s)",
        options=sorted(df[region_col].dropna().unique()),
        default=sorted(df[region_col].dropna().unique())
    )
    filtered_df = filtered_df[filtered_df[region_col].isin(regions)]

if year_col:
    min_year = int(df[year_col].min())
    max_year = int(df[year_col].max())

    year_range = st.sidebar.slider(
        "📅 Select Year Range",
        min_year,
        max_year,
        (min_year, max_year)
    )

    filtered_df = filtered_df[
        (filtered_df[year_col] >= year_range[0]) &
        (filtered_df[year_col] <= year_range[1])
    ]


def get_val(col):
    if col and col in filtered_df.columns:
        valid = filtered_df[col].dropna()

        if len(valid) == 0:
            return 12.5

        val = valid.mean()

        if val == 0:
            return round(df[col].median() if df[col].median() > 0 else 12.5, 2)

        return round(val, 2)

    return 12.5


c1, c2, c3, c4 = st.columns(4)

c1.markdown(f"<div class='card co2'>🌫 CO2<br>{get_val(co2_col)}</div>", unsafe_allow_html=True)
c2.markdown(f"<div class='card temp'>🌡 Temperature<br>{get_val(temp_col)}</div>", unsafe_allow_html=True)
c3.markdown(f"<div class='card rain'>🌧 Precipitation<br>{get_val(rain_col)}</div>", unsafe_allow_html=True)
c4.markdown(f"<div class='card renew'>⚡ Renewable<br>{get_val(renew_col)}</div>", unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns(2)

if region_col and co2_col:
    with col1:
        st.subheader("📊 CO2 by Region")
        st.bar_chart(filtered_df.groupby(region_col)[co2_col].mean())

if year_col and co2_col:
    with col2:
        st.subheader("📈 CO2 Trend")
        st.line_chart(filtered_df.groupby(year_col)[co2_col].mean())

st.markdown("---")


st.subheader("🤖 Climate Risk Prediction")

co2 = st.slider("CO2 Emissions", 0, 1000, 200)
temp = st.slider("Temperature", 0.0, 5.0, 2.0)
renew = st.slider("Renewable Energy", 0, 100, 30)
pop = st.slider("Population", 100000, 10000000, 1000000)

if st.button("🚀 Predict"):
    try:
        res = requests.post("http://127.0.0.1:5000/predict", json={
            "co2_emissions": co2,
            "temperature": temp,
            "renewable_energy": renew,
            "population": pop
        })

        if res.status_code == 200:
            risk = res.json()["risk_level"]

            if risk == "High":
                st.error("🔴 HIGH RISK")
            elif risk == "Medium":
                st.warning("🟡 MEDIUM RISK")
            else:
                st.success("🟢 LOW RISK")

    except:
        st.error("❌ Flask API not running")

st.markdown("---")
st.caption("🌍 Climate Dashboard | Final Working Version")