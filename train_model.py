import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib


df = pd.read_csv("cleaned_climate_data.csv", low_memory=False)

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")


cols = ['co2_emissions', 'temperature_change', 'renewable_energy', 'population']


for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()


def risk_label(row):
    if row["co2_emissions"] > 700 and row["temperature_change"] > 3:
        return "High"
    elif row["co2_emissions"] > 400:
        return "Medium"
    else:
        return "Low"

df["risk_level"] = df.apply(risk_label, axis=1)

if "High" not in df["risk_level"].values:
    high_samples = df.sample(500, replace=True).copy()
    high_samples["co2_emissions"] = 900
    high_samples["temperature_change"] = 4.5
    high_samples["renewable_energy"] = 10
    high_samples["risk_level"] = "High"
    df = pd.concat([df, high_samples])


X = df[cols]
y = df["risk_level"]

model = RandomForestClassifier(n_estimators=120, random_state=42)
model.fit(X, y)


joblib.dump(model, "model.pkl")

print("✅ Model trained successfully!")