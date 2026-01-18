import pandas as pd
import numpy as np

# Load your training data
df = pd.read_csv('./training_data_200k_v3_param_viols.csv')  # or however you load it

print("="*70)
print("TRAINING DATA DISTRIBUTION ANALYSIS")
print("="*70)

# Ambient RH analysis
print("\n[AMBIENT RH]")
print(f"Total boards: {len(df)}")
print(f"Boards with RH > USL (50): {(df['Ambient RH'] > 50).sum()} ({(df['Ambient RH'] > 50).mean()*100:.2f}%)")
print(f"Boards with RH > 55: {(df['Ambient RH'] > 55).sum()} ({(df['Ambient RH'] > 55).mean()*100:.2f}%)")

print("\nRH risk distribution:")
print(df['ambient_rh_risk'].describe())

print("\nRH risk for violated boards (RH > 50):")
if (df['Ambient RH'] > 50).sum() > 0:
    print(df[df['Ambient RH'] > 50]['ambient_rh_risk'].describe())

# Ambient Temperature analysis
print("\n" + "="*70)
print("[AMBIENT TEMPERATURE]")
print(f"Total boards: {len(df)}")
print(f"Boards with Temp > USL (26): {(df['Ambient temperature'] > 26).sum()} ({(df['Ambient temperature'] > 26).mean()*100:.2f}%)")
print(f"Boards with Temp > 27: {(df['Ambient temperature'] > 27).sum()} ({(df['Ambient temperature'] > 27).mean()*100:.2f}%)")

print("\nTemp risk distribution:")
print(df['ambient_temperature_risk'].describe())

print("\nTemp risk for violated boards (Temp > 26):")
if (df['Ambient temperature'] > 26).sum() > 0:
    print(df[df['Ambient temperature'] > 26]['ambient_temperature_risk'].describe())

# Overall risk comparison
print("\n" + "="*70)
print("[RISK SCORE COMPARISON ACROSS PARAMETERS]")
print("="*70)

risk_cols = ['paste_volume_risk', 'stencil_thickness_risk', 'paste_viscosity_risk',
             'ambient_rh_risk', 'ambient_temperature_risk']

for col in risk_cols:
    high_risk = (df[col] > 0.70).sum()
    print(f"{col:30s}: Mean={df[col].mean():.4f}, High-risk={high_risk:>6,} ({high_risk/len(df)*100:5.2f}%)")