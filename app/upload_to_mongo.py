import pandas as pd
from pymongo import MongoClient
from datetime import datetime, timezone
from .constants import DB, COL, MONGO_URI

# -----------------------------
# CONFIG
# -----------------------------
CSV_PATH = "./training_data_200k_v4.csv"

# -----------------------------
# HELPERS
# -----------------------------
def normalize_mechanisms(mech_str):
    """
    Returns list of mechanisms.
    Supports:
      - single cause
      - 'cause1; cause2'
    """
    if pd.isna(mech_str) or mech_str.strip() == "":
        return []
    return [m.strip() for m in mech_str.split(";")]

# -----------------------------
# LOAD CSV
# -----------------------------
df = pd.read_csv(CSV_PATH)

client = MongoClient(MONGO_URI)
collection = client[DB][COL]

documents = []
risk_columns = [col for col in df.columns if "risk" in col.lower()]

for _, row in df.iterrows():
    parameter_violations = {}
    for risk_col in risk_columns:
        risk_value = float(row[risk_col])
        parameter_violations["_".join(risk_col.lower().split())] = risk_value

    doc = {
        "board_id": f"PCB{int(row['board_number'])}",
        "batch_id": int(row["batch_id"]),
        "board_number": int(row["board_number"]),

        "parameters": {
            "paste_volume": float(row["Paste volume per aperture"]),
            "stencil_thickness": float(row["Stencil thickness"]),
            "paste_viscosity": float(row["Paste viscosity"]),
            "ambient_rh": float(row["Ambient RH"]),
            "ambient_temperature": float(row["Ambient temperature"])
        },
        "parameter_violations": parameter_violations,
        "labels": {
            "defect": row["Defect"],
            "mechanism_causes": (
                None if pd.isna(row["mech causes"]) or row["mech causes"] == ""
                else row["mech causes"]
            ),
            "solder_printing_mechanism": row['Solder Printing Mechanism'],
            "reflow_mechanism": row['Reflow Mechanism']
        },

        "temporal": {
            "hour_of_day": int(row["hour_of_day"]),
            "stencil_batch": (
                None if pd.isna(row["stencil_batch"]) or row["stencil_batch"] == ""
                else row["stencil_batch"]
            )
        },

        "metadata": {
            "generated_at": datetime.now(timezone.utc),
            "data_source": "synthetic_generator_v1",
            "schema_version": "1.0"
        }
    }

    documents.append(doc)

# -----------------------------
# INSERT
# -----------------------------
if documents:
    collection.insert_many(documents)

print(f"✅ Inserted {len(documents)} board documents into MongoDB")
