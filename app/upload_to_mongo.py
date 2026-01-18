import pandas as pd
from pymongo import MongoClient
from datetime import datetime, timezone
from collections import Counter, defaultdict

# -----------------------------
# CONFIG
# -----------------------------
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "pcb_manufacturing"
COLLECTION_NAME = "synthetic_boards_v2"
CSV_PATH = "./training_data_200k_v3.csv"

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
collection = client[DB_NAME][COLLECTION_NAME]

documents = []

for _, row in df.iterrows():

    doc = {
        "board_id": f"PCB{int(row['board_number'])}",
        "batch_id": int(row["batch_id"]),
        "board_number": int(row["board_number"]),

        "parameters": {
            "paste_volume_per_aperture": float(row["Paste volume per aperture"]),
            "stencil_thickness": float(row["Stencil thickness"]),
            "paste_viscosity": float(row["Paste viscosity"]),
            "ambient_rh": float(row["Ambient RH"]),
            "ambient_temperature": float(row["Ambient temperature"])
        },

        "labels": {
            "defect": row["Defect"],
            "mechanism_causes": (
                None if pd.isna(row["mech causes"]) or row["mech causes"] == ""
                else row["mech causes"]
            ),
            "root_causes": (
                None if pd.isna(row["root causes"]) or row["root causes"] == ""
                else row["root causes"]
            )
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
