import pandas as pd
from sklearn.model_selection import train_test_split
import os

CSV_PATH = "data/MRI_metadata.csv"
SPLIT_DIR = "data/splits"

# Load metadata
df = pd.read_csv(CSV_PATH)
print(f"Loaded MRI_metadata.csv with {len(df)} rows")

# ---------------- SAFETY CHECKS ---------------- #

required_cols = ["ImageID", "Group"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

# Map labels
label_map = {
    "CN": 0,
    "MCI": 1,
    "AD": 2
}

df["label"] = df["Group"].map(label_map)
df = df.dropna(subset=["label"])

if df.empty:
    raise ValueError("No valid samples after label mapping ❌")

# Optional: keep only downloaded scans
if "Downloaded" in df.columns:
    df = df[df["Downloaded"].astype(str).str.lower() == "yes"]

# Create filename column (matches MRI files)
df["file_name"] = df["ImageID"].astype(str) + ".nii"

print("Class distribution:")
print(df["label"].value_counts())

# ------------------------------------------------ #

# 70 / 15 / 15 split (stratified)
train, temp = train_test_split(
    df,
    test_size=0.3,
    stratify=df["label"],
    random_state=42
)

val, test = train_test_split(
    temp,
    test_size=0.5,
    stratify=temp["label"],
    random_state=42
)

os.makedirs(SPLIT_DIR, exist_ok=True)

train[["file_name", "label"]].to_csv(
    os.path.join(SPLIT_DIR, "train.csv"), index=False
)
val[["file_name", "label"]].to_csv(
    os.path.join(SPLIT_DIR, "val.csv"), index=False
)
train[["file_name", "label"]].to_csv(
    "data/splits/train.csv", index=False
)

val[["file_name", "label"]].to_csv(
    "data/splits/val.csv", index=False
)

test[["file_name", "label"]].to_csv(
    "data/splits/test.csv", index=False
)
