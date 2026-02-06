import pandas as pd
import os

META = pd.read_csv("data/filtered_metadata.csv")
NIFTI_DIR = "data/nifti"

available = [f.replace(".nii.gz", "") for f in os.listdir(NIFTI_DIR)]

df = META[META["Subject"].isin(available)]

df = df[["Subject", "Group"]]
df.columns = ["subject_id", "label"]

df.to_csv("data/processed_mri/labels.csv", index=False)

print("Final subjects:", len(df))
print(df["label"].value_counts())
