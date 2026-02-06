import pandas as pd

df = pd.read_csv("data/mri_metadata.csv")

# keep only MRI + downloaded + baseline
df = df[
    (df["Modality"] == "MRI") &
    (df["Downloaded"] == "Yes") &
    (df["Visit"] == "bl")
]

print("Total usable scans:", len(df))
print(df[["ImageID", "Subject", "Group"]].head())

df.to_csv("data/filtered_metadata.csv", index=False)
