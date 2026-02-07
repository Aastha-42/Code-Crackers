import pandas as pd
import numpy as np
import nibabel as nib
import os

DATA_DIR = "data/processed_mri/final"

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    X = []
    y = []

    for _, row in df.iterrows():
        file_name = row["file_name"]
        label = row["label"]

        img_path = os.path.join(DATA_DIR, file_name)

        if not os.path.exists(img_path):
            continue

        img = nib.load(img_path).get_fdata()

        # ---- SIMPLE FEATURE EXTRACTION ----
        # Option 1 (safe for logistic regression)
        features = [
            img.mean(),
            img.std(),
            img.max(),
            img.min()
        ]

        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)
