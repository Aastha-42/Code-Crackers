import os
import numpy as np
import pandas as pd
import nibabel as nib

DATA_DIR = "data/processed_mri/final"

# def load_dataset(csv_path):
#     df = pd.read_csv(csv_path)

#     print(f"📄 Loaded CSV: {data\MRI_metadata.csv}")
#     print("Columns:", df.columns.tolist())
#     print("Total rows:", len(df))

#     X, y = [], []

#     for i, row in df.iterrows():
#         file_name = row.iloc[0]   # safer than name
#         label = row.iloc[1]

#         img_path = os.path.join(DATA_DIR, str(file_name))

#         if not os.path.exists(img_path):
#             print(f"❌ Missing MRI: {img_path}")
#             continue

#         img = nib.load(img_path).get_fdata()

#         features = [
#             img.mean(),
#             img.std(),
#             img.max(),
#             img.min()
#         ]

#         X.append(features)
#         y.append(label)

#     print(f"✅ Loaded samples: {len(X)}")

#     return np.array(X), np.array(y)
def load_dataset(csv_path):

    df = pd.read_csv(csv_path)

    print(f"Loaded CSV: {csv_path}")
    print("Columns:", df.columns.tolist())
    print("Total rows:", len(df))

    X, y = [], []
    for i, row in df.iterrows():

        file_name = row.iloc[0]
        label = row.iloc[1]

        img_path = os.path.join(DATA_DIR, str(file_name))

        if not os.path.exists(img_path):
            print(f"Missing MRI: {img_path}")
            continue

        img = nib.load(img_path).get_fdata()

        features = [
            img.mean(),
            img.std(),
            img.max()
        ]

        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)
