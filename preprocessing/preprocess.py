import os
import logging
import pandas as pd
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import warnings

# Optional: N4 bias-field correction using SimpleITK (uncomment if you want)
# import SimpleITK as sitk

# -----------------------
# CONFIG
# -----------------------
CSV_PATH = "data/raw_mri/MRI_metadata.csv"
MRI_FOLDER = "data/raw_mri/mri"
SAVE_FOLDER = "data/processed_mri"
TARGET_SHAPE = (128, 128, 128)   # desired output shape
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)
USE_BIAS_CORRECTION = False      # set True if you want to apply N4 (requires SimpleITK)
REORIENT = True                  # set False to skip reorientation
CLIP_PERCENTILES = (1, 99)       # (low_pct, high_pct) for intensity clipping
NORMALIZE_METHOD = "minmax"      # options: "minmax" or "zscore"
STRATIFY_BY_SUBJECT = True       # if true and multiple scans per subject, split by subject

os.makedirs(SAVE_FOLDER, exist_ok=True)

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# -----------------------
# LOAD CSV and basic cleaning
# -----------------------
df = pd.read_csv(CSV_PATH)
df = df.rename(columns={
    "Subject": "subject_id",
    "Group": "diagnosis",
    "ImageID": "image_id"
})

logging.info(f"CSV loaded, {len(df)} rows")
print("CSV columns:", df.columns.tolist())

# drop rows missing critical fields
required_cols = ["subject_id", "diagnosis"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"CSV missing required column: {c}")

df = df.dropna(subset=required_cols).copy()

# standardize subject_id to string
df["subject_id"] = df["subject_id"].astype(str).str.strip()

# optional: if your CSV has a filename column, use it; else assume subject_id + .nii or .nii.gz
df["file_name"] = df["image_id"].astype(str).str.strip()

# ensure extension
df["file_name"] = df["file_name"].apply(
    lambda x: x if x.endswith((".nii", ".nii.gz")) else x + ".nii.gz"
)

# label mapping
label_map = {"CN": 0, "MCI": 1, "AD": 2}
df["label"] = df["diagnosis"].map(label_map)
missing_labels = df["label"].isna().sum()
if missing_labels:
    logging.warning(f"{missing_labels} rows have diagnoses not in label_map and will be dropped")
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

# -----------------------
# ensure files exist (try common extensions)
# -----------------------
def find_file(fn):
    # check the provided name first, then .nii.gz, then .nii
    candidates = [
        os.path.join(MRI_FOLDER, fn),
        os.path.join(MRI_FOLDER, fn + ".nii.gz"),
        os.path.join(MRI_FOLDER, fn + ".nii"),
        os.path.join(MRI_FOLDER, fn.replace(".nii.gz", ".nii")),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

df["filepath"] = df["file_name"].apply(find_file)
n_missing_files = df["filepath"].isna().sum()
if n_missing_files:
    logging.warning(f"{n_missing_files} scans missing on disk — they will be dropped")
df = df.dropna(subset=["filepath"]).reset_index(drop=True)
logging.info(f"{len(df)} rows remain after file existence check")

# if multiple scans per subject and we want subject-level split later, keep note
scan_counts = df["subject_id"].value_counts()
if (scan_counts > 1).any():
    logging.info("Some subjects have multiple scans (counts):")
    logging.info(scan_counts.loc[scan_counts > 1].head().to_dict())

# -----------------------
# processing helpers
# -----------------------
def reorient_to_canonical(img):
    # returns nibabel image reoriented to closest canonical RAS
    return nib.as_closest_canonical(img)

def n4_bias_field_correction(nib_img):
    """Optional: perform N4 bias correction using SimpleITK. Requires SimpleITK."""
    sitk_img = sitk.GetImageFromArray(nib_img.get_fdata().astype(np.float32))
    sitk_img.SetSpacing(nib_img.header.get_zooms())
    maskImage = sitk.OtsuThreshold(sitk_img, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = corrector.Execute(sitk_img, maskImage)
    arr = sitk.GetArrayFromImage(corrected)
    return arr

def resize_volume(img, target_shape=TARGET_SHAPE):
    # img: numpy array
    factors = [t / s for s, t in zip(img.shape, target_shape)]
    # order=1 : linear interpolation (good tradeoff)
    return zoom(img, factors, order=1)

def normalize_volume(img, method=NORMALIZE_METHOD, clip_pcts=CLIP_PERCENTILES):
    # clip by percentiles to reduce extreme outliers
    if np.isnan(img).any():
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.nanpercentile(img, clip_pcts)
    if high == low:
        # degenerate case
        return np.zeros_like(img, dtype=np.float32)
    img = np.clip(img, low, high)
    if method == "minmax":
        img = (img - low) / (high - low)
    elif method == "zscore":
        m = img.mean()
        s = img.std() if img.std() > 0 else 1.0
        img = (img - m) / s
    else:
        raise ValueError("normalize method unknown")
    return img.astype(np.float32)

def process_one(row):
    """Process one scan. Returns dict with metadata and processed array (or None on error)."""
    subject_id = row["subject_id"]
    label = int(row["label"])
    fp = row["filepath"]
    try:
        nib_image = nib.load(fp)
        if REORIENT:
            nib_image = reorient_to_canonical(nib_image)
        arr = nib_image.get_fdata()
        # some scans may have 4th dim = 1, squeeze
        if arr.ndim == 4 and arr.shape[3] == 1:
            arr = np.squeeze(arr, axis=3)
        if arr.ndim != 3:
            raise ValueError(f"Unexpected arr ndim={arr.ndim}")
        # optional bias correction
        if USE_BIAS_CORRECTION:
            # needs SimpleITK; do only if configured
            arr = n4_bias_field_correction(nib_image)
        # handle NaNs
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        # resize and normalize
        arr_resized = resize_volume(arr, target_shape=TARGET_SHAPE)
        arr_norm = normalize_volume(arr_resized)  # returns float32
        meta = {
            "subject_id": subject_id,
            "original_file": fp,
            "label": label,
            "orig_shape": arr.shape,
            "processed_shape": arr_norm.shape,
            "min": float(arr_norm.min()),
            "max": float(arr_norm.max()),
            "mean": float(arr_norm.mean()),
            "std": float(arr_norm.std())
        }
        return meta, arr_norm
    except Exception as e:
        return {"error_for": fp, "error": str(e)}, None

# -----------------------
# process dataset (parallel)
# -----------------------
processed_meta = []
processed_arrays = []
failed = []

logging.info(f"Starting processing with {NUM_WORKERS} workers...")

rows = df.to_dict(orient="records")
with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
    futures = [ex.submit(process_one, r) for r in rows]
    for fut in tqdm(futures, total=len(futures)):
        try:
            meta, arr = fut.result()
            if arr is None:
                failed.append(meta)
            else:
                processed_meta.append(meta)
                processed_arrays.append(arr)
        except Exception as e:
            failed.append({"error": str(e)})

logging.info(f"Processed {len(processed_arrays)} scans successfully, {len(failed)} failed")

if len(processed_arrays) == 0:
    raise RuntimeError("No scans processed successfully. Check logs.")

# -----------------------
# convert to numpy arrays & metadata dataframe
# -----------------------
X = np.stack(processed_arrays).astype(np.float32)
y = np.array([m["label"] for m in processed_meta], dtype=np.int64)
meta_df = pd.DataFrame(processed_meta)

logging.info(f"Final dataset shape: {X.shape}, labels shape: {y.shape}")
meta_df.to_csv(os.path.join(SAVE_FOLDER, "processed_metadata.csv"), index=False)

# -----------------------
# train/val/test split
# -----------------------
if STRATIFY_BY_SUBJECT:
    # do subject-level split: take unique subject -> choose train/val/test subjects
    subj_df = meta_df[["subject_id", "label"]].drop_duplicates().reset_index(drop=True)
    strat = subj_df["label"].values
    train_subj, temp_subj = train_test_split(subj_df["subject_id"], test_size=0.3, random_state=42, stratify=strat)
    val_subj, test_subj = train_test_split(temp_subj, test_size=0.5, random_state=42,
                                          stratify=subj_df.set_index("subject_id").loc[temp_subj]["label"].values)
    # map scans to splits
    meta_df["split"] = meta_df["subject_id"].apply(lambda s: "train" if s in set(train_subj) else ("val" if s in set(val_subj) else "test"))
else:
    # split by scans with stratify by label
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    # save and exit
    np.savez_compressed(os.path.join(SAVE_FOLDER, "dataset_splits.npz"),
                        X_train=X_train, y_train=y_train,
                        X_val=X_val, y_val=y_val,
                        X_test=X_test, y_test=y_test)
    logging.info("Saved compressed splits (scan-level).")
    raise SystemExit("Completed (scan-level split).")

# if subject-level: build arrays for each split
train_mask = meta_df["split"] == "train"
val_mask = meta_df["split"] == "val"
test_mask = meta_df["split"] == "test"

X_train = X[train_mask.values]
y_train = y[train_mask.values]
X_val = X[val_mask.values]
y_val = y[val_mask.values]
X_test = X[test_mask.values]
y_test = y[test_mask.values]

logging.info("Split counts (train/val/test): %d / %d / %d" % (len(y_train), len(y_val), len(y_test)))

# -----------------------
# save arrays and metadata
# -----------------------
np.savez_compressed(os.path.join(SAVE_FOLDER, "X_y_splits.npz"),
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    X_test=X_test, y_test=y_test)

meta_df.to_csv(os.path.join(SAVE_FOLDER, "processed_metadata_with_splits.csv"), index=False)
logging.info("Saved processed arrays and metadata. ALL DONE 🎉")

