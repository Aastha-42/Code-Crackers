import nibabel as nib
import numpy as np
import os

IN_DIR = "data/processed_mri/cropped"
OUT_DIR = "data/processed_mri/final"
os.makedirs(OUT_DIR, exist_ok=True)

for f in os.listdir(IN_DIR):

    # ✅ Skip non-NIfTI files (important!)
    if not f.endswith((".nii", ".nii.gz")):
        continue

    in_path = os.path.join(IN_DIR, f)

    try:
        img = nib.load(in_path)
        x = img.get_fdata()

        # safety: avoid division by zero
        min_val = x.min()
        max_val = x.max()

        if max_val - min_val == 0:
            print(f"Skipping {f}: constant volume")
            continue

        # Min-Max normalization → [0, 1]
        x = (x - min_val) / (max_val - min_val)

        nib.save(
            nib.Nifti1Image(x.astype(np.float32), img.affine),
            os.path.join(OUT_DIR, f)
        )

    except Exception as e:
        print(f"Skipping {f}: {e}")

print("Final preprocessing done ✅")
