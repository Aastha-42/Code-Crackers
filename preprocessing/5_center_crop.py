import nibabel as nib
import numpy as np
import os

IN_DIR = "data/processed_mri/normalized"
OUT_DIR = "data/processed_mri/cropped"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_SHAPE = (128, 128, 128)

for f in os.listdir(IN_DIR):

    # ✅ skip non-NIfTI files
    if not f.endswith((".nii", ".nii.gz")):
        continue

    in_path = os.path.join(IN_DIR, f)

    try:
        img = nib.load(in_path)
        x = img.get_fdata()

        # ensure 3D
        if x.ndim != 3:
            print(f"Skipping {f}: not 3D")
            continue

        # center crop
        start = [
            max(0, (x.shape[i] - TARGET_SHAPE[i]) // 2)
            for i in range(3)
        ]
        end = [
            start[i] + min(TARGET_SHAPE[i], x.shape[i])
            for i in range(3)
        ]

        cropped = x[
            start[0]:end[0],
            start[1]:end[1],
            start[2]:end[2]
        ]

        # pad if image is smaller than target
        pad_width = [
            (0, TARGET_SHAPE[i] - cropped.shape[i])
            for i in range(3)
        ]
        cropped = np.pad(cropped, pad_width, mode="constant")

        nib.save(
            nib.Nifti1Image(cropped.astype(np.float32), img.affine),
            os.path.join(OUT_DIR, f)
        )

    except Exception as e:
        print(f"Skipping {f}: {e}")

print("Center crop done ✅")
