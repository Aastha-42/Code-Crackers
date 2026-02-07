import nibabel as nib
import numpy as np
import os

IN_DIR = "data/processed_mri/stripped"
OUT_DIR = "data/processed_mri/normalized"
os.makedirs(OUT_DIR, exist_ok=True)

for f in os.listdir(IN_DIR):

    # ✅ skip non-NIfTI files
    if not f.endswith((".nii", ".nii.gz")):
        continue

    in_path = os.path.join(IN_DIR, f)

    try:
        img = nib.load(in_path)
        x = img.get_fdata()

        # guard against empty / constant images
        std = x.std()
        if std == 0:
            print(f"Skipping {f}: zero variance")
            continue

        x = (x - x.mean()) / (std + 1e-8)

        nib.save(
            nib.Nifti1Image(x.astype(np.float32), img.affine),
            os.path.join(OUT_DIR, f)
        )

    except Exception as e:
        print(f"Skipping {f}: {e}")

print("Normalization done ✅")
