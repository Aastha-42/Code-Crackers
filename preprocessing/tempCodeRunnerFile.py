import nibabel as nib
import numpy as np
import os

IN_DIR = "data/processed_mri/stripped"
OUT_DIR = "data/processed_mri/normalized"
os.makedirs(OUT_DIR, exist_ok=True)

for f in os.listdir(IN_DIR):
    img = nib.load(os.path.join(IN_DIR, f))
    x = img.get_fdata()

    x = (x - x.mean()) / (x.std() + 1e-8)

    nib.save(
        nib.Nifti1Image(x, img.affine),
        os.path.join(OUT_DIR, f)
    )

print("Normalization done ✅")
  