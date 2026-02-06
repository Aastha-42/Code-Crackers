import nibabel as nib
import numpy as np
import os

IN_DIR = "data/nifti"
OUT_DIR = "data/processed_mri/stripped"
os.makedirs(OUT_DIR, exist_ok=True)

for f in os.listdir(IN_DIR):
    img = nib.load(os.path.join(IN_DIR, f))
    data = img.get_fdata()

    mask = data > np.mean(data)
    brain = data * mask

    nib.save(
        nib.Nifti1Image(brain, img.affine),
        os.path.join(OUT_DIR, f)
    )

print("Skull stripping done ✅")
