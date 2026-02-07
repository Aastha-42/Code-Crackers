import nibabel as nib
import numpy as np
import os

IN_DIR = "data/nifti"
OUT_DIR = "data/processed_mri/stripped"
os.makedirs(OUT_DIR, exist_ok=True)

for f in os.listdir(IN_DIR):

    # 🔴 Skip non-NIfTI files
    if not f.endswith((".nii", ".nii.gz")):
        continue

    in_path = os.path.join(IN_DIR, f)

    try:
        img = nib.load(in_path)
        data = img.get_fdata()

        mask = data > np.mean(data)
        brain = data * mask

        nib.save(
            nib.Nifti1Image(brain.astype(np.float32), img.affine),
            os.path.join(OUT_DIR, f)
        )

    except Exception as e:
        print(f"Skipping {f}: {e}")

print("Skull stripping done ✅")
