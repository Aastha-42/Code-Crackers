import nibabel as nib
import numpy as np
import os

TARGET = (160, 160, 160)

def crop(x):
    s = [(x.shape[i] - TARGET[i]) // 2 for i in range(3)]
    return x[
        s[0]:s[0]+TARGET[0],
        s[1]:s[1]+TARGET[1],
        s[2]:s[2]+TARGET[2]
    ]

IN_DIR = "data/processed_mri/normalized"
OUT_DIR = "data/processed_mri/cropped"
os.makedirs(OUT_DIR, exist_ok=True)

for f in os.listdir(IN_DIR):
    img = nib.load(os.path.join(IN_DIR, f))
    cropped = crop(img.get_fdata())

    nib.save(
        nib.Nifti1Image(cropped, img.affine),
        os.path.join(OUT_DIR, f)
    )

print("Center cropping done ✅")
