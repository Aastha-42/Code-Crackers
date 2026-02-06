import os
import dicom2nifti
import pandas as pd

RAW_DICOM = "data/raw_dicom"
OUT_NIFTI = "data/nifti"
META = pd.read_csv("data/filtered_metadata.csv")

os.makedirs(OUT_NIFTI, exist_ok=True)

for _, row in META.iterrows():
    image_id = row["ImageID"]
    dicom_folder = os.path.join(RAW_DICOM, image_id)
    out_file = os.path.join(OUT_NIFTI, f"{row['Subject']}.nii.gz")

    if os.path.exists(dicom_folder) and not os.path.exists(out_file):
        dicom2nifti.convert_directory(
            dicom_folder,
            OUT_NIFTI,
            compression=True,
            reorient=True
        )

print("DICOM → NIfTI conversion done ✅")
