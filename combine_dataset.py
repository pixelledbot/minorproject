import os
import shutil

# Path to original dataset
source_dataset = "Dataset/Baackground_Removed"

# New combined dataset
target_dataset = "dataset_combined"

# Category mapping
mapping = {

    "sharps": [
        "scalpels",
        "episiotomy_scissors",
        "mayo_scissors",
        "stitch_removal_scissors",
        "forceps",
        "tweezers",
        "hemostats"
    ],

    "infectious": [
        "blood_soaked_bandages",
        "human_organs",
        "used_masks",
        "used_medical_gloves"
    ],

    "pharmaceutical": [
        "ampoules_full",
        "ampoules_broken"
    ],

    "general": [
        "syrup_bottles",
        "iv_bottles",
        "disinfectant_bottles",
        "used_medical_paper",
        "general_organic_waste",
        "waterbottles"
    ]
}

# Create category folders
for category in mapping:
    os.makedirs(os.path.join(target_dataset, category), exist_ok=True)

# Copy images
for category, folders in mapping.items():

    print(f"\nProcessing category: {category}")

    for folder in folders:

        source_folder = os.path.join(source_dataset, folder)

        if not os.path.exists(source_folder):
            print(f"Folder not found: {source_folder}")
            continue

        images = os.listdir(source_folder)

        print(f"Copying {len(images)} images from {folder}")

        for img in images:

            src = os.path.join(source_folder, img)
            dst = os.path.join(target_dataset, category, img)

            shutil.copy(src, dst)

print("\nDataset successfully combined!")
