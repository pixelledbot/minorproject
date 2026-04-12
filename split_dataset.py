import os
import shutil
import random
import sys

# ---------------- FIX WINDOWS ENCODING ----------------
sys.stdout.reconfigure(encoding='utf-8')

# ---------------- PATHS ----------------
CLEAN_DIR = r"c:\Users\pixel_t\Downloads\archive\dataset_clean"
RAW_DIR   = r"c:\Users\pixel_t\Downloads\archive\dataset_raw"

OUTPUT_DIR = r"c:\Users\pixel_t\Downloads\archive\dataset_split"

# ---------------- SPLIT RATIOS ----------------
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

# ---------------- GET CLASSES ----------------
classes = os.listdir(CLEAN_DIR)

splits = ["train", "val", "test"]

# ---------------- CREATE OUTPUT FOLDERS ----------------
for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

# ---------------- GET IMAGES ----------------
def get_images(folder):
    images = []
    for f in os.listdir(folder):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            images.append(os.path.join(folder, f))
    return images

# ---------------- PROCESS CLASS ----------------
def process_class(cls):
    clean_path = os.path.join(CLEAN_DIR, cls)
    raw_path = os.path.join(RAW_DIR, cls)

    all_images = []

    # add clean images
    if os.path.exists(clean_path):
        all_images += get_images(clean_path)

    # add raw images
    if os.path.exists(raw_path):
        all_images += get_images(raw_path)

    if len(all_images) == 0:
        print(f"[WARN] No images found for class: {cls}")
        return

    random.shuffle(all_images)

    total = len(all_images)
    train_end = int(total * TRAIN_RATIO)
    val_end = int(total * (TRAIN_RATIO + VAL_RATIO))

    train_imgs = all_images[:train_end]
    val_imgs = all_images[train_end:val_end]
    test_imgs = all_images[val_end:]

    def copy_files(file_list, split_name):
        for img_path in file_list:
            filename = os.path.basename(img_path)
            dst = os.path.join(OUTPUT_DIR, split_name, cls, filename)
            shutil.copy2(img_path, dst)

    copy_files(train_imgs, "train")
    copy_files(val_imgs, "val")
    copy_files(test_imgs, "test")

    print(f"[OK] {cls}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")

# ---------------- RUN ----------------
for cls in classes:
    process_class(cls)

print("\n[DONE] Dataset successfully split at:", OUTPUT_DIR)