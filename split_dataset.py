import os
import shutil
import random

source_dir = "dataset_combined"
train_dir = "data/train"
test_dir = "data/test"

split_ratio = 0.8

classes = ["general","infectious","pharmaceutical","sharps"]

for cls in classes:

    src_folder = os.path.join(source_dir, cls)

    train_folder = os.path.join(train_dir, cls)
    test_folder = os.path.join(test_dir, cls)

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    images = os.listdir(src_folder)
    random.shuffle(images)

    split_index = int(len(images) * split_ratio)

    train_images = images[:split_index]
    test_images = images[split_index:]

    for img in train_images:
        shutil.copy(
            os.path.join(src_folder, img),
            os.path.join(train_folder, img)
        )

    for img in test_images:
        shutil.copy(
            os.path.join(src_folder, img),
            os.path.join(test_folder, img)
        )

print("Dataset split complete!")