import os
from glob import glob

# Paths
dataset_path = "path to siim-acr-pneumothorax"
image_dir = os.path.join(dataset_path, "png_images")
mask_dir = os.path.join(dataset_path, "png_masks")
train_file = os.path.join(dataset_path, "train.txt")
val_file = os.path.join(dataset_path, "val.txt")

# Split ratio for train and validation sets
train_ratio = 0.8

# Get all image filenames (assumes images are in .jpg format)
image_files = sorted(glob(os.path.join(image_dir, "*.png")))
mask_files = sorted(glob(os.path.join(mask_dir, "*.png")))

# Check if the number of images and masks match
assert len(image_files) == len(mask_files), "Mismatch between image and mask count!"

# Extract base filenames (without extensions)
base_names = [os.path.splitext(os.path.basename(file))[0] for file in image_files]

# Shuffle and split into train and validation sets
import random
random.seed(21)
random.shuffle(base_names)

split_idx = int(len(base_names) * train_ratio)
train_names = base_names[:split_idx]
val_names = base_names[split_idx:]

# Write to train.txt
with open(train_file, "w") as f:
    f.write("\n".join(train_names))

# Write to val.txt
with open(val_file, "w") as f:
    f.write("\n".join(val_names))

print(f"`train.txt` and `val.txt` created successfully in {dataset_path}.")
