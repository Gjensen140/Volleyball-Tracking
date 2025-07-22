import kagglehub
import shutil
import os

# Step 1: Download dataset
dataset_path = kagglehub.dataset_download("pythonistasamurai/volleyball-ball-object-detection-dataset")
print(f"Dataset downloaded to: {dataset_path}")

# Step 2: Move it to data path
target_path = "data/volleyball"
if not os.path.exists(target_path):
    shutil.copytree(dataset_path, target_path)
    print(f"Copied dataset to {target_path}")
else:
    print(f"{target_path} already exists.")
