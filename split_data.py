import os
import shutil
import random
def split_dataset(parent_folder, output_folder="dataset", train_count=300, val_count=200):
    random.seed(42)  # For reproducibility
    train_folder = os.path.join(output_folder, "train")
    val_folder = os.path.join(output_folder, "val")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    for class_name in os.listdir(parent_folder):
        class_path = os.path.join(parent_folder, class_name)
        if not os.path.isdir(class_path):
            continue  # Skip files that are not folders

        images = [
            f for f in os.listdir(class_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'))
        ]
        if len(images) < train_count + val_count:
            print(f" Not enough images in {class_name}, found {len(images)}.")
            continue
        random.shuffle(images)
        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        # Create class-specific folders
        train_class_folder = os.path.join(train_folder, class_name)
        val_class_folder = os.path.join(val_folder, class_name)
        os.makedirs(train_class_folder, exist_ok=True)
        os.makedirs(val_class_folder, exist_ok=True)
        # Copy images
        for img in train_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_class_folder, img))
        for img in val_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(val_class_folder, img))
        print(f"{class_name}: {len(train_images)} train + {len(val_images)} val images copied.")
    print(f"\Dataset split completed in '{output_folder}'.")

# Example usage:
split_dataset("C:/Users/Nick/Desktop/FinalDataset")























