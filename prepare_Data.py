import os
import shutil
import random
import math 

# Base folder paths
BASE_FOLDER = "D:/Work/WatchTower/Yolo-v8/YOLOV8/"
IMAGE_DATA_PATH = os.path.join(BASE_FOLDER, 'yolo/v1/images/')
LABEL_DATA_PATH = os.path.join(BASE_FOLDER, 'yolo/v1/labels/')

# Output folder paths
BASE_OUTPUT_FOLDER = "D:/Work/WatchTower/Yolo-v8/YOLOV8/datasets/"
TRAIN_IMAGE_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, 'images/train')
VAL_IMAGE_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, 'images/val')
TRAIN_LABEL_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, 'labels/train')
VAL_LABEL_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, 'labels/val')

# File types
LABEL_TYPE = '.txt'
IMG_TYPE = '.jpg'

def copy_data(dataset_name, image_folder, label_folder, target_image_folder, target_label_folder):
    """Copy images and labels to the target folders with new names."""
    dataset_name = dataset_name.replace('dataset', '').replace('_', '')
    image_files = [f for f in os.listdir(image_folder) if f.endswith(IMG_TYPE)]

    for img_file in image_files:
        new_img_file = f"{dataset_name}{img_file}"
        label_file = img_file.replace(IMG_TYPE, LABEL_TYPE)
        new_label_file = f"{dataset_name}{label_file}"

        # Copy image and label files
        shutil.copy(os.path.join(image_folder, img_file), os.path.join(target_image_folder, new_img_file))
        shutil.copy(os.path.join(label_folder, label_file), os.path.join(target_label_folder, new_label_file))

    print(f"Data copy complete for {dataset_name}: {len(image_files)} images copied.")

def get_all_folders(base_folder):
    """Return a list of all non-empty subdirectories in the specified base folder."""
    non_empty_folders = []
    for item in os.listdir(base_folder):
        item_path = os.path.join(base_folder, item)
        if os.path.isdir(item_path) and os.listdir(item_path):  # Check if the directory is not empty
            non_empty_folders.append(item)
    return non_empty_folders

def split_folders(all_folders, val_size=0.1, random_seed=42):
    """Randomly split folders into training and validation sets."""

    random.seed(random_seed)

    train_folders = []
    val_folders = []
    
    # Shuffle the list to randomize the order
    random.shuffle(all_folders)
    
    num_val_folders = round(val_size*len(all_folders))
    # print(num_val_folders)
    
    # random_indices = random.sample(range(len(all_folders)), num_val_folders)
    # print("random_indices",random_indices)
    # for i in random_indices: 
    #     print("i",i)
    #     val_folders.append(all_folders.pop(i-1))

    i = 0 
    while( i<round(num_val_folders)):
        i += 1
        # Pick a random number between 0 and len(all_folders) - 1
        random_index = random.randint(0, len(all_folders) - 1)
        # Output the random index and the corresponding folder
        val_folders.append(all_folders.pop(round(random_index)))
    
    train_folders = all_folders
    
    print("Training folders:", train_folders)
    print("Validation folders:", val_folders)

    for folder in train_folders:
        copy_data(folder, os.path.join(IMAGE_DATA_PATH, folder), os.path.join(LABEL_DATA_PATH, folder), 
                  TRAIN_IMAGE_FOLDER, TRAIN_LABEL_FOLDER)

    for folder in val_folders:
        copy_data(folder, os.path.join(IMAGE_DATA_PATH, folder), os.path.join(LABEL_DATA_PATH, folder), 
                  VAL_IMAGE_FOLDER, VAL_LABEL_FOLDER)



def prepareData():
    # Clear the output folder if it exists
    if os.path.exists(BASE_OUTPUT_FOLDER):
        shutil.rmtree(BASE_OUTPUT_FOLDER, ignore_errors=True)

    # Create the output folder structure
    os.makedirs(TRAIN_IMAGE_FOLDER, exist_ok=True)
    os.makedirs(VAL_IMAGE_FOLDER, exist_ok=True)
    os.makedirs(TRAIN_LABEL_FOLDER, exist_ok=True)
    os.makedirs(VAL_LABEL_FOLDER, exist_ok=True)
    # Get all non-empty folders and split them
    all_folders = get_all_folders(LABEL_DATA_PATH)
    print("Non-empty folders found:", all_folders)
    split_folders(all_folders, val_size=0.1 , random_seed=40)
    
if __name__ == "__main__":
    prepareData()

