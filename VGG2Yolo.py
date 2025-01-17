import os
import json
import cv2
import shutil
import zipfile

# base_folder = "/Users/esmaeilahmadi/Documents/Work/WatchTower/Yolo-v8/train0"
base_folder = "D:/Work/WatchTower/Yolo-v8/train0"
json_path = os.path.join(base_folder, "via/v1/")
json_path = os.path.join(base_folder, "via/v2/")
image_path = os.path.join(base_folder, "Dataset/")
image_path = os.path.join(base_folder, "via/v2/")
output_path = os.path.join(base_folder, "yolo/v1/")
img_type = '.jpg'
annot_type = '.txt'


def unzip_file(zip_file_path, extract_to):
    """Unzip the specified zip file to the given directory."""
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def process_json_file(dataset_name):
    """Process a single JSON file and extract annotations."""
    with open(os.path.join(json_path, dataset_name), 'r') as json_file:
        data = json.load(json_file)

    # Create output folders for the dataset
    # dataset_name_cleaned = dataset_name.replace('.json', '').replace('dataset', '')
    dataset_name_cleaned = dataset_name.replace('.json', '').replace('dataset', '')

    print(dataset_name_cleaned)
    os.makedirs(os.path.join(output_path, 'labels/', dataset_name_cleaned), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images/', dataset_name_cleaned), exist_ok=True)

    # Iterate through image metadata
    for image_id, metadata in data['_via_img_metadata'].items():
        filename = str(metadata['filename']).replace(img_type, '')
        regions = metadata['regions']

        if not regions:
            continue

        img_path = os.path.join(image_path, dataset_name_cleaned, filename + img_type)

        # Check if the image file exists
        if not os.path.isfile(img_path):
            # print(f"Image not found: {img_path}")
            continue  # Skip to the next iteration if the image does not exist

        # Read the image
        img = cv2.imread(img_path)
        # Check if the image was read successfully
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue  # Skip to the next iteration if the image could not be read

        # Get image dimensions
        img_height, img_width, _ = img.shape

        # Create a text file for the image annotations
        with open(os.path.join(output_path, 'labels/', dataset_name_cleaned, filename + annot_type), 'w') as text_file:
            for region in regions:
                # Get polyline coordinates
                x_coords = region["shape_attributes"]["all_points_x"]
                y_coords = region["shape_attributes"]["all_points_y"]

                # Find the bounding box that encloses the polyline
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                # Calculate the center and size of the bounding box
                x_center = (x_min + x_max) / 2.0
                y_center = (y_min + y_max) / 2.0

                # Normalize coordinates (relative to image size)
                x_center_norm = x_center / img_width
                y_center_norm = y_center / img_height
                width_norm = (x_max - x_min) / img_width
                height_norm = (y_max - y_min) / img_height

                # Assuming class is stored under "Class" in region_attributes
                class_id = int(region["region_attributes"].get("Class", -1))
                if class_id not in [0, 1]:
                    print(f"{filename} class {class_id} is not valid")
                    continue

                # Create YOLO format annotation
                yolo_annotation = f"{class_id} {x_center_norm:.4f} {y_center_norm:.4f} {width_norm:.4f} {height_norm:.4f}\n"
                text_file.write(yolo_annotation)

        # Copy the image to the output directory
        shutil.copy(img_path, os.path.join(output_path, "images/", dataset_name_cleaned, filename + img_type))


# Clear output path if it exists
if os.path.exists(output_path):
    # os.rmdir(output_path)
    shutil.rmtree(output_path, ignore_errors=True)

# Create output directories
if not os.path.exists(output_path):
    os.makedirs(os.path.join(output_path, 'labels/'))
    os.makedirs(os.path.join(output_path, 'images/'))
    
    
# Process each JSON annotation file in the specified directory
for dataset_name in os.listdir(json_path):
    if dataset_name.endswith(('.json', '.JSON')):
        zip_file_path = os.path.join(image_path, dataset_name.replace('.json', '.zip'))
        
        # Check if the zip file exists
        if os.path.isfile(zip_file_path):
            extract_to = image_path
            unzip_file(zip_file_path, extract_to)
            process_json_file(dataset_name)
        else:
            print(f"Zip file not found for {dataset_name}: {zip_file_path}")
        