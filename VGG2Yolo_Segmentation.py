import os
import json
import cv2
import shutil
import zipfile

# BASE_FOLDER = "/Users/esmaeilahmadi/Documents/Work/WatchTower/Yolo-v8/train0"
BASE_FOLDER = "D:/Work/WatchTower/Yolo-v8/Yolov8"
JSON_PATH = os.path.join(BASE_FOLDER, "via/v2/")
IMAGE_PATH = os.path.join(BASE_FOLDER, "via/v2/")
OUTPUT_PATH = os.path.join(BASE_FOLDER, "yolo/v1/")
img_type = '.jpg'
annot_type = '.txt'


def unzip_file(zip_file_path, extract_to):
    """Unzip the specified zip file to the given directory."""
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def process_json_file(dataset_name):
    """Process a single JSON file and extract annotations."""
    with open(os.path.join(JSON_PATH, dataset_name), 'r') as json_file:
        data = json.load(json_file)

    # Create output folders for the dataset
    # dataset_name_cleaned = dataset_name.replace('.json', '').replace('dataset', '')
    dataset_name_cleaned = dataset_name.replace('.json', '').replace('dataset', '')

    print(dataset_name_cleaned)
    os.makedirs(os.path.join(OUTPUT_PATH, 'labels/', dataset_name_cleaned), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, 'images/', dataset_name_cleaned), exist_ok=True)

    # Iterate through image metadata
    for image_id, metadata in data['_via_img_metadata'].items():
        filename = str(metadata['filename']).replace(img_type, '')
        regions = metadata['regions']

        if not regions:
            continue

        img_path = os.path.join(IMAGE_PATH, dataset_name_cleaned, filename + img_type)

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
        with open(os.path.join(OUTPUT_PATH, 'labels/', dataset_name_cleaned, filename + annot_type), 'w') as text_file:
            for region in regions:
                # Assuming class is stored under "Class" in region_attributes
                class_id = int(region["region_attributes"].get("Class", -1))
                if class_id not in [0, 1]:
                    print(f"{filename} class {class_id} is not valid")
                    continue
                
                # Get polyline coordinates
                all_points_x = region["shape_attributes"]["all_points_x"]
                all_points_y = region["shape_attributes"]["all_points_y"]

                values = [] 
                values.append(class_id) # Adding The Class
                
                for i in range(min(len(all_points_x), len(all_points_y))):
                    values.append(round(all_points_x[i]/img_width, 4))
                    values.append(round(all_points_y[i]/img_height, 4))
                
                # Create YOLO format annotation
                region_annotation = ' '.join(map(str, values))
                
                # Write YOLO format annotation
                text_file.write(region_annotation+'\n')

        # Copy the image to the output directory
        shutil.copy(img_path, os.path.join(OUTPUT_PATH, "images/", dataset_name_cleaned, filename + img_type))


# Clear output path if it exists
if os.path.exists(OUTPUT_PATH):
    # os.rmdir(OUTPUT_PATH)
    shutil.rmtree(OUTPUT_PATH, ignore_errors=True)

# Create output directories
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(os.path.join(OUTPUT_PATH, 'labels/'))
    os.makedirs(os.path.join(OUTPUT_PATH, 'images/'))
    
    
# Process each JSON annotation file in the specified directory
def converVGGtoYolo():
    for dataset_name in os.listdir(JSON_PATH):
        if dataset_name.endswith(('.json', '.JSON')):
            zip_file_path = os.path.join(IMAGE_PATH, dataset_name.replace('.json', '.zip'))
            
            # Check if the zip file exists
            if os.path.isfile(zip_file_path):
                extract_to = IMAGE_PATH
                unzip_file(zip_file_path, extract_to)
                process_json_file(dataset_name)
            else:
                print(f"Zip file not found for {dataset_name}: {zip_file_path}")
            
            
if __name__ == "__main__":
    converVGGtoYolo()
        