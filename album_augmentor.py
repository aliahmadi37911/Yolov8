import cv2
import numpy as np
import albumentations as album
import os

BASE_FOLDER = "D:/Work/WatchTower/Yolo-v8/YOLOv8/"
IMG_PATH = os.path.join(BASE_FOLDER, "yolo/v1/images")
LABEL_PATH = os.path.join(BASE_FOLDER, "yolo/v1/labels")

def read_bboxes_from_file(file_path):
    bboxes = []
    class_labels = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            class_index = int(parts[0])  # Class index
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            bboxes.append([x_center, y_center, width, height])
            class_labels.append(class_index)
    return np.array(bboxes), class_labels

def augment_image(image, bboxes, class_labels):
    # Define the augmentation pipeline
    transform = album.Compose([
        album.Rotate(limit=0,p=0),
        album.VerticalFlip(p=0),
        album.HorizontalFlip(p=0.5),
        album.RandomBrightnessContrast(p=0.2),
        album.Resize(height=640, width=640),  # Resize to YOLOv8 input size
        album.GaussianBlur(p=0.2),
        album.GaussNoise(p=0.1),
        album.InvertImg(p=0.2),
        album.ToGray(p=0.1)
    ], bbox_params=album.BboxParams(format='yolo', label_fields=['class_labels']))

    try:
        # Perform the augmentation
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        return augmented['image'], augmented['bboxes']
    except Exception as e:
        print(f"Error during augmentation: {e}")
        # Return the original image and bounding boxes if an error occurs
        return [], []


def process_images_in_folder(images_folder, labels_folder):

    # Iterate through all files in the image folder
    for foldername in  os.listdir(images_folder):
        print(f'{foldername}')
        for filename in os.listdir(os.path.join(images_folder, foldername)):
            if filename.endswith(('.jpg', '.jpeg', '.png')):  # Check for image file types
                image_path = os.path.join(images_folder, foldername)
                image_path = os.path.join(image_path, filename)
                
                bbox_path = os.path.join(labels_folder, foldername)
                bbox_path = os.path.join(bbox_path, filename.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))

                # Load the image
                image = cv2.imread(image_path)

                # Read bounding boxes from the corresponding text file
                bboxes, class_labels = read_bboxes_from_file(bbox_path)

                # Augment the image and bounding boxes
                # print(f'{foldername}/{filename}')
                augmented_image, augmented_bboxes = augment_image(image, bboxes, class_labels)
                
                if(len(augmented_image) and len(augmented_bboxes)):
                    # Save the augmented image with "aug_" prefix
                    output_image_path = os.path.join(images_folder, foldername)
                    output_image_path = os.path.join(output_image_path, f'aug_{filename}')
                    cv2.imwrite(output_image_path, augmented_image)

                    # Optionally, save the augmented bounding boxes to a new text file
                    output_bbox_path = os.path.join(labels_folder, foldername)
                    output_bbox_path = os.path.join(output_bbox_path, f'aug_{filename.replace(".jpg", ".txt").replace(".jpeg", ".txt").replace(".png", ".txt")}')
                    with open(output_bbox_path, 'w') as f:
                        i = 0
                        for bbox in augmented_bboxes:
                            # Assuming class labels are still the same
                            f.write(f'{class_labels[i]} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n')  
                            i += 1
                    
if __name__ == "__main__":
    process_images_in_folder(IMG_PATH, LABEL_PATH)
