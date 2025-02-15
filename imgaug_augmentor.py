import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
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
    # Define the augmentation sequence
    seq = iaa.Sequential([
        iaa.Rotate(rotate=(0, 0)),  # Rotate between -30 and 30 degrees
        iaa.Fliplr(0),                # Horizontal flip with 50% probability
        iaa.Flipud(0.5),                # Vertical flip with 50% probability
        iaa.Multiply((0.9, 1.1)),       # Change brightness
        iaa.LinearContrast((0.9, 1.1)), # Change contrast
        iaa.Resize({"height": 640, "width": 640})  # Resize to 640x640
    ])

    # Convert bounding boxes to imgaug format
    bboxes_imgaug = ia.BoundingBoxesOnImage(
        [ia.BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]) for bbox in bboxes],
        shape=image.shape
    )

    # Augment the image and bounding boxes
    image_aug, bboxes_aug = seq(image=image, bounding_boxes=bboxes_imgaug)

    # Convert back to the original format (list of bounding boxes)
    bboxes_aug = bboxes_aug.bounding_boxes
    bboxes_aug_list = [[bbox.x1, bbox.y1, bbox.x2, bbox.y2] for bbox in bboxes_aug]

    return image_aug, bboxes_aug_list


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
                        for bbox in augmented_bboxes:
                            # Assuming class labels are still the same
                            f.write(f'0 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n')  # Change '0' to the appropriate class index if needed
                    
if __name__ == "__main__":
    process_images_in_folder(IMG_PATH, LABEL_PATH)
