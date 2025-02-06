from ultralytics import YOLO
import ultralytics
ultralytics.checks()
import os
import torch

torch.cuda.set_device(0)
# Path to the YAML configuration file that contains dataset information
yamlPath = "D:/Work/WatchTower/Yolo-v8/Yolov8/fire-seg.yaml"

# Load the model (You can choose the model based on your task: detection or segmentation)
# Example for detection:
model = YOLO('yolov8n.pt')  # Load YOLOv8 model for detection

# Train the model on CPUpip uninstall torchvision        

# results_cpu = model.train(data=yamlPath, epochs=50, imgsz=720, device='cpu')
# print("Training complete on CPU.")

# Train the model on a single GPU (GPU 0)
results_gpu_1 = model.train(data=yamlPath, epochs=500, imgsz=640, device=0, workers=0, scale=0.9, fliplr=1, bgr = 0.1, mosaic=0.01, crop_fraction=1.0)
print("Training complete on GPU 0.")

# Evaluate the model's performance on the validation set
# This can be done after the training is complete for each result
results = model.val()
print("Evaluation complete.")
