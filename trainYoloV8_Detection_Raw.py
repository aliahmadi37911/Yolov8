from ultralytics import YOLO
import ultralytics
ultralytics.checks()

import torch

torch.cuda.set_device(0)

# Path to the YAML configuration file that contains dataset information
yamlPath = "D:/Work/WatchTower/Yolo-v8/Yolov8/fire-seg.yaml"

# yamlPath = "D:/Work/WatchTower/Yolo-v8/train0/coco8-seg.yaml"

# coco_gt = coco.loadAnns(coco.getAnnIds())
# coco_dt = coco.loadRes(predictions)
# coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
# coco_eval.evaluate()
# coco_eval.accumulate()
# coco_eval.summarize()



# Load the model 

# Load YOLOv8 Small model for Detection
# model = YOLO('yolov8s.pt')  

# Load YOLOv8 Nano model for Detection
model = YOLO('yolov8n.pt')  

# Load YOLOv8 Medium model for Detection
# model = YOLO('yolov8m.pt')  


# Train the model on CPUpip uninstall torchvision        
# results_cpu = model.train(data=yamlPath, epochs=50, imgsz=720, device='cpu')
# print("Training complete on CPU.")

# Train the model on a single GPU (GPU 0)
results_gpu_1 = model.train(data=yamlPath, epochs=500, imgsz=640, device=0, workers=0)
print("Training complete on GPU 0.")

# Train the model on multiple GPUs (GPU 0 and GPU 1)
# results_gpu_2 = model.train(data=yamlPath, epochs=500, imgsz=640, device=[0, 1])
# print("Training complete on GPUs 0 and 1.")

# Evaluate the model's performance on the validation set
results = model.val()
print("Evaluation complete.")
