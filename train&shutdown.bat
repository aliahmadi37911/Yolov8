@echo off
REM Change directory to where your training script is located
cd D:\Work\WatchTower\Yolo-v8\Yolov8\

REM Run your training command
python %1

REM Schedule shutdown after training is complete
shutdown /s /t 60

REM Cancle Schedule 
REM shutdown /a
