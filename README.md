Yolov8 Model Training with custom dataset

1. First Run VGG2Yolo.py for changing VGG annotations to Yolo Supported Format. 
2. Second Run imgaug_augmentor.py or album_augmentor.py for implementing augmentation
3. Third Run prepare_Data.py for spliting data into val and train and transforming to Yolo Format
4. Fourth Run trainYoloV8Detection.py for training model
5. Fifth Run track_video.py for testin