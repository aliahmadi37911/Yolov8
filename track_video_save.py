from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime

# Get the current time
current_time = datetime.now().strftime("%Y%m%d_%H%M")

import torch
torch.cuda.set_device(0)

VIDEO_PATH = "./videos/2.mov"
YOLO_MODEL_PATH = "best.pt"

def track_video(model_path, video_path):
    # load the model
    model = YOLO(model_path)
    # open the video file
    cap = cv2.VideoCapture(video_path)
    track_history = defaultdict(lambda: [])
    # get the video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # define the codec and create VideoWriter object
    output_path = f"./videos/output_tracked_video_{current_time}.mp4"  # Output video file path
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
    )
    # loop through the video frames
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # CPU
            # results = model.track(frame, persist=True)
            # GPU 
            results = model.track(frame, persist=True, device=0)
            boxes = results[0].boxes.xywh.cpu()
            track_ids = (
                results[0].boxes.id.int().cpu().tolist()
                if results[0].boxes.id is not None
                else None
            )
            annotated_frame = results[0].plot()
            # plot the tracks
            if track_ids:
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 30 tracks for 30 frames
                        track.pop(0)
                    # draw the tracking lines
                    points = np.array(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(
                        annotated_frame,
                        [points],
                        isClosed=False,
                        color=(230, 230, 230),
                        thickness=2,
                    )
            # write the annotated frame
            out.write(annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    # release the video capture object and close the display window
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return output_path

if __name__ == "__main__":
    track_video(YOLO_MODEL_PATH, VIDEO_PATH)