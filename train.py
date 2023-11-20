import ultralytics
from ultralyticsplus import YOLO, render_result
import torch
from roboflow import Roboflow

if __name__ == '__main__':
    model = YOLO('./yolov8n.pt')
    # model = YOLO('./runs/detect/train/weights/best.pt')
    model.train(data='data.yaml', epochs=500, imgsz=640, device=0, verbose=True, patience=100)
    # rf = Roboflow(api_key="")
    # project = rf.workspace().project("input-fields-detection")
    # model = project.version(3).model

