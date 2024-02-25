from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(data='/Users/vzhan/Documents/projects/ballStats/basketball-1/data.yaml', epochs=3)