from ultralytics import YOLO
import os
import time
from os import listdir

#for training
#model = YOLO("yolov8n.yaml")
#results = model.train(data="config.yaml", epochs=50)

#result loop
model = YOLO("runs/detect/train3/weights/best.pt")
# get the path/directory
folder_dir = "/home/john/VR_IT/Source/pythonProject/yolo/testImages/"
output = []
for image in sorted(os.listdir(folder_dir)):
    singleInfo = []
    # check if the image ends with png
    if image.endswith(".png"):
        print(image)
        #results = model.predict(folder_dir+image, conf=0.2, max_det=30)
        results = model.track(folder_dir+image, conf=0.2, max_det=30, show=True, persist=True, tracker="bytetrack.yaml")
        for box in results[0].boxes:
            boxInfo = []
            label = results[0].names[box.cls[0].item()]
            boxInfo.append(label)
            x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            boxInfo.append([x2-x1, y2-y1])
            boxInfo.append([x_center, y_center])
            singleInfo.append(boxInfo)
    output.append(singleInfo)
input("Press Enter to continue...")
print(output)
