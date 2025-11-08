import cv2
import torch
import numpy as np
from torchvision import models, transforms
from datetime import datetime
import time

# MobilenetSSD AI Model (detects people and the classes underneath)
prototxt = "deploy.prototxt"
model_caffe = "mobilenet_iter_73000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model_caffe)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

map_img = cv2.imread("map_img.png")  

map_w, map_h = 150, 100
map_img = cv2.resize(map_img, (map_w, map_h))  

altitude = 100  
aqi = 42       
weather = "Sunny"

# Deeplab AI Model (detects rivers and lakes and big bodies of water)
deeplab = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True).eval()
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
WATER_CLASS = 10

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

last_segmentation_time = 0
mask_colored_resized = None

map_h, map_w = 100, 150
fake_map = np.full((map_h, map_w, 3), 200, dtype=np.uint8) 
cv2.rectangle(fake_map, (0,0), (map_w-1, map_h-1), (255,255,255), 2) 
cv2.line(fake_map, (10, 20), (140, 20), (50,50,50), 2)
cv2.line(fake_map, (10, 50), (140, 50), (50,50,50), 2)
cv2.line(fake_map, (10, 80), (140, 80), (50,50,50), 2)

def draw_arrow(img, center, size=15, color=(0,255,0), angle=0):
    pts = np.array([
        [0, -size],     
        [-size//2, size], 
        [size//2, size]
    ], np.float32)

    theta = np.radians(angle)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts = pts @ R.T
    pts += np.array(center)
    pts = pts.astype(np.int32)
    cv2.fillPoly(img, [pts], color)

def draw_compass(img, center, radius=20, heading=0):
    cv2.circle(img, center, radius, (0, 0, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    offset = radius + 5
    cv2.putText(img, 'N', (center[0]-5, center[1]-offset), font, font_scale, (0,0,0), thickness)
    cv2.putText(img, 'S', (center[0]-5, center[1]+offset+5), font, font_scale, (0,0,0), thickness)
    cv2.putText(img, 'E', (center[0]+offset-5, center[1]+5), font, font_scale, (0,0,0), thickness)
    cv2.putText(img, 'W', (center[0]-offset-5, center[1]+5), font, font_scale, (0,0,0), thickness)

    theta = np.radians(heading)
    arrow_length = radius - 5
    tip = (int(center[0] + arrow_length * np.sin(theta)),
           int(center[1] - arrow_length * np.cos(theta)))
    cv2.arrowedLine(img, center, tip, (0, 0, 255), 2, tipLength=0.3)


while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]

    # Detects bottles (water) and people
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")

            if label == "person":
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
                cv2.putText(frame, f"Person: {confidence:.2f}",
                            (startX, startY - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            elif label == "bottle":
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (255, 0, 0), 2)
                cv2.putText(frame, f"Water: {confidence:.2f}",
                            (startX, startY - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    # Detects water, or at least tries to detect lakes and riveres every 2 seconds
    current_time = time.time()
    if current_time - last_segmentation_time > 2:
        input_tensor = preprocess(frame).unsqueeze(0)
        with torch.no_grad():
            output = deeplab(input_tensor)['out'][0]
        output_predictions = output.argmax(0).byte().cpu().numpy()

        mask = (output_predictions == WATER_CLASS).astype(np.uint8) * 255
        mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        mask_colored_resized = cv2.resize(mask_colored, (w, h), interpolation=cv2.INTER_NEAREST)

        last_segmentation_time = current_time

    if mask_colored_resized is not None:
        overlay = cv2.addWeighted(frame, 0.7, mask_colored_resized, 0.3, 0)
    else:
        overlay = frame.copy()

    # Real Time
    now = datetime.now().strftime("%H:%M:%S")
    cv2.putText(overlay, f"{now}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Fake GPS and Map
    map_x, map_y = w - map_w - 10, 10
    overlay[map_y:map_y+map_h, map_x:map_x+map_w] = map_img.copy()
    draw_arrow(overlay, (map_x + map_w//2, map_y + map_h//2),
               size=15, color=(0,255,0), angle=0) 

    compass_center = (map_x + 25, map_y + map_h - 25)
    draw_compass(overlay, compass_center, radius=20, heading=45)  

    bar_height = 30
    cv2.rectangle(overlay, (0, h - bar_height), (w, h), (50, 50, 50), -1)  

    status_text = f"Altitude: {altitude}m | Air Quality: {aqi} AQI | Current Weather: {weather}"
    cv2.putText(overlay, status_text, (10, h - 8),
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Fake Recording
    flash_interval = 0.5
    current_time = time.time()

    circle_radius = 8
    circle_x = w - 110  
    circle_y = h - 50 
    text_x = circle_x + circle_radius + 5
    text_y = circle_y + 5

    if int(current_time / flash_interval) % 2 == 0:
        cv2.circle(overlay, (circle_x, circle_y), circle_radius, (0, 0, 255), -1)

    cv2.putText(overlay, "Recording", (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)



    cv2.imshow("AR Hiking Monocle", overlay)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
