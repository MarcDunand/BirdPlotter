from ultralytics import YOLO
import cv2
import numpy as np
from pyaxidraw import axidraw
import math
import torch

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

# Define drawing functions
def circle(x, y, r):
    x = 279.4 - x
    y = 215.9 - y
    axi.moveto(x, y + r)
    steps = 30
    for i in range(steps + 1):
        theta = (i * 2 * math.pi) / steps
        axi.lineto(x + r * math.sin(theta), y + r * math.cos(theta))
    axi.penup()

def rect(x, y, w, h):
    x = 275 - x
    y = 200 - y
    axi.moveto(x, y)
    axi.lineto(x + w, y)
    axi.lineto(x + w, y + h)
    axi.lineto(x, y + h)
    axi.lineto(x, y)
    axi.penup()

# Startup the AxiDraw
axi = axidraw.AxiDraw()
axi.plot_setup()
axi.interactive()
if not axi.connect():
    print("Not connected")
    quit()
print("Connected to AxiDraw!")
axi.options.units = 2
axi.update()

# Load the YOLO model
model = YOLO('yolov8m.pt')
model.to(device)

# Open the webcam video feed
cap = cv2.VideoCapture(0)  # Use '0' for the default webcam

frame_count = 0

# Main loop
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam. Exiting...")
        break

    # Resize and process frame
    frame = cv2.resize(frame, (4656, 3496))  # Adjust resolution as needed

    # Run YOLO detection
    print("Beginning to run YOLO")
    results = model(frame, conf=0.3, imgsz=(4672, 3520))
    print("YOLO detection completed")

    foundBirds = []
    for detection in results[0].boxes:
        x_min, y_min, x_max, y_max = detection.xyxy[0]
        confidence = detection.conf[0]
        cls = int(detection.cls[0])

        if model.names[cls] in ["bird", "airplane", "kite"]:
            cv2.rectangle(frame, (int(x_min) - 10, int(y_min) - 10),
                          (int(x_max) + 10, int(y_max) + 10), (0, 0, 255), 2)
            position = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
            cv2.putText(frame, f"Bird at {position}", (int(x_min), int(y_min) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

            # Save detected frame and prepare for AxiDraw
            cv2.imwrite(f"./bird_frames/bird_frame_{frame_count}.jpg", frame)
            frame_count += 1
            ratio = 0.065
            if x_min > 200 and y_min > 200 and x_max < 4400 and y_max < 3300:
                foundBirds.append([float(x_min) * ratio, float(y_min) * ratio,
                               float(x_max - x_min) * ratio, float(y_max - y_min) * ratio])

    # Display the frame
    display_frame = cv2.resize(frame, (1280, 960))
    cv2.imshow("Bird Detection", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # If birds are found, draw bounding boxes with AxiDraw synchronously
    if foundBirds:
        print("Drawing bounding boxes with AxiDraw")
        for bird in foundBirds:
            rect(bird[0], bird[1], bird[2], bird[3])
        print("AxiDraw completed drawing the bounding boxes.")

# Cleanup
cap.release()
cv2.destroyAllWindows()

axi.moveto(0, 0)
axi.disconnect()
axi.options.mode = "align"
axi.plot_run()
