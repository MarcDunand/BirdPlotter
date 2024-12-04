from ultralytics import YOLO
import cv2
import numpy as np
from pyaxidraw import axidraw
import math
import torch
import os  # Import os module for file handling

# Configuration: Set to True to use the webcam, False to use frames from readFrames
use_webcam = False  # Change to True to use the webcam

# Path to the frames directory (used only if use_webcam is False)
frames_directory = './readFrames/'  # Update this path to your frames directory

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

# Define drawing functions
def arc(x, y, rx, ry, thetaS, thetaE, raisePen):
    if(thetaE < thetaS):
        print("Make starting angle less than ending angle")
        return

    if raisePen:
        axi.penup()

    axi.goto(x + rx * math.sin(thetaS), y + ry * math.cos(thetaS))
    dTheta = thetaE - thetaS
    steps = int((dTheta / (2 * math.pi)) * 23)
    for i in range(steps + 1):
        theta = thetaS + (i * dTheta) / steps
        axi.lineto(x + rx * math.sin(theta), y + ry * math.cos(theta))
    if raisePen:
        axi.penup()

def rect(x, y, w, h):
    axi.moveto(x, y)
    axi.lineto(x + w, y)
    axi.lineto(x + w, y + h)
    axi.lineto(x, y + h)
    axi.lineto(x, y)
    axi.penup()

def drawBird(x, y, w, h):
    cutoff = 1
    arc(x + np.cos((np.pi / 2) - cutoff) * (w / 4), y - h / 4, w / 4, h, cutoff * -1, cutoff, False)
    arc(x + np.cos((np.pi / 2) - cutoff) * (3 * w / 4), y - h / 4, w / 4, h, cutoff * -1, cutoff, False)
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
model = YOLO('yolov8x.pt')
model.to(device)

frame_count = 0

# Prepare the list of frames if not using webcam
if not use_webcam:
    print("Using frames from readFrames directory.")
    # Get a sorted list of image file paths from the frames directory
    frame_files = sorted([os.path.join(frames_directory, f) for f in os.listdir(frames_directory)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
    total_frames = len(frame_files)
    print(f"Total frames to process: {total_frames}")
else:
    # Open the webcam video feed
    cap = cv2.VideoCapture(1)  # Use '0' for the default webcam
    print("Using webcam as video source.")

# Main loop
while True:
    if use_webcam:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam. Exiting...")
            break
    else:
        # Check if all frames have been processed
        if frame_count >= total_frames:
            print("All frames have been processed. Exiting...")
            break

        # Read the frame from the file
        frame_path = frame_files[frame_count]
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Failed to read frame: {frame_path}. Skipping...")
            frame_count += 1
            continue

    # Resize and process frame
    frame = cv2.resize(frame, (4656, 3496))  # Adjust resolution as needed

    # Run YOLO detection
    print(f"Processing frame {frame_count + 1}/{total_frames if not use_webcam else ''}")
    print("Beginning to run YOLO")
    results = model(frame, conf=0.4, imgsz=(4672, 3520))
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
            ratio = 0.065
            if x_min > 200 and y_min > 200 and x_max < 4400 and y_max < 3300 and (x_max - x_min) * (y_max - y_min) < 40000:
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
            bird[0] = (275 - bird[0])
            bird[1] = (200 - bird[1])
            # rect(bird[0], bird[1], bird[2], bird[3])  # Uncomment for debugging
            drawBird(bird[0], bird[1], bird[2], bird[3])
        print("AxiDraw completed drawing the bounding boxes.")

    # Increment frame count
    frame_count += 1

# Cleanup
if use_webcam:
    cap.release()
cv2.destroyAllWindows()

axi.moveto(0, 0)
axi.disconnect()
axi.options.mode = "align"
axi.plot_run()
