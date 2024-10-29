from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLO model
model = YOLO('yolov8m.pt')  # Choose your model version

# Open the prerecorded video file
cap = cv2.VideoCapture('birdClip.mkv')

# Initialize background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize to full resolution for YOLO processing
    frame = cv2.resize(frame, (4656, 3496))

    # Apply background subtraction to detect motion
    fg_mask = bg_subtractor.apply(frame)

    # Calculate the number of non-zero pixels in the mask (indicating motion)
    motion_pixels = np.count_nonzero(fg_mask)

    # Set a threshold for detecting significant motion
    if motion_pixels > 5000:  # Adjust threshold based on your scene
        # Run YOLO on frames with significant motion
        results = model(frame, conf=0.15, imgsz=(4672, 3520))
        print('Movement Detected:')

    if motion_pixels < 5000:  # Adjust threshold based on your scene
        # Run YOLO on frames with significant motion
        results = model(frame, conf=0.25, imgsz=(4672, 3520))

    # Process detections
    for detection in results[0].boxes:
        x_min, y_min, x_max, y_max = detection.xyxy[0]
        confidence = detection.conf[0]
        cls = int(detection.cls[0])

        # Check if the detected object is a bird
        if model.names[cls] == "bird":
            # Draw bounding box and display bird's position
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
            position = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
            cv2.putText(frame, f"Bird at {position}", (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        else:
            # Draw bounding box and display detected object's label
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
            position = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
            cv2.putText(frame, model.names[cls], (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

    # Resize the frame for display purposes
    display_frame = cv2.resize(frame, (1280, 960))  # Adjust to a suitable display size

    # Display the resized frame
    cv2.imshow("Bird Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
