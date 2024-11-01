from ultralytics import YOLO
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Load the YOLO model
model = YOLO('yolov8n.pt')  # Choose your model version

# Open the prerecorded video file
cap = cv2.VideoCapture('birdClip2_trimmed.mkv')

frame_count = 0  # Track the frame number for unique filenames
prev_frame = None  # To store the previous frame
similarity_threshold = 0.99  # Threshold for skipping similar frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize to full resolution for YOLO processing
    frame = cv2.resize(frame, (4656, 3496))

    # Convert the current frame to grayscale for easier comparison
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Check similarity with the previous frame using SSIM
    if prev_frame is not None:
        score, _ = ssim(prev_frame, gray_frame, full=True)
        # Skip this frame if it is very similar to the previous one
        if score > similarity_threshold:
            continue  # Skip processing if frames are very similar

    # Update the previous frame after confirming it’s unique or sufficiently different
    prev_frame = gray_frame.copy()

    # Run YOLO detection
    results = model(frame, conf=0.3, imgsz=(4672, 3520))

    # Process detections
    for detection in results[0].boxes:
        x_min, y_min, x_max, y_max = detection.xyxy[0]
        confidence = detection.conf[0]
        cls = int(detection.cls[0])

        # Check if the detected object is a bird, airplane, or kite
        if model.names[cls] in ["bird", "airplane", "kite"]:
            # Draw bounding box and display bird's position
            cv2.rectangle(frame, (int(x_min) - 10, int(y_min) - 10), (int(x_max) + 10, int(y_max) + 10), (0, 0, 255), 2)
            position = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
            cv2.putText(frame, f"Bird at {position}", (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

            # Save the frame with the bird detected
            cv2.imwrite(f"./bird_frames/bird_frame_{frame_count}.jpg", frame)
            frame_count += 1

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
