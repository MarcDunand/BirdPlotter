import cv2
import os

# Path to the video file
video_path = 'birdClip2_trimmed.MKV'  # Replace with your video file name

# Output directory for saved frames
output_dir = 'readFrames'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

print(f"Video FPS: {fps}")
print(f"Total frames: {total_frames}")
print(f"Video duration (seconds): {duration}")

# Option A: Extract specific frame numbers
frame_numbers = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]  # Replace with your desired frame numbers

# Option B: Extract frames at specific intervals (e.g., every 5 seconds)
# interval = 5  # Time interval in seconds
# frame_numbers = [int(fps * t) for t in range(0, int(duration), interval)]

for frame_num in frame_numbers:
    if frame_num >= total_frames:
        print(f"Frame {frame_num} exceeds total number of frames. Skipping.")
        continue

    # Set the video to the specified frame number
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    # Read the frame
    ret, frame = cap.read()

    if ret:
        # Construct the filename
        frame_filename = os.path.join(output_dir, f'frame_{frame_num}.jpg')

        # Save the frame as an image
        cv2.imwrite(frame_filename, frame)
        print(f"Saved frame {frame_num} to {frame_filename}")
    else:
        print(f"Warning: Could not read frame {frame_num}")

# Release the video capture object
cap.release()
