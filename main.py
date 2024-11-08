import cv2
from picamera2 import Picamera2
import time
import os
from processing import process_frame  # Importing the extracted function

print("[INFO] initializing camera...")
# Initialize the camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (1920, 1080)}))
picam2.start()

frame_count = 0
start_time = time.time()
fps = 0

# Detect if running over SSH
is_ssh = 'SSH_CONNECTION' in os.environ or 'SSH_CLIENT' in os.environ

def draw_results(frame, face_locations, face_names):
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled
        top *= frame_height
        right *= frame_width
        bottom *= frame_height
        left *= frame_width

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 3)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left -3, top - 35), (right+3, top), (244, 42, 3), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)

    return frame

def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps

print("[INFO] starting main loop...")
try:
    while True:
        # Start timing
        cycle_start_time = time.time()

        # Capture a frame from camera
        frame = picam2.capture_array()

        # Process the frame with the function
        processed_frame, face_locations, face_names, timings = process_frame(
            frame
        )

        # Calculate and update FPS
        fps = calculate_fps()

        # Output timings for processing steps
        print(f"fps: {fps:.1f}, "
            # f"Resize: {timings['resize']:.2f} ms, "
            # f"Color Conversion: {timings['color_conversion']:.2f} ms, "
            f"Face Location: {timings['face_location']:.2f} ms, "
            f"Face Encoding: {timings['face_encoding']:.2f} ms, "
            f"Face Matching: {timings['face_matching']:.2f} ms"
        )

        # Display everything over the video feed.
        if not is_ssh:
            # Get the text and boxes to be drawn based on the processed frame
            display_frame = draw_results(processed_frame, face_locations, face_names)

            # Resize the display_frame to 360p
            display_frame = cv2.resize(display_frame, (640, 360))

            # Attach FPS counter to the text and boxes
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (display_frame.shape[1] - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Video', display_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) == ord("q"):
                break

        # time.sleep(0.01)  # Small delay to prevent high CPU usage
except KeyboardInterrupt:
    # Allow script to be stopped with Ctrl+C when running over SSH
    pass

# By breaking the loop we run this code here which closes everything
if not is_ssh:
    cv2.destroyAllWindows()
picam2.stop()