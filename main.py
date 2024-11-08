import cv2
from picamera2 import Picamera2
import time
import os
from find_faces import full_scan, delta_scan
from servo_control import servo_control

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
        # (and convert them from float to int)
        top = int(top * frame_height)
        right = int(right * frame_width)
        bottom = int(bottom * frame_height)
        left = int(left * frame_width)

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

# Initialize previous face data
previous_face_locations = []
previous_face_names = []
last_full_scan_time = time.time()

print("[INFO] starting main loop...")
try:
    while True:
        # Start timing
        cycle_start_time = time.time()

        # Capture a frame from camera
        frame = picam2.capture_array()

        # Check if it's time for a full scan
        time_since_last_full_scan = time.time() - last_full_scan_time
        do_full_scan = False
        if previous_face_locations is None:
            print("First run, performing full scan")
            do_full_scan = True
        elif len(previous_face_locations) == 0:
            print("No faces detected in previous frame, performing full scan")
            do_full_scan = True
        elif time_since_last_full_scan > 10:
            print("Time since last full scan > 10 seconds, performing full scan")
            do_full_scan = True

        if not do_full_scan:
            result = delta_scan(frame, previous_face_locations, previous_face_names)
            if result:
                face_locations, face_names, timings = result
            else:
                do_full_scan = True

        if do_full_scan:
            face_locations, face_names, timings = full_scan(frame)
            last_full_scan_time = time.time()

        # Update previous face data
        previous_face_locations = face_locations
        previous_face_names = face_names

        servo_control(face_locations)

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
            display_frame = draw_results(frame, face_locations, face_names)

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