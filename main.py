import face_recognition
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import pickle
from adafruit_servokit import ServoKit

# Load pre-trained face encodings
print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

print("[INFO] initializing camera...")
# Initialize the camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (1920, 1080)}))
picam2.start()

# Initialize our variables
cv_scaler = 4 # this has to be a whole number

face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0

print("[INFO] initializing servo...")
kit = ServoKit(channels=16)

def process_frame(frame):
    global face_locations, face_encodings, face_names

    timings = {}

    # Resize the frame
    resize_start = time.time()
    resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
    timings['resize'] = (time.time() - resize_start) * 1000  # milliseconds

    # Color conversion
    color_conversion_start = time.time()
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    timings['color_conversion'] = (time.time() - color_conversion_start) * 1000  # milliseconds

    # Face location
    face_location_start = time.time()
    face_locations = face_recognition.face_locations(rgb_resized_frame)
    timings['face_location'] = (time.time() - face_location_start) * 1000  # milliseconds

    # Face encoding
    face_encoding_start = time.time()
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations, model='large')
    timings['face_encoding'] = (time.time() - face_encoding_start) * 1000  # milliseconds

    # Face matching
    face_matching_start = time.time()
    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)
    timings['face_matching'] = (time.time() - face_matching_start) * 1000  # milliseconds

    # Output timings for processing steps
    print(f"Resize: {timings['resize']:.2f} ms, Color Conversion: {timings['color_conversion']:.2f} ms, "
          f"Face Location: {timings['face_location']:.2f} ms, Face Encoding: {timings['face_encoding']:.2f} ms, "
          f"Face Matching: {timings['face_matching']:.2f} ms")

    # After identifying face_locations
    # Calculate average x-position of faces
    if face_locations:
        avg_x = sum([(left + right) / 2 for (top, right, bottom, left) in face_locations]) / len(face_locations)
        # Map x-position (0-1920) to servo angle (0-180)
        servo_angle = (avg_x / (1920 / cv_scaler)) * 180
        # Move servo to the angle
        kit.servo[0].angle = servo_angle

    return frame

def draw_results(frame):
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler

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
while True:
    # Start timing
    cycle_start_time = time.time()

    # Capture a frame from camera
    capture_start = time.time()
    frame = picam2.capture_array()
    capture_time = (time.time() - capture_start) * 1000  # milliseconds

    # Process the frame with the function
    process_start = time.time()
    processed_frame = process_frame(frame)
    process_time = (time.time() - process_start) * 1000  # milliseconds

    # Get the text and boxes to be drawn based on the processed frame
    draw_start = time.time()
    display_frame = draw_results(processed_frame)
    draw_time = (time.time() - draw_start) * 1000  # milliseconds

    # Calculate and update FPS
    fps = calculate_fps()

    # Attach FPS counter to the text and boxes
    cv2.putText(display_frame, f"FPS: {fps:.1f}", (display_frame.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Resize the display_frame to 360p
    resize_display_start = time.time()
    display_frame = cv2.resize(display_frame, (640, 360))
    resize_display_time = (time.time() - resize_display_start) * 1000  # milliseconds

    # Display everything over the video feed.
    imshow_start = time.time()
    cv2.imshow('Video', display_frame)
    imshow_time = (time.time() - imshow_start) * 1000  # milliseconds

    # Total cycle time
    cycle_time = (time.time() - cycle_start_time) * 1000  # milliseconds

    # Output timings to console
    print(f"Capture: {capture_time:.2f} ms, Process: {process_time:.2f} ms, Draw: {draw_time:.2f} ms, "
          f"Resize Display: {resize_display_time:.2f} ms, Imshow: {imshow_time:.2f} ms, "
          f"Total Cycle: {cycle_time:.2f} ms")

    # Break the loop and stop the script if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

    # Add a delay to slow down the frame rate to 1 FPS
    # time.sleep(1)

# By breaking the loop we run this code here which closes everything
cv2.destroyAllWindows()
picam2.stop()