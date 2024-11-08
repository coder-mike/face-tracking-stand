import time
import cv2
import face_recognition
import pickle
import numpy as np
from mtcnn import MTCNN

detector = MTCNN()

# Load pre-trained face encodings
print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

def full_scan(frame):
    """
    Process a single frame for face recognition and servo control.

    Args:
        frame (ndarray): The image frame to process.

    Returns:
        tuple: A tuple containing:
            - face_locations (list): List of face locations found in the frame, normalized to (0 to 1.0) range.
            - face_names (list): List of names corresponding to detected faces.
            - timings (dict): Dictionary of timing measurements for processing steps.
    """
    # Initialize variables
    face_locations = []
    face_encodings = []
    face_names = []
    timings = {}

    # Downscale the frame to speed up face detection
    scale = 0.5
    resized_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

    # Color conversion
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Face location using MTCNN
    face_location_start = time.time()
    detections = detector.detect_faces(rgb_resized_frame)
    timings['face_location'] = (time.time() - face_location_start) * 1000  # milliseconds

    # Extract face locations
    for detection in detections:
        x, y, width, height = detection['box']
        top, right, bottom, left = y, x + width, y + height, x
        face_locations.append((top, right, bottom, left))

    # Face encoding
    face_encoding_start = time.time()
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations, model='large')
    timings['face_encoding'] = (time.time() - face_encoding_start) * 1000  # milliseconds

    # Face matching
    face_matching_start = time.time()
    face_names = []
    live = []
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
        live.append(True)
    timings['face_matching'] = (time.time() - face_matching_start) * 1000  # milliseconds

    # Normalize face locations to 0..1.0 range so that servo control is independent of camera resolution
    resized_frame_width = resized_frame.shape[1]
    resized_frame_height = resized_frame.shape[0]
    face_locations = [
        (
            top / resized_frame_height,
            right / resized_frame_width,
            bottom / resized_frame_height,
            left / resized_frame_width
        )
        for (top, right, bottom, left) in face_locations
    ]

    return face_locations, face_names, live, timings


def delta_scan(frame, previous_face_locations, previous_face_names):
    face_locations = []
    face_names = []
    live = []
    # Initialize timings with zero values for consistency
    timings = {
        'face_location': 0.0,
        'face_encoding': 0.0,
        'face_matching': 0.0
    }

    frame_height, frame_width, _ = frame.shape

    for (top_norm, right_norm, bottom_norm, left_norm), name in zip(previous_face_locations, previous_face_names):
        # Scale normalized coordinates to pixel values and round them
        top = int(top_norm * frame_height)
        right = int(right_norm * frame_width)
        bottom = int(bottom_norm * frame_height)
        left = int(left_norm * frame_width)

        # Calculate margins
        margin_v = int((bottom - top) * 0.5)
        margin_h = int((right - left) * 0.5)

        # Define new region with margins and ensure coordinates are within frame bounds
        top_new = max(0, top - margin_v)
        bottom_new = min(frame_height, bottom + margin_v)
        left_new = max(0, left - margin_h)
        right_new = min(frame_width, right + margin_h)

        # Crop the region
        cropped_frame = frame[top_new:bottom_new, left_new:right_new]

        # Resize to speed up face detection
        width_new = 50
        scale = width_new / (right_new - left_new)
        height_new = int((bottom_new - top_new) * scale)
        resized_cropped_frame = cv2.resize(cropped_frame, (width_new, height_new))

        # Face detection on cropped image using MTCNN
        face_location_start = time.time()
        rgb_cropped_frame = cv2.cvtColor(resized_cropped_frame, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(rgb_cropped_frame)
        timings['face_location'] += (time.time() - face_location_start) * 1000  # Accumulate time in milliseconds

        if len(detections) != 1:
            # Fall back to previous face location
            live.append(False)
            face_locations.append((top_norm, right_norm, bottom_norm, left_norm))
            face_names.append(name)
            continue

        # Extract face location from detection
        detection = detections[0]
        x, y, width, height = detection['box']
        top_cropped, right_cropped, bottom_cropped, left_cropped = y, x + width, y + height, x

        # Scale back up to the size of the cropped image
        top_rescaled = top_new + int(top_cropped / height_new * (bottom_new - top_new))
        bottom_rescaled = top_new + int(bottom_cropped / height_new * (bottom_new - top_new))
        left_rescaled = left_new + int(left_cropped / width_new * (right_new - left_new))
        right_rescaled = left_new + int(right_cropped / width_new * (right_new - left_new))

        # Append the normalized face location
        face_locations.append((
            top_rescaled / frame_height,
            right_rescaled / frame_width,
            bottom_rescaled / frame_height,
            left_rescaled / frame_width
        ))
        face_names.append(name)
        live.append(True)

    return face_locations, face_names, live, timings