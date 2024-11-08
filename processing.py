import time
import cv2
import face_recognition
import pickle
import numpy as np

cv_scaler = 4 # this has to be a whole number

# Load pre-trained face encodings
print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

def process_frame(frame):
    """
    Process a single frame for face recognition and servo control.

    Args:
        frame (ndarray): The image frame to process.
        cv_scaler (int): Scaling factor for resizing the frame.

    Returns:
        tuple: A tuple containing:
            - frame (ndarray): The processed frame (unchanged in this function).
            - face_locations (list): List of face locations found in the frame, normalized to (0 to 1.0) range.
            - face_names (list): List of names corresponding to detected faces.
            - timings (dict): Dictionary of timing measurements for processing steps.
    """
    # Initialize variables
    face_locations = []
    face_encodings = []
    face_names = []
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

    # Normalize face locations to 0..1.0 range
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



    return frame, face_locations, face_names, timings