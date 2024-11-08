import time
import cv2
import face_recognition

def process_frame(frame, known_face_encodings, known_face_names, cv_scaler, kit):
    """
    Process a single frame for face recognition and servo control.

    Args:
        frame (ndarray): The image frame to process.
        known_face_encodings (list): List of known face encodings.
        known_face_names (list): List of names corresponding to the known face encodings.
        cv_scaler (int): Scaling factor for resizing the frame.
        kit (ServoKit): Instance of the servo kit to control the servo motor.

    Returns:
        tuple: A tuple containing:
            - frame (ndarray): The processed frame (unchanged in this function).
            - face_locations (list): List of face locations found in the frame.
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

    # After identifying face_locations
    # Calculate average x-position of faces
    if face_locations:
        avg_x = sum([(left + right) / 2 for (top, right, bottom, left) in face_locations]) / len(face_locations)
        # Map x-position (0-1920) to servo angle (0-180)
        servo_angle = (avg_x / (1920 / cv_scaler)) * 180
        # Move servo to the angle
        kit.servo[0].angle = servo_angle

    return frame, face_locations, face_names, timings