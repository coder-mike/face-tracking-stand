
from adafruit_servokit import ServoKit

print("[INFO] initializing servo...")
kit = ServoKit(channels=16)

def servo_control(face_locations):
    # Move the servo to the average x-position of the faces
    if face_locations:
        avg_x = sum([(left + right) / 2 for (top, right, bottom, left) in face_locations]) / len(face_locations)
        # Map x-position (0-1920) to servo angle (0-180)
        servo_angle = avg_x * 180
        # Move servo to the angle
        kit.servo[0].angle = servo_angle