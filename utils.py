import cv2
import numpy as np

def extract_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face = image[y:y+h, x:x+w]
    face = cv2.resize(face, (224, 224))
    return face


def predict(image):
    face = extract_face(image)

    if face is None:
        return None, None

    # Convert to grayscale
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # Calculate image sharpness (deterministic)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Convert to percentage (0–100)
    fake_percentage = min(100, max(0, variance / 10))

    return face, round(fake_percentage, 2)
