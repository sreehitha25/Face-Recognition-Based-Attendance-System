"""
register_faces.py
------------------
Captures face images from the webcam for a new person and saves them
under dataset/<person_id>_<name>/ for later training.

Run standalone:
    python register_faces.py "Sreehitha"

Or import register_person() and call it from the Streamlit app.
"""

import cv2
import os
import sys
import database as db

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
SAMPLES_TO_CAPTURE = 30  # number of face images to collect per person
DATASET_DIR = "dataset"


def register_person(name: str, samples: int = SAMPLES_TO_CAPTURE, progress_callback=None):
    """
    Opens the webcam, detects faces, and saves `samples` cropped grayscale
    face images for the given person. Returns the assigned person_id.

    progress_callback (optional): a function called with (count, samples)
    after each captured image — useful for updating a Streamlit progress bar.
    """
    db.init_db()
    person_id = db.get_next_person_id()

    person_dir = os.path.join(DATASET_DIR, f"{person_id}_{name}")
    os.makedirs(person_dir, exist_ok=True)

    face_detector = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    cam = cv2.VideoCapture(0)

    count = 0
    while count < samples:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (200, 200))
            cv2.imwrite(os.path.join(person_dir, f"{count}.jpg"), face_img)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured {count}/{samples}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if progress_callback:
                progress_callback(count, samples)
            break  # only capture one face per frame

        cv2.imshow("Registering Face - Press Q to cancel", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if count >= samples:
            break

    cam.release()
    cv2.destroyAllWindows()

    if count > 0:
        db.add_person(person_id, name)

    return person_id, count


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python register_faces.py <person_name>")
        sys.exit(1)

    person_name = sys.argv[1]
    pid, captured = register_person(person_name)
    print(f"Registered '{person_name}' with ID {pid}. Captured {captured} images.")
    print("Now run train_model.py to update the recognizer.")
