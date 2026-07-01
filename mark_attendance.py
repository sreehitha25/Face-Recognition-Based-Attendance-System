"""
mark_attendance.py
-------------------
Opens the webcam, recognizes faces using the trained LBPH model, and
logs attendance into the database (once per person per day).

Run standalone:
    python mark_attendance.py
Press 'q' to stop the webcam session.
"""

import cv2
import os
import database as db

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
TRAINER_PATH = os.path.join("trainer", "trainer.yml")

# Lower distance = more confident match. Tune this if you get false matches.
CONFIDENCE_THRESHOLD = 70


def run_attendance_session(status_callback=None):
    """
    Runs a live webcam session marking attendance for recognized faces.
    status_callback(message): optional hook to print/log status messages
    (e.g., for displaying in a Streamlit app).
    """
    if not os.path.exists(TRAINER_PATH):
        raise FileNotFoundError("No trained model found. Run train_model.py first.")

    db.init_db()
    people = dict(db.get_all_people())  # {id: name}

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_PATH)
    face_detector = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    cam = cv2.VideoCapture(0)
    marked_this_session = set()

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
            person_id, distance = recognizer.predict(face_img)

            if distance < CONFIDENCE_THRESHOLD and person_id in people:
                name = people[person_id]
                label = f"{name} ({round(100 - distance, 1)}%)"
                color = (0, 255, 0)

                if person_id not in marked_this_session:
                    newly_marked = db.mark_attendance(person_id, name)
                    marked_this_session.add(person_id)
                    if newly_marked and status_callback:
                        status_callback(f"Attendance marked: {name}")
            else:
                label = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Attendance - Press Q to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
    return marked_this_session


if __name__ == "__main__":
    marked = run_attendance_session(status_callback=print)
    print(f"Session ended. {len(marked)} unique people marked present.")
