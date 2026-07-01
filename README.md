# Face Recognition Attendance System

A real-time attendance system that recognizes registered faces via webcam
and logs attendance automatically, with an analytics dashboard built in
Streamlit.

**Tech Stack:** Python, OpenCV (Haar Cascade + LBPH Face Recognizer), SQLite,
Streamlit, Pandas

---

## How it works

1. **Register** — Capture ~30 face images per person via webcam (Haar
   Cascade detects the face region).
2. **Train** — An LBPH (Local Binary Patterns Histograms) recognizer is
   trained on all registered faces and saved to `trainer/trainer.yml`.
3. **Mark Attendance** — A live webcam session recognizes faces in
   real time and logs each person's attendance (once per day) to a SQLite
   database.
4. **Dashboard** — View attendance records, per-person counts, daily
   trends, and export to CSV.

---

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

This will open a browser tab with the dashboard UI. Note: the webcam
feed for Register/Attendance opens as a **separate native OpenCV window**
(not inside the browser) — this is expected behavior for OpenCV's
`cv2.imshow`.

### Command-line alternative
You can also run each step directly without the UI:
```bash
python register_faces.py "Your Name"
python train_model.py
python mark_attendance.py
```

---

## Project structure

```
attendance_system/
├── app.py                # Streamlit UI (register/train/attendance/dashboard)
├── database.py            # SQLite schema + queries
├── register_faces.py      # Webcam face capture for new people
├── train_model.py         # Trains LBPH recognizer on captured faces
├── mark_attendance.py     # Live recognition + attendance logging
├── requirements.txt
├── dataset/                # Captured face images (created at runtime)
├── trainer/trainer.yml     # Trained model (created at runtime)
└── attendance.db           # SQLite database (created at runtime)
```

---

## Possible extensions (good for interview talking points)

- Swap Haar Cascade for a DNN-based face detector (e.g., OpenCV's
  `res10_300x300_ssd`) for better accuracy in low light / angles.
- Add attendance % thresholds and flag "at-risk" students/employees
  (ties into classification/prediction skills).
- Deploy the dashboard (Streamlit Cloud) while keeping capture local,
  since webcam access doesn't work in a cloud browser session.
- Add email/SMS alerts for absentees using a scheduled job.

---

## Suggested resume bullet points

**Face Recognition-Based Attendance System**
- Built a real-time attendance system using OpenCV (Haar Cascade face
  detection + LBPH recognizer) to automatically identify and log
  registered individuals via webcam.
- Designed a SQLite-backed data pipeline to store face embeddings'
  metadata and attendance logs, preventing duplicate same-day entries.
- Developed an interactive Streamlit dashboard to visualize attendance
  trends, per-person attendance counts, and exportable reports.
- **Tech Stack:** Python, OpenCV, SQLite, Streamlit, Pandas.

*(Once you run it, replace with your actual numbers — e.g., "recognized
faces with ~95% accuracy across X registered users" — using your real
test results.)*
