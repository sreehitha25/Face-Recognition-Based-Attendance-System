"""
app.py
------
Streamlit front-end for the Face Recognition Attendance System.

Tabs:
1. Register  - capture a new person's face samples via webcam
2. Train     - (re)train the LBPH recognizer on all registered faces
3. Attendance- open a live webcam session that recognizes faces and logs attendance
4. Dashboard - view attendance records and analytics (charts, trends, %)

Run with:
    streamlit run app.py

Note: webcam capture (Register/Attendance tabs) opens a native OpenCV
window on your machine — it will NOT stream inside the browser tab.
This is expected; look for the separate "Registering Face" /
"Attendance" window that pops up.
"""

import streamlit as st
import pandas as pd
import database as db
from register_faces import register_person
from train_model import train_recognizer
from mark_attendance import run_attendance_session

st.set_page_config(page_title="Face Recognition Attendance System", layout="wide")
db.init_db()

st.title("🎓 Face Recognition Attendance System")
st.caption("OpenCV (Haar Cascade + LBPH) · SQLite · Streamlit")

tab_register, tab_train, tab_attendance, tab_dashboard = st.tabs(
    ["➕ Register", "🧠 Train Model", "📸 Mark Attendance", "📊 Dashboard"]
)

# ---------------------------------------------------------------- Register
with tab_register:
    st.subheader("Register a new person")
    st.write(
        "Enter a name and click **Start Capture**. A webcam window will "
        "open and collect 30 face samples automatically — look at the camera "
        "and move your head slightly for varied angles."
    )
    name_input = st.text_input("Full name")

    if st.button("Start Capture", type="primary", disabled=not name_input):
        progress_bar = st.progress(0, text="Starting webcam...")

        def update_progress(count, total):
            progress_bar.progress(count / total, text=f"Captured {count}/{total} images")

        with st.spinner("Look at the webcam window..."):
            person_id, captured = register_person(name_input, progress_callback=update_progress)

        if captured > 0:
            st.success(f"Registered '{name_input}' (ID {person_id}) with {captured} images.")
            st.info("Go to the **Train Model** tab next to update the recognizer.")
        else:
            st.error("No face captured. Check your webcam and try again.")

    st.divider()
    st.subheader("Registered people")
    people = db.get_all_people()
    if people:
        st.dataframe(pd.DataFrame(people, columns=["ID", "Name"]), use_container_width=True)
    else:
        st.write("No one registered yet.")

# ------------------------------------------------------------------ Train
with tab_train:
    st.subheader("Train the recognizer")
    st.write(
        "Run this after registering one or more people, and again "
        "any time you add someone new."
    )

    if st.button("Train Now", type="primary"):
        progress_bar = st.progress(0, text="Training...")

        def update_train_progress(done, total):
            progress_bar.progress(done / total, text=f"Processing person {done}/{total}")

        try:
            n_people, n_images = train_recognizer(progress_callback=update_train_progress)
            st.success(f"Training complete — {n_people} people, {n_images} images.")
        except ValueError as e:
            st.error(str(e))

# ------------------------------------------------------------- Attendance
with tab_attendance:
    st.subheader("Mark attendance")
    st.write(
        "Click **Start Session** to open a live webcam window. Recognized "
        "faces are marked present automatically (once per day). Press **Q** "
        "in the webcam window to end the session."
    )

    if st.button("Start Session", type="primary"):
        messages = []

        def log_status(msg):
            messages.append(msg)

        try:
            with st.spinner("Session running — check the webcam window. Press Q there to stop."):
                marked = run_attendance_session(status_callback=log_status)
            st.success(f"Session ended. {len(marked)} unique people marked present.")
            for m in messages:
                st.write(f"✅ {m}")
        except FileNotFoundError as e:
            st.error(str(e))

# --------------------------------------------------------------- Dashboard
with tab_dashboard:
    st.subheader("Attendance records & analytics")
    records = db.get_attendance_records()

    if not records:
        st.write("No attendance marked yet.")
    else:
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Total check-ins logged", len(df))
        col2.metric("Unique people", df["name"].nunique())
        col3.metric("Days with attendance", df["date"].nunique())

        st.divider()

        left, right = st.columns(2)

        with left:
            st.write("**Attendance count per person**")
            per_person = df["name"].value_counts()
            st.bar_chart(per_person)

        with right:
            st.write("**Check-ins over time**")
            per_day = df.groupby(df["date"].dt.date).size()
            st.line_chart(per_day)

        st.divider()
        st.write("**All records**")
        st.dataframe(
            df.sort_values("date", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download as CSV", csv, "attendance_records.csv", "text/csv")
