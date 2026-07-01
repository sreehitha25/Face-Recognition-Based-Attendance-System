"""
database.py
------------
Handles all SQLite database operations for the Face Recognition
Attendance System: storing registered people, and logging attendance
records with timestamps.
"""

import sqlite3
from datetime import datetime, date

DB_PATH = "attendance.db"


def get_connection():
    """Return a new SQLite connection."""
    return sqlite3.connect(DB_PATH)


def init_db():
    """Create the required tables if they don't already exist."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY,      -- same as the numeric label used by LBPH
            name TEXT NOT NULL UNIQUE,
            registered_on TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            FOREIGN KEY (person_id) REFERENCES people (id)
        )
    """)

    conn.commit()
    conn.close()


def add_person(person_id: int, name: str):
    """Register a new person (or ignore if the name already exists)."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO people (id, name, registered_on) VALUES (?, ?, ?)",
        (person_id, name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    conn.close()


def get_all_people():
    """Return list of (id, name) for all registered people."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM people ORDER BY id")
    rows = cur.fetchall()
    conn.close()
    return rows


def get_next_person_id():
    """Auto-increment helper for assigning a new numeric label."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT MAX(id) FROM people")
    max_id = cur.fetchone()[0]
    conn.close()
    return 1 if max_id is None else max_id + 1


def mark_attendance(person_id: int, name: str):
    """
    Log attendance for a person, but only once per calendar day
    to avoid duplicate entries from repeated frame detections.
    Returns True if a new record was inserted, False if already marked today.
    """
    today = date.today().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM attendance WHERE person_id = ? AND date = ?",
        (person_id, today),
    )
    already_marked = cur.fetchone() is not None

    if not already_marked:
        cur.execute(
            "INSERT INTO attendance (person_id, name, date, time) VALUES (?, ?, ?, ?)",
            (person_id, name, today, now_time),
        )
        conn.commit()

    conn.close()
    return not already_marked


def get_attendance_records():
    """Return all attendance records as a list of dicts, most recent first."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT name, date, time FROM attendance ORDER BY date DESC, time DESC"
    )
    rows = cur.fetchall()
    conn.close()
    return [{"name": r[0], "date": r[1], "time": r[2]} for r in rows]
