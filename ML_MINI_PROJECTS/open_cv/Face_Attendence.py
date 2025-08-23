import streamlit as st
import cv2
import numpy as np
import face_recognition as fr
import pickle
import csv
import os
from datetime import datetime
import pyttsx3

engine = pyttsx3.init()    # initialize text-to-speech engine

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ------------------ File Names ------------------
DATA_FILE = "face_data.pkl"        # stores encodings, names, ids
ATTENDANCE_FILE = "attendance.csv" # stores attendance records (all days)
TODAY_FILE = "today.csv"           # stores today's attendance only

# ------------------ Persistence ------------------
def save_data(known_face_encodings, known_face_names, known_face_ids):
    with open(DATA_FILE, "wb") as f:
        pickle.dump((known_face_encodings, known_face_names, known_face_ids), f)

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "rb") as f:
            return pickle.load(f)
    return [], [], []

def ensure_attendance_files():
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "ID", "Date", "Time"])

    # Reset today.csv daily
    today_str = datetime.now().strftime("%Y-%m-%d")
    reset_today = True
    if os.path.exists(TODAY_FILE):
        with open(TODAY_FILE, "r") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            first_row = next(reader, None)
            if first_row and first_row[2] == today_str:
                reset_today = False
    if reset_today:
        with open(TODAY_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "ID", "Date", "Time"])


def load_marked_ids_today():
    """Load IDs already marked today from today's file."""
    marked_ids = set()
    if os.path.exists(TODAY_FILE):
        with open(TODAY_FILE, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    marked_ids.add(row[1])
    return marked_ids
def mark_attendance(name, id_no, marked_ids_today):
    date = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")

    if id_no not in marked_ids_today:
        # Always add to global attendance
        with open(ATTENDANCE_FILE, "a", newline="") as f:
            csv.writer(f).writerow([name, id_no, date, time_str])

        # Add to today's attendance
        with open(TODAY_FILE, "a", newline="") as f:
            csv.writer(f).writerow([name, id_no, date, time_str])

        marked_ids_today.add(id_no)
        return True
    return False


# ------------------ Image/encoding utils ------------------
def bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    """Decode raw bytes to OpenCV BGR image."""
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)           #cv2.imdecode() → takes the byte array (still compressed, e.g., JPEG/PNG format) and decodes it into an actual image matrix.
    return bgr

def get_face_encodings_from_bgr(bgr_image, model="hog"):       # hog is CPU-based, faster, less accurate   and CNN is slower in cpu , and it requires GPU for speed, more accurate
    """
    Return list of face encodings from a BGR image.
    model: "hog" (CPU, fast) or "cnn" (GPU support, slower on CPU)
    """
    if bgr_image is None:
        return []
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    locations = fr.face_locations(rgb, model=model)
    encodings = fr.face_encodings(rgb, locations)
    return encodings, locations

def register_person(name: str, id_no: str, image_bytes: bytes,
                    known_encodings, known_names, known_ids, tolerance: float = 0.4):
    """Add a new person to database if a face is detected; returns (ok, msg)."""
    if not name or not id_no:
        return False, "Please enter both Name and ID."

    bgr = bytes_to_bgr(image_bytes)
    encodings, _ = get_face_encodings_from_bgr(bgr)
    if len(encodings) == 0:
        return False, "No face detected in the provided image."
    if len(encodings) > 1:
        return False, "Multiple faces detected. Please upload an image with only one face."

    new_encoding = encodings[0]

    # Check duplicate ID
    if id_no in known_ids:
        return False, f"ID '{id_no}' already exists."

    # ✅ Check duplicate face (using tolerance)
    if len(known_encodings) > 0:
        matches = fr.compare_faces(known_encodings, new_encoding, tolerance=tolerance)
        if any(matches):
            existing_index = matches.index(True)
            existing_name = known_names[existing_index]
            existing_id = known_ids[existing_index]
            return False, f"This face already exists in the system with ID  {existing_id}."

    # If no duplicates, save
    known_encodings.append(new_encoding)
    known_names.append(name)
    known_ids.append(id_no)
    save_data(known_encodings, known_names, known_ids)
    speak(f"{name} you have Registered successfully")
    return True, f"Registered {name} ({id_no}) successfully."

# ------------------ Recognition ------------------

def recognize_from_image(image_bytes: bytes, known_encodings, known_names, known_ids, tolerance: float = 0.4):
    """
    Recognize first face in image_bytes.
    Returns (recognized: bool, name, id_no, distance, annotated_image_bgr)
    If not recognized, name/id_no = 'Unknown'/'N/A'
    """
    bgr = bytes_to_bgr(image_bytes)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    face_locations = fr.face_locations(rgb)
    face_encodings = fr.face_encodings(rgb, face_locations)

    name, id_no, distance = "Unknown", "N/A", None

    if len(face_encodings) == 0:
        return None, None, None, None, bgr   # <--- mark as "no face"


    # We'll process the first detected face (you can loop all if needed)
    face_encoding = face_encodings[0]
    matches = fr.compare_faces(known_encodings, face_encoding, tolerance=tolerance)
    face_distances = fr.face_distance(known_encodings, face_encoding) if len(known_encodings) else np.array([])       #distance 0.3 → strong match, distance 0.7 → weak match.

    best_match_index = None
    if len(face_distances) > 0:
        best_match_index = int(np.argmin(face_distances))

    recognized = False
    if best_match_index is not None and matches[best_match_index]:
        recognized = True
        name = known_names[best_match_index]
        id_no = known_ids[best_match_index]
        distance = float(face_distances[best_match_index])

    # Draw on image
    if len(face_locations) > 0:
        (top, right, bottom, left) = face_locations[0]
        color = (0, 200, 0) if recognized else (0, 0, 255)
        cv2.rectangle(bgr, (left, top), (right, bottom), color, 2)
        label = f"{name} ({id_no})" if recognized else "Unknown"
        cv2.putText(bgr, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return recognized, name, id_no, distance, bgr

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Face Attendance", page_icon="📸", layout="wide")
# st.title("📸 Face Recognition Attendance System")
# Place this near the top of your script (after st.set_page_config)
st.markdown(
    """
    <style>
    /* ---------------- General Styling ---------------- */
    body {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }


    /* ---------------- Tabs Styling ---------------- */
    .stTabs [role="tablist"] {
        gap: 20px;
        justify-content: center;
    }
    .stTabs [role="tab"] {
        background: #1e293b;
        color: white;
        border-radius: 15px;
        padding: 10px 20px;
        transition: all 0.3s ease-in-out;
        border: 2px solid transparent;
    }
    .stTabs [role="tab"]:hover {
        background: #334155;
        border-color: #22d3ee;
        transform: scale(1.05);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #22d3ee, #06b6d4);
        color: black !important;
        font-weight: bold;
        border: none;
    }

    /* ---------------- Input Fields ---------------- */
    .stTextInput > div > div > input {
        background: #1e293b;
        color: white;
        border-radius: 12px;
        border: 1px solid #475569;
        transition: 0.3s;
    }
    .stTextInput > div > div > input:focus {
        border-color: #22d3ee;
        box-shadow: 0 0 10px #22d3ee;
    }

    /* ---------------- Buttons ---------------- */
    button {
        border-radius: 12px !important;
        background: linear-gradient(90deg, #06b6d4, #3b82f6);
        color: white !important;
        font-weight: bold;
        padding: 10px 25px !important;
        border: none;
        transition: 0.3s ease-in-out;
    }
    button:hover {
        background: linear-gradient(90deg, #3b82f6, #06b6d4);
        transform: scale(1.05);
        box-shadow: 0px 0px 12px #22d3ee;
    }

    div.stButton > button {
        width: 100%;
        height: 50px;
        border-radius: 12px;
        background: #22c55e; /* green */
        color: white;
        font-weight: bold;
        font-size: 18px;
        border: none;
        margin-top: 8px;
    }
    div.stButton > button:hover {
        background: #16a34a; /* darker green on hover */
        transform: scale(1.02);
    }
    /* ---------------- Alerts ---------------- */
    .stAlert {
        border-radius: 12px;
        padding: 15px;
        font-size: 1rem;
        animation: fadeIn 0.5s ease-in-out;
    }
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }

    /* ---------------- Images ---------------- */
    img {
        border-radius: 20px;
        box-shadow: 0 0 20px rgba(0,0,0,0.5);
        transition: transform 0.3s;
    }
    img:hover {
        transform: scale(1.03);
    }

    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    .title {
        font-size: 38px;
        font-weight: bold;
        color: #ff4d4d;
        text-shadow: 0 0 10px #ff1a1a, 0 0 20px #ff1a1a, 0 0 30px #ff1a1a;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
    }
    .title img {
        width: 40px;
        height: 40px;
        filter: drop-shadow(0 0 5px #ff1a1a);
    }
    .title {
        text-align: center;
        background: linear-gradient(90deg, #ff416c, #ff4b2b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em !important;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { text-shadow: 0 0 5px #ff416c; }
        50% { text-shadow: 0 0 20px #ff4b2b; }
        100% { text-shadow: 0 0 5px #ff416c; }
    }
    
    # /* Background Image */
    # .stApp {
    #     background-image: url("https://repository-images.githubusercontent.com/334098583/7b863500-be31-11eb-927d-1eea89be6b70"); /* Example: students classroom */
    #     background-size: cover;
    #     background-position: center;
    #     background-attachment: fixed;
    # }

    # /* Semi-transparent overlay for readability */
    # .stApp::before {
    #     content: "";
    #     position: fixed;
    #     top: 0;
    #     left: 0;
    #     right: 0;
    #     bottom: 0;
    #     background-color: rgba(255, 255, 255, 0.75); /* light white overlay */
    #     z-index: -1;
    # }
    </style>

    <div class="title">
        <img src="https://cdn-icons-png.flaticon.com/512/709/709579.png" alt="camera">
        Face Recognition Attendance System
    </div>
    
    
    
    
    """,
    unsafe_allow_html=True
)

# Ensure attendance file exists
ensure_attendance_files()

# Load known data once per session
if "known_face_encodings" not in st.session_state:
    kfe, kfn, kfi = load_data()
    st.session_state.known_face_encodings = kfe
    st.session_state.known_face_names = kfn
    st.session_state.known_face_ids = kfi

# Load today's marked IDs (persist for session)
if "marked_ids_today" not in st.session_state:
    st.session_state.marked_ids_today = load_marked_ids_today()


tab1, tab2 = st.tabs(["📝 Register", "✅ Attendance"])

# ------------------ Register Tab ------------------
with tab1:
    st.subheader("Register a New Person")
    col1, col2 = st.columns([1, 1])

    name = st.text_input("Name" , placeholder="Enter your Name to get Registered :")
    id_no = st.text_input("ID" ,placeholder="Enter your Unique ID Number :")

    st.markdown("**Provide a face image (upload or camera):**")
    upload_img = st.file_uploader("Upload face image", type=["jpg", "jpeg", "png"])
    cam_img = st.camera_input("Or take a picture")

    img_bytes = None
    if cam_img is not None:
        img_bytes = cam_img.getvalue()
    elif upload_img is not None:
        img_bytes = upload_img.read()

    if st.button("Register"):
        if img_bytes is None:
            st.warning("Please upload or capture a face image.")
            speak("Please upload or capture a face image.")
        else:
            ok, msg = register_person(
                name.strip(),
                id_no.strip(),
                img_bytes,
                st.session_state.known_face_encodings,
                st.session_state.known_face_names,
                st.session_state.known_face_ids
            )
            if ok:
                st.success(msg)
            else:
                st.error(msg)

# ------------------ Attendance Tab ------------------
with tab2:
    st.subheader("Mark Attendance")
    st.markdown("Use your camera to capture a frame and we’ll recognize the face.")
    
    cam_shot = st.camera_input("Capture for attendance")
    if cam_shot is not None:
        recognized, name, id_no, distance, annotated_bgr = recognize_from_image(
            cam_shot.getvalue(),
            st.session_state.known_face_encodings,
            st.session_state.known_face_names,
            st.session_state.known_face_ids,
            tolerance=0.4
        )

        # Show annotated image
        if annotated_bgr is not None:
            st.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), caption="Processed Frame", use_container_width=True)

        if recognized is None:  # ✅ No face detected at all
            st.warning("⚠️ No face detected.(or) Face is Not visible. Please take a clear photo with your face visible.")
            speak("No face detected.(or) Face is Not visible. Please take a clear photo with your face visible.")
        elif recognized:
            added = mark_attendance(name, id_no, st.session_state.marked_ids_today)
            if added:
                st.success(f"✅ Attendance marked for {name} ({id_no}).")
                speak(f"Welcome {name}. Your attendance has been marked.")
            else:
                st.info(f"ℹ️ {name} ({id_no}) is already marked today.")
            if distance is not None:
                st.caption(f"Match distance: {distance:.3f}")
        else:
            st.warning("👤 Face not recognized. Please register first in the **Register** tab.")
            # speak("Face not recognized. Please register first in the Register tab.")
