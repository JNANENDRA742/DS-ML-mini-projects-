# Face Attendance System (OpenCV + Face Recognition)

A real-time **webcam attendance** app.  
When a person faces the camera, the app identifies them, **logs attendance to a CSV** (first time only per session/day), and **stores/loads face encodings in a PKL** file to speed up future runs.

> Auto-generated files (`*.csv`, `*.pkl`) are ignored by Git via `.gitignore`.

---

## ✨ Features
- Live face detection & recognition from webcam
- First-time attendance logging (prevents duplicate entries)
- Creates `Attendance.csv` and `encodings.pkl` automatically if missing
- Simple one-command run
- Easy to add new people (Name + ID)

---

## 📁 Project Structure
