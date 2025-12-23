import cv2
import math
import sqlite3
import os
from datetime import datetime

# ----------------- Setup -----------------
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not opened")
    exit()

LINE_Y = 250
DIST_THRESHOLD = 50

tracked_faces = {}
# New Set to keep track of faces we have already saved
counted_ids = set() 

face_id = 0
in_count = 0

# ----------------- Create folder -----------------
if not os.path.exists("faces"):
    os.makedirs("faces")

# ----------------- Database -----------------
conn = sqlite3.connect("face_data.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT,
    date TEXT,
    time TEXT
)
""")
conn.commit()

# ----------------- Helper function -----------------
def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# ----------------- Main Loop -----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw the line
    #cv2.line(frame, (0, LINE_Y), (640, LINE_Y), (255, 0, 0), 2)

    current_faces = []

    for (x, y, w, h) in faces:
        cx = x + w//2
        cy = y + h//2
        current_faces.append((cx, cy, x, y, w, h))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    # Create a copy of keys to iterate safely (optional but good practice)
    tracked_ids_current_frame = []

    for cx, cy, x, y, w, h in current_faces:
        matched = False

        for fid, (px, py) in tracked_faces.items():
            if distance((cx, cy), (px, py)) < DIST_THRESHOLD:
                
                tracked_ids_current_frame.append(fid) # Mark this ID as active
                
                # --------- IN ENTRY ---------
                # Check crossing AND check if we haven't counted this ID yet
                if py < LINE_Y and cy >= LINE_Y and fid not in counted_ids:
                    
                    in_count += 1
                    counted_ids.add(fid) # MARK AS COUNTED

                    # Crop face
                    face_img = frame[y:y+h, x:x+w]

                    timestamp = datetime.now()
                    img_name = f"faces/face_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(img_name, face_img)

                    # Save to database
                    cursor.execute(
                        "INSERT INTO entries (image_path, date, time) VALUES (?, ?, ?)",
                        (img_name, timestamp.date().isoformat(), timestamp.time().strftime("%H:%M:%S"))
                    )
                    conn.commit()
                    print(f"Saved: {img_name}")

                tracked_faces[fid] = (cx, cy)
                matched = True
                break

        if not matched:
            tracked_faces[face_id] = (cx, cy)
            face_id += 1

    # Optional: Clean up old faces (basic garbage collection)
    # If tracked_faces gets too big, it slows down. 
    # This is a simple way to keep the dictionary small.
    if len(tracked_faces) > 50:
        tracked_faces.clear()
        counted_ids.clear()

    cv2.putText(frame, f"IN COUNT: {in_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("AI Face Counter with DB", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
conn.close()
cv2.destroyAllWindows()