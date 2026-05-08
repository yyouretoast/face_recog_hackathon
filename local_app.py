import cv2
import json
import numpy as np
from deepface import DeepFace
import os

DATABASE_FILE = 'database.json'

def load_database():
    try:
        with open(DATABASE_FILE, 'r') as f:
            data = json.load(f)
        known_embeddings = [np.array(p['embedding']) for p in data]
        known_names = [p['name'] for p in data]
        return data, known_embeddings, known_names
    except:
        return [], [], []

def save_to_database(name, embedding):
    try:
        with open(DATABASE_FILE, 'r') as f:
            database = json.load(f)
    except:
        database = []
    
    database.append({'name': name, 'embedding': embedding.tolist()})
    
    with open(DATABASE_FILE, 'w') as f:
        json.dump(database, f)
    
    print(f"✅ {name} added to database")

def find_match(embedding, known_embeddings, known_names, threshold=0.8):
    if not known_embeddings:
        return "Unknown", 0
    
    embedding = np.array(embedding)
    distances = [np.linalg.norm(embedding - k) for k in known_embeddings]
    min_dist = min(distances)
    
    if min_dist < threshold:
        return known_names[np.argmin(distances)], round((1 - min_dist) * 100, 1)
    return "Unknown", 0

# State
database, known_embeddings, known_names = load_database()
frame_count = 0
current_results = []
adding_mode = False
capture_embedding = None

video = cv2.VideoCapture(0)
print("Camera started")
print("Press 'A' to add current unknown face to database")
print("Press 'Q' to quit")

while True:
    ret, frame = video.read()
    frame = cv2.flip(frame, 1)
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    if not ret:
        break

    frame_count += 1

    if frame_count % 5 == 0:
        try:
            faces = DeepFace.extract_faces(frame, enforce_detection=False)
            current_results = []

            for face in faces:
                x = face['facial_area']['x']
                y = face['facial_area']['y']
                w = face['facial_area']['w']
                h = face['facial_area']['h']

                face_img = frame[y:y+h, x:x+w]

                try:
                    embedding = DeepFace.represent(face_img,
                                                  model_name='Facenet',
                                                  enforce_detection=False)[0]['embedding']
                    name, confidence = find_match(embedding, known_embeddings, known_names)
                    
                    # Store last unknown embedding for adding
                    if name == "Unknown":
                        capture_embedding = np.array(embedding)
                        
                except:
                    name, confidence = "Unknown", 0

                current_results.append((x, y, w, h, name, confidence))
        except:
            pass

    for (x, y, w, h, name, confidence) in current_results:
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        label = f"{name} ({confidence}%)" if name != "Unknown" else "Unknown - Press A to add"
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Instructions on screen
    cv2.putText(frame, "A: Add face | Q: Quit", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if adding_mode:
        cv2.putText(frame, "Type name in terminal and press Enter", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow('DEBI Face Recognition', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('a'):
        if capture_embedding is not None:
            video.release()
            cv2.destroyAllWindows()
            
            name = input("Enter name for this face: ").strip()
            
            if name:
                save_to_database(name, capture_embedding)
                database, known_embeddings, known_names = load_database()
                print(f" {name} added! Restarting camera...")
            
            video = cv2.VideoCapture(0)
        else:
            print("No unknown face detected to add")

video.release()
cv2.destroyAllWindows()