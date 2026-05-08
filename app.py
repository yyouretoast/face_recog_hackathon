import cv2
import json
import numpy as np
import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

st.set_page_config(page_title="DEBI Face Recognition", page_icon="👤", layout="wide")

# Import DeepFace AFTER page config so Streamlit UI doesn't time out during heavy TensorFlow loading
with st.spinner("Initializing Deep Learning models... This may take a few moments on the first run."):
    from deepface import DeepFace

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

def find_match(embedding, known_embeddings, known_names, threshold=0.8):
    if not known_embeddings:
        return "Unknown", 0
    
    embedding = np.array(embedding)
    distances = [np.linalg.norm(embedding - k) for k in known_embeddings]
    min_dist = min(distances)
    
    if min_dist < threshold:
        return known_names[np.argmin(distances)], round((1 - min_dist) * 100, 1)
    return "Unknown", 0

# Custom CSS for Premium Design Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Base fonts */
    html, body {
        font-family: 'Outfit', sans-serif;
    }
    
    h1 {
        background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
        text-align: center;
        margin-bottom: 0.5rem !important;
        font-size: 3rem !important;
    }
    
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Card-like styling for columns */
    div[data-testid="column"] {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease;
    }
    
    div[data-testid="column"]:hover {
        transform: translateY(-5px);
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        box-shadow: 0 0 15px rgba(139, 92, 246, 0.5);
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>DEBI Face Recognition App</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Real-time Face Detection & Recognition Platform built for the DEBI Hackathon 2026</p>", unsafe_allow_html=True)

class FaceRecognizer(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.current_results = []
        _, self.known_embeddings, self.known_names = load_database()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        self.frame_count += 1
        
        # Process every 5th frame for performance
        if self.frame_count % 5 == 0:
            try:
                faces = DeepFace.extract_faces(img, enforce_detection=False)
                self.current_results = []
                
                for face in faces:
                    x = face['facial_area']['x']
                    y = face['facial_area']['y']
                    w = face['facial_area']['w']
                    h = face['facial_area']['h']
                    
                    face_img = img[y:y+h, x:x+w]
                    
                    try:
                        embedding = DeepFace.represent(face_img, model_name='Facenet', enforce_detection=False)[0]['embedding']
                        name, confidence = find_match(embedding, self.known_embeddings, self.known_names)
                    except:
                        name, confidence = "Unknown", 0
                        
                    self.current_results.append((x, y, w, h, name, confidence))
            except:
                pass
                
        for (x, y, w, h, name, confidence) in self.current_results:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            label = f"{name} ({confidence}%)" if name != "Unknown" else "Unknown"
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            # Display text inside the box as per hackathon guidelines
            cv2.putText(img, label, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Camera Feed")
    webrtc_streamer(
        key="face-recognition",
        video_processor_factory=FaceRecognizer,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    )

with col2:
    st.subheader("Database Management")
    st.write("Add a new person to the database by uploading their image.")
    
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    new_name = st.text_input("Person's Name")
    
    if st.button("Add to Database"):
        if uploaded_file is not None and new_name:
            with st.spinner("Processing image..."):
                with open("temp.jpg", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                try:
                    embedding = DeepFace.represent("temp.jpg", model_name='Facenet', enforce_detection=True)[0]['embedding']
                    save_to_database(new_name, np.array(embedding))
                    st.success(f"✅ Successfully added {new_name} to the database! Please restart the video stream to apply changes.")
                except Exception as e:
                    st.error(f"❌ Could not detect a face in the image. Please try another image.")
                
                if os.path.exists("temp.jpg"):
                    os.remove("temp.jpg")
        else:
            st.warning("⚠️ Please upload an image and enter a name.")
            
    st.divider()
    st.write("Current Database Entries:")
    database, _, _ = load_database()
    if database:
        for entry in database:
            st.write(f"- 👤 {entry['name']}")
    else:
        st.write("Database is empty.")