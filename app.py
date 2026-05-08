import cv2
import json
import numpy as np
import os
import threading
import streamlit as st
import av
import face_recognition

_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="DEBI Face Recognition", page_icon="👤", layout="wide")

# ─── Database helpers ────────────────────────────────────────────────────────
DATABASE_FILE = os.path.join(_DIR, 'database.json')

def load_database():
    try:
        if not os.path.exists(DATABASE_FILE):
            return [], [], []
        with open(DATABASE_FILE, 'r') as f:
            data = json.load(f)
        known_embeddings = [np.array(p['embedding']) for p in data]
        known_names = [p['name'] for p in data]
        return data, known_embeddings, known_names
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return [], [], []

def save_to_database(name, embedding):
    try:
        database = []
        if os.path.exists(DATABASE_FILE):
            with open(DATABASE_FILE, 'r') as f:
                database = json.load(f)

        database.append({'name': name, 'embedding': embedding.tolist()})
        with open(DATABASE_FILE, 'w') as f:
            json.dump(database, f)
    except Exception as e:
        st.error(f"Error saving to database: {e}")

def clear_database():
    try:
        with open(DATABASE_FILE, 'w') as f:
            json.dump([], f)
        st.success("🗑️ Database cleared!")
    except Exception as e:
        st.error(f"Error clearing database: {e}")

def find_match(embedding, known_embeddings, known_names, threshold=0.6):
    if not known_embeddings:
        return "Unknown", 0

    # Use face_recognition distance (Euclidean)
    distances = face_recognition.face_distance([embedding], known_embeddings)[0]
    min_dist = np.min(distances)
    best_name = known_names[np.argmin(distances)]

    if min_dist < threshold:
        # Convert distance to a pseudo-confidence percentage
        confidence = round((1 - (min_dist / threshold)) * 100, 1) if min_dist < threshold else 0
        return best_name, max(0, confidence)

    return "Unknown", 0

# ─── Shared processing function ─────────────────────────────────────────────
def process_image(img, threshold=0.6):
    """Detect faces, match embeddings, return annotated image + results."""
    _, known_embeddings, known_names = load_database()
    results = []
    annotated = img.copy()

    # face_recognition requires RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        face_locations = face_recognition.face_locations(rgb_img)
        face_encodings = face_recognition.face_encodings(rgb_img, known_face_locations=face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Convert (top, right, bottom, left) to (x, y, w, h)
            x, y, w, h = left, top, right - left, bottom - top

            if w < 20 or h < 20:
                continue

            name, confidence = find_match(face_encoding, known_embeddings, known_names, threshold)
            results.append({"name": name, "confidence": confidence})

            # Draw bounding box
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            label = f"{name} ({confidence}%)" if name != "Unknown" else "Unknown"
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 3)

            # Label background inside the box
            lbl_sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(annotated, (x, y), (x + lbl_sz[0] + 10, y + lbl_sz[1] + 15), color, -1)
            cv2.putText(annotated, label, (x + 5, y + lbl_sz[1] + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    except Exception:
        pass
    return annotated, results

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
html, body { font-family: 'Outfit', sans-serif; }
h1 {
    background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700 !important;
    text-align: center;
    font-size: 2.5rem !important;
}
.subtitle { text-align: center; color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem; }
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white; border: none; border-radius: 8px;
    padding: 0.6rem 1rem; font-weight: 600;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
    box-shadow: 0 0 20px rgba(139, 92, 246, 0.4);
    transform: scale(1.02);
}
</style>""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("<h1>DEBI Face Recognition</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Real-time Face Detection &amp; Recognition · DEBI Hackathon 2026</p>", unsafe_allow_html=True)

# ─── Sidebar: Database Management ───────────────────────────────────────────
with st.sidebar:
    st.header("👤 Database")

    # Sensitivity Slider
    st.subheader("⚙️ Tuning")
    threshold = st.slider("Recognition Sensitivity", 0.1, 1.0, 0.6, 0.05,
                         help="Lower = Stricter (Higher accuracy), Higher = Looser (More detections)")

    st.divider()
    st.caption("Add faces to the recognition database")
    uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
    new_name = st.text_input("Person's Name")
    if st.button("➕ Add to Database"):
        if uploaded_file is not None and new_name:
            with st.spinner("Extracting face embedding..."):
                file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                try:
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(rgb_img)
                    if len(encodings) > 0:
                        save_to_database(new_name, np.array(encodings[0]))
                        st.success(f"✅ Added **{new_name}**!")
                        st.rerun()
                    else:
                        st.error("❌ No face detected. Try another image.")
                except Exception:
                    st.error("❌ Error processing image.")
        else:
            st.warning("Upload an image and enter a name.")
    st.divider()
    st.subheader("Registered Faces")
    database, _, _ = load_database()
    if database:
        names = {}
        for entry in database:
            names[entry['name']] = names.get(entry['name'], 0) + 1
        for name, count in names.items():
            st.write(f"👤 **{name}** ({count} embedding{'s' if count > 1 else ''})")

        if st.button("🗑️ Clear All Faces"):
            clear_database()
            st.rerun()
    else:
        st.info("No faces registered yet.")

# ─── Main Content ────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📸 Snapshot Mode", "🎥 Live Mode"])

# ── Tab 1: Snapshot Mode ───────────────────────────────────
with tab1:
    st.caption("Take a photo or upload an image to recognize faces instantly.")
    col1, col2 = st.columns([1, 1])
    with col1:
        input_method = st.radio("Input", ["📷 Camera", "📁 Upload"], horizontal=True, label_visibility="collapsed")
        img_source = None
        if input_method == "📷 Camera":
            camera_photo = st.camera_input("Take a photo")
            if camera_photo:
                file_bytes = np.frombuffer(camera_photo.getvalue(), np.uint8)
                img_source = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        else:
            uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="snap")
            if uploaded:
                file_bytes = np.frombuffer(uploaded.read(), np.uint8)
                img_source = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    with col2:
        if img_source is not None:
            with st.spinner("🔍 Analyzing faces..."):
                annotated, results = process_image(img_source, threshold)
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                             caption="Recognition Results", use_container_width=True)
                if results:
                    for r in results:
                        icon = "✅" if r["name"] != "Unknown" else "❓"
                        conf = f" — {r['confidence']}% match" if r["name"] != "Unknown" else ""
                        st.markdown(f"{icon} **{r['name']}**{conf}")
                else:
                    st.info("No faces detected.")
        else:
            st.info("👈 Take a photo or upload an image to begin.")

# ── Tab 2: Live Mode ─────────────────────────────────────────
with tab2:
    st.caption("Real-time recognition via webcam. Requires a stable connection.")
    try:
        from streamlit_webrtc import webrtc_streamer

        _, known_emb, known_nm = load_database()
        state = {"count": 0, "results": []}
        lock = threading.Lock()

        def video_callback(frame):
            img = frame.to_ndarray(format="bgr24")
            with lock:
                state["count"] += 1
                cnt = state["count"]
            if cnt % 15 == 0:
                try:
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_img)
                    face_encodings = face_recognition.face_encodings(rgb_img, known_face_locations=face_locations)

                    new_res = []
                    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                        # Translate to (x, y, w, h) for the state
                        x, y, w, h = left, top, right - left, bottom - top
                        nm, cf = find_match(encoding, known_emb, known_nm, threshold)
                        new_res.append((x, y, w, h, nm, cf))
                    with lock:
                        state["results"] = new_res
                except Exception:
                    pass
            with lock:
                cur = list(state["results"])
            for (x, y, w, h, nm, cf) in cur:
                color = (0, 255, 0) if nm != "Unknown" else (0, 0, 255)
                label = f"{nm} ({cf}%)" if nm != "Unknown" else "Unknown"
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
                lsz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(img, (x, y), (x + lsz[0] + 10, y + lsz[1] + 15), color, -1)
                cv2.putText(img, label, (x + 5, y + lsz[1] + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(
            key="live",
            video_frame_callback=video_callback,
            rtc_configuration={"iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
            ]},
            media_stream_constraints={"video": True, "audio": False},
        )
        st.caption("💡 If video doesn't connect, use Snapshot Mode instead.")
    except ImportError:
        st.warning("Live mode unavailable. Use Snapshot Mode.")
    except Exception as e:
        st.error(f"Live mode error: {e}")
