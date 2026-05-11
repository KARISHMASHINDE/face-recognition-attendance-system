import streamlit as st
import face_rec
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import streamlit.components.v1 as components
import av
import time
from queue import Queue, Empty

# 1. PAGE CONFIG
st.set_page_config(page_title="Attendance System", page_icon="📷", layout="wide")

# Hide the invisible HTML component used for audio
st.markdown("""
<style>
    iframe[title="streamlit_components.v1.html"] { display: none; } 
    .stAlert { margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

# 2. SHARED RESOURCES (Caching)
@st.cache_resource
def get_attendance_queue():
    return Queue()

attendance_queue = get_attendance_queue()

@st.cache_data(ttl=600)
def load_face_db(db_name):
    return face_rec.retrive_data(db_name)

@st.cache_resource
def load_model(db_name):
    return face_rec.RealTimePred(db_name)

# 3. AUTH & SESSION CHECKS
if not st.session_state.get("logged_in"):
    st.warning("Please login first 🔐")
    st.stop()

if "audio_unlocked" not in st.session_state:
    st.session_state.audio_unlocked = False

if not st.session_state.audio_unlocked:
    st.title("📷 System Activation")
    if st.button("🚀 Unlock Audio & Start Monitoring"):
        st.session_state.audio_unlocked = True
        st.rerun()
    st.stop()

# 4. LOAD DATA
face_db = load_face_db(st.session_state.db_name)
if face_db is None or face_db.empty:
    st.error("Could not load face database. Check MySQL connection.")
    st.stop()

realtimepred = load_model(st.session_state.db_name)

# 5. SIDEBAR
st.sidebar.markdown(f"### 👤 {st.session_state.get('user_name')}\n**DB:** `{st.session_state.get('db_name')}`")
if st.sidebar.button("🚪 Logout"):
    st.session_state.clear()
    st.rerun()

# 6. AUDIO FUNCTION (JavaScript)
def speak_js(text):
    if not text: return
    clean_text = text.replace("'", "\\'").replace("\n", " ")
    ts = time.time()
    components.html(f"""
        <div id="{ts}"></div>
        <script>
            window.speechSynthesis.cancel(); 
            const msg = new SpeechSynthesisUtterance('{clean_text}');
            msg.lang = 'en-US';
            msg.rate = 0.9; // Slightly slower for clarity
            window.speechSynthesis.speak(msg);
        </script>
    """, height=0)

# 7. VIDEO CALLBACK
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    try:
        # Prediction logic from face_rec.py
        pred_img, detected, event = realtimepred.face_prediction(
            img, face_db, "facial_features", ["Name", "EmployeeID"], 0.65
        )
        if event: 
            attendance_queue.put(event)
    except Exception as e:
        pred_img = img
    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")

# 8. MAIN UI
st.title("📷 Face Attendance System")

webrtc_ctx = webrtc_streamer(
    key="attendance_camera",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]}
        ]
    },
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

# 9. EVENT PROCESSING (Fixed: Removed while True)
message_placeholder = st.empty()

try:
    # Process all pending items in the queue for this rerun
    while not attendance_queue.empty():
        event = attendance_queue.get_nowait()
        name, status = event[0], event[1]
        duration = event[2] if event[2] else ""
        
        if status == "IN": 
            msg = f"{name}, Check-in successful"
        elif status == "OUT": 
            msg = f"{name}, Check-out successful. Worked Time: {duration}"
        elif status == "DONE": 
            msg = f"{name}, Attendance already marked"
        else: 
            msg = f"Hello {name}"

        # Display and Speak
        message_placeholder.success(msg)
        st.toast(msg)
        speak_js(msg)
        
except Empty:
    pass

# 10. SMART RERUN
# Only rerun the script if the camera is active to check for new events
if webrtc_ctx and webrtc_ctx.state.playing:
    time.sleep(0.5) # Balanced delay to prevent CPU spikes
    st.rerun()