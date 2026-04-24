import streamlit as st
import face_rec
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import streamlit.components.v1 as components
import av
import time
import logging
from queue import Queue, Empty

# =========================
# PAGE CONFIG & CSS
# =========================
st.set_page_config(page_title="Attendance System", page_icon="📷", layout="wide")

st.markdown("""
    <style>
    iframe[title="streamlit_components.v1.html"] { display: none; }
    .stAlert { margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

# =========================
# THREAD-SAFE GLOBAL QUEUE
# =========================
@st.cache_resource
def get_attendance_queue():
    """Provides a thread-safe pipe between the camera thread and UI thread."""
    return Queue()

attendance_queue = get_attendance_queue()

# =========================
# LOGIN CHECK
# =========================
if not st.session_state.get("logged_in"):
    st.warning("Please login first 🔐")
    st.stop()

if st.session_state.get("user_role") not in ["system", "super"]:
    st.error("⛔ Unauthorized Access")
    st.stop()

# =========================
# SIDEBAR
# =========================
st.sidebar.markdown(f"""
### 👤 User Info
**Name:** {st.session_state.get("user_name", "")}  
**DB:** `{st.session_state.get("db_name", "")}`  
**Role:** {st.session_state.get("user_role", "")}
""")

if st.sidebar.button("🚪 Logout"):
    st.session_state.clear()
    st.rerun()

# =========================
# SESSION STATE INIT
# =========================
st.session_state.setdefault("last_detection_time", {})
st.session_state.setdefault("last_event_time", 0)
st.session_state.setdefault("current_msg", None)

message_placeholder = st.empty()
COOLDOWN_SECONDS = 180

# =========================
# LOAD RESOURCES
# =========================
@st.cache_data(ttl=300)
def load_face_db(db_name):
    return face_rec.retrive_data(db_name)

@st.cache_resource
def load_model(db_name):
    return face_rec.RealTimePred(db_name)

face_db = load_face_db(st.session_state.db_name)
realtimepred = load_model(st.session_state.db_name)

# =========================
# VOICE FUNCTION
# =========================
def speak_js(text):
    clean_text = text.replace("'", "")
    components.html(f"""
        <script>
            window.speechSynthesis.cancel();
            const msg = new SpeechSynthesisUtterance('{clean_text}');
            window.speechSynthesis.speak(msg);
        </script>
    """, height=0)

# =========================
# VIDEO CALLBACK
# =========================
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    try:
        # Use the global queue, NOT st.session_state (fixes the thread error)
        pred_img, detected, event = realtimepred.face_prediction(
            img, face_db, "facial_features", ["Name", "EmployeeID"], 0.7
        )
        if event:
            attendance_queue.put(event)
    except Exception:
        pred_img = img
    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")

# =========================
# WEBRTC CAMERA
# =========================
st.title("📷 Real-Time Face Attendance System")

webrtc_ctx = webrtc_streamer(
    key="attendance_camera",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

# =========================
# PROCESS QUEUE (UI UPDATE)
# =========================
try:
    while True:
        event = attendance_queue.get_nowait()
        name, status = event[0].strip(), event[1].strip()
        now = time.time()
        
        last_time = st.session_state.last_detection_time.get(name, 0)
        
        if status == "OUT" or (now - last_time > COOLDOWN_SECONDS):
            st.session_state.last_detection_time[name] = now
            st.session_state.last_event_time = now
            
            msg_map = {
                "IN": f"{name}, check-in successful",
                "OUT": f"{name}, check-out successful",
                "DONE": f"{name}, attendance already completed today"
            }
            msg = msg_map.get(status, f"Hello {name}")
            st.session_state.current_msg = msg

            # Feedback
            message_placeholder.success(msg)
            st.toast(msg)
            speak_js(msg)

except Empty:
    pass

# =========================
# STABILITY CONTROLLER
# =========================
# Clear status message after 5 seconds
if st.session_state.current_msg and (time.time() - st.session_state.last_event_time > 3):
    st.session_state.current_msg = None
    message_placeholder.empty()

# SAFETY: Only rerun if the camera is stable and playing
# This prevents the "NoneType" crashes during connection negotiation
if webrtc_ctx and webrtc_ctx.state.playing:
    time.sleep(0.5) 
    st.rerun()
else:
    if not webrtc_ctx.state.playing:
        st.info("System Ready. Please click 'Start' to begin.")