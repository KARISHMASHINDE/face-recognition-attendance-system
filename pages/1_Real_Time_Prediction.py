import streamlit as st
import face_rec
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import streamlit.components.v1 as components
import av
import time
from queue import Queue, Empty

# ==========================================
# 1. PAGE CONFIG & STYLING
# ==========================================
st.set_page_config(page_title="Attendance System", page_icon="📷", layout="wide")

st.markdown("""
<style>
    iframe[title="streamlit_components.v1.html"] { display: none; } 
    .stAlert { margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_attendance_queue():
    return Queue()

attendance_queue = get_attendance_queue()

# --- AUTH CHECK ---
if not st.session_state.get("logged_in"):
    st.warning("Please login first 🔐")
    st.stop()

if st.session_state.get("user_role") not in ["system", "super"]:
    st.error("⛔ Unauthorized Access")
    st.stop()

# ==========================================
# 2. AUDIO UNLOCK (Production Guard)
# ==========================================
if "audio_unlocked" not in st.session_state:
    st.session_state.audio_unlocked = False

if not st.session_state.audio_unlocked:
    st.title("📷 System Activation")
    st.info("Browser security requires a manual start to enable Voice Feedback.")
    if st.button("🚀 Unlock Audio & Start Monitoring"):
        st.session_state.audio_unlocked = True
        st.rerun()
    st.stop()

# ==========================================
# 3. SESSION STATE & MODEL LOADING
# ==========================================
st.session_state.setdefault("last_detection_time", {})
st.session_state.setdefault("last_event_time", 0)
st.session_state.setdefault("current_msg", None)
message_placeholder = st.empty()
COOLDOWN_SECONDS = 180

@st.cache_data(ttl=300)
def load_face_db(db_name):
    return face_rec.retrive_data(db_name)

@st.cache_resource
def load_model(db_name):
    return face_rec.RealTimePred(db_name)

face_db = load_face_db(st.session_state.db_name)
realtimepred = load_model(st.session_state.db_name)

# ==========================================
# 4. FIXED SPEECH FUNCTION (No 'key' Error)
# ==========================================
def speak_js(text):
    """Compatible with all Streamlit versions. Uses a div ID to force re-render."""
    if not text: return
    clean_text = text.replace("'", "\\'").replace("\n", " ")
    
    # Using a timestamp inside the HTML string to force the component to update
    ts = time.time()
    
    components.html(
        f"""
        <div id="tts-{ts}" style="display:none;"></div>
        <script>
            (function() {{
                window.speechSynthesis.cancel(); 
                const msg = new SpeechSynthesisUtterance('{clean_text}');
                msg.lang = 'en-US';
                msg.rate = 1.0;
                window.speechSynthesis.speak(msg);
            }})();
        </script>
        """, 
        height=0
    )

# ==========================================
# 5. VIDEO CALLBACK
# ==========================================
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    try:
        pred_img, detected, event = realtimepred.face_prediction(
            img, face_db, "facial_features", ["Name", "EmployeeID"], 0.7
        )
        if event:
            attendance_queue.put(event)
    except Exception:
        pred_img = img
    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")

# ==========================================
# 6. UI & WEBRTC
# ==========================================
st.sidebar.markdown(f"### 👤 User Info\n**Name:** {st.session_state.get('user_name')} \n**DB:** `{st.session_state.get('db_name')}` \n**Role:** {st.session_state.get('user_role')}")
if st.sidebar.button("🚪 Logout"):
    st.session_state.clear()
    st.rerun()

st.title("📷 Real-Time Face Attendance System")

webrtc_ctx = webrtc_streamer(
    key="attendance_camera",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

# ==========================================
# 7. QUEUE PROCESSING
# ==========================================
try:
    while True:
        event = attendance_queue.get_nowait()
        name, status = event[0].strip(), event[1].strip()
        duration = event[2] if len(event) > 2 else None
        now = time.time()
        
        last_time = st.session_state.last_detection_time.get(name, 0)
        
        # Logic: Always process OUT, process IN only if cooldown passed
        if status == "OUT" or (now - last_time > COOLDOWN_SECONDS):
            st.session_state.last_detection_time[name] = now
            st.session_state.last_event_time = now
            
            if status == "IN":
                msg = f"{name}, check-in successful"
            elif status == "OUT":
                msg = f"{name}, check-out successful. Total work time: {duration}"
            elif status == "DONE":
                msg = f"{name}, attendance already completed today"
            else:
                msg = f"Hello {name}"

            st.session_state.current_msg = msg
            message_placeholder.success(msg)
            st.toast(msg)
            speak_js(msg)
except Empty:
    pass

# Clear feedback after 4 seconds
if st.session_state.current_msg and (time.time() - st.session_state.last_event_time > 4):
    st.session_state.current_msg = None
    message_placeholder.empty()

# Keep live loop running
if webrtc_ctx and webrtc_ctx.state.playing:
    time.sleep(0.1) # Faster response time
    st.rerun()
else:
    if not webrtc_ctx.state.playing:
        st.info("System Ready. Please click 'Start' to begin camera stream.")