import streamlit as st
import face_rec
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import streamlit.components.v1 as components
import av
import time
from queue import Queue, Empty

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

# --- AUTH & AUDIO ---
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

# --- LOAD RESOURCES ---
@st.cache_data(ttl=600)
def load_face_db(db_name):
    return face_rec.retrive_data(db_name)

@st.cache_resource
def load_model(db_name):
    return face_rec.RealTimePred(db_name)

face_db = load_face_db(st.session_state.db_name)
realtimepred = load_model(st.session_state.db_name)

# --- UI ELEMENTS ---
st.sidebar.markdown(f"### 👤 {st.session_state.get('user_name')}\n**DB:** `{st.session_state.get('db_name')}`")
if st.sidebar.button("🚪 Logout"):
    st.session_state.clear()
    st.rerun()

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
            window.speechSynthesis.speak(msg);
        </script>
    """, height=0)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    try:
        pred_img, detected, event = realtimepred.face_prediction(img, face_db, "facial_features", ["Name", "EmployeeID"], 0.65)
        if event: attendance_queue.put(event)
    except:
        pred_img = img
    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")

st.title("📷 Face Attendance System")

webrtc_ctx = webrtc_streamer(
    key="attendance_camera",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": True, 
        "audio": False  
    },
    async_processing=True
)

# --- EVENT PROCESSING ---
message_placeholder = st.empty()
try:
    while True:
        event = attendance_queue.get_nowait()
        name, status = event[0], event[1]
        duration = event[2] if len(event) > 2 else ""
        
        if status == "IN": msg = f"{name}, Check-in successful"
        elif status == "OUT": msg = f"{name}, Check-out successful. Your work time is: {duration}"
        elif status == "DONE": msg = f"{name}, Attendance already marked"
        else: msg = f"Hello {name}"

        message_placeholder.success(msg)
        st.toast(msg)
        speak_js(msg)
        time.sleep(2) # Prevent speech overlap
except Empty:
    pass

if webrtc_ctx and webrtc_ctx.state.playing:
    time.sleep(0.1)
    st.rerun()