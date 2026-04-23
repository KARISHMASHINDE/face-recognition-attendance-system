import streamlit as st
import face_rec
from streamlit_webrtc import webrtc_streamer
import av
import time

st.set_page_config(page_title="Attendance", page_icon="📷")

# ------------------------------
# LOGIN
# ------------------------------
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first 🔐")
    st.stop()

# ------------------------------
# SIDEBAR
# ------------------------------
st.sidebar.markdown(f"""
### 👤 Logged In
**Name:** {st.session_state.get('user_name','')}  
**DB:** `{st.session_state.get('db_name','')}`  
**Role:** {st.session_state.get('user_role','')}
""")

if st.sidebar.button("🚪 Logout"):
    st.session_state.clear()
    st.rerun()

# ------------------------------
# ROLE
# ------------------------------
if st.session_state.get("user_role") not in ["system", "super"]:
    st.error("⛔ Unauthorized Access")
    st.stop()

st.subheader("Real-Time Attendance System")

message_placeholder = st.empty()

# ------------------------------
# LOAD DB
# ------------------------------
if "face_db" not in st.session_state:
    st.session_state.face_db = face_rec.retrive_data(st.session_state.db_name)

face_db = st.session_state.face_db

if face_db.empty:
    st.warning("⚠️ No registered users found.")

# ------------------------------
# INIT MODEL
# ------------------------------
if "realtimepred" not in st.session_state:
    st.session_state.realtimepred = face_rec.RealTimePred(
        st.session_state.db_name
    )

realtimepred = st.session_state.realtimepred

# ------------------------------
# STATES
# ------------------------------
if "camera_on" not in st.session_state:
    st.session_state.camera_on = True

if "last_event" not in st.session_state:
    st.session_state.last_event = None

if "event_time" not in st.session_state:
    st.session_state.event_time = None


# ------------------------------
# CALLBACK (NO session_state here!)
# ------------------------------
def video_frame_callback(frame):

    img = frame.to_ndarray(format="bgr24")

    try:
        pred_img, detected, event = realtimepred.face_prediction(
            img,
            face_db,
            "facial_features",
            ["Name", "EmployeeID"],
            0.7
        )

        # ✅ STORE INSIDE OBJECT (SAFE)
        if event:
            realtimepred.latest_event = event

    except Exception as e:
        #print("[ERROR]", e)
        pred_img = img

    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")


# ------------------------------
# CAMERA
# ------------------------------
webrtc_ctx = None

if st.session_state.camera_on:
    webrtc_ctx = webrtc_streamer(
        key="cam",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    # ✅ CHECK EVENT IN MAIN THREAD
    if hasattr(realtimepred, "latest_event") and realtimepred.latest_event:

        st.session_state.last_event = realtimepred.latest_event
        st.session_state.event_time = time.time()

        realtimepred.latest_event = None

        # 🔥 STOP CAMERA SAFELY
        if webrtc_ctx and webrtc_ctx.state.playing:
            webrtc_ctx.stop()

        st.session_state.camera_on = False
        st.rerun()

else:
    if st.button("▶️ Restart Camera"):
        st.session_state.camera_on = True
        st.session_state.last_event = None
        st.session_state.event_time = None
        st.rerun()


# ------------------------------
# MESSAGE DISPLAY
# ------------------------------
if st.session_state.last_event:

    name, log_type = st.session_state.last_event

    if log_type == "IN":
        message_placeholder.success(f"✅ {name} CHECK-IN successful")

    elif log_type == "OUT":
        message_placeholder.warning(f"👋 {name} CHECK-OUT successful")

    elif log_type == "DONE":
        message_placeholder.info(f"🛑 {name}, you are done for today")

    # Auto clear after 3 sec
    if st.session_state.event_time:
        if time.time() - st.session_state.event_time > 3:
            st.session_state.last_event = None
            st.session_state.event_time = None