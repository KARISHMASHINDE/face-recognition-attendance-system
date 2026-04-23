import streamlit as st
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import time

from db_connection import get_mysql_connection

st.set_page_config(page_title="Register Employee", page_icon="📝")

# -------------------------
# LOGIN + DB CHECK 🔐
# -------------------------
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first 🔐")
    st.stop()

if "db_name" not in st.session_state:
    st.error("❌ Database not selected. Please login again.")
    st.stop()
# -------------------------
# FACE MODEL
# -------------------------
faceapp = FaceAnalysis(name='buffalo_sc', root='insightface_model')
faceapp.prepare(ctx_id=-1, det_size=(640,640))

# -------------------------
# SESSION STATE
# -------------------------
if "embeddings" not in st.session_state:
    st.session_state.embeddings = []

if "last_face_time" not in st.session_state:
    st.session_state.last_face_time = time.time()

# ✅ NEW (AUTO CAPTURE CONTROL)
if "last_capture_time" not in st.session_state:
    st.session_state.last_capture_time = 0

# -------------------------
# CONFIG
# -------------------------
CAPTURE_INTERVAL = 1.5   # seconds
MAX_SAMPLES = 3

# -------------------------
# TITLE
# -------------------------
st.title("Employee Registration")

# -------------------------
# INPUT
# -------------------------
name = st.text_input("Employee Name").strip()
emp_id = st.text_input("Employee ID").strip()

st.info("📷 Keep your face steady. Samples will be captured automatically.")

st.write("Samples collected:", len(st.session_state.embeddings))

# -------------------------
# CHECK DUPLICATE ID
# -------------------------
def check_employee_id_exists(emp_id):
    conn = get_mysql_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT 1 FROM employee_face_data WHERE employee_id=%s", (emp_id,))
    result = cursor.fetchone()

    conn.close()
    return result is not None

# -------------------------
# LOAD EXISTING EMBEDDINGS
# -------------------------
def load_all_embeddings():
    conn = get_mysql_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT face_embedding FROM employee_face_data")
    rows = cursor.fetchall()

    conn.close()

    embeddings = []
    for (emb,) in rows:
        embeddings.append(np.frombuffer(emb, dtype=np.float32))

    return embeddings

existing_embeddings = load_all_embeddings()

# -------------------------
# FACE MATCH CHECK
# -------------------------
def is_face_already_registered(new_embedding, threshold=0.6):

    if not existing_embeddings:
        return False

    sims = cosine_similarity(
        np.array(existing_embeddings),
        new_embedding.reshape(1, -1)
    ).flatten()

    return np.max(sims) > threshold

# -------------------------
# CAMERA (AUTO CAPTURE)
# -------------------------
camera = st.camera_input("Capture Face")

if camera is not None:

    file_bytes = np.asarray(bytearray(camera.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    results = faceapp.get(img)

    if len(results) > 0:

        st.session_state.last_face_time = time.time()

        embedding = results[0].embedding

        # 🚨 DUPLICATE FACE CHECK
        if is_face_already_registered(embedding):
            st.error("⚠️ This face is already registered!")

        else:
            current_time = time.time()

            # ✅ AUTO CAPTURE LOGIC
            if (
                len(st.session_state.embeddings) < MAX_SAMPLES and
                current_time - st.session_state.last_capture_time > CAPTURE_INTERVAL
            ):
                st.session_state.embeddings.append(embedding)
                st.session_state.last_capture_time = current_time

                st.success(f"✅ Sample {len(st.session_state.embeddings)} captured")

    else:
        st.warning("No face detected")

# -------------------------
# PROGRESS BAR
# -------------------------
progress = len(st.session_state.embeddings) / MAX_SAMPLES
st.progress(progress)

if len(st.session_state.embeddings) == MAX_SAMPLES:
    st.success("🎉 All samples captured automatically!")

# -------------------------
# AUTO STOP CAMERA (10 sec)
# -------------------------
if time.time() - st.session_state.last_face_time > 10:
    st.warning("⚠️ No face detected for 10 seconds. Please restart capture.")

# -------------------------
# REGISTER
# -------------------------
if st.button("Register"):

    # VALIDATION
    if not name:
        st.error("Enter employee name")
        st.stop()

    if not emp_id:
        st.error("Enter employee ID")
        st.stop()

    if check_employee_id_exists(emp_id):
        st.error("⚠️ Employee ID already exists")
        st.stop()

    if len(st.session_state.embeddings) < MAX_SAMPLES:
        st.error(f"Capture at least {MAX_SAMPLES} samples")
        st.stop()

    # MEAN EMBEDDING
    x = np.array(st.session_state.embeddings)
    mean_embedding = x.mean(axis=0).astype(np.float32)

    # FINAL FACE CHECK
    if is_face_already_registered(mean_embedding):
        st.error("⚠️ This face is already registered (final check)")
        st.stop()

    # SAVE
    conn = get_mysql_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO employee_face_data
        (employee_id, employee_name, face_embedding)
        VALUES (%s,%s,%s)
    """, (emp_id, name, mean_embedding.tobytes()))

    conn.commit()
    conn.close()

    st.success("✅ Employee Registered Successfully")

    # RESET
    st.session_state.embeddings = []
    st.session_state.last_capture_time = 0