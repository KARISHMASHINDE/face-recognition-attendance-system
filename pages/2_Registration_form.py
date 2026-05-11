import streamlit as st
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import time
import logging

# Assuming get_mysql_connection is handled in your db_connection file
from db_connection import get_mysql_connection

# 1. LOGGING CONFIGURATION
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. APP CONFIGURATION
st.set_page_config(page_title="Biometric Registration", page_icon="📝", layout="centered")

# -------------------------
# SECURITY & AUTH CHECK
# -------------------------
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first 🔐")
    st.stop()

# -------------------------
# MODELS & CACHING
# -------------------------
@st.cache_resource(show_spinner="Loading Face Recognition Models...")
def load_face_model():
    try:
        # ctx_id=-1 uses CPU. Set to 0 if you have a GPU/CUDA configured.
        faceapp = FaceAnalysis(name='buffalo_sc', root='insightface_model')
        faceapp.prepare(ctx_id=-1, det_size=(640, 640))
        return faceapp
    except Exception as e:
        logger.error(f"Failed to load InsightFace model: {e}")
        st.error("Critical Error: Face Analysis model could not be initialized.")
        return None

faceapp = load_face_model()

# -------------------------
# DATABASE UTILITIES
# -------------------------
def fetch_employee_from_db(emp_id):
    """Securely fetches employee details using parameterized queries."""
    conn = None
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor(dictionary=True)
        # Using parameterized query to prevent SQL Injection
        query = "SELECT employee_name, designation FROM employee_salary WHERE emp_id = %s LIMIT 1"
        cursor.execute(query, (emp_id,))
        return cursor.fetchone()
    except Exception as e:
        logger.error(f"DB Fetch Error for ID {emp_id}: {e}")
        return None
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

def check_existing_biometrics(emp_id):
    conn = None
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM employee_face_data WHERE employee_id=%s", (emp_id,))
        return cursor.fetchone() is not None
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

@st.cache_data(ttl=60) # Cache for 60 seconds to reduce DB load
def get_all_registered_embeddings():
    conn = None
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT face_embedding FROM employee_face_data")
        rows = cursor.fetchall()
        return [np.frombuffer(row[0], dtype=np.float32) for row in rows]
    except Exception:
        return []
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

# -------------------------
# SESSION STATE MANAGEMENT
# -------------------------
if "embeddings" not in st.session_state:
    st.session_state.embeddings = []
if "last_capture_time" not in st.session_state:
    st.session_state.last_capture_time = 0

# -------------------------
# UI - MAIN FORM
# -------------------------
st.title("🛡️ Biometric Enrollment")
st.markdown("Register employee face data for the attendance system.")

input_emp_id = st.text_input("Enter Employee ID (emp_id)", help="Input the unique ID from employee_salary table").strip()

# Logic flow control
valid_id = False
employee_data = None

if input_emp_id:
    employee_data = fetch_employee_from_db(input_emp_id)
    
    if employee_data:
        st.success(f"✅ **Identity Verified:** {employee_data['employee_name']} ({employee_data['designation']})")
        valid_id = True
    else:
        st.error(f"🔍 ID '{input_emp_id}' not found in the master database. Please check and try again.")
        st.session_state.embeddings = [] # Clear data if ID is changed to invalid

# -------------------------
# REGISTRATION WORKFLOW
# -------------------------
if valid_id:
    # Read-only fields for confirmation
    col1, col2 = st.columns(2)
    col1.text_input("Name", value=employee_data['employee_name'], disabled=True)
    col2.text_input("Designation", value=employee_data['designation'], disabled=True)

    st.divider()
    
    # 1. Camera Input
    st.info("💡 **Instructions:** Look directly at the camera. 3 samples will be taken automatically.")
    camera = st.camera_input("Biometric Scan")

    if camera is not None:
        file_bytes = np.asarray(bytearray(camera.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        results = faceapp.get(img)

        if len(results) > 0:
            current_embedding = results[0].embedding
            
            # 2. Duplicate Face Prevention
            existing_faces = get_all_registered_embeddings()
            if existing_faces:
                sims = cosine_similarity(np.array(existing_faces), current_embedding.reshape(1, -1)).flatten()
                if np.max(sims) > 0.65: # Threshold for similarity
                    st.error("🚫 **Duplicate Found:** This face is already registered in the system.")
                    st.stop()

            # 3. Automatic Sample Collection
            if len(st.session_state.embeddings) < 3:
                now = time.time()
                if now - st.session_state.last_capture_time > 1.5: # 1.5s interval
                    st.session_state.embeddings.append(current_embedding)
                    st.session_state.last_capture_time = now
                    st.toast(f"Captured Sample {len(st.session_state.embeddings)}/3", icon="📸")
        else:
            st.warning("⚠️ No face detected. Please adjust lighting or position.")

    # 4. Progress Tracking
    progress_val = len(st.session_state.embeddings) / 3
    st.progress(progress_val)
    st.write(f"Enrollment Progress: {len(st.session_state.embeddings)} / 3 Samples")

    # 5. Final Save Action
    if st.button("Finalize Registration", type="primary", use_container_width=True):
        if check_existing_biometrics(input_emp_id):
            st.error("This Employee ID already has registered face data.")
        elif len(st.session_state.embeddings) < 3:
            st.error("Insufficient samples. Please allow the camera to capture 3 frames.")
        else:
            try:
                # Calculate the average (mean) embedding for higher accuracy
                final_embedding = np.array(st.session_state.embeddings).mean(axis=0).astype(np.float32)
                
                conn = get_mysql_connection()
                cursor = conn.cursor()
                sql = "INSERT INTO employee_face_data (employee_id, employee_name, face_embedding) VALUES (%s, %s, %s)"
                cursor.execute(sql, (input_emp_id, employee_data['employee_name'], final_embedding.tobytes()))
                conn.commit()
                
                st.balloons()
                st.success(f"🎉 Registration complete for {employee_data['employee_name']}!")
                
                # Cleanup session
                st.session_state.embeddings = []
                st.session_state.last_capture_time = 0
                time.sleep(2)
                st.rerun() # Refresh page for next registration
                
            except Exception as e:
                logger.error(f"Failed to save biometrics: {e}")
                st.error("Database Save Error. Contact administrator.")
            finally:
                if conn and conn.is_connected():
                    cursor.close()
                    conn.close()
else:
    st.info("👋 Awaiting valid Employee ID to begin enrollment.")