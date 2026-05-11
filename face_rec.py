import streamlit as st
import numpy as np
import pandas as pd
import cv2
import mysql.connector
from mysql.connector import pooling
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise
from datetime import datetime
import time

# ==========================================
# 1. DATABASE CONNECTION POOLING
# ==========================================
db_config = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "karishma123",
}

db_pool = None

def get_db_pool(db_name):
    global db_pool
    try:
        # If pool exists but for a different database, we must reset it or handle it
        if db_pool is None:
            db_pool = mysql.connector.pooling.MySQLConnectionPool(
                pool_name="attendance_pool",
                pool_size=10, # Increase this if many users are logging in
                database=db_name,
                **db_config
            )
        return db_pool
    except mysql.connector.Error as e:
        st.error(f"Failed to connect to MySQL: {e}")
        return None
# ==========================================
# 2. MODEL LOAD
# ==========================================
faceapp = FaceAnalysis(name='buffalo_sc', root='insightface_model')
faceapp.prepare(ctx_id=-1, det_size=(640, 640))

def retrive_data(db_name):
    pool = get_db_pool(db_name)
    conn = pool.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT employee_name, employee_id, face_embedding FROM employee_face_data")
    rows = cursor.fetchall()
    cursor.close()
    conn.close() 
    
    data = []
    for name, emp_id, emb in rows:
        if emb:
            feature = np.frombuffer(emb, dtype=np.float32)
            data.append([name, emp_id, feature])
    
    df = pd.DataFrame(data, columns=['Name', 'EmployeeID', 'facial_features'])
    if not df.empty:
        df['stacked_features'] = list(np.vstack(df['facial_features'].values))
    return df

class RealTimePred:
    def __init__(self, db_name):
        self.db_name = db_name
        self.min_repeat_gap = 5 
        self.min_out_gap = 60   # Shortened for testing/flexibility
        self.last_seen_time = {}
        self.blink_counter = {}
        self.blink_verified = {}
        self.db_pool = get_db_pool(db_name)

    def is_blinking(self, emp_id, landmarks):
        if landmarks is None or len(landmarks) < 2:
            return False
        left_eye, right_eye = landmarks[0], landmarks[1]
        eye_distance = np.linalg.norm(left_eye - right_eye)
        eye_height = abs(left_eye[1] - right_eye[1]) 
        ratio = eye_height / (eye_distance + 1e-6)

        if emp_id not in self.blink_counter: self.blink_counter[emp_id] = 0
        if ratio < 0.15:
            self.blink_counter[emp_id] += 1
            return False
        else:
            if self.blink_counter[emp_id] >= 1:
                self.blink_counter[emp_id] = 0
                return True
            self.blink_counter[emp_id] = 0
        return False

    def saveLogs_mysql(self, name, emp_id, time_str):
        conn = self.db_pool.get_connection()
        cursor = conn.cursor()
        try:
            # Get the very last log for today to determine the next step
            cursor.execute("""
                SELECT log_type, attendance_time FROM employee_daily_attendance 
                WHERE employee_id=%s AND attendance_date=CURDATE() 
                ORDER BY attendance_time DESC LIMIT 1
            """, (emp_id,))
            result = cursor.fetchone()
            
            log_type = "IN" 
            duration_str = None

            if result:
                last_type, last_time = result
                # Consistency in time formatting
                if isinstance(last_time, str):
                    last_time = datetime.strptime(last_time, "%Y-%m-%d %H:%M:%S")
                elif hasattr(last_time, 'total_seconds'): # Handle timedelta
                    last_time = datetime.combine(datetime.today(), (datetime.min + last_time).time())

                diff_sec = (datetime.now() - last_time).total_seconds()
                if diff_sec < self.min_out_gap: return None 

                # Toggle Logic: If last was IN, now OUT. If last was OUT, now IN.
                if last_type == "IN":
                    log_type = "OUT"
                    
                    # Calculate cumulative time for the day
                    cursor.execute("""
                        SELECT log_type, attendance_time FROM employee_daily_attendance 
                        WHERE employee_id=%s AND attendance_date=CURDATE() 
                        ORDER BY attendance_time ASC
                    """, (emp_id,))
                    all_logs = cursor.fetchall()
                    
                    total_seconds = 0
                    temp_in = None
                    for l_type, l_time in all_logs:
                        if l_type == "IN":
                            temp_in = l_time
                        elif l_type == "OUT" and temp_in:
                            total_seconds += (l_time - temp_in).total_seconds()
                            temp_in = None
                    
                    # Add current session
                    total_seconds += diff_sec
                    hrs, rem = divmod(int(total_seconds), 3600)
                    mins = rem // 60
                    duration_str = f"{hrs}h {mins}m"
                else:
                    log_type = "IN"

            cursor.execute("""
                INSERT INTO employee_daily_attendance 
                (employee_id, employee_name, attendance_time, attendance_date, log_type)
                VALUES (%s, %s, %s, CURDATE(), %s)
            """, (emp_id, name, time_str, log_type))
            conn.commit()
            return name, log_type, duration_str
        finally:
            cursor.close()
            conn.close()

    def face_prediction(self, img, df, col, roles, thresh):
        if df.empty: return img, False, None
        results = faceapp.get(img)
        if not results: return img, False, None

        main_face = max(results, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
        bbox, emb, landmarks = main_face.bbox.astype(int), main_face.embedding, main_face.kps
        
        X = np.array(df['stacked_features'].tolist())
        sim = pairwise.cosine_similarity(X, emb.reshape(1, -1)).flatten()
        
        name, emp_id = "Unknown", "Unknown"
        if np.max(sim) >= thresh:
            idx = np.argmax(sim)
            name, emp_id = df.iloc[idx]['Name'], df.iloc[idx]['EmployeeID']

        event = None
        if name != "Unknown":
            if self.is_blinking(emp_id, landmarks):
                self.blink_verified[emp_id] = time.time()

            if emp_id in self.blink_verified and (time.time() - self.blink_verified[emp_id] <= 3):
                cv2.putText(img, "VERIFIED", (bbox[0], bbox[1]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                now_ts = datetime.now()
                last_seen = self.last_seen_time.get(emp_id, 0)
                if (time.time() - last_seen) > self.min_repeat_gap:
                    event = self.saveLogs_mysql(name, emp_id, now_ts.strftime("%Y-%m-%d %H:%M:%S"))
                    self.last_seen_time[emp_id] = time.time()
            else:
                cv2.putText(img, "PLEASE BLINK", (bbox[0], bbox[1]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        color = (0,255,0) if name != "Unknown" else (0,0,255)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(img, f"{name} ({np.max(sim):.2f})", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return img, True, event