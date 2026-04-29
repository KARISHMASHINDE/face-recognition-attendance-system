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

# Global pool to prevent connection overhead
db_pool = None

def get_db_pool(db_name):
    global db_pool
    if db_pool is None:
        db_pool = mysql.connector.pooling.MySQLConnectionPool(
            pool_name="attendance_pool",
            pool_size=10,
            database=db_name,
            **db_config
        )
    return db_pool

# ==========================================
# 2. MODEL LOAD
# ==========================================
# Pre-load the model globally
faceapp = FaceAnalysis(name='buffalo_sc', root='insightface_model')
faceapp.prepare(ctx_id=-1, det_size=(640, 640))

def retrive_data(db_name):
    pool = get_db_pool(db_name)
    conn = pool.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT employee_name, employee_id, face_embedding FROM employee_face_data")
    rows = cursor.fetchall()
    conn.close() # Return to pool
    
    data = []
    for name, emp_id, emb in rows:
        if emb:
            feature = np.frombuffer(emb, dtype=np.float32)
            data.append([name, emp_id, feature])
    
    df = pd.DataFrame(data, columns=['Name', 'EmployeeID', 'facial_features'])
    # Pre-stack features for lightning-fast search
    if not df.empty:
        df['stacked_features'] = list(np.vstack(df['facial_features'].values))
    return df

class RealTimePred:
    def __init__(self, db_name):
        self.db_name = db_name
        self.min_repeat_gap = 5 # Seconds between same-person scans
        self.min_out_gap = 60    # Minimum time between IN and OUT
        self.last_seen_time = {}
        self.blink_counter = {}
        self.blink_verified = {}
        self.db_pool = get_db_pool(db_name)

    def is_blinking(self, emp_id, landmarks):
        if landmarks is None or len(landmarks) < 2:
            return False
        left_eye, right_eye = landmarks[0], landmarks[1]
        
        # Calculate Eye Aspect Ratio (Simplified for 5-point KPS)
        eye_distance = np.linalg.norm(left_eye - right_eye)
        # Vertical height is harder with 5-points; we use a fixed threshold on eye height movement
        # Here we use the ratio logic you provided but optimized
        eye_height = abs(left_eye[1] - right_eye[1]) 
        ratio = eye_height / (eye_distance + 1e-6)

        if emp_id not in self.blink_counter: self.blink_counter[emp_id] = 0

        if ratio < 0.15:
            self.blink_counter[emp_id] += 1
            return False
        else:
            if self.blink_counter[emp_id] >= 1: # Lowered to 1 for faster detection in WebRTC
                self.blink_counter[emp_id] = 0
                return True
            self.blink_counter[emp_id] = 0
        return False

    def saveLogs_mysql(self, name, emp_id, time_str):
        conn = self.db_pool.get_connection()
        cursor = conn.cursor()
        try:
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
                # Handle various time formats from MySQL
                if isinstance(last_time, str):
                    last_time = datetime.strptime(last_time, "%Y-%m-%d %H:%M:%S")
                elif hasattr(last_time, 'seconds'):
                    last_time = datetime.combine(datetime.today(), (datetime.min + last_time).time())

                diff_sec = (datetime.now() - last_time).total_seconds()

                if last_type == "IN":
                    if diff_sec < self.min_out_gap: return None
                    log_type = "OUT"
                    hrs, rem = divmod(int(diff_sec), 3600)
                    mins = rem // 60
                    duration_str = f"{hrs}h {mins}m"
                else:
                    return name, "DONE", None

            cursor.execute("""
                INSERT INTO employee_daily_attendance 
                (employee_id, employee_name, attendance_time, attendance_date, log_type)
                VALUES (%s, %s, %s, CURDATE(), %s)
            """, (emp_id, name, time_str, log_type))
            conn.commit()
            return name, log_type, duration_str
        finally:
            conn.close()

    def face_prediction(self, img, df, col, roles, thresh):
        if df.empty: return img, False, None
        
        results = faceapp.get(img)
        if not results: return img, False, None

        # PRODUCTION FIX: Process only the largest face (the person in front)
        main_face = max(results, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
        
        bbox, emb, landmarks = main_face.bbox.astype(int), main_face.embedding, main_face.kps
        
        # Optimized Matrix Search
        X = np.array(df['stacked_features'].tolist())
        sim = pairwise.cosine_similarity(X, emb.reshape(1, -1)).flatten()
        
        name, emp_id = "Unknown", "Unknown"
        if np.max(sim) >= thresh:
            idx = np.argmax(sim)
            name, emp_id = df.iloc[idx]['Name'], df.iloc[idx]['EmployeeID']

        event = None
        if name != "Unknown":
            # Check Blink
            if self.is_blinking(emp_id, landmarks):
                self.blink_verified[emp_id] = time.time()

            # Process attendance if blink verified in last 3 seconds
            if emp_id in self.blink_verified and (time.time() - self.blink_verified[emp_id] <= 3):
                cv2.putText(img, "VERIFIED", (bbox[0], bbox[1]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                now_ts = datetime.now()
                # Cooldown check
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