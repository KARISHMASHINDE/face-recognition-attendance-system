import numpy as np
import pandas as pd
import cv2
import mysql.connector
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise
from datetime import datetime
import time

# ------------------------------
# MODEL LOAD
# ------------------------------
faceapp = FaceAnalysis(name='buffalo_sc', root='insightface_model')
faceapp.prepare(ctx_id=-1, det_size=(640, 640))

# ------------------------------
# LOAD FACE DATA
# ------------------------------
def retrive_data(db_name):
    conn = mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="karishma123",
        database=db_name
    )
    cursor = conn.cursor()
    cursor.execute("""
        SELECT employee_name, employee_id, face_embedding
        FROM employee_face_data
    """)
    rows = cursor.fetchall()
    conn.close()
    data = []
    for name, emp_id, emb in rows:
        if emb:
            feature = np.frombuffer(emb, dtype=np.float32)
            if feature.shape[0] > 0:
                data.append([name, emp_id, feature])
    return pd.DataFrame(data, columns=['Name', 'EmployeeID', 'facial_features'])

# ------------------------------
# FACE MATCHING
# ------------------------------
def ml_search_algorithm(df, col, test_vec, roles, thresh):
    if df.empty:
        return 'Unknown', 'Unknown'
    try:
        X = np.vstack(df[col].values)
    except:
        return 'Unknown', 'Unknown'
    sim = pairwise.cosine_similarity(X, test_vec.reshape(1, -1)).flatten()
    if np.max(sim) >= thresh:
        idx = np.argmax(sim)
        person = df.iloc[idx]
        return person[roles[0]], person[roles[1]]
    return 'Unknown', 'Unknown'

# ------------------------------
# REALTIME CLASS
# ------------------------------
class RealTimePred:
    def __init__(self, db_name):
        self.db_name = db_name
        self.min_repeat_gap = 5
        self.min_out_gap = 60
        self.last_seen_time = {}
        self.blink_counter = {}
        self.blink_verified = {}

    def can_process(self, emp_id):
        now = datetime.now()
        last = self.last_seen_time.get(emp_id)
        if last is None:
            self.last_seen_time[emp_id] = now
            return True
        if (now - last).total_seconds() >= self.min_repeat_gap:
            self.last_seen_time[emp_id] = now
            return True
        return False

    def is_blinking(self, emp_id, landmarks):
        if landmarks is None or len(landmarks) < 2:
            return False
        left_eye, right_eye = landmarks[0], landmarks[1]
        eye_distance = abs(left_eye[0] - right_eye[0])
        eye_height = abs(left_eye[1] - right_eye[1])
        ratio = eye_height / (eye_distance + 1e-6)
        if emp_id not in self.blink_counter:
            self.blink_counter[emp_id] = 0
        if ratio < 0.15:
            self.blink_counter[emp_id] += 1
        else:
            if self.blink_counter[emp_id] >= 2:
                self.blink_counter[emp_id] = 0
                return True
            self.blink_counter[emp_id] = 0
        return False

    def saveLogs_mysql(self, name, emp_id, time_str):
        conn = mysql.connector.connect(
            host="127.0.0.1", user="root", password="karishma123", database=self.db_name
        )
        cursor = conn.cursor()
        cursor.execute("""
            SELECT log_type, attendance_time
            FROM employee_daily_attendance
            WHERE employee_id=%s AND attendance_date=CURDATE()
            ORDER BY attendance_time DESC LIMIT 1
        """, (emp_id,))
        result = cursor.fetchone()
        
        log_type = "IN"
        duration_str = None

        if result:
            last_type, last_time = result
            if isinstance(last_time, str):
                last_time = datetime.strptime(last_time, "%Y-%m-%d %H:%M:%S")
            elif hasattr(last_time, 'seconds'): # Handle timedelta
                last_time = datetime.combine(datetime.today(), (datetime.min + last_time).time())

            diff_sec = (datetime.now() - last_time).total_seconds()

            if last_type == "IN":
                if diff_sec < self.min_out_gap:
                    conn.close()
                    return None
                log_type = "OUT"
                # Calculate Duration
                hrs, rem = divmod(int(diff_sec), 3600)
                mins = rem // 60
                duration_str = f"{hrs}h {mins}m"
            else:
                conn.close()
                return name, "DONE", None

        cursor.execute("""
            INSERT INTO employee_daily_attendance
            (employee_id, employee_name, attendance_time, attendance_date, log_type)
            VALUES (%s, %s, %s, CURDATE(), %s)
        """, (emp_id, name, time_str, log_type))
        conn.commit()
        conn.close()
        return name, log_type, duration_str

    def face_prediction(self, img, df, col, roles, thresh):
        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        results = faceapp.get(img)
        detected, event = False, None

        if len(results) > 1:
            cv2.putText(img, "Only one person allowed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            return img, False, None

        for r in results:
            detected = True
            bbox, emb, landmarks = r.bbox.astype(int), r.embedding, r.kps
            name, emp_id = ml_search_algorithm(df, col, emb, roles, thresh)

            if name != "Unknown":
                if self.is_blinking(emp_id, landmarks):
                    self.blink_verified[emp_id] = time.time()

                if emp_id in self.blink_verified and (time.time() - self.blink_verified[emp_id] <= 3):
                    cv2.putText(img, "Blink Verified", (bbox[0], bbox[1]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    if self.can_process(emp_id):
                        event = self.saveLogs_mysql(name, emp_id, time_str)
                else:
                    cv2.putText(img, "Please Blink", (bbox[0], bbox[1]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            color = (0,255,0) if name != "Unknown" else (0,0,255)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(img, name, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return img, detected, event