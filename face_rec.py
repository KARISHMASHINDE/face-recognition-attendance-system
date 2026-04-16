import numpy as np
import pandas as pd
import cv2
import mysql.connector
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise
from datetime import datetime

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
# REALTIME CLASS (FIXED)
# ------------------------------
class RealTimePred:

    def __init__(self, db_name):
        self.db_name = db_name

        # controls
        self.min_repeat_gap = 5     # avoid frame spam
        self.min_out_gap = 60       # allow OUT only after IN delay

        # tracking
        self.last_seen_time = {}

    # ------------------------------
    # PREVENT FRAME SPAM
    # ------------------------------
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

    # ------------------------------
    # SAVE LOGIC (IN / OUT / DONE)
    # ------------------------------
    def saveLogs_mysql(self, name, emp_id, time_str):

        conn = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            password="karishma123",
            database=self.db_name
        )

        cursor = conn.cursor()

        # Get last entry
        cursor.execute("""
            SELECT log_type, attendance_time
            FROM employee_daily_attendance
            WHERE employee_id=%s AND attendance_date=CURDATE()
            ORDER BY attendance_time DESC LIMIT 1
        """, (emp_id,))

        result = cursor.fetchone()

        # FIRST TIME → IN
        if result is None:
            log_type = "IN"

        else:
            last_type, last_time = result

            if isinstance(last_time, str):
                last_time = datetime.strptime(last_time, "%Y-%m-%d %H:%M:%S")

            diff = (datetime.now() - last_time).total_seconds()

            # IN → OUT
            if last_type == "IN":
                if diff < self.min_out_gap:
                    conn.close()
                    return None
                log_type = "OUT"

            # OUT → DONE
            else:
                conn.close()
                return name, "DONE"

        print(f"[DB SAVE] {name} | {emp_id} | {log_type}")

        cursor.execute("""
            INSERT INTO employee_daily_attendance
            (employee_id, employee_name, attendance_time, attendance_date, log_type)
            VALUES (%s, %s, %s, CURDATE(), %s)
        """, (emp_id, name, time_str, log_type))

        conn.commit()
        conn.close()

        return name, log_type

    # ------------------------------
    # FACE PREDICTION
    # ------------------------------
    def face_prediction(self, img, df, col, roles, thresh):

        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d %H:%M:%S")

        results = faceapp.get(img)

        detected = False
        event = None

        for r in results:

            detected = True

            bbox = r.bbox.astype(int)
            emb = r.embedding

            name, emp_id = ml_search_algorithm(df, col, emb, roles, thresh)

            if name != "Unknown":

                if self.can_process(emp_id):

                    result = self.saveLogs_mysql(name, emp_id, time_str)

                    if result:
                        event = result

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(img, name, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return img, detected, event