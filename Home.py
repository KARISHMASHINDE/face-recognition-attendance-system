import streamlit as st
import mysql.connector

# ------------------------------
# Page Configuration
# ------------------------------
if __name__ == "__main__":
    st.set_page_config(
        page_title='Inmate Attendance System',
        layout='wide'
    )

# ------------------------------
# Header
# ------------------------------
st.header('Inmate Attendance System using Face Recognition')

# ------------------------------
# MySQL Connection
# ------------------------------
def connect_mysql():
    try:
        conn = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            password="Zeeshan@123",
            database="loginpath"
        )
        return conn
    except mysql.connector.Error as err:
        st.error(f"MySQL Connection Error: {err}")
        return None


# ------------------------------
# Load System
# ------------------------------
with st.spinner("Loading Models and Connecting to MySQL db..."):

    import face_rec

    loginpath_db = connect_mysql()

# ------------------------------
# Status
# ------------------------------
st.success('Model loaded successfully')

if loginpath_db:
    st.success('MySQL db successfully connected')
else:
    st.error('Database connection failed')