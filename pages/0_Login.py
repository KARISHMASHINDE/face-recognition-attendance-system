import streamlit as st
import mysql.connector
import time

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Login", page_icon="🔐")

# -------------------------
# SESSION DEFAULTS
# -------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "login_attempts" not in st.session_state:
    st.session_state.login_attempts = 0

if "last_attempt_time" not in st.session_state:
    st.session_state.last_attempt_time = 0

SESSION_TIMEOUT = 60 * 30
MAX_ATTEMPTS = 5
LOCK_TIME = 60

# -------------------------
# AUTO REDIRECT
# -------------------------
if st.session_state.logged_in:
    st.switch_page("pages/1_Real_Time_Prediction.py")

# -------------------------
# CENTER ALIGN UI
# -------------------------
col1, col2, col3 = st.columns([1,2,1])

with col2:

    st.markdown("## Ashram Login")

    # SUCCESS MESSAGE
    if "login_msg" in st.session_state:
        st.success(st.session_state.login_msg)
        del st.session_state.login_msg

    # INPUTS
    ref = st.text_input("Ashram Code")
    role = st.selectbox("Select Role", ["attendance"])
    password = st.text_input("Password", type="password")

    # -------------------------
    # LOGIN FUNCTION
    # -------------------------
    def login_user(ref, role, password):
        try:
            conn = mysql.connector.connect(
                host="127.0.0.1",
                user="root",
                password="karishma123",
                database="loginpath"
            )

            cursor = conn.cursor()

            cursor.execute("""
                SELECT Edb, Ename, Erole, Epass 
                FROM administration
                WHERE Arefcode=%s AND Erole=%s
            """, (ref, role))

            result = cursor.fetchone()
            cursor.close()
            conn.close()

            if result:
                Edb, Ename, Erole, db_password = result

                if password == db_password:
                    return (Edb, Ename, Erole)

            return None

        except Exception as e:
            st.error(f"Database Error: {e}")
            return None

    # -------------------------
    # LOGIN BUTTON
    # -------------------------
    if st.button("Login", width='stretch'):

        # BRUTE FORCE PROTECTION
        if st.session_state.login_attempts >= MAX_ATTEMPTS:
            if time.time() - st.session_state.last_attempt_time < LOCK_TIME:
                st.error("⛔ Too many attempts. Try again after 1 minute.")
                st.stop()
            else:
                st.session_state.login_attempts = 0

        # VALIDATION
        if not ref.strip():
            st.warning("⚠️ Please enter Ashram Code")
            st.stop()

        if not password.strip():
            st.warning("⚠️ Please enter Password")
            st.stop()

        # LOGIN
        with st.spinner("Authenticating..."):
            result = login_user(ref, role, password)

        if result:
            Edb, Ename, Erole = result

            st.session_state.db_name = Edb
            st.session_state.user_name = Ename
            st.session_state.user_role = Erole
            st.session_state.logged_in = True
            st.session_state.login_time = time.time()

            st.session_state.login_attempts = 0
            st.session_state.login_msg = f"✅ Welcome {Ename}"

            st.rerun()
        else:
            st.session_state.login_attempts += 1
            st.session_state.last_attempt_time = time.time()

            remaining = MAX_ATTEMPTS - st.session_state.login_attempts
            st.error(f"❌ Invalid credentials. Attempts left: {remaining}")