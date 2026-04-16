import streamlit as st

st.set_page_config(page_title="Logout", page_icon="🚪")

st.title("🚪 Logout")

# If user is logged in → logout
if "logged_in" in st.session_state and st.session_state.logged_in:

    st.success("You have been logged out successfully ✅")

    # Clear session
    st.session_state.clear()

    # Redirect to login page after 2 sec
    st.info("Redirecting to login page...")

    import time
    time.sleep(2)

    st.switch_page("pages/0_Login.py")

else:
    st.warning("You are not logged in")
    st.switch_page("pages/0_Login.py")