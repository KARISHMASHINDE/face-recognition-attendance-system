import streamlit as st
import mysql.connector

def get_mysql_connection():

    if "db_name" not in st.session_state:
        raise Exception("Database not selected. Please login first.")

    return mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="Zeeshan@123",
        database=st.session_state.db_name   # ✅ dynamic DB
    )