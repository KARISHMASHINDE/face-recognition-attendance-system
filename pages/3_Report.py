import streamlit as st
import pandas as pd
import numpy as np
import mysql.connector
from datetime import datetime, date, timedelta
from db_connection import get_mysql_connection

st.set_page_config(page_title="Reporting", layout="wide")
st.subheader("Reporting Dashboard")

def load_attendance_logs():
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT employee_name, employee_id, attendance_time, attendance_date, log_type
            FROM employee_daily_attendance ORDER BY attendance_time DESC
        """)
        rows = cursor.fetchall()
        conn.close()
        df = pd.DataFrame(rows, columns=["Name", "EmployeeID", "Timestamp", "AttendanceDate", "log_type"])
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df["Date"] = df["Timestamp"].dt.date
        return df
    except Exception as e:
        st.error(f"Error loading logs: {e}")
        return pd.DataFrame()

def load_registered_users():
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT employee_name, employee_id FROM employee_face_data")
        rows = cursor.fetchall()
        conn.close()
        return pd.DataFrame(rows, columns=["Name", "EmployeeID"])
    except Exception as e:
        st.error(f"Error loading users: {e}")
        return pd.DataFrame()

# Load Data
logs = load_attendance_logs()
users_df = load_registered_users()

# --- FILTER SECTION ---
st.markdown("### 🔎 Filters")
col1, col2, col3, col4 = st.columns(4)
today = date.today()
first_day = today.replace(day=1)

from_date = col1.date_input("From Date", value=first_day)
to_date = col2.date_input("To Date", value=today)

name_list = ["All"] + sorted(logs["Name"].unique().tolist()) if not logs.empty else ["All"]
selected_name = col3.selectbox("Employee Name", name_list)

emp_list = ["All"] + sorted(logs["EmployeeID"].astype(str).unique().tolist()) if not logs.empty else ["All"]
selected_emp = col4.selectbox("Employee ID", emp_list)

# --- APPLY FILTERS ---
filtered_logs = logs.copy()
if not filtered_logs.empty:
    filtered_logs = filtered_logs[(filtered_logs["Date"] >= from_date) & (filtered_logs["Date"] <= to_date)]
    if selected_name != "All": 
        filtered_logs = filtered_logs[filtered_logs["Name"] == selected_name]
    if selected_emp != "All": 
        filtered_logs = filtered_logs[filtered_logs["EmployeeID"].astype(str) == selected_emp]

tab1, tab2, tab3, tab4 = st.tabs(["Registered Users", "Raw Logs", "Daily Report", "Monthly Report"])

with tab1:
    st.subheader("All Registered Employees")
    st.dataframe(users_df, use_container_width=True)

with tab2:
    st.subheader("Raw Attendance Records")
    st.dataframe(filtered_logs, use_container_width=True)

with tab3:
    st.subheader("Daily Attendance Report")
    if filtered_logs.empty:
        st.warning("No data found for the selected filters.")
    else:
        report_data = []
        grouped = filtered_logs.groupby(["Date", "Name", "EmployeeID"])
        
        for (att_date, name, emp), group in grouped:
            group = group.sort_values("Timestamp")
            
            # Count In/Outs
            total_in = len(group[group["log_type"] == "IN"])
            total_out = len(group[group["log_type"] == "OUT"])
            
            total_seconds = 0
            temp_in = None
            first_in = group[group["log_type"] == "IN"]["Timestamp"].min()
            last_out = group[group["log_type"] == "OUT"]["Timestamp"].max()

            for _, row in group.iterrows():
                if row["log_type"] == "IN":
                    temp_in = row["Timestamp"]
                elif row["log_type"] == "OUT" and temp_in:
                    total_seconds += (row["Timestamp"] - temp_in).total_seconds()
                    temp_in = None
            
            working_hours = round(total_seconds / 3600, 2)
            status = "Full Day" if working_hours >= 8 else ("Half Day" if working_hours >= 4 else "Short Hours")
            
            report_data.append([att_date, name, emp, first_in, last_out, total_in, total_out, working_hours, status])

        daily_df = pd.DataFrame(report_data, columns=[
            "Date", "Name", "EmployeeID", "First In", "Last Out", "Total IN", "Total OUT", "Net Hours", "Status"
        ])
        st.dataframe(daily_df, use_container_width=True)

with tab4:
    st.subheader("Monthly Attendance & Absenteeism Summary")
    if filtered_logs.empty:
        st.warning("No data found.")
    else:
        # 1. Calculate Presence Value per day
        monthly_data = []
        grouped = filtered_logs.groupby(["Date", "Name", "EmployeeID"])
        for (att_date, name, emp), group in grouped:
            group = group.sort_values("Timestamp")
            sec = 0
            tin = None
            for _, row in group.iterrows():
                if row["log_type"] == "IN": tin = row["Timestamp"]
                elif row["log_type"] == "OUT" and tin:
                    sec += (row["Timestamp"] - tin).total_seconds()
                    tin = None
            hrs = sec / 3600
            val = 1 if hrs >= 4 else 0 # Minimum 4 hours to be counted as present
            monthly_data.append([att_date, name, emp, val])

        m_df = pd.DataFrame(monthly_data, columns=["Date", "Name", "EmployeeID", "Value"])
        
        # 2. Aggregate by Month and Employee
        m_df["Month"] = pd.to_datetime(m_df["Date"]).dt.to_period("M")
        summary = m_df.groupby(["Month", "Name", "EmployeeID"])["Value"].sum().reset_index()
        summary.rename(columns={"Value": "Total Present Days"}, inplace=True)
        
        # 3. Calculate Absent Days
        # Total days in selected range
        delta = to_date - from_date
        total_days_in_range = delta.days + 1
        
        # We subtract present days from total range days to get absent days
        summary["Absent Days"] = total_days_in_range - summary["Total Present Days"]
        
        # Ensure absent days isn't negative (edge cases)
        summary["Absent Days"] = summary["Absent Days"].clip(lower=0)
        
        st.dataframe(summary, use_container_width=True)