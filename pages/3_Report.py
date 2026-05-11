import streamlit as st
import pandas as pd
import numpy as np
import mysql.connector
from datetime import datetime, date, timedelta
from db_connection import get_mysql_connection

# --- CONFIGURATION ---
st.set_page_config(page_title="Attendance Reporting", layout="wide")

# --- 1. LOAD SHIFT MAPPING FROM EXCEL ---
@st.cache_data
def get_shift_requirements():
    try:
        # Note: Ensure openpyxl is installed: pip install openpyxl
        df_map = pd.read_excel('working_hours_mapping.xlsx')
        def calc_required_hours(shift_str):
            try:
                # Expects format "9:00 AM - 6:00 PM"
                start_str, end_str = shift_str.split(' - ')
                fmt = '%I:%M %p'
                t1 = datetime.strptime(start_str.strip(), fmt)
                t2 = datetime.strptime(end_str.strip(), fmt)
                delta = t2 - t1
                return delta.total_seconds() / 3600
            except:
                return 9.0  # Default fallback
        
        df_map['ReqHours'] = df_map['Working Hours'].apply(calc_required_hours)
        return df_map.set_index('Designation')['ReqHours'].to_dict()
    except Exception as e:
        st.error(f"Error loading working_hours_mapping.xlsx: {e}")
        return {}

shift_hours_dict = get_shift_requirements()

# --- 2. DATA LOADING FUNCTIONS ---
def load_attendance_logs():
    try:
        conn = get_mysql_connection()
        query = """
            SELECT 
                a.employee_name, 
                a.employee_id, 
                a.attendance_time, 
                a.attendance_date, 
                a.log_type,
                s.designation
            FROM employee_daily_attendance a
            LEFT JOIN employee_salary s ON a.employee_id = s.emp_id
            ORDER BY a.attendance_time DESC
        """
        df = pd.read_sql(query, conn)
        #conn.close()
        
        df["Timestamp"] = pd.to_datetime(df["attendance_time"], errors="coerce")
        df["Date"] = pd.to_datetime(df["attendance_date"]).dt.date
        return df
    except Exception as e:
        st.error(f"Error loading logs: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def load_registered_users():
    try:
        conn = get_mysql_connection()
        query = """
            SELECT 
                s.emp_id AS 'Employee ID', 
                s.employee_name AS 'Name', 
                s.designation AS 'Designation', 
                s.department AS 'Department'
            FROM employee_salary s
            INNER JOIN employee_face_data f ON s.emp_id = f.employee_id
            WHERE s.deleted = 0
        """
        df = pd.read_sql(query, conn)
        #conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading users: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

# Load initial data
logs = load_attendance_logs()
users_df = load_registered_users()

# --- 3. UI LAYOUT ---
st.title("📊 Attendance Reporting System")

# --- FILTER SECTION ---
with st.expander("🔎 Filter Records", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    today = date.today()
    first_day = today.replace(day=1)

    from_date = col1.date_input("From Date", value=first_day)
    to_date = col2.date_input("To Date", value=today)

    name_list = ["All"] + sorted(logs["employee_name"].unique().tolist()) if not logs.empty else ["All"]
    selected_name = col3.selectbox("Employee Name", name_list)

    emp_list = ["All"] + sorted(logs["employee_id"].astype(str).unique().tolist()) if not logs.empty else ["All"]
    selected_emp = col4.selectbox("Employee ID", emp_list)

# --- APPLY FILTERS ---
filtered_logs = logs.copy()
if not filtered_logs.empty:
    filtered_logs = filtered_logs[(filtered_logs["Date"] >= from_date) & (filtered_logs["Date"] <= to_date)]
    if selected_name != "All": 
        filtered_logs = filtered_logs[filtered_logs["employee_name"] == selected_name]
    if selected_emp != "All": 
        filtered_logs = filtered_logs[filtered_logs["employee_id"].astype(str) == selected_emp]

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["👥 Registered Employees", "📄 Raw Logs", "📅 Daily Report", "📈 Monthly Summary"])

with tab1:
    st.subheader("Employees with Registered Face Data")
    if users_df.empty:
        st.info("No employees found with registered face data.")
    else:
        st.dataframe(users_df, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Raw Punch Records")
    if not filtered_logs.empty:
        st.dataframe(filtered_logs[["Date", "employee_name", "employee_id", "designation", "Timestamp", "log_type"]], use_container_width=True)

with tab3:
    st.subheader("Daily Working Hours & Status")
    if filtered_logs.empty:
        st.warning("No data found for the selected range.")
    else:
        report_data = []
        grouped = filtered_logs.groupby(["Date", "employee_name", "employee_id", "designation"])
        
        for (att_date, name, emp_id, desig), group in grouped:
            group = group.sort_values("Timestamp")
            
            total_seconds = 0
            temp_in_time = None
            for _, row in group.iterrows():
                if row["log_type"] == "IN":
                    temp_in_time = row["Timestamp"]
                elif row["log_type"] == "OUT" and temp_in_time:
                    total_seconds += (row["Timestamp"] - temp_in_time).total_seconds()
                    temp_in_time = None
            
            working_hours = round(total_seconds / 3600, 2)
            req_hours = shift_hours_dict.get(desig, 9.0)
            
            # Status Logic
            if working_hours >= (req_hours * 0.85):
                status = "Full Day"
            elif working_hours >= 4.0:
                status = "Half Day"
            elif working_hours > 0:
                status = "Short Leave"
            else:
                status = "Absent"
            
            first_in = group[group["log_type"] == "IN"]["Timestamp"].min()
            last_out = group[group["log_type"] == "OUT"]["Timestamp"].max()

            report_data.append({
                "Date": att_date,
                "Name": name,
                "ID": emp_id,
                "Designation": desig,
                "In Time": first_in.strftime('%I:%M %p') if pd.notnull(first_in) else "N/A",
                "Out Time": last_out.strftime('%I:%M %p') if pd.notnull(last_out) else "N/A",
                "Worked Hours": working_hours,
                "Shift Req": req_hours,
                "Status": status
            })

        daily_df = pd.DataFrame(report_data)

        def style_status(val):
            if val == "Full Day": return 'background-color: #d4edda; color: #155724'
            if val == "Half Day": return 'background-color: #fff3cd; color: #856404'
            return 'background-color: #f8d7da; color: #721c24'

        st.dataframe(daily_df.style.applymap(style_status, subset=['Status']), use_container_width=True)

with tab4:
    st.subheader("Monthly Attendance Statistics")
    if not filtered_logs.empty and 'daily_df' in locals():
        # Add Month column for segregation
        summary_df = daily_df.copy()
        summary_df['Month'] = pd.to_datetime(summary_df['Date']).dt.strftime('%B %Y')

        # Group by Month AND Employee details
        summary = summary_df.groupby(["Month", "Name", "ID", "Designation"]).agg({
            "Worked Hours": "sum",
            "Status": [
                lambda x: (x == "Full Day").sum(), 
                lambda x: (x == "Half Day").sum(),
                lambda x: (x == "Short Leave").sum(),
                lambda x: (x == "Absent").sum()
            ]
        }).reset_index()
        
        # Flatten column names
        summary.columns = ["Month", "Name", "ID", "Designation", "Total Hours", "Full Days", "Half Days", "Short Leaves", "Absents"]
        
        # Calculation for Payable Days
        summary["Payable Days"] = summary["Full Days"] + (summary["Half Days"] * 0.5)
        
        # Sort by Month (Chronological)
        summary['sort_date'] = pd.to_datetime(summary['Month'], format='%B %Y')
        summary = summary.sort_values(by=['sort_date', 'Name'], ascending=[False, True]).drop(columns=['sort_date'])

        # Display UI
        st.dataframe(summary, use_container_width=True, hide_index=True)
        
        # Download Option
        csv = summary.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Monthly Summary", data=csv, file_name="monthly_attendance.csv", mime="text/csv")
    else:
        st.info("No data available to generate a monthly summary.")