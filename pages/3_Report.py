import streamlit as st
import pandas as pd
import mysql.connector

# ==============================================
# PAGE CONFIGURATION
# ==============================================
st.set_page_config(page_title='Reporting', layout='wide')
st.subheader('Reporting')

# ==============================================
# MYSQL CONNECTION
# ==============================================
from db_connection import get_mysql_connection


# ==============================================
# DATA LOADING FUNCTIONS
# ==============================================
def load_attendance_logs():

    conn = get_mysql_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT employee_name, employee_id, attendance_time, attendance_date, log_type
        FROM employee_daily_attendance
        ORDER BY attendance_time DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    df = pd.DataFrame(rows, columns=[
        'Name','EmployeeID','Timestamp','Date','log_type'
    ])

    return df


def load_registered_users():

    conn = get_mysql_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT employee_name, employee_id
        FROM employee_face_data
    """)

    rows = cursor.fetchall()
    conn.close()

    df = pd.DataFrame(rows, columns=['Name','EmployeeID'])

    return df


# ==============================================
# PAGE LAYOUT
# ==============================================
tab1, tab2, tab3, tab4 = st.tabs([
    'Registered Users',
    'Raw Logs',
    'Attendance Report',
    'Monthly Report'
])

# ==============================================
# TAB 1 - REGISTERED USERS
# ==============================================
with tab1:

    st.subheader('Registered Users')

    if st.button('Refresh User Data'):

        with st.spinner('Loading user data...'):
            user_data = load_registered_users()
            st.dataframe(user_data)


# ==============================================
# TAB 2 - RAW ATTENDANCE LOGS
# ==============================================
with tab2:

    st.subheader('Raw Attendance Logs')

    if st.button('Refresh Log Data'):

        logs = load_attendance_logs()
        st.dataframe(logs)


# ==============================================
# TAB 3 - ATTENDANCE REPORT
# ==============================================
with tab3:

    st.subheader('Processed Attendance Report')

    logs = load_attendance_logs()

    if logs.empty:
        st.warning("No attendance data found")

    else:
        logs['Timestamp'] = pd.to_datetime(logs['Timestamp'])
        logs['Date'] = logs['Timestamp'].dt.date

        report_data = []

        for (date, name, emp), group in logs.groupby(['Date','Name','EmployeeID']):

            group = group.sort_values('Timestamp')

            check_in = group[group['log_type'] == 'IN']['Timestamp'].min()
            check_out = group[group['log_type'] == 'OUT']['Timestamp'].max()

            working_hours = 0
            status = "Absent"

            if pd.notna(check_in) and pd.notna(check_out):

                duration = (check_out - check_in).total_seconds() / 3600

                # ✅ Deduct 1 hour lunch
                duration = max(0, duration - 1)

                working_hours = round(duration, 2)

                # ✅ Status Logic
                if working_hours >= 8:
                    status = "Full Day"
                elif working_hours >= 4:
                    status = "Half Day"
                else:
                    status = "Absent"

            report_data.append([
                date, name, emp, check_in, check_out, working_hours, status
            ])

        report = pd.DataFrame(report_data, columns=[
            'Date','Name','EmployeeID','Check_In','Check_Out','Working_Hours','Status'
        ])

        st.dataframe(report)

        # ✅ Download Option
        st.download_button(
            "⬇️ Download Report",
            report.to_csv(index=False),
            file_name="daily_attendance.csv"
        )
        
# ==============================================
# TAB 4 - MONTHLY REPORT
with tab4:

    st.subheader('Monthly Attendance Summary')

    logs = load_attendance_logs()

    if logs.empty:
        st.warning("No attendance data found")

    else:
        logs['Timestamp'] = pd.to_datetime(logs['Timestamp'])
        logs['Month'] = logs['Timestamp'].dt.to_period('M')

        monthly_report = logs.groupby(['Month','Name','EmployeeID'])['log_type'].apply(
            lambda x: (x=='IN').sum()
        ).reset_index(name='Days_Present')

        st.dataframe(monthly_report)

        # ✅ Download Option
        st.download_button(
            "⬇️ Download Monthly Report",
            monthly_report.to_csv(index=False),
            file_name="monthly_attendance_summary.csv"
        )