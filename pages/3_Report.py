import streamlit as st
import pandas as pd
import mysql.connector

# ==============================================
# PAGE CONFIG
# ==============================================
st.set_page_config(page_title='Reporting', layout='wide')
st.subheader('Reporting')

from db_connection import get_mysql_connection


# ==============================================
# LOAD DATA
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

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Date'] = df['Timestamp'].dt.date

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

    return pd.DataFrame(rows, columns=['Name','EmployeeID'])


# ==============================================
# BASE DATA
# ==============================================
logs = load_attendance_logs()


# ==============================================
# 🔎 FILTERS (FRONT UI)
# ==============================================
st.markdown("### 🔎 Filters")

col1, col2, col3, col4 = st.columns(4)

min_date = logs['Date'].min()
max_date = logs['Date'].max()

with col1:
    from_date = st.date_input(
        "From Date",
        value=min_date,
        key="from_date"
    )

with col2:
    to_date = st.date_input(
        "To Date",
        value=max_date,
        key="to_date"
    )

with col3:
    name_list = ["All"] + sorted(logs['Name'].dropna().unique().tolist())
    selected_name = st.selectbox("Employee Name", name_list)

with col4:
    emp_list = ["All"] + sorted(logs['EmployeeID'].dropna().astype(str).unique().tolist())
    selected_emp = st.selectbox("Employee ID", emp_list)


# ==============================================
# APPLY FILTERS
# ==============================================
filtered_logs = logs.copy()

# Date filter
filtered_logs = filtered_logs[
    (filtered_logs['Date'] >= from_date) &
    (filtered_logs['Date'] <= to_date)
]

# Name filter
if selected_name != "All":
    filtered_logs = filtered_logs[filtered_logs['Name'] == selected_name]

# Employee filter
if selected_emp != "All":
    filtered_logs = filtered_logs[
        filtered_logs['EmployeeID'].astype(str) == selected_emp
    ]


# ==============================================
# TABS
# ==============================================
tab1, tab2, tab3, tab4 = st.tabs([
    'Registered Users',
    'Raw Logs',
    'Attendance Report',
    'Monthly Report'
])


# ==============================================
# TAB 1 - USERS
# ==============================================
with tab1:
    st.subheader('Registered Users')

    if st.button('Refresh User Data'):
        st.dataframe(load_registered_users())


# ==============================================
# TAB 2 - RAW LOGS
# ==============================================
with tab2:
    st.subheader('Raw Attendance Logs')
    st.dataframe(filtered_logs)


# ==============================================
# TAB 3 - DAILY REPORT
# ==============================================
with tab3:

    st.subheader('Processed Attendance Report')

    if filtered_logs.empty:
        st.warning("No attendance data found")

    else:

        report_data = []

        for (date, name, emp), group in filtered_logs.groupby(['Date','Name','EmployeeID']):

            group = group.sort_values('Timestamp')

            check_in = group[group['log_type'] == 'IN']['Timestamp'].min()
            check_out = group[group['log_type'] == 'OUT']['Timestamp'].max()

            working_hours = 0
            status = "Absent"

            if pd.notna(check_in) and pd.notna(check_out):

                duration = (check_out - check_in).total_seconds() / 3600
                duration = max(0, duration - 1)

                working_hours = round(duration, 2)

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

        st.download_button(
            "⬇️ Download Report",
            report.to_csv(index=False),
            file_name="daily_attendance.csv"
        )


# ==============================================
# TAB 4 - MONTHLY REPORT
# ==============================================
with tab4:

    st.subheader('Monthly Attendance Summary')

    if filtered_logs.empty:
        st.warning("No attendance data found")

    else:

        report_data = []

        for (date, name, emp), group in filtered_logs.groupby(['Date','Name','EmployeeID']):

            group = group.sort_values('Timestamp')

            check_in = group[group['log_type'] == 'IN']['Timestamp'].min()
            check_out = group[group['log_type'] == 'OUT']['Timestamp'].max()

            status = "Absent"

            if pd.notna(check_in) and pd.notna(check_out):

                duration = (check_out - check_in).total_seconds() / 3600
                duration = max(0, duration - 1)

                if duration >= 8:
                    status = "Full Day"
                elif duration >= 4:
                    status = "Half Day"
                else:
                    status = "Absent"

            report_data.append([date, name, emp, status])

        report = pd.DataFrame(report_data, columns=[
            'Date','Name','EmployeeID','Status'
        ])

        def status_to_value(x):
            if x == "Full Day":
                return 1
            elif x == "Half Day":
                return 0.5
            return 0

        report['Value'] = report['Status'].apply(status_to_value)
        report['Month'] = pd.to_datetime(report['Date']).dt.to_period('M')

        monthly_report = report.groupby(
            ['Month','Name','EmployeeID']
        )['Value'].sum().reset_index(name='Total_Days')

        st.dataframe(monthly_report)

        st.download_button(
            "⬇️ Download Monthly Report",
            monthly_report.to_csv(index=False),
            file_name="monthly_attendance_summary.csv"
        )