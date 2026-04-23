import streamlit as st
import pandas as pd
import mysql.connector
from datetime import datetime, date

from db_connection import get_mysql_connection

# ==============================================
# PAGE CONFIG
# ==============================================
st.set_page_config(page_title="Reporting", layout="wide")
st.subheader("Reporting Dashboard")


# ==============================================
# LOAD ATTENDANCE LOGS
# ==============================================
def load_attendance_logs():
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT employee_name,
                   employee_id,
                   attendance_time,
                   attendance_date,
                   log_type
            FROM employee_daily_attendance
            ORDER BY attendance_time DESC
        """)

        rows = cursor.fetchall()
        conn.close()

        df = pd.DataFrame(rows, columns=[
            "Name",
            "EmployeeID",
            "Timestamp",
            "AttendanceDate",
            "log_type"
        ])

        if df.empty:
            return df

        # Safe datetime conversion
        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"],
            errors="coerce"
        )

        # Create clean Date column
        df["Date"] = df["Timestamp"].dt.date

        return df

    except Exception as e:
        st.error(f"Error loading attendance logs: {e}")
        return pd.DataFrame()


# ==============================================
# LOAD REGISTERED USERS
# ==============================================
def load_registered_users():
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT employee_name, employee_id
            FROM employee_face_data
        """)

        rows = cursor.fetchall()
        conn.close()

        return pd.DataFrame(
            rows,
            columns=["Name", "EmployeeID"]
        )

    except Exception as e:
        st.error(f"Error loading users: {e}")
        return pd.DataFrame()


# ==============================================
# BASE DATA
# ==============================================
logs = load_attendance_logs()


# ==============================================
# FILTER SECTION
# ==============================================
st.markdown("### 🔎 Filters")

col1, col2, col3, col4 = st.columns(4)

# Default dates
today = date.today()
first_day_of_month = today.replace(day=1)

# Safe min/max dates
if logs.empty:
    min_date = first_day_of_month
    max_date = today
else:
    logs["Date"] = pd.to_datetime(
        logs["Date"],
        errors="coerce"
    ).dt.date

    valid_dates = logs["Date"].dropna()

    if valid_dates.empty:
        min_date = first_day_of_month
        max_date = today
    else:
        min_date = valid_dates.min()
        max_date = valid_dates.max()


# From Date
with col1:
    from_date = st.date_input(
        "From Date",
        value=first_day_of_month,
        key="from_date"
    )

# To Date
with col2:
    to_date = st.date_input(
        "To Date",
        value=max_date,
        key="to_date"
    )

# Employee Name Filter
with col3:
    if logs.empty:
        name_list = ["All"]
    else:
        name_list = ["All"] + sorted(
            logs["Name"].dropna().unique().tolist()
        )

    selected_name = st.selectbox(
        "Employee Name",
        name_list
    )

# Employee ID Filter
with col4:
    if logs.empty:
        emp_list = ["All"]
    else:
        emp_list = ["All"] + sorted(
            logs["EmployeeID"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    selected_emp = st.selectbox(
        "Employee ID",
        emp_list
    )


# ==============================================
# APPLY FILTERS
# ==============================================
filtered_logs = logs.copy()

if not filtered_logs.empty:
    # Date filter
    filtered_logs = filtered_logs[
        (filtered_logs["Date"] >= from_date) &
        (filtered_logs["Date"] <= to_date)
    ]

    # Name filter
    if selected_name != "All":
        filtered_logs = filtered_logs[
            filtered_logs["Name"] == selected_name
        ]

    # Employee filter
    if selected_emp != "All":
        filtered_logs = filtered_logs[
            filtered_logs["EmployeeID"].astype(str) == selected_emp
        ]


# ==============================================
# TABS
# ==============================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Registered Users",
    "Raw Logs",
    "Daily Report",
    "Monthly Report"
])


# ==============================================
# TAB 1 - REGISTERED USERS
# ==============================================
with tab1:
    st.subheader("Registered Users")

    users_df = load_registered_users()

    if users_df.empty:
        st.warning("No registered users found")
    else:
        st.dataframe(users_df, use_container_width=True)

        st.download_button(
            "⬇️ Download Users",
            users_df.to_csv(index=False),
            file_name="registered_users.csv"
        )


# ==============================================
# TAB 2 - RAW LOGS
# ==============================================
with tab2:
    st.subheader("Raw Attendance Logs")

    if filtered_logs.empty:
        st.warning("No logs found")
    else:
        st.dataframe(
            filtered_logs,
            use_container_width=True
        )

        st.download_button(
            "⬇️ Download Raw Logs",
            filtered_logs.to_csv(index=False),
            file_name="raw_logs.csv"
        )


# ==============================================
# TAB 3 - DAILY REPORT
# ==============================================
with tab3:
    st.subheader("Daily Attendance Report")

    if filtered_logs.empty:
        st.warning("No attendance data found")

    else:
        report_data = []

        grouped = filtered_logs.groupby(
            ["Date", "Name", "EmployeeID"]
        )

        for (att_date, name, emp), group in grouped:
            group = group.sort_values("Timestamp")

            check_in = group[
                group["log_type"] == "IN"
            ]["Timestamp"].min()

            check_out = group[
                group["log_type"] == "OUT"
            ]["Timestamp"].max()

            working_hours = 0
            status = "Absent"

            if pd.notna(check_in) and pd.notna(check_out):
                duration = (
                    check_out - check_in
                ).total_seconds() / 3600

                # Lunch break deduction
                duration = max(0, duration - 1)

                working_hours = round(duration, 2)

                if working_hours >= 8:
                    status = "Full Day"
                elif working_hours >= 4:
                    status = "Half Day"
                else:
                    status = "Absent"

            report_data.append([
                att_date,
                name,
                emp,
                check_in,
                check_out,
                working_hours,
                status
            ])

        daily_report = pd.DataFrame(
            report_data,
            columns=[
                "Date",
                "Name",
                "EmployeeID",
                "Check In",
                "Check Out",
                "Working Hours",
                "Status"
            ]
        )

        st.dataframe(
            daily_report,
            use_container_width=True
        )

        st.download_button(
            "⬇️ Download Daily Report",
            daily_report.to_csv(index=False),
            file_name="daily_attendance_report.csv"
        )


# ==============================================
# TAB 4 - MONTHLY REPORT
# ==============================================
with tab4:
    st.subheader("Monthly Attendance Summary")

    if filtered_logs.empty:
        st.warning("No attendance data found")

    else:
        report_data = []

        grouped = filtered_logs.groupby(
            ["Date", "Name", "EmployeeID"]
        )

        for (att_date, name, emp), group in grouped:
            group = group.sort_values("Timestamp")

            check_in = group[
                group["log_type"] == "IN"
            ]["Timestamp"].min()

            check_out = group[
                group["log_type"] == "OUT"
            ]["Timestamp"].max()

            status = "Absent"

            if pd.notna(check_in) and pd.notna(check_out):
                duration = (
                    check_out - check_in
                ).total_seconds() / 3600

                duration = max(0, duration - 1)

                if duration >= 8:
                    status = "Full Day"
                elif duration >= 4:
                    status = "Half Day"
                else:
                    status = "Absent"

            report_data.append([
                att_date,
                name,
                emp,
                status
            ])

        monthly_df = pd.DataFrame(
            report_data,
            columns=[
                "Date",
                "Name",
                "EmployeeID",
                "Status"
            ]
        )

        def status_to_value(status):
            if status == "Full Day":
                return 1
            elif status == "Half Day":
                return 0.5
            return 0

        monthly_df["Present_Value"] = monthly_df["Status"].apply(status_to_value)

        monthly_df["Month"] = pd.to_datetime(
            monthly_df["Date"]
        ).dt.to_period("M")

        # Total present days
        monthly_report = monthly_df.groupby(
            ["Month", "Name", "EmployeeID"]
        )["Present_Value"].sum().reset_index()

        monthly_report.rename(
            columns={
                "Present_Value": "Total Present Days"
            },
            inplace=True
        )

        # Calculate total working days in selected range
        total_working_days = len(
            pd.date_range(from_date, to_date)
        )
        # total_working_days = len([
        #     day for day in pd.date_range(from_date, to_date)
        #     if day.weekday() != 6
        # ]) # Exclude Sundays
        
        # Leave days
        monthly_report["Leave Days"] = (
            total_working_days - monthly_report["Total Present Days"]
        )

        monthly_report["Leave Days"] = monthly_report["Leave Days"].apply(
            lambda x: max(0, x)
        )

        st.dataframe(
            monthly_report,
            use_container_width=True
        )

        st.download_button(
            "⬇️ Download Monthly Report",
            monthly_report.to_csv(index=False),
            file_name="monthly_attendance_report.csv"
        )