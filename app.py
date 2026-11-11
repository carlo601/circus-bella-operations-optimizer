import io
import re
import pandas as pd
import pulp
import requests
import streamlit as st

# -----------------------------------------------------------------------------
# Configuration & Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Circus Bella Bravura 2026 - Ops & Strike Optimizer",
    page_icon="🎪",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_SHEET_ID = "1mrqG1ENDb2p3OvQGv6KwjCRgHs1by2s6KUjHJ1ZC9OQ"


# -----------------------------------------------------------------------------
# Time Helpers & Overlap Detection
# -----------------------------------------------------------------------------
def parse_time_str(t_str: str) -> int:
    """Converts time strings like '11:00a', '2:00p', '14:00', '18:00p' to minutes from midnight."""
    t_str = t_str.strip().lower()
    is_pm = "p" in t_str
    is_am = "a" in t_str
    clean_t = re.sub(r"[ap]", "", t_str).strip()

    if ":" in clean_t:
        parts = clean_t.split(":")
        h, m = int(parts[0]), int(parts[1])
    else:
        h, m = int(clean_t), 0

    if is_pm and h < 12:
        h += 12
    elif is_am and h == 12:
        h = 0
    elif not is_am and not is_pm and h < 8:
        # Defaults afternoon circus hours if 2:00-6:00 without p
        h += 12

    return h * 60 + m


def time_interval_overlaps(slot1: str, slot2: str) -> bool:
    """Checks if two shift intervals on the same day overlap in time."""
    try:
        p1 = [p.strip() for p in slot1.split("-")]
        p2 = [p.strip() for p in slot2.split("-")]
        if len(p1) != 2 or len(p2) != 2:
            return False

        s1, e1 = parse_time_str(p1[0]), parse_time_str(p1[1])
        s2, e2 = parse_time_str(p2[0]), parse_time_str(p2[1])

        # Overlap if max(start) < min(end)
        return max(s1, s2) < min(e1, e2)
    except Exception:
        # Default fallback: if identical slot string, they overlap
        return slot1 == slot2


# -----------------------------------------------------------------------------
# Data Ingestion & Sanitization
# -----------------------------------------------------------------------------
def extract_time_slots_from_cell(cell_val) -> list:
    """Extracts line-separated time blocks from survey response cells."""
    if pd.isna(cell_val):
        return []
    raw = str(cell_val).replace("\r", "\n").split("\n")
    return [line.strip() for line in raw if line.strip()]


def parse_raw_dataframe(raw_df: pd.DataFrame):
    """Sanitizes names, removes TEST rows, and decomposes columns into concrete shifts."""
    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Combine First and Last Name
    first_col = next((c for c in df.columns if "first name" in c.lower()), None)
    last_col = next((c for c in df.columns if "last name" in c.lower()), None)

    if first_col and last_col:
        df["Full Name"] = (
            df[first_col].fillna("").astype(str).str.strip()
            + " "
            + df[last_col].fillna("").astype(str).str.strip()
        )
    elif first_col:
        df["Full Name"] = df[first_col].astype(str).str.strip()
    else:
        df["Full Name"] = df[df.columns[0]].astype(str).str.strip()

    name_col = "Full Name"

    # Filter out TEST submissions
    test_filter = (
        df[name_col].str.contains(r"test|dummy|sample", case=False, na=False)
        | (
            df["Email"].astype(str).str.contains(r"test@test\.com", case=False)
            if "Email" in df.columns
            else False
        )
        | (df[name_col] == "")
    )
    df = df[~test_filter].copy()

    # Deduplicate: Keep latest submission per person
    if "Submission ID" in df.columns:
        df = df.drop_duplicates(subset=[name_col], keep="last")
    else:
        df = df.drop_duplicates(subset=[name_col], keep="last")
    df = df.reset_index(drop=True)

    # Identify date/shift columns (ignore demographic and note columns)
    ignore_cols = {
        first_col,
        last_col,
        name_col,
        "Email",
        "Phone Number",
        "Submission ID",
    }
    date_cols = [
        c
        for c in df.columns
        if c not in ignore_cols and "conflict" not in c.lower()
    ]

    # Notes column
    notes_col = next((c for c in df.columns if "conflict" in c.lower()), None)

    # Decompose columns & cell contents into Concrete Shifts
    concrete_shifts = []
    # Dict: shift_key -> {"day": day, "time": slot, "role": role}
    shift_meta = {}
    # Availability matrix: (crew_member, shift_key) -> bool
    availability_matrix = {}

    for col in date_cols:
        col_lower = col.lower()
        if "strike" in col_lower:
            role = "Strike Crew"
        elif "concession" in col_lower and "training" in col_lower:
            role = "Concessions Training"
        elif "set-up" in col_lower:
            role = "Set-up Crew"
        else:
            role = "Operations (FOH / Ushers / Concessions)"

        # Find all unique time blocks listed in this column
        unique_slots = set()
        for val in df[col].dropna():
            for slot in extract_time_slots_from_cell(val):
                unique_slots.add(slot)

        if not unique_slots:
            unique_slots.add("Standard Shift")

        for slot in sorted(list(unique_slots)):
            shift_key = f"{col} [{slot}]"
            concrete_shifts.append(shift_key)
            shift_meta[shift_key] = {
                "day": col,
                "time": slot,
                "role": role,
            }

    # Populate Availability Matrix
    crew_list = sorted(df[name_col].unique().tolist())
    for _, row in df.iterrows():
        person = row[name_col]
        for shift_key in concrete_shifts:
            day = shift_meta[shift_key]["day"]
            slot = shift_meta[shift_key]["time"]
            cell_slots = extract_time_slots_from_cell(row.get(day, None))
            availability_matrix[(person, shift_key)] = slot in cell_slots

    return (
        df,
        crew_list,
        concrete_shifts,
        shift_meta,
        availability_matrix,
        notes_col,
    )


# -----------------------------------------------------------------------------
# Optimization Model (PuLP MILP)
# -----------------------------------------------------------------------------
def solve_schedule(
    crew_list: list,
    shifts: list,
    shift_meta: dict,
    availability_matrix: dict,
    shift_targets: dict,
    max_shifts_per_person: int,
):
    """Solves the assignment problem with load-balancing and time overlap exclusion."""
    prob = pulp.LpProblem(
        "Circus_Bella_Operations_Optimizer", pulp.LpMaximize
    )

    # Decision variables: x[p, s] in {0, 1}
    x = {}
    for p in crew_list:
        for s in shifts:
            if availability_matrix.get((p, s), False):
                # Clean variable names for PuLP
                clean_p = re.sub(r"[^a-zA-Z0-9_]", "_", p)
                clean_s = re.sub(r"[^a-zA-Z0-9_]", "_", s)
                x[p, s] = pulp.LpVariable(
                    f"x_{clean_p}_{clean_s}", cat=pulp.LpBinary
                )

    # Slack variable for each shift to prevent infeasibility if understaffed
    shortfall = {}
    for s in shifts:
        clean_s = re.sub(r"[^a-zA-Z0-9_]", "_", s)
        shortfall[s] = pulp.LpVariable(
            f"shortfall_{clean_s}", lowBound=0, cat=pulp.LpInteger
        )

    # Load-balancing tier variables: y[p, k] = 1 if person works >= k shifts
    y = {}
    for p in crew_list:
        clean_p = re.sub(r"[^a-zA-Z0-9_]", "_", p)
        for k in range(1, max_shifts_per_person + 1):
            y[p, k] = pulp.LpVariable(
                f"tier_{clean_p}_{k}", cat=pulp.LpBinary
            )

    # 1. Shift Target Constraints
    for s in shifts:
        assigned_p = [x[p, s] for p in crew_list if (p, s) in x]
        needed = shift_targets.get(s, 0)
        prob += pulp.lpSum(assigned_p) + shortfall[s] >= needed
        prob += pulp.lpSum(assigned_p) <= needed

    # 2. Max Shifts Per Person & Fairness Tier Linking
    for p in crew_list:
        assigned_s = [x[p, s] for s in shifts if (p, s) in x]
        prob += pulp.lpSum(assigned_s) <= max_shifts_per_person
        prob += pulp.lpSum(assigned_s) == pulp.lpSum(
            [y[p, k] for k in range(1, max_shifts_per_person + 1)]
        )
        for k in range(1, max_shifts_per_person):
            prob += y[p, k] >= y[p, k + 1]

    # 3. Dynamic Overlap Constraints (Same day, overlapping time blocks)
    day_to_shifts = {}
    for s in shifts:
        day_to_shifts.setdefault(shift_meta[s]["day"], []).append(s)

    for day, day_shifts in day_to_shifts.items():
        if len(day_shifts) > 1:
            for i in range(len(day_shifts)):
                for j in range(i + 1, len(day_shifts)):
                    s1 = day_shifts[i]
                    s2 = day_shifts[j]
                    t1 = shift_meta[s1]["time"]
                    t2 = shift_meta[s2]["time"]

                    if time_interval_overlaps(t1, t2):
                        for p in crew_list:
                            if (p, s1) in x and (p, s2) in x:
                                prob += x[p, s1] + x[p, s2] <= 1

    # Objective: Heavily penalize unfilled slots (-10,000 each)
    # Tiered marginal penalty (2 * tier) ensures shifts are balanced evenly
    unfilled_penalty = 10000 * pulp.lpSum([shortfall[s] for s in shifts])
    fairness_penalty = pulp.lpSum(
        [
            (k * 2) * y[p, k]
            for p in crew_list
            for k in range(1, max_shifts_per_person + 1)
        ]
    )

    prob += -unfilled_penalty - fairness_penalty

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=0)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]

    assignments = {s: [] for s in shifts}
    crew_totals = {p: 0 for p in crew_list}
    crew_shifts = {p: [] for p in crew_list}

    if status in ["Optimal", "Feasible"]:
        for (p, s), var in x.items():
            if var.varValue is not None and var.varValue > 0.5:
                assignments[s].append(p)
                crew_totals[p] += 1
                crew_shifts[p].append(s)

    shortfall_counts = {
        s: int(shortfall[s].varValue) if shortfall[s].varValue is not None else 0
        for s in shifts
    }

    return status, assignments, crew_totals, crew_shifts, shortfall_counts


# -----------------------------------------------------------------------------
# Streamlit UI & Application Layout
# -----------------------------------------------------------------------------
def main():
    st.title("🎪 Circus Bella Bravura 2026")
    st.subheader("Operations & Strike Shift Optimizer")

    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("📥 Data Source")

    data_source_mode = st.sidebar.radio(
        "Choose Ingestion Method:",
        ["Direct CSV Upload (Instant)", "Google Sheet Live Sync"],
    )

    raw_df = pd.DataFrame()

    if data_source_mode == "Direct CSV Upload (Instant)":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Survey CSV",
            type=["csv"],
            help="Export from Jotform or Google Sheets and drop it here.",
        )
        if uploaded_file is not None:
            raw_df = pd.read_csv(uploaded_file)
    else:
        sheet_url = st.sidebar.text_input(
            "Google Sheet Link or ID",
            value=DEFAULT_SHEET_ID,
            help="Paste full URL or Sheet ID.",
        )
        gid_val = st.sidebar.text_input(
            "Worksheet GID",
            value="",
            placeholder="e.g. 174829392 (check end of sheet URL)",
        )
        if st.sidebar.button(
            "🔄 Sync with Google Sheet", use_container_width=True
        ):
            st.cache_data.clear()

        # Ingestion logic with fallbacks
        sheet_id = sheet_url.strip()
        id_m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", sheet_url)
        if id_m:
            sheet_id = id_m.group(1)
        gid_m = re.search(r"[#&?]gid=([0-9]+)", sheet_url)
        if gid_m and not gid_val:
            gid_val = gid_m.group(1)

        urls = []
        if gid_val:
            urls.append(
                f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&gid={gid_val}"
            )
            urls.append(
                f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid_val}"
            )
        urls.append(
            f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv"
        )
        urls.append(
            f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        )

        for u in urls:
            try:
                res = requests.get(u, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
                if res.status_code == 200 and len(res.content) > 50:
                    raw_df = pd.read_csv(io.StringIO(res.text))
                    break
            except Exception:
                continue

    if raw_df.empty:
        st.info(
            "👈 Please upload your survey CSV or connect the Google Sheet in the sidebar to begin."
        )
        st.stop()

    (
        df,
        crew_list,
        concrete_shifts,
        shift_meta,
        availability_matrix,
        notes_col,
    ) = parse_raw_dataframe(raw_df)

    st.sidebar.markdown("---")
    st.sidebar.metric("Active Crew Members", len(crew_list))
    st.sidebar.metric("Concrete Shifts Detected", len(concrete_shifts))

    max_shifts_slider = st.sidebar.slider(
        "Max Shifts Per Person",
        min_value=1,
        max_value=max(len(concrete_shifts), 1),
        value=min(6, max(len(concrete_shifts), 1)),
        help="Hard cap on total shifts assigned to any single individual.",
    )

    # Tabs
    tab_setup, tab_roster, tab_optimizer, tab_crew_view = st.tabs([
        "📋 Shift Targets & Roles",
        "👥 Survey Responses & Notes",
        "🚀 Run Optimizer & Master Schedule",
        "👤 Crew-Centric View",
    ])

    # -------------------------------------------------------------
    # TAB 1: SHIFT REQUIREMENTS & HEADCOUNT
    # -------------------------------------------------------------
    with tab_setup:
        st.markdown(
            "Adjust the **Needed Headcount** for each shift. Shifts occurring at the same time on the same date will automatically prevent double-booking."
        )

        default_config = []
        for s in concrete_shifts:
            meta = shift_meta[s]
            role = meta["role"]
            needed = 4 if role == "Strike Crew" else 3
            default_config.append({
                "Shift Identifier": s,
                "Day / Event": meta["day"],
                "Time Window": meta["time"],
                "Role / Scope": role,
                "Needed Headcount": needed,
            })

        config_df = pd.DataFrame(default_config)

        edited_config = st.data_editor(
            config_df,
            column_config={
                "Shift Identifier": st.column_config.TextColumn(
                    "Shift", disabled=True
                ),
                "Day / Event": st.column_config.TextColumn(
                    "Day", disabled=True
                ),
                "Time Window": st.column_config.TextColumn(
                    "Time", disabled=True
                ),
                "Role / Scope": st.column_config.SelectboxColumn(
                    "Role Category",
                    options=[
                        "Operations (FOH / Ushers / Concessions)",
                        "Concessions Training",
                        "Set-up Crew",
                        "Strike Crew",
                    ],
                    required=True,
                ),
                "Needed Headcount": st.column_config.NumberColumn(
                    "Needed Headcount", min_value=0, max_value=50, step=1
                ),
            },
            hide_index=True,
            use_container_width=True,
        )

        shift_targets = dict(
            zip(
                edited_config["Shift Identifier"],
                edited_config["Needed Headcount"].astype(int),
            )
        )
        total_slots_needed = sum(shift_targets.values())
        total_slots_offered = sum(availability_matrix.values())

        c1, c2 = st.columns(2)
        c1.metric("Total Headcount Needed", total_slots_needed)
        c2.metric("Total Availabilities Submitted", total_slots_offered)

    # -------------------------------------------------------------
    # TAB 2: CREW AVAILABILITY & NOTES
    # -------------------------------------------------------------
    with tab_roster:
        st.markdown(
            f"**Cleaned Submissions ({len(df)} crew members)** — *Test entries purged, duplicates removed.*"
        )

        crew_overview = []
        for p in crew_list:
            avail_count = sum(
                availability_matrix.get((p, s), False) for s in concrete_shifts
            )
            note = (
                df[df["Full Name"] == p][notes_col].values[0]
                if notes_col and not df[df["Full Name"] == p].empty
                else ""
            )
            crew_overview.append({
                "Crew Member": p,
                "Total Shifts Available": avail_count,
                "Special Requests & Notes": note if pd.notna(note) else "",
            })

        st.dataframe(
            pd.DataFrame(crew_overview),
            hide_index=True,
            use_container_width=True,
        )

    # -------------------------------------------------------------
    # TAB 3: RUN OPTIMIZER & MASTER SCHEDULE
    # -------------------------------------------------------------
    with tab_optimizer:
        st.markdown("### Run PuLP Optimization Engine")

        if st.button(
            "🚀 Run Optimization", type="primary", use_container_width=True
        ):
            with st.spinner("Finding optimal schedule with PuLP CBC..."):
                status, assignments, crew_totals, crew_shifts, shortfall_counts = (
                    solve_schedule(
                        crew_list=crew_list,
                        shifts=concrete_shifts,
                        shift_meta=shift_meta,
                        availability_matrix=availability_matrix,
                        shift_targets=shift_targets,
                        max_shifts_per_person=max_shifts_slider,
                    )
                )

            st.session_state["opt_results"] = {
                "status": status,
                "assignments": assignments,
                "crew_totals": crew_totals,
                "crew_shifts": crew_shifts,
                "shortfall_counts": shortfall_counts,
                "shift_targets": shift_targets,
                "edited_config": edited_config,
            }

        if "opt_results" in st.session_state:
            results = st.session_state["opt_results"]
            status = results["status"]
            assignments = results["assignments"]
            shortfall_counts = results["shortfall_counts"]
            shift_targets_ref = results["shift_targets"]
            cfg = results["edited_config"]

            if status in ["Optimal", "Feasible"]:
                total_short = sum(shortfall_counts.values())
                if total_short == 0:
                    st.success("🎉 Schedule is 100% staffed with zero gaps!")
                else:
                    st.warning(
                        f"⚠️ Notice: **{total_short} slot(s) unfilled** due to volunteer availability or Max Shift limits."
                    )

                master_rows = []
                for _, row in cfg.iterrows():
                    s = row["Shift Identifier"]
                    role = row["Role / Scope"]
                    needed = shift_targets_ref[s]
                    assigned = assignments.get(s, [])
                    assigned_count = len(assigned)
                    short = shortfall_counts.get(s, 0)

                    status_str = "✅ Full" if short == 0 else f"🚨 Short {short}"
                    assigned_str = ", ".join(assigned) if assigned else "—"

                    master_rows.append({
                        "Day / Event": row["Day / Event"],
                        "Time Window": row["Time Window"],
                        "Role": role,
                        "Needed": needed,
                        "Assigned": assigned_count,
                        "Status": status_str,
                        "Assigned Crew": assigned_str,
                    })

                master_df = pd.DataFrame(master_rows)
                st.dataframe(master_df, hide_index=True, use_container_width=True)

                csv_buf = io.StringIO()
                master_df.to_csv(csv_buf, index=False)
                st.download_button(
                    label="📥 Download Master Schedule as CSV",
                    data=csv_buf.getvalue(),
                    file_name="circus_bella_master_schedule.csv",
                    mime="text/csv",
                )

    # -------------------------------------------------------------
    # TAB 4: CREW-CENTRIC VIEW
    # -------------------------------------------------------------
    with tab_crew_view:
        if "opt_results" in st.session_state:
            results = st.session_state["opt_results"]
            crew_shifts = results["crew_shifts"]

            crew_rows = []
            for p in sorted(crew_list):
                shifts_assigned = crew_shifts.get(p, [])
                crew_rows.append({
                    "Crew Member": p,
                    "Total Assigned Shifts": len(shifts_assigned),
                    "Assigned Schedule": (
                        " | ".join(shifts_assigned)
                        if shifts_assigned
                        else "No shifts assigned (Backup)"
                    ),
                })

            crew_df = pd.DataFrame(crew_rows).sort_values(
                by="Total Assigned Shifts", ascending=False
            )
            st.dataframe(crew_df, hide_index=True, use_container_width=True)

            crew_csv_buf = io.StringIO()
            crew_df.to_csv(crew_csv_buf, index=False)
            st.download_button(
                label="📥 Download Crew Summary as CSV",
                data=crew_csv_buf.getvalue(),
                file_name="circus_bella_crew_summary.csv",
                mime="text/csv",
            )
        else:
            st.info("Run the optimization in Tab 3 to view individual schedules.")


if __name__ == "__main__":
    main()
