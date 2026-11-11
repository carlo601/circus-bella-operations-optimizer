import io
import re
import urllib.parse
import pandas as pd
import pulp
import streamlit as st

# -----------------------------------------------------------------------------
# Configuration & Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Circus Bella Bravura - Operations & Strike Optimizer",
    page_icon="🎪",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_SHEET_ID = "1mrqG1ENDb2p3OvQGv6KwjCRgHs1by2s6KUjHJ1ZC9OQ"


# -----------------------------------------------------------------------------
# Data Ingestion & Sanitization
# -----------------------------------------------------------------------------
@st.cache_data(ttl=60)
def load_survey_data(sheet_id: str, gid: str = "0") -> pd.DataFrame:
    """Fetches CSV export directly from Google Sheets."""
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Failed to fetch data from Google Sheet. Error: {e}")
        return pd.DataFrame()


def clean_and_parse_roster(df: pd.DataFrame):
    """Sanitizes names, removes TEST rows, deduplicates, and extracts shift columns."""
    if df.empty:
        return pd.DataFrame(), [], ""

    # Normalize column names
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Detect Timestamp column
    timestamp_col = next(
        (c for c in df.columns if "timestamp" in c.lower()), None
    )

    # Detect Crew Name column
    name_col = next(
        (
            c
            for c in df.columns
            if any(k in c.lower() for k in ["name", "full name", "first name"])
        ),
        None,
    )
    if not name_col:
        # Fallback to second column if timestamp is first
        name_col = (
            df.columns[1]
            if len(df.columns) > 1 and timestamp_col == df.columns[0]
            else df.columns[0]
        )

    # 1. Clean Names & remove TEST submissions
    df[name_col] = df[name_col].astype(str).str.strip()
    df = df[
        ~df[name_col].str.contains(
            r"^\s*$|test|sample|dummy", case=False, na=False
        )
    ]

    # 2. Deduplicate: Keep latest submission per person
    if timestamp_col:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        df = df.sort_values(by=timestamp_col, ascending=True)

    df = df.drop_duplicates(subset=[name_col], keep="last").reset_index(
        drop=True
    )

    # 3. Identify Shift Columns (exclude metadata columns like Timestamp, Email, Notes, Phone)
    meta_keywords = [
        "timestamp",
        "name",
        "email",
        "phone",
        "comments",
        "notes",
        "dietary",
        "experience",
        "shirt",
    ]
    shift_cols = [
        c
        for c in df.columns
        if not any(k in c.lower() for k in meta_keywords)
        and c not in [timestamp_col, name_col]
    ]

    return df, shift_cols, name_col


def is_available(value) -> bool:
    """Interprets survey cell values into Boolean availability."""
    if pd.isna(value):
        return False
    val_str = str(value).strip().lower()
    negative_answers = {
        "no",
        "n",
        "unavailable",
        "not available",
        "0",
        "false",
        "none",
        "",
        "-",
    }
    return val_str not in negative_answers


# -----------------------------------------------------------------------------
# Optimization Model (PuLP MILP)
# -----------------------------------------------------------------------------
def solve_schedule(
    crew_list: list,
    shifts: list,
    availability_matrix: dict,
    shift_targets: dict,
    overlap_groups: dict,
    max_shifts_per_person: int,
):
    """Solves the Operations & Strike assignment problem with load-balancing."""
    prob = pulp.LpProblem(
        "Circus_Bella_Operations_Optimizer", pulp.LpMaximize
    )

    # Decision variables: x[crew, shift] in {0, 1}
    x = {}
    for p in crew_list:
        for s in shifts:
            if availability_matrix.get((p, s), False):
                x[p, s] = pulp.LpVariable(f"x_{p}_{s}", cat=pulp.LpBinary)

    # Shortfall variables to guarantee feasibility when understaffed
    shortfall = {}
    for s in shifts:
        shortfall[s] = pulp.LpVariable(
            f"shortfall_{s}", lowBound=0, cat=pulp.LpInteger
        )

    # Fairness linearization variables: y[p, k] = 1 if person p works >= k shifts
    y = {}
    for p in crew_list:
        for k in range(1, max_shifts_per_person + 1):
            y[p, k] = pulp.LpVariable(f"tier_{p}_{k}", cat=pulp.LpBinary)

    # ------------------ CONSTRAINTS ------------------

    # 1. Shift Capacity / Target
    for s in shifts:
        assigned_crew = [x[p, s] for p in crew_list if (p, s) in x]
        needed = shift_targets.get(s, 0)
        # assigned + shortfall >= needed
        prob += (
            pulp.lpSum(assigned_crew) + shortfall[s] >= needed,
            f"Demand_Min_{s}",
        )
        # Never assign more than needed
        prob += pulp.lpSum(assigned_crew) <= needed, f"Demand_Max_{s}"

    # 2. Maximum shifts per person & Fairness Tiers
    for p in crew_list:
        assigned_shifts = [x[p, s] for s in shifts if (p, s) in x]
        # Max limit
        prob += (
            pulp.lpSum(assigned_shifts) <= max_shifts_per_person,
            f"Max_Shifts_{p}",
        )
        # Link shift count to tier variables: sum(x) == sum(y)
        prob += pulp.lpSum(assigned_shifts) == pulp.lpSum(
            [y[p, k] for k in range(1, max_shifts_per_person + 1)]
        ), f"Tier_Link_{p}"
        # Order tiers: y[p, 1] >= y[p, 2] >= ...
        for k in range(1, max_shifts_per_person):
            prob += y[p, k] >= y[p, k + 1], f"Tier_Order_{p}_{k}"

    # 3. Overlap Constraints (No concurrent shifts for the same person)
    for group_name, grouped_shifts in overlap_groups.items():
        if len(grouped_shifts) > 1:
            for p in crew_list:
                prob += (
                    pulp.lpSum(
                        [x[p, s] for s in grouped_shifts if (p, s) in x]
                    )
                    <= 1,
                    f"No_Overlap_{p}_{group_name}",
                )

    # ------------------ OBJECTIVE FUNCTION ------------------
    # 1) Heavy penalty for unfilled required slots (-10,000 per missing crew)
    # 2) Progressive marginal penalty for assigning extra shifts to the same person
    #    (shift 1 costs 2, shift 2 costs 4, shift 3 costs 6, etc.) -> balances workload evenly
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

    # Extract Results
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
    st.sidebar.header("⚙️ Configuration")

    sheet_id = st.sidebar.text_input("Google Sheet ID", value=DEFAULT_SHEET_ID)
    gid = st.sidebar.text_input("Worksheet GID", value="0")

    if st.sidebar.button("🔄 Sync with Google Sheet", use_container_width=True):
        st.cache_data.clear()
        st.sidebar.success("Cache cleared! Reloaded live data.")

    # Load Data
    raw_df = load_survey_data(sheet_id, gid)

    if raw_df.empty:
        st.warning("⚠️ No data loaded. Check the Google Sheet ID and permissions.")
        st.stop()

    df, detected_shifts, name_col = clean_and_parse_roster(raw_df)

    if not detected_shifts:
        st.error(
            "Could not detect shift columns automatically. Please verify survey columns."
        )
        st.dataframe(raw_df.head(5))
        st.stop()

    crew_list = sorted(df[name_col].dropna().unique().tolist())

    # Build Availability Lookup Matrix
    availability_matrix = {}
    for _, row in df.iterrows():
        person = row[name_col]
        for s in detected_shifts:
            availability_matrix[(person, s)] = is_available(row.get(s, None))

    # Sidebar Metrics
    st.sidebar.markdown("---")
    st.sidebar.metric("Active Crew Members", len(crew_list))
    st.sidebar.metric("Detected Shifts", len(detected_shifts))

    max_shifts_slider = st.sidebar.slider(
        "Max Shifts Per Person",
        min_value=1,
        max_value=max(len(detected_shifts), 1),
        value=min(4, max(len(detected_shifts), 1)),
        help="Hard cap on the number of shifts any single crew member can be scheduled for.",
    )

    # --- TABS INTERFACE ---
    tab_setup, tab_roster, tab_optimizer, tab_crew_view = st.tabs([
        "📋 Shift Targets & Roles",
        "👥 Survey Data & Availability",
        "🚀 Run Optimizer & Master Schedule",
        "👤 Crew-Centric View",
    ])

    # -------------------------------------------------------------
    # TAB 1: SHIFT REQUIREMENTS & OVERLAP SETUP
    # -------------------------------------------------------------
    with tab_setup:
        st.markdown(
            "Configure the **Needed Headcount**, assigned **Role / Scope**, and **Time Slot** for each shift."
        )
        st.caption(
            "💡 Shifts with the same *Overlap Group* will never assign the same person to both (e.g. FOH and Concessions during the same show)."
        )

        # Prepopulate shift configuration DataFrame
        default_config = []
        for s in detected_shifts:
            # Smart default role guessing
            s_lower = s.lower()
            if "strike" in s_lower:
                role = "Strike Crew"
                group = "Strike Tear Down"
            elif "concession" in s_lower:
                role = "Concessions"
                group = re.sub(
                    r"concession[s]?", "", s, flags=re.IGNORECASE
                ).strip()
            elif "usher" in s_lower:
                role = "Ushers"
                group = re.sub(r"usher[s]?", "", s, flags=re.IGNORECASE).strip()
            elif "foh" in s_lower or "front of house" in s_lower:
                role = "Front of House"
                group = re.sub(
                    r"foh|front of house", "", s, flags=re.IGNORECASE
                ).strip()
            else:
                role = "General Operations"
                group = s

            default_config.append({
                "Shift Name": s,
                "Role / Scope": role,
                "Overlap Group (Concurrent)": group or s,
                "Needed Headcount": 3,  # Sensible default for circus FOH/concessions
            })

        config_df = pd.DataFrame(default_config)

        # Interactive Data Editor
        edited_config = st.data_editor(
            config_df,
            column_config={
                "Shift Name": st.column_config.TextColumn(
                    "Shift Column", disabled=True
                ),
                "Role / Scope": st.column_config.SelectboxColumn(
                    "Role Category",
                    options=[
                        "Front of House",
                        "Concessions",
                        "Ushers",
                        "Strike Crew",
                        "General Operations",
                    ],
                    required=True,
                ),
                "Overlap Group (Concurrent)": st.column_config.TextColumn(
                    "Overlap Group",
                    help="Shifts sharing the exact same group name cannot be assigned to the same volunteer.",
                ),
                "Needed Headcount": st.column_config.NumberColumn(
                    "Needed Headcount", min_value=0, max_value=50, step=1
                ),
            },
            hide_index=True,
            use_container_width=True,
        )

        # Build dictionaries from edited configuration
        shift_targets = dict(
            zip(
                edited_config["Shift Name"],
                edited_config["Needed Headcount"].astype(int),
            )
        )
        total_slots_needed = sum(shift_targets.values())

        # Group shifts by overlap
        overlap_groups = {}
        for _, row in edited_config.iterrows():
            grp = row["Overlap Group (Concurrent)"]
            s_name = row["Shift Name"]
            overlap_groups.setdefault(grp, []).append(s_name)

        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Total Crew Slots Needed", total_slots_needed)
        total_avail_slots = sum(availability_matrix.values())
        col_m2.metric("Total Volunteer Availabilities", total_avail_slots)

    # -------------------------------------------------------------
    # TAB 2: SURVEY DATA & AVAILABILITY MATRIX
    # -------------------------------------------------------------
    with tab_roster:
        st.markdown(
            f"**Cleaned Submissions ({len(df)} crew members)** — *Deduplicated & Test entries removed*"
        )

        avail_summary = []
        for p in crew_list:
            count = sum(
                availability_matrix.get((p, s), False) for s in detected_shifts
            )
            avail_summary.append({name_col: p, "Shifts Offered": count})
        avail_summary_df = pd.DataFrame(avail_summary).sort_values(
            by="Shifts Offered", ascending=False
        )

        col_left, col_right = st.columns([1, 2])
        with col_left:
            st.dataframe(avail_summary_df, hide_index=True)

        with col_right:
            # Boolean visual grid
            display_grid = df[[name_col] + detected_shifts].copy()
            for s in detected_shifts:
                display_grid[s] = display_grid[s].apply(
                    lambda v: "✅ Yes" if is_available(v) else "❌ No"
                )
            st.dataframe(display_grid, hide_index=True)

    # -------------------------------------------------------------
    # TAB 3: OPTIMIZER & MASTER SCHEDULE
    # -------------------------------------------------------------
    with tab_optimizer:
        st.markdown("### Run Mathematical Optimization Engine")
        st.write(
            "This will solve for the optimal assignments respecting all time overlaps, availability, and capacity targets."
        )

        if st.button(
            "🚀 Run Optimization", type="primary", use_container_width=True
        ):
            with st.spinner("Finding optimal schedule with PuLP CBC..."):
                status, assignments, crew_totals, crew_shifts, shortfall_counts = (
                    solve_schedule(
                        crew_list=crew_list,
                        shifts=detected_shifts,
                        availability_matrix=availability_matrix,
                        shift_targets=shift_targets,
                        overlap_groups=overlap_groups,
                        max_shifts_per_person=max_shifts_slider,
                    )
                )

            # Save state in session_state
            st.session_state["opt_results"] = {
                "status": status,
                "assignments": assignments,
                "crew_totals": crew_totals,
                "crew_shifts": crew_shifts,
                "shortfall_counts": shortfall_counts,
                "shift_targets": shift_targets,
                "edited_config": edited_config,
            }

        # Check if results are stored
        if "opt_results" in st.session_state:
            results = st.session_state["opt_results"]
            status = results["status"]
            assignments = results["assignments"]
            shortfall_counts = results["shortfall_counts"]
            shift_targets_ref = results["shift_targets"]
            cfg = results["edited_config"]

            if status in ["Optimal", "Feasible"]:
                total_shortfall = sum(shortfall_counts.values())
                if total_shortfall == 0:
                    st.success("🎉 Schedule is fully staffed with zero gaps!")
                else:
                    st.warning(
                        f"⚠️ Staffing capacity warning: **{total_shortfall} unfilled slots** across the schedule. See flagged shifts below."
                    )

                # Build Master Schedule Table
                master_rows = []
                for _, row in cfg.iterrows():
                    s = row["Shift Name"]
                    role = row["Role / Scope"]
                    needed = shift_targets_ref[s]
                    assigned = assignments.get(s, [])
                    assigned_count = len(assigned)
                    missing = shortfall_counts.get(s, 0)

                    status_str = "✅ Full" if missing == 0 else f"🚨 Short {missing}"
                    assigned_str = ", ".join(assigned) if assigned else "—"

                    master_rows.append({
                        "Shift Name": s,
                        "Role": role,
                        "Needed": needed,
                        "Assigned Count": assigned_count,
                        "Status": status_str,
                        "Assigned Crew": assigned_str,
                    })

                master_df = pd.DataFrame(master_rows)
                st.dataframe(master_df, hide_index=True, use_container_width=True)

                # Export CSV
                csv_buffer = io.StringIO()
                master_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Master Schedule as CSV",
                    data=csv_buffer.getvalue(),
                    file_name="circus_bella_master_schedule.csv",
                    mime="text/csv",
                )
            else:
                st.error(
                    f"Solver failed to find a valid solution. Solver status: {status}"
                )

    # -------------------------------------------------------------
    # TAB 4: CREW-CENTRIC SUMMARY
    # -------------------------------------------------------------
    with tab_crew_view:
        if "opt_results" in st.session_state:
            results = st.session_state["opt_results"]
            crew_totals = results["crew_totals"]
            crew_shifts = results["crew_shifts"]

            st.markdown("### Individual Crew Rosters & Workload Distribution")

            crew_rows = []
            for p in sorted(crew_list):
                shifts_assigned = crew_shifts.get(p, [])
                crew_rows.append({
                    "Crew Member": p,
                    "Total Shifts Assigned": len(shifts_assigned),
                    "Assigned Shifts List": (
                        " | ".join(shifts_assigned)
                        if shifts_assigned
                        else "None (Available Backup)"
                    ),
                })

            crew_df = pd.DataFrame(crew_rows).sort_values(
                by="Total Shifts Assigned", ascending=False
            )
            st.dataframe(crew_df, hide_index=True, use_container_width=True)

            # Export Crew View CSV
            crew_csv_buffer = io.StringIO()
            crew_df.to_csv(crew_csv_buffer, index=False)
            st.download_button(
                label="📥 Download Crew Roster Summary as CSV",
                data=crew_csv_buffer.getvalue(),
                file_name="circus_bella_crew_summary.csv",
                mime="text/csv",
            )
        else:
            st.info(
                "Please run the optimization in Tab 3 to view the crew summary."
            )


if __name__ == "__main__":
    main()