import streamlit as st
import pandas as pd
import numpy as np

from transportation import (
    solve_northwest_corner, solve_least_cost, solve_vam,
    validate_problem as validate_transportation, compare_methods
)
from scheduling import (
    validate_jobs, schedule_fcfs, schedule_spt, schedule_edd, schedule_priority
)

st.set_page_config(page_title="Optimization Methods Agent", page_icon="🧮", layout="wide")
st.title("🧮 Optimization Methods Agent")
st.caption("Transportation Problems + Single-Machine Scheduling · API-free")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def render_result(result):
    st.markdown(f"### {result['method']}")
    st.metric("Total transportation cost" if "total_cost" in result else "Average completion time",
              f"{result['total_cost']:,.2f}" if "total_cost" in result else f"{result['metrics']['average_completion_time']:.2f}")
    if "allocation" in result:
        st.dataframe(pd.DataFrame(result["allocation"]), use_container_width=True)
        st.markdown(f"**Cost calculation:** {result['cost_expression']} = **{result['total_cost']:,.2f}**")
    else:
        st.dataframe(pd.DataFrame(result["schedule"]), use_container_width=True, hide_index=True)
        st.markdown("**Performance metrics**")
        st.json(result["metrics"])
    st.markdown("**Step-by-step calculations**")
    for i, step in enumerate(result["steps"], start=1):
        st.markdown(f"{i}. {step}")

transport_tab, schedule_tab = st.tabs(["🚚 Transportation", "📅 Scheduling"])

with transport_tab:
    st.header("Transportation Problem Solver")
    st.write("Enter the cost matrix, supply, and demand. This module preserves the three original transportation methods.")
    with st.sidebar:
        st.subheader("Transportation dimensions")
        tm = st.number_input("Number of sources", 1, 12, 3, key="tm")
        tn = st.number_input("Number of destinations", 1, 12, 3, key="tn")

    source_names = [f"Source {i+1}" for i in range(int(tm))]
    dest_names = [f"Destination {j+1}" for j in range(int(tn))]

    def resize_df(key, rows, cols, row_names, col_names):
        old = st.session_state.get(key)
        if isinstance(old, pd.DataFrame) and old.shape == (rows, cols):
            return old
        new = pd.DataFrame(0.0, index=row_names, columns=col_names)
        if isinstance(old, pd.DataFrame):
            r, c = min(rows, old.shape[0]), min(cols, old.shape[1])
            if r and c:
                new.iloc[:r, :c] = old.iloc[:r, :c].to_numpy()
        st.session_state[key] = new
        return new

    cdf = resize_df("tcost", int(tm), int(tn), source_names, dest_names)
    sdf = resize_df("tsupply", int(tm), 1, source_names, ["Supply"])
    ddf = resize_df("tdemand", 1, int(tn), ["Demand"], dest_names)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Cost matrix")
        cdf = st.data_editor(cdf, key="tcost_editor", num_rows="fixed", use_container_width=True)
        st.session_state.tcost = cdf
    with c2:
        st.subheader("Supply")
        sdf = st.data_editor(sdf, key="tsupply_editor", num_rows="fixed", use_container_width=True)
        st.session_state.tsupply = sdf
        st.subheader("Demand")
        ddf = st.data_editor(ddf, key="tdemand_editor", num_rows="fixed", use_container_width=True)
        st.session_state.tdemand = ddf

    if st.button("Load transportation sample", key="load_tsample"):
        st.session_state.tcost = pd.DataFrame([[8,6,10],[9,12,13],[14,9,16]],
            index=["Source 1","Source 2","Source 3"],
            columns=["Destination 1","Destination 2","Destination 3"], dtype=float)
        st.session_state.tsupply = pd.DataFrame({"Supply":[20,30,25]},
            index=["Source 1","Source 2","Source 3"], dtype=float)
        st.session_state.tdemand = pd.DataFrame([[10,35,30]], index=["Demand"],
            columns=["Destination 1","Destination 2","Destination 3"], dtype=float)
        st.rerun()

    tmethod = st.selectbox("Choose transportation method",
        ["North-West Corner", "Least Cost", "Vogel's Approximation Method (VAM)", "Compare all three"],
        key="tmethod")
    tchat = st.text_input("Or describe your transportation request",
        placeholder="e.g. Compare all three methods", key="tchat")
    if st.button("Solve transportation problem", type="primary", key="solve_t"):
        costs = cdf.to_numpy(dtype=float)
        supply = sdf["Supply"].to_numpy(dtype=float)
        demand = ddf.iloc[0].to_numpy(dtype=float)
        errors = validate_transportation(costs, supply, demand)
        request = (tchat or "").lower()
        chosen = tmethod
        if any(k in request for k in ["compare", "minimum cost", "lowest cost", "choose yourself", "best method"]):
            chosen = "Compare all three"
        elif "north-west" in request or "northwest" in request or "north west" in request:
            chosen = "North-West Corner"
        elif "least cost" in request:
            chosen = "Least Cost"
        elif "vam" in request or "vogel" in request:
            chosen = "Vogel's Approximation Method (VAM)"
        if errors:
            st.error("Please correct these inputs:\n\n- " + "\n- ".join(errors))
        else:
            if chosen == "North-West Corner":
                results = [solve_northwest_corner(costs, supply, demand)]
            elif chosen == "Least Cost":
                results = [solve_least_cost(costs, supply, demand)]
            elif chosen == "Vogel's Approximation Method (VAM)":
                results = [solve_vam(costs, supply, demand)]
            else:
                results = compare_methods(costs, supply, demand)
            if len(results) > 1:
                winner = min(results, key=lambda x: x["total_cost"])
                st.success(f"Lowest cost among the three initial-solution methods: {winner['method']} = {winner['total_cost']:,.2f}")
                st.dataframe(pd.DataFrame([{"Method": r["method"], "Total cost": r["total_cost"]} for r in results]).sort_values("Total cost"), hide_index=True)
                st.caption("This comparison does not guarantee global optimality.")
            for result in results:
                with st.expander(f"{result['method']} — cost {result['total_cost']:,.2f}", expanded=True):
                    render_result(result)

with schedule_tab:
    st.header("Single-Machine Scheduling")
    st.write("Each job has a processing time. Optional arrival times, due dates, and priorities are used by the relevant methods.")
    num_jobs = st.number_input("Number of jobs", min_value=1, max_value=30, value=5, step=1, key="num_jobs")
    default_jobs = pd.DataFrame({
        "Job": [f"J{i+1}" for i in range(int(num_jobs))],
        "Processing Time": [6, 2, 8, 3, 5][:int(num_jobs)] if int(num_jobs) <= 5 else [1]*int(num_jobs),
        "Arrival Time": [0]*int(num_jobs),
        "Due Date": [10, 8, 20, 12, 15][:int(num_jobs)] if int(num_jobs) <= 5 else [10]*int(num_jobs),
        "Priority": [3, 1, 4, 2, 5][:int(num_jobs)] if int(num_jobs) <= 5 else list(range(1, int(num_jobs)+1)),
    })
    existing = st.session_state.get("jobs_df")
    if not isinstance(existing, pd.DataFrame) or len(existing) != int(num_jobs):
        if isinstance(existing, pd.DataFrame):
            r = min(len(existing), int(num_jobs))
            default_jobs.iloc[:r, :min(default_jobs.shape[1], existing.shape[1])] = existing.iloc[:r, :min(default_jobs.shape[1], existing.shape[1])].to_numpy()
        st.session_state.jobs_df = default_jobs

    jobs_df = st.data_editor(
        st.session_state.jobs_df, key="jobs_editor", num_rows="fixed", use_container_width=True,
        column_config={
            "Job": st.column_config.TextColumn("Job", required=True),
            "Processing Time": st.column_config.NumberColumn("Processing Time", min_value=0.000001, step=1.0),
            "Arrival Time": st.column_config.NumberColumn("Arrival Time", min_value=0, step=1.0),
            "Due Date": st.column_config.NumberColumn("Due Date", step=1.0),
            "Priority": st.column_config.NumberColumn("Priority (1 = highest)", min_value=1, step=1),
        }
    )
    st.session_state.jobs_df = jobs_df

    smethod = st.selectbox("Choose scheduling method", [
        "FCFS (First Come, First Served)",
        "SPT (Shortest Processing Time)",
        "EDD (Earliest Due Date)",
        "Priority Scheduling",
        "Compare all scheduling methods",
    ], key="smethod")
    schat = st.text_input("Scheduling request (optional)",
        placeholder="e.g. Use shortest processing time", key="schat")
    st.caption("Priority convention: 1 is the highest priority. Ties are resolved deterministically.")
    if st.button("Schedule jobs", type="primary", key="solve_s"):
        jobs = jobs_df.copy()
        jobs.columns = [str(c).strip() for c in jobs.columns]
        errors = validate_jobs(jobs)
        req = (schat or "").lower()
        method = smethod
        if any(k in req for k in ["fcfs", "first come", "arrival order"]):
            method = "FCFS (First Come, First Served)"
        elif any(k in req for k in ["shortest", "spt", "sjf"]):
            method = "SPT (Shortest Processing Time)"
        elif any(k in req for k in ["earliest due", "edd", "deadline"]):
            method = "EDD (Earliest Due Date)"
        elif "priority" in req:
            method = "Priority Scheduling"
        elif any(k in req for k in ["compare", "all methods", "compare methods"]):
            method = "Compare all scheduling methods"

        if errors:
            st.error("Please correct these inputs:\n\n- " + "\n- ".join(errors))
        else:
            methods = {
                "FCFS (First Come, First Served)": schedule_fcfs,
                "SPT (Shortest Processing Time)": schedule_spt,
                "EDD (Earliest Due Date)": schedule_edd,
                "Priority Scheduling": schedule_priority,
            }
            if method == "Compare all scheduling methods":
                results = [fn(jobs) for fn in methods.values()]
                comparison = pd.DataFrame([
                    {
                        "Method": r["method"],
                        "Makespan": r["metrics"]["makespan"],
                        "Average completion time": r["metrics"]["average_completion_time"],
                        "Average waiting time": r["metrics"]["average_waiting_time"],
                        "Maximum lateness": r["metrics"]["maximum_lateness"],
                        "Maximum tardiness": r["metrics"]["maximum_tardiness"],
                        "Total tardiness": r["metrics"]["total_tardiness"],
                    } for r in results
                ])
                st.subheader("Scheduling method comparison")
                st.dataframe(comparison, use_container_width=True, hide_index=True)
                st.caption("Different methods optimize different objectives. Choose the metric relevant to your assignment; no single method is best for every objective.")
            else:
                results = [methods[method] (jobs)]
            for result in results:
                with st.expander(f"{result['method']} — average completion time {result['metrics']['average_completion_time']:.2f}", expanded=True):
                    render_result(result)
                    st.subheader("Processing-time chart")
                    st.bar_chart(result["schedule"], x="Job", y="Processing Time")
                    st.caption("The table gives exact start and completion times; this chart compares job processing durations.")

st.divider()
with st.expander("Agent conversation log"):
    st.write("This API-free agent uses deterministic keyword routing plus a user-selectable method menu. All mathematical results are calculated by Python.")
