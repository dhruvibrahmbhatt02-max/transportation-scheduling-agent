"""Deterministic single-machine scheduling algorithms with step-by-step traces."""
import math
import pandas as pd


REQUIRED_COLUMNS = ["Job", "Processing Time", "Arrival Time", "Due Date", "Priority"]


def validate_jobs(jobs):
    errors = []
    if not isinstance(jobs, pd.DataFrame):
        return ["Jobs must be entered in a table."]
    missing = [c for c in REQUIRED_COLUMNS if c not in jobs.columns]
    if missing:
        return [f"Missing required columns: {', '.join(missing)}."]
    if jobs.empty:
        return ["Enter at least one job."]
    if jobs["Job"].isna().any() or (jobs["Job"].astype(str).str.strip() == "").any():
        errors.append("Every job must have a non-empty name.")
    if jobs["Job"].astype(str).duplicated().any():
        errors.append("Job names must be unique.")
    for col in ["Processing Time", "Arrival Time", "Due Date", "Priority"]:
        values = pd.to_numeric(jobs[col], errors="coerce")
        if values.isna().any():
            errors.append(f"{col} must contain numeric values.")
        elif not all(math.isfinite(float(v)) for v in values):
            errors.append(f"{col} must contain finite values.")
    if errors:
        return errors
    if (jobs["Processing Time"].astype(float) <= 0).any():
        errors.append("Processing Time must be greater than zero.")
    if (jobs["Arrival Time"].astype(float) < 0).any():
        errors.append("Arrival Time cannot be negative.")
    if (jobs["Priority"].astype(float) < 1).any():
        errors.append("Priority must be 1 or greater; 1 means highest priority.")
    return errors


def _schedule(jobs, method, ordering, explanation):
    df = jobs.copy()
    df["Job"] = df["Job"].astype(str)
    df["Processing Time"] = df["Processing Time"].astype(float)
    df["Arrival Time"] = df["Arrival Time"].astype(float)
    df["Due Date"] = df["Due Date"].astype(float)
    df["Priority"] = df["Priority"].astype(float)

    clock = 0.0
    rows = []
    steps = [explanation]
    for rank, (_, job) in enumerate(ordering.iterrows(), start=1):
        arrival = float(job["Arrival Time"])
        processing = float(job["Processing Time"])
        start = max(clock, arrival)
        idle = max(0.0, arrival - clock)
        completion = start + processing
        waiting = start - arrival
        turnaround = completion - arrival
        lateness = completion - float(job["Due Date"])
        tardiness = max(0.0, lateness)
        rows.append({
            "Sequence": rank,
            "Job": job["Job"],
            "Arrival Time": arrival,
            "Processing Time": processing,
            "Due Date": float(job["Due Date"]),
            "Priority": float(job["Priority"]),
            "Start": start,
            "Completion": completion,
            "Waiting Time": waiting,
            "Turnaround Time": turnaround,
            "Lateness": lateness,
            "Tardiness": tardiness,
        })
        steps.append(
            f"Position {rank}: schedule job {job['Job']} at time {start:g} "
            f"(max(previous completion {clock:g}, arrival {arrival:g})). "
            f"Processing time = {processing:g}; completion = {start:g} + {processing:g} = {completion:g}; "
            f"waiting = {start:g} - {arrival:g} = {waiting:g}; "
            f"turnaround = {completion:g} - {arrival:g} = {turnaround:g}; "
            f"lateness = {completion:g} - due date {float(job['Due Date']):g} = {lateness:g}; "
            f"tardiness = max(0, lateness) = {tardiness:g}."
        )
        if idle > 0:
            steps.append(f"The machine is idle from time {clock:g} to {arrival:g} because the next job has not arrived.")
        clock = completion

    result_df = pd.DataFrame(rows)
    metrics = {
        "makespan": float(result_df["Completion"].max()),
        "average_completion_time": float(result_df["Completion"].mean()),
        "average_waiting_time": float(result_df["Waiting Time"].mean()),
        "average_turnaround_time": float(result_df["Turnaround Time"].mean()),
        "maximum_lateness": float(result_df["Lateness"].max()),
        "maximum_tardiness": float(result_df["Tardiness"].max()),
        "total_tardiness": float(result_df["Tardiness"].sum()),
    }
    steps.append(
        "Summary: makespan = maximum completion time; average completion time = mean of completion times; "
        "average waiting time = mean of waiting times; maximum lateness = maximum(completion − due date); "
        "tardiness is max(0, lateness)."
    )
    return {"method": method, "schedule": result_df, "metrics": metrics, "steps": steps}


def _validate_or_raise(jobs):
    errors = validate_jobs(jobs)
    if errors:
        raise ValueError("\n".join(errors))


def schedule_fcfs(jobs):
    _validate_or_raise(jobs)
    ordering = jobs.sort_values(["Arrival Time", "Job"], kind="stable")
    return _schedule(
        jobs, "FCFS (First Come, First Served)", ordering,
        "FCFS orders jobs by earliest arrival time. Ties are broken by job name for reproducibility."
    )


def schedule_spt(jobs):
    _validate_or_raise(jobs)
    # For release times, choose the shortest available job whenever the machine is free.
    remaining = jobs.copy()
    chosen_indices = []
    now = 0.0
    while len(remaining):
        available = remaining[remaining["Arrival Time"].astype(float) <= now]
        if available.empty:
            next_arrival = remaining["Arrival Time"].astype(float).min()
            now = float(next_arrival)
            available = remaining[remaining["Arrival Time"].astype(float) <= now]
        chosen = available.sort_values(["Processing Time", "Arrival Time", "Job"], kind="stable").index[0]
        chosen_indices.append(chosen)
        now = max(now, float(remaining.loc[chosen, "Arrival Time"])) + float(remaining.loc[chosen, "Processing Time"])
        remaining = remaining.drop(index=chosen)
    ordering = jobs.loc[chosen_indices]
    return _schedule(
        jobs, "SPT (Shortest Processing Time)", ordering,
        "SPT selects the shortest processing time among jobs available when the machine becomes free. "
        "With all jobs available at time zero, SPT minimizes average completion/flow time."
    )


def schedule_edd(jobs):
    _validate_or_raise(jobs)
    ordering = jobs.sort_values(["Due Date", "Arrival Time", "Job"], kind="stable")
    return _schedule(
        jobs, "EDD (Earliest Due Date)", ordering,
        "EDD orders jobs by earliest due date; ties are broken by arrival time and then job name. "
        "For the standard single-machine setting with all jobs available at time zero, EDD minimizes maximum lateness."
    )


def schedule_priority(jobs):
    _validate_or_raise(jobs)
    ordering = jobs.sort_values(["Priority", "Arrival Time", "Job"], kind="stable")
    return _schedule(
        jobs, "Priority Scheduling", ordering,
        "Priority scheduling uses the convention 1 = highest priority. Ties are broken by arrival time and then job name."
    )
