# Transportation + Scheduling Agent (API-free)

## Features
- Transportation: North-West Corner, Least Cost, VAM, and optional comparison
- Scheduling: FCFS, SPT, EDD, Priority Scheduling
- Streamlit editable tables and method selectors
- Chat-like keyword routing (no external API)
- Validation and step-by-step calculations
- Scheduling metrics: makespan, average completion time, average waiting time, average turnaround time, maximum lateness, maximum tardiness, total tardiness

## Run on Windows
```powershell
py -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Scheduling method meanings
- FCFS: jobs sorted by arrival time
- SPT: among currently available jobs, choose shortest processing time
- EDD: sort by due date; under the standard all-jobs-available single-machine assumptions, minimizes maximum lateness
- Priority: lower number means higher priority

The scheduling module is deterministic and requires no API key. The chat box uses keyword routing; the method dropdown is the reliable explicit selector.
Transportation automatic comparison chooses the lowest cost among the three initial-solution methods, not necessarily the global optimum.

## Files
- `app.py`: interface and routing
- `transportation.py`: existing transportation algorithms
- `scheduling.py`: new scheduling algorithms
- `requirements.txt`: dependencies
