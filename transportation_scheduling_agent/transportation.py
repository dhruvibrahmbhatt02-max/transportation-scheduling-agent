"""Transportation algorithms preserved for the combined app."""
import numpy as np
TOL = 1e-9

def validate_problem(costs, supply, demand):
    messages = []
    try:
        costs = np.asarray(costs, dtype=float)
        supply = np.asarray(supply, dtype=float)
        demand = np.asarray(demand, dtype=float)
    except (TypeError, ValueError):
        return ["All costs, supply values, and demand values must be numeric."]
    if costs.ndim != 2 or costs.shape[0] == 0 or costs.shape[1] == 0:
        return ["The cost matrix must have at least one source and one destination."]
    if supply.ndim != 1 or len(supply) != costs.shape[0]:
        messages.append("Supply must contain exactly one value per source.")
    if demand.ndim != 1 or len(demand) != costs.shape[1]:
        messages.append("Demand must contain exactly one value per destination.")
    if not np.all(np.isfinite(costs)): messages.append("Every cost must be finite.")
    if not np.all(np.isfinite(supply)): messages.append("Every supply value must be finite.")
    if not np.all(np.isfinite(demand)): messages.append("Every demand value must be finite.")
    if np.any(costs < 0): messages.append("Costs cannot be negative.")
    if np.any(supply < 0): messages.append("Supply cannot be negative.")
    if np.any(demand < 0): messages.append("Demand cannot be negative.")
    if messages: return messages
    if abs(float(supply.sum()) - float(demand.sum())) > TOL:
        messages.append(f"Problem is unbalanced: total supply = {supply.sum():g}, total demand = {demand.sum():g}. Adjust totals to match.")
    return messages

def _prepare(costs, supply, demand):
    costs, supply, demand = np.asarray(costs, float), np.asarray(supply, float).copy(), np.asarray(demand, float).copy()
    errors = validate_problem(costs, supply, demand)
    if errors: raise ValueError("\n".join(errors))
    return costs, supply, demand

def _finish(method, costs, allocation, steps):
    total = float(np.sum(costs * allocation))
    terms = [f"{allocation[i,j]:g} × {costs[i,j]:g}" for i,j in zip(*np.where(allocation > TOL))]
    return {"method":method, "allocation":allocation, "total_cost":total,
            "cost_expression":" + ".join(terms) if terms else "0", "steps":steps}

def solve_northwest_corner(costs, supply, demand):
    costs, s, d = _prepare(costs, supply, demand)
    m,n = costs.shape; a=np.zeros((m,n)); i=j=0
    steps=["Start at the top-left cell. North-West Corner does not consider unit costs when selecting cells."]
    while i<m and j<n:
        q=min(s[i],d[j]); bs,bd=s[i],d[j]; a[i,j]+=q; s[i]-=q; d[j]-=q
        steps.append(f"Allocate min(S{i+1}={bs:g}, D{j+1}={bd:g}) = {q:g} to (S{i+1}, D{j+1}); unit cost {costs[i,j]:g}. Remaining S{i+1}={s[i]:g}, D{j+1}={d[j]:g}.")
        rd=s[i]<=TOL; cd=d[j]<=TOL
        if rd: i+=1
        if cd: j+=1
    steps.append("Total cost is the sum of allocation × unit cost across all cells.")
    return _finish("North-West Corner",costs,a,steps)

def solve_least_cost(costs, supply, demand):
    costs,s,d=_prepare(costs,supply,demand); m,n=costs.shape; a=np.zeros((m,n))
    rows=set(range(m)); cols=set(range(n)); steps=["Repeatedly choose the lowest-cost active cell. Ties go to the earliest row, then column."]
    k=0
    while rows and cols:
        candidates=[(costs[i,j],i,j) for i in rows for j in cols if s[i]>TOL and d[j]>TOL]
        if not candidates: break
        c,i,j=min(candidates,key=lambda x:(x[0],x[1],x[2]))
        q=min(s[i],d[j]); bs,bd=s[i],d[j]; a[i,j]+=q; s[i]-=q; d[j]-=q; k+=1
        steps.append(f"Iteration {k}: lowest active cost {c:g} at (S{i+1}, D{j+1}); allocate min({bs:g}, {bd:g}) = {q:g}. Remaining S{i+1}={s[i]:g}, D{j+1}={d[j]:g}.")
        if s[i]<=TOL: rows.discard(i)
        if d[j]<=TOL: cols.discard(j)
    if np.any(s>TOL) or np.any(d>TOL): raise ValueError("Could not satisfy all supply and demand.")
    steps.append("Total cost is the sum of allocation × unit cost across all cells.")
    return _finish("Least Cost",costs,a,steps)

def _penalty(costs, indices, fixed, axis):
    vals=sorted((costs[fixed,j],j) for j in indices) if axis=="row" else sorted((costs[i,fixed],i) for i in indices)
    if not vals: return None
    return (vals[1][0]-vals[0][0] if len(vals)>1 else vals[0][0], vals[0][1], vals[0][0])

def solve_vam(costs, supply, demand):
    costs,s,d=_prepare(costs,supply,demand); m,n=costs.shape; a=np.zeros((m,n))
    rows=set(range(m)); cols=set(range(n))
    steps=["Calculate each active row/column penalty as the difference between its two lowest active costs. If one cost remains, use it as the penalty."]
    k=0
    while rows and cols:
        rows={i for i in rows if s[i]>TOL}; cols={j for j in cols if d[j]>TOL}
        if not rows or not cols: break
        opts=[]
        for i in sorted(rows):
            p,j,c=_penalty(costs,sorted(cols),i,"row")
            opts.append((p,c,0,i,j,"row"))
        for j in sorted(cols):
            p,i,c=_penalty(costs,sorted(rows),j,"column")
            opts.append((p,c,1,i,j,"column"))
        p,c,_,i,j,kind=sorted(opts,key=lambda x:(-x[0],x[1],x[2],x[3],x[4]))[0]
        q=min(s[i],d[j]); bs,bd=s[i],d[j]; a[i,j]+=q; s[i]-=q; d[j]-=q; k+=1
        steps.append(f"Iteration {k}: select {kind} {i+1 if kind=='row' else j+1}, penalty {p:g}; cheapest cell (S{i+1}, D{j+1}) costs {c:g}. Allocate min({bs:g}, {bd:g}) = {q:g}. Remaining S{i+1}={s[i]:g}, D{j+1}={d[j]:g}.")
    if np.any(s>TOL) or np.any(d>TOL): raise ValueError("VAM could not satisfy all supply and demand.")
    steps.append("Total cost is the sum of allocation × unit cost across all cells.")
    return _finish("Vogel's Approximation Method (VAM)",costs,a,steps)

def compare_methods(costs,supply,demand):
    return [solve_northwest_corner(costs,supply,demand),solve_least_cost(costs,supply,demand),solve_vam(costs,supply,demand)]
