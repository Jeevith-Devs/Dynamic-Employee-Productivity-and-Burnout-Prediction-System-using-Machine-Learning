"""
generate_synthetic_employee_weeks.py

Creates a synthetic employee-week dataset for Productivity Index (PI) and Burnout Risk.
Default: 1000 employees * 20 weeks = 20,000 rows.

Usage:
    python generate_synthetic_employee_weeks.py

Adjust parameters at the top of the file.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# -----------------------------
# CONFIGURABLE PARAMETERS
# -----------------------------
SEED = 42
n_employees = 1000       # number of distinct employees
n_weeks = 20             # weeks per employee -> n_employees * n_weeks ~ 20k
start_date = datetime(2024, 1, 1)
departments = ['engineering', 'sales', 'hr', 'support', 'product', 'marketing']
role_levels = ['junior', 'mid', 'senior', 'lead']

# target fraction of rows that should be labeled "High" burnout (0.0 .. 1.0)
# If actual generated high < target, the script will nudge random rows to reach it.
target_high_frac = 0.03  # e.g., 3% high burnout; increase to get more high examples

output_csv = "synthetic_employee_weeks_20k.csv"
np.random.seed(SEED)

# -----------------------------
# GENERATION LOGIC
# -----------------------------
rows = []
for emp in range(1, n_employees + 1):
    # per-employee latent traits
    base_skill = np.random.normal(70, 10)
    base_skill = float(np.clip(base_skill, 30, 100))
    baseline_stress = float(np.random.beta(2, 6))    # low by default
    baseline_motivation = float(np.random.beta(5, 2))# tends higher
    dept = np.random.choice(departments, p=[0.25,0.15,0.08,0.2,0.17,0.15])
    role = np.random.choice(role_levels, p=[0.25,0.45,0.2,0.1])
    manager_id = int(np.ceil(emp / 10))
    dept_task_diff_factor = {'engineering':1.1, 'sales':0.9, 'hr':0.8, 'support':0.9, 'product':1.0, 'marketing':0.95}[dept]
    dept_msg_factor = {'engineering':0.8, 'sales':1.3, 'hr':0.9, 'support':1.2, 'product':1.0, 'marketing':1.1}[dept]

    stress = baseline_stress
    motivation = baseline_motivation

    for w in range(n_weeks):
        week_start = start_date + timedelta(weeks=w)
        # workload — seasonal + occasional deadline spikes
        seasonal = 1.0 + 0.25 * np.sin(2 * np.pi * (w / 13))
        deadline_spike = 1.0 + (0.5 if ((w + emp) % 8 == 6) else 0.0)  # distribute spikes across employees
        workload = float(np.clip(np.random.normal(1.0, 0.18) * seasonal * deadline_spike, 0.5, 1.8))

        # hours/overtime
        expected_hours = 40
        overtime = float(max(0.0, np.random.normal((workload - 1) * 12, 3)))
        hours_worked = float(max(20.0, expected_hours + overtime + np.random.normal(0, 3)))

        # idle ratio
        idle_time_ratio = float(np.clip(np.random.normal(0.12 + 0.15*stress + 0.02*(1-workload), 0.05), 0.02, 0.6))

        # messages and meetings
        msg_count = int(max(0, np.random.normal(50 * dept_msg_factor * (1 - idle_time_ratio), 12)))
        meetings = int(np.clip(np.random.normal(4 * workload, 1.8), 0, 30))

        # tasks and difficulty
        avg_task_difficulty = float(np.clip(np.random.normal(2.8 * workload * dept_task_diff_factor, 0.5), 1.0, 5.0))
        tasks_completed = int(np.round(np.clip(np.random.normal((base_skill/10) * motivation / workload, 1.5), 0, 50)))

        # on-time
        task_on_time_ratio = float(np.clip(np.random.normal(0.92 - 0.12*(workload-1) - 0.15*stress, 0.06), 0.2, 1.0))

        # sentiment
        avg_sentiment = float(np.clip(np.random.normal(0.35 - 0.6*stress - 0.15*(workload-1), 0.25), -1, 1))

        # leave days (probabilistic)
        p0 = 0.94 - 0.03 * stress
        p1 = 0.05 + 0.04 * stress
        p2 = 0.006 + 0.01 * stress
        p3 = 0.004
        probs = np.array([p0, p1, p2, p3])
        probs = np.clip(probs, 0, 1)
        probs = probs / probs.sum()
        leave_days = int(np.random.choice([0,1,2,3], p=probs))

        # Productivity Index (PI)
        pi_raw = (
            0.45 * (tasks_completed / 20) + 
            0.18 * task_on_time_ratio + 
            0.18 * (1 - idle_time_ratio) + 
            0.12 * (0.5 + avg_sentiment/2) - 
            0.07 * (meetings / 10)
        )
        PI = float(np.clip(100 * (pi_raw / 1.2), 0, 100))

        # burnout score and class
        burnout_score = float(0.45 * stress + 0.25 * np.tanh(overtime / 12) + 0.15 * (1 - task_on_time_ratio) + 0.15 * (1 - (PI/100)))
        if burnout_score < 0.33:
            burnout_risk = "Low"
        elif burnout_score < 0.66:
            burnout_risk = "Medium"
        else:
            burnout_risk = "High"

        rows.append({
            "emp_id": emp,
            "week_start": week_start.date().isoformat(),
            "week": w+1,
            "department": dept,
            "role_level": role,
            "manager_id": manager_id,
            "base_skill": round(base_skill,1),
            "baseline_stress": round(baseline_stress,3),
            "workload": round(workload,3),
            "tasks_completed": tasks_completed,
            "avg_task_difficulty": round(avg_task_difficulty,2),
            "hours_worked": round(hours_worked,1),
            "overtime_hours": round(overtime,2),
            "idle_time_ratio": round(idle_time_ratio,3),
            "msg_count": msg_count,
            "avg_sentiment": round(avg_sentiment,3),
            "meetings": meetings,
            "leave_days": leave_days,
            "task_on_time_ratio": round(task_on_time_ratio,3),
            "productivity_index": round(PI,2),
            "burnout_score": round(burnout_score,3),
            "burnout_risk": burnout_risk
        })

        # carry-over updates
        stress = float(np.clip(stress + 0.08*(workload-1) - 0.12*(leave_days>0) - 0.04*(PI>70) + np.random.normal(0,0.02), 0, 1))
        motivation = float(np.clip(motivation - 0.06*stress + 0.02*(PI>75) + np.random.normal(0,0.03), 0.2, 1.0))

df = pd.DataFrame(rows)
total_rows = df.shape[0]
print(f"Generated rows: {total_rows}")

# -----------------------------
# ADJUST to meet target_high_frac (if needed)
# -----------------------------
current_high = (df['burnout_risk'] == 'High').sum()
current_high_frac = current_high / total_rows
desired_high = int(round(target_high_frac * total_rows))

print(f"Current High count: {current_high} ({current_high_frac:.4f}), Desired: {desired_high}")

if current_high < desired_high:
    # Select candidates that are not High and nudge them by increasing 'burnout_score' components:
    need = desired_high - current_high
    print(f"Injecting High cases by nudging {need} rows...")
    # Choose candidates proportionally from Medium and Low (prefer recent weeks)
    candidates = df[df['burnout_risk'] != 'High'].sample(n=need, random_state=SEED)
    for idx in candidates.index:
        # bump stress and overtime a bit for the chosen row
        df.at[idx, 'baseline_stress'] = min(1.0, df.at[idx, 'baseline_stress'] + np.random.uniform(0.2, 0.45))
        # increase overtime artificially
        df.at[idx, 'overtime_hours'] = df.at[idx, 'overtime_hours'] + np.random.uniform(4, 18)
        # decrease on-time ratio and sentiment
        df.at[idx, 'task_on_time_ratio'] = max(0.01, df.at[idx, 'task_on_time_ratio'] - np.random.uniform(0.15, 0.4))
        df.at[idx, 'avg_sentiment'] = max(-1.0, df.at[idx, 'avg_sentiment'] - np.random.uniform(0.3, 0.8))
        # recompute PI and burnout score & class for that row
        tasks_completed = df.at[idx, 'tasks_completed']
        idle_time_ratio = df.at[idx, 'idle_time_ratio']
        meetings = df.at[idx, 'meetings']
        task_on_time_ratio = df.at[idx, 'task_on_time_ratio']
        avg_sentiment = df.at[idx, 'avg_sentiment']
        pi_raw = (
            0.45 * (tasks_completed / 20) + 
            0.18 * task_on_time_ratio + 
            0.18 * (1 - idle_time_ratio) + 
            0.12 * (0.5 + avg_sentiment/2) - 
            0.07 * (meetings / 10)
        )
        PI = float(np.clip(100 * (pi_raw / 1.2), 0, 100))
        df.at[idx, 'productivity_index'] = round(PI, 2)
        burnout_score = float(0.45 * df.at[idx, 'baseline_stress'] + 0.25 * np.tanh(df.at[idx, 'overtime_hours'] / 12) + 0.15 * (1 - task_on_time_ratio) + 0.15 * (1 - (PI/100)))
        df.at[idx, 'burnout_score'] = round(burnout_score, 3)
        # new class
        if burnout_score < 0.33:
            df.at[idx, 'burnout_risk'] = "Low"
        elif burnout_score < 0.66:
            df.at[idx, 'burnout_risk'] = "Medium"
        else:
            df.at[idx, 'burnout_risk'] = "High"

# Final counts & save
counts = df['burnout_risk'].value_counts().to_dict()
print("Final burnout counts:", counts)
print("Saving CSV to:", output_csv)
df.to_csv(output_csv, index=False)

# quick summary stats
print("\nSample rows:")
print(df.head(6).to_string(index=False))
print("\nSummary of productivity_index (min, mean, median, max):",
      df['productivity_index'].min(), round(df['productivity_index'].mean(),2),
      df['productivity_index'].median(), df['productivity_index'].max())
