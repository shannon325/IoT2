import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pulp import *

# 1. Data Loading and Preprocessing
def load_and_preprocess_data(filename):
    # Read both sheets from Excel
    data = pd.read_excel(filename, sheet_name=None)
    tasks_df = data["User & Task ID"]
    costs_df = data["PredictiveGuidelinePricing"]
    
    # Split User & Task ID into separate columns
    tasks_df[['User', 'Task ID']] = tasks_df['User & Task ID'].str.split('_task', expand=True)
    tasks_df['Task ID'] = tasks_df['Task ID'].astype(int)
    
    # Rename columns for clarity
    tasks_df = tasks_df.rename(columns={
        'Ready Time': 'ready_time',
        'Deadline': 'deadline',
        'Maximum scheduled energy per hour': 'max_energy',
        'Energy Demand': 'demand'
    })
    
    return tasks_df, costs_df['Unit Cost'].tolist()

# 2. Linear Programming Solver
def solve_user_optimization(user_tasks, unit_costs):
    # Create optimization problem
    prob = LpProblem(f"Energy_Scheduling_{user_tasks['User'].iloc[0]}", LpMinimize)
    
    # Decision variables
    time_slots = range(24)  # 24-hour period
    vars_dict = {}
    
    for _, task in user_tasks.iterrows():
        for t in range(task['ready_time'], task['deadline'] + 1):
            vars_dict[(task['Task ID'], t)] = LpVariable(
                f"Task_{task['Task ID']}_Hour_{t}",
                0,
                task['max_energy']
            )
    
    # Objective function
    prob += lpSum(vars_dict[(task['Task ID'], t)] * unit_costs[t]
                 for _, task in user_tasks.iterrows()
                 for t in range(task['ready_time'], task['deadline'] + 1))
    
    # Constraints
    for _, task in user_tasks.iterrows():
        # Energy demand constraint
        prob += lpSum(vars_dict[(task['Task ID'], t)]
                     for t in range(task['ready_time'], task['deadline'] + 1)) == task['demand']
    
    # Solve
    prob.solve(PULP_CBC_CMD(msg=False))
    
    # Extract results
    schedule = np.zeros(24)  # Initialize 24-hour schedule
    task_schedules = {}
    
    for (task_id, t), var in vars_dict.items():
        if var.value() is not None and var.value() > 0:
            schedule[t] += var.value()
            if task_id not in task_schedules:
                task_schedules[task_id] = np.zeros(24)
            task_schedules[task_id][t] = var.value()
    
    return {
        'total_cost': value(prob.objective),
        'schedule': schedule,
        'task_schedules': task_schedules
    }

# 3. Visualization
def plot_user_results(user_results, unit_costs, user_id):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot hourly energy consumption
    ax1.bar(range(24), user_results['schedule'], color='skyblue', alpha=0.6)
    ax1.set_title(f'Hourly Energy Consumption - User {user_id}')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Energy (kWh)')
    ax1.grid(True, alpha=0.3)
    
    # Plot unit costs
    ax2.plot(range(24), unit_costs, color='red', marker='o')
    ax2.set_title('Unit Costs Over 24 Hours')
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Cost')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"User_{user_id}_Results.png")
    plt.show()

# Main execution
def main():
    # Load data
    tasks_df, unit_costs = load_and_preprocess_data("IMSE7143CW2Input.xlsx")
    
    # Process each user
    all_results = {}
    for user in tasks_df['User'].unique():
        user_tasks = tasks_df[tasks_df['User'] == user]
        results = solve_user_optimization(user_tasks, unit_costs)
        all_results[user] = results
        
        # Print results
        print(f"\n=== Results for {user} ===")
        print(f"Total Cost: {results['total_cost']:.2f}")
        print(f"Average Hourly Energy: {np.mean(results['schedule']):.2f} kWh")
        print(f"Peak Energy Hour: {np.argmax(results['schedule'])} (with {np.max(results['schedule']):.2f} kWh)")
        
        # Plot results
        plot_user_results(results, unit_costs, user)

# Run the analysis
if __name__ == "__main__":
    main()