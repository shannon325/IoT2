import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pulp import *

# 1. Data Loading and Preprocessing
def load_and_preprocess_data(filename):
    data = pd.read_excel(filename, sheet_name=None)
    tasks_df = data["User & Task ID"]
    costs_df = data["PredictiveGuidelinePricing"]
    
    tasks_df[['User', 'Task ID']] = tasks_df['User & Task ID'].str.split('_task', expand=True)
    tasks_df['Task ID'] = tasks_df['Task ID'].astype(int)
    
    tasks_df = tasks_df.rename(columns={
        'Ready Time': 'ready_time',
        'Deadline': 'deadline',
        'Maximum scheduled energy per hour': 'max_energy',
        'Energy Demand': 'demand'
    })
    
    return tasks_df, costs_df['Unit Cost'].tolist()

# 2. Calculate quadratic price cost
def calculate_quadratic_cost(energy):
    """
    Calculate cost using quadratic pricing model
    Cost = 0.5 * E²
    where E is the total energy consumption
    """
    return 0.5 * energy * energy

# 3. Solve community optimization
def solve_community_optimization(tasks_df, unit_costs):
    prob = LpProblem("Community_Energy_Scheduling", LpMinimize)
    
    # Decision variables
    vars_dict = {}
    for _, task in tasks_df.iterrows():
        for t in range(task['ready_time'], task['deadline'] + 1):
            vars_dict[(task['User'], task['Task ID'], t)] = LpVariable(
                f"Task_{task['User']}_{task['Task ID']}_{t}",
                0,
                task['max_energy']
            )
    
    # Objective with linear approximation of quadratic costs
    # Note: We use linear approximation here since PuLP doesn't support quadratic objectives
    prob += lpSum(vars_dict[(task['User'], task['Task ID'], t)] * unit_costs[t]
                 for _, task in tasks_df.iterrows()
                 for t in range(task['ready_time'], task['deadline'] + 1))
    
    # Task completion constraints
    for _, task in tasks_df.iterrows():
        prob += lpSum(vars_dict[(task['User'], task['Task ID'], t)]
                     for t in range(task['ready_time'], task['deadline'] + 1)) == task['demand']
    
    # Solve the problem
    prob.solve(PULP_CBC_CMD(msg=False))
    
    # Extract results
    schedule = np.zeros(24)
    user_schedules = {user: np.zeros(24) for user in tasks_df['User'].unique()}
    task_schedules = {}
    
    for (user, task_id, t), var in vars_dict.items():
        if var.value() is not None and var.value() > 0:
            schedule[t] += var.value()
            user_schedules[user][t] += var.value()
            if (user, task_id) not in task_schedules:
                task_schedules[(user, task_id)] = np.zeros(24)
            task_schedules[(user, task_id)][t] = var.value()
    
    # Calculate costs using both pricing schemes
    linear_costs = np.zeros(24)
    quadratic_costs = np.zeros(24)
    for t in range(24):
        linear_costs[t] = schedule[t] * unit_costs[t]
        quadratic_costs[t] = calculate_quadratic_cost(schedule[t])
    
    return {
        'schedule': schedule,
        'user_schedules': user_schedules,
        'task_schedules': task_schedules,
        'linear_total_cost': np.sum(linear_costs),
        'quadratic_total_cost': np.sum(quadratic_costs),
        'linear_costs': linear_costs,
        'quadratic_costs': quadratic_costs
    }

# 4. Visualization
def plot_results(results, unit_costs):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Community Energy Consumption and Individual Users
    bottom = np.zeros(24)
    for user, schedule in results['user_schedules'].items():
        ax1.bar(range(24), schedule, bottom=bottom, label=f'User {user}', alpha=0.6)
        bottom += schedule
    ax1.set_title('Community Energy Consumption by User')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Energy (kWh)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cost Comparison
    ax2.plot(range(24), results['linear_costs'], 'b-', 
             label='Linear Cost (Guideline Price)', marker='o')
    ax2.plot(range(24), results['quadratic_costs'], 'r-', 
             label='Quadratic Cost (0.5E²)', marker='o')
    ax2.set_title('Cost Comparison')
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Cost')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Task3.png')
    plt.show()

# Main execution
def main():
    # Load data
    tasks_df, unit_costs = load_and_preprocess_data("IMSE7143CW2Input.xlsx")
    
    # Solve optimization
    results = solve_community_optimization(tasks_df, unit_costs)
    
    # Print results
    print("\n=== Results ===")
    print(f"Total cost (Linear pricing): {results['linear_total_cost']:.2f}")
    print(f"Total cost (Quadratic pricing): {results['quadratic_total_cost']:.2f}")
    
    # Plot results
    plot_results(results, unit_costs)

if __name__ == "__main__":
    main()