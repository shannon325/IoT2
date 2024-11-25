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

# 2. Solve optimization for one user
def solve_user_optimization(user_tasks, unit_costs, peak_hours=None, peak_hour_limit=None):
    prob = LpProblem(f"Energy_Scheduling_{user_tasks['User'].iloc[0]}", LpMinimize)
    
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
    
    # Basic constraints
    for _, task in user_tasks.iterrows():
        prob += lpSum(vars_dict[(task['Task ID'], t)]
                     for t in range(task['ready_time'], task['deadline'] + 1)) == task['demand']
    
    # Peak hour constraints
    if peak_hours is not None and peak_hour_limit is not None:
        for hour in peak_hours:
            prob += lpSum(vars_dict[(task['Task ID'], hour)]
                        for _, task in user_tasks.iterrows()
                        if hour >= task['ready_time'] and hour <= task['deadline']) <= peak_hour_limit
    
    prob.solve(PULP_CBC_CMD(msg=False))
    
    schedule = np.zeros(24)
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

# 3. Find peak hours
def find_peak_hours(schedules, n=3):
    total_energy = np.sum(schedules, axis=0)
    peak_indices = np.argsort(total_energy)[-n:]
    return peak_indices, total_energy

# 4. Check if any hour exceeds limit
def check_peak_limit(schedules, limit=10.0):
    total_energy = np.sum(schedules, axis=0)
    return np.max(total_energy) <= limit, total_energy

# 5. Visualization
def plot_results(all_iteration_data, unit_costs):
    n_iterations = len(all_iteration_data)
    fig, axes = plt.subplots(n_iterations + 1, 1, figsize=(15, 5*n_iterations))
    
    if n_iterations == 1:
        axes = [axes]
    
    # Plot each iteration
    for i, data in enumerate(all_iteration_data):
        total_energy = data['total_energy']
        peak_hours = data['peak_hours']
        
        axes[i].bar(range(24), total_energy, color='skyblue', alpha=0.6)
        axes[i].set_title(f'Iteration {i+1} - Total Energy Consumption\n'
                        f'Max Energy: {np.max(total_energy):.2f} kWh, '
                        f'Total Cost: {data["total_cost"]:.2f}')
        axes[i].set_xlabel('Hour')
        axes[i].set_ylabel('Energy (kWh)')
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=10, color='red', linestyle='--', label='10 kWh limit')
        
        # Mark peak hours
        for hour in peak_hours:
            axes[i].axvline(x=hour, color='red', linestyle=':', alpha=0.5)
            # axes[i].text(hour, total_energy[hour], f'{total_energy[hour]:.1f}', 
            #             rotation=90, verticalalignment='bottom')
        
        axes[i].legend()
    
    # Plot unit costs
    axes[-1].plot(range(24), unit_costs, color='red', marker='o')
    axes[-1].set_title('Unit Costs Over 24 Hours')
    axes[-1].set_xlabel('Hour')
    axes[-1].set_ylabel('Cost')
    axes[-1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Task2.png')
    plt.show()
    

# Main execution
def main():
    # Load data
    tasks_df, unit_costs = load_and_preprocess_data("IMSE7143CW2Input.xlsx")
    
    iteration = 1
    maximum_iterations = 6
    all_iteration_data = []
    best_solution = None
    best_max_energy = float('inf')
    
    while True:
        print(f"\n=== Iteration {iteration} ===")
        
        current_schedules = []
        iteration_results = {}
        
        # First iteration without constraints
        peak_hours = None
        peak_hour_limit = None
        
        if iteration > 1:
            # Use previous iteration's peak hours
            peak_hours = all_iteration_data[-1]['peak_hours']
            peak_hour_limit = 2.0  # Limit per user for peak hours
        
        # Optimize for each user
        for user in tasks_df['User'].unique():
            user_tasks = tasks_df[tasks_df['User'] == user]
            results = solve_user_optimization(user_tasks, unit_costs, peak_hours, peak_hour_limit)
            iteration_results[user] = results
            current_schedules.append(results['schedule'])
        
        # Analyze results
        peak_hours, total_energy = find_peak_hours(current_schedules)
        max_energy = np.max(total_energy)
        total_cost = sum(result['total_cost'] for result in iteration_results.values())
        
        # Store iteration data
        iteration_data = {
            'total_energy': total_energy,
            'peak_hours': peak_hours,
            'max_energy': max_energy,
            'total_cost': total_cost
        }
        all_iteration_data.append(iteration_data)
        
        print(f"Maximum energy usage: {max_energy:.2f} kWh")
        print(f"Total cost: {total_cost:.2f}")
        print(f"Peak hours: {peak_hours}")
        
        # Check if solution meets requirements
        meets_limit, _ = check_peak_limit(current_schedules)
        
        if meets_limit:
            print(f"\nFound solution with all hours below 10 kWh after {iteration} iterations!")
            best_solution = iteration_results
            break
        
        # Update best solution if better than previous
        if max_energy < best_max_energy:
            best_max_energy = max_energy
            best_solution = iteration_results
        
        # Check for convergence
        if iteration > 1 and abs(all_iteration_data[-1]['max_energy'] - 
                               all_iteration_data[-2]['max_energy']) < 0.001:
            print("\nConverged - no further improvement possible")
            break
        
        if iteration >= maximum_iterations:  # Maximum iterations
            print("\nReached maximum number of iterations")
            break
        
        iteration += 1
    
    # Display final results
    print("\n=== Final Results ===")
    print(f"Best solution found after {iteration} iterations:")
    print(f"Maximum energy usage: {best_max_energy:.2f} kWh")
    print(f"Total cost: {sum(result['total_cost'] for result in best_solution.values()):.2f}")
    
    # Plot results
    plot_results(all_iteration_data, unit_costs)

if __name__ == "__main__":
    main()