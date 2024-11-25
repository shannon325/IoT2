import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

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

# 2. Solve optimization with quadratic cost
def solve_quadratic_optimization(tasks_df):
    # Create decision variables matrix
    num_tasks = len(tasks_df)
    vars_matrix = cp.Variable((num_tasks, 24))  # Each task has 24 hours of possible scheduling
    
    # Objective: minimize sum of 0.5 * (sum of energy per hour)^2
    hourly_sums = cp.sum(vars_matrix, axis=0)  # Sum energy usage per hour
    objective = cp.Minimize(cp.sum(0.5 * cp.power(hourly_sums, 2)))
    
    # Constraints
    constraints = []
    
    # 1. Non-negativity
    constraints.append(vars_matrix >= 0)
    
    # 2. Maximum energy per hour constraint
    for i, task in tasks_df.iterrows():
        constraints.append(vars_matrix[i, :] <= task['max_energy'])
    
    # 3. Zero energy outside time window
    for i, task in tasks_df.iterrows():
        # Before ready time
        if task['ready_time'] > 0:
            constraints.append(vars_matrix[i, :task['ready_time']] == 0)
        # After deadline
        if task['deadline'] < 23:
            constraints.append(vars_matrix[i, (task['deadline']+1):] == 0)
    
    # 4. Total energy demand constraint
    for i, task in tasks_df.iterrows():
        constraints.append(cp.sum(vars_matrix[i, :]) == task['demand'])
    
    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    # Extract results
    if prob.status == 'optimal':
        schedule_matrix = vars_matrix.value
        
        # Process results
        total_schedule = np.sum(schedule_matrix, axis=0)
        user_schedules = {}
        for user in tasks_df['User'].unique():
            user_mask = tasks_df['User'] == user
            user_schedules[user] = np.sum(schedule_matrix[user_mask], axis=0)
        
        # Calculate costs
        quadratic_costs = 0.5 * np.power(total_schedule, 2)
        
        return {
            'schedule': total_schedule,
            'user_schedules': user_schedules,
            'quadratic_costs': quadratic_costs,
            'total_cost': np.sum(quadratic_costs),
            'status': 'optimal'
        }
    else:
        return {'status': 'infeasible'}

# 3. Solve linear optimization (for comparison)
def solve_linear_optimization(tasks_df, unit_costs):
    # Create decision variables matrix
    num_tasks = len(tasks_df)
    vars_matrix = cp.Variable((num_tasks, 24))
    
    # Objective: minimize sum of linear costs
    hourly_sums = cp.sum(vars_matrix, axis=0)
    objective = cp.Minimize(cp.sum(cp.multiply(hourly_sums, unit_costs)))
    
    # Constraints (same as quadratic optimization)
    constraints = []
    constraints.append(vars_matrix >= 0)
    
    for i, task in tasks_df.iterrows():
        constraints.append(vars_matrix[i, :] <= task['max_energy'])
        
        if task['ready_time'] > 0:
            constraints.append(vars_matrix[i, :task['ready_time']] == 0)
        if task['deadline'] < 23:
            constraints.append(vars_matrix[i, (task['deadline']+1):] == 0)
            
        constraints.append(cp.sum(vars_matrix[i, :]) == task['demand'])
    
    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    # Extract results
    if prob.status == 'optimal':
        schedule_matrix = vars_matrix.value
        total_schedule = np.sum(schedule_matrix, axis=0)
        
        user_schedules = {}
        for user in tasks_df['User'].unique():
            user_mask = tasks_df['User'] == user
            user_schedules[user] = np.sum(schedule_matrix[user_mask], axis=0)
        
        linear_costs = total_schedule * unit_costs
        
        return {
            'schedule': total_schedule,
            'user_schedules': user_schedules,
            'linear_costs': linear_costs,
            'total_cost': np.sum(linear_costs),
            'status': 'optimal'
        }
    else:
        return {'status': 'infeasible'}

# 4. Visualization
def plot_comparison(linear_results, quadratic_results, unit_costs):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot 1: Linear pricing schedule
    bottom = np.zeros(24)
    for user, schedule in linear_results['user_schedules'].items():
        ax1.bar(range(24), schedule, bottom=bottom, label=f'User {user}', alpha=0.6)
        bottom += schedule
    ax1.set_title('Energy Schedule with Linear Pricing')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Energy (kWh)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Quadratic pricing schedule
    bottom = np.zeros(24)
    for user, schedule in quadratic_results['user_schedules'].items():
        ax2.bar(range(24), schedule, bottom=bottom, label=f'User {user}', alpha=0.6)
        bottom += schedule
    ax2.set_title('Energy Schedule with Quadratic Pricing (0.5E²)')
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Energy (kWh)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cost comparison
    ax3.plot(range(24), linear_results['linear_costs'], 'b-', 
             label='Linear Cost', marker='o')
    ax3.plot(range(24), quadratic_results['quadratic_costs'], 'r-', 
             label='Quadratic Cost (0.5E²)', marker='o')
    ax3.set_title('Cost Comparison')
    ax3.set_xlabel('Hour')
    ax3.set_ylabel('Cost')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Task4.png')  
    plt.show()

# Main execution
def main():
    # Load data
    tasks_df, unit_costs = load_and_preprocess_data("IMSE7143CW2Input.xlsx")
    
    # Solve both optimization problems
    linear_results = solve_linear_optimization(tasks_df, unit_costs)
    quadratic_results = solve_quadratic_optimization(tasks_df)
    
    if linear_results['status'] == 'optimal' and quadratic_results['status'] == 'optimal':
        print("\n=== Results ===")
        print(f"Total cost (Linear pricing): {linear_results['total_cost']:.2f}")
        print(f"Total cost (Quadratic pricing): {quadratic_results['total_cost']:.2f}")
        
        # Plot comparison
        plot_comparison(linear_results, quadratic_results, unit_costs)
    else:
        print("One or both optimization problems could not be solved optimally.")

if __name__ == "__main__":
    main()