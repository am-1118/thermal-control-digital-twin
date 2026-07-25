import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

from physics_engine import LaserHeatingPlant, SapphireParameters
from estimators import ExtendedKalmanFilter
from controller import LQIController

def run_mutated_lqg_simulation(
    true_params: SapphireParameters, 
    nominal_params: SapphireParameters,
    sim_time: float = 200.0, 
    dt: float = 0.1, 
    target_temp_celsius: float = 350.0,
    measurement_noise_std: float = 2.0
) -> np.ndarray:
    
    T_eq = target_temp_celsius + 273.15
    P_eq = 53.44 
    
    # 1. Initialize True Plant (Mutated)
    plant = LaserHeatingPlant(
        params=true_params, 
        initial_temperature=true_params.t_ambient, 
        measurement_std=measurement_noise_std
    )
    
    # 2. Initialize Controllers with Nominal Linearization
    m_cp = nominal_params.mass * nominal_params.cp
    A_c = -(nominal_params.area / m_cp) * (
        nominal_params.h_conv + 4 * nominal_params.emissivity * nominal_params.sigma * T_eq**3
    )
    B_c = nominal_params.absorptivity / m_cp
    
    Q = np.array([[10.0, 0.0], [0.0, 0.5]])
    R = np.array([[1.0]])
    
    lqi = LQIController(
        A_c=A_c, B_c=B_c, dt=dt, Q=Q, R=R, 
        T_eq=T_eq, P_eq=P_eq, u_min=0.0, u_max=100.0
    )
    
    # 3. Initialize EKF (Nominal)
    def ekf_physics(T: float, P: float) -> float:
        q_in = nominal_params.absorptivity * P
        q_conv = nominal_params.h_conv * nominal_params.area * (T - nominal_params.t_ambient)
        q_rad = nominal_params.emissivity * nominal_params.sigma * nominal_params.area * (T**4 - nominal_params.t_ambient**4)
        return (q_in - q_conv - q_rad) / m_cp
        
    def ekf_jacobian(T: float) -> float:
        return -(nominal_params.area / m_cp) * (
            nominal_params.h_conv + 4 * nominal_params.emissivity * nominal_params.sigma * T**3
        )
        
    ekf = ExtendedKalmanFilter(
        physics_model=ekf_physics, jacobian_model=ekf_jacobian,
        dt=dt, Q=0.01, R=measurement_noise_std**2, P0=10.0, x0=nominal_params.t_ambient
    )
    
    # 4. Main Event Loop
    time_steps = np.arange(0, sim_time, dt)
    true_temperatures = []
    
    current_estimate = nominal_params.t_ambient
    
    for _ in time_steps:
        P_laser = lqi.compute(measured_value=current_estimate)
        T_true = plant.step(dt=dt, P_laser=P_laser)
        z_measured, _ = plant.measure_temperature()
        
        ekf.predict(u=P_laser)
        current_estimate = ekf.update(z=z_measured)
        
        true_temperatures.append(T_true)
        
    return np.array(true_temperatures)

def run_lqg_monte_carlo(N: int = 50):
    print(f"Initiating LQG Monte Carlo Sweep with {N} iterations...")
    
    nominal_params = SapphireParameters()
    sim_time = 200.0
    dt = 0.1
    time_steps = np.arange(0, sim_time, dt)
    
    results_matrix = np.zeros((N, len(time_steps)))
    
    for i in range(N):
        mutated_params = deepcopy(nominal_params)
        
        # Inject 10% Mass Variance, 15% Emissivity Variance, 5 K Ambient Variance
        mutated_params.mass = np.random.normal(loc=nominal_params.mass, scale=0.10 * nominal_params.mass)
        mutated_params.emissivity = np.random.normal(loc=nominal_params.emissivity, scale=0.15 * nominal_params.emissivity)
        mutated_params.t_ambient = np.random.normal(loc=nominal_params.t_ambient, scale=5.0)
        
        results_matrix[i, :] = run_mutated_lqg_simulation(
            true_params=mutated_params, 
            nominal_params=nominal_params,
            sim_time=sim_time, 
            dt=dt
        )
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{N} simulations.")
            
    # Statistical Bounds
    mean_response = np.mean(results_matrix, axis=0)
    std_response = np.std(results_matrix, axis=0)
    upper_bound = mean_response + 2 * std_response
    lower_bound = mean_response - 2 * std_response
    
    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11})
    
    for i in range(N):
        plt.plot(time_steps, results_matrix[i, :], color='gray', alpha=0.15, linewidth=0.5)
        
    plt.plot(time_steps, mean_response, color='black', linewidth=2, label=r'Mean Response ($\mu$)')
    plt.fill_between(time_steps, lower_bound, upper_bound, color='green', alpha=0.2, label=r'95% Confidence ($\pm 2\sigma$)')
    
    target_temp_k = 350.0 + 273.15
    plt.axhline(target_temp_k, color='red', linestyle='--', label='Target Setpoint (623.15 K)')
    
    plt.title(f'Optimal LQG Monte Carlo Robustness Sweep ($N={N}$ Runs)\n10% Mass Variance, 15% Emissivity Variance')
    plt.xlabel('Time (s)')
    plt.ylabel('True Plant Temperature (K)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_lqg_monte_carlo(N=50)