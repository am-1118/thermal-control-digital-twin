import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

from physics_engine import LaserHeatingPlant, SapphireParameters
from estimators import ExtendedKalmanFilter
from controller import PIDController

def run_mutated_simulation(
    true_params: SapphireParameters, 
    nominal_params: SapphireParameters,
    sim_time: float = 200.0, 
    dt: float = 0.1, 
    target_temp_celsius: float = 350.0,
    measurement_noise_std: float = 2.0
) -> pd.DataFrame:
    """
    Runs a single simulation where the physical plant uses 'true_params' 
    (with injected variance), but the EKF and Controller assume 'nominal_params'.
    """
    T_ambient_true = true_params.t_ambient
    T_ambient_nominal = nominal_params.t_ambient
    T_setpoint = target_temp_celsius + 273.15
    
    # 1. Initialize True Plant with MUTATED parameters
    plant = LaserHeatingPlant(
        params=true_params, 
        initial_temperature=T_ambient_true, 
        measurement_std=measurement_noise_std
    )
    
    # 2. Initialize Controller with detuned, stable nominal gains
    controller = PIDController(Kp=2.0, Ki=0.1, Kd=0.0, dt=dt, u_min=0.0, u_max=100.0)
    
    # 3. Initialize EKF with NOMINAL parameters (Ignorant of the mutation)
    nominal_thermal_mass = nominal_params.mass * nominal_params.cp
    
    def ekf_physics(T: float, P: float) -> float:
        q_in = nominal_params.absorptivity * P
        q_conv = nominal_params.h_conv * nominal_params.area * (T - nominal_params.t_ambient)
        q_rad = nominal_params.emissivity * nominal_params.sigma * nominal_params.area * (T**4 - nominal_params.t_ambient**4)
        return (q_in - q_conv - q_rad) / nominal_thermal_mass
        
    def ekf_jacobian(T: float) -> float:
        return -(nominal_params.area / nominal_thermal_mass) * (
            nominal_params.h_conv + 4 * nominal_params.emissivity * nominal_params.sigma * T**3
        )
        
    ekf = ExtendedKalmanFilter(
        physics_model=ekf_physics,
        jacobian_model=ekf_jacobian,
        dt=dt, Q=0.01, R=measurement_noise_std**2, P0=10.0, x0=T_ambient_nominal
    )
    
    # 4. Main Event Loop
    time_steps = np.arange(0, sim_time, dt)
    true_temperatures = []
    
    current_estimate = T_ambient_nominal
    
    for _ in time_steps:
        P_laser = controller.compute(setpoint=T_setpoint, measured_value=current_estimate)
        T_true = plant.step(dt=dt, P_laser=P_laser)
        z_measured, _ = plant.measure_temperature()
        
        ekf.predict(u=P_laser)
        current_estimate = ekf.update(z=z_measured)
        
        true_temperatures.append(T_true)
        
    return np.array(true_temperatures)

def run_monte_carlo_sweep(N: int = 50):
    """
    Executes a Monte Carlo parameter sweep and visualizes the statistical robustness.
    """
    print(f"Initiating Monte Carlo Sweep with {N} iterations...")
    
    nominal_params = SapphireParameters()
    sim_time = 200.0
    dt = 0.1
    time_steps = np.arange(0, sim_time, dt)
    
    # Matrix to store the true temperature traces of all N runs
    results_matrix = np.zeros((N, len(time_steps)))
    
    for i in range(N):
        # Create a mutated copy of the physical parameters
        mutated_params = deepcopy(nominal_params)
        
        # Inject Gaussian noise (variance) into physical properties
        # e.g., Mass varies by 10%, Emissivity varies by 15%, Ambient Temp by 5 K
        mutated_params.mass = np.random.normal(loc=nominal_params.mass, scale=0.10 * nominal_params.mass)
        mutated_params.emissivity = np.random.normal(loc=nominal_params.emissivity, scale=0.15 * nominal_params.emissivity)
        mutated_params.t_ambient = np.random.normal(loc=nominal_params.t_ambient, scale=5.0)
        
        # Run simulation and store the trace
        results_matrix[i, :] = run_mutated_simulation(
            true_params=mutated_params, 
            nominal_params=nominal_params,
            sim_time=sim_time, 
            dt=dt
        )
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{N} simulations.")
            
    # Calculate Statistical Bounds
    mean_response = np.mean(results_matrix, axis=0)
    std_response = np.std(results_matrix, axis=0)
    upper_bound = mean_response + 2 * std_response
    lower_bound = mean_response - 2 * std_response
    
    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11})
    
    # Plot all individual runs faintly
    for i in range(N):
        plt.plot(time_steps, results_matrix[i, :], color='gray', alpha=0.15, linewidth=0.5)
        
    # Plot statistical aggregates
    plt.plot(time_steps, mean_response, color='black', linewidth=2, label=r'Mean Response ($\mu$)')
    plt.fill_between(time_steps, lower_bound, upper_bound, color='blue', alpha=0.2, label=r'95% Confidence ($\pm 2\sigma$)')
    
    target_temp_k = 350.0 + 273.15
    plt.axhline(target_temp_k, color='red', linestyle='--', label='Target Setpoint (623.15 K)')
    
    plt.title(f'Monte Carlo Robustness Sweep ($N={N}$ Runs)\n10% Mass Variance, 15% Emissivity Variance')
    plt.xlabel('Time (s)')
    plt.ylabel('True Plant Temperature (K)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_monte_carlo_sweep(N=50)