import numpy as np
import pandas as pd
from typing import Dict, Any

from physics_engine import LaserHeatingPlant, SapphireParameters
from estimators import LinearKalmanFilter, ExtendedKalmanFilter
from controller import PIDController
from tuning import ziegler_nichols_tuning  # <-- Import the tuning function

def run_single_simulation(
    sim_time: float = 300.0, 
    dt: float = 0.1, 
    target_temp_celsius: float = 350.0,
    measurement_noise_std: float = 2.0
) -> pd.DataFrame:
    
    # --- 1. System Initialization ---
    params = SapphireParameters()
    T_ambient = params.t_ambient
    T_setpoint = target_temp_celsius + 273.15  # Convert to Kelvin
    
    # Initialize True Plant
    plant = LaserHeatingPlant(
        params=params, 
        initial_temperature=T_ambient, 
        measurement_std=measurement_noise_std
    )
    
    # --- 1.5 Calibration Phase (Manual Override) ---
    print("Bypassing Z-N Auto-Tuning. Applying Manual Detuned Gains...")
    
    # Manually detuned gains for smooth thermal control
    Kp_manual = 2.0
    Ki_manual = 0.1
    Kd_manual = 0.0
    
    # Initialize Controller using manual gains
    controller = PIDController(
        Kp=Kp_manual, 
        Ki=Ki_manual, 
        Kd=Kd_manual, 
        dt=dt, 
        u_min=0.0, 
        u_max=100.0
    )

    # --- 2. Estimator Setup ---
    # Common Noise Covariances
    Q = 0.01  # Process noise (model uncertainty)
    R = measurement_noise_std**2  # Measurement noise variance
    P0 = 10.0 # Initial uncertainty
    
    # A. Linear Kalman Filter (LKF) Setup
    # We linearize around the target setpoint (T_ss)
    T_ss = T_setpoint
    thermal_mass = params.mass * params.cp
    
    Ac = -(params.area / thermal_mass) * (params.h_conv + 4 * params.emissivity * params.sigma * T_ss**3)
    Bc = params.absorptivity / thermal_mass
    
    # Calculate steady-state power required to hold T_ss (for LKF deviations)
    q_conv_ss = params.h_conv * params.area * (T_ss - T_ambient)
    q_rad_ss = params.emissivity * params.sigma * params.area * (T_ss**4 - T_ambient**4)
    P_ss = (q_conv_ss + q_rad_ss) / params.absorptivity
    
    lkf = LinearKalmanFilter(
        F=1.0 + Ac * dt, 
        G=Bc * dt, 
        H=1.0, 
        Q=Q, R=R, P0=P0, 
        x0_dev=T_ambient - T_ss  # Initial deviation from setpoint
    )
    
    # B. Extended Kalman Filter (EKF) Setup
    def ekf_physics(T: float, P: float) -> float:
        q_in = params.absorptivity * P
        q_conv = params.h_conv * params.area * (T - params.t_ambient)
        q_rad = params.emissivity * params.sigma * params.area * (T**4 - params.t_ambient**4)
        return (q_in - q_conv - q_rad) / thermal_mass
        
    def ekf_jacobian(T: float) -> float:
        return -(params.area / thermal_mass) * (params.h_conv + 4 * params.emissivity * params.sigma * T**3)
        
    ekf = ExtendedKalmanFilter(
        physics_model=ekf_physics,
        jacobian_model=ekf_jacobian,
        dt=dt, Q=Q, R=R, P0=P0, x0=T_ambient
    )
    
    # --- 3. Telemetry Storage ---
    time_steps = np.arange(0, sim_time, dt)
    history = {
        "time": [],
        "true_temperature": [],
        "measured_temperature": [],
        "lkf_estimate": [],
        "ekf_estimate": [],
        "laser_power": [],
        "lkf_error_cov": [],
        "ekf_error_cov": []
    }
    
    # We must prime the controller with an initial estimate.
    # We will use the EKF's output to drive the PID.
    current_estimate = T_ambient
    
    # --- 4. The Main Discrete Event Loop ---
    for t in time_steps:
        # Step A: Controller calculates input based on the latest state estimate
        # (Using EKF estimate to close the loop)
        P_laser = controller.compute(setpoint=T_setpoint, measured_value=current_estimate)
        
        # Step B: Physical plant advances one time step (True continuous physics)
        T_true = plant.step(dt=dt, P_laser=P_laser)
        
        # Step C: Sensor samples the true state with noise
        z_measured, _ = plant.measure_temperature()
        
        # Step D: Observer State Updates (Predict & Update)
        # 1. Update LKF (Operates on deviations from T_ss and P_ss)
        lkf.predict(u_dev=P_laser - P_ss)
        T_lkf_dev = lkf.update(z_dev=z_measured - T_ss)
        T_lkf_absolute = T_lkf_dev + T_ss
        
        # 2. Update EKF (Operates on absolute values and non-linear physics)
        ekf.predict(u=P_laser)
        T_ekf_absolute = ekf.update(z=z_measured)
        
        # Feedback for the next control cycle
        current_estimate = T_ekf_absolute
        
        # Step E: Log Telemetry
        history["time"].append(t)
        history["true_temperature"].append(T_true)
        history["measured_temperature"].append(z_measured)
        history["lkf_estimate"].append(T_lkf_absolute)
        history["ekf_estimate"].append(T_ekf_absolute)
        history["laser_power"].append(P_laser)
        history["lkf_error_cov"].append(lkf.P)
        history["ekf_error_cov"].append(ekf.P)
        
    return pd.DataFrame(history)

if __name__ == "__main__":
    print("Running single simulation loop...")
    
    # Run the simulation
    df_manual = run_single_simulation(sim_time=200.0, dt=0.1)
    
    # Save the telemetry to a CSV file in your repository
    df_manual.to_csv("telemetry_manual_tuning.csv", index=False)
    print("Simulation Complete. Data saved to 'telemetry_ZN_tuning.csv'.")
    
    # Display the final steady-state results
    print("\nFinal State (t = 200s):")
    print(df_manual.tail(1)[['true_temperature', 'ekf_estimate', 'laser_power']].to_string(index=False))