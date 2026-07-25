import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from physics_engine import LaserHeatingPlant, SapphireParameters
from estimators import ExtendedKalmanFilter
from controller import LQIController

def run_lqg_simulation():
    print("Initializing LQG Digital Twin...")
    
    # 1. Simulation Setup
    sim_time = 200.0
    dt = 0.1
    T_eq = 623.15  # 350 Celsius target
    P_eq = 53.44   # Steady-state power required
    
    # 2. Physics Parameters
    nominal_params = SapphireParameters()
    
    # We will simulate the true plant with a slight initial temperature offset and measurement noise
    plant = LaserHeatingPlant(
        params=nominal_params, 
        initial_temperature=nominal_params.t_ambient, 
        measurement_std=2.0
    )
    
    # 3. Calculate Linearized Jacobians (Ac, Bc) at Equilibrium
    m_cp = nominal_params.mass * nominal_params.cp
    A_c = -(nominal_params.area / m_cp) * (
        nominal_params.h_conv + 4 * nominal_params.emissivity * nominal_params.sigma * T_eq**3
    )
    B_c = nominal_params.absorptivity / m_cp
    
    # 4. LQR Cost Matrices Definition
    # Q penalizes [Temperature Error, Integral Error]
    Q = np.array([
        [10.0, 0.0],
        [0.0,  0.5]
    ])
    
    # R penalizes [Laser Power Effort]
    R = np.array([[1.0]])
    
    # 5. Initialize Controllers and Estimators
    lqi = LQIController(
        A_c=A_c, B_c=B_c, dt=dt, Q=Q, R=R, 
        T_eq=T_eq, P_eq=P_eq, u_min=0.0, u_max=100.0
    )
    
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
        physics_model=ekf_physics, 
        jacobian_model=ekf_jacobian, 
        dt=dt, Q=0.01, R=2.0**2, P0=10.0, x0=nominal_params.t_ambient
    )
    
    # 6. Main Discrete Event Loop
    time_steps = np.arange(0, sim_time, dt)
    history = {
        "time": [], "true_temperature": [], "measured_temperature": [], 
        "ekf_estimate": [], "ekf_error_cov": [], "laser_power": []
    }
    
    current_estimate = nominal_params.t_ambient
    
    for t in time_steps:
        # LQI Controller computes optimal power based on EKF estimate
        P_laser = lqi.compute(measured_value=current_estimate)
        
        # Plant physically steps forward
        T_true = plant.step(dt=dt, P_laser=P_laser)
        z_measured, _ = plant.measure_temperature()
        
        # EKF updates state
        ekf.predict(u=P_laser)
        current_estimate = ekf.update(z=z_measured)
        
        # Log telemetry
        history["time"].append(t)
        history["true_temperature"].append(T_true)
        history["measured_temperature"].append(z_measured)
        history["ekf_estimate"].append(current_estimate)
        history["ekf_error_cov"].append(ekf.P)
        history["laser_power"].append(P_laser)
        
    df = pd.DataFrame(history)
    print(f"LQI Optimal Gains -> K_temp: {lqi.K_temp:.4f}, K_integral: {lqi.K_int:.4f}")
    
    # 7. Visualization
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11})
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    
    # Thermal Response
    ax1.plot(df['time'], df['true_temperature'], color='black', label='True Plant')
    ax1.plot(df['time'], df['ekf_estimate'], color='red', linestyle='--', label='EKF Estimate')
    ax1.axhline(T_eq, color='blue', linestyle=':', label=f'Setpoint ({T_eq} K)')
    ax1.set_title('Optimal LQG Thermal Response')
    ax1.set_ylabel('Temperature (K)')
    ax1.legend(loc='lower right')
    
    # Actuation
    ax2.plot(df['time'], df['laser_power'], color='darkorange', label='Commanded Power')
    ax2.axhline(100.0, color='red', linestyle=':', alpha=0.7, label='Hardware Max')
    ax2.axhline(P_eq, color='gray', linestyle='--', alpha=0.7, label=f'Equilibrium ({P_eq} W)')
    ax2.set_title('LQI Controller Actuation')
    ax2.set_ylabel('Power (W)')
    ax2.legend(loc='upper right')
    
    # EKF Accuracy
    error = df['true_temperature'] - df['ekf_estimate']
    sigma = np.sqrt(df['ekf_error_cov'])
    ax3.plot(df['time'], error, color='purple', label='Estimation Error')
    ax3.fill_between(df['time'], -2*sigma, 2*sigma, color='purple', alpha=0.2, label=r'$\pm 2\sigma$ Bounds')
    ax3.axhline(0.0, color='black', linewidth=1)
    ax3.set_title('Extended Kalman Filter Accuracy')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Error (K)')
    ax3.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_lqg_simulation()