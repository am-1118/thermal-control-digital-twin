import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Ensure the app can find the backend modules in the src/ directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from physics_engine import LaserHeatingPlant, SapphireParameters
from estimators import ExtendedKalmanFilter, RecursiveLeastSquares
from controller import LQIController

st.set_page_config(page_title="LQG Digital Twin Dashboard", layout="wide")
st.title("Sapphire Wafer Thermal Control: Optimal LQG Twin")
st.markdown("Adjust the LQR cost matrices to balance tracking aggressiveness versus energy conservation. The RLS estimator tracks physical mutations in real-time.")

# --- Sidebar: The LQR Cost Panel ---
st.sidebar.header("LQR Cost Matrices (Q & R)")
st.sidebar.markdown("Define the mathematical penalties for the Riccati Equation.")

q_temp_ui = st.sidebar.slider("Temperature Error Penalty (Q11)", 0.0, 50.0, 5.0, 0.5)
q_int_ui = st.sidebar.slider("Integral Error Penalty (Q22)", 0.0, 5.0, 0.5, 0.1)
r_power_ui = st.sidebar.slider("Control Effort Penalty (R)", 0.1, 10.0, 1.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.header("Physical Plant Uncertainties")
mass_variance = st.sidebar.slider("Mass Variance (%)", -20, 20, 10, 1)
absorptivity_variance = st.sidebar.slider("Absorptivity Variance (%)", -20, 20, 15, 1)
measurement_noise = st.sidebar.slider("Sensor Noise (Std Dev K)", 0.0, 5.0, 2.0, 0.1)

# --- Backend Integration ---
def run_lqg_interactive_simulation():
    sim_time = 200.0
    dt = 0.1
    T_eq = 350.0 + 273.15
    P_eq = 53.44
    
    nominal_params = SapphireParameters()
    true_params = SapphireParameters()
    true_params.mass = nominal_params.mass * (1.0 + (mass_variance / 100.0))
    true_params.absorptivity = nominal_params.absorptivity * (1.0 + (absorptivity_variance / 100.0))
    
    plant = LaserHeatingPlant(params=true_params, initial_temperature=true_params.t_ambient, measurement_std=measurement_noise)
    
    # 1. Calculate Continuous Jacobians for the LQI Controller
    m_cp = nominal_params.mass * nominal_params.cp
    A_c = -(nominal_params.area / m_cp) * (
        nominal_params.h_conv + 4 * nominal_params.emissivity * nominal_params.sigma * T_eq**3
    )
    B_c = nominal_params.absorptivity / m_cp
    
    Q_matrix = np.array([[q_temp_ui, 0.0], [0.0, q_int_ui]])
    R_matrix = np.array([[r_power_ui]])
    
    # Initialize Optimal Controller
    lqi = LQIController(
        A_c=A_c, B_c=B_c, dt=dt, Q=Q_matrix, R=R_matrix, 
        T_eq=T_eq, P_eq=P_eq, u_min=0.0, u_max=100.0
    )
    
    # Initialize EKF
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
        dt=dt, Q=0.02, R=measurement_noise**2, P0=10.0, x0=nominal_params.t_ambient
    )
    
    # Initialize RLS Estimator
    rls = RecursiveLeastSquares(num_params=2, lambda_factor=0.99, P0_diagonal=1000.0)
    
    time_steps = np.arange(0, sim_time, dt)
    history = {
        "time": [], "true_temperature": [], "measured_temperature": [], "ekf_estimate": [], 
        "ekf_error_cov": [], "laser_power": [], "estimated_mass": [], "estimated_alpha": []
    }
    
    # Initialize initial conditions before the loop
    current_estimate = nominal_params.t_ambient
    prev_estimate = nominal_params.t_ambient
    P_laser = P_eq  # Start actuator at equilibrium assumption
    
    my_bar = st.progress(0, text="Calculating Discrete Algebraic Riccati Equation & Simulating...")
    
    for i, t in enumerate(time_steps):
        # ---------------------------------------------------------
        # STEP 1: The Physical Plant (The Real World)
        # ---------------------------------------------------------
        # Plant physically steps forward using the laser power commanded in the PREVIOUS step
        T_true = plant.step(dt=dt, P_laser=P_laser)
        
        # Pyrometer takes a raw, noisy measurement
        z_measured, _ = plant.measure_temperature()
        
        # ---------------------------------------------------------
        # STEP 2: Extended Kalman Filter (State Estimation)
        # ---------------------------------------------------------
        ekf.predict(u=P_laser)
        
        # EKF absorbs the noisy pyrometer data and outputs the optimal state
        current_estimate = ekf.update(z=z_measured)
        
        # ---------------------------------------------------------
        # STEP 3: Recursive Least Squares (Passive Parameter ID)
        # ---------------------------------------------------------
        # RLS strictly uses the smooth EKF estimate to calculate the derivative
        dT_dt = (current_estimate - prev_estimate) / dt
        
        q_conv = nominal_params.h_conv * nominal_params.area * (current_estimate - nominal_params.t_ambient)
        q_rad = nominal_params.emissivity * nominal_params.sigma * nominal_params.area * (current_estimate**4 - nominal_params.t_ambient**4)
        q_loss = q_conv + q_rad
        
        y_k = q_loss
        phi_k = np.array([[-nominal_params.cp * dT_dt], [P_laser]])
        
        # Persistent Excitation threshold = 1.5
        theta_hat = rls.update(phi=phi_k, y=y_k, freeze_update=(abs(dT_dt) <= 1.5))
        
        # ---------------------------------------------------------
        # STEP 4: LQI Controller (Optimal Action)
        # ---------------------------------------------------------
        # Controller exclusively uses the EKF estimate, completely ignoring z_measured
        P_laser = lqi.compute(measured_value=current_estimate)
        
        # ---------------------------------------------------------
        # Logging & Housekeeping
        # ---------------------------------------------------------
        history["time"].append(t)
        history["true_temperature"].append(T_true)
        history["measured_temperature"].append(z_measured)
        history["ekf_estimate"].append(current_estimate)
        history["ekf_error_cov"].append(ekf.P)
        history["laser_power"].append(P_laser)
        history["estimated_mass"].append(theta_hat[0, 0])
        history["estimated_alpha"].append(theta_hat[1, 0])
        
        prev_estimate = current_estimate
        
        if i % (len(time_steps) // 10) == 0:
            my_bar.progress(i / len(time_steps), text="Simulating Digital Twin...")
            
    my_bar.empty()
    return pd.DataFrame(history), true_params

# --- Visualization ---
if st.sidebar.button("Run Simulation", type="primary"):
    df, true_params = run_lqg_interactive_simulation()
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
    
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    
    time = df['time']
    target_temp_k = 350.0 + 273.15
    
    # Subplot 1: Temperature Tracking
    ax1.plot(time, df['true_temperature'], color='black', label='True Plant')
    ax1.plot(time, df['ekf_estimate'], color='red', linestyle='--', label='EKF Estimate')
    ax1.axhline(target_temp_k, color='blue', linestyle=':', label='Setpoint')
    ax1.set_title('Optimal LQG Thermal Response')
    ax1.set_ylabel('Temp (K)')
    ax1.legend(loc='lower right')
    
    # Subplot 2: Control Actuation
    ax2.plot(time, df['laser_power'], color='darkorange', label='Commanded Power')
    ax2.set_title('LQI Controller Actuation')
    ax2.set_ylabel('Power (W)')
    ax2.legend(loc='upper right')
    
    # Subplot 3: EKF Estimation Error
    error = df['true_temperature'] - df['ekf_estimate']
    sigma = np.sqrt(df['ekf_error_cov'])
    ax3.plot(time, error, color='purple', label='Error')
    ax3.fill_between(time, -2*sigma, 2*sigma, color='purple', alpha=0.2, label=r'$\pm 2\sigma$ Bounds')
    ax3.axhline(0.0, color='black', linewidth=1)
    ax3.set_title('Extended Kalman Filter Accuracy')
    ax3.set_ylabel('Error (K)')
    ax3.legend(loc='upper right')
    
    # Subplot 4: Mass Tracking
    ax4.plot(time, df['estimated_mass'], color='green', label='RLS Estimated Mass')
    ax4.axhline(true_params.mass, color='black', linestyle=':', label='True Mutated Mass')
    ax4.set_title('Online Parameter Identification: Mass ($m$)')
    ax4.set_ylabel('Mass (kg)')
    ax4.legend(loc='lower right')
    
    # Subplot 5: Absorptivity Tracking
    ax5.plot(time, df['estimated_alpha'], color='indigo', label=r'RLS Estimated $\alpha$')
    ax5.axhline(true_params.absorptivity, color='black', linestyle=':', label=r'True Mutated $\alpha$')
    ax5.set_title(r'Online Parameter Identification: Absorptivity ($\alpha$)')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Absorptivity')
    ax5.legend(loc='lower right')
    
    plt.tight_layout()
    st.pyplot(fig)