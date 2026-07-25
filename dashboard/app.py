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
from controller import PIDController

st.set_page_config(page_title="Digital Twin Dashboard", layout="wide")
st.title("Sapphire Wafer Thermal Control: Digital Twin")
st.markdown("Adjust the PID parameters and inject physical variances. The RLS estimator will attempt to identify the hidden variances in real-time.")

# --- Sidebar ---
st.sidebar.header("PID Controller Tuning")
Kp_ui = st.sidebar.slider("Proportional Gain (Kp)", 0.0, 50.0, 2.0, 0.5)
Ki_ui = st.sidebar.slider("Integral Gain (Ki)", 0.0, 250.0, 0.1, 0.1)
Kd_ui = st.sidebar.slider("Derivative Gain (Kd)", 0.0, 10.0, 0.1, 0.1)

st.sidebar.header("Physical Plant Uncertainties")
mass_variance = st.sidebar.slider("Mass Variance (%)", -20, 20, 10, 1)
absorptivity_variance = st.sidebar.slider("Absorptivity Variance (%)", -20, 20, 15, 1)
measurement_noise = st.sidebar.slider("Sensor Noise (Std Dev K)", 0.0, 5.0, 2.0, 0.1)

# --- Backend Integration ---
def run_interactive_simulation():
    sim_time = 200.0
    dt = 0.1
    target_temp_k = 350.0 + 273.15
    
    nominal_params = SapphireParameters()
    true_params = SapphireParameters()
    true_params.mass = nominal_params.mass * (1.0 + (mass_variance / 100.0))
    true_params.absorptivity = nominal_params.absorptivity * (1.0 + (absorptivity_variance / 100.0))
    
    plant = LaserHeatingPlant(params=true_params, initial_temperature=true_params.t_ambient, measurement_std=measurement_noise)
    
    # Corrected: Passing the UI slider variable for Derivative Gain
    controller = PIDController(Kp=Kp_ui, Ki=Ki_ui, Kd=Kd_ui, dt=dt, u_min=0.0, u_max=100.0)
    
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
        dt=dt, Q=0.01, R=measurement_noise**2, P0=10.0, x0=nominal_params.t_ambient
    )
    
    # Corrected: Using the exact initialization structure from estimators.py
    rls = RecursiveLeastSquares(num_params=2, lambda_factor=0.99, P0_diagonal=1000.0)
    
    time_steps = np.arange(0, sim_time, dt)
    history = {"time": [], "true_temperature": [], "ekf_estimate": [], "laser_power": [], "estimated_mass": [], "estimated_alpha": []}
    
    current_estimate = nominal_params.t_ambient
    prev_estimate = nominal_params.t_ambient
    
    my_bar = st.progress(0, text="Simulating Digital Twin...")
    
    for i, t in enumerate(time_steps):
        # 1. Control & Plant Step
        P_laser = controller.compute(setpoint=target_temp_k, measured_value=current_estimate)
        T_true = plant.step(dt=dt, P_laser=P_laser)
        z_measured, _ = plant.measure_temperature()
        
        # 2. EKF Step
        ekf.predict(u=P_laser)
        current_estimate = ekf.update(z=z_measured)
        
        # 3. RLS Parameter Tracking
        dT_dt = (current_estimate - prev_estimate) / dt
        
        q_conv = nominal_params.h_conv * nominal_params.area * (current_estimate - nominal_params.t_ambient)
        q_rad = nominal_params.emissivity * nominal_params.sigma * nominal_params.area * (current_estimate**4 - nominal_params.t_ambient**4)
        q_loss = q_conv + q_rad
        
        y_k = q_loss
        phi_k = np.array([[-nominal_params.cp * dT_dt], [P_laser]])
        
        # Route the excitation check directly to the estimator backend
        theta_hat = rls.update(phi=phi_k, y=y_k, freeze_update=(abs(dT_dt) <= 2.0))
        
        # Log Data
        history["time"].append(t)
        history["true_temperature"].append(T_true)
        history["ekf_estimate"].append(current_estimate)
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
    df, true_params = run_interactive_simulation()
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    time = df['time']
    target_temp_k = 350.0 + 273.15
    
    # Subplot 1: Temperature
    ax1.plot(time, df['true_temperature'], color='black', label='True Plant')
    ax1.plot(time, df['ekf_estimate'], color='red', linestyle='--', label='EKF Estimate')
    ax1.axhline(target_temp_k, color='blue', linestyle=':', label='Setpoint')
    ax1.set_title('Thermal Response')
    ax1.legend(loc='lower right')
    
    # Subplot 2: Power
    ax2.plot(time, df['laser_power'], color='darkorange', label='Commanded Power')
    ax2.set_title('Controller Actuation')
    ax2.legend(loc='upper right')
    
    # Subplot 3: Mass Tracking
    ax3.plot(time, df['estimated_mass'], color='green', label='RLS Estimated Mass')
    ax3.axhline(true_params.mass, color='black', linestyle=':', label='True Mutated Mass')
    ax3.set_title('Online Parameter Identification: Mass ($m$)')
    ax3.legend(loc='upper right')
    
    # Subplot 4: Absorptivity Tracking
    ax4.plot(time, df['estimated_alpha'], color='purple', label=r'RLS Estimated Absorptivity ($\alpha$)')
    ax4.axhline(true_params.absorptivity, color='black', linestyle=':', label=r'True Mutated $\alpha$')
    ax4.set_title(r'Online Parameter Identification: Absorptivity ($\alpha$)')
    ax4.set_xlabel('Time (s)')
    ax4.legend(loc='upper right')
    
    plt.tight_layout()
    st.pyplot(fig)