import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_simulation_results(df: pd.DataFrame, target_temp_celsius: float = 350.0):
    """
    Generates publication-quality plots of the digital twin simulation telemetry 
    from a loaded DataFrame.
    """
    # 1. Publication Styling Configuration
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'legend.fontsize': 10,
        'lines.linewidth': 1.5,
        'figure.dpi': 150
    })
    
    target_temp_k = target_temp_celsius + 273.15
    time = df['time']
    
    # 2. Create Figure and Subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    
    # --- Subplot 1: Temperature Tracking ---
    # Plot sensor data first (lightly alpha-blended so it doesn't dominate)
    ax1.scatter(time, df['measured_temperature'], color='gray', s=2, alpha=0.3, label='Noisy Pyrometer ($z_k$)')
    ax1.plot(time, df['true_temperature'], color='black', label='True Plant ($T_{true}$)')
    ax1.plot(time, df['ekf_estimate'], color='red', linestyle='--', label='EKF Estimate ($\hat{T}_{EKF}$)')
    ax1.axhline(target_temp_k, color='blue', linestyle=':', label=f'Setpoint ({target_temp_k:.2f} K)')
    
    ax1.set_title('Sapphire Wafer Thermal Response')
    ax1.set_ylabel('Temperature (K)')
    ax1.legend(loc='lower right')
    
    # --- Subplot 2: Control Effort (Laser Power) ---
    ax2.plot(time, df['laser_power'], color='darkorange', label='Commanded Power ($P_{laser}$)')
    ax2.axhline(100.0, color='red', linestyle=':', alpha=0.7, label='Hardware Max (100 W)')
    ax2.axhline(0.0, color='red', linestyle=':', alpha=0.7)
    
    ax2.set_title('Controller Actuation (PID Output)')
    ax2.set_ylabel('Laser Power (W)')
    ax2.legend(loc='upper right')
    
    # --- Subplot 3: EKF Estimation Error ---
    ekf_error = df['true_temperature'] - df['ekf_estimate']
    ax3.plot(time, ekf_error, color='purple', label='Error ($T_{true} - \hat{T}_{EKF}$)')
    ax3.axhline(0.0, color='black', linewidth=1)
    
    # Plot the 2-sigma confidence bounds from the EKF covariance
    sigma = np.sqrt(df['ekf_error_cov'])
    ax3.fill_between(time, -2*sigma, 2*sigma, color='purple', alpha=0.2, label=r'$\pm 2\sigma$ Confidence')
    
    ax3.set_title('Extended Kalman Filter Accuracy')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Error (K)')
    ax3.legend(loc='upper right')
    
    # Clean up layout and display
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define the target file to visualize
    csv_filename = r"src\telemetry_manual_tuning.csv"  # Change to 'telemetry_manual_tuning.csv' to see the stable run
    
    print(f"Attempting to load data from: {csv_filename}...")
    
    if os.path.exists(csv_filename):
        # Load the saved telemetry
        df_telemetry = pd.read_csv(csv_filename)
        print("Data loaded successfully. Generating plots...")
        
        # Plot the results
        plot_simulation_results(df_telemetry, target_temp_celsius=350.0)
    else:
        print(f"Error: The file '{csv_filename}' was not found in the current directory.")
        print("Please ensure the simulation script has been run and the CSV is saved.")