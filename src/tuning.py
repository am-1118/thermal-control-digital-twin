import numpy as np
from physics_engine import LaserHeatingPlant, SapphireParameters

def ziegler_nichols_tuning(step_power: float = 100.0, simulation_time: float = 60.0, dt: float = 0.1):
    """
    Performs an open-loop step test on the physical plant and calculates 
    optimal PID parameters using the Ziegler-Nichols Reaction Curve method.
    """
    # 1. Initialize the pristine physical plant
    params = SapphireParameters()
    plant = LaserHeatingPlant(params=params, initial_temperature=params.t_ambient, measurement_std=0.0)
    
    times = np.arange(0, simulation_time, dt)
    temperatures = np.zeros_like(times)
    
    # 2. Run the open-loop step response simulation
    for i, t in enumerate(times):
        # We sample the true, noise-free temperature for tuning
        temperatures[i] = plant.true_temperature
        plant.step(dt=dt, P_laser=step_power)
        
    # 3. Calculate the numerical derivative (dT/dt)
    dT_dt = np.gradient(temperatures, dt)
    
    # 4. Find the inflection point (maximum slope)
    max_slope_idx = np.argmax(dT_dt)
    R = dT_dt[max_slope_idx]
    t_inflect = times[max_slope_idx]
    T_inflect = temperatures[max_slope_idx]
    
    # 5. Calculate Dead Time (L)
    T_initial = temperatures[0]
    L = t_inflect - ((T_inflect - T_initial) / R)
    
    # Safety check: if L is extremely small or negative, the system 
    # responds too instantly for Z-N to work perfectly without adjustment.
    L = max(L, dt)  # Lower bound L to the sample time
    
    # 6. Calculate Z-N PID Parameters
    Kp = (1.2 * step_power) / (R * L)
    Ti = 2.0 * L
    Td = 0.5 * L
    
    Ki = Kp / Ti
    Kd = Kp * Td
    
    print("--- Ziegler-Nichols Tuning Results ---")
    print(f"Max Slope (R): {R:.4f} K/s")
    print(f"Dead Time (L): {L:.4f} s")
    print(f"Kp: {Kp:.4f}")
    print(f"Ki: {Ki:.4f}")
    print(f"Kd: {Kd:.4f}")
    
    return Kp, Ki, Kd

if __name__ == "__main__":
    ziegler_nichols_tuning()