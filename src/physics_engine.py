import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Tuple

@dataclass
class SapphireParameters:
    """
    Thermophysical parameters for a 2-inch single-crystal sapphire wafer.
    Used for lumped-capacitance thermal modeling.
    """
    mass: float = 0.004             # Mass (kg)
    cp: float = 750.0               # Specific heat capacity (J/(kg K))
    area: float = 0.00405           # Total exposed surface area (m^2)
    absorptivity: float = 0.8       # Absorptivity at 10.6 um (CO2 laser)
    emissivity: float = 0.7         # Broadband thermal emissivity
    h_conv: float = 15.0            # Natural convection coefficient (W/(m^2 K))
    sigma: float = 5.67e-8          # Stefan-Boltzmann constant (W/(m^2 K^4))
    t_ambient: float = 298.15       # Ambient room temperature (K)


class LaserHeatingPlant:
    """
    Simulates the continuous 'True Plant' physics of the laser heating process 
    and generates synthetic, noisy sensor data for the digital twin observer.
    """
    
    def __init__(self, params: SapphireParameters, initial_temperature: float = 298.15, measurement_std: float = 2.0):
        """
        Initializes the physical plant simulator.

        Args:
            params (SapphireParameters): The thermophysical properties of the target.
            initial_temperature (float): Starting true temperature in Kelvin.
            measurement_std (float): Standard deviation of the synthetic pyrometer noise (K).
        """
        self.params = params
        self.true_temperature = initial_temperature
        self.measurement_std = measurement_std
        
        # Pre-calculate thermal inertia for optimization
        self.thermal_inertia = self.params.mass * self.params.cp

    def _heat_equation(self, t: float, T: float, P_laser: float) -> float:
        """
        The governing non-linear ordinary differential equation (ODE).
        
        Args:
            t (float): Current time (required by SciPy solver API).
            T (float): Current true temperature (K).
            P_laser (float): Incident laser power (W).
            
        Returns:
            float: The rate of change of temperature, dT/dt.
        """
        q_in = self.params.absorptivity * P_laser
        
        q_conv = self.params.h_conv * self.params.area * (T - self.params.t_ambient)
        q_rad = self.params.emissivity * self.params.sigma * self.params.area * (T**4 - self.params.t_ambient**4)
        
        dT_dt = (q_in - q_conv - q_rad) / self.thermal_inertia
        return dT_dt

    def step(self, dt: float, P_laser: float) -> float:
        """
        Advances the physical simulation forward by one discrete time step.
        Uses a Zero-Order Hold (ZOH) assumption for the laser power input.
        
        Args:
            dt (float): The time step duration (seconds).
            P_laser (float): The commanded laser power during this step (W).
            
        Returns:
            float: The new true temperature at the end of the time step (K).
        """
        # Ensure laser power is non-negative
        P_laser = max(0.0, P_laser)
        
        # Integrate the non-linear ODE over [0, dt] using RK45
        solution = solve_ivp(
            fun=self._heat_equation,
            t_span=(0.0, dt),
            y0=[self.true_temperature],
            args=(P_laser,),
            method='RK45',
            rtol=1e-6,
            atol=1e-9
        )
        
        # Extract the final integrated state
        self.true_temperature = solution.y[0][-1]
        
        return self.true_temperature

    def measure_temperature(self) -> Tuple[float, float]:
        """
        Samples the current true state and applies zero-mean Gaussian white noise
        to simulate an imperfect physical sensor (e.g., infrared pyrometer).
        
        Returns:
            Tuple[float, float]: 
                - The noisy measured temperature (z_k)
                - The true, uncorrupted temperature (T_k)
        """
        noise = np.random.normal(loc=0.0, scale=self.measurement_std)
        measured_temp = self.true_temperature + noise
        
        return measured_temp, self.true_temperature