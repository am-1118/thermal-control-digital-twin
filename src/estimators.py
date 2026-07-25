import numpy as np
from typing import Tuple, Callable

class LinearKalmanFilter:
    """
    A standard discrete-time Linear Kalman Filter (LKF).
    Operates on a linearized state-space model around a fixed operating point.
    """
    def __init__(self, F: float, G: float, H: float, Q: float, R: float, P0: float, x0_dev: float = 0.0):
        """
        Args:
            F (float): Discrete state transition scalar (1 + Ac*dt).
            G (float): Discrete control input scalar (Bc*dt).
            H (float): Observation scalar (1.0 for direct measurement).
            Q (float): Process noise covariance.
            R (float): Measurement noise covariance.
            P0 (float): Initial error covariance.
            x0_dev (float): Initial state deviation from the operating point.
        """
        self.F = F
        self.G = G
        self.H = H
        self.Q = Q
        self.R = R
        
        # State and covariance tracking
        self.x_hat = x0_dev
        self.P = P0

    def predict(self, u_dev: float) -> float:
        """
        Time Update: Project the state and covariance forward.
        """
        # Linear state prediction
        self.x_hat = self.F * self.x_hat + self.G * u_dev
        
        # Covariance prediction
        self.P = self.F * self.P * self.F + self.Q
        return self.x_hat

    def update(self, z_dev: float) -> float:
        """
        Measurement Update: Correct the state estimate using the noisy sensor data.
        """
        # Calculate Kalman Gain
        S = self.H * self.P * self.H + self.R
        K = (self.P * self.H) / S
        
        # Update estimate with measurement innovation (residual)
        self.x_hat = self.x_hat + K * (z_dev - self.H * self.x_hat)
        
        # Update error covariance
        self.P = (1.0 - K * self.H) * self.P
        
        return self.x_hat


class ExtendedKalmanFilter:
    """
    A continuous-discrete Extended Kalman Filter (EKF).
    Uses the exact non-linear ODE for state prediction and dynamically 
    calculates the Jacobian for covariance updates.
    """
    def __init__(self, 
                 physics_model: Callable[[float, float], float], 
                 jacobian_model: Callable[[float], float], 
                 dt: float, Q: float, R: float, P0: float, x0: float):
        """
        Args:
            physics_model (Callable): The exact non-linear ODE f(T, P).
            jacobian_model (Callable): Function returning the continuous Jacobian Fc at a given T.
            dt (float): Discrete time step duration.
            Q (float): Process noise covariance.
            R (float): Measurement noise covariance.
            P0 (float): Initial error covariance.
            x0 (float): Initial absolute temperature estimate (K).
        """
        self.physics = physics_model
        self.jacobian = jacobian_model
        self.dt = dt
        self.H = 1.0  # Direct temperature measurement
        self.Q = Q
        self.R = R
        
        self.x_hat = x0
        self.P = P0

    def predict(self, u: float) -> float:
        """
        Time Update: Non-linear state projection and dynamic covariance projection.
        """
        # 1. Non-linear state prediction (Forward Euler integration)
        dT_dt = self.physics(self.x_hat, u)
        self.x_hat = self.x_hat + dT_dt * self.dt
        
        # 2. Dynamic Jacobian calculation
        Fc = self.jacobian(self.x_hat)
        Fk = 1.0 + Fc * self.dt  # Discretized state transition scalar
        
        # 3. Covariance prediction using local Jacobian
        self.P = Fk * self.P * Fk + self.Q
        return self.x_hat

    def update(self, z: float) -> float:
        """
        Measurement Update: Correct the state estimate using the noisy sensor data.
        """
        S = self.H * self.P * self.H + self.R
        K = (self.P * self.H) / S
        
        self.x_hat = self.x_hat + K * (z - self.H * self.x_hat)
        self.P = (1.0 - K * self.H) * self.P
        
        return self.x_hat
    
class RecursiveLeastSquares:
    """
    Recursive Least Squares (RLS) estimator for online parameter identification.
    Includes anti-windup protection for periods lacking persistent excitation.
    """
    def __init__(self, num_params: int, lambda_factor: float = 0.99, P0_diagonal: float = 1000.0):
        self.num_params = num_params
        self.lambda_factor = lambda_factor
        # Parameters initialized to zero, will rapidly converge during initial excitation
        self.theta = np.zeros((num_params, 1))
        self.P = np.eye(num_params) * P0_diagonal

    def update(self, phi: np.ndarray, y: float, freeze_update: bool = False) -> np.ndarray:
        """
        Executes a single RLS update step.
        
        Args:
            phi: The regressor vector (column vector).
            y: The target measurement.
            freeze_update: Supervisory flag. If True, suspends the update to 
                           prevent covariance windup and parameter drift when 
                           the system lacks persistent excitation.
        """
        # --- Anti-Windup Safeguard ---
        if freeze_update:
            return self.theta

        # --- Standard RLS Mathematics ---
        # 1. Calculate Kalman Gain
        P_phi = np.dot(self.P, phi)
        denominator = self.lambda_factor + np.dot(phi.T, P_phi)
        
        # Prevent division by zero if regressor is completely dead
        if denominator[0, 0] < 1e-12:
            return self.theta
            
        K = P_phi / denominator
        
        # 2. Calculate A Priori Error
        error = y - np.dot(phi.T, self.theta)[0, 0]
        
        # 3. Update Parameter Estimates
        self.theta = self.theta + K * error
        
        # 4. Update Covariance Matrix (with forgetting factor)
        self.P = (self.P - np.dot(K, P_phi.T)) / self.lambda_factor
        
        return self.theta