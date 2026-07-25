import numpy as np
import scipy.linalg as la

class PIDController:
    """
    Discrete-time PID controller with conditional integration (clamping) 
    anti-windup logic and derivative on measurement.
    """
    def __init__(self, Kp: float, Ki: float, Kd: float, dt: float, u_min: float = 0.0, u_max: float = 100.0):
        """
        Args:
            Kp (float): Proportional gain.
            Ki (float): Integral gain.
            Kd (float): Derivative gain.
            dt (float): Discrete time step (seconds).
            u_min (float): Minimum control output (e.g., 0 W).
            u_max (float): Maximum control output (e.g., 100 W).
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.u_min = u_min
        self.u_max = u_max
        
        # State registers
        self.integral_error = 0.0
        self.prev_measured = 0.0  # Used to prevent derivative kick
        
    def reset(self):
        """Clears the internal state registers."""
        self.integral_error = 0.0
        self.prev_measured = 0.0

    def compute(self, setpoint: float, measured_value: float) -> float:
        """
        Computes the control output for the current time step.
        
        Args:
            setpoint (float): The target temperature.
            measured_value (float): The current estimated temperature.
            
        Returns:
            float: The commanded laser power (clamped).
        """
        error = setpoint - measured_value
        
        # Proportional term
        P_term = self.Kp * error
        
        # Derivative on measurement (prevents massive spikes if setpoint suddenly changes)
        derivative = -(measured_value - self.prev_measured) / self.dt
        D_term = self.Kd * derivative
        
        # Nominal Control Effort (Predicting without integration yet)
        u_nominal = P_term + self.Ki * (self.integral_error + error * self.dt) + D_term
        
        # --- Clamping Anti-Windup Logic ---
        # 1. Is the actuator saturated?
        is_saturated = (u_nominal >= self.u_max) or (u_nominal <= self.u_min)
        
        # 2. Are the error and the nominal control effort in the same direction?
        # We use np.sign to determine the direction of the push.
        same_sign = np.sign(error) == np.sign(u_nominal)
        
        # 3. Conditional Integration
        if not (is_saturated and same_sign):
            self.integral_error += error * self.dt
            
        I_term = self.Ki * self.integral_error
        
        # Final Output Calculation
        u_out = P_term + I_term + D_term
        
        # Enforce hard hardware limits
        u_actual = max(self.u_min, min(self.u_max, u_out))
        
        # Update register for next step
        self.prev_measured = measured_value
        
        return u_actual

class LQIController:
    """
    Linear Quadratic Integral (LQI) Controller.
    Calculates optimal state-feedback gains by solving the Discrete Algebraic Riccati Equation (DARE).
    """
    def __init__(self, A_c: float, B_c: float, dt: float, Q: np.ndarray, R: np.ndarray, 
                 T_eq: float, P_eq: float, u_min: float = 0.0, u_max: float = 100.0):
        
        self.dt = dt
        self.T_eq = T_eq
        self.P_eq = P_eq
        self.u_min = u_min
        self.u_max = u_max
        
        # 1. Discretize the continuous plant dynamics (Forward Euler)
        A_d = 1.0 + A_c * self.dt
        B_d = B_c * self.dt
        
        # 2. Construct the Augmented LQI Matrices
        self.A_aug = np.array([
            [A_d, 0.0],
            [self.dt, 1.0]
        ])
        
        self.B_aug = np.array([
            [B_d],
            [0.0]
        ])
        
        # 3. Solve the Discrete Algebraic Riccati Equation (DARE)
        S = la.solve_discrete_are(self.A_aug, self.B_aug, Q, R)
        
        # 4. Extract the Optimal Feedback Gain Matrix (K)
        term1 = np.linalg.inv(R + self.B_aug.T @ S @ self.B_aug)
        term2 = self.B_aug.T @ S @ self.A_aug
        self.K_lqr = term1 @ term2  # Shape: (1, 2)
        
        self.K_temp = self.K_lqr[0, 0]
        self.K_int = self.K_lqr[0, 1]
        
        # 5. Initialize the Integral State
        self.integral_error = 0.0

    def compute(self, measured_value: float) -> float:
        """
        Computes the optimal laser power given the current temperature estimate.
        """
        # 1. Calculate state deviation (x_k)
        x_k = measured_value - self.T_eq
        
        # 2. Calculate optimal control deviation (Delta P)
        delta_P = -(self.K_temp * x_k + self.K_int * self.integral_error)
        
        # 3. Convert back to absolute physical power
        u_cmd = self.P_eq + delta_P
        
        # 4. Hardware Saturation & Anti-Windup Clamping
        if u_cmd > self.u_max:
            u_cmd = self.u_max
        elif u_cmd < self.u_min:
            u_cmd = self.u_min
        else:
            # Only accumulate integral error if the actuator is NOT saturated
            self.integral_error += x_k * self.dt
            
        return u_cmd