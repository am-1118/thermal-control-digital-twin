# Thermal Control Digital Twin

This repository contains a closed-loop digital twin built to manage the thermal processing of a sapphire wafer. It simulates physical temperature dynamics and compares two different control strategies: Proportional-Integral-Derivative (PID) and Linear Quadratic Gaussian (LQG). The system tracks physical uncertainties in real-time, specifically unmodeled variances in mass and absorptivity.

---

## 1. System Dynamics & Numerical Integration

The core of the simulation is a physics engine representing a laser heating plant. 

### Physical Model
The plant's temperature dynamics are calculated by balancing the incoming heat against heat lost to the environment. The equations account for three main heat transfer components:
*   **Heat Input ($q_{in}$):** The energy absorbed from the laser, calculated using the laser power and the wafer's absorptivity.
*   **Convective Loss ($q_{conv}$):** Heat lost to the surrounding air, determined by the convective heat transfer coefficient, surface area, and the temperature difference between the wafer and ambient air.
*   **Radiative Loss ($q_{rad}$):** Heat emitted as radiation, calculated using the Stefan-Boltzmann law, the wafer's emissivity, and the difference between the fourth powers of the wafer temperature and ambient temperature.

### Numerical Integration
The continuous physics equations are integrated digitally over discrete time steps. 
*   The simulation uses a fixed time step of 0.1 seconds. 
*   At each step, the plant uses the commanded laser power to step the true temperature forward in time.

---

## 2. Clarification on Controller Input

To end the confusion regarding what data is fed into the controllers: **neither the PID nor the LQG controller receives the raw, noisy pyrometer data**. 
*   In the PID script, the controller receives the `current_estimate` variable.
*   In the LQG script, the controller also receives the `current_estimate` variable.
*   In both architectures, this `current_estimate` is the smoothed, filtered output produced by the Extended Kalman Filter (EKF), not the raw sensor measurement.

---

## 3. Baseline Architecture: PID Control

The first architecture utilizes a standard PID controller alongside an EKF and a Recursive Least Squares (RLS) estimator.

### Reasons for Selection
PID is straightforward to tune and implement without requiring complex matrix math. The user interface allows for direct manual tuning of the Proportional ($K_p$), Integral ($K_i$), and Derivative ($K_d$) gains. 

### Known Flaws
While simple, PID struggles to elegantly handle physical actuator limits. It also reacts to errors rather than predicting them, making it less optimal for the highly non-linear radiative losses that occur at higher temperatures.

### Data Flow Pipeline
*   The PID controller calculates the required laser power using the temperature setpoint and the EKF's current temperature estimate.
*   The physical plant advances one time step using this commanded power.
*   The system takes a noisy pyrometer measurement.
*   The EKF predicts the next state based on the laser power.
*   The EKF updates its internal estimate using the new noisy measurement.
*   The RLS evaluates the rate of temperature change. 
*   If the absolute rate of change is 2.0 K/s or less, the RLS updates its real-time estimates for mass and absorptivity.

---

## 4. Optimal Architecture: LQG Control

The second architecture replaces the PID loop with a Linear Quadratic Integral (LQI) controller, forming a complete LQG system. 

### Reasons for Selection
LQG provides mathematically optimal control by balancing tracking performance against energy consumption. The interface allows tuning of the $Q$ matrix (penalizing state and integral errors) and the $R$ matrix (penalizing control effort). This inherently minimizes actuator chatter and guarantees zero steady-state error.

### Known Flaws
The system exposes the Kalman Filter tuning paradox. Setting the EKF process noise ($Q$) to 0.02 forces the filter to heavily trust its internal model, which successfully filters sensor noise but introduces a mathematical bias when physical mutations occur. Additionally, setting the RLS persistent excitation threshold too tight (1.5 K/s) causes the estimator to accidentally digest steady-state noise, leading to degraded parameter tracking.

### Data Flow Pipeline
*   The physical plant advances one time step using the previous laser power command.
*   The system takes a noisy pyrometer measurement.
*   The EKF predicts the state based on the previous power command.
*   The EKF immediately updates its estimate using the new noisy measurement.
*   The RLS calculates the rate of temperature change.
*   If the absolute rate of change is 1.5 K/s or less, the RLS updates its mass and absorptivity estimates.
*   The LQI controller calculates the new optimal laser power using only the newly updated EKF estimate.

---

## 5. Applications and Material Selection

Sapphire was selected as the control material due to its extreme thermal stability and relevance in advanced computing hardware. 

Precise thermal control of synthetic sapphire is a critical requirement in the fabrication of next-generation quantum computing chips and specialized microelectronics. Qubits are highly sensitive to thermal fluctuations during physical vapor deposition and annealing processes. By utilizing robust LQG digital twins, hardware fabrication facilities can maintain strict thermal tolerances, directly improving qubit coherence times and wafer yield. Furthermore, the low computational overhead of these state-space architectures makes them ideal for direct deployment onto FPGAs for ultra-low-latency edge computing applications and AI-on-chip thermal management.
