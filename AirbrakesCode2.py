import numpy as np
from scipy.interpolate import interp1d
import time

# -----------------------------
# Flight computer / structures data
# -----------------------------
altitude = 277.2055969238281  # m
velocity = 54.36523438        # m/s
pitch = 2                     # degrees

# Data from Structures
Cd_ref = 1.3
min_area = 0
max_area = 0.001
width = 0.04
mass = 6
CdA_r = 0.00453
target_apogee = 420  # meters
dt_control = 0.1      # control timestep
n_deployments = 100   # discrete deployment steps

# -----------------------------
# Atmosphere properties
# -----------------------------
def air_properties(altitude):
    g = 9.80665
    R = 287.05
    if altitude < 11000.0:
        T = 288.15 - 0.0065 * altitude
        p = 101325.0 * (T / 288.15) ** (-g / (-0.0065 * R))
    elif altitude < 20000.0:
        T = 216.65
        p = 22632.06 * np.exp(-g * (altitude - 11000.0) / (R * T))
    elif altitude <= 32000.0:
        T = 216.65 + 0.001 * (altitude - 20000.0)
        p = 5474.89 * (T / 216.65) ** (-g / (0.001 * R))
    else:
        T = 228.65
        p = 5474.89 * np.exp(-g * (altitude - 32000.0) / (R * T))
    rho = p / (R * T)
    return rho

# Precompute air density lookup table
alt_grid = np.linspace(0, 35000, 35001)
rho_grid = np.array([air_properties(z) for z in alt_grid])
rho_lookup = interp1d(alt_grid, rho_grid, kind='linear', fill_value='extrapolate')

# -----------------------------
# CdA for airbrakes
# -----------------------------
def airbrake_CdA(velocity, altitude, area, width, Cd_ref, Re_ref=1e6):
    CdA = Cd_ref * area * 4
    return CdA

def generate_CdA_table(velocity, altitude, Cd_ref, min_area, max_area, width, n_steps=10):
    A_values = np.linspace(min_area, max_area, n_steps)
    CdA_list = np.array([airbrake_CdA(velocity, altitude, A, width, Cd_ref) for A in A_values])
    fractions = (A_values - min_area) / (max_area - min_area)
    return fractions, CdA_list

deployment_fractions, CdA_airbrakes_table = generate_CdA_table(
    velocity, altitude, Cd_ref, min_area, max_area, width, n_deployments
)

# -----------------------------
# Vectorized RK4 coast-to-apogee
# -----------------------------
def predict_apogee_vectorized(altitude, velocity, pitch, mass, CdA_r, CdA_airbrakes_array, dt=0.05, max_time=500.0):
    g = 9.80665
    pitch_rad = np.radians(pitch)

    n_candidates = len(CdA_airbrakes_array)
    state = np.zeros((n_candidates, 4))
    state[:, 1] = altitude
    state[:, 2] = velocity * np.sin(pitch_rad)
    state[:, 3] = velocity * np.cos(pitch_rad)

    t = 0.0
    while t < max_time:
        z_s = state[:, 1]
        vx_s = state[:, 2]
        vz_s = state[:, 3]

        rho = rho_lookup(np.maximum(z_s, 0.0))
        v_mag = np.hypot(vx_s, vz_s)
        v_mag[v_mag == 0] = 1e-8

        a_drag_x = -0.5 * rho * (CdA_r + CdA_airbrakes_array) / mass * v_mag * vx_s
        a_drag_z = -0.5 * rho * (CdA_r + CdA_airbrakes_array) / mass * v_mag * vz_s - g

        k1 = np.column_stack([vx_s, vz_s, a_drag_x, a_drag_z])
        k2 = np.column_stack([vx_s + 0.5 * dt * k1[:, 2], vz_s + 0.5 * dt * k1[:, 3], a_drag_x, a_drag_z])
        k3 = np.column_stack([vx_s + 0.5 * dt * k2[:, 2], vz_s + 0.5 * dt * k2[:, 3], a_drag_x, a_drag_z])
        k4 = np.column_stack([vx_s + dt * k3[:, 2], vz_s + dt * k3[:, 3], a_drag_x, a_drag_z])

        state += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        if np.all(state[:, 3] <= 0):
            break
        t += dt

    return state[:, 1]  # final altitudes

# -----------------------------
# MPC-like loop (first control cycle example)
# -----------------------------
start_time = time.time()

# Track current deployment fraction (example: start at 50%)
current_fraction = 0
max_change = 0.25
first_run = True  # <<< track if it's the first control cycle

# Determine feasible deployment fractions for this control step
min_possible = max(current_fraction - max_change, 0.0)
max_possible = min(current_fraction + max_change, 1.0)
possible_mask = (deployment_fractions >= min_possible) & (deployment_fractions <= max_possible)
possible_fractions = deployment_fractions[possible_mask]
possible_CdAs = CdA_airbrakes_table[possible_mask]

# Always include 0% deployment for reference
CdA_zero = Cd_ref * min_area * 4
fractions_all = np.concatenate(([0.0], possible_fractions))
CdAs_all = np.concatenate(([CdA_zero], possible_CdAs))

# Predict apogees only for feasible deployments (+ zero)
predicted_apogees = predict_apogee_vectorized(
    altitude, velocity, pitch, mass, CdA_r, CdAs_all, dt=0.02, max_time=500.0
)

# Match predicted apogees to fractions
apogees_dict = dict(zip(fractions_all, predicted_apogees))

# Target selection logic (NO unconstrained computation)
max_predicted = max(predicted_apogees)
if first_run:
    # First run: if best feasible predicted is below target, lower target slightly
    if max_predicted < target_apogee:
        adjusted_target = (max_predicted // 50) * 50
        target_apogee = adjusted_target
else:
    # Subsequent runs: only lower target if it's more than 10 m below
    if max_predicted < target_apogee - 10:
        adjusted_target = (max_predicted // 50) * 50
        target_apogee = adjusted_target

# Choose deployment (from feasible set) closest to target
idx_best = np.argmin(np.abs(predicted_apogees - target_apogee))
# NOTE: target_fraction is now the chosen feasible fraction (not an unconstrained value)
target_fraction = fractions_all[idx_best]
target_CdA = CdAs_all[idx_best]
target_apogee_pred = predicted_apogees[idx_best]

# Compute actual (limited) deployment to command
delta = target_fraction - current_fraction
if abs(delta) > max_change:
    delta = np.sign(delta) * max_change
new_fraction = np.clip(current_fraction + delta, 0.0, 1.0)

# Compute new predicted apogee with constrained deployment (single case)
CdA_constrained = Cd_ref * (min_area + new_fraction * (max_area - min_area)) * 4
apogee_constrained = predict_apogee_vectorized(
    altitude, velocity, pitch, mass, CdA_r, np.array([CdA_constrained]),
    dt=0.02, max_time=500.0
)[0]

# Servo mapping
servo_min, servo_max = 0.0, 60.0
servo_command = servo_min + new_fraction * (servo_max - servo_min)

# -----------------------------
# Print results (no unconstrained/optimal target printed)
# -----------------------------
end_time = time.time()
print(f"Target Apogee: {target_apogee:.1f} m")
print(f"Current deployment: {current_fraction*100:.1f}%")
print(f"Feasible range this step: {min_possible*100:.1f}%–{max_possible*100:.1f}%")
print(f"Chosen feasible deployment fraction: {target_fraction*100:.1f}%")
print(f"New (limited) deployment fraction commanded: {new_fraction*100:.1f}%")
print(f"Predicted apogee with limited deployment: {apogee_constrained:.1f} m")
print(f"Predicted apogee with no airbrakes: {apogees_dict[0.0]:.1f} m")
print(f"Runtime: {end_time - start_time:.4f} seconds")
