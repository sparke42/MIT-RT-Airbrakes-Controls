import numpy as np
from scipy.interpolate import interp1d
import time

# -----------------------------
# Flight computer / structures data
# -----------------------------
# Data from flight computer
altitude = 33200 #m
velocity = 152.4 #m/s
pitch = 2 #degrees
# Data from Structures
Cd_ref = 0.5937
min_area = 0
max_area = 0.2
width = 0.1
mass = 36
CdA_r = 1.6327
target_apogee = 33528  # meters (~110,000 ft)
dt_control = 0.1       # control timestep
n_deployments = 10     # discrete deployment steps

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
    # Precompute Re if desired, or just use Cd_ref for simplicity
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
    # States: columns = x, z, v_x, v_z
    state = np.zeros((n_candidates, 4))
    state[:,1] = altitude           # z
    state[:,2] = velocity * np.sin(pitch_rad)  # v_x
    state[:,3] = velocity * np.cos(pitch_rad)  # v_z

    t = 0.0
    while t < max_time:
        z_s = state[:,1]
        vx_s = state[:,2]
        vz_s = state[:,3]

        # Interpolate rho
        rho = rho_lookup(np.maximum(z_s, 0.0))

        v_mag = np.hypot(vx_s, vz_s)
        v_mag[v_mag == 0] = 1e-8  # avoid division by zero

        a_drag_x = -0.5 * rho * (CdA_r + CdA_airbrakes_array) / mass * v_mag * vx_s
        a_drag_z = -0.5 * rho * (CdA_r + CdA_airbrakes_array) / mass * v_mag * vz_s - g

        # RK4 step (simplified: use same a_drag for all k1..k4)
        k1 = np.column_stack([vx_s, vz_s, a_drag_x, a_drag_z])
        k2 = np.column_stack([vx_s + 0.5*dt*k1[:,2], vz_s + 0.5*dt*k1[:,3], a_drag_x, a_drag_z])
        k3 = np.column_stack([vx_s + 0.5*dt*k2[:,2], vz_s + 0.5*dt*k2[:,3], a_drag_x, a_drag_z])
        k4 = np.column_stack([vx_s + dt*k3[:,2], vz_s + dt*k3[:,3], a_drag_x, a_drag_z])

        state += dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        # Check if all candidates have reached apogee
        if np.all(state[:,3] <= 0):
            break
        t += dt

    # Interpolate exact apogee for candidates that passed v_z <= 0
    z_apogee = state[:,1]
    return z_apogee

# -----------------------------
# MPC-like loop
# -----------------------------
start_time = time.time()

# Vectorized prediction for all deployments at once
predicted_apogees = predict_apogee_vectorized(
    altitude, velocity, pitch, mass, CdA_r, CdA_airbrakes_table, dt=0.02, max_time=500.0
)

# Choose deployment fraction closest to target
idx_best = np.argmin(np.abs(predicted_apogees - target_apogee))
best_fraction = deployment_fractions[idx_best]
best_CdA = CdA_airbrakes_table[idx_best]
best_apogee = predicted_apogees[idx_best]

# Map fraction to servo command
servo_min, servo_max = 0.0, 60.0
servo_command = servo_min + best_fraction * (servo_max - servo_min)

end_time = time.time()

print(f"Best airbrakes deployment fraction: {best_fraction*100:.1f}%")
print(f"Predicted apogee: {best_apogee:.1f} m")
print(f"Servo command: {servo_command:.1f}°")
print(f"Runtime: {end_time - start_time:.4f} seconds")
