import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# USER CONFIGURATION
# ============================================================
csv_path = '/Users/sydneyparke/Documents/MIT/Rocket Team/xanthus.csv'

unit_system = "metric"     # 'imperial' or 'metric'
airbrake_areas = [0.004, 0.001, 0.015]  # example areas in m²
Cd_airbrake_base = 1.2       # base drag coefficient for airbrakes
target_reductions = [5, 8, 10]  # meters
deployment_modes = ['full']
burnout_time = 1.0            # seconds, airbrakes can only deploy after this

# ------------------------------------------------------
# Extract columns from CSV
# ------------------------------------------------------
data = pd.read_csv(csv_path, header=None, sep=',')
time_s  = data.iloc[:, 0].to_numpy()           # time in seconds
alt_all = data.iloc[:, 1].to_numpy()           # altitude
vel_all = data.iloc[:, 2].to_numpy()           # velocity
angleofattack_all = data.iloc[:, 3].to_numpy() # angle of attack
cd_all  = data.iloc[:, 4].to_numpy()           # Cd

# Unit conversion if needed
if unit_system.lower() == "imperial":
    alt_all = alt_all * 0.3048
    vel_all = vel_all * 0.3048

# Pre-apogee mask
pre_apogee_mask = vel_all > 0
time = time_s[pre_apogee_mask]
alt = alt_all[pre_apogee_mask]
vel = vel_all[pre_apogee_mask]
cd  = cd_all[pre_apogee_mask]
aoa = angleofattack_all[pre_apogee_mask]

# ------------------------------------------------------
# Post-burnout mask
# ------------------------------------------------------
post_burnout_mask = time >= burnout_time
alt_pb = alt[post_burnout_mask]
vel_pb = vel[post_burnout_mask]
cd_pb  = cd[post_burnout_mask]
aoa_pb = aoa[post_burnout_mask]


# Rocket constants
m = 13 * 0.453592        # kg
g = 9.81                 # m/s^2
D = 0.1016               # 6 in diameter in meters
A_rocket = np.pi * (D/2)**2
rho = 1.225              # kg/m^3

# Baseline (no airbrakes)
F_drag_rocket = 0.5 * rho * cd_pb * A_rocket * vel_pb**2
delta_h = vel_pb**2 / (2*g) - (F_drag_rocket / (m*g) * (vel_pb/g))
baseline_apogee = alt_pb[0] + np.max(delta_h)
print(f"Baseline predicted apogee: {baseline_apogee:.2f} m\n")

# ------------------------------------------------------
# Function: deployment mask
# ------------------------------------------------------
def get_deployment_mask(velocity, mode):
    if mode == 'full':
        return np.ones_like(velocity, dtype=bool)
    elif mode == 'transonic_subsonic':
        return velocity < 343
    elif mode == 'subsonic':
        return velocity < 300
    else:
        raise ValueError(f"Unknown mode: {mode}")

# Function: altitude range for a mask
def print_altitude_range(mask, name):
    if np.any(mask):
        alt_min = np.min(alt_pb[mask])
        alt_max = np.max(alt_pb[mask])
        print(f"{name}: from {alt_min:.2f} m to {alt_max:.2f} m")
    else:
        print(f"{name}: No points in this region")

# ------------------------------------------------------
# Compute flight curves and minimum areas
# ------------------------------------------------------
all_results = {}       # flight curves
min_area_results = {}  # minimum airbrake areas

for mode in deployment_modes:
    deployment_mask = get_deployment_mask(vel_pb, mode)
    print_altitude_range(deployment_mask, f"Deployment region ({mode.replace('_',' ').title()})")

    results = []
    flight_curves = []

    # Compute flight curves for fixed airbrake areas
    for A_ab in airbrake_areas:
        # Drag depends on angle of attack (simple linear scaling)
        Cd_eff = Cd_airbrake_base * (1 + 0.01 * aoa_pb)  # 1% increase per degree
        F_drag_airbrake = 0.5 * rho * Cd_eff * A_ab * vel_pb**2 * deployment_mask
        total_drag = F_drag_rocket + F_drag_airbrake
        delta_h_airbrake = vel_pb**2 / (2*g) - (total_drag / (m*g) * (vel_pb/g))
        predicted_apogee = alt_pb[0] + np.max(delta_h_airbrake)

        results.append({
            'Airbrake Area (m^2)': A_ab,
            'Airbrake Drag Force (avg N)': np.mean(F_drag_airbrake),
            'Predicted Apogee (m)': predicted_apogee
        })
        flight_curves.append(alt_pb[0] + delta_h_airbrake)

    all_results[mode] = {
        'results': pd.DataFrame(results),
        'flight_curves': flight_curves
    }

    # Compute minimum area to reduce apogee by target reductions
    min_area_dict = {}
    A_test_vals = np.linspace(0.0001, 0.5, 5000)  # fine grid
    for target in target_reductions:
        for A_test in A_test_vals:
            Cd_eff = Cd_airbrake_base * (1 + 0.01 * aoa_pb)
            F_drag_airbrake = 0.5 * rho * Cd_eff * A_test * vel_pb**2 * deployment_mask
            total_drag = F_drag_rocket + F_drag_airbrake
            delta_h_airbrake = vel_pb**2 / (2*g) - (total_drag / (m*g) * (vel_pb/g))
            predicted_apogee = alt_pb[0] + np.max(delta_h_airbrake)
            if baseline_apogee - predicted_apogee >= target:
                min_area_dict[target] = A_test
                break
        else:
            min_area_dict[target] = None  # target not achievable
    min_area_results[mode] = min_area_dict

# ------------------------------------------------------
# Print results
# ------------------------------------------------------
for mode, data_dict in all_results.items():
    print(f"\n--- Deployment Mode: {mode.replace('_',' ').title()} ---")
    print(data_dict['results'])

print("\n--- Minimum Airbrake Areas for Target Apogee Reductions ---")
for mode, targets in min_area_results.items():
    print(f"\nMode: {mode.replace('_',' ').title()}")
    for target, area in targets.items():
        if area is not None:
            print(f"Reduce by {target} m: {area:.4f} m²")
        else:
            print(f"Reduce by {target} m: Not achievable")

# ------------------------------------------------------
# Plot flight curves
# ------------------------------------------------------
plt.figure(figsize=(10,6))
plt.plot(vel_pb, alt_pb[0] + delta_h, label='No Airbrakes', linewidth=2, color='black')

colors = ['blue', 'green', 'red']
for i, mode in enumerate(deployment_modes):
    for j, A_ab in enumerate(airbrake_areas):
        plt.plot(vel_pb, all_results[mode]['flight_curves'][j], linestyle='--',
                 color=colors[i], alpha=0.7,
                 label=f'{mode.replace("_"," ").title()} - {A_ab:.3f} m²')

plt.xlabel('Velocity (m/s)')
plt.ylabel('Altitude (m)')
plt.title('Effect of Airbrakes on Apogee (All Deployment Modes)')
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------
# Plot minimum area vs target reductions
# ------------------------------------------------------
plt.figure(figsize=(8,6))
for i, mode in enumerate(deployment_modes):
    areas = [min_area_results[mode][t] for t in target_reductions]
    plt.plot(target_reductions, areas, marker='o', color=colors[i], label=mode.replace('_',' ').title())

plt.xlabel('Target Apogee Reduction (m)')
plt.ylabel('Minimum Airbrake Area (m²)')
plt.title('Minimum Airbrake Area Required by Deployment Mode')
plt.grid(True)
plt.legend()
plt.show()
