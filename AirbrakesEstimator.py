import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load data from CSV
# -----------------------------
# Replace 'flight_data.csv' with the path to your CSV
# Assumes column 1 = altitude (m), column 2 = velocity (m/s)
data = pd.read_csv('/Users/sydneyparke/Documents/RT-MPC/Xanthus-altvsvel.csv', header=None)  # no headers in your CSV
alt_all = data.iloc[:,0].to_numpy()  # altitude in meters
vel_all = data.iloc[:,1].to_numpy()  # velocity in m/s

# Use only pre-apogee data (velocity > 0)
pre_apogee_mask = vel_all > 0
alt = alt_all[pre_apogee_mask]
vel = vel_all[pre_apogee_mask]

# -------------------------
# Rocket constants
# -------------------------
m = 13 * 0.453592       # mass in kg
g = 9.81                # gravity m/s^2
Cd_rocket = 0.75        # rocket drag coefficient
D = 0.1016              # 4 in diameter in meters
A_rocket = np.pi*(D/2)**2
rho = 1.225             # air density kg/m^3

# -------------------------
# Baseline forces
# -------------------------
F_gravity = m * g
F_drag_rocket = 0.5 * rho * Cd_rocket * A_rocket * vel**2

# -------------------------
# Baseline apogee estimate
# -------------------------
drag_correction = F_drag_rocket / (m*g) * (vel/g)  # approximate
delta_h = vel**2 / (2*g) - drag_correction
baseline_apogee = alt[0] + np.max(delta_h)

print(f"Gravity force: {F_gravity:.2f} N")
print(f"Rocket drag force (avg): {np.mean(F_drag_rocket):.2f} N")
print(f"Baseline predicted apogee (approx): {baseline_apogee:.2f} m")

# -------------------------
# Airbrake parameters
# -------------------------
Cd_airbrake = 1.28
airbrake_areas = np.linspace(0.002, 0.01, 10)  # curve plotting

# -------------------------
# Compute apogee with airbrakes
# -------------------------
results = []
flight_curves = []

for A_ab in airbrake_areas:
    F_drag_airbrake = 0.5 * rho * Cd_airbrake * A_ab * vel**2
    total_drag = F_drag_rocket + F_drag_airbrake
    drag_correction = total_drag / (m*g) * (vel/g)
    delta_h_airbrake = vel**2 / (2*g) - drag_correction
    predicted_apogee = alt[0] + np.max(delta_h_airbrake)

    results.append({
        'Airbrake Area (m^2)': A_ab,
        'Airbrake Drag Force (N, avg)': np.mean(F_drag_airbrake),
        'Predicted Apogee (m)': predicted_apogee
    })

    flight_curves.append(alt[0] + delta_h_airbrake)

# -------------------------
# Estimate minimum airbrake area to reduce apogee by 50m
# -------------------------
target_reduction = 5  # meters
A_test = 0.001
while True:
    F_drag_airbrake = 0.5 * rho * Cd_airbrake * A_test * vel**2
    total_drag = F_drag_rocket + F_drag_airbrake
    drag_correction = total_drag / (m*g) * (vel/g)
    delta_h_airbrake = vel**2 / (2*g) - drag_correction
    predicted_apogee = alt[0] + np.max(delta_h_airbrake)

    if baseline_apogee - predicted_apogee >= target_reduction:
        min_area_for_50m_drop = A_test
        break
    A_test += 0.0001  # small increment for precision

print(f"\nMinimum airbrake area to reduce apogee by at least 5 m: {min_area_for_50m_drop:.4f} m²")

# -------------------------
# Print results table
# -------------------------
df_results = pd.DataFrame(results)
print("\nAirbrake effect on apogee:")
print(df_results)

# -------------------------
# Plot flight path curves
# -------------------------
plt.figure(figsize=(10,6))

# baseline
plt.plot(vel, alt[0]+delta_h, label='No Airbrakes', linewidth=2, color='black')

# with airbrakes
for i, A_ab in enumerate(airbrake_areas):
    plt.plot(vel, flight_curves[i], label=f'Airbrake {A_ab:.3f} m²', linestyle='--')

plt.xlabel('Velocity (m/s)')
plt.ylabel('Altitude (m)')
plt.title('Effect of Airbrakes on Rocket Flight Path')
plt.legend()
plt.grid(True)
plt.show()
