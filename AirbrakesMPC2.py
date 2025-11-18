import time
import matplotlib.pyplot as plt

# -----------------------------
# Flight computer / structures data
# -----------------------------
altitude = 2584.918  # m
velocity = 319.785        # m/s
pitch = 0.0                    # degrees; 0 = vertical

Cd_ref = 1.28
min_area = 0.0
max_area = 0.00165
width = 0.066
mass = 37.65
CdA_r = 0.00958
target_apogee = 5500

# control params
dt_control = 0.1
n_deployments = 50
current_fraction = 0.0
max_change = 0.25
first_run = True
run_one_step_only = False  # <--- Set True to run only one MPC step

# -----------------------------
# Atmosphere
# -----------------------------
def air_properties(alt):
    g = 9.80665
    R = 287.05
    if alt < 11000.0:
        T = 288.15 - 0.0065 * alt
        p = 101325.0 * (T / 288.15) ** (-g / (-0.0065 * R))
    elif alt < 20000.0:
        T = 216.65
        p = 22632.06 * pow(2.718281828, -g * (alt - 11000.0) / (R * T))
    elif alt <= 32000.0:
        T = 216.65 + 0.001 * (alt - 20000.0)
        p = 5474.89 * (T / 216.65) ** (-g / (0.001 * R))
    else:
        T = 228.65
        p = 5474.89 * pow(2.718281828, -g * (alt - 32000.0) / (R * T))
    rho = p / (R * T)
    return rho

# -----------------------------
# CdA table
# -----------------------------
def CdA_from_fraction(fraction):
    return Cd_ref * (min_area + fraction * (max_area - min_area)) * 4.0

deployment_fractions = [i / (n_deployments - 1) for i in range(n_deployments)]
CdA_airbrakes_table = [CdA_from_fraction(f) for f in deployment_fractions]

# -----------------------------
# RK4 apogee simulation
# -----------------------------
def simulate_until_apogee(initial_alt, initial_speed, pitch_deg, mass, CdA_r, CdA_schedule_func,
                          dt_inner=0.02, max_time=500.0):
    g = 9.80665
    pitch_rad = pitch_deg * 3.14159265 / 180.0

    vx = initial_speed * pow(1.0 - pitch_rad**2, 0)  # horizontal component not used
    vz = initial_speed  # vertical since pitch=0 is vertical
    z = initial_alt
    x = 0.0
    t = 0.0

    prev_z = z
    prev_vz = vz
    prev_t = t

    if vz <= 0.0:
        return float(z)

    while t < max_time:
        vmag = vz
        if vmag == 0:
            vmag = 1e-8
        rho = air_properties(max(z,0.0))
        total_CdA = CdA_r + CdA_schedule_func(t, z)
        ax = 0.0
        az = -0.5 * rho * total_CdA / mass * vmag * vz - g

        # RK4 integration
        k1z = vz
        k1vz = az

        k2z = vz + 0.5 * dt_inner * k1vz
        k2vz = -0.5 * rho * total_CdA / mass * vmag * (vz + 0.5 * dt_inner * k1vz) - g

        k3z = vz + 0.5 * dt_inner * k2vz
        k3vz = -0.5 * rho * total_CdA / mass * vmag * (vz + 0.5 * dt_inner * k2vz) - g

        k4z = vz + dt_inner * k3vz
        k4vz = -0.5 * rho * total_CdA / mass * vmag * (vz + dt_inner * k3vz) - g

        z_new = z + dt_inner / 6.0 * (k1z + 2*k2z + 2*k3z + k4z)
        vz_new = vz + dt_inner / 6.0 * (k1vz + 2*k2vz + 2*k3vz + k4vz)

        if vz_new <= 0.0:
            denom = prev_vz - vz_new
            if abs(denom) < 1e-12:
                return float(max(prev_z, z_new))
            alpha = prev_vz / denom
            apogee_z = prev_z + alpha * (z_new - prev_z)
            return float(apogee_z)

        if z_new <= 0.0:
            return 0.0

        prev_z, prev_vz, prev_t = z_new, vz_new, t + dt_inner
        z, vz, t = z_new, vz_new, t + dt_inner

    return float(z)

# -----------------------------
# Generate evenly spaced sequences
# -----------------------------
def generate_sequences(curr_fraction, deployment_fractions, horizon, K_options, max_change):
    sequences = []

    def recurse(prefix):
        if len(prefix) == horizon + 1:
            sequences.append(prefix[1:].copy())
            return
        prev = prefix[-1]
        min_val = max(prev - max_change, 0.0)
        max_val = min(prev + max_change, 1.0)
        if K_options == 1:
            feasible = [prev]
        else:
            feasible = [min_val + i*(max_val-min_val)/(K_options-1) for i in range(K_options)]
        for f in feasible:
            prefix.append(f)
            recurse(prefix)
            prefix.pop()

    recurse([curr_fraction])
    return sequences

# -----------------------------
# CdA schedule from sequence
# -----------------------------
def make_CdA_schedule_from_sequence(base_fraction, seq_fractions, dt_control):
    def CdA_schedule(t, z):
        idx = int(t / dt_control)
        if idx >= len(seq_fractions):
            frac = seq_fractions[-1]
        else:
            frac = seq_fractions[idx]
        return CdA_from_fraction(frac)
    return CdA_schedule

# -----------------------------
# Apogee scan (diagnostic)
# -----------------------------
# apogee_scan = []
# print("=== Apogee vs Deployment Scan ===")
# for i in range(n_deployments):
#     frac = i / (n_deployments - 1)
#     schedule = lambda t, z, f=frac: CdA_from_fraction(f)
#     apogee = simulate_until_apogee(altitude, velocity, pitch, mass, CdA_r, schedule)
#     apogee_scan.append(apogee)
#     print(f"Fraction: {frac:.3f}, Predicted apogee: {apogee:.2f} m")

# plt.figure()
# plt.plot([i/(n_deployments-1) for i in range(n_deployments)], apogee_scan, '-o')
# plt.xlabel('Deployment fraction')
# plt.ylabel('Predicted apogee (m)')
# plt.title('Apogee vs Airbrake Deployment')
# plt.grid(True)
# plt.show()

# -----------------------------
# MPC loop
# -----------------------------
horizon = 5
K_options = 3
max_control_steps = 1
tolerance = 1.0
control_log = []

start_time = time.time()

for step in range(max_control_steps):

    # ----- target relaxation -----
    test_schedule = make_CdA_schedule_from_sequence(current_fraction, [current_fraction], dt_control)
    pred_ap = simulate_until_apogee(altitude, velocity, pitch, mass, CdA_r, test_schedule)
    if first_run:
        if pred_ap < target_apogee:
            target_apogee = int(pred_ap // 10) * 10
    else:
        if pred_ap < target_apogee - 10:
            target_apogee = int(pred_ap // 10) * 10

    print(f"\nCurrent target_apogee: {target_apogee:.1f} m")

    # ----- generate candidate sequences -----
    seqs = generate_sequences(current_fraction, deployment_fractions, horizon, K_options, max_change)

    best_seq = None
    best_cost = float('inf')
    best_apogee = None

    t_step_start = time.time()

    for seq in seqs:
        # print("Trying deployment sequence:", seq)
        schedule = make_CdA_schedule_from_sequence(current_fraction, seq, dt_control)
        apogee_seq = simulate_until_apogee(altitude, velocity, pitch, mass, CdA_r, schedule)
        cost = abs(apogee_seq - target_apogee)
        if cost < best_cost:
            best_cost = cost
            best_seq = seq
            best_apogee = apogee_seq

    requested_action = best_seq[0]
    delta = requested_action - current_fraction
    if abs(delta) > max_change:
        delta = (delta/abs(delta)) * max_change
    implemented_fraction = max(0.0, min(1.0, current_fraction + delta))

    # predicted apogee if held
    schedule_hold = lambda t, z: CdA_from_fraction(implemented_fraction)
    apogee_hold = simulate_until_apogee(altitude, velocity, pitch, mass, CdA_r, schedule_hold)

    servo_min, servo_max = 0.0, 60.0
    servo_command = servo_min + implemented_fraction * (servo_max - servo_min)

    t_step_end = time.time()

    print(f"\nMPC step {step+1}")
    print(f" Current fraction: {current_fraction:.3f}")
    print(f" Best sequence (first 3 of {horizon}): {[round(f,3) for f in best_seq[:3]]} ...")
    print(f" Predicted apogee of best seq: {best_apogee:.3f} m (cost {best_cost:.2f} m)")
    print(f" Implementing first action (requested {requested_action:.3f}) -> limited to {implemented_fraction:.3f}")
    print(f" Predicted apogee with limited action held: {apogee_hold:.3f} m")
    print(f" Step runtime: {t_step_end - t_step_start:.4f} s")
    print(f" Servo command: {servo_command:.2f} deg")

    control_log.append({
        'step': step,
        'current_fraction': current_fraction,
        'requested_action': requested_action,
        'implemented_fraction': implemented_fraction,
        'pred_apogee_best_seq': best_apogee,
        'pred_apogee_implemented': apogee_hold,
        'servo': servo_command,
        'cost': best_cost,
        'step_runtime': t_step_end - t_step_start
    })

    current_fraction = implemented_fraction
    first_run = False

    if abs(apogee_hold - target_apogee) <= tolerance:
        print(f"Reached target within ±{tolerance} m; stopping MPC loop.")
        break

    if run_one_step_only:
        print("Run-one-step mode enabled; stopping MPC loop after first step.")
        break

end_time = time.time()
print(f"\nFinal commanded deployment: {current_fraction*100:.2f}% -> servo {servo_command:.2f} deg")
print(f"Predicted apogee with final command: {apogee_hold:.3f} m")
print(f"Total MPC runtime: {end_time - start_time:.3f} s")

# # -----------------------------
# # Graph: best deployment fraction + predicted apogee vs altitude
# # -----------------------------
# best_deployments = []
# predicted_apogees = []

# # Example list of altitudes and velocities (replace with your 20 points)
# altitude_list = [
#     2342.76, 2404.804, 2465.832, 2525.865, 2584.918, 2643.006, 2700.146, 2756.352,
#     2811.638, 2866.015, 2920.643, 2972.098, 3023.827, 3074.693, 3124.708, 3173.884,
#     3222.228, 3279.155, 3334.912, 3389.515
# ]

# velocity_list = [
#     250.228, 246.13, 242.11, 238.162, 234.271, 230.446, 226.685, 222.975,
#     219.317, 215.711, 212.159, 208.651, 205.183, 201.756, 198.374, 195.035,
#     191.726, 187.797, 183.924, 180.104
# ]

# for alt_test, vel_test in zip(altitude_list, velocity_list):
#     # Find best deployment fraction
#     seqs = generate_sequences(0.0, deployment_fractions, horizon, K_options, max_change)
#     best_seq = None
#     best_cost = float('inf')
#     best_apogee = None

#     for seq in seqs:
#         schedule = make_CdA_schedule_from_sequence(0.0, seq, dt_control)
#         apogee_seq = simulate_until_apogee(alt_test, vel_test, pitch, mass, CdA_r, schedule)
#         cost = abs(apogee_seq - target_apogee)
#         if cost < best_cost:
#             best_cost = cost
#             best_seq = seq
#             best_apogee = apogee_seq

#     best_deployments.append(best_seq[0])
#     predicted_apogees.append(best_apogee)

# # Plot with secondary axis
# fig, ax1 = plt.subplots()

# color1 = 'tab:blue'
# ax1.set_xlabel('Initial Altitude (m)')
# ax1.set_ylabel('Best Deployment Fraction', color=color1)
# ax1.plot(altitude_list, best_deployments, '-o', color=color1, label='Deployment Fraction')
# ax1.tick_params(axis='y', labelcolor=color1)
# ax1.grid(True)

# ax2 = ax1.twinx()  # secondary y-axis
# color2 = 'tab:red'
# ax2.set_ylabel('Predicted Apogee (m)', color=color2)
# ax2.plot(altitude_list, predicted_apogees, '-s', color=color2, label='Predicted Apogee')
# ax2.tick_params(axis='y', labelcolor=color2)

# fig.tight_layout()
# plt.title('Best Airbrake Deployment & Predicted Apogee vs Altitude')
# plt.show()

# -----------------------------
# Best sequence vs predicted apogee with dual y-axis
# -----------------------------
alt_test = altitude
vel_test = velocity

# # Find best sequence
# seqs = generate_sequences(current_fraction, deployment_fractions, horizon, K_options, max_change)
# best_seq = None
# best_cost = float('inf')
# best_apogee = None

# for seq in seqs:
#     schedule = make_CdA_schedule_from_sequence(current_fraction, seq, dt_control)
#     apogee_seq = simulate_until_apogee(alt_test, vel_test, pitch, mass, CdA_r, schedule)
#     cost = abs(apogee_seq - target_apogee)
#     if cost < best_cost:
#         best_cost = cost
#         best_seq = seq
#         best_apogee = apogee_seq

# Simulate predicted apogee at each step of sequence
apogees_during_sequence = []
print(best_seq)
for i in range(len(best_seq)):
    partial_schedule = make_CdA_schedule_from_sequence(current_fraction, best_seq[:i+1], dt_control)
    apogee_partial = simulate_until_apogee(alt_test, vel_test, pitch, mass, CdA_r, partial_schedule)
    apogees_during_sequence.append(apogee_partial)

# Plot with dual y-axis
fig, ax1 = plt.subplots()

color1 = 'tab:blue'
ax1.set_xlabel('Step in Deployment Sequence')
ax1.set_ylabel('Airbrake Deployment Fraction', color=color1)
ax1.plot(range(1, len(best_seq)+1), best_seq, '-o', color=color1, label='Deployment Fraction')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, 1)
ax1.grid(True)

ax2 = ax1.twinx()  # secondary y-axis for apogee
color2 = 'tab:red'
ax2.set_ylabel('Predicted Apogee (m)', color=color2)
ax2.plot(range(1, len(best_seq)+1), apogees_during_sequence, '-s', color=color2, label='Predicted Apogee')
ax2.tick_params(axis='y', labelcolor=color2)

fig.tight_layout()
plt.title('Best Airbrake Deployment Sequence vs Predicted Apogee')
plt.show()
