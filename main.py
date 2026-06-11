import numpy as np
from config import *
from calculations import *
import matplotlib.pyplot as plt



def run_simulation(strength=STRENGTH, steps=NUMBER_OF_STEPS, V0=V0, width=INITIAL_WIDTH, initial_position=INITIAL_POSITION, num_points=NUM_POINTS, box_size=BOX_SIZE):

    x_axis = np.linspace(-box_size / 2, box_size / 2, num_points, endpoint=False)
    dx = x_axis[1] - x_axis[0]
    px = np.fft.fftfreq(num_points, d=dx) * 2 *np.pi # 1d array


    inital_packet = create_wave_packet(x_axis) # maybe just x_axis
    current_packet = normalize_wave_packet(inital_packet, dx)


    B = np.zeros(num_points)
    B[(x_axis >= V_START) & (x_axis <= V_END)] = V0
    # Set the potential barrier height for the masked region
    # The barrier is defined as a rectangular region in the x-axis between V_START and V_END, and it extends infinitely in the y and z directions.

    # Absorbing boundary conditions: add an imaginary potential that increases towards the edges of the box to prevent reflections
    #does not affect the results since the wave packet should not reach the edges of the box, but just in case it does, we want to prevent reflections from the boundaries that would interfere with our results.
    absorb = np.zeros(num_points)
    m = num_points // 8
    ramp = np.linspace(0, 1, m)**2

    absorb[:m]  = strength * ramp[::-1]
    absorb[-m:] = strength * ramp

    kick = calculate_kick(absorb, B)
    drift = calculate_drift(px)


    for step in range(int(steps)):
        current_packet = split_step(kick, drift, current_packet)

    # past barrier but before the right absorber
    right_absorber_start = x_axis[-m]
    window = (x_axis > V_END) & (x_axis < right_absorber_start)

    T = np.sum(np.abs(current_packet[window])**2) * dx    
    E0 = calculate_most_probable_energy()
    kappa = np.sqrt(2 * MASS * (V0 - E0)) / HBAR
    kappa_L = kappa * V_WIDTH
    T_theory_avg = calculate_energy_averaged_transmission(inital_packet, px, V0)
    percent_error = calculate_percent_error(T, T_theory_avg)

    return x_axis, current_packet, T, E0, kappa, kappa_L, percent_error, T_theory_avg

best_params = np.zeros(4)        # strength, steps, num_points, box_size
current_percent_error = np.inf   # start high so first run always wins


strengths = logspace(-22, -16, 10)
steps = [int(round(x)) for x in logspace(3, 5, 10)]  
num_points = [256, 512, 1024, 2048, 4096]
box_sizes = np.array([40e-9, 60e-9, 80e-9, 100e-9, 120e-9]) 


total = len(strengths) * len(steps) * len(num_points) * len(box_sizes) # total number of simulations to run
count = 0

# Sweep over parameters to find the best combination that minimizes the percent error - brute-force approach
for s in strengths:
    for k in steps:
        for n in num_points:
            for bx in box_sizes:

                count += 1
                print(f"\r{count}/{total} ({100*count/total:.2f}%)", end="", flush=True)

                x_axis, current_packet, T, E0, kappa, kappa_L, percent_error, T_theory_avg = run_simulation(strength=s, steps=k, num_points=n, box_size=bx)


                if percent_error < current_percent_error:
                    current_percent_error = percent_error
                    print(f"\nnew best: strength={s:.2e} steps={k} N={n} box={bx:.2e} err={percent_error:.2f}%")

                    best_params = np.array([s, k, n, bx])

s, k, n, bx = best_params
print(f"\nBest: strength={s:.2e} steps={k} N={n} box={bx:.2e}  error={current_percent_error:.2f}%")

x_axis, current_packet, T, E0, kappa, kappa_L, percent_error, T_theory_avg = run_simulation(strength=3e-21, steps=70000, num_points=4096, box_size=200e-9)
print(f"simulated:   {T:.4f}")
print(f"avg theory:  {T_theory_avg:.4f}")
print(f"single-E:    {calculate_theoretical_transmission(V0, calculate_most_probable_energy()):.4f}")
print(f"error:       {percent_error:.1f}%")





"""print(f"Theoretical Transmission: {calculate_theoretical_transmission(V0, E0)}")

print(f"Percent Error: {abs(T - calculate_theoretical_transmission(V0, E0)) / calculate_theoretical_transmission(V0, E0) * 100:.2f}%")

print(f"Most Probable Energy E0: {E0:.2e} J")
print(f"Barrier Height V0: {V0:.2e} J")
print(f"Barrier Width: {V_WIDTH:.2e} m")
print(f"kappa: {np.sqrt(2 * MASS * (V0 - E0)) / HBAR:.2e} 1/m")
print(f"kappa * L: {np.sqrt(2 * MASS * (V0 - E0)) / HBAR * V_WIDTH:.2f}")




fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x_axis, np.abs(current_packet)**2, label='|ψ|²')
ax.axvspan(V_START, V_END, alpha=0.2, color='red', label='barrier')
ax.set_xlabel("x position")
ax.set_ylabel("probability density")
ax.legend()
plt.savefig("wave_packet_evolution.png")
plt.show()"""