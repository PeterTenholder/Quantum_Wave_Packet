import numpy as np
from config import *
from calculations import *
import matplotlib.pyplot as plt



def run_simulation(strength=STRENGTH, steps=NUMBER_OF_STEPS, V0=V0, width=INITIAL_WIDTH, initial_position=INITIAL_POSITION, num_points=NUM_POINTS, box_size=BOX_SIZE):
    x_axis = np.linspace(-box_size / 2, box_size / 2, num_points, endpoint=False)

    dx = x_axis[1] - x_axis[0]

    px = np.fft.fftfreq(num_points, d=dx) * 2 *np.pi # 1d array
    py = px
    pz = px
    #PX, PY, PZ = np.meshgrid(px, py, pz, indexing='ij') #shapes to 3d

    dpx = px[1] - px[0]
    '''print("Spatial Grid Shape: ", x_axis.shape)
    print("Momentumn Grid Shape: ", px.shape)
    '''
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

    kick = np.exp(-1j * (B - 1j*absorb) * dt / (2*HBAR))   # imaginary part causes decay
    drift = np.exp(-1j * HBAR * px**2 * dt / (2*MASS))


    for step in range(int(steps)):

        # Should be ~1 
        if step % 1000 == 0:
            norm = np.sum(np.abs(current_packet)**2) * dx
            #print(f"step {step:>6}   norm = {norm:.6f}")

        # First half kick: apply half the potential operator in position space
        psi = current_packet * kick

        # Kinetic drift in momentum space:
        #   FFT to momentum space -> apply kinetic operator -> inverse FFT back
        psi_momentum = np.fft.fft(psi)
        psi_drifted = np.fft.ifft(psi_momentum * drift)

        # Second half kick: apply the remaining half of the potential operator
        current_packet = psi_drifted * kick
    #make potential barrier
    past = x_axis > V_END
    # clean region: past barrier but before the right absorber
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
steps = [int(round(x)) for x in logspace(3, 5, 10)]  # ints: 1000, 1668, 2783, ... 100000
num_points = [256, 512, 1024, 2048, 4096]
box_sizes = np.array([40e-9, 60e-9, 80e-9, 100e-9, 120e-9])                   # the lever that matters




# i=0  -> 10^-20.00 = 1.000e-20
# i=1  -> 10^-19.56 = 2.783e-20
# i=2  -> 10^-19.11 = 7.743e-20
# i=3  -> 10^-18.67 = 2.154e-19
# i=4  -> 10^-18.22 = 5.995e-19
# i=5  -> 10^-17.78 = 1.668e-18
# i=6  -> 10^-17.33 = 4.642e-18
# i=7  -> 10^-16.89 = 1.292e-17
# i=8  -> 10^-16.44 = 3.594e-17
# i=9  -> 10^-16.00 = 1.000e-16



total = len(strengths) * len(steps) * len(num_points) * len(box_sizes)
count = 0

for s in strengths:
    for k in steps:
        for n in num_points:
            for bx in box_sizes:
                count += 1
                print(f"\r{count}/{total} ({100*count/total:.2f}%)", end="", flush=True)




                #print(f"Running simulation with strength={i:.2e}, steps={k}, V0={l:.2e}")
                x_axis, current_packet, T, E0, kappa, kappa_L, percent_error, T_theory_avg = run_simulation(strength=s, steps=k, num_points=n, box_size=bx)


                if percent_error < current_percent_error:
                    current_percent_error = percent_error
                    print(f"\nnew best: strength={s:.2e} steps={k} N={n} box={bx:.2e} err={percent_error:.2f}%")

                    best_params = np.array([s, k, n, bx])

s, k, n, bx = best_params
print(f"\nBest: strength={s:.2e} steps={k} N={n} box={bx:.2e}  error={current_percent_error:.2f}%")


*_, T, _, _, _, err, Tavg = run_simulation(strength=3e-21, steps=70000, num_points=4096, box_size=200e-9)
print(f"simulated:   {T:.4f}")
print(f"avg theory:  {Tavg:.4f}")
print(f"single-E:    {calculate_theoretical_transmission(V0, calculate_most_probable_energy()):.4f}")
print(f"error:       {err:.1f}%")





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