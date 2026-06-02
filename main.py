import numpy as np
from config import *
from calculations import *
import matplotlib.pyplot as plt

num_points = NUM_POINTS
box_size = BOX_SIZE

x_axis = np.linspace(-box_size / 2, box_size / 2, num_points, endpoint=False)
#y_axis = x_axis
#z_axis = x_axis

dx = x_axis[1] - x_axis[0]
#dy = y_axis[1] - y_axis[0]
#dz = z_axis[1] - z_axis[0]

px = np.fft.fftfreq(num_points, d=dx) * 2 *np.pi # 1d array
py = px
pz = px
#PX, PY, PZ = np.meshgrid(px, py, pz, indexing='ij') #shapes to 3d

dpx = px[1] - px[0]



print("Spatial Grid Shape: ", x_axis.shape)
print("Momentumn Grid Shape: ", px.shape)

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
strength = 3e-22
absorb[:m]  = strength * ramp[::-1]
absorb[-m:] = strength * ramp

kick = np.exp(-1j * (B - 1j*absorb) * dt / (2*HBAR))   # imaginary part causes decay
drift = np.exp(-1j * px**2 * dt / (2*MASS*HBAR))



for step in range(int(1.4e4)):

    # Should be ~1 
    if step % 1000 == 0:
        norm = np.sum(np.abs(current_packet)**2) * dx
        print(f"step {step:>6}   norm = {norm:.6f}")

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
print(f"Transmission: {T}")

E0 = calculate_most_probable_energy()

print(f"Theoretical Transmission: {calculate_theoretical_transmission(V0, E0)}")

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
plt.show()