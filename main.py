import numpy as np
from config import *
from calculations import *

num_points = 64  # Number of grid points along each side
box_size = 10.0  # Size of the physical box

x_axis = np.linspace(-box_size / 2, box_size / 2, num_points, endpoint=False)
y_axis = x_axis
z_axis = x_axis
X, Y, Z = np.meshgrid(x_axis, y_axis, z_axis, indexing='ij')

dx = x_axis[1] - x_axis[0]

px = np.fft.fftfreq(num_points, d=dx) * 2 *np.pi # 1d array
py = px
pz = px
PX, PY, PZ = np.meshgrid(px, py, pz, indexing='ij') #shapes to 3d

dpx = px[1] - px[0]



print("Spatial Grid Shape: ", X.shape)
print("Momentumn Grid Shape: ", PX.shape)

inital_packet = create_wave_packet(X) # maybe just X
normalized_packet = nomrmalize_wave_packet(inital_packet, x_axis)
for i in range(10):
    # first half kick 
    # mulitpoly position wave funciton by first half of potential energy operator V(x) 
    psi_new = normalized_packet * np.exp(-1 *1j * V * dt / (2 * HBAR))

    # transform wave function from position space into momentum space with fft
    phi_new = np.fft.fftn(psi_new)


    # kinetic drift
    # multiply kinetic energy operator in momentum space 
    phi_new_drifted = phi_new * np.exp(-1 *1j * HBAR * (PX**2 + PY**2 + PZ**2) * dt / (2 * MASS)) # k^2 = px^2 + py^2 + pz^2

    # reutn to position space with inverse fft

    psi_new_drifted = np.fft.ifftn(phi_new_drifted)

    # second half kick 
    # multoply by second half of potential energy operator V(x) to get final wave function at time t+dt
    psi_final = psi_new_drifted * np.exp(-1 *1j * V * dt / (2 * HBAR))

    print(np.trapezoid(np.abs(psi_final)**2, x_axis)) # should be 1, check normalization