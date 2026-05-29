import numpy as np

num_points = 64  # Number of grid points along each side
box_size = 10.0  # Size of the physical box

x_axis = np.linspace(-box_size / 2, box_size / 2, num_points, endpoint=False)
y_axis = x_axis
z_axis = x_axis
X, Y, Z = np.meshgrid(x_axis, y_axis, z_axis, indexing='ij')

dx = x_axis[1] - x_axis[0]

px = np.fft.fftfreq(num_points, d=dx) * 2 *np.pi
py = px
pz = px
PX, PY, PZ = np.meshgrid(px, py, pz, indexing='ij')

dpx = px[1] - px[0]



print("Spatial Grid Shape: ", X.shape)
print("Momentumn Grid Shape: ", PX.shape)