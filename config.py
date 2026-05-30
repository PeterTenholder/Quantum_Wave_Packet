import numpy as np

dt = 1e-18  # Time step for evolution

INITIAL_WIDTH = 1e-9
H= 6.62607015e-34
HBAR = 1.0545718e-34
K_O = 1e10
MASS = 9.10938356e-31 #electron mass
INITIAL_POSITION = 0

V = 0.5 * MASS * (K_O * HBAR / MASS)**2 # potential energy at the center of the wave packet, unsure 
