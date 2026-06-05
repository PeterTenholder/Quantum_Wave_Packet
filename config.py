import numpy as np

dt = 1e-18  # Time step for evolution

NUMBER_OF_STEPS = 1.4e4

INITIAL_WIDTH = 1e-9
H= 6.62607015e-34
HBAR = 1.0545718e-34
K_O = 1e10
MASS = 9.10938356e-31 #electron mass
INITIAL_POSITION = -12e-9 # initial position of the wave packet, should be negative to start on the left side of the box

potent = 0.5 * MASS * (K_O * HBAR / MASS)**2 # potential energy at the center of the wave packet, unsure 

V0 = 6.3e-19 # height of the potential barrier
V_WIDTH = 1e-9 # width of the potential barrier
V_START = 0 # start position of the potential barrier
V_END = V_START + V_WIDTH # end position of the potential barrier

NUM_POINTS = 4096  # Number of grid points along each side
BOX_SIZE = 40e-9  # Size of the physical box

STRENGTH = 1e-18 # strength of the absorbing boundary, should be small enough to prevent significant reflections while still effectively absorbing outgoing waves at the boundaries. The strength is scaled by HBAR and dx in the main code to ensure it has the correct units and magnitude for the simulation.