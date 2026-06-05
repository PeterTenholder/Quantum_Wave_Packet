import numpy as np
from config import *

def calculate_A(width):
    return (1 / (np.pi * 2 * width**2)) ** 0.25

def create_wave_packet(x, t=0):
    return calculate_A(INITIAL_WIDTH) * np.exp(-(x - INITIAL_POSITION)**2 / (4 * INITIAL_WIDTH**2)) * np.exp(1j * K_O * x)

def calculate_evolution(x, t, width, group_velocity):
    A = calculate_A(width)
    return A * (1 / (np.sqrt(1+1j * (HBAR * t / (2 * MASS * INITIAL_WIDTH**2))))) * np.exp(-1 * (x -INITIAL_POSITION - group_velocity * t)**2 / (4 * width**2) * ( 1 - 1j * (HBAR * t / (2 * MASS * INITIAL_WIDTH**2))) + 1j *K_O * x - 1j * (HBAR * K_O**2 * t / (2 * MASS))) 

def normalize_wave_packet(wave_packet, dx):
    norm = np.sum(np.abs(wave_packet)**2) * dx
    return wave_packet / np.sqrt(norm)

def calculate_theoretical_transmission(B0, E):
    kappa = np.sqrt(2 * MASS * (B0 - E)) / HBAR
    return (1 + (B0**2 / (4 * E * (B0 - E))) * np.sinh(kappa * V_WIDTH)**2) ** (-1)

def calculate_most_probable_energy():
    # The most probable energy of the initial wave packet 
    # no one single energy, but can find mean energy from the momentum distribution of the initial wave packet
    return (HBAR * K_O)**2 / (2 * MASS)

def calculate_percent_error(simulated, theoretical):
    return abs(simulated - theoretical) / theoretical * 100

def logspace(lo_exp, hi_exp, n):  # 10^lo to 10^hi, n points
    return [10**(lo_exp + (hi_exp-lo_exp)*i/(n-1)) for i in range(n)]