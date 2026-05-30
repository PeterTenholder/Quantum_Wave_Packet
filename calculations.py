import numpy as np
from config import *

def calculate_A(width):
    return (1 / (np.pi * 2 * width**2)) ** 0.25

def create_wave_packet(x, t=0):
    return calculate_A(INITIAL_WIDTH) * ((-1 * (x-INITIAL_POSITION)**2) / (4 * INITIAL_WIDTH**2)) * np.exp(1j * K_O * x)

def calculate_evolution(x, t, width, group_velocity):
    A = calculate_A(width)
    return A * (1 / (np.sqrt(1+1j * (HBAR * t / (2 * MASS * INITIAL_WIDTH**2))))) * np.exp(-1 * (x -INITIAL_POSITION - group_velocity * t)**2 / (4 * width**2) * ( 1 - 1j * (HBAR * t / (2 * MASS * INITIAL_WIDTH**2))) + 1j *K_O * x - 1j * (HBAR * K_O**2 * t / (2 * MASS))) 

def nomrmalize_wave_packet(wave_packet, x):
    norm = np.trapezoid(np.abs(wave_packet)**2, x)
    return wave_packet / np.sqrt(norm)

