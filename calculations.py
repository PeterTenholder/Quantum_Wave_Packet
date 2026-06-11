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
    #example: logspace(-22, -16, 10) gives 10 points from 1e-22 to 1e-16
    return [10**(lo_exp + (hi_exp-lo_exp)*i/(n-1)) for i in range(n)]

def calculate_energy_averaged_transmission(initial_packet, px, V0):
    # To get the energy-averaged transmission, we can decompose the initial wave packet into its momentum components (via FFT), calculate the transmission for each momentum (energy) component, 
    # and then take a weighted average based on the momentum distribution of the initial packet.
    phi = np.fft.fft(initial_packet)
    weight = np.abs(phi)**2
    fwd = px > 0
    E = (HBAR * px[fwd])**2 / (2 * MASS)
    w = weight[fwd]

    Tvals = np.zeros_like(E)
    # Calculate transmission for each energy component using the rectangular barrier formula
    #tvals is the same length as E, which is the number of forward-moving momentum components in the initial wave packet. 
    # Each component has a certain energy E[i] and a corresponding weight w[i] that represents how much of the initial wave packet's momentum distribution is concentrated at that energy.  
    for i, e in enumerate(E):
        if e <= 0:
            Tvals[i] = 0.0
        elif e < V0:
            kappa = np.sqrt(2*MASS*(V0 - e)) / HBAR
            arg = kappa * V_WIDTH 
            if arg > 30:                      # sinh overflows; T ~ 0
                Tvals[i] = 0.0
            else:
                Tvals[i] = 1.0 / (1 + (V0**2/(4*e*(V0-e))) * np.sinh(arg)**2)
        else:  # E > V0: over-barrier, partial reflection still occurs
            kprime = np.sqrt(2*MASS*(e - V0)) / HBAR
            Tvals[i] = 1.0 / (1 + (V0**2 / (4*e*(e - V0))) * np.sin(kprime * V_WIDTH)**2)

    denom = np.sum(w)
    if denom == 0:
        return np.nan
    
    # this helps because the initial wave packet is not monoenergetic, but has a distribution of energies. Each energy component contributes to the overall transmission according to its weight in the initial momentum distribution. 
    # By summing w[i] * Tvals[i], we get the total transmitted probability, and dividing by the sum of weights gives us the average transmission coefficient for the entire wave packet.
    return np.sum(w * Tvals) / denom

def calculate_kick(absorb, B):
    return np.exp(-1j * (B - 1j*absorb) * dt / (2*HBAR)) 

def calculate_drift(px):
    return np.exp(-1j * HBAR * px**2 * dt / (2*MASS))

def split_step(kick, drift, current_packet):
    # Should be ~1, checker
    '''if step % 1000 == 0:
        norm = np.sum(np.abs(current_packet)**2) * dx
        print(f"step {step:>6}   norm = {norm:.6f}")'''

    # First half kick: apply half the potential operator in position space
    psi = current_packet * kick

    # Kinetic drift in momentum space:
    #   FFT to momentum space -> apply kinetic operator -> inverse FFT back
    psi_momentum = np.fft.fft(psi)
    psi_drifted = np.fft.ifft(psi_momentum * drift)

    # Second half kick: apply the remaining half of the potential operator
    current_packet = psi_drifted * kick

    return current_packet