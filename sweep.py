from main import run_simulation
from config import *
from calculations import * 
import matplotlib.pyplot as plt


# --- fixed baseline (the "good enough" defaults to hold constant) ---
BASE_STRENGTH = 3e-21
BASE_BOX   = 250e-9
BASE_STEPS = 70000
BASE_N        = 2048
V0_FIXED      = 6.3e-19   # one physics value to validate against

theory = calculate_theoretical_transmission(V0_FIXED, calculate_most_probable_energy())
print(f"Theoretical T = {theory:.6e}\n")

print("=== sweep N (resolution) ===")
for n in [512, 1024, 2048, 4096, 8192]:
    *_, T, _, _, _, err = run_simulation(
        strength=BASE_STRENGTH, steps=BASE_STEPS, num_points=n, box_size=BASE_BOX, V0=V0_FIXED)
    print(f"N={n:>5}   T={T:.6e}   err={err:.2f}%")
print()

print("=== sweep box_size ===")
for bx in [150e-9, 200e-9, 250e-9, 300e-9, 400e-9]:
    *_, T, _, _, _, err = run_simulation(
        strength=BASE_STRENGTH, steps=BASE_STEPS, num_points=BASE_N, box_size=bx, V0=V0_FIXED)
    print(f"box={bx:.2e}   T={T:.6e}   err={err:.2f}%")
print()

print("=== sweep steps (timing) ===")
for k in [40000, 55000, 70000, 85000, 100000]:
    *_, T, _, _, _, err = run_simulation(
        strength=BASE_STRENGTH, steps=k, num_points=BASE_N, box_size=BASE_BOX, V0=V0_FIXED)
    print(f"steps={k:>6}   T={T:.6e}   err={err:.2f}%")
print()

print("=== sweep strength (absorber) ===")
for s in np.logspace(-22, -19, 7):
    *_, T, _, _, _, err = run_simulation(
        strength=s, steps=BASE_STEPS, num_points=BASE_N, box_size=BASE_BOX, V0=V0_FIXED)
    print(f"strength={s:.2e}   T={T:.6e}   err={err:.2f}%")
print()

Ns, Ts = [], []
for n in [512, 1024, 2048, 4096, 8192]:
    *_, T, _, _, _, err, Tavg = run_simulation(strength=BASE_STRENGTH, steps=BASE_STEPS,
                                          num_points=n, box_size=BASE_BOX, V0=V0_FIXED)
    Ns.append(n); Ts.append(T)

plt.figure()
plt.plot(Ns, Ts, 'o-', label='simulated T')
plt.axhline(theory, color='r', ls='--', label='theory')
plt.xscale('log'); plt.xlabel('num_points'); plt.ylabel('T')
plt.legend(); plt.title('Convergence in resolution')
plt.show()


