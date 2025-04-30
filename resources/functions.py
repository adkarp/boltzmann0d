import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"

import time

from scipy.constants import e, k, m_e
from scipy.integrate import solve_ivp

constants = {
    "mstar": 0.26 * m_e,
    "q": e,
    "k": k,
    "T0": 300,
    "tau_c": 200e-15,
    "ni": 1.45e16,
}
E = np.logspace(3, 6, 4)

def f0(v, **constants):
    mstar = constants["mstar"]
    kB = constants["k"]
    T = constants["T0"]
    ni = constants["ni"]

    prefactor = np.sqrt(mstar / (2 * np.pi * kB * T))
    expfactor = np.exp(-mstar * np.abs(v) ** 2 / (2 * k * T))

    return ni * prefactor * expfactor

def df(v, f, E, **constants):
    mstar = constants["mstar"]
    tau_c = constants["tau_c"]
    q = constants["q"]

    alpha = mstar / (q * E * tau_c)

    return alpha * (f0(v, **constants) - f)

def evalBoltzmann(E, f0, dfdv, userinput=True, IC=None, plot=True, **constants):
    if userinput == True:
        N = int(input("Grid Size: "))
        Vmax = float(input("Max Velocity: "))
        Vmin = float(input("Min Velocity (mag): "))
    else:
        N = IC["N"]
        Vmax = IC["Vmax"]
        Vmin = IC["Vmin"]

    start_time = time.time()

    V = np.linspace(-Vmin, Vmax, N)

    cmap = plt.get_cmap("plasma", len(E))
    colors = cmap(np.arange(len(E)))

    if plot == True:
        plt.figure(figsize=(12, 6))

    solutions = []
    for Efield, color in zip(E, colors):

        f_init = f0(-Vmax, **constants)
        sol = solve_ivp(
            fun=lambda v, f: dfdv(v, f, E=Efield, **constants),
            t_span=(-Vmin, Vmax),
            y0=[f_init],
            t_eval=V,
            method="RK45",
        )
        v = sol.t
        f = sol.y[0]

        solutions.append({"E": Efield, "v": v, "f": f})
        if plot == True:
            plt.plot(v, f / np.max(f), label=f"E = {Efield}", color=color)

    if plot == True:
        plt.plot(
            v, f0(v, **constants) / np.max(f0(v, **constants)), "--", label=r"$f_0(v)$"
        )

        plt.xlabel("Velcoity (m/s)")
        plt.ylabel("Normalized $f(v)$")
        plt.legend()
        plt.title(
            f"RK4 Simulatiuon of the 0D Boltzmann Equation with an Applied E-Field ({N} Mesh Pts.)"
        )
        plt.show()

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Computation took {round(total_time,3)} seconds.")

    return solutions, total_time, f0(v, **constants)

def MonteCarloCollision(E, Np, **constants):
    kB = constants["k"]
    T0 = constants["T0"]
    q = constants["q"]
    mstar = constants["mstar"]
    tau_c = constants["tau_c"]

    vth = np.sqrt(kB * T0 / mstar)

    acceleration = q * E / mstar
    V0 = np.random.normal(0, vth, size=Np)
    dt = np.random.exponential(tau_c, size=Np)

    v_samp = V0 + acceleration * dt
    return v_samp

def evalBoltzmannMC(E, f0, userinput=True, IC=None, plot=True, **constants):
    # Inputs
    if userinput == True:
        N = int(input("Number of Monte Carlo samples: "))
        Nbins = int(input("Velocity grid size (histogram bins): "))
        Vmax = float(input("Max velocity for histogram (+): "))
        Vmin = float(input("Min velocity (mag) for histogram (-): "))
    else:
        N = IC["N"]
        Nbins = IC["Nbins"]
        Vmax = IC["Vmax"]
        Vmin = IC["Vmin"]

    start_time = time.time()

    # Bin Creation
    edges = np.linspace(-Vmin, Vmax, Nbins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    if plot == True:
        plt.figure(figsize=(12, 6))

    cmap = plt.get_cmap("plasma", len(E))
    colors = cmap(np.arange(len(E)))

    solutions = []
    # Iterating through all E-Fields
    for Efield, color in zip(E, colors):
        v_samples = MonteCarloCollision(Efield, N, **constants)

        hist, _ = np.histogram(v_samples, bins=edges, density=True)
        hist /= np.max(hist)

        if plot == True:
            plt.plot(centers, hist, label=f"E = {Efield} V/m", color=color)

        solutions.append({"E": Efield, "v_samples": v_samples, "hist": hist})
    
    f0_vals = f0(centers, **constants)
    
    if plot == True:
        plt.plot(centers, f0_vals / np.max(f0_vals), "--", label=r"$f_0(v)$")

        plt.xlabel("Velocity (m/s)")
        plt.ylabel("Normalized $f(v)$")
        plt.title(
            f"Monte Carlo Simulation of the 0D Boltzmann with an Applied E-Field ({N:.0E} Particles with {Nbins} Bins)"
        )
        plt.legend()
        plt.show()

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Computation took {round(total_time,3)} seconds.")

    return solutions, total_time, f0_vals