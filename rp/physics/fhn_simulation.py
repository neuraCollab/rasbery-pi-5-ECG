import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd


def fitzhugh_nagumo(t, y, a=0.7, b=0.8, tau=12.5, I=0.5):
    v, w = y
    dvdt = v - (v ** 3) / 3 - w + I
    dwdt = (v + a - b * w) / tau
    return [dvdt, dwdt]


def generate_data(duration=50, fs=1000, noise_std=0.02):
    t = np.linspace(0, duration, int(fs * duration))
    y0 = [-1, 1]  # начальные условия
    sol = solve_ivp(fitzhugh_nagumo, [0, duration], y0, t_eval=t)

    v = sol.y[0]
    # добавим шум для реалистичности
    noisy_v = v + np.random.normal(0, noise_std, len(v))

    # сохраняем данные
    df = pd.DataFrame({"t": t, "v": noisy_v})
    df.to_csv("synthetic_ecg.csv", index=False)

    # график
    plt.figure(figsize=(10, 4))
    plt.plot(t, noisy_v, label="Simulated ECG-like signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (a.u.)")
    plt.title("FitzHugh–Nagumo Model Simulation")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fhn_simulation.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    generate_data()
