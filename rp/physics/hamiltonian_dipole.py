"""
hamiltonian_dipole.py

Гамильтонова интерпретация динамики электрического диполя.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

OUT_DIR = "recordings/dipole_results"
P_PATH = os.path.join(OUT_DIR, "p_est.npy")
os.makedirs(OUT_DIR, exist_ok=True)

p = np.load(P_PATH)  # shape (3, T)
FS = 100
t = np.arange(p.shape[1]) / FS

# производная (скорость изменения диполя)
dp = np.gradient(p, axis=1) * FS

# параметры модели осциллятора
m = 1.0
k = 10.0
omega0 = np.sqrt(k/m)

# энергия Гамильтона
T_energy = 0.5 * (1/m) * np.sum(dp**2, axis=0)
U_energy = 0.5 * k * np.sum(p**2, axis=0)
H = T_energy + U_energy

# === визуализация ===
fig, axs = plt.subplots(3, 2, figsize=(10, 8))
for i in range(3):
    axs[i,0].plot(t, p[i], label=f"p_{i}")
    axs[i,0].plot(t, dp[i], label=f"dp_{i}/dt", alpha=0.6)
    axs[i,0].legend(fontsize=8)
    axs[i,0].set_xlabel("Time (s)")
    axs[i,0].set_ylabel("Amplitude")

    axs[i,1].plot(p[i], dp[i])
    axs[i,1].set_xlabel(f"p_{i}")
    axs[i,1].set_ylabel(f"dp_{i}/dt")
    axs[i,1].set_title(f"Phase space: p_{i}")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "hamiltonian_phase.png"), dpi=150)

# энергия
plt.figure(figsize=(8,4))
plt.plot(t, H, label="H(t)")
plt.xlabel("Time (s)")
plt.ylabel("Hamiltonian Energy")
plt.title("Total energy of dipole oscillation")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "hamiltonian_energy.png"), dpi=150)

print(f"✅ Графики сохранены в {OUT_DIR}")
