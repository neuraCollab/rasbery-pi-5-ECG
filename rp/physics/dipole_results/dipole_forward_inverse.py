"""
dipole_forward_inverse.py

Квазистационарная дипольная модель + обратная задача (Tikhonov) для
реалистичных записей ЭКГ в формате *_filtered.npy (shape (T,6)).

Как использовать:
    python dipole_forward_inverse.py

Результаты будут в папке recordings/dipole_results/
(графики, summary.txt, numpy-данные).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from numpy.linalg import inv

# Параметры (можешь подправить)
RECORDINGS_DIR = "recordings"
OUT_DIR = os.path.join(RECORDINGS_DIR, "dipole_results")
os.makedirs(OUT_DIR, exist_ok=True)

# Электрофизические параметры
SIGMA = 0.2  # проводимость ткани, S/m (примерное значение)
FOURIER_FACTOR = 1.0 / (4.0 * np.pi * SIGMA)

# Геометрия: положение источника и 6 виртуальных электродов (в метрах).
# Простая модель: центр сердца в (0,0,0), электроды на полусфере вокруг грудной клетки.
R_SOURCE = np.array([0.0, 0.0, 0.0])  # сердце в центре координат
ELECTRODE_RADIUS = 0.18  # ~18 cm от центра (пример)
# 6 позиций примерно соответствуют проекциям для шести отведений фронтов
ELECTRODE_POSITIONS = np.array([
    [ 0.0,  ELECTRODE_RADIUS, 0.0],   # передняя средняя верх
    [ ELECTRODE_RADIUS*0.6, -ELECTRODE_RADIUS*0.4, 0.0],
    [-ELECTRODE_RADIUS*0.6, -ELECTRODE_RADIUS*0.4, 0.0],
    [ ELECTRODE_RADIUS*0.9, 0.0,  ELECTRODE_RADIUS*0.1],
    [-ELECTRODE_RADIUS*0.9, 0.0,  ELECTRODE_RADIUS*0.1],
    [ 0.0, -ELECTRODE_RADIUS, 0.0]
])

def build_forward_matrix(elec_pos, r0=R_SOURCE, sigma=SIGMA):
    """
    Строит матрицу G (M x 3), такую что V = G @ p,
    где p = (p_x, p_y, p_z) -- дипольный момент.
    """
    M = elec_pos.shape[0]
    G = np.zeros((M, 3), dtype=float)
    for j in range(M):
        r = elec_pos[j]
        r_vec = r - r0
        r_norm = np.linalg.norm(r_vec)
        if r_norm < 1e-6:
            raise ValueError("Electrode coincides with source position")
        G[j, :] = FOURIER_FACTOR * (r_vec) / (r_norm**3)
    return G  # shape (M,3)

def tikhonov_inversion(G, Vt, lam=1e-6):
    """
    Оценка p(t) по V(t) через Tikhonov-регуляризацию (решение для каждого момента).
    Vt: shape (M, T)
    Возвращает p(t): shape (3, T)
    """
    M, T = Vt.shape
    # матричный предрасчёт
    GT = G.T  # (3,M)
    GTG = GT @ G  # (3,3)
    A = GTG + lam * np.eye(3)
    Ainv = inv(A)
    # p(t) = Ainv @ GT @ V(:,t)
    Pt = Ainv @ GT @ Vt
    return Pt  # (3, T)

def load_latest_filtered(recordings_dir=RECORDINGS_DIR):
    files = sorted([f for f in os.listdir(recordings_dir) if f.endswith('_filtered.npy')])
    if not files:
        raise FileNotFoundError("Нет файлов *_filtered.npy в recordings/")
    path = os.path.join(recordings_dir, files[-1])
    data = np.load(path)  # (T, 6)
    return data, files[-1]

def compute_metrics(V, Vrec):
    """
    RMSE по всем каналам и R^2 объяснённая дисперсия.
    V, Vrec: shape (M, T)
    """
    mse = np.mean((V - Vrec)**2)
    rmse = np.sqrt(mse)
    var = np.var(V)
    r2 = 1.0 - mse / (var + 1e-12)
    return rmse, r2

def plot_time_series(t, V_orig, V_rec, p, out_dir=OUT_DIR, base="result"):
    # V_orig / V_rec shape (M,T), p shape (3,T)
    M, T = V_orig.shape
    plt.figure(figsize=(10, 6))
    for i in range(M):
        plt.subplot(M+1, 1, i+1)
        plt.plot(t, V_orig[i, :], label=f"V{i+1}")
        plt.plot(t, V_rec[i, :], linestyle='--', label=f"V{i+1}_rec", alpha=0.8)
        if i == 0:
            plt.legend(loc='upper right', fontsize=8)
        plt.ylabel("u.a.")
    # plot dipole components
    plt.subplot(M+1, 1, M+1)
    plt.plot(t, p[0,:], label="p_x")
    plt.plot(t, p[1,:], label="p_y")
    plt.plot(t, p[2,:], label="p_z")
    plt.legend(loc='upper right', fontsize=8)
    plt.ylabel("dipole")
    plt.xlabel("time (s)")
    plt.tight_layout()
    out = os.path.join(out_dir, f"{base}_timeseries.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print("Saved time series plot:", out)

def plot_spectra(p, fs, out_dir=OUT_DIR, base="result"):
    # p shape (3, T)
    fig, axs = plt.subplots(3, 1, figsize=(8,8))
    for i in range(3):
        f, Pxx = welch(p[i,:], fs=fs, nperseg=512)
        axs[i].semilogy(f, Pxx)
        axs[i].set_xlim(0, 40)
        axs[i].set_ylabel(f"Pxx p_{i}")
    axs[-1].set_xlabel("Hz")
    plt.tight_layout()
    out = os.path.join(out_dir, f"{base}_dipole_spectra.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print("Saved dipole spectra:", out)

def main():
    # загрузка
    data, fname = load_latest_filtered()
    # data shape (T, 6) -- переупорядочим в (6, T)
    V = data.T.astype(float)
    T = V.shape[1]
    # предположение о дискретизации — если известно FS в проекте — 100 Hz
    FS = 100.0
    t = np.arange(T) / FS

    # Центрируем данные (убираем DC)
    V = V - V.mean(axis=1, keepdims=True)

    # Построим матрицу G для наших 6 электродов
    G = build_forward_matrix(ELECTRODE_POSITIONS, r0=R_SOURCE, sigma=SIGMA)  # (6,3)

    # Оценка диполя по всем моментам времени
    lam = 1e-3  # регуляризация, можно варьировать
    P_est = tikhonov_inversion(G, V, lam=lam)  # (3, T)

    # Реконструкция измерений
    V_rec = G @ P_est  # (6, T)

    # Метрики
    rmse, r2 = compute_metrics(V, V_rec)
    print(f"File: {fname}  RMSE={rmse:.6f}  R^2={r2:.4f}")

    # Сохранение numpy
    np.save(os.path.join(OUT_DIR, "G_matrix.npy"), G)
    np.save(os.path.join(OUT_DIR, "p_est.npy"), P_est)
    np.save(os.path.join(OUT_DIR, "V_rec.npy"), V_rec)

    # Плоты
    plot_time_series(t, V, V_rec, P_est, out_dir=OUT_DIR, base="dipole")
    plot_spectra(P_est, fs=FS, out_dir=OUT_DIR, base="dipole")

    # Сводный файл
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(f"source_file: {fname}\n")
        f.write(f"electrode_positions:\n{ELECTRODE_POSITIONS}\n")
        f.write(f"regularization_lambda: {lam}\n")
        f.write(f"RMSE: {rmse:.6e}\n")
        f.write(f"R2: {r2:.6f}\n")
    print("Saved summary to", os.path.join(OUT_DIR, "summary.txt"))

if __name__ == "__main__":
    main()
