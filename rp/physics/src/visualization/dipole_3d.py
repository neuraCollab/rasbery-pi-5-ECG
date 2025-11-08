# src/visualization/dipole_3d.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os
from pathlib import Path
import params

def create_dipole_animation(dipole_path):
    """
    Создаёт анимацию, сравнивающую истинный и восстановленный диполи.
    """
    # 1. Загрузка данных
    p_est = np.load(dipole_path)
    dipole_path_obj = Path(dipole_path)
    p_true_path = dipole_path_obj.parent / dipole_path_obj.name.replace("_p_est.npy", "_p_true.npy")
    p_true = np.load(p_true_path)

    T = p_true.shape[1]
    fs = params.FS
    t = np.arange(T) / fs

    # 2. Нормализация для визуализации (чтобы оба вектора были в одном масштабе)
    all_data = np.concatenate([p_true, p_est], axis=1)
    scale = np.max(np.linalg.norm(all_data, axis=0)) * 1.1
    p_true_norm = p_true / scale
    p_est_norm = p_est / scale

    # 3. Настройка графика
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Сравнение диполей: Истина (зелёный) vs Оценка (красный)")

    # Истинный диполь
    true_vec, = ax.plot([0, p_true_norm[0,0]], [0, p_true_norm[1,0]], [0, p_true_norm[2,0]], 
                        color="green", lw=3, label="p_true")
    true_trail, = ax.plot([], [], [], color="green", lw=1, alpha=0.5)

    # Восстановленный диполь
    est_vec, = ax.plot([0, p_est_norm[0,0]], [0, p_est_norm[1,0]], [0, p_est_norm[2,0]], 
                       color="red", lw=3, linestyle='--', label="p_est")
    est_trail, = ax.plot([], [], [], color="red", lw=1, alpha=0.5)

    ax.legend()

    # 4. Функция обновления для анимации
    def update(frame):
        # Истинный диполь
        true_vec.set_data([0, p_true_norm[0,frame]], [0, p_true_norm[1,frame]])
        true_vec.set_3d_properties([0, p_true_norm[2,frame]])
        true_trail.set_data(p_true_norm[0,:frame], p_true_norm[1,:frame])
        true_trail.set_3d_properties(p_true_norm[2,:frame])
        
        # Восстановленный диполь
        est_vec.set_data([0, p_est_norm[0,frame]], [0, p_est_norm[1,frame]])
        est_vec.set_3d_properties([0, p_est_norm[2,frame]])
        est_trail.set_data(p_est_norm[0,:frame], p_est_norm[1,:frame])
        est_trail.set_3d_properties(p_est_norm[2,:frame])
        
        ax.set_title(f"Сравнение диполей — t={t[frame]:.2f}s\n(Зелёный: Истина, Красный: Оценка)")
        return true_vec, true_trail, est_vec, est_trail

    # 5. Создание и сохранение анимации
    ani = FuncAnimation(fig, update, frames=T, interval=30, blit=False, repeat=True)

    out_path = os.path.join(params.RESULTS_DIR, "dipole_comparison.gif")
    ani.save(out_path, writer="pillow", fps=20)
    plt.close(fig)
    print(f"✅ Анимация сравнения сохранена: {out_path}")
    return out_path