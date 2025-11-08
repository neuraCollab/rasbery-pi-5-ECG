"""
visualize_dipole_3d.py

Анимация вектора диполя, восстановленного из ЭКГ.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from matplotlib.animation import FuncAnimation

OUT_DIR = "recordings/dipole_results"
P_PATH = os.path.join(OUT_DIR, "p_est.npy")
SAVE_MP4 = False

p = np.load(P_PATH)  # shape (3, T)
T = p.shape[1]
FS = 100
t = np.arange(T) / FS

# нормализация масштаба
p /= np.max(np.linalg.norm(p, axis=0))

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Вектор диполя сердца (время)")

vec, = ax.plot([0, p[0,0]], [0, p[1,0]], [0, p[2,0]], color="red", lw=3)
trail, = ax.plot([], [], [], color="gray", lw=1, alpha=0.5)

def update(frame):
    vec.set_data([0, p[0,frame]], [0, p[1,frame]])
    vec.set_3d_properties([0, p[2,frame]])
    trail.set_data(p[0,:frame], p[1,:frame])
    trail.set_3d_properties(p[2,:frame])
    ax.set_title(f"Вектор диполя сердца — t={t[frame]:.2f}s")
    return vec, trail

ani = FuncAnimation(fig, update, frames=T, interval=30, blit=False)

if SAVE_MP4:
    out_path = os.path.join(OUT_DIR, "dipole_animation.mp4")
    ani.save(out_path, writer="ffmpeg", fps=30, dpi=150)
else:
    out_path = os.path.join(OUT_DIR, "dipole_animation.gif")
    ani.save(out_path, writer="pillow", fps=30)
    print(f"✅ Анимация сохранена: {out_path}")

