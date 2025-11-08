# src/analysis/hamiltonian.py
import numpy as np
import matplotlib.pyplot as plt
import os
import params

def calculate_hamiltonian(p, fs=params.FS, m=1.0, k=10.0):
    """
    Вычисляет Гамильтониан (полную энергию) для 3D-диполя.
    
    Аргументы:
        p (np.ndarray): Диполь с формой (3, T).
        fs (float): Частота дискретизации.
        m (float): Масса (параметр модели осциллятора).
        k (float): Жёсткость (параметр модели осциллятора).
        
    Возвращает:
        tuple: (время, полная_энергия, кинетическая_энергия, потенциальная_энергия)
    """
    T = p.shape[1]
    t = np.arange(T) / fs

    # Производная (скорость)
    dp = np.gradient(p, axis=1) * fs

    # Энергии
    T_energy = 0.5 * (1/m) * np.sum(dp**2, axis=0)  # Кинетическая
    U_energy = 0.5 * k * np.sum(p**2, axis=0)        # Потенциальная
    H = T_energy + U_energy                           # Полная

    return t, H, T_energy, U_energy

def analyze_hamiltonian(dipole_path):
    """
    Основная функция анализа. Сравнивает восстановленный диполь с истинным.
    dipole_path в данном контексте — это путь к p_est.npy, 
    но мы сами найдём p_true.npy в той же директории.
    """
    from pathlib import Path

    # 1. Загрузка данных
    p_est = np.load(dipole_path)
    # Находим путь к p_true.npy
    dipole_path_obj = Path(dipole_path)
    p_true_path = dipole_path_obj.parent / dipole_path_obj.name.replace("_p_est.npy", "_p_true.npy")
    p_true = np.load(p_true_path)

    print(f"Загружены диполи: p_true {p_true.shape}, p_est {p_est.shape}")

    # 2. Вычисление ошибки
    mse = np.mean((p_true - p_est)**2)
    mae = np.mean(np.abs(p_true - p_est))
    print(f"Метрики ошибки восстановления:")
    print(f"  Среднеквадратичная ошибка (MSE): {mse:.2e}")
    print(f"  Средняя абсолютная ошибка (MAE):  {mae:.2e}")

    # 3. Расчёт Гамильтонианов
    t, H_true, T_true, U_true = calculate_hamiltonian(p_true)
    _, H_est, T_est, U_est = calculate_hamiltonian(p_est)

    # 4. Визуализация
    plt.figure(figsize=(12, 5))

    # График полной энергии
    plt.subplot(1, 2, 1)
    plt.plot(t, H_true, 'g-', label='H_true (Истина)', linewidth=2)
    plt.plot(t, H_est, 'r--', label='H_est (Оценка)', linewidth=2)
    plt.xlabel("Время (с)")
    plt.ylabel("Полная энергия H")
    plt.title("Сравнение полной энергии")
    plt.legend()
    plt.grid(True)

    # График ошибки во времени
    plt.subplot(1, 2, 2)
    error_norm = np.linalg.norm(p_true - p_est, axis=0)
    plt.plot(t, error_norm, 'b-', linewidth=1)
    plt.xlabel("Время (с)")
    plt.ylabel("Норма ошибки ||p_true - p_est||")
    plt.title("Ошибка восстановления во времени")
    plt.grid(True)

    plt.tight_layout()
    out_path = os.path.join(params.RESULTS_DIR, "hamiltonian_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ График сравнения сохранён: {out_path}")

    # 5. Сохранение метрик в файл 
    summary_path = os.path.join(params.RESULTS_DIR, "analysis_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f: # <-- Добавлено encoding='utf-8'
        f.write("=== Анализ качества восстановления диполя ===\n")
        f.write(f"Среднеквадратичная ошибка (MSE): {mse:.6e}\n")
        f.write(f"Средняя абсолютная ошибка (MAE):  {mae:.6e}\n")
        f.write(f"Средняя полная энергия (истина): {np.mean(H_true):.6e}\n")
        f.write(f"Средняя полная энергия (оценка): {np.mean(H_est):.6e}\n")

    print(f"✅ Отчёт сохранён: {summary_path}")
    return summary_path