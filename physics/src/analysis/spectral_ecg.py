# src/analysis/spectral_ecg.py
"""
Модуль для спектрального анализа сгенерированного ЭКГ-сигнала.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, spectrogram
import os
import params

def analyze_ecg_spectrum(ecg_filtered_path):
    """
    Выполняет спектральный анализ ЭКГ.
    
    Аргументы:
        ecg_filtered_path (str): Путь к файлу *_filtered.npy.
    """
    # 1. Загрузка данных
    ecg = np.load(ecg_filtered_path) # shape (6, T)
    # Для анализа возьмём первое отведение как пример
    signal = ecg[0, :]
    fs = params.FS

    # 2. Анализ с помощью Welch (PSD)
    f_welch, Pxx = welch(signal, fs=fs, nperseg=256, noverlap=128, scaling='density')

    # 3. Построение спектрограммы
    f_spec, t_spec, Sxx = spectrogram(signal, fs=fs, nperseg=128, noverlap=120, scaling='spectrum')

    # 4. Визуализация
    plt.figure(figsize=(14, 10))

    # График исходного сигнала
    plt.subplot(3, 1, 1)
    T = len(signal)
    t = np.arange(T) / fs
    plt.plot(t, signal, linewidth=0.8)
    plt.title("Отведение I: Синтетический ЭКГ-сигнал")
    plt.xlabel("Время (с)")
    plt.ylabel("Амплитуда")
    plt.grid(True)

    # График PSD (Welch)
    plt.subplot(3, 1, 2)
    plt.semilogy(f_welch, Pxx, 'b')
    plt.title("Спектральная плотность мощности (Welch)")
    plt.xlabel("Частота (Гц)")
    plt.ylabel("PSD (В²/Гц)")
    plt.xlim(0, 40)
    plt.grid(True)

    # Спектрограмма
    plt.subplot(3, 1, 3)
    im = plt.pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx + 1e-12), shading='gouraud', cmap='viridis')
    plt.title("Спектрограмма")
    plt.ylabel("Частота (Гц)")
    plt.xlabel("Время (с)")
    plt.ylim(0, 40)
    plt.colorbar(im, label='Мощность (дБ)')
    plt.grid(True)

    plt.tight_layout()
    out_path = os.path.join(params.RESULTS_DIR, "ecg_spectral_analysis.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"✅ Спектральный анализ ЭКГ сохранён: {out_path}")
    return out_path