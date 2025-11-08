# main.py
"""
Основной скрипт для запуска всего пайплайна.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_generation.mock_ecg import generate_mock_ecg
from src.dipole_model.inverse_solver import solve_dipole_inverse
from src.analysis.hamiltonian import analyze_hamiltonian
from src.analysis.spectral_ecg import analyze_ecg_spectrum # <-- Новый импорт
from src.visualization.dipole_3d import create_dipole_animation

if __name__ == "__main__":
    print("🚀 Запуск полного пайплайна...")
    
    # 1. Генерация моковых данных
    ecg_path = generate_mock_ecg(model='fhn')
    # Найдём путь к filtered-файлу для спектрального анализа
    ecg_filtered_path = ecg_path.replace("_raw.npy", "_filtered.npy")
    
    # 1b. Спектральный анализ ЭКГ
    analyze_ecg_spectrum(ecg_filtered_path) # <-- Новый шаг
    
    # 2. Решение обратной задачи
    dipole_path = solve_dipole_inverse(ecg_path)
    
    # 3. Анализ качества диполя
    analyze_hamiltonian(dipole_path)
    
    # 4. Визуализация диполя
    create_dipole_animation(dipole_path)
    
    print("✅ Пайплайн завершён успешно!")