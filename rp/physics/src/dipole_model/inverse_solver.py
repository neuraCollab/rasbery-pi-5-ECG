# src/dipole_model/inverse_solver.py
import numpy as np
import os
import params

def load_lead_matrix(n_leads=params.N_LEADS, n_dipole=params.N_DIPOLE, seed=42):
    """
    Загружает ИЛИ генерирует ту же самую матрицу отведений L,
    что и при генерации данных. Для воспроизводимости используем тот же seed.
    В реальном сценарии L была бы известна из калибровки или анатомии.
    """
    np.random.seed(seed)
    L = np.random.randn(n_leads, n_dipole)
    L = L / np.linalg.norm(L, axis=1, keepdims=True)
    return L

def solve_dipole_inverse(ecg_path):
    """
    Решает обратную задачу ЭКГ: восстанавливает диполь p_est из ЭКГ.
    
    Аргументы:
        ecg_path (str): Путь к файлу *_raw.npy или *_filtered.npy.
        
    Возвращает:
        str: Путь к файлу с сохранённой оценкой диполя p_est.npy.
    """
    # 1. Загрузка ЭКГ
    if ecg_path.endswith('_raw.npy'):
        # Если подали сырые данные, нужно их "денормализовать"
        adc = np.load(ecg_path)
        ecg = (adc.astype(np.float32) - 512.0) / 200.0
    else:
        # Предполагаем, что это уже *_filtered.npy
        ecg = np.load(ecg_path)
    
    # ecg.shape = (N_LEADS, T)
    print(f"Загружена ЭКГ с формой: {ecg.shape}")

    # 2. Загрузка/генерация матрицы отведений L
    L = load_lead_matrix(n_leads=ecg.shape[0], n_dipole=params.N_DIPOLE)
    print(f"Сгенерирована матрица отведений L с формой: {L.shape}")
    
    # 3. Решение обратной задачи.
    # У нас есть система: ecg = L @ p_est
    # ecg: (6, T), L: (6, 3) -> p_est: (3, T)
    #
    # Используем псевдообратную матрицу (метод наименьших квадратов)
    L_pinv = np.linalg.pinv(L)  # L_pinv будет иметь форму (3, 6)
    print(f"Псевдообратная матрица L_pinv имеет форму: {L_pinv.shape}")
    
    # Теперь просто умножаем: p_est = L_pinv @ ecg
    p_est = L_pinv @ ecg  # (3, 6) @ (6, T) -> (3, T)
    print(f"Восстановленный диполь p_est имеет форму: {p_est.shape}")

    # 4. Сохранение результата
    base_path = os.path.join(params.RECORDINGS_DIR, params.MOCK_BASENAME)
    p_est_path = f"{base_path}_p_est.npy"
    np.save(p_est_path, p_est.astype(np.float32))
    
    print(f"✅ Обратная задача решена. Оценка диполя сохранена: {p_est_path}")
    return p_est_path