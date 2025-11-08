# src/data_generation/mock_ecg.py
import numpy as np
import os
# Импортируем параметры напрямую
import params # Или: from params import FS, DURATION, HR_BPM, NOISE_STD, RECORDINGS_DIR, MOCK_BASENAME, N_LEADS, N_DIPOLE
from scipy.integrate import solve_ivp

def synth_dipole_3d_fhn(t, hr_bpm=params.HR_BPM, noise_std=0.0, seed=42):
    """
    Синтезирует "истинный" 3D-диполь сердца, используя упрощённую 
    динамическую модель на основе ФитцХью-Нагумо.
    """
    np.random.seed(seed)
    duration = t[-1]
    fs = 1 / (t[1] - t[0])

    # Начальные условия и параметры для 3D-осциллятора
    # Мы будем использовать модифицированную систему,
    # где основная переменная v генерирует колебания,
    # а остальные компоненты — это её производные или фазовые сдвиги.
    def heart_oscillator(t, y):
        # y = [v, w, phi_x, phi_y, phi_z]
        v, w, phi_x, phi_y, phi_z = y
        
        # Параметры ФитцХью-Нагумо
        I = 0.5
        a = 0.7
        b = 0.8
        tau = 12.5
        
        # Уравнения для основного осциллятора (v, w)
        dvdt = v - v**3/3 - w + I
        dwdt = (v + a - b * w) / tau
        
        # Уравнения для фаз, чтобы создать 3D-траекторию
        # Фазы медленно дрейфуют, создавая сложную, но периодическую петлю
        dphix_dt = 0.1 * np.cos(phi_x)  # Очень медленное изменение
        dphiy_dt = 0.1 * np.sin(phi_y)
        dphiz_dt = 0.1 * np.cos(phi_z + phi_x)
        
        return [dvdt, dwdt, dphix_dt, dphiy_dt, dphiz_dt]

    # Начальные условия
    y0 = [-1.0, 1.0, 0.0, np.pi/4, np.pi/2]
    
    # Решаем систему ОДУ
    sol = solve_ivp(heart_oscillator, [0, duration], y0, t_eval=t, method='RK45')
    
    if not sol.success:
        raise RuntimeError("Не удалось решить ОДУ для генерации диполя.")
    
    v = sol.y[0]  # Основной осциллятор
    phi_x, phi_y, phi_z = sol.y[2], sol.y[3], sol.y[4]
    
    # Формируем 3D-диполь из осциллятора и фаз
    p_true = np.zeros((params.N_DIPOLE, len(t)))
    
    # Амплитуда может модулироваться самим осциллятором v
    amp_x = 1.0 + 0.2 * np.sin(2 * np.pi * (hr_bpm/60) * t)
    amp_y = 0.8 + 0.1 * np.cos(2 * np.pi * (hr_bpm/60) * t)
    amp_z = 0.5 + 0.05 * np.sin(4 * np.pi * (hr_bpm/60) * t)
    
    p_true[0] = amp_x * v * np.cos(phi_x)
    p_true[1] = amp_y * v * np.sin(phi_y)
    p_true[2] = amp_z * v * np.cos(phi_z)

    # Нормализуем по амплитуде
    p_true /= np.max(np.abs(p_true)) * 1.1

    if noise_std > 0:
        p_true += noise_std * np.random.randn(*p_true.shape)

    return p_true
 
def generate_lead_matrix(n_leads=params.N_LEADS, n_dipole=params.N_DIPOLE, seed=42):
    """
    Генерирует фиксированную, физиологически правдоподобную матрицу отведений.
    В реальности она определяется анатомией пациента.
    """
    np.random.seed(seed) # Для воспроизводимости
    # Матрица случайна, но её можно заменить на реальные значения из литературы
    L = np.random.randn(n_leads, n_dipole)
    # Нормализуем строки, чтобы амплитуды отведений были сопоставимы
    L = L / np.linalg.norm(L, axis=1, keepdims=True)
    return L

def generate_mock_ecg(model='fhn'):  # <-- ВАЖНО: есть параметр model со значением по умолчанию
    """
    Основная функция генерации моковых данных.
    Аргументы:
        model (str): 'fhn' для динамической модели или 'simple' для синусоид.
    """ 

    # 1. Временная ось
    T = int(params.FS * params.DURATION)
    t = np.arange(T) / params.FS

    # 2. Синтез "истинного" диполя
    if model == 'fhn':
        p_true = synth_dipole_3d_fhn(t, hr_bpm=params.HR_BPM, noise_std=0.001)
    else: # model == 'simple'
        p_true = synth_dipole_3d_simple(t, hr_bpm=params.HR_BPM, noise_std=0.001)

    # 3. Генерация матрицы отведений
    L = generate_lead_matrix()

    # 4. Генерация ЭКГ: V = L * p
    ecg_filtered = L @ p_true

    # 5. Добавление шума и моделирование АЦП
    ecg_filtered += params.NOISE_STD * np.random.randn(*ecg_filtered.shape)
    adc = 512 + (200 * ecg_filtered)
    adc = np.clip(adc, 0, 1023).astype(np.int16)

    # 6. Сохранение данных
    base_path = os.path.join(params.RECORDINGS_DIR, params.MOCK_BASENAME)
    np.save(f"{base_path}_filtered.npy", ecg_filtered.astype(np.float32))
    np.save(f"{base_path}_raw.npy", adc.astype(np.int16))
    np.save(f"{base_path}_p_true.npy", p_true.astype(np.float32))

    print(f"✅ Моковые данные и истинный диполь сохранены.")
    print(f"   Файлы: {base_path}_*.npy")

    return f"{base_path}_raw.npy"

# Не забудьте сохранить старую функцию для простоты!
def synth_dipole_3d_simple(t, hr_bpm=params.HR_BPM, noise_std=0.0):
    """Простая синусоидальная модель (старая версия)."""
    hr_hz = hr_bpm / 60.0
    f_qrs = hr_hz
    f_twave = hr_hz * 0.5
    p_true = np.zeros((params.N_DIPOLE, len(t)))
    p_true[0] = (1.0 * np.sin(2 * np.pi * f_qrs * t) + 0.3 * np.sin(2 * np.pi * f_twave * t))
    p_true[1] = (0.8 * np.sin(2 * np.pi * f_qrs * t + np.pi/4) + 0.25 * np.sin(2 * np.pi * f_twave * t + np.pi/6))
    p_true[2] = (0.5 * np.sin(2 * np.pi * f_qrs * t + np.pi/2) + 0.15 * np.sin(2 * np.pi * f_twave * t + np.pi/3))
    if noise_std > 0:
        p_true += noise_std * np.random.randn(*p_true.shape)
    return p_true