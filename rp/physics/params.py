# params.py
import os

# --- Основные параметры сигнала ---
FS = 100.0          # Частота дискретизации (Гц)
DURATION = 10.0     # Длительность записи (с)
HR_BPM = 72         # Частота сердцебиения (уд/мин)
NOISE_STD = 0.02    # Стандартное отклонение шума

# --- Пути к директориям ---
RECORDINGS_DIR = "recordings"
RESULTS_DIR = "results"
os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Параметры дипольной модели ---
N_LEADS = 6         # Количество ЭКГ-отведений
N_DIPOLE = 3        # Размерность диполя (3D)

# --- Имена файлов ---
MOCK_BASENAME = "ecg_mock_0001"