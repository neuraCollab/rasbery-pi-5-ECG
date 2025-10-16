import asyncio
import serial
import threading
import time
import json
import os
import numpy as np
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from collections import deque
from typing import Optional
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import io

# === Настройка логирования ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("ecg_app.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === Настройки ===
PORT_I = "/dev/ttyUSB0"
PORT_II = "/dev/ttyUSB1"
BAUD_RATE = 115200
SEQ_LEN = 1000
RECORDINGS_DIR = "recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

templates = Jinja2Templates(directory="templates")
active_connections: list[WebSocket] = []
stop_event = threading.Event()

buffer_I = deque(maxlen=SEQ_LEN + 100)
buffer_II = deque(maxlen=SEQ_LEN + 100)
buffer_lock = threading.Lock()

recording_ready = True
last_save_time = 0

ser_I: Optional[serial.Serial] = None
ser_II: Optional[serial.Serial] = None


# === PyTorch и модель ===
import torch
import torch.nn.functional as F

# Загружаем модель один раз при старте
MODEL_PATH = "ecg_model_traced.pt"
device = torch.device("cpu")  # На Pi только CPU
try:
    # TorchScript модели загружаются через torch.jit.load
    model = torch.jit.load(MODEL_PATH, map_location=device)
    # У ScriptModule нет .eval(), но он и так в eval-режиме
    logger.info(f"✅ TorchScript-модель загружена: {MODEL_PATH}")
except Exception as e:
    model = None
    logger.error(f"❌ Не удалось загрузить модель: {e}")

from scipy import signal as scipy_signal

def adc_to_uv(adc_value):
    # АЦП: 0–1023 соответствует 0–5.0 В
    voltage_at_adc = adc_value * (5.0 / 1023.0)  # в вольтах

    # Но сигнал от AD8232: 0–3.3 В → смещение 1.65 В = изоэлектрическая линия
    voltage_diff = voltage_at_adc - 1.65  # в вольтах

    # Учёт усиления AD8232 (GAIN = 100)
    input_uv = (voltage_diff / 100.0) * 1_000_000  # в микровольтах

    return input_uv

def preprocess_single_ecg(signal: np.ndarray, fs=100) -> np.ndarray:
    """
    Применяет ТОЧНО ТУ ЖЕ предобработку, что и при обучении на PTB-XL:
    - Полосовой фильтр 0.5–40 Гц
    - Z-score нормализация по каждому отведению
    """
    sos = scipy_signal.butter(4, [0.5, 40], btype='band', fs=fs, output='sos')
    filtered = scipy_signal.sosfilt(sos, signal, axis=0)

    mean = filtered.mean(axis=0, keepdims=True)
    std = filtered.std(axis=0, keepdims=True) + 1e-6
    normalized = (filtered - mean) / std

    # ❌ НЕ ОБРЕЗАЕМ до [-1, 1] — это нарушает совместимость с обучением!
    # ✅ Оставляем как есть: z-score может быть [-3, +3] и более — это нормально.
    return normalized

def adc_to_microvolts(signal_adc: np.ndarray) -> np.ndarray:
    """
    Переводит значения analogRead от Arduino + AD8232 в микровольты (µV),
    учитывая baseline 512 и усиление 110.
    """
    volts_per_step = 5.0 / 1024.0
    gain = 110.0
    baseline = 512

    microvolts = (signal_adc - baseline) * volts_per_step * 10 / gain
    return microvolts


from scipy.stats import iqr

def predict_ecg(signal: np.ndarray) -> dict:
    if model is None:
        return {"error": "Модель не загружена"}

    try:
        if signal.shape != (1000, 6):
            return {"error": f"Ожидалась форма (1000, 6), получена {signal.shape}"}

        # === Шаг 1: Центрирование ===
        signal_centered = signal - 512.0

        # === Шаг 2: Полосовой фильтр 0.5–40 Гц ===
        sos = scipy_signal.butter(4, [0.5, 40], btype='band', fs=100, output='sos')
        filtered = scipy_signal.sosfilt(sos, signal_centered, axis=0)

        # === Шаг 3: СГЛАЖИВАНИЕ QRS (ключевое!) ===
        from scipy.ndimage import gaussian_filter1d
        filtered_smooth = np.zeros_like(filtered)
        for i in range(6):
            filtered_smooth[:, i] = gaussian_filter1d(filtered[:, i], sigma=1.0)
        filtered = filtered_smooth

        # === Шаг 4: Нормализация по амплитуде ===
        max_abs = np.max(np.abs(filtered), axis=0, keepdims=True) + 1e-6
        normalized = 1.7 * filtered / max_abs  # 1.7 — хороший баланс

        # === Шаг 5: Clip ===
        normalized = np.clip(normalized, -4.0, 4.0)

        # === Подготовка тензора ===
        x = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)
        x = x.permute(0, 2, 1)

        # === Предсказание ===
        with torch.no_grad():
            probs = model(x).cpu().numpy()[0]

        class_names = [
            'is_sinus_rhythm', 'is_afib', 'is_aflt', 'is_pac', 'is_pvc',
            'is_svt', 'is_sinus_arrhythmia', 'has_1avb', 'has_2avb', 'has_3avb',
            'has_rbbb', 'has_lbbb', 'has_irbbb', 'has_ilbbb', 'has_lafb',
            'has_lpfb', 'has_wpw', 'has_bigeminy', 'has_trigeminy'
        ]

        threshold = 0.1
        predictions = {
            class_names[i]: {
                "probability": float(probs[i] * 10),
                "predicted": bool(probs[i] * 10 >= threshold)
            }
            for i in range(len(class_names))
        }

        return {"prediction": predictions}

    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}")
        return {"error": str(e)}
    

def debug_plot_signal(signal: np.ndarray, path: str):
    import matplotlib.pyplot as plt
    t = np.arange(signal.shape[0])
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF']
    plt.figure(figsize=(12, 6))
    for i in range(6):
        plt.subplot(6, 1, i+1)
        plt.plot(t, signal[:, i])
        plt.ylabel(leads[i])
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    
def safe_readline(ser_obj):
    try:
        if ser_obj and ser_obj.in_waiting > 0:
            line = ser_obj.readline().decode('utf-8', errors='ignore').strip()
            if line.isdigit():
                return int(line)
    except Exception as e:
        logger.debug(f"Ошибка чтения из порта: {e}")
    return None

def butter_lowpass_filter(data, cutoff, fs, order=5):
    """Фильтр Баттерворта нижних частот."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def moving_average(signal: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Скользящее среднее."""
    if len(signal) < window_size:
        return signal
    weights = np.ones(window_size) / window_size
    return np.convolve(signal, weights, mode='valid')

def compute_all_leads(lead_I: np.ndarray, lead_II: np.ndarray) -> np.ndarray:
    lead_III = lead_II - lead_I
    lead_aVR = -(lead_I + lead_II) / 2.0
    lead_aVL = lead_I - lead_II / 2.0
    lead_aVF = lead_II - lead_I / 2.0
    return np.stack([lead_I, lead_II, lead_III, lead_aVR, lead_aVL, lead_aVF], axis=1)

def save_ecg_gif(six_leads: np.ndarray, base_path: str):
    """Сохраняет анимированный GIF всех 6 отведений."""
    try:
        t = np.arange(six_leads.shape[0])
        leads_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF']
        frames = []
        step = max(1, len(t) // 100)  # не более 100 кадров

        for i in range(step, len(t) + 1, step):
            fig, axes = plt.subplots(6, 1, figsize=(12, 8), sharex=True)
            for j in range(6):
                axes[j].plot(t[:i], six_leads[:i, j], 'b-', linewidth=1.5)
                axes[j].set_ylabel(leads_names[j], fontsize=9)
                axes[j].grid(True, linestyle='--', alpha=0.6)
            axes[-1].set_xlabel("Время (отсчёты)")
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=80)
            buf.seek(0)
            frames.append(Image.open(buf))
            plt.close(fig)

        gif_path = f"{base_path}.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0
        )
        for f in frames:
            f.close()
        logger.info(f"GIF сохранён: {gif_path}")
    except Exception as e:
        logger.error(f"Ошибка при создании GIF: {e}")

def save_recording(raw_I: list, raw_II: list, filtered_I: list, filtered_II: list):
    """Сохраняет сырые и отфильтрованные данные + GIF."""
    try:
        # Сырые данные
        raw_I_arr = np.array(raw_I[:SEQ_LEN], dtype=np.float32)
        raw_II_arr = np.array(raw_II[:SEQ_LEN], dtype=np.float32)
        raw_six_leads = compute_all_leads(raw_I_arr, raw_II_arr)

        # Отфильтрованные данные
        filtered_I_arr = np.array(filtered_I[:SEQ_LEN], dtype=np.float32)
        filtered_II_arr = np.array(filtered_II[:SEQ_LEN], dtype=np.float32)
        filtered_six_leads = compute_all_leads(filtered_I_arr, filtered_II_arr)

        idx = len([f for f in os.listdir(RECORDINGS_DIR) if f.endswith('.npy')]) + 1
        base_path = os.path.join(RECORDINGS_DIR, f"ecg_{idx:04d}")

        # Сохраняем
        np.save(f"{base_path}_raw.npy", raw_six_leads)
        np.save(f"{base_path}_filtered.npy", filtered_six_leads)
        logger.info(f"Сохранены данные: {base_path}_raw.npy, {base_path}_filtered.npy")

        # Сохраняем GIF
        save_ecg_gif(filtered_six_leads, base_path)

    except Exception as e:
        logger.error(f"Ошибка сохранения записи: {e}")

def read_serial_data(loop):
    global recording_ready, last_save_time
    logger.info("Запущен поток чтения данных с портов")
    while not stop_event.is_set():
        val_I = safe_readline(ser_I)
        val_II = safe_readline(ser_II)

        if val_I is not None or val_II is not None:
            with buffer_lock:
                if val_I is not None:
                    buffer_I.append(val_I)
                if val_II is not None:
                    buffer_II.append(val_II)

            # Фильтруем последние 50 значений
            with buffer_lock:
                if len(buffer_I) >= 50 and len(buffer_II) >= 50:
                    I_raw = np.array(list(buffer_I)[-50:])
                    II_raw = np.array(list(buffer_II)[-50:])

                    # 1. Фильтр Баттерворта
                    I_butter = butter_lowpass_filter(I_raw, cutoff=40, fs=100, order=5)
                    II_butter = butter_lowpass_filter(II_raw, cutoff=40, fs=100, order=5)

                    # 2. Скользящее среднее (окно 3)
                    I_smooth_arr = moving_average(I_butter, window_size=3)
                    II_smooth_arr = moving_average(II_butter, window_size=3)

                    # Берём последнее значение
                    I_final = float(I_smooth_arr[-1]) if len(I_smooth_arr) > 0 else float(I_butter[-1])
                    II_final = float(II_smooth_arr[-1]) if len(II_smooth_arr) > 0 else float(II_butter[-1])

                    III = II_final - I_final
                    aVR = -(I_final + II_final) / 2.0
                    aVL = I_final - II_final / 2.0
                    aVF = II_final - I_final / 2.0

                    data = {
                        "I": I_final,
                        "II": II_final,
                        "III": round(III, 1),
                        "aVR": round(aVR, 1),
                        "aVL": round(aVL, 1),
                        "aVF": round(aVF, 1),
                        "timestamp": time.time()
                    }

                    for ws in active_connections[:]:
                        try:
                            asyncio.run_coroutine_threadsafe(
                                ws.send_text(json.dumps(data)),
                                loop
                            )
                        except Exception as e:
                            logger.debug(f"Ошибка отправки по WebSocket: {e}")

        now = time.time()
        if recording_ready and now - last_save_time >= 10.0:
            with buffer_lock:
                if len(buffer_I) >= SEQ_LEN and len(buffer_II) >= SEQ_LEN:
                    logger.info("Накоплено 1000 отсчётов. Начинаю сохранение записи...")
                    # Сохраняем сырые данные
                    raw_I_snapshot = list(buffer_I)[:SEQ_LEN]
                    raw_II_snapshot = list(buffer_II)[:SEQ_LEN]

                    # Применяем фильтр ко всему буферу для отфильтрованной записи
                    I_raw_full = np.array(raw_I_snapshot)
                    II_raw_full = np.array(raw_II_snapshot)

                    I_filtered_full = butter_lowpass_filter(I_raw_full, cutoff=40, fs=100, order=5)
                    II_filtered_full = butter_lowpass_filter(II_raw_full, cutoff=40, fs=100, order=5)

                    I_filtered_smooth = moving_average(I_filtered_full, window_size=3)
                    II_filtered_smooth = moving_average(II_filtered_full, window_size=3)

                    # Дополняем до длины SEQ_LEN
                    I_filtered_smooth_padded = np.pad(I_filtered_smooth, (0, SEQ_LEN - len(I_filtered_smooth)), mode='edge')
                    II_filtered_smooth_padded = np.pad(II_filtered_smooth, (0, SEQ_LEN - len(II_filtered_smooth)), mode='edge')

                    # Сохраняем в фоновом потоке
                    threading.Thread(
                        target=save_recording,
                        args=(raw_I_snapshot, raw_II_snapshot, I_filtered_smooth_padded.tolist(), II_filtered_smooth_padded.tolist()),
                        daemon=True
                    ).start()

                    # Очищаем буферы
                    buffer_I.clear()
                    buffer_II.clear()
                    last_save_time = now
                    recording_ready = False
                    logger.info("Буферы очищены. Ожидание следующей записи...")

        if not recording_ready and now - last_save_time >= 10.0:
            recording_ready = True

        time.sleep(0.001)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ser_I, ser_II
    try:
        ser_I = serial.Serial(PORT_I, BAUD_RATE, timeout=0.01)
        ser_II = serial.Serial(PORT_II, BAUD_RATE, timeout=0.01)
        time.sleep(1.5)
        ser_I.reset_input_buffer()
        ser_II.reset_input_buffer()
        logger.info(f"✅ Подключены: {PORT_I}, {PORT_II}")
    except Exception as e:
        logger.error(f"❌ Ошибка подключения к портам: {e}")
        ser_I = ser_II = None

    if ser_I and ser_II:
        stop_event.clear()
        loop = asyncio.get_running_loop()
        threading.Thread(target=read_serial_data, args=(loop,), daemon=True).start()

    yield

    stop_event.set()
    if ser_I: ser_I.close()
    if ser_II: ser_II.close()
    logger.info("🔌 Порты закрыты")

app = FastAPI(title="6-канальный ЭКГ монитор", lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"Новое WebSocket-соединение. Всего: {len(active_connections)}")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        active_connections.remove(websocket)
        logger.info(f"WebSocket-соединение закрыто. Осталось: {len(active_connections)}")
        
from fastapi import File, UploadFile
from fastapi.responses import JSONResponse

@app.get("/predict_latest")
async def predict_latest():
    """Предсказывает по последней сохранённой записи."""
    try:
        files = sorted([f for f in os.listdir(RECORDINGS_DIR) if f.endswith('_raw.npy')])
        if not files:
            return JSONResponse({"error": "Нет сохранённых записей"}, status_code=404)

        latest_raw = os.path.join(RECORDINGS_DIR, files[-1])
        latest_filtered = latest_raw.replace('_raw.npy', '_filtered.npy')

        raw_data = np.load(latest_raw)      # (1000, 6)
        filtered_data = np.load(latest_filtered)
        
        # debug_plot_signal(raw_data, "debug_signal.png")

        pred_raw = predict_ecg(raw_data)
        pred_filtered = predict_ecg(filtered_data)

        return {
            "raw_prediction": pred_raw,
            "filtered_prediction": pred_filtered,
            "filename": files[-1].replace('_raw.npy', '')
        }
    except Exception as e:
        logger.error(f"Ошибка в /predict_latest: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/predict_upload")
async def predict_upload(file: UploadFile = File(...)):
    """Предсказывает по загруженному пользователем .npy файлу."""
    if not file.filename.endswith('.npy'):
        return JSONResponse({"error": "Только .npy файлы"}, status_code=400)

    try:
        contents = await file.read()
        buf = io.BytesIO(contents)
        data = np.load(buf)

        if data.shape != (1000, 6):
            return JSONResponse({"error": f"Ожидалась форма (1000, 6), получена {data.shape}"}, status_code=400)

        pred = predict_ecg(data)
        return {"prediction": pred, "filename": file.filename}
    except Exception as e:
        logger.error(f"Ошибка загрузки/предсказания: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)