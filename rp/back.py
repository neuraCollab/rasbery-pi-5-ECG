import asyncio
import serial
import threading
import time
import json
import os
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from collections import deque
from typing import Optional
from scipy.signal import butter, filtfilt

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

def safe_readline(ser_obj):
    try:
        if ser_obj and ser_obj.in_waiting > 0:
            line = ser_obj.readline().decode('utf-8', errors='ignore').strip()
            if line.isdigit():
                return int(line)
    except Exception:
        pass
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

def save_recording(lead_I: list, lead_II: list):
    """Сохраняет СЫРЫЕ данные (без фильтрации) в .npy"""
    try:
        lead_I = np.array(lead_I[:SEQ_LEN], dtype=np.float32)
        lead_II = np.array(lead_II[:SEQ_LEN], dtype=np.float32)
        six_leads = compute_all_leads(lead_I, lead_II)
        idx = len([f for f in os.listdir(RECORDINGS_DIR) if f.endswith('.npy')]) + 1
        base_path = os.path.join(RECORDINGS_DIR, f"ecg_{idx:04d}")
        np.save(f"{base_path}.npy", six_leads)
        print(f"✅ Запись сохранена: {base_path}.npy")
    except Exception as e:
        print(f"❌ Ошибка сохранения: {e}")

def read_serial_data(loop):
    global recording_ready, last_save_time
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
                        except Exception:
                            pass

        now = time.time()
        if recording_ready and now - last_save_time >= 10.0:
            with buffer_lock:
                if len(buffer_I) >= SEQ_LEN and len(buffer_II) >= SEQ_LEN:
                    lead_I_snapshot = list(buffer_I)[:SEQ_LEN]
                    lead_II_snapshot = list(buffer_II)[:SEQ_LEN]
                    buffer_I.clear()
                    buffer_II.clear()
                    threading.Thread(
                        target=save_recording,
                        args=(lead_I_snapshot, lead_II_snapshot),
                        daemon=True
                    ).start()
                    last_save_time = now
                    recording_ready = False

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
        print(f"✅ Подключены: {PORT_I}, {PORT_II}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        ser_I = ser_II = None

    if ser_I and ser_II:
        stop_event.clear()
        loop = asyncio.get_running_loop()
        threading.Thread(target=read_serial_data, args=(loop,), daemon=True).start()

    yield

    stop_event.set()
    if ser_I: ser_I.close()
    if ser_II: ser_II.close()
    print("🔌 Порты закрыты")

app = FastAPI(title="6-канальный ЭКГ монитор", lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        active_connections.remove(websocket)