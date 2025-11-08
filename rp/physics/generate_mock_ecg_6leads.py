"""
Генератор моковых ЭКГ для тестирования дипольной модели.
Создаёт 6-канальные *_filtered.npy и *_raw.npy в recordings/.
"""

import os
import numpy as np

RECORDINGS_DIR = "recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

FS = 100.0
DURATION = 10.0
T = int(FS * DURATION)
t = np.arange(T) / FS

def synth_single_lead(t, hr=72, noise=0.02, drift=0.1, phase_shift=0.0):
    hr_hz = hr / 60.0
    beats = np.arange(0, t[-1], 1/hr_hz)
    sig = np.zeros_like(t)
    for b in beats:
        sig += np.exp(-0.5*((t-b)/0.05)**2)
        sig += 0.3*np.exp(-0.5*((t-(b-0.15))/0.08)**2)
        sig += 0.5*np.exp(-0.5*((t-(b+0.25))/0.1)**2)
    sig /= np.max(np.abs(sig))
    sig += drift * np.sin(2*np.pi*0.3*t + phase_shift)
    sig += noise * np.random.randn(len(t))
    return sig

# Синтез 6 отведений с небольшими фазовыми сдвигами
six_leads = np.stack([
    synth_single_lead(t, phase_shift=i*np.pi/12)
    for i in range(6)
], axis=1)

# Масштаб до диапазона Arduino (0–1023)
adc = 512 + 200 * six_leads
adc = np.clip(adc, 0, 1023).astype(np.int16)

base = os.path.join(RECORDINGS_DIR, "ecg_mock_0001")
np.save(f"{base}_filtered.npy", six_leads.astype(np.float32))
np.save(f"{base}_raw.npy", adc.astype(np.int16))

print(f"✅ Моковые данные сохранены: {base}_filtered.npy, shape={six_leads.shape}")
