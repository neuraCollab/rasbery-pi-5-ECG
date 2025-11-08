# ecg_spectral_analysis.py (без cwt)
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, spectrogram
from numpy.fft import rfft, rfftfreq
import csv, os
from scipy.signal import welch


FS = 500
DURATION = 10.0
HR_BPM = 72
NOISE_STD = 0.02
BASELINE_DRIFT = 0.2
OUT_CSV = "synthetic_ad8232.csv"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

def synth_ecg(fs=FS, duration=DURATION, hr_bpm=HR_BPM,
              noise_std=NOISE_STD, baseline_drift=BASELINE_DRIFT):
    t = np.arange(0, duration, 1.0/fs)
    hr_hz = hr_bpm / 60.0
    beats = np.arange(0, duration, 1.0/hr_hz)
    signal = np.zeros_like(t)

    for b in beats:
        qrs_w = 0.06
        signal += 1.0 * np.exp(-0.5*((t - b)/qrs_w)**2)
        signal += 0.2 * np.exp(-0.5*((t - (b - 0.18))/0.09)**2)
        signal += 0.35 * np.exp(-0.5*((t - (b + 0.28))/0.12)**2)

    signal = signal / np.max(np.abs(signal))
    drift = baseline_drift*(0.6*np.sin(2*np.pi*0.25*t)+0.4*np.sin(2*np.pi*0.05*t))
    signal += drift
    signal += noise_std*np.random.randn(len(signal))
    adc = 512 + (signal * 200)
    adc = np.clip(adc, 0, 1023).astype(np.int16)
    return t, signal, adc

def save_csv(adc, path=OUT_CSV):
    with open(path, "w", newline='') as f:
        w = csv.writer(f)
        w.writerow(["adc"])
        for v in adc:
            w.writerow([int(v)])
    print("Saved:", path)
 
def analyze_fft(signal, fs=FS):
    # убираем среднее (DC) и небольшую лин. тенденцию
    x = signal - np.mean(signal)
    # используем Welch — даёт PSD в единицах "мощность/Hz"
    f, Pxx = welch(x, fs=fs, nperseg=1024, noverlap=512, scaling='density')
    # ограничим диапазон для поиска пика (0.5..5 Hz)
    mask = (f >= 0.5) & (f <= 5.0)
    if np.any(mask):
        domf = f[mask][np.argmax(Pxx[mask])]
    else:
        domf = f[np.argmax(Pxx)]
    return f, Pxx, domf

def analyze_spectrogram(signal, fs=FS, nperseg=256, noverlap=200):
    f, t_spec, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg,
                                 noverlap=noverlap, scaling='spectrum')
    return f, t_spec, Sxx

def main():
    t, signal_norm, adc = synth_ecg()
    save_csv(adc)
    real_signal = (adc.astype(float) - 512.0) / 1024.0

    xf, psd, domf = analyze_fft(signal_norm, FS)
    print(f"Dominant freq: {domf:.3f} Hz (expected {HR_BPM/60:.2f})")

    df = xf[1]-xf[0]
    bands = {
        "heart (0.5–5Hz)": psd[(xf>=0.5)&(xf<=5.0)].sum()*df,
        "resp (0.1–0.5Hz)": psd[(xf>=0.1)&(xf<=0.5)].sum()*df,
        "high (>40Hz)": psd[(xf>40)].sum()*df
    }
    for k,v in bands.items():
        print(f"{k}: {v:.4e}")

    f_spec, t_spec, Sxx = analyze_spectrogram(signal_norm, FS)

    # === Графики ===
    plt.figure(figsize=(10,3))
    plt.plot(t, signal_norm)
    plt.xlabel("Time (s)"); plt.title("Synthetic ECG-like signal")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "signal.png"), dpi=150)

    plt.figure(figsize=(10,4))
    plt.semilogy(xf, psd)
    plt.xlim(0,50)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (power / Hz)")
    plt.title("Power spectrum (Welch)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fft_welch.png"), dpi=150)



    plt.figure(figsize=(10,4))
    plt.pcolormesh(t_spec, f_spec, 10*np.log10(Sxx+1e-12), shading='gouraud')
    plt.ylabel('Freq [Hz]'); plt.xlabel('Time [s]')
    plt.title("Spectrogram (STFT)"); plt.ylim(0,40)
    plt.colorbar(label='dB'); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "spectrogram.png"), dpi=150)

    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as sf:
        sf.write(f"Dominant frequency: {domf:.4f} Hz\n")
        for k,v in bands.items():
            sf.write(f"{k}: {v:.4e}\n")
    print("Saved results to", OUT_DIR)

if __name__ == "__main__":
    main()
