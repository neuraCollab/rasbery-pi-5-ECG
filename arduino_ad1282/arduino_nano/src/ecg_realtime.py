import serial
import matplotlib.pyplot as plt
import numpy as np
import glob
import sys

# === Автоматическое определение порта ===
def find_arduino_port():
    ports = glob.glob('/dev/tty{USB,ACM}*')
    if not ports:
        print("❌ Arduino не найден. Подключите устройство.")
        sys.exit(1)
    return ports[0]

# === Настройка ===
PORT = find_arduino_port()
BAUD = 9600
BUFFER_SIZE = 500  # Количество точек на графике

print(f"✅ Подключаюсь к {PORT}...")
try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
except Exception as e:
    print(f"❌ Ошибка подключения: {e}")
    sys.exit(1)

# === Настройка графика ===
plt.ion()  # Интерактивный режим
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_ylim(0, 1023)
ax.set_xlim(0, BUFFER_SIZE)
ax.set_title('ЭКГ в реальном времени (AD8232 + Arduino Nano)', fontsize=14)
ax.set_xlabel('Время')
ax.set_ylabel('Амплитуда')
ax.grid(True, linestyle='--', alpha=0.6)

x = np.arange(BUFFER_SIZE)
y = np.zeros(BUFFER_SIZE)
line, = ax.plot(x, y, 'g-', linewidth=1.5)  # Зелёная линия

print("📡 Ожидание данных... Приложите пальцы к электродам.")

try:
    while True:
        raw = ser.readline().decode('utf-8').strip()
        
        if raw == "!":
            print("⚠️ Электроды отключены")
            continue
            
        if raw.isdigit():
            value = int(raw)
            # Сдвигаем буфер влево и добавляем новое значение
            y = np.roll(y, -1)
            y[-1] = value
            
            line.set_ydata(y)
            fig.canvas.draw()
            fig.canvas.flush_events()

except KeyboardInterrupt:
    print("\n⏹️  Остановка...")
finally:
    ser.close()
    plt.ioff()
    plt.show()