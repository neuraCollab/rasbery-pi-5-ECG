#include <Arduino.h>

// AD8232 ЭКГ — простой вывод в Serial Plotter
const int ECG_OUTPUT = A0;
const int LO_PLUS  = 10;
const int LO_MINUS = 11;

void setup() {
  Serial.begin(115200);
  pinMode(LO_PLUS, INPUT);
  pinMode(LO_MINUS, INPUT);
}

void loop() {
  int ecgValue = analogRead(ECG_OUTPUT);
  
  // Проверка отсоединения электродов
  if (digitalRead(LO_PLUS) == HIGH || digitalRead(LO_MINUS) == HIGH) {
    // Электроды отключены — можно отправить маркер
    Serial.println(512); // или 1023 — "flatline"
  } else {
    Serial.println(ecgValue);
  }

  delay(10); // ~100 Гц — достаточно для ЭКГ
}