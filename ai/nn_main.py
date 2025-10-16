# %%
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F

# %%
# Загрузка
X_train = np.load("ecg_data/X_train.npy")
X_test = np.load("ecg_data/X_test.npy")
y_train = np.load("ecg_data/y_train.npy")
y_test = np.load("ecg_data/y_test.npy")
Y_train = pd.read_csv("ecg_data/Y_train.csv", index_col=0)

# Проверка
print("X_train.shape:", X_train.shape)
print(Y_train.shape)
print("y_train.shape:", y_train.shape)
print("Целевые переменные:", open("ecg_data/target_columns.txt").read().splitlines())


# %%
class ECGNet(nn.Module):
    def __init__(self, in_channels=6, num_classes=19, seq_len=1000):
        super().__init__()
        self.conv1 = nn.Conv1d(6, 19, 2, 1, 1)

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels=64, kernel_size=7, padding="same"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # (64, 500)
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # (128, 260)
            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),  # (256, 125)
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(10),  # -> (512, 10) — фиксированная длина
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 10, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_block(x)
        return self.classifier(x)

    def get_item(self, x, idx):
        return x[idx]


# %%
from torch.utils.data import TensorDataset, DataLoader

x_train_t = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
y_train_t = torch.tensor(y_train, dtype=torch.float32)

train_dataset = TensorDataset(x_train_t, y_train_t)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ECGNet().to(device=device)

pos_weights = torch.tensor([
    (y_train.shape[0] - y_train[:, i].sum()) / (y_train[:, i].sum() + 1e-5)
    for i in range(19)
], dtype=torch.float32)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# %%
model.train()
for _e in range(10):
    total_loss = 0

    for x_batch, y_batch in train_dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {_e+1}, LossL {total_loss/len(train_dataloader):.4f}")

# %%
from sklearn.metrics import average_precision_score, f1_score

model.eval()
all_preds = []
all_targets = []

X_test_t = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)
y_test_t = torch.tensor(y_train, dtype=torch.float32)

with open('ecg_data/target_columns.txt') as f:
    target_names = f.read().splitlines()

print(f"X_test.shape: {X_test_t.shape}")
print(f"y_test.shape: {y_test.shape}")
print(f"Классы: {target_names}")


with torch.no_grad():
    y_pred_proba = model(X_test_t)
    y_pred_proba = y_pred_proba.cpu().numpy()

y_pred_binary = (y_pred_proba > 0.5).astype(int)

# A. Micro-mAP (общая эффективность)
mAP_micro = average_precision_score(y_test, y_pred_proba, average="micro")

# B. Weighted-mAP (по поддержке класса)
mAP_weighted = np.average(
    [average_precision_score(y_test[:, i], y_pred_proba[:, i]) for i in range(19)],
    weights=y_test.sum(axis=0)  # вес = число позитивов
)

# C. AP только по "значимым" классам (например, > 0.5%)
significant_classes = [i for i, freq in enumerate(y_test.mean(axis=0)) if freq >= 0.005]
mAP_significant = np.mean([
    average_precision_score(y_test[:, i], y_pred_proba[:, i]) 
    for i in significant_classes
])

print(f"mAP (micro):     {mAP_micro:.4f}")
print(f"mAP (weighted):  {mAP_weighted:.4f}")
print(f"mAP (significant): {mAP_significant:.4f}")

valid_classes = [i for i in range(19) if y_test[:, i].sum() > 0]
mAP_valid = np.mean([
    average_precision_score(y_test[:, i], y_pred_proba[:, i]) 
    for i in valid_classes
])
print(f"mAP (только по классам с позитивами): {mAP_valid:.4f}")

# %%
key_classes = ['is_afib', 'is_pvc', 'is_sinus_arrhythmia', 'has_lbbb', 'has_irbbb']
for name in key_classes:
    i = target_names.index(name)
    ap = average_precision_score(y_test[:, i], y_pred_proba[:, i])
    print(f"{name:<25}: AP = {ap:.4f} (частота: {y_test[:, i].mean()*100:.2f}%)")
# %%
# Посчитай частоту каждого класса в y_test (и в y_train!)
print("Доля положительных примеров в y_test:")
for name, freq in zip(target_names, y_test.mean(axis=0)):
    print(f"{name:<25}: {freq:.4f} ({freq*100:.2f}%)")
# %%
idx = target_names.index("is_sinus_rhythm")
ap_sinus = average_precision_score(y_test[:, idx], y_pred_proba[:, idx])
f1_sinus = f1_score(y_test[:, idx], y_pred_binary[:, idx])

print(f"AP для is_sinus_rhythm: {ap_sinus:.4f}")
print(f"F1 для is_sinus_rhythm: {f1_sinus:.4f}")
# %%
# Трейсинг модели
example_input = torch.randn(1, 6, 1000)

traced_model = torch.jit.trace(model, example_input)

# Сохраните
traced_model.save("ecg_model_traced.pt")
print("✅ Модель экспортирована в TorchScript")
# %%
X_test_t
# %%
