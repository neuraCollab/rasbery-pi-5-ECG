# %% [markdown]
# Тут стоит поиграться с fine-tuning и как-то привести данные к 2D (а не 3D)
# План:
# - Извлечь HRV-признаки через neurokit2
# - Посмотреть распределение классов
# - Использовать scale_pos_weight для редких классов
# Но, возможно, это не имеет смысла — лучше использовать DL (см. nn_main.py)
# %%

import os
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import average_precision_score, label_ranking_average_precision_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import joblib

# %%
# Загрузка
X_train = np.load('ecg_data/X_train.npy')
X_test  = np.load('ecg_data/X_test.npy')
y_train = np.load('ecg_data/y_train.npy')
y_test  = np.load('ecg_data/y_test.npy')  # убедись, что он есть!

# Проверка
print("X_train.shape:", X_train.shape)
print("y_train.shape:", y_train.shape)
print("X_test.shape:", X_test.shape)
print("y_test.shape:", y_test.shape)
print("Целевые переменные:", open('ecg_data/target_columns.txt').read().splitlines())

# %%
# === Feature Engineering ===
def extract_ecg_features(X):
    n_samples, timesteps, n_channels = X.shape
    features_list = []

    for ch in range(n_channels):
        x_ch = X[:, :, ch]
        mean = np.mean(x_ch, axis=1)
        std = np.std(x_ch, axis=1)
        min_val = np.min(x_ch, axis=1)
        max_val = np.max(x_ch, axis=1)
        median = np.median(x_ch, axis=1)
        skew = pd.DataFrame(x_ch).skew(axis=1).values
        kurt = pd.DataFrame(x_ch).kurt(axis=1).values
        ptp = max_val - min_val
        rms = np.sqrt(np.mean(x_ch**2, axis=1))
        zcr = np.sum((x_ch[:, :-1] * x_ch[:, 1:]) < 0, axis=1) / (timesteps - 1)
        
        ch_features = np.stack([mean, std, min_val, max_val, median, skew, kurt, ptp, rms, zcr], axis=1)
        features_list.append(ch_features)

    # Межканальные корреляции
    corr_features = []
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            corr = np.array([np.corrcoef(X[s, :, i], X[s, :, j])[0, 1] for s in range(n_samples)])
            corr_features.append(corr)
    corr_features = np.column_stack(corr_features)

    # FFT
    fft_features = []
    for ch in range(n_channels):
        x_ch = X[:, :, ch]
        fft_vals = np.fft.fft(x_ch, axis=1)
        amp = np.abs(fft_vals[:, :10])
        fft_features.append(amp)
    fft_features = np.concatenate(fft_features, axis=1)

    all_features = np.concatenate(features_list + [corr_features, fft_features], axis=1)
    print(f"Извлечено признаков: {all_features.shape[1]}")
    return all_features

# Извлекаем признаки
X_train_feat = extract_ecg_features(X_train)
X_test_feat = extract_ecg_features(X_test)

# Загружаем названия классов
with open('ecg_data/target_columns.txt') as f:
    target_names = f.read().splitlines()

# %%
# === Вспомогательные функции ===
def compute_pos_weights(y):
    """Вычисляет веса для scale_pos_weight."""
    weights = []
    for i in range(y.shape[1]):
        pos = y[:, i].sum()
        neg = y.shape[0] - pos
        weight = neg / (pos + 1e-5)
        weights.append(weight)
    return np.array(weights)

def safe_multilabel_ap(y_true, y_pred_proba, average="samples"):
    """
    Устойчивый расчёт AP: игнорирует классы без позитивов.
    """
    valid_classes = np.where(y_true.sum(axis=0) > 0)[0]
    if len(valid_classes) == 0:
        return 0.0
    return average_precision_score(
        y_true[:, valid_classes],
        y_pred_proba[:, valid_classes],
        average=average
    )

# Вычисляем веса
pos_weights = compute_pos_weights(y_train)

# %%
# === Обучение моделей вручную (с scale_pos_weight) ===
print("Обучение моделей с учётом дисбаланса...")

models = {}

# XGBoost
print("Обучение XGBoost...")
xgb_models = {}
for i, target in enumerate(target_names):
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        scale_pos_weight=pos_weights[i],
        eval_metric='logloss',
        n_jobs=1
    )
    model.fit(X_train_feat, y_train[:, i])
    xgb_models[target] = model
models["XGBoost"] = xgb_models

# LightGBM
print("Обучение LightGBM...")
lgb_models = {}
for i, target in enumerate(target_names):
    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        is_unbalance=True,  # LightGBM аналог scale_pos_weight
        verbose=-1
    )
    model.fit(X_train_feat, y_train[:, i])
    lgb_models[target] = model
models["LightGBM"] = lgb_models

# CatBoost
print("Обучение CatBoost...")
cb_models = {}
for i, target in enumerate(target_names):
    model = cb.CatBoostClassifier(
        n_estimators=200,
        depth=6,
        learning_rate=0.1,
        random_state=42,
        auto_class_weights='Balanced',  # обработка дисбаланса
        verbose=0
    )
    model.fit(X_train_feat, y_train[:, i])
    cb_models[target] = model
models["CatBoost"] = cb_models

# %%
# === Оценка на тесте ===
def predict_proba_multilabel(models_dict, X):
    probs = []
    for target in target_names:
        prob = models_dict[target].predict_proba(X)[:, 1]
        probs.append(prob)
    return np.column_stack(probs)

results = {}
for name, model_dict in models.items():
    y_pred_proba = predict_proba_multilabel(model_dict, X_test_feat)
    
    # Основная метрика: samples AP
    ap_samples = safe_multilabel_ap(y_test, y_pred_proba, average="samples")
    
    # Дополнительно: label ranking AP (ещё устойчивее)
    lr_ap = label_ranking_average_precision_score(y_test, y_pred_proba)
    
    results[name] = {
        "AP (samples)": ap_samples,
        "Label Ranking AP": lr_ap,
        "proba": y_pred_proba
    }
    print(f"{name}:")
    print(f"  AP (samples)       = {ap_samples:.4f}")
    print(f"  Label Ranking AP   = {lr_ap:.4f}")

# Выбираем лучшую модель по Label Ranking AP
best_model_name = max(results.keys(), key=lambda k: results[k]["Label Ranking AP"])
best_proba = results[best_model_name]["proba"]

print(f"\nЛучшая модель: {best_model_name}")

# Сохраняем
np.save("y_pred_proba.npy", best_proba)
joblib.dump(models[best_model_name], f"best_model_{best_model_name}.pkl")
print("Вероятности и модель сохранены!")

# %%
# === Анализ по ключевым классам ===
print("\nAP по ключевым диагнозам:")
key_classes = ['is_sinus_rhythm', 'is_afib', 'is_pvc', 'is_sinus_arrhythmia']
for cls in key_classes:
    if cls in target_names:
        i = target_names.index(cls)
        ap = average_precision_score(y_test[:, i], best_proba[:, i])
        freq = y_test[:, i].mean()
        print(f"{cls:<25}: AP = {ap:.4f} (частота: {freq:.2%})")

# %%
# === SHAP для XGBoost (если он лучший или для анализа) ===
try:
    import shap
    if "XGBoost" in models:
        idx = target_names.index("is_afib")
        xgb_single = models["XGBoost"][target_names[idx]]
        
        # Генерируем осмысленные названия признаков
        feature_names = []
        for ch in range(6):
            for stat in ['mean', 'std', 'min', 'max', 'median', 'skew', 'kurt', 'ptp', 'rms', 'zcr']:
                feature_names.append(f"ch{ch}_{stat}")
        for i in range(6):
            for j in range(i+1, 6):
                feature_names.append(f"corr_{i}_{j}")
        for ch in range(6):
            for k in range(10):
                feature_names.append(f"fft_ch{ch}_k{k}")
        
        explainer = shap.TreeExplainer(xgb_single)
        shap_values = explainer.shap_values(X_test_feat[:100])
        shap.summary_plot(shap_values, X_test_feat[:100], feature_names=feature_names, show=False)
        import matplotlib.pyplot as plt
        plt.savefig("shap_summary.png", bbox_inches='tight')
        print("SHAP-график сохранён в 'shap_summary.png'")
except Exception as e:
    print("SHAP недоступен или ошибка:", e)