# Pothole Detector (YOLOv8)

Модель компьютерного зрения для обнаружения дорожных ям на изображениях и видео. Основана на YOLOv8n с дообучением на кастомном датасете.

## 📊 Эффективность модели

### Ключевые метрики (после 50 эпох):
- **mAP50**: 0.479
- **Precision**: 0.721 (макс.)
- **Recall**: 0.569 (макс.)
- **Время инференса**: ~15 мс/кадр (на NVIDIA Tesla T4)

### Визуализация результатов:
| Реальная разметка | Предсказания модели |
|-------------------|---------------------|
| ![Реальная разметка](https://github.com/MrFireDeN/pothole-detector/blob/main/pothole_detector/yolov8n_custom/val_batch0_labels.jpg) | ![Предсказанная разметка](https://github.com/MrFireDeN/pothole-detector/blob/main/pothole_detector/yolov8n_custom/val_batch0_pred.jpg) |

### 📈 Графики обучения 
![Графики обучения](https://github.com/MrFireDeN/pothole-detector/blob/main/pothole_detector/yolov8n_custom/results.png)

## 🎬 Пример работы на видео
https://github.com/user-attachments/assets/ef48b0a6-12d8-4ba0-aea2-3ef16f0095bb

## 🛠️ Использование

1. Скачайте модель из [релизов](https://github.com/MrFireDeN/pothole-detector/releases) (`pothole_detector.pt`)
2. Для инференса:

```python
from ultralytics import YOLO

model = YOLO('pothole_detector.pt')
results = model.predict('input.jpg', conf=0.5)
```

Или через командную строку:

```python
yolo detect predict model=pothole_detector.pt source=video.mp4
```

## Авторы

- [**Свиридов Денис**](https://github.com/MrFireDeN) - Обучение модели
- [**Подмосковнов Илья**](https://github.com/rokosvlg) - Создание датасета для обучения
