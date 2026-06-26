<div align="center">

# Find a Photo of Your Child 🧒

**Автоматический поиск фотографий по лицу** — находит снимки с конкретным человеком среди тысяч фото с праздников, утренников и мероприятий.

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![InsightFace](https://img.shields.io/badge/InsightFace-0.7+-red?style=for-the-badge)](https://github.com/deepinsight/insightface)
[![OpenCV](https://img.shields.io/badge/OpenCV-YuNet-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## ✨ Возможности

| Режим | Описание |
|-------|----------|
| **🔍 all** | Находит **все** фотографии с любыми лицами |
| **🎯 reference** | Находит только фото, где лицо **совпадает с эталоном** (ваш ребёнок) |
| ⚡ **Многопоточность** | Ускорение за счёт параллельной обработки |
| 🖼️ **Форматы** | JPG, JPEG, PNG, HEIC, WEBP |
| 📄 **Отчёт** | Текстовый файл с результатами по каждому файлу |
| 📁 **Копирование** | Найденные фото сохраняются в отдельную папку |

---

## 🧠 Как это работает

```
Эталонное фото          Все фото в папках
     │                        │
     ▼                        ▼
  InsightFace              InsightFace
  (embedding)              (detect faces)
     │                        │
     └────────┬───────────────┘
              ▼
      Сравнение embedding'ов
         (cosine distance)
              │
              ▼
      Совпадает? → Копия в DESTINATION_DIR
```

- **Режим reference** — InsightFace извлекает embedding лица из эталонного фото и сравнивает с лицами на всех снимках
- **Режим all** — OpenCV (YuNet) находит все лица на фото
- Изображения масштабируются до 1200px по длинной стороне для ускорения

---

## 🚀 Быстрый старт

```bash
# 1. Установка зависимостей
pip install insightface onnxruntime opencv-python-headless pillow tqdm

# 2. Подготовьте эталонные фото
# Положите фото ребёнка в папку reference/

# 3. Настройте пути в Findchild.py:
SEARCH_PATHS = ["/путь/к/фото"]
REFERENCE_DIR = "reference"
DESTINATION_DIR = "found_photos"

# 4. Запуск
python Findchild.py --mode reference    # поиск конкретного лица
python Findchild.py --mode all          # поиск всех лиц
```

---

## 📁 Структура

```
Find-a-photo-of-your-child/
├── Findchild.py          # Главный скрипт
├── README.md
├── reference/            # 📥 Сюда положить эталонные фото
└── found_photos/         # 📤 Сюда сохраняются результаты
```

---

## 📊 Пример вывода

```
=== FACE DETECTION REPORT ===
Total files processed: 3420
Files with faces found: 156
Files with matching face: 23
Copied to destination: 23

Time elapsed: 145.3 seconds
```

---

## 💡 Зачем это нужно

> У меня накопились сотни фотографий с детских утренников, где мой ребёнок в толпе других детей. Я устал вручную просматривать все снимки.
>
> Этот скрипт за пару минут нашёл и скопировал только те фото, где есть мой сын. Экономия времени — колоссальная.

---

## ⚠️ Замечания

- Не проверено на Windows (только macOS/Linux)
- Для HEIC/WEBP требуется Pillow с поддержкой форматов
- Режим `reference` требует `insightface` (автоматически скачивает модель при первом запуске)
- На "чистой" системе могут потребоваться дополнительные системные библиотеки

---

<div align="center">

**Made with ❤️ for parents**

[Report Bug](https://github.com/KrivchenkoEgor/Find-a-photo-of-your-child/issues) · [Request Feature](https://github.com/KrivchenkoEgor/Find-a-photo-of-your-child/issues)

</div>
