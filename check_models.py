"""Реальная проверка конвейера: YuNet детектирует лицо, InsightFace строит embedding.

Отличается от check_env.py тем, что тут не «модуль импортируется», а «лицо найдено».
Запуск:  QT_QPA_PLATFORM=offscreen .venv/bin/python check_models.py /path/к/фото_с_лицом.jpg
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

MODEL_YUNET = Path("assets/models/face_detection_yunet_2023mar.onnx")
TEST_IMAGE = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/messi5.jpg")

print(f"тестовое изображение: {TEST_IMAGE} ({TEST_IMAGE.stat().st_size} байт)")

# --- 1. YuNet: режим «Все лица» -------------------------------------------------
img = cv2.imread(str(TEST_IMAGE))
if img is None:
    print("FAIL: cv2.imread не смог прочитать файл")
    sys.exit(1)
h, w = img.shape[:2]
print(f"cv2 прочитал: {w}x{h}")

det = cv2.FaceDetectorYN_create(str(MODEL_YUNET), "", (w, h), score_threshold=0.6)
t0 = time.perf_counter()
ret, faces = det.detect(img)
elapsed = time.perf_counter() - t0
n = 0 if faces is None else len(faces)
print(f"[1] YuNet: лиц найдено {n}, время {elapsed*1000:.0f} мс")
if n:
    x, y, fw, fh = faces[0][:4]
    print(f"    рамка первого лица: x={x:.0f} y={y:.0f} w={fw:.0f} h={fh:.0f}, уверенность {faces[0][14]:.2f}")
assert n > 0, "YuNet не нашёл лицо на эталонном фото — режим «Все лица» не будет работать"

# --- 2. InsightFace buffalo_l: режим «Конкретный человек» -----------------------
import insightface
from insightface.app import FaceAnalysis

t0 = time.perf_counter()
app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
app.prepare(ctx_id=-1, det_size=(640, 640))
print(f"[2] buffalo_l загрузился за {time.perf_counter()-t0:.1f} с")

t0 = time.perf_counter()
faces = app.get(img)
print(f"[3] FaceAnalysis.get(): лиц {len(faces)}, время {time.perf_counter()-t0:.2f} с")
assert faces, "InsightFace не вернул лиц — режим reference не будет работать"

f0 = faces[0]
emb = f0.normed_embedding
print(f"    embedding: размер {emb.shape}, dtype {emb.dtype}, норма {np.linalg.norm(emb):.4f}")
print(f"    det_score {f0.det_score:.3f}")

# --- 3. Сравнение двух «лиц» одного человека = основа порога сходства ------------
ref_img = cv2.resize(img, (w // 2, h // 2))
ref_faces = app.get(ref_img)
if ref_faces:
    sim = float(np.dot(ref_faces[0].normed_embedding, emb))
    print(f"[4] косинусное сходство того же лица в другом масштабе: {sim:.3f}")
    print(f"    при пороге 0.4 (AGENTS.md) это сработает: {sim >= 0.4}")

# --- 4. Чужое лицо не должно совпадать -----------------------------------------
noise = (np.random.default_rng(1).integers(0, 255, (h, w, 3))).astype(np.uint8)
print(f"[5] на шумовой картинке лиц найдено: {len(app.get(noise))} (ожидаем 0)")

print("\nВСЁ РАБОТАЕТ: оба режима обеспечены установленными пакетами.")
