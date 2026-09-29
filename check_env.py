"""Проверка окружения: не «установилось», а «работоспособно».

Запуск:  .venv/bin/python check_env.py
"""

import sys
import traceback

RESULTS = []


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append(("OK", name, detail))
    except Exception as exc:
        RESULTS.append(("FAIL", name, f"{type(exc).__name__}: {exc}"))
        traceback.print_exc(limit=3)


def v_mod():
    import numpy
    import cv2
    return f"numpy {numpy.__version__} / cv2 {cv2.__version__}"


def v_pyside():
    from PySide6 import QtCore, QtGui, QtWidgets
    return f"Qt {QtCore.__version__}"


def v_offscreen_widget():
    """Qt должен собирать виджеты без экрана (offscreen) — это проверка для PyInstaller-сборки."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    label = QtWidgets.QLabel("тест")
    label.setText("ок")
    return f"QLabel текст={label.text()!r}, platform={app.platformName()}"


def v_insightface_api():
    import insightface
    from insightface.app import FaceAnalysis
    import inspect
    sig = str(inspect.signature(FaceAnalysis.__init__))
    return f"insightface {getattr(insightface, '__version__', '?')}, FaceAnalysis{sig}"


def v_yunet_api():
    import cv2
    if not hasattr(cv2, "FaceDetectorYN_create"):
        raise AttributeError("в cv2 нет FaceDetectorYN_create — YuNet недоступен")
    return "cv2.FaceDetectorYN_create присутствует"


def v_onnx_providers():
    import onnxruntime as ort
    return f"onnxruntime {ort.__version__}, providers={ort.get_available_providers()}"


def v_heif_roundtrip():
    """Пишем и читаем HEIF — проверяем, что iPhone-формат действительно открывается."""
    import io
    from PIL import Image
    from pillow_heif import register_heif_opener
    register_heif_opener()
    img = Image.new("RGB", (64, 48), (120, 200, 90))
    buf = io.BytesIO()
    img.save(buf, format="HEIF")
    buf.seek(0)
    back = Image.open(buf)
    back.load()
    return f"{back.format} {back.size} режим={back.mode}"


def v_numpy2_compat():
    """numpy 2.x ломает старые сборки cv2 — проверяем совместимость на живом массиве."""
    import numpy as np
    import cv2
    arr = (np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (32, 32))
    return f"cvtColor+resize ok, форма {resized.shape}"


def v_python():
    return sys.version.split()[0]


def v_disk_and_paths():
    import shutil
    from pathlib import Path
    free_gb = shutil.disk_usage(Path.cwd()).free / 1e9
    return f"свободно {free_gb:.1f} ГБ, venv={Path(sys.executable).parent}"


check("Python", v_python)
check("диск/venv", v_disk_and_paths)
check("numpy + cv2 импорты", v_mod)
check("cv2 работает с numpy 2", v_numpy2_compat)
check("PySide6 импорты", v_pyside)
check("PySide6 offscreen-виджет", v_offscreen_widget)
check("insightface API", v_insightface_api)
check("YuNet API", v_yunet_api)
check("onnxruntime провайдеры", v_onnx_providers)
check("HEIF запись/чтение", v_heif_roundtrip)

print("\n================ ИТОГ ================")
fails = 0
for status, name, detail in RESULTS:
    print(f"[{status:4}] {name:28} {detail}")
    fails += status == "FAIL"
print(f"======================================")
print(f"провалов: {fails}")
sys.exit(1 if fails else 0)
