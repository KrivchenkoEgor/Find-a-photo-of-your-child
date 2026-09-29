# -*- mode: python ; coding: utf-8 -*-
"""FindChild.spec — из чего собирается приложение.

Файл, а не длинная команда в терминале, по той же причине, по которой тесты лежат в
репозитории: сборку нужно уметь повторить буквально. Команда из истории shell теряется,
спецификация остаётся, и в ней видно, какие веса и какие плагины попали в архив.

Запуск:

    .venv/bin/python -m PyInstaller FindChild.spec

Что попадает внутрь и почему

* **YuNet (230 КБ)** — один файл OpenCV. Без него не работает режим «все лица», а
  скачать его приложение не может: это не кэш, а часть сборки.
* **`buffalo_l` (191 МБ, два файла из пяти)** — детектор `det_10g.onnx` и распознаватель
  `w600k_r50.onnx`. Три остальных файла пакета (3D- и 2D-ориентиры, пол/возраст)
  приложению не нужны: `InsightEngine` просит `allowed_modules=['detection',
  'recognition']`. Остальные 280 МБ — это вес, который человек носит в кармане зря.
* **`assets/samples/test_face.jpg` не упаковывается**: он нужен тестам и калибровке,
  а не работе окна.

Веса берутся из кэша insightface (`~/.insightface/models/buffalo_l`), а не из
репозитория: 191 МБ чужих бинарников в git класть нельзя, и на этой машине они уже
есть. Поэтому сборка на новом компьютере сначала требует одного запуска приложения
из исходников (оно докачает веса) — либо `FINDCHILD_BUFFALO_L=/путь/к/buffalo_l`.

Спецификация падает, если весов нет: молча собрать приложение без распознавания —
значит отдать человеку архив, в котором половина кнопок не работает, и узнать об этом
он должен не дома, а на чужом Маке.
"""

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs, copy_metadata

# SPECPATH — каталог этого файла, который PyInstaller подставляет сам. Относительные
# пути в datas иначе зависели бы от того, из какой папки запущен сборщик.
KOREN = Path(SPECPATH)

# Минимальная версия macOS у собранного Python: измеряется `otool -l` по бинарнику
# интерпретатора (см. LESSONS). Занижать её — обещать запуск там, где он не случится.
MIN_MACOS = os.environ.get("FINDCHILD_MIN_MACOS", "26.0")

# ----------------------------------------------------------------- веса распознавания

VESI_BUFFALO = Path(
    os.environ.get("FINDCHILD_BUFFALO_L",
                   str(Path.home() / ".insightface" / "models" / "buffalo_l"))
)
NUZHNYE_VESA = ("det_10g.onnx", "w600k_r50.onnx")

otsutstvuyut = [f for f in NUZHNYE_VESA if not (VESI_BUFFALO / f).is_file()]
if otsutstvuyut:
    raise SystemExit(
        "Сборка остановлена: внутри архива не будет весов распознавания.\n"
        f"  ждали файлы {', '.join(otsutstvuyut)}\n"
        f"  в папке     {VESI_BUFFALO}\n"
        "  запустите один раз приложение из исходников (оно докачает buffalo_l) или\n"
        "  укажите свою папку: FINDCHILD_BUFFALO_L=/путь/к/buffalo_l")

# ----------------------------------------------------------------- что кладём в архив

DATY = [
    (str(KOREN / "assets" / "models" / "face_detection_yunet_2023mar.onnx"),
     "assets/models"),
    *[(str(VESI_BUFFALO / f), "models/buffalo_l") for f in NUZHNYE_VESA],
]

# Метаданные Pillow нужны `PIL.Image` для реестра плагинов, а pillow-heif регистрирует
# открыватель HEIC именно через Pillow. Без них на чужом Маке .heic не откроется, и
# выглядит это как «приложение не видит фото с iPhone».
DATY += copy_metadata("Pillow")

BINARIKI = [
    # onnxruntime держит свои .dylib в подкаталоге `_deps`, и обычная графа зависимостей
    # их не видит: без явного сбора на чужой машине падает импорт распознавателя.
    *collect_dynamic_libs("onnxruntime"),
]

# Импорт `insightface` и `_pillow_heif` живут внутри функций и в корне site-packages,
# статический анализ их пропускает.
SKRYTYE_IMPORTY = ["insightface", "_pillow_heif", "pillow_heif"]

# Что не тащим сознательно: эти пакеты приложение не импортирует, а hooks-цепочки
# тянут их за собой сотнями мегабайт.
OTSEK = [
    "tkinter", "matplotlib", "IPython", "pytest", "setuptools", "pip",
    "PySide6.QtWebEngineCore", "PySide6.QtQml", "PySide6.QtQuick",
    "PySide6.QtMultimedia", "PySide6.Qt3DCore", "PySide6.QtCharts",
    "PySide6.QtDataVisualization", "PySide6.QtGraphs", "PySide6.QtPdf",
    "PySide6.QtPositioning", "PySide6.QtRemoteObjects", "PySide6.QtScxml",
    "PySide6.QtSensors", "PySide6.QtSerialPort", "PySide6.QtTextToSpeech",
    "PySide6.QtWebChannel", "PySide6.QtWebSockets", "PySide6.QtBluetooth",
    "PySide6.QtNfc", "PySide6.QtHelp", "PySide6.QtDesigner", "PySide6.QtUiTools",
]

IKONKA = KOREN / "assets" / "icon.icns"
if not IKONKA.is_file():
    # `BUNDLE(icon=None)` не собирает: PyInstaller вызывает os.path.exists(None) и
    # падает внятным только для него «Icon input file None not found».
    raise SystemExit(
        f"Сборка остановлена: нет иконки {IKONKA}.\n"
        "  собирается из PNG набором `iconutil`: см. раздел «Сборка» в README.")

a = Analysis(
    [str(KOREN / "src" / "main.py")],
    pathex=[str(KOREN / "src")],
    binaries=BINARIKI,
    datas=DATY,
    hiddenimports=SKRYTYE_IMPORTY,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(KOREN / "hooks" / "runtime_hook_cv2.py")],
    excludes=OTSEK,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,               # один каталог: тяжёлое лежит рядом, а не в exe
    name="FindChild",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,                       # окно без терминала: человек запускает двойным кликом
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,                    # архитектура этого Mac: arm64
    codesign_identity=None,              # ad-hoc подпись по умолчанию
    entitlements_file=None,
)

coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="FindChild")

app = BUNDLE(
    coll,
    name="FindChild.app",
    icon=str(IKONKA) if IKONKA.is_file() else None,
    bundle_identifier="local.FindChild",
    version="1.0",
    info_plist={
        # Имя в строке меню: короткое, с пробелами, без «.app». Полное название видно
        # в Finder и в переключателе окон — за него отвечает DisplayName.
        "CFBundleName": "Поиск фото",
        "CFBundleDisplayName": "Поиск фото ребёнка",
        "CFBundleVersion": "1.0",
        "LSMinimumSystemVersion": MIN_MACOS,
        "NSHumanReadableCopyright": "Локальная обработка фото. Ни один снимок "
                                    "не покидает этот компьютер.",
    },
)
