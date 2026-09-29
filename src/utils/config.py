"""Пути и сохранённые настройки.

Служебная папка — не рядом с фото: в семейном архиве не должно быть наших файлов.

Здесь собраны три разных дома приложения, и путать их нельзя:

| что                    | где                                             | кто чинит |
|------------------------|-------------------------------------------------|-----------|
| настройки, оглавление  | `app_dir()` — служебная папка пользователя      | приложение |
| `buffalo_l` (191 МБ)   | `buffalo_l_dir()` — внутри приложения, а если   | пересборка |
|                        | копии нет — кэш insightface в `~/.insightface`  | или человек |
| модель YuNet (230 КБ)  | `yunet_model_path()` — внутри самого приложения | только пересборка |

Именно поэтому «путь к моделям» — это два пути, а не один, и настройкой они не
становятся: в полосе настроек их нет. Оба называет прямо сообщение рабочего потока
(см. `worker.ModeliNeZagruzilis`) — там, где человек их и увидит. Свести их к одному
пути значило бы отправить человека с недостающим YuNet в `.insightface`, где лежит
только пакет распознавания.

Геттеры путей ничего не создают: на вопрос «куда писать» не должно появляться каталогов
— ни у реального пользователя, ни в тестовой песочнице. Создают папки пишущие: `Qt`
пишет `settings.ini` сам, `FaceCache` делает `mkdir` для оглавления.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSettings

APP_NAME = "FindChild"
DEFAULT_THRESHOLD = 0.38
DEFAULT_ENGINE = "both"
DEFAULT_MAX_DIM = 2400
THRESHOLD_RANGE = (0.30, 0.60)

# Ключи режимов — те же, что принимает фабрика `core.engine.make_engine`. Список
# держим здесь, потому что проверяется значение при чтении файла: строка, которую
# человек дописал руками, не должна доходить до фабрики и ронять разбор ValueError.
ENGINE_KEYS = ("both", "yunet", "insight")

# Разумные границы длинной стороны снимка, px. Меньше 600 лицо ребёнка на групповом
# фото перестаёт находиться, больше 6000 — это уже описка в файле настроек.
MAX_DIM_RANGE = (600, 6000)


def _dom() -> Path:
    """Домашняя папка пользователя. Отдельно, чтобы тесты подменяли её через HOME."""
    return Path.home()


def app_dir() -> Path:
    """Служебная папка приложения: macOS `~/Library/Application Support/FindChild`,
    Windows `%APPDATA%\\FindChild`, Linux `~/.local/share/FindChild` (или `$XDG_DATA_HOME`).

    Никакого реестра и никаких файлов внутри архива с фото: настройки и оглавление
    живут в одном предсказуемом месте, которое человек может удалить руками, если
    захочет начать с чистого листа. Кнопка «Очистить оглавление» чистит базу, а путь
    сюда показывает строка состояния окна: «оглавление очищено: удалено записей N».
    """
    if sys.platform == "darwin":
        base = _dom() / "Library" / "Application Support"
    elif sys.platform == "win32":
        # %APPDATA% нет при служебном входе и в портативном запуске: тогда
        # %USERPROFILE%\\AppData\\Roaming — то самое место, куда его подставил бы Windows.
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else _dom() / "AppData" / "Roaming"
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        base = Path(xdg) if xdg else _dom() / ".local" / "share"
    return base / APP_NAME


def cache_file() -> Path:
    """Оглавление отпечатков: переживает перезапуск и перечитывает папку только заново."""
    return app_dir() / "cache.sqlite3"


def settings_file() -> Path:
    """Файл сохранённых настроек. Отдельный геттер — чтобы показать путь в интерфейсе."""
    return app_dir() / "settings.ini"


def models_dir() -> Path:
    """Корень кэша insightface: здесь лежит распакованный `buffalo_l`.

    Повторяет правило самой библиотеки (`root='~/.insightface'`, подкаталог `models`)
    без импорта: тестам этот модуль доступен без insightface и без скачанных весов, а
    правило не менялось между версиями. Папки может не быть — это ровно тот случай,
    когда разбор не начинается и человек получает два настоящих пути в тексте ошибки
    (`worker.ModeliNeZagruzilis`), поэтому здесь только читают, но не создают.
    """
    return _dom() / ".insightface" / "models"


def buffalo_l_dir() -> Path:
    """Каталог весов распознавания: сначала внутри собранного приложения, потом кэш.

    Два дома у одного файла не придирка, а следствие того, что `.app` переносимый: на
    чужом Маке `~/.insightface` нет, и без копии внутри архива приложение попросило бы
    интернет и 290 МБ. Поэтому правило такое — если в архиве лежит `models/buffalo_l`,
    веса берутся оттуда; иначе (режим разработки или сборка без весов) — прежний
    кэш insightface.

    Именно существование этой папки и есть ответ на «модели не загрузились»: её путь
    человек читает в тексте сбоя, который печатает `worker.ModeliNeZagruzilis`.
    """
    v_arhive = _koren_reshursov() / "models" / "buffalo_l"
    return v_arhive if v_arhive.is_dir() else models_dir() / "buffalo_l"


def _koren_reshursov() -> Path:
    """Где лежат ресурсы, едущие внутри приложения.

    В собранном `.app` их раскрывает PyInstaller (`sys._MEIPASS`), в исходниках это
    корень проекта. Один владелец ответа нужен потому, что ресурсов теперь два — YuNet
    и веса распознавания, — и у двух мест с одним правилом путь разъезжается при
    первой же пересборке архива.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    return Path(meipass) if meipass else Path(__file__).resolve().parents[2]


def yunet_model_path() -> Path:
    """Модель YuNet: в собранном приложении — внутри архива PyInstaller.

    `sys._MEIPASS` ставит PyInstaller; в исходниках путь ведёт в `assets/models` рядом
    с проектом. YuNet весит 230 КБ и едет вместе с приложением, поэтому его дом — не
    кэш insightface (см. `models_dir`).
    """
    return _koren_reshursov() / "assets" / "models" / "face_detection_yunet_2023mar.onnx"


class Settings:
    """Настройки между запусками: ini-файл в служебной папке приложения.

    `QSettings` собирается конструктором «файл + IniFormat», а не «организация +
    приложение»: второй вариант Qt пишет в реестр Windows и в каталог пользователя, и
    тесты с подменой пути молча правили бы живые настройки человека. Путь входит в
    конструктор, поэтому проверять его можно простым файлом в `tmp_path`.

    Значения проверяются на чтении, а не на доверии. Файл переживает старую версию
    приложения и правку руками, а настройка, уронившая запуск, хуже отсутствующей:
    похожесть возвращается внутри `THRESHOLD_RANGE` (иначе число на экране и число,
    по которому режутся кучки, разойдутся), режим — только один из `ENGINE_KEYS`, размер
    снимка — из разумного диапазона.

    `path` — туда реально пишется: полоса настроек показывает его человеку, когда
    объясняет, что удалять при чистке.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._put = Path(path) if path is not None else settings_file()
        self._s = QSettings(str(self._put), QSettings.IniFormat)

    @property
    def path(self) -> Path:
        return self._put

    def _chislo(self, klyuch: str, default: float) -> float:
        """Число из файла; всё остальное — молча умолчалка.

        Ловим и `ValueError` («abc», пустая строка), и `TypeError`: Qt толкует
        «0,5» как список строк, а `float(['0', '5'])` — уже TypeError. Отдельно
        отбрасываем NaN и Inf: `float('nan')` не бросает ничего, но до первого
        `int(nan)` в размере снимка дело доходит до ValueError.
        """
        try:
            znachenie = float(self._s.value(klyuch, default))
        except (TypeError, ValueError):
            return default
        return znachenie if math.isfinite(znachenie) else default

    @property
    def threshold(self) -> float:
        niz, verh = THRESHOLD_RANGE
        return min(max(self._chislo("threshold", DEFAULT_THRESHOLD), niz), verh)

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._s.setValue("threshold", float(value))

    @property
    def engine(self) -> str:
        znachenie = self._s.value("engine", DEFAULT_ENGINE)
        stroka = "" if znachenie is None else str(znachenie)
        return stroka if stroka in ENGINE_KEYS else DEFAULT_ENGINE

    @engine.setter
    def engine(self, value: str) -> None:
        self._s.setValue("engine", str(value))

    @property
    def max_dim(self) -> int:
        znachenie = int(self._chislo("max_dim", float(DEFAULT_MAX_DIM)))
        niz, verh = MAX_DIM_RANGE
        # Зажимать нельзя: 0 и 42 — не «слишком мало», а испорченный файл. Здесь
        # честнее вернуться к проверенным 2400, чем молча начать искать по 600 px.
        return znachenie if niz <= znachenie <= verh else DEFAULT_MAX_DIM

    @max_dim.setter
    def max_dim(self, value: int) -> None:
        self._s.setValue("max_dim", int(value))

    def _stroka(self, klyuch: str) -> str:
        znachenie: Any = self._s.value(klyuch, "")
        return "" if znachenie is None else str(znachenie)

    @property
    def last_source(self) -> str:
        return self._stroka("last_source")

    @last_source.setter
    def last_source(self, value: str) -> None:
        self._s.setValue("last_source", str(value))

    @property
    def last_dest(self) -> str:
        return self._stroka("last_dest")

    @last_dest.setter
    def last_dest(self, value: str) -> None:
        self._s.setValue("last_dest", str(value))

    def save(self) -> bool:
        """Сбросить буфер на диск. Возвращает True, только если запись действительно удалась.

        Без `sync()` Qt дописывает файл на выходе из приложения, а у `QThread` с тяжёлым
        разбором этот выход может не наступить вовсе: человек выключает окно кнопкой.

        Статус спрашиваем сразу, и это не формальность: `sync()` не бросает исключений,
        а неудачу копит в `status()`. Без проверки `save()` возвращал бы None и тогда,
        когда на диск не легло ничего, — человек нажал бы «сохранить», увидел бы свой
        порог на экране и узнал о промахе только при следующем запуске, когда настройки
        вернулись бы к умолчалкам. `AccessError` сюда приходит с недоступной на запись
        папкой, со снятым томом и с заполненным диском; `FormatError` — если файл по
        пути подменили чужим. Вызывающий обязан сказать об этом человеку (задача 15:
        «не удалось сохранить настройки в …»), потому что молчаливое «сохранили» было бы
        обещанием, которое приложение не выполнило.
        """
        self._s.sync()
        return self._s.status() == QSettings.NoError
