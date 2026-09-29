"""Оглавление отпечатков лиц. Тяжёлый разбор папки происходит один раз.

Храним только координаты и отпечатки, сами фото не копируем.

Ключ записи — (путь, наносекунды изменения, размер, движок, разрешение). Смена режима
детекции или разрешения скана сама обнуляет оглавление: старые отпечатки не могут
молча подставиться вместо новых, и пользователь получает честный пересчёт.

Формат содержимого помечен версией (`PAYLOAD_VERSION`): файл оглавления живёт у
пользователя долго и переживёт не одну версию приложения. Версия не та — это промах,
и фото разберётся заново. Так будущее изменение `to_dict` делает старые строки
неактуальными, а не падающими посреди разбора. Повреждённую строку `get` ещё и удаляет:
оглавление восстанавливается с нуля, и висеть мусором оно не должно.

Кроме лиц строка помнит, сколько отпечатков было отброшено на этом снимке
(`"dropped"`). Это не справка для отладки, а единственная память о потерях, которая
переживает запуск: отбраковку считает разбор, а разбор снимка из оглавления не делает
вовсе. Без этого поля тёплый прогон врал бы отчёту — «потерь нет» про архив, в котором
лица терялись на первом, холодном прогоне (см. `worker.ScanStats.dropped_faces`).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Sequence

from .engine import Face

# Версия формата отпечатка. Меняется при любом изменении to_dict/from_dict:
# старая запись тогда не будет прочитана криво — она просто не найдётся и фото
# разберётся заново.
#   1 — только список лиц;
#   2 — добавлено число отброшенных отпечатков (`dropped`): без него тёплый прогон
#       не может узнать о потерях прошлого запуска и печатает «потерь нет».
PAYLOAD_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS faces (
    path     TEXT    NOT NULL,
    mtime    INTEGER NOT NULL,
    size     INTEGER NOT NULL,
    engine   TEXT    NOT NULL,
    max_dim  INTEGER NOT NULL,
    payload  TEXT    NOT NULL,
    PRIMARY KEY (path, mtime, size, engine, max_dim)
)
"""


def _celoe(znachenie: object) -> int:
    """Целое из чужой строки JSON: не целое, не число, NaN или `true` — это 0.

    Поле `dropped` читается из файла оглавления, который пережил не одну версию
    приложения и правку руками. Его испорченное значение не имеет права ни уронить
    разбор, ни превратиться в `int(True) == 1` — это справка для отчёта, а не данные
    сравнения, и ответ «потерь не видно» здесь честнее исключения.
    """
    if isinstance(znachenie, bool) or not isinstance(znachenie, (int, float)):
        return 0
    try:
        velichina = int(znachenie)              # NaN и Inf бросают здесь, а не в отчёте
    except (TypeError, ValueError, OverflowError):
        return 0
    return velichina if velichina > 0 else 0


class FaceCache:
    """SQLite-оглавление: список лиц по каждому фото и каждой комбинации настроек.

    Работа в несколько потоков. Оглавление открывает поток интерфейса, а читает и пишет
    рабочий QThread (задача 9): без `check_same_thread=False` sqlite3 на первом же чужом
    потоке бросает ProgrammingError. Каждая операция спрятана под `self._lock` — по
    замеру без блокировки 6 потоков дали 5 падений `SystemError: error return without
    exception set` и потеряли часть записей.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._lock = threading.Lock()
        # Сколько строк выбросили как повреждённые. Для `get` битая строка и отсутствие
        # записи — одно и то же (`None`), а разбору папки этого мало: отчёт обязан
        # отличать «этого снимка ещё не было в оглавлении» от «оглавление
        # самопочинилось, строку пришлось удалить». Иначе архив, где половина строк
        # записана старой версией с NaN-отпечатками, выглядит как «всё хорошо», хотя
        # пересчитывается моделями заново на каждый прогон.
        self.damage_count = 0
        with self._lock:
            self._db.execute(_SCHEMA)
            self._db.commit()

    def _key(self, photo: Path, stamp: tuple[int, int], engine: str,
             max_dim: int) -> tuple[str, int, int, str, int]:
        # int(...) держит ключ целым: наносекундный штамп — это ~1,8·10^18, а точные
        # целые у дробного числа только до 2^53 (~9·10^15). Утонуло бы в округлении —
        # два разных снимка получили бы одинаковый штамп и чужие отпечатки.
        return (str(Path(photo)), int(stamp[0]), int(stamp[1]), engine, int(max_dim))

    def get(self, photo: Path, stamp: tuple[int, int], engine: str,
            max_dim: int) -> list[Face] | None:
        """Список лиц, если он есть для этих же настроек; None — сканировать заново.

        Обёртка над `get_s_poterjami` для тех, кому лица нужны сами по себе. Разбор
        папки берёт вторую функцию: ему вместе с лицами положено помнить, сколько
        отпечатков отбросил снимок на прошлом прогоне.
        """
        hit = self.get_s_poterjami(photo, stamp, engine, max_dim)
        return None if hit is None else hit[0]

    def get_s_poterjami(self, photo: Path, stamp: tuple[int, int], engine: str,
                        max_dim: int) -> tuple[list[Face], int] | None:
        """Лица и число отброшенных на этом снимке отпечатков; None — записи нет.

        Пустой список — это попадание: фото без лиц тоже лежит в оглавлении и второй
        раз не читается. Проверять нужно `is not None`, а не истинность.

        Битая или устаревшая по формату запись — это тоже None. Оглавление восстанавливается
        с нуля, поэтому ронять из-за одной строки разбор всей папки нельзя: без этой
        обработки `json.loads` и `Face.from_dict` бросают исключение прямо в рабочий
        поток, и прогон обрывается с непереведённым traceback'ом. Такую запись заодно
        удаляем, чтобы она не висела мусором до ближайшей очистки.

        Число потерь в старой строке может и не оказаться (его добавили версией 2), и
        испорченным оно быть не обязано: это справка для отчёта, а не данные сравнения.
        Поэтому за отсутствующее или нечисловое значение отвечаем нулём, а строку
        выбрасываем только когда под угрозой сами лица.
        """
        key = self._key(photo, stamp, engine, max_dim)
        with self._lock:
            row = self._db.execute(
                "SELECT payload FROM faces"
                " WHERE path=? AND mtime=? AND size=? AND engine=? AND max_dim=?",
                key,
            ).fetchone()
        if row is None:
            return None
        try:
            data = json.loads(row[0])
            if data.get("v") != PAYLOAD_VERSION:
                return None
            lica = [Face.from_dict(d) for d in data["faces"]]
        except (ValueError, KeyError, TypeError, AttributeError):
            # ValueError закрывает сразу три беды: битый JSON (JSONDecodeError — его
            # подкласс), битый base64 (binascii.Error — тоже его подкласс) и отпечаток
            # неверной длины. TypeError — отпечаток не-строкой, KeyError — нет нужного
            # поля, AttributeError — строка старого формата, где лежал голый список лиц.
            with self._lock:
                # Заодно счётчик: без него «строка была и исчезла» неотличимо от
                # «строки не было», и отчёт молча потерял бы пересчёт половины архива.
                self.damage_count += 1
                self._db.execute(
                    "DELETE FROM faces"
                    " WHERE path=? AND mtime=? AND size=? AND engine=? AND max_dim=?",
                    key,
                )
                self._db.commit()
            return None
        return lica, _celoe(data.get("dropped"))

    def put(self, photo: Path, stamp: tuple[int, int], engine: str, max_dim: int,
            faces: Sequence[Face], dropped: int = 0) -> None:
        """Записать лица по этому ключу; повторная запись того же ключа перезаписывает.

        `dropped` — сколько отпечатков на этом снимке отбросил разбор как непригодные.
        Записывается всегда, даже нулём: ноль в строке — это «мы смотрели и ничего не
        потеряли», а отсутствие поля — «мы не знаем», и путать их нельзя.
        """
        # Конверт с версией формата и плотными разделителями JSON: метка остаётся
        # байт-в-байт `"v":2`, чтобы её можно было найти в файле простым запросом,
        # а пробелы после двоеточий лишь добавляют ~12 байт на лицо.
        payload = json.dumps({"v": PAYLOAD_VERSION,
                              "faces": [f.to_dict() for f in faces],
                              "dropped": _celoe(dropped)},
                             ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO faces (path, mtime, size, engine, max_dim, payload)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (*self._key(photo, stamp, engine, max_dim), payload),
            )
            self._db.commit()

    def clear(self) -> int:
        """Удалить всё оглавление; возвращает число стёртых записей.

        Запись — это один снимок при одном наборе настроек, а не одно лицо: в подсказке
        интерфейса («очищено N») это число фото, а не число найденных лиц.
        """
        with self._lock:
            n = self._db.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
            self._db.execute("DELETE FROM faces")
            self._db.commit()
        return n

    def close(self) -> None:
        with self._lock:
            self._db.close()
