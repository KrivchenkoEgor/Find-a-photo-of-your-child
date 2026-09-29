"""Поиск фотофайлов в папках. Сами изображения здесь не читаются — этим занимается
`load_photo` в этом же модуле, когда приложение попросит разбор."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()   # без этого Pillow не открывает .heic

PHOTO_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"})
# Служебные каталоги без точки в начале имени. Всё, что начинается с точки
# (.Trash, .Trash-501, .thumbnails, .qtsmedia), отсекает проверка на точку —
# держать их здесь отдельными строками значит создать видимость чёрного списка,
# который никогда не срабатывает: сравнение через `in` глоб не понимает.
SKIP_DIR_NAMES = frozenset({"__pycache__"})


def _skip_dir(part: str) -> bool:
    """Служебный каталог: имя из чёрного списка или начинающееся с точки."""
    return part in SKIP_DIR_NAMES or part.startswith(".")


def find_photos(dirs: Iterable[Path]) -> list[Path]:
    """Все фото во всех папках, рекурсивно, в стабильном порядке.

    Пропускает скрытые и служебные каталоги: в корзине и в кэшах macOS искать нечего.
    Порядок — сначала полный путь внутри каждой папки, затем порядок самих папок:
    «Фото», потом «Камера», потом «Отпуск» читается человеком, а алфавит папок — нет.
    Глобальной сортировки здесь нет нарочно (см. `test_poryadok_pri_neskolkih_papkah_govorit_pravdu`).

    Каждая папка обходится сама по себе, поэтому вложенная друг в друга пара вернёт
    общие файлы дважды. Окно это переживает: `set_source_folders` снимает дубли через
    `dict.fromkeys`, и ровно для этого — а не для красоты — порядок возврата тут
    стабилен.
    """
    found: list[Path] = []
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for path in sorted(d.rglob("*")):      # порядок по полному пути — детерминирован
            if not path.is_file():
                continue
            if path.suffix.lower() not in PHOTO_SUFFIXES:
                continue
            if any(_skip_dir(part) for part in path.relative_to(d).parts[:-1]):
                continue
            if path.name.startswith("."):
                continue
            found.append(path)
    return found


def file_stamp(path: Path) -> tuple[int, int]:
    """Пара (время изменения, размер) — ключ, по которому кэш понимает, что фото изменилось.

    Наносекунды, а не секунды: `st_mtime` округляется вниз, и файл, перезаписанный
    в ту же секунду без смены размера, сохранил бы устаревшие отпечатки молча.
    """
    st = Path(path).stat()
    return int(st.st_mtime_ns), int(st.st_size)


def _read_with_pillow(path: Path) -> Image.Image:
    """Читает через Pillow, а не cv2.imread: OpenCV не декодирует HEIC — основной формат iPhone.

    Ориентацию из EXIF свежий OpenCV применяет и сам, но лишь пока вызывающий не передал
    флаг IMREAD_IGNORE_ORIENTATION; полагаться на это нельзя, поэтому exif_transpose здесь.
    """
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def _to_bgr(img: Image.Image) -> np.ndarray:
    """Pillow отдаёт RGB, а OpenCV (YuNet, norm_crop) ждут BGR — меняем порядок каналов."""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def load_photo(path: Path, max_dim: int = 2400) -> np.ndarray | None:
    """Читает фото, поворачивает по EXIF и уменьшает до max_dim. None — если не читается.

    Уменьшение до 1200 px (наследие Findchild.py) стоит 3–7 процентных пунктов сходства:
    лицо 110 px превращается в 22 px. Поэтому по умолчанию 2400.
    """
    try:
        img = _read_with_pillow(Path(path))
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            # round, а не int: иначе длинная сторона может стать max_dim - 1.
            # max(1, ...): у снимка 4900x1 короткая сторона обнулилась бы, и Pillow
            # бросил бы ValueError наружу — а обещание модуля «верни None, не падай».
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                             Image.LANCZOS)
        return _to_bgr(img)
    except Exception:
        return None
