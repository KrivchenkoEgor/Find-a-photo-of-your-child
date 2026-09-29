"""Чтение фото: ориентация из EXIF, уменьшение по длинной стороне, HEIC, битые файлы.

Картинки собираются синтетически через Pillow во временной папке теста: реальные снимки
из data/ и ref/ — фото конкретного ребёнка, они личные и вне git, в тестах не используются.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from core.scanner import load_photo


def _jpeg(path: Path, size: tuple[int, int] = (4000, 3000),
          color: tuple[int, int, int] = (10, 200, 30)) -> Path:
    """Сплошной RGB-снимок заданного размера.

    Цвет выбран застрахованным: пиксель сплошного JPEG переживает и сжатие, и LANCZOS
    без изменения (проверено), поэтому ассерты ниже можно писать точными числами.
    """
    Image.new("RGB", size, color).save(path, "JPEG")
    return path


def test_umenshaet_do_max_dim_po_dlinnoy_storone(tmp_path) -> None:
    p = _jpeg(tmp_path / "a.jpg")
    img = load_photo(p, max_dim=1200)
    assert max(img.shape[:2]) == 1200
    assert img.shape[2] == 3
    # разрешение по умолчанию — 2400, а не 1200 из старого Findchild.py
    assert max(load_photo(p).shape[:2]) == 2400


def test_ne_umenshaet_esli_foto_i_tak_malenkoe(tmp_path) -> None:
    p = _jpeg(tmp_path / "b.jpg", size=(600, 400))
    img = load_photo(p, max_dim=2400)
    assert img.shape[:2] == (400, 600)


def test_povorachivaet_po_exif(tmp_path) -> None:
    """Снимок iPhone лежит повёрнутым, настоящая ориентация — только в теге EXIF.

    Пиксели в файле — альбомные (400×200), а тег говорит «повернуть на 90° по часовой».
    Тест падает, если из пайплайна убрать ImageOps.exif_transpose: на выходе окажется
    200 строк и 400 колонок вместо обратных.

    Оговорка: OpenCV 5.0 применяет ориентацию JPEG/PNG/WEBP и сам, поэтому один этот тест
    возврат на cv2.imread не ловит — от него бережёт test_chitaet_heic (HEIF OpenCV не
    декодирует вовсе).
    """
    p = tmp_path / "rot.jpg"
    base = Image.new("RGB", (400, 200), (255, 0, 0))
    exif = Image.Exif()
    exif[0x0112] = 6                      # Orientation = 6 (90° по часовой)
    base.save(p, "JPEG", exif=exif)
    img = load_photo(p, max_dim=2400)
    assert img.shape[0] > img.shape[1]    # после поворота высота больше ширины


def test_vozvrashchaet_bgr_uint8(tmp_path) -> None:
    """OpenCV и norm_crop ждут BGR uint8, а Pillow отдаёт RGB — порядок каналов важен."""
    p = _jpeg(tmp_path / "c.jpg", size=(600, 400))
    img = load_photo(p, max_dim=2400)
    assert img is not None
    assert img.dtype == np.uint8
    assert img[200, 300].tolist() == [30, 200, 10]   # RGB(10, 200, 30) → каналы в порядке BGR


def test_chitaet_heic(tmp_path) -> None:
    """HEIC — формат iPhone по умолчанию: не откроешь его, и свежие снимки не читаются.

    Это же страховка от возврата на cv2.imread: OpenCV HEIF не декодирует и вернул бы
    None на каждом фото с iPhone. Зелёным тест держат два условия — вызов
    register_heif_opener() и чтение именно через Pillow.
    """
    p = tmp_path / "iphone.heic"
    Image.new("RGB", (800, 600), (10, 200, 30)).save(p, "HEIF")
    img = load_photo(p)
    assert img is not None
    assert img.shape[:2] == (600, 800)    # (h, w), уменьшение не нужно: обе стороны < 2400


def test_nechitaemyy_fail_dast_None(tmp_path) -> None:
    """Повреждённый файл — это None, а не исключение: разбор папки не должен вставать.

    В листинге брифа здесь стояло `write_bytes(b"не картинка")`, но байтовый литерал
    кириллицу не принимает — SyntaxError ещё до запуска теста. Тот же смысл записан
    через write_text с явной кодировкой utf-8.
    """
    p = tmp_path / "битое.jpg"
    p.write_text("не картинка", encoding="utf-8")
    assert load_photo(p) is None


def test_ne_padaet_na_ekstremalnom_sootnoshenii(tmp_path) -> None:
    """Снимок 4900x1: короткая сторона при уменьшении обнулилась бы, и Pillow упал бы
    ValueError'ом наружу. Задача 9 на это не защищена — контракт «None, а не исключение».
    """
    p = _jpeg(tmp_path / "panorama.jpg", size=(4900, 1))
    img = load_photo(p, max_dim=2400)
    assert img is not None and min(img.shape[:2]) >= 1
