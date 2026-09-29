"""Фоновый разбор папки: кэш, отмена, частичный результат и видимые потери.

Тесты чистые: ни insightface, ни OpenCV-моделей, ни реальных снимков. Движок
подставляется фиктивный, фото пишутся синтетические через Pillow в `tmp_path`
(личные фото из data/ и ref/ в тестах не участвуют). `scan_photos` — функция без
Qt, поэтому она проверяется напрямую; `ScanWorker` — тонкая обёртка, и её тесты
только и подтверждают, что она ничего не теряет на пути в поток и обратно.
"""

import sqlite3
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pytest import MonkeyPatch

from core.cache import FaceCache
from core.engine import Face
from core.report import write_report
from core.scanner import file_stamp
from core.worker import (ModeliNeZagruzilis, OglavlenieNedostupno, ScanStats, ScanWorker,
                         obespechit_zagruzku, scan_photos)


def lico(side: float = 100.0, emb: np.ndarray | None = None) -> Face:
    return Face(box=(0.0, 0.0, side, side), landmarks=None,
                embedding=np.ones(512, dtype=np.float32) / np.sqrt(512)
                if emb is None else emb,
                detector="insight")


def neprigodnyy_vektor(imya: str = "nan") -> np.ndarray:
    """Отпечаток, который `prigoden_otpechatok` не пропустит: таким он бывает у
    вырожденного кропа на границе кадра и в строках, записанных старыми версиями."""
    return {"nan": np.full(512, np.nan, dtype=np.float32),
            "nuli": np.zeros(512, dtype=np.float32)}[imya]


class FakeEngine:
    name = "fake"

    def __init__(self, faces: list[Face] | None = None, raises: bool = False) -> None:
        self.faces = faces if faces is not None else [lico()]
        self.raises = raises
        self.seen: list[Any] = []

    def load(self) -> None:
        pass

    def detect(self, image: np.ndarray) -> list[Face]:
        self.seen.append(image.shape)
        if self.raises:
            raise RuntimeError("модель упала")
        return list(self.faces)


class DvizhokSOtchyotom:
    """Движок, который считает `load()` и `detect()` и честно знает про `loaded`.

    По нему и проверяется главное обещание рабочего потока: модели читаются один
    раз на сессию, а не по разу на каждый прогон папки (13–21 с на лишний круг).
    """

    name = "fake"

    def __init__(self, faces: list[Face] | None = None) -> None:
        self.faces = faces if faces is not None else [lico()]
        self.zagruzki = 0
        self.detektsii = 0
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self.zagruzki += 1
        self._loaded = True

    def detect(self, image: np.ndarray) -> list[Face]:
        self.detektsii += 1
        return list(self.faces)


class DvizhokBezModeley:
    """Движок, у которого нет весов: `load()` падает до первого кадра."""

    name = "fake"
    loaded = False

    def load(self) -> None:
        raise FileNotFoundError("нет модели face_detection_yunet_2023mar.onnx")

    def detect(self, image: np.ndarray) -> list[Face]:
        raise AssertionError("после неудачной загрузки разбора быть не должно")


class KeshKOtdannyom:
    """Настоящее оглавление, которое после N записанных снимков перестаёт отвечать.

    Так выглядит «диск кончился» посреди разбора: SQLite бросает `OperationalError`
    из записи, и весь уже снятый труд остаётся в настоящей базе — её и проверяем
    после сбоя, а не слова сообщения.
    """

    def __init__(self, osnovnoy: FaceCache, pitayetsya: int) -> None:
        self.osnovnoy = osnovnoy
        self.pitayetsya = pitayetsya
        self.zapisey = 0

    @property
    def damage_count(self) -> int:
        return self.osnovnoy.damage_count

    def get(self, photo: Path, stamp: tuple[int, int], engine: str,
            max_dim: int) -> list[Face] | None:
        return self.osnovnoy.get(photo, stamp, engine, max_dim)

    def get_s_poterjami(self, photo: Path, stamp: tuple[int, int], engine: str,
                        max_dim: int) -> tuple[list[Face], int] | None:
        return self.osnovnoy.get_s_poterjami(photo, stamp, engine, max_dim)

    def put(self, photo: Path, stamp: tuple[int, int], engine: str,
            max_dim: int, faces: list[Face], dropped: int = 0) -> None:
        if self.zapisey >= self.pitayetsya:
            raise sqlite3.OperationalError("database or disk is full")
        self.osnovnoy.put(photo, stamp, engine, max_dim, faces, dropped=dropped)
        self.zapisey += 1


class MertvoeOglavlenie:
    """Оглавление, мёртвое с самого начала: база не открылась вообще."""

    damage_count = 0

    def get(self, *args: Any, **kwargs: Any) -> list[Face] | None:
        raise sqlite3.DatabaseError("disk I/O error")

    def get_s_poterjami(self, *args: Any, **kwargs: Any) -> tuple[list[Face], int] | None:
        raise sqlite3.DatabaseError("disk I/O error")

    def put(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("в мёртвое оглавление не пишут")


def _obrezannyy_dzhpeg(p: Path) -> Path:
    """Заголовок JPEG цел, поток данных обрезан — так выглядит недокачанный снимок."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (300, 200), (7, 7, 7)).save(buf, "JPEG")
    data = buf.getvalue()
    p.write_bytes(data[: len(data) // 2])
    return p


def _gniloe_heic(p: Path) -> Path:
    """Настоящий HEIC с отрезанным хвостом: контейнер опознаётся, а декодер спотыкается.

    Именно ради этого случая архив iPhone и требует собственную причину: по строке
    «не читается» такой снимок неотличим от мусора, переименованного в .jpg.
    """
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (300, 200), (5, 9, 12)).save(buf, format="HEIF")
    p.write_bytes(buf.getvalue()[:-12])
    return p


@pytest.fixture
def cache(tmp_path: Path) -> Iterator[FaceCache]:
    c = FaceCache(tmp_path / "cache.sqlite3")
    yield c
    c.close()


def _photos(tmp_path: Path, n: int) -> list[Path]:
    from PIL import Image

    out: list[Path] = []
    for i in range(n):
        p = tmp_path / f"f{i}.jpg"
        Image.new("RGB", (300, 200), (i, 0, 0)).save(p, "JPEG")
        out.append(p)
    return out


def zabrat(dannye: list[Any], app: Any, predel: float = 5.0) -> bool:
    """Дожидаться доставки сигнала в поток интерфейса, крутя его очередь событий.

    Слот, присоединённый к сигналу из рабочего потока, живёт в очереди событий,
    а тестовый прогон её сам не крутит — ровно этим занимается окно.
    """
    do_konca = time.monotonic() + predel
    while not dannye and time.monotonic() < do_konca:
        app.processEvents()
        time.sleep(0.005)
    return bool(dannye)


# --- разбор, кэш, частичный результат ---------------------------------------------


def test_zapisaet_lica_i_v_kesh_i_v_otvet(tmp_path: Path, cache: FaceCache) -> None:
    files = _photos(tmp_path, 2)
    photos, stats = scan_photos(files, FakeEngine(), cache, max_dim=1200)
    assert len(photos) == 2 and stats.scanned == 2 and stats.failures == []
    again, stats2 = scan_photos(files, FakeEngine(), cache, max_dim=1200)
    assert stats2.cached == 2          # второй раз модели не вызывались


def test_otmena_sohranyaet_sdelannoe_v_kesh(tmp_path: Path, cache: FaceCache) -> None:
    files = _photos(tmp_path, 4)
    done: list = []
    photos, stats = scan_photos(files, FakeEngine(), cache, max_dim=1200,
                                should_stop=lambda: len(done) >= 2,
                                on_progress=lambda d, t, p: done.append(p))
    assert len(photos) == 2 and stats.scanned == 2
    assert cache.get(files[0], file_stamp(files[0]), "fake", 1200) is not None
    assert cache.get(files[3], file_stamp(files[3]), "fake", 1200) is None


def test_bityy_fayl_popadaet_v_failures_i_ne_glushit_progon(tmp_path: Path,
                                                            cache: FaceCache) -> None:
    files = _photos(tmp_path, 2)
    (tmp_path / "broken.jpg").write_bytes(b"not an image")
    photos, stats = scan_photos(files + [tmp_path / "broken.jpg"], FakeEngine(), cache,
                                max_dim=1200)
    assert stats.scanned == 2 and len(photos) == 2
    assert len(stats.failures) == 1 and stats.failures[0][0].endswith("broken.jpg")
    assert stats.failures[0][1].startswith("файл не читается")
    assert stats.failures[0][1] != "файл не читается", "без причины отчёт слепой"


def test_padenie_modeli_ne_ostanavivaet_progon(tmp_path: Path,
                                               cache: FaceCache) -> None:
    files = _photos(tmp_path, 1)
    photos, stats = scan_photos(files, FakeEngine(raises=True), cache, max_dim=1200)
    assert photos == {} and len(stats.failures) == 1


# --- два разных сбоя прогона: загрузка моделей и оглавление -------------------------


def test_sboy_oglavleniya_posredi_progona_nazyvaet_oglavlenie_a_ne_modeli(
        tmp_path: Path, cache: FaceCache) -> None:
    """Диск кончился на третьем снимке из четырёх. Прочитать про это человек обязан
    в строке ошибки: задача 15 показывает `error` как есть, и этап должен быть назван
    здесь, а не выдуман вызывающим кодом.
    """
    files = _photos(tmp_path, 4)
    slabyy = KeshKOtdannyom(cache, pitayetsya=2)

    with pytest.raises(OglavlenieNedostupno) as lovushka:
        scan_photos(files, FakeEngine(), slabyy, max_dim=1200)

    soobshchenie = str(lovushka.value)
    assert soobshchenie.startswith("оглавление недоступно:")
    assert "уже снятые лица сохранены" in soobshchenie
    assert "2 снимка" in soobshchenie
    assert "не удалось загрузить модели" not in soobshchenie
    # сырая строка SQLite родителю не подсказка: переводим, а не показываем
    assert "disk is full" not in soobshchenie
    assert "свободное место" in soobshchenie
    assert lovushka.value.sohraneno == 2
    assert isinstance(lovushka.value.__cause__, sqlite3.OperationalError)

    # и главное: «сохранены» — не слова об оглавлении, а факт из настоящей базы
    assert cache.get(files[0], file_stamp(files[0]), "fake", 1200) is not None
    assert cache.get(files[1], file_stamp(files[1]), "fake", 1200) is not None
    assert cache.get(files[2], file_stamp(files[2]), "fake", 1200) is None


def test_sboy_oglavleniya_do_pervogo_foto_ne_vraet_chto_chto_to_sohraneno(
        tmp_path: Path) -> None:
    """Если оглавление сдохло на первом же чтении, писать «уже снятые лица сохранены»
    — врать. Тот же тип, но честный хвост: снимать было нечего.
    """
    files = _photos(tmp_path, 2)
    with pytest.raises(OglavlenieNedostupno) as lovushka:
        scan_photos(files, FakeEngine(), MertvoeOglavlenie(), max_dim=1200)

    soobshchenie = str(lovushka.value)
    assert soobshchenie.startswith("оглавление недоступно:")
    assert "уже снятые лица сохранены" not in soobshchenie
    assert "ни один снимок ещё не разобран" in soobshchenie
    assert "диск не отвечает" in soobshchenie
    assert lovushka.value.sohraneno == 0


# --- настоящие ответы SQLite: переводить надо те строки, которые база и говорит ------
#
# Перевод сырой строки в русское действие — единственная причина, по которой рабочий
# поток вообще знает про SQLite. Список ниже составлен по настоящим ответам библиотеки:
# про испорченный файл базы данных она говорит «database disk image is malformed», а не
# «database is corrupted». Прежний ключ держил вторую, несуществующую формулировку, и
# битое оглавление проваливалось в общий ответ «база не отвечает» — то есть человек
# терял ровно тот совет, ради которого строка и заведена: очистить оглавление.


# Имена случаев — латиницей: кириллицу pytest экранирует последовательностями вида
# «\u043e\u0447», и вывод прогона становится нечитаемым ровно там, где он нужен.
# Имена случаев — латиницей: совет по-русски попадает в параметр, pytest экранирует
# кириллицу последовательностями вида «\u043e\u0447», и вывод прогона становится
# нечитаемым ровно там, где он и нужен.
NASTOJASHHIE_OTVETY_SQLITE = (
    pytest.param("database disk image is malformed", "очистите", id="malformed"),
    pytest.param("file is not a database", "очистите", id="ne-baza"),
    pytest.param("database or disk is full", "свободное место", id="polnyy-disk"),
    pytest.param("database is locked", "другой копией", id="zanyato"),
    pytest.param("attempt to write a readonly database", "проверьте права",
                 id="tolko-chtenie"),
    pytest.param("unable to open database file", "не открывается", id="net-fayla"),
    pytest.param("disk I/O error", "диск не отвечает", id="io"),
    pytest.param("sqlite3_step failure", "база оглавления не отвечает", id="neizvestnoe"),
)


class OtdaetStokuSQLite:
    """Оглавление, которое на первое чтение отвечает заданной строкой SQLite."""

    damage_count = 0

    def __init__(self, stoka: str) -> None:
        self._stoka = stoka

    def get(self, *args: Any, **kwargs: Any) -> list[Face] | None:
        raise sqlite3.DatabaseError(self._stoka)

    def get_s_poterjami(self, *args: Any, **kwargs: Any) -> None:
        raise sqlite3.DatabaseError(self._stoka)

    def put(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("до записи дело не доходит")


@pytest.mark.parametrize("stoka, sovet", NASTOJASHHIE_OTVETY_SQLITE)
def test_kazhdyj_nastoyashchiy_otvet_sqlite_pereveden_v_dejstvie(
        tmp_path: Path, stoka: str, sovet: str) -> None:
    """Совет обязан доходить до человека на каждом настоящем ответе базы.

    Проверка идёт через `scan_photos`, а не через приватный переводчик: мало знать
    нужную строку, её ещё должен увидеть тот текст, который окно печатает как есть.
    """
    with pytest.raises(OglavlenieNedostupno) as lovushka:
        scan_photos(_photos(tmp_path, 1), FakeEngine(), OtdaetStokuSQLite(stoka),
                    max_dim=1200)

    soobshchenie = str(lovushka.value)
    assert sovet in soobshchenie, f"SQLite сказал {stoka!r}, а совета нет: {soobshchenie}"
    assert "не удалось загрузить модели" not in soobshchenie


def test_mertsvyh_klyuchov_v_tablice_perewoda_net() -> None:
    """Строки, которой SQLite не говорит никогда, в таблице быть не должно.

    Мёртвый ключ безвреден ровно до того дня, когда под ним прячется настоящий:
    «database is corrupted» звучит правдоподобно, поэтому битое оглавление годами
    доходило до человека как «база не отвечает».
    """
    from core.worker import _SBBOI_OGLAVLENIYA

    priznaki = [stoka for stoka, _ in _SBBOI_OGLAVLENIYA]
    assert "database is corrupted" not in priznaki
    assert "database disk image is malformed" in priznaki


def test_neudachnaya_zagruzka_modeley_eto_sboy_zagruzki_a_ne_oglavleniya(
        tmp_path: Path, cache: FaceCache) -> None:
    """Зеркальный случай: весов нет, дело не дошло ни до одного снимка. Текст не должен
    ни обещать сохранённые лица, ни упоминать оглавление.
    """
    with pytest.raises(ModeliNeZagruzilis) as lovushka:
        scan_photos(_photos(tmp_path, 1), DvizhokBezModeley(), cache, max_dim=1200)

    soobshchenie = str(lovushka.value)
    assert soobshchenie.startswith("не удалось загрузить модели:")
    assert "нет модели" in soobshchenie
    assert "ни один снимок не разобран" in soobshchenie
    assert "оглавление" not in soobshchenie
    assert "сохранены" not in soobshchenie
    assert isinstance(lovushka.value.__cause__, FileNotFoundError)
    # оглавление при этом здорово: прогон до него не дошёл ни на одну строку
    assert cache.clear() == 0


def test_sboy_zagruzki_nazyvaet_dva_nastoyashchih_puti(tmp_path: Path,
                                                       cache: FaceCache) -> None:
    """Текст об отсутствующих весах обязан вести туда, где человек реально может чинить.

    Прежняя формулировка «проверьте в настройках путь к моделям» отправляла человека в
    полосу настроек, где такой строки нет и быть не может: пути к весам — не настройка.
    Теперь названы оба настоящих пути теми же геттерами, по которым ходит само
    приложение, и сказано, какой из них качается, а какой лежит в пакете.
    """
    from utils.config import buffalo_l_dir, yunet_model_path

    with pytest.raises(ModeliNeZagruzilis) as lovushka:
        scan_photos(_photos(tmp_path, 1), DvizhokBezModeley(), cache, max_dim=1200)

    soobshchenie = str(lovushka.value)
    assert str(yunet_model_path()) in soobshchenie, "путь поиска лиц не назван"
    assert str(buffalo_l_dir()) in soobshchenie, "путь распознавания не назван"
    assert "в настройках" not in soobshchenie, "ссылка на настройку, которой нет"
    # два пути — это два разных разговора: один чинит пересборка, другой человек
    assert "230 КБ" in soobshchenie and "290 МБ" in soobshchenie


def test_net_fayla_yuneta_v_tekste_est_ssylka(tmp_path: Path,
                                                                  monkeypatch:
                                                                  MonkeyPatch) -> None:
    """`net modeli YuNet. Скачайте face_detection_yunet_2023mar.onnx` — это не действие.

    По названию файла человек не найдёт, откуда его взять. Движок обязан отдать
    настоящую ссылку OpenCV, и она доезжает до текста сбоя нетронутой.
    """
    import core.yunet as yunet_mod
    from core.yunet import YUNET_URL, YunetEngine

    monkeypatch.setattr(yunet_mod.cv2, "FaceDetectorYN_create", lambda *a, **k: object())
    monkeypatch.setattr(yunet_mod, "_load_recognition", lambda: object())
    netsushchestvuyushchaya = tmp_path / "net" / "yunet.onnx"
    with pytest.raises(FileNotFoundError) as lovushka:
        YunetEngine(netsushchestvuyushchaya).load()

    tekst = str(lovushka.value)
    assert str(netsushchestvuyushchaya) in tekst, "куда класть файл — не сказано"
    assert YUNET_URL in tekst and tekst.startswith("нет файла детектора лиц")
    assert "github.com" in YUNET_URL


def test_scan_worker_sboy_oglavleniya_ne_nazyvaetsya_sboyem_modeley(
        tmp_path: Path, cache: FaceCache) -> None:
    """Тот же сбой глазами интерфейса: в `error` приходит текст с этапом, `done` — нет.

    Проверка нужна именно на обёртке: до правки оба сбоя склеивались в `str(exc)` без
    названия этапа, и задача 15 подписывала полный диск как «распознавание не
    запустилось».
    """
    files = _photos(tmp_path, 3)
    oshibki: list[str] = []
    polucheno: list = []
    worker = ScanWorker(files, FakeEngine(), KeshKOtdannyom(cache, pitayetsya=1), 1200)
    worker.error.connect(oshibki.append)
    worker.done.connect(lambda payload: polucheno.append(payload))
    worker.run()

    assert len(oshibki) == 1, oshibki
    assert oshibki[0].startswith("оглавление недоступно:")
    assert "уже снятые лица сохранены — 1 снимок" in oshibki[0]
    assert "не удалось загрузить модели" not in oshibki[0]
    assert polucheno == [], "сбой прогона не притворяется завершением"


def test_neozhidannyy_sboy_ne_pripisyvaetsya_nikakomu_iz_etapov(
        tmp_path: Path, cache: FaceCache, monkeypatch: MonkeyPatch) -> None:
    """«Не знаю, что сломалось» не имеет права притворяться известным этапом.

    Любой неопознанный сбой уходит с нейтральным заголовком: иначе человек с битой
    библиотекой читал бы «не удалось загрузить модели» и искал бы веса.
    """
    def bochaya_proverka(path: Path) -> tuple[int, int]:
        raise ValueError("неожиданная поломка")

    monkeypatch.setattr("core.worker.file_stamp", bochaya_proverka)
    oshibki: list[str] = []
    worker = ScanWorker(_photos(tmp_path, 1), FakeEngine(), cache, 1200)
    worker.error.connect(oshibki.append)
    worker.run()

    assert oshibki == ["разбор прервался: неожиданная поломка"]
    assert not oshibki[0].startswith(("не удалось загрузить модели",
                                      "оглавление недоступно"))


# --- причина нечитаемости доживает до отчёта ----------------------------------------


def test_prichina_nechitaemogo_snimka_razlichaet_bityy_kadr_i_gniloy_heic(
        tmp_path: Path, cache: FaceCache) -> None:
    """Три потери — три разных текста. «Файл не читается» на сотнях строк отчёта не
    даёт никакой зацепки, а архив-то iPhone: там HEIC, который не раскрылся, и
    обрезанный JPEG из iCloud — две разные беды с разными советами.
    """
    files = _photos(tmp_path, 1)
    musor = tmp_path / "musor.jpg"
    musor.write_bytes(b"not an image")
    obrezannyy = _obrezannyy_dzhpeg(tmp_path / "obrezannyy.jpg")
    gniloe = _gniloe_heic(tmp_path / "gniloe.heic")

    photos, stats = scan_photos(files + [musor, obrezannyy, gniloe], FakeEngine(), cache,
                                max_dim=1200)
    assert stats.scanned == 1 and len(photos) == 1

    prichiny = {Path(p).name: reason for p, reason in stats.failures}
    assert len(prichiny) == 3
    for imya in ("musor.jpg", "obrezannyy.jpg", "gniloe.heic"):
        assert prichiny[imya].startswith("файл не читается: ")
        assert prichiny[imya] != "файл не читается: "
    assert len(set(prichiny.values())) == 3, "причины слились в одну строку"

    # подстроки принадлежат Pillow и pillow-heif: меняются редко, а ловят суть
    assert "UnidentifiedImageError" in prichiny["musor.jpg"]      # формат не опознан
    assert "OSError" in prichiny["obrezannyy.jpg"]                # поток оборван
    assert "truncated" in prichiny["obrezannyy.jpg"]
    assert "ValueError" in prichiny["gniloe.heic"]                # декодер споткнулся
    assert "end of file" in prichiny["gniloe.heic"]

    # путь в причине не дублирует первое поле `failures`: в отчёте он уже назван
    assert str(musor) not in prichiny["musor.jpg"]
    assert "musor.jpg" in prichiny["musor.jpg"]


def test_isheznuvshiy_fayl_popadaet_v_failures_a_ne_ronyaet_progon(
        tmp_path: Path, cache: FaceCache) -> None:
    """Между «нашли файл в папке» и «разобрали» проходит до нескольких минут: архив
    в это время перемещают, флешку вытаскивают. Исключение от одного снимка ушло бы
    наружу вместе со всеми уже снятыми лицами, а не легло в сводку."""
    files = _photos(tmp_path, 2)
    net = tmp_path / "net.jpg"
    photos, stats = scan_photos(files[:1] + [net] + files[1:], FakeEngine(), cache,
                                max_dim=1200)
    assert stats.scanned == 2 and len(photos) == 2
    assert len(stats.failures) == 1
    assert stats.failures[0][0].endswith("net.jpg")
    assert "файл недоступен" in stats.failures[0][1]


def test_progress_idet_po_kazhdomu_foto_vklyuchaya_ohibki(tmp_path: Path,
                                                          cache: FaceCache) -> None:
    """Полоса прогресса не должна «застревать» на битом файле: счётчик идёт по числу
    обработанных снимков, а не по числу найденных лиц."""
    files = _photos(tmp_path, 2)
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    zapisi: list[tuple[int, int, str]] = []
    scan_photos(files[:1] + [broken] + files[1:], FakeEngine(), cache, max_dim=1200,
                on_progress=lambda d, t, p: zapisi.append((d, t, p.name)))
    assert [d for d, _, _ in zapisi] == [1, 2, 3]
    assert {t for _, t, _ in zapisi} == {3}


def test_foto_bez_lic_keshiruetsya_kak_pustoe(tmp_path: Path,
                                              cache: FaceCache) -> None:
    """Пустой список — это ответ, а не промах: иначе каждый прогон читал бы снимок заново."""
    files = _photos(tmp_path, 1)
    photos, stats = scan_photos(files, FakeEngine(faces=[]), cache, max_dim=1200)
    assert photos == {files[0]: []} and stats.scanned == 1
    _, stats2 = scan_photos(files, FakeEngine(faces=[]), cache, max_dim=1200)
    assert stats2.cached == 1 and stats2.scanned == 0


# --- требование 1: модели читаются ровно один раз на сессию ------------------------


def test_pustoy_spisok_ne_gruzyt_modeli(cache: FaceCache) -> None:
    """Пустая папка не должна стоить 300 МБ весов и 15 секунд ожидания.

    Сводка остаётся дефолтной: разбор не начинался, и отчитываться не о чем.
    """
    engine = DvizhokSOtchyotom()
    photos, stats = scan_photos([], engine, cache, max_dim=1200)
    assert photos == {} and stats == ScanStats() and engine.zagruzki == 0


def test_dva_potoka_ne_sobirayut_modeli_odnovremenno() -> None:
    """Гонка двух загрузок: `load()` у `BothEngine` не атомарен.

    Флажок проверяется до того, как начались 13–21 с чтения весов, поэтому два
    потока, решившие «движок не готов», собрали бы их дважды — а второй заход ещё и
    падает на `adopt_recognition()`. Общий замок в `obespechit_zagruzku` это держит:
    второй поток дожидается первого и видит движок готовым.
    """
    class DolgiyDvizhok:
        name = "fake"

        def __init__(self) -> None:
            self.zagruzki = 0
            self.aktivnyh = 0
            self.odnovremenno = 0
            self._loaded = False

        @property
        def loaded(self) -> bool:
            return self._loaded

        def load(self) -> None:
            self.aktivnyh += 1
            self.odnovremenno = max(self.odnovremenno, self.aktivnyh)
            time.sleep(0.05)               # окно, за которое гонка и раскрывается
            self.zagruzki += 1
            self.aktivnyh -= 1
            self._loaded = True

        def detect(self, image: np.ndarray) -> list[Face]:
            return [lico()]

    engine = DolgiyDvizhok()
    potoki = [threading.Thread(target=obespechit_zagruzku, args=(engine,))
              for _ in range(4)]
    for potok in potoki:
        potok.start()
    for potok in potoki:
        potok.join()

    assert engine.zagruzki == 1
    assert engine.odnovremenno == 1, "модели читались в два потока одновременно"


def test_dvizhok_gruzitsya_odin_raz_na_dva_progona(tmp_path: Path,
                                                   cache: FaceCache) -> None:
    """Второй `load()` на живом `BothEngine` — это не «ещё 15 секунд», а гарантированная
    ошибка: юнет после загрузки отказывается принимать распознаватель. Флаг `loaded`
    обязан останавливать второй заход до обращения к движку."""
    engine = DvizhokSOtchyotom()
    files = _photos(tmp_path, 1)
    scan_photos(files, engine, cache, max_dim=1200)
    cache.clear()                       # оглавление пусто: модели точно нужны снова
    photos, stats = scan_photos(files, engine, cache, max_dim=1200)
    assert engine.zagruzki == 1
    assert engine.detektsii == 2 and stats.scanned == 1 and len(photos) == 1


def test_vtoroy_scan_worker_na_tom_zhe_dvizhke_ne_perezagruzaet_modeli(
        tmp_path: Path, cache: FaceCache, qapp: Any) -> None:
    """То же самое через обёртку: пользователь жмёт «Разобрать папку» второй раз.

    `run()` вызывается напрямую, в этом же потоке: здесь важен порядок вызовов к
    движку, а не многопоточность — за ней в тест ниже.
    """
    engine = DvizhokSOtchyotom()
    files = _photos(tmp_path, 1)
    for _ in range(2):
        polucheno: list = []
        worker = ScanWorker(files, engine, cache, 1200)
        worker.done.connect(lambda payload: polucheno.append(payload))
        worker.run()
        assert len(polucheno) == 1, "прогон обязан закончиться результатом"
        cache.clear()
    assert engine.zagruzki == 1
    assert engine.detektsii == 2


def test_zaranee_zagruzhennyy_dvizhok_ne_tropyaet_vovse(tmp_path: Path,
                                                        cache: FaceCache) -> None:
    """Интерфейс вправе подготовить модели до запуска потока — тогда первый кадр
    не превращается в 15 секунд ожидания внутри разбора."""
    engine = DvizhokSOtchyotom()
    engine.load()
    photos, stats = scan_photos(_photos(tmp_path, 1), engine, cache, max_dim=1200)
    assert engine.zagruzki == 1 and stats.scanned == 1 and len(photos) == 1


def test_tri_realnyh_dvizhka_soblyudayut_kontrakt_loaded(
        tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """`loaded` есть у каждой реализации `FaceEngine`, а не только у фиктивных.

    Проверка `getattr(engine, "loaded", False)` в рабочем коде прощает отсутствующее
    свойство — то есть тихо грузит модели заново на каждый прогон. Именно это здесь и
    ловится: без свойства тест проходил бы молча, а пользователь получил бы второй
    `FaceAnalysis` посреди второй папки.
    """
    import core.yunet as yunet_mod
    from core.both import BothEngine
    from core.insight import InsightEngine
    from core.yunet import YunetEngine

    def ladno() -> Any:
        return object()

    def oryv() -> Any:
        raise OSError("[Errno 28] Failed to retrieve the model")

    monkeypatch.setattr(yunet_mod.cv2, "FaceDetectorYN_create", lambda *a, **k: object())
    monkeypatch.setattr(yunet_mod, "_load_recognition", ladno)
    model = tmp_path / "yunet.onnx"
    model.write_bytes(b"")

    yunet = YunetEngine(model)
    assert yunet.loaded is False
    yunet.load()
    assert yunet.loaded is True

    # --- половинчатая загрузка: детектор поднялся, распознаватель нет --------------
    #
    # Ровно то, что делает первый запуск человека в режиме «Быстрее» при оборванной
    # закачке 290 МБ. `detect()` без распознавателя не работает, а прежний ответ
    # `loaded` строился только на детекторе — значит `obespechit_zagruzku` поверил бы
    # ему навсегда, повторных попыток не было бы, и все снимки архива умерли бы на
    # `'NoneType' object has no attribute 'get_feat'`.
    monkeypatch.setattr(yunet_mod, "_load_recognition", oryv)
    polovina = YunetEngine(model)
    with pytest.raises(OSError):
        polovina.load()
    assert polovina.loaded is False, "неполная загрузка притворилась готовностью"
    assert polovina._detector is None, "наполовину собранный движок оставили детектору"

    # ...и повторная попытка возможна: та же закачка, которая оборвалась первый раз
    monkeypatch.setattr(yunet_mod, "_load_recognition", ladno)
    polovina.load()
    assert polovina.loaded is True

    # `load()` у InsightEngine не зовём: он тянет insightface и ~300 МБ весов.
    assert InsightEngine().loaded is False

    oba = BothEngine(FakeEngine(), FakeEngine())
    assert oba.loaded is False
    oba.load()
    assert oba.loaded is True


# --- требование 2: отброшенная работа видна числом, а не молчанием -----------------


def test_neprigodnyy_otpechatok_viden_v_svodke_i_ne_otravlyaet_kesh(
        tmp_path: Path, cache: FaceCache) -> None:
    """Лицо с непригодным отпечатком движок вернуть может, а в оглавлении ему не
    место: `Face.from_dict` на такой строке бросает ValueError, кэш считает запись
    битой и удаляет. Запиши мы её — снимок перечитывался бы моделями на каждом
    прогоне до конца жизни архива, молча и бесконечно."""
    files = _photos(tmp_path, 1)
    engine = FakeEngine(faces=[lico(), lico(side=50, emb=neprigodnyy_vektor("nan"))])
    photos, stats = scan_photos(files, engine, cache, max_dim=1200)
    assert [f.size for f in photos[files[0]]] == [100.0]
    assert stats.scanned == 1
    assert stats.dropped_faces == {files[0]: 1}
    assert stats.cache_damage == 0, "это не повреждение оглавления: случай другой"

    # ядовитое лицо не записано: второй прогон берёт снимок из оглавления
    _, stats2 = scan_photos(files, FakeEngine(), cache, max_dim=1200)
    assert stats2.cached == 1 and stats2.scanned == 0
    # ...но про потерянное лицо второй прогон знает: число лежит в самой строке
    # оглавления. Именно здесь решается, соврёт ли report.txt тёплого запуска
    # «потерь нет» об архив, где лица уже теряли.
    assert stats2.dropped_faces == {files[0]: 1}, "тёплый прогон потерял число потерь"


def test_teplyj_progon_snimaet_poteri_iz_oglavleniya_a_ne_vymyslyaet_nol(
        tmp_path: Path, cache: FaceCache) -> None:
    """Холодный прогон теряет два лица, тёплый обязан потерять те же два в отчёте.

    Модели на тёплом прогоне к этим снимкам не прикасались вовсе, и без поля потерь в
    оглавлении сводка второго прогона была бы пустой — а с ней и «потерь нет: ни один
    файл не пропущен» в report.txt. Снимок при этом тот же и беда та же.

    Второе число нужно не меньше первого: чистый снимок обязан принести ноль, иначе
    «потерь нет» стало бы «потерь не считали».
    """
    files = _photos(tmp_path, 2)
    engine = FakeEngine(faces=[lico(),
                               lico(side=50, emb=neprigodnyy_vektor("nan")),
                               lico(side=40, emb=neprigodnyy_vektor("nuli"))])
    _, stats = scan_photos(files, engine, cache, max_dim=1200)
    assert stats.scanned == 2
    assert stats.dropped_faces == {files[0]: 2, files[1]: 2}

    # чистый снимок: своя строка оглавления со своим числом потерь
    chistyj = tmp_path / "chistyj.jpg"
    from PIL import Image
    Image.new("RGB", (300, 200), (9, 9, 9)).save(chistyj, "JPEG")
    cache.put(chistyj, file_stamp(chistyj), "fake", 1200, [lico()], dropped=3)
    lica, poteri = cache.get_s_poterjami(chistyj, file_stamp(chistyj), "fake", 1200)
    assert len(lica) == 1 and poteri == 3

    _, teplyj = scan_photos(files, FakeEngine(), cache, max_dim=1200)
    assert teplyj.cached == 2 and teplyj.scanned == 0
    assert teplyj.dropped_faces == {files[0]: 2, files[1]: 2}

    itog = write_report(tmp_path / "report.txt", settings={}, rows=[],
                        failures=[], copied=[],
                        dropped_faces=teplyj.dropped_faces, cache_damage=0)
    tekst = itog.read_text(encoding="utf-8")
    assert "лиц отброшено: 4" in tekst
    assert "потерь нет" not in tekst, "тёплый запуск соврал, что потерь не было"


def test_pustaya_stroka_oglavleniya_peredaet_svoi_nuli(tmp_path: Path,
                                                       cache: FaceCache) -> None:
    """Фото без лиц — попадание с нулём потерь: «потерь нет» обязано быть посчитанным,
    а не отсутствующим, иначе отчёт не отличит чистый снимок от несчитанного."""
    files = _photos(tmp_path, 1)
    engine = FakeEngine(faces=[])
    _, stats = scan_photos(files, engine, cache, max_dim=1200)
    assert stats.dropped_faces == {}
    _, teplyj = scan_photos(files, FakeEngine(), cache, max_dim=1200)
    assert teplyj.cached == 1 and teplyj.dropped_faces == {}


def test_bitaya_zapis_v_keshe_schitaetsya_otdelno_ot_neprigodnogo_otpechatka(
        tmp_path: Path, cache: FaceCache) -> None:
    """Старые версии писали отпечаток без проверки — в оглавлении лежат NaN, Inf,
    нули и чужая ширина. Разбор такое фото не теряет, а пересчитывает, и отчёт
    обязан уметь сказать «оглавление починено: N», не путая это с «лицо отброшено
    как непригодное».
    """
    files = _photos(tmp_path, 1)
    cache.put(files[0], file_stamp(files[0]), "fake", 1200,
              [lico(emb=neprigodnyy_vektor("nuli"))])      # как писала старая версия

    photos, stats = scan_photos(files, FakeEngine(), cache, max_dim=1200)
    assert stats.scanned == 1 and len(photos) == 1
    assert stats.cache_damage == 1
    assert stats.dropped_faces == {}, "движок вернул одно годное лицо — потерь нет"

    # и только один раз: следующая сводка обязана быть чистой
    _, stats2 = scan_photos(files, FakeEngine(), cache, max_dim=1200)
    assert stats2.cached == 1 and stats2.cache_damage == 0


def test_vse_lica_neprigodnye_eto_nulev_likvid_i_odna_poterya(
        tmp_path: Path, cache: FaceCache) -> None:
    """Ни одного годного лица — это не «фото не обработано»: снимок прочитан, модели
    отработали. Он уходит в оглавление пустым и больше не читается."""
    files = _photos(tmp_path, 1)
    engine = FakeEngine(faces=[lico(emb=neprigodnyy_vektor("nan")),
                               lico(side=50, emb=neprigodnyy_vektor("nuli"))])
    photos, stats = scan_photos(files, engine, cache, max_dim=1200)
    assert photos == {files[0]: []} and stats.scanned == 1 and stats.failures == []
    assert stats.dropped_faces == {files[0]: 2}
    _, stats2 = scan_photos(files, FakeEngine(), cache, max_dim=1200)
    assert stats2.cached == 1


def test_poteri_vnutri_dvizhka_popadayut_v_tu_zhe_svodku(tmp_path: Path,
                                                          cache: FaceCache) -> None:
    """Настоящие движки роняют непригодное лицо внутри `detect()` — к разбору оно
    приходит уже исчезнувшим. Счётчик движка обязан попасть в ту же сводку и по
    каждому файлу отдельно, иначе «лицо отброшено как непригодное» в отчёте
    означало бы «повезло, движок доложил».
    """
    class DvizhokSSchetom:
        name = "fake"

        def __init__(self) -> None:
            self.otsortirovannye_lica = 0

        def load(self) -> None:
            pass

        def detect(self, image: np.ndarray) -> list[Face]:
            self.otsortirovannye_lica += 1        # модель выдала NaN-вектор, лицо провалилось
            return [lico()]

    files = _photos(tmp_path, 2)
    photos, stats = scan_photos(files, DvizhokSSchetom(), cache, max_dim=1200)
    assert stats.dropped_faces == {files[0]: 1, files[1]: 1}
    assert stats.scanned == 2 and len(photos[files[0]]) == 1


# --- требование 3: отмена между фото, начатый снимок доводится до конца --------------


def test_otmena_sredi_foto_dozakrivaet_nachatoe_i_pishet_ego_v_kesh(
        tmp_path: Path, cache: FaceCache) -> None:
    """«Остановить» нажали, пока снимок уже в моделях. Бросать его посреди кадра
    нельзя: оглавление останется без строки, и следующий запуск начнёт с нуля.
    Здесь кнопка нажимается ровно на втором фото — оно обязано дописаться."""
    files = _photos(tmp_path, 4)

    class DvizhokSNazhatoyOtkmenoy:
        name = "fake"

        def __init__(self) -> None:
            self.detektsii = 0
            self.stopped = False

        def load(self) -> None:
            pass

        def detect(self, image: np.ndarray) -> list[Face]:
            self.detektsii += 1
            if self.detektsii == 2:          # «Остановить» нажато во время второго фото
                self.stopped = True
            return [lico()]

    engine = DvizhokSNazhatoyOtkmenoy()
    photos, stats = scan_photos(files, engine, cache, max_dim=1200,
                                should_stop=lambda: engine.stopped)
    assert stats.scanned == 2 and len(photos) == 2
    assert engine.detektsii == 2, "после отмены модели трогать нельзя"
    for p in (files[0], files[1]):           # начатое фото дописано в оглавление
        assert cache.get(p, file_stamp(p), "fake", 1200) is not None
    for p in (files[2], files[3]):
        assert cache.get(p, file_stamp(p), "fake", 1200) is None


def test_otmena_do_pervogo_foto_daeet_pustoy_no_chestnyy_otvet(
        tmp_path: Path, cache: FaceCache) -> None:
    """Нажатая заранее кнопка — не ошибка и не зависание: пустой результат и нули."""
    files = _photos(tmp_path, 2)
    photos, stats = scan_photos(files, FakeEngine(), cache, max_dim=1200,
                                should_stop=lambda: True)
    assert photos == {} and stats.scanned == 0 and stats.cached == 0


def test_scan_worker_otmena_iz_potoka_dozakrivaet_foto_v_kesh(
        tmp_path: Path, cache: FaceCache, qapp: Any) -> None:
    """Отмена из потока интерфейса (`request_stop()` — чужой поток) и возврат
    частичного результата обратно.

    Здесь ровно тот путь, которым пользуется окно: `start()`, сигнал из рабочего
    потока, `wait()`.
    """
    files = _photos(tmp_path, 3)

    class DvizhokSOtmenoy:
        name = "fake"

        def __init__(self) -> None:
            self.detektsii = 0
            self.prosba: Any = None

        def load(self) -> None:
            pass

        def detect(self, image: np.ndarray) -> list[Face]:
            self.detektsii += 1
            if self.detektsii == 2 and self.prosba is not None:
                self.prosba()                # нажатие «Остановить» в рабочем потоке
            return [lico()]

    engine = DvizhokSOtmenoy()
    worker = ScanWorker(files, engine, cache, 1200)
    polucheno: list = []
    oshibki: list[str] = []
    worker.done.connect(lambda payload: polucheno.append(payload))
    worker.error.connect(oshibki.append)
    engine.prosba = worker.request_stop
    worker.start()
    assert worker.wait(30_000), "поток не завершился за 30 секунд"
    assert zabrat(polucheno, qapp), "результат не дошёл до потока интерфейса"

    assert not oshibki, oshibki
    photos, stats = polucheno[0]
    assert stats.scanned == 2 and len(photos) == 2
    assert cache.get(files[1], file_stamp(files[1]), "fake", 1200) is not None


def test_scan_worker_ne_gruzit_modeli_povtorno_i_dostaet_progress(
        tmp_path: Path, cache: FaceCache, qapp: Any) -> None:
    """Настоящий прогон в настоящем потоке: один `load()` и живой прогресс.

    Это единственный тест, который идёт через `start()`, а не через прямой `run()`:
    он и подтверждает, что обёртка доносит сигналы из чужого потока и не трогает
    веса второй раз. Ошибки движка — в тесте ниже.
    """
    engine = DvizhokSOtchyotom()
    files = _photos(tmp_path, 1)
    progress_zapisi: list[tuple[int, int, str]] = []
    polucheno: list = []
    worker = ScanWorker(files, engine, cache, 1200)
    worker.progress.connect(lambda d, t, name: progress_zapisi.append((d, t, name)))
    worker.done.connect(lambda payload: polucheno.append(payload))
    worker.start()
    assert worker.wait(30_000)
    assert zabrat(polucheno, qapp)

    assert engine.zagruzki == 1
    assert progress_zapisi == [(1, 1, "f0.jpg")]
    photos, stats = polucheno[0]
    assert len(photos) == 1 and stats.scanned == 1


def test_scan_worker_oibka_dvizhka_uhodit_v_signal_a_ne_v_padenie(
        tmp_path: Path, cache: FaceCache) -> None:
    """Не поднялись модели (нет весов, нет диска) — интерфейс обязан узнать об этом
    текстом, а не молча получить пустой список."""
    class DvizhokSBedoy:
        name = "fake"
        loaded = False

        def load(self) -> None:
            raise FileNotFoundError("нет модели face_detection_yunet_2023mar.onnx")

        def detect(self, image: np.ndarray) -> list[Face]:
            raise AssertionError("после неудачной загрузки разбора быть не должно")

    oshibki: list[str] = []
    polucheno: list = []
    worker = ScanWorker(_photos(tmp_path, 1), DvizhokSBedoy(), cache, 1200)
    worker.error.connect(oshibki.append)
    worker.done.connect(lambda payload: polucheno.append(payload))
    worker.run()
    assert len(oshibki) == 1 and "нет модели" in oshibki[0]
    assert polucheno == []
    assert not worker.isRunning()


def test_scan_worker_sozdaetsya_tri_signala_na_meste(tmp_path: Path, qapp: Any) -> None:
    """Смоук по сигнатуре из брифа: объект строится, три сигнала на месте."""
    kesh = FaceCache(tmp_path / "smoke.sqlite3")
    try:
        worker = ScanWorker([], DvizhokSOtchyotom(), kesh, 2400)
        assert [s for s in ("progress", "done", "error") if hasattr(worker, s)] == \
            ["progress", "done", "error"]
    finally:
        kesh.close()
