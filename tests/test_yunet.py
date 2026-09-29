"""Разбор строк `cv2.FaceDetectorYN.detect` в объекты Face.

Тесты чистые: никакого insightface и никаких моделей, только numpy-массивы строк
и подставные детектор с распознаванием. Реальные фото из data/ и ref/ здесь не
участвуют — это снимки конкретного ребёнка.

Пути «строки → лица» в модуле ровно один: `YunetEngine.detect`. Отдельной функции
разбора строк здесь больше нет — она отдавала лица с нулевым отпечатком, то есть
ровно те, которые `prigoden_otpechatok` заверает, и противоречила контракту самого
модуля: лицо без пригодного отпечатка считается ненайденным. Всё, что ниже,
проверяется на честном пути, где отпечаток достраивается на месте.
"""

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pytest import MonkeyPatch

from core.engine import prigoden_otpechatok
from core.yunet import YunetEngine


# --- ниже: движок целиком, но по-прежнему без моделей --------------------------------
#
# Все проверки работают на подставном детекторе и подставном распознавателе, поэтому
# остаются юнит-тестами: ни один из них не трогает ~/.insightface и не зависит от того,
# скачаны веса или нет.


def stroka_po_x(x: float, w: float = 100.0, h: float = 150.0,
                score: float = 0.9) -> list[float]:
    """Строка, у которой левый глаз стоит ровно на x рамки.

    По этой метке тест видит, с какой строки слеплен отпечаток: точки у строк с разными
    x разные, и подмена отпечатка между двумя лицами осталась бы видимой.
    """
    row = [x, 20.0, w, h]
    row += [x, 22.0, x + 40.0, 22.0, x + 20.0, 45.0, x + 10.0, 60.0, x + 30.0, 60.0]
    row.append(score)
    return row


class FakeDetector:
    """Отдаёт заготовленные строки вместо настоящего YuNet."""

    def __init__(self, rows: np.ndarray) -> None:
        self._rows = rows

    def setInputSize(self, size: tuple[int, int]) -> None:
        pass

    def detect(self, image: np.ndarray) -> tuple[int, np.ndarray]:
        return 0, self._rows


def krop_s_metkoy_x(img: np.ndarray, landmark: np.ndarray, image_size: int = 112,
                    mode: str = "arcface") -> np.ndarray:
    """Кроп, в пикселе [0,0,0] которого живёт x лица-источника."""
    crop = np.zeros((image_size, image_size, 3), dtype=np.float32)
    crop[0, 0, 0] = landmark[0][0]
    return crop


def engine_s_stubami(tmp_path: Path, monkeypatch: MonkeyPatch, rows: np.ndarray,
                     recognition: Any) -> YunetEngine:
    """YunetEngine с подставленными детектором, распознавателем и norm_crop."""
    engine = YunetEngine(tmp_path / "yunet.onnx")
    engine._detector = FakeDetector(rows)
    engine._recognition = recognition
    monkeypatch.setattr("core.yunet._norm_crop", krop_s_metkoy_x)
    return engine


class HorosheeRaspoznavanie:
    """Распознаватель, который всегда отдаёт годный единичный вектор с меткой x."""

    def get_feat(self, crop: np.ndarray) -> np.ndarray:
        v = float(crop[0, 0, 0]) / 1000.0
        out = np.zeros(512, dtype=np.float32)
        out[0] = v
        out[1] = float(np.sqrt(max(0.0, 1.0 - v * v)))
        return out


def test_pyatnadcat_kolonek_stanovyatsya_lico_s_otpechatkom(
        tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Пятнадцать колонок строки YuNet — это рамка (x, y, x+w, y+h), десять чисел точек
    и отпечаток единичной длины.

    Покрытие переехало на `detect` вместе с функцией разбора строк: путь в модуле один,
    и проверять надо именно его — иначе тест разрешал бы жизнь второму, «удобному» разбору
    строк, который отдаёт лица с нулевым отпечатком.
    """
    rows = np.array([stroka_po_x(500.0)], dtype=np.float32)
    engine = engine_s_stubami(tmp_path, monkeypatch, rows, HorosheeRaspoznavanie())

    lica = engine.detect(np.zeros((10, 10, 3), dtype=np.uint8))

    assert len(lica) == 1
    f = lica[0]
    assert (f.x1, f.y1, f.x2, f.y2) == (500.0, 20.0, 600.0, 170.0)
    assert f.detector == "yunet"
    assert len(f.landmarks) == 10
    assert f.landmarks[0] == pytest.approx(500.0), "точки переставлены"
    assert prigoden_otpechatok(f.embedding), "отпечаток годного лица непригоден"
    assert np.linalg.norm(f.embedding) == pytest.approx(1.0)


def test_melkoe_lico_ne_dostaet_otpechatka_vovse(tmp_path: Path,
                                                 monkeypatch: MonkeyPatch) -> None:
    """Лицо в 8 px на 24-мегапиксельном снимке — шум: его рамку отсекает `MIN_FACE_PX`,
    и распознаватель на неё не зовётся.

    Второе важнее первого: счётчик отбраковки обязан молчать. Мелкая рамка — решение
    детектора, а не испорченные данные, и в отчёте «лиц отброшено» про неё появились бы
    там, где терять было нечего.
    """
    vyzovy: list = []

    class Sledzhet(HorosheeRaspoznavanie):
        def get_feat(self, crop: np.ndarray) -> np.ndarray:
            vyzovy.append(float(crop[0, 0, 0]))
            return super().get_feat(crop)

    rows = np.array([stroka_po_x(500.0, w=8.0, h=9.0), stroka_po_x(700.0)],
                    dtype=np.float32)
    engine = engine_s_stubami(tmp_path, monkeypatch, rows, Sledzhet())

    lica = engine.detect(np.zeros((10, 10, 3), dtype=np.uint8))

    assert [f.x1 for f in lica] == [700.0]
    assert vyzovy == [700.0], "мелкую рамку всё равно читали моделью"
    assert engine.otsortirovannye_lica == 0


def test_otpechatok_ne_uezhaet_na_drugoe_lico(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Главный тихий отказ этого модуля.

    Если сначала отфильтровать строки, а потом сопоставить их с отпечатками по номеру,
    первое принятое лицо получит отпечаток отвергнутого — без всякой ошибки.
    Проверяется без моделей: подставляем детектор, распознавание и norm_crop так,
    чтобы в отпечатке было зашифровано, с какой именно строки он получен.
    """
    # строка 0 отвергнута по уверенности, строка 1 принята; у них разные x
    rows = np.array([stroka_po_x(10.0, score=0.1), stroka_po_x(500.0, score=0.9)],
                    dtype=np.float32)

    class FakeRec:
        def get_feat(self, crop: np.ndarray) -> np.ndarray:
            # crop[0, 0, 0] — это x лица, куда его положил наш krop_s_metkoy_x.
            # detect() нормализует вектор до длины 1, поэтому метку кладём в единичный
            # вектор: после нормировки нулевой элемент остаётся ровно x/1000
            # (500 → 0.5, а отвергнутые 10 → 0.01).
            v = float(crop[0, 0, 0]) / 1000.0
            out = np.zeros(512, dtype=np.float32)
            out[0] = v
            out[1] = float(np.sqrt(1.0 - v * v))
            return out

    engine = engine_s_stubami(tmp_path, monkeypatch, rows, FakeRec())

    faces = engine.detect(np.zeros((10, 10, 3), dtype=np.uint8))

    assert len(faces) == 1
    assert faces[0].x1 == 500.0                                    # рамка от принятого лица
    assert faces[0].embedding[0] == pytest.approx(500.0 / 1000.0)  # отпечаток от него, не 0.01
    assert np.linalg.norm(faces[0].embedding) == pytest.approx(1.0)


def test_nulevoy_otpechatok_ne_popadaet_v_arhiv(tmp_path: Path,
                                                monkeypatch: MonkeyPatch) -> None:
    """Нулевой вектор — не «лицо с нулевым отпечатком», а не найденное лицо.

    Деление на нулевую норму дало бы NaN: он без единой ошибки проходит через base64
    в оглавлении и дальше отравляет каждое сравнение в задачах 8 и 9. Предупреждение
    numpy о делении превращаем в ошибку — иначе в логах осталась бы тихая деградация.
    """
    rows = np.array([stroka_po_x(100.0), stroka_po_x(500.0)], dtype=np.float32)

    class FakeRec:
        def get_feat(self, crop: np.ndarray) -> np.ndarray:
            out = np.zeros(512, dtype=np.float32)
            if float(crop[0, 0, 0]) == 100.0:
                return out              # вырожденный кроп: модель отдаёт одни нули
            out[0] = 3.0                # нормальный вектор, норма которого не 1
            return out

    engine = engine_s_stubami(tmp_path, monkeypatch, rows, FakeRec())

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        faces = engine.detect(np.zeros((10, 10, 3), dtype=np.uint8))

    assert len(faces) == 1
    assert faces[0].x1 == 500.0                             # лицо с нулём отброшено
    assert np.isfinite(faces[0].embedding).all()            # NaN в архив не попадает
    assert faces[0].embedding[0] == pytest.approx(1.0)


def test_otbrochennoe_lico_dvizhok_otschityvaet(tmp_path: Path,
                                                monkeypatch: MonkeyPatch) -> None:
    """Отбраковка обязана быть числом, а не молчанием.

    Лицо, завернувшееся по `prigoden_otpechatok`, исчезает внутри `detect()` — до
    разбора папки (задача 9) оно уже не доживает. Счётчик — то, по чему отчёт
    (задача 10) отличит «лицо отброшено как непригодное» от «лица не было вовсе».
    Отсеянные по уверенности и по размеру рамки в счётчик не входят: это решение
    детектора, а не испорченные данные.
    """
    rows = np.array([stroka_po_x(100.0), stroka_po_x(300.0), stroka_po_x(500.0)],
                    dtype=np.float32)

    class FakeRec:
        def get_feat(self, crop: np.ndarray) -> np.ndarray:
            x = float(crop[0, 0, 0])
            if x == 100.0:
                return np.zeros(512, dtype=np.float32)          # вырожденный кроп
            if x == 300.0:
                return np.full(512, np.nan, dtype=np.float32)   # сбой модели
            out = np.zeros(512, dtype=np.float32)
            out[0] = 2.0
            return out

    engine = engine_s_stubami(tmp_path, monkeypatch, rows, FakeRec())
    assert engine.otsortirovannye_lica == 0, "до разбора потерь нет"

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        faces = engine.detect(np.zeros((10, 10, 3), dtype=np.uint8))

    assert len(faces) == 1 and faces[0].x1 == 500.0
    assert engine.otsortirovannye_lica == 2
    # накопительный: второй кадр добавляет свои потери, а не обнуляет прежние
    engine.detect(np.zeros((10, 10, 3), dtype=np.uint8))
    assert engine.otsortirovannye_lica == 4


def test_nan_ot_modeli_toze_ne_lico(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Та же страховка для NaN, который модель выдала сама (норма тоже не конечна)."""
    rows = np.array([stroka_po_x(500.0)], dtype=np.float32)

    class FakeRec:
        def get_feat(self, crop: np.ndarray) -> np.ndarray:
            out = np.full(512, np.nan, dtype=np.float32)
            return out

    engine = engine_s_stubami(tmp_path, monkeypatch, rows, FakeRec())

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert engine.detect(np.zeros((10, 10, 3), dtype=np.uint8)) == []


def test_otpechatok_chuzhoy_shiriny_toze_ne_lico(tmp_path: Path,
                                                 monkeypatch: MonkeyPatch) -> None:
    """Правило едино на все пути: вектор не в 512 чисел — не лицо (см. `prigoden_otpechatok`).

    До правки этот путь проверял только норму, и урезанный отпечаток доезжал до
    `matmul` в сравнении, где ронял весь прогон.
    """
    rows = np.array([stroka_po_x(100.0), stroka_po_x(500.0)], dtype=np.float32)

    class FakeRec:
        def get_feat(self, crop: np.ndarray) -> np.ndarray:
            if float(crop[0, 0, 0]) == 100.0:
                return np.full(256, 0.1, dtype=np.float32)      # «вектор из другой модели»
            out = np.zeros(512, dtype=np.float32)
            out[0] = 3.0
            return out

    engine = engine_s_stubami(tmp_path, monkeypatch, rows, FakeRec())

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        faces = engine.detect(np.zeros((10, 10, 3), dtype=np.uint8))

    assert [f.x1 for f in faces] == [500.0]


# --- распознаватель, подставленный извне (нужен режиму «Оба», задача 7) -----------
#
# BothEngine грузит InsightEngine первым, и внутри его FaceAnalysis уже есть готовая
# arcface. Если юнет всё равно пойдёт в `_load_recognition()`, те же 174 МБ весов
# прочтутся второй раз и лягут в память вторым экземпляром сессии ONNX Runtime.
# Поэтому движок умеет принять готовый распознаватель — здесь это проверяется
# без единой модели.


def yunet_gotov_k_zagruzke(tmp_path: Path, monkeypatch: MonkeyPatch) -> Path:
    """Пустой файл вместо YuNet-онкс: сам детектор в этой проверке не нужен."""
    model_path = tmp_path / "yunet.onnx"
    model_path.write_bytes(b"")
    monkeypatch.setattr("core.yunet.cv2.FaceDetectorYN_create",
                        lambda *a, **k: object())
    return model_path


def test_gotovaya_model_ne_zastavlyaet_chitat_174_megbayt_vtoroy_raze(
        tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    def nikogda() -> Any:
        raise AssertionError("_load_recognition() не вызывается, когда модель уже дали")

    monkeypatch.setattr("core.yunet._load_recognition", nikogda)
    recognition = object()

    engine = YunetEngine(yunet_gotov_k_zagruzke(tmp_path, monkeypatch),
                         recognition=recognition)
    engine.load()

    assert engine._recognition is recognition


def test_razpoznavatel_mozno_peredat_i_posle_razpakovki_dvizhka(
        tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """BothEngine отдаёт модель методом, а не через приватное поле чужого класса."""
    def nikogda() -> Any:
        raise AssertionError("вторую копию модели читать нельзя")

    monkeypatch.setattr("core.yunet._load_recognition", nikogda)
    recognition = object()

    engine = YunetEngine(yunet_gotov_k_zagruzke(tmp_path, monkeypatch))
    engine.adopt_recognition(recognition)
    engine.load()

    assert engine._recognition is recognition


def test_posle_zagruzki_razpoznavatel_uzhe_ne_menyayut(
        tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Молчаливый отказ здесь хуже ошибки: лицо поехало бы чужим отпечатком."""
    monkeypatch.setattr("core.yunet._load_recognition", lambda: object())
    engine = YunetEngine(yunet_gotov_k_zagruzke(tmp_path, monkeypatch))
    engine.load()
    with pytest.raises(RuntimeError):
        engine.adopt_recognition(object())


def test_tot_zhe_samyy_razpoznavatel_posle_zagruzki_ne_oshibka(
        tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Повторить `load()` с той же моделью — не «сменить распознаватель на чужой».

    Второй `load()` делает рабочий поток задачи 9 на каждом прогоне папки, и падать
    должно только настоящее расхождение: когда после загрузки приходит ДРУГОЙ объект.
    """
    model = object()
    monkeypatch.setattr("core.yunet._load_recognition", lambda: model)
    engine = YunetEngine(yunet_gotov_k_zagruzke(tmp_path, monkeypatch))
    engine.load()
    engine.adopt_recognition(model)             # тот же самый — молча, без RuntimeError
    engine.load()
    assert engine._recognition is model


def test_bez_podstavki_model_gruzitsya_kak_razhe(
        tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Обычный путь `make_engine('yunet', ...)` не сломан: без модели она грузится."""
    sentinel = object()
    vyzov: list[int] = []

    def razraz() -> Any:
        vyzov.append(1)
        return sentinel

    monkeypatch.setattr("core.yunet._load_recognition", razraz)

    engine = YunetEngine(yunet_gotov_k_zagruzke(tmp_path, monkeypatch))
    engine.load()

    assert vyzov == [1]
    assert engine._recognition is sentinel
