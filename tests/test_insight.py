"""Конвертация объектов insightface в `Face`.

Тесты чистые: лицо подставляется фиктивным, модели не загружаются, скачивание
buffalo_l не происходит. Личные фото из data/ и ref/ не участвуют.
"""

import sys
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest

from core.insight import InsightEngine, convert_insight_faces


class FakeFace:
    """Ровно те поля insightface, которые читает конвертер."""

    def __init__(self, box: Sequence[float], kps: Any, emb: Sequence[float],
                 score: float = 0.8) -> None:
        self.bbox = np.array(box, dtype=np.float32)
        self.kps = np.array(kps, dtype=np.float32)
        self.normed_embedding = np.array(emb, dtype=np.float32)
        self.det_score = score


def test_preshodit_bbox_x1y1x2y2() -> None:
    # точки заведомо разные: фикстура из нулей проглотила бы перестановку глаз
    kps = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
    f = FakeFace([5, 6, 55, 106], np.array(kps, dtype=np.float32), [1.0] + [0.0] * 511)
    face = convert_insight_faces([f])[0]
    assert face.box == (5.0, 6.0, 55.0, 106.0)
    assert face.size == 50.0
    assert face.detector == "insight"
    # порядок точек сверяем целиком, а не длиной: перестановка глаз роняет сходство
    # 0.898 → 0.403 (bench_calibrate), и проверка `len(landmarks) == 10` её не ловит
    assert face.landmarks == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)


def test_pravilnyy_embedding_float32() -> None:
    emb = [1.0] + [0.0] * 511
    face = convert_insight_faces([FakeFace([0, 0, 10, 10], np.zeros((5, 2)), emb)])[0]
    assert face.embedding.dtype == np.float32
    assert face.embedding[0] == 1.0


def test_otpechatok_ne_gljadit_na_massiv_insightface() -> None:
    """Отпечаток — копия, а не вид массива insightface.

    `np.asarray(..., dtype=np.float32)` при уже float32 возвращает тот же массив. Лицо
    из `app.get()` живёт внутри одного вызова, но следующий кадр перезапишет буфер
    движка — и сохранённый отпечаток уедет в архиве чужим лицом. Поэтому копия, как
    в `Face.from_dict`.
    """
    emb = [1.0] + [0.0] * 511
    f = FakeFace([0, 0, 10, 10], np.zeros((5, 2)), emb)
    face = convert_insight_faces([f])[0]

    f.normed_embedding[0] = 0.0        # так же делает движок на следующем кадре

    assert face.embedding[0] == 1.0
    assert not np.shares_memory(face.embedding, f.normed_embedding)


def test_pustoy_vhod_dast_pustoy_vyhod() -> None:
    assert convert_insight_faces([]) == []


# --- вырожденный отпечаток не попадает в архив ------------------------------------
#
# YuNet-путь такую проверку уже держит (задача 5). Здесь то же самое для InsightFace:
# `normed_embedding` даёт сама модель, и на сбойном кропе это могут быть нули или NaN.
# Такой вектор без единой ошибки проезжает base64 в оглавлении и отравляет каждое
# сравнение дальше: сходство с ним выходит nan, и фото исчезает из выдачи молча,
# без единого признака того, что детектор лицо всё-таки нашёл.


def test_nulevoy_otpechatok_ne_popadaet_v_arhiv() -> None:
    lico = FakeFace([0, 0, 60, 60], np.zeros((5, 2)), [1.0] + [0.0] * 511)
    degenerat = FakeFace([100, 100, 160, 160], np.zeros((5, 2)), [0.0] * 512)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        faces = convert_insight_faces([degenerat, lico])

    assert len(faces) == 1
    assert faces[0].box == (0.0, 0.0, 60.0, 60.0)
    assert np.isfinite(faces[0].embedding).all()


def test_nan_ot_modeli_toze_ne_lico() -> None:
    normalnaya = [1.0] + [0.0] * 511
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        faces = convert_insight_faces([
            FakeFace([0, 0, 60, 60], np.zeros((5, 2)), normalnaya),
            FakeFace([100, 100, 160, 160], np.zeros((5, 2)), [np.nan] + normalnaya[1:]),
            FakeFace([200, 200, 260, 260], np.zeros((5, 2)), [np.inf] + normalnaya[1:]),
        ])
    assert len(faces) == 1
    assert faces[0].box == (0.0, 0.0, 60.0, 60.0)


def test_otpechatok_chuzhoy_shiriny_toze_ne_lico() -> None:
    """Правило теперь одно на все пути: годен только вектор ровно в 512 чисел.

    Модель другой сборки (или урезанный кэш buffalo) отдаёт другую ширину, и такой
    отпечаток живёт до первого `matmul` — там он падает на весь прогон.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        faces = convert_insight_faces([
            FakeFace([0, 0, 60, 60], np.zeros((5, 2)), [1.0] * 256),
            FakeFace([100, 100, 160, 160], np.zeros((5, 2)), [1.0] + [0.0] * 511),
        ])
    assert len(faces) == 1
    assert faces[0].box == (100.0, 100.0, 160.0, 160.0)


def test_otbrochennoe_lico_dvizhok_otschityvaet() -> None:
    """Отбраковка по отпечатку — числом, а не молчанием (см. то же у `YunetEngine`).

    Разбор папки читает счётчик и кладёт его в сводку, чтобы отчёт сказал
    «лицо отброшено как непригодное», а не «лица не было». Мелкая рамка, отсеянная
    по `min_face`, в счётчик не входит: это решение детектора, а не испорченные данные.
    """
    normalnaya = [1.0] + [0.0] * 511

    class Prilozhenie:
        def __init__(self, lica: list[Any]) -> None:
            self._lica = lica

        def get(self, image: np.ndarray) -> list[Any]:
            return list(self._lica)

    engine = InsightEngine()
    engine._app = Prilozhenie([
        FakeFace([0, 0, 60, 60], np.zeros((5, 2)), normalnaya),
        FakeFace([100, 100, 160, 160], np.zeros((5, 2)), [0.0] * 512),      # нули
        FakeFace([200, 200, 260, 260], np.zeros((5, 2)), [1.0] * 256),      # чужая ширина
        FakeFace([300, 300, 310, 310], np.zeros((5, 2)), normalnaya),       # 10 px — мелко
    ])
    assert engine.otsortirovannye_lica == 0

    lica = engine.detect(np.zeros((10, 10, 3), dtype=np.uint8))

    assert len(lica) == 1 and lica[0].box == (0.0, 0.0, 60.0, 60.0)
    assert engine.otsortirovannye_lica == 2


# --- распознаватель для разделения с YuNet (режим «Оба») --------------------------


class FakeApp:
    """Копия устройства FaceAnalysis: словарь моделей по имени задачи."""

    def __init__(self, models: dict[str, Any]) -> None:
        self.models = models

    def get(self, image: np.ndarray) -> list[Any]:
        return []


def test_dvizhek_o_tdaet_gotovyy_razpoznavatel() -> None:
    """BothEngine берёт arcface отсюда, вместо второй загрузки 174 МБ весов."""
    recognition = object()
    engine = InsightEngine()
    assert engine.recognition_model is None        # до load() модели нет
    engine._app = FakeApp({"detection": object(), "recognition": recognition})
    assert engine.recognition_model is recognition
    assert engine.detect(np.zeros((10, 10, 3), dtype=np.uint8)) == []


# --- откуда берутся веса: архив приложения или кэш (задача T25) ------------------------


class _KartaFaceAnalysis:
    """Записывает, чем его построили, и притворяется поднявшимся `FaceAnalysis`.

    Настоящий конструктор читает 191 МБ onnx, поэтому здесь только журнал вызовов:
    проверяется единственный вопрос — что именно передали в `name`.
    """

    vyzovy: list[dict[str, Any]] = []

    def __init__(self, name=None, allowed_modules=None, **kwargs) -> None:
        type(self).vyzovy.append({"name": name, "allowed_modules": allowed_modules})
        self.models: dict[str, Any] = {}

    def prepare(self, ctx_id: int, **kwargs) -> None:
        pass


def _podstavit_insightface(monkeypatch: pytest.MonkeyPatch) -> type[_KartaFaceAnalysis]:
    """В подменённый `insightface` локальный импорт внутри `load()` берёт карту."""
    import types

    karta = type("FaceAnalysis", (_KartaFaceAnalysis,), {"vyzovy": []})
    modul_prilozheniya = types.ModuleType("insightface.app")
    modul_prilozheniya.FaceAnalysis = karta
    koren = types.ModuleType("insightface")
    koren.app = modul_prilozheniya
    monkeypatch.setitem(sys.modules, "insightface", koren)
    monkeypatch.setitem(sys.modules, "insightface.app", modul_prilozheniya)
    return karta


def test_zagruzka_peredayet_papku_vesam_esli_ona_na_meste(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Каталог весов внутри архива передаётся путём: библиотека тогда не лезет в сеть.

    `FaceAnalysis(name="buffalo_l")` при отсутствии кэша `~/.insightface` скачивает
    290 МБ. Переданный каталог insightface 2.0 принимает как есть — и на чужом Маке
    приложение работает офлайн.
    """
    papka = tmp_path / "arhiv" / "models" / "buffalo_l"
    papka.mkdir(parents=True)
    karta = _podstavit_insightface(monkeypatch)

    InsightEngine(buffalo_l=papka).load()

    assert karta.vyzovy[0]["name"] == str(papka), karta.vyzovy
    assert karta.vyzovy[0]["allowed_modules"] == ["detection", "recognition"]


def test_bez_papki_vesam_peredayotsja_ime_dlja_dokachki(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Каталога нет (или его не передали) — прежний путь: имя пакета, пусть качает.

    Молча построить FaceAnalysis по несуществующему пути значило бы убрать единственный
    способ достать веса, который у приложения был.
    """
    karta = _podstavit_insightface(monkeypatch)

    InsightEngine().load()

    assert karta.vyzovy[0]["name"] == "buffalo_l", karta.vyzovy


def test_fabrika_peredayet_papku_vesam_obom_dvizkam(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`make_engine` обязан донести путь до распознавателя в обоих режимах.

    Забытый аргумент не виден на этом Маке: здесь веса и так лежат в `~/.insightface`.
    На чужой машине он проявился бы как «приложение просит интернет», и проверить это
    можно только здесь.
    """
    from core.engine import make_engine

    papka = tmp_path / "arhiv" / "models" / "buffalo_l"
    papka.mkdir(parents=True)
    karta = _podstavit_insightface(monkeypatch)

    make_engine("insight", tmp_path / "net.onnx", papka).load()
    # В режиме «Оба» грузим только распознаватель: YuNet читает файл с диска, а его-то
    # как раз и нет — проверяем не его, а то, что путь дошёл до второго движка.
    make_engine("both", tmp_path / "net.onnx", papka)._insight.load()

    assert [v["name"] for v in karta.vyzovy] == [str(papka), str(papka)], karta.vyzovy
