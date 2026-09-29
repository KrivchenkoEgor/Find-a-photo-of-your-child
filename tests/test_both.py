"""Режим «Оба»: слияние лиц двух детекторов и фабрика движков.

Тесты чистые: никаких insightface и никаких моделей — движки подставляются
фиктивными, а лица строятся из numpy-нулей. Личные фото из data/ и ref/ здесь
не участвуют.
"""

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest
from pytest import MonkeyPatch

from core.both import BothEngine, iou, merge_faces
from core.engine import Face, make_engine


def lico(box: Sequence[float], detector: str = "insight",
         emb: np.ndarray | None = None) -> Face:
    return Face(box=box, landmarks=None,
                embedding=emb if emb is not None else np.zeros(512, dtype=np.float32),
                detector=detector)


def test_iou_na_odinakovyh_ramkah_odin() -> None:
    a = lico((0, 0, 10, 10))
    assert iou(a, a) == pytest.approx(1.0)


def test_iou_bez_peresecheniya_nol() -> None:
    assert iou(lico((0, 0, 10, 10)), lico((100, 100, 120, 130))) == 0.0


def test_soedinennyh_lic_ne_dublirodet() -> None:
    """YuNet нашёл то же лицо чуть иначе — оставляем одну рамку, более крупную."""
    primary = [lico((0, 0, 100, 100), "insight")]
    secondary = [lico((2, 2, 98, 98), "yunet")]
    assert len(merge_faces(primary, secondary)) == 1


def test_pri_soedinenii_ostaetsya_ramka_krupnee() -> None:
    primary = [lico((0, 0, 50, 50), "insight")]
    secondary = [lico((-5, -5, 60, 60), "yunet")]
    merged = merge_faces(primary, secondary)
    assert len(merged) == 1
    assert merged[0].box == (-5.0, -5.0, 60.0, 60.0)


def test_raznye_lica_ostaetsya_oboe() -> None:
    primary = [lico((0, 0, 40, 40), "insight")]
    secondary = [lico((200, 200, 260, 260), "yunet")]
    assert len(merge_faces(primary, secondary)) == 2


def test_pustye_vhody() -> None:
    assert merge_faces([], []) == []


# --- правило слияния против обычного IoU ------------------------------------------
#
# Специфика требует «пересечение больше половины площади МЕНЬШЕЙ рамки». Здесь это
# не украшение: вложенная мелкая рамка — типичный случай, когда два детектора нашли
# одно и то же лицо чуть по-разному.


def klassicheskij_iou(a: Face, b: Face) -> float:
    """Пересечение через объединение — то, чем измеряют рамки обычно."""
    ax1, ay1, ax2, ay2 = a.box
    bx1, by1, bx2, by2 = b.box
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def test_vlozhennaya_ramka_sitaetsya_odnoj_litsom() -> None:
    """Мелкая рамка внутри крупной — это одно лицо. Обычный IoU его бы удвоил."""
    big = lico((0, 0, 100, 100), "insight")
    small = lico((40, 40, 60, 60), "yunet")
    # пересечение равно площади мелкой рамки: относительно неё — 100%
    assert iou(big, small) == pytest.approx(1.0)
    # классический IoU при этом 0.04 — под порог 0.5 то же лицо прошло бы вторым
    classic = klassicheskij_iou(big, small)
    assert classic == pytest.approx(0.04)
    assert classic < 0.5
    merged = merge_faces([big], [small])
    assert len(merged) == 1
    assert merged[0].box == (0.0, 0.0, 100.0, 100.0)      # осталась крупная рамка


def test_obratnyy_poryadok_vhodov_ne_menayet_resultat() -> None:
    """Мелкая рамка пришла первой: она всё равно уступает место крупной."""
    big = lico((0, 0, 100, 100), "insight")
    small = lico((40, 40, 60, 60), "yunet")
    merged = merge_faces([small], [big])
    assert len(merged) == 1
    assert merged[0].box == (0.0, 0.0, 100.0, 100.0)


def test_pri_odinakovyh_ramkah_ostaetsya_primary() -> None:
    """При равных размерах никто никого не вытесняет: лицо остаётся у основного."""
    primary = lico((0, 0, 50, 50), "insight")
    secondary = lico((1, 1, 51, 51), "yunet")
    merged = merge_faces([primary], [secondary])
    assert len(merged) == 1
    assert merged[0] is primary


def test_krupnee_eto_menshaya_storona_a_ne_ploschad_ramki() -> None:
    """Квадрат 60×60 крупнее полосы 100×40, хотя площадь у полосы больше.

    Раньше этот выбор не проверялся ничем: все рамки в тестах были квадратами, и
    правило «меньшая сторона» от правила «площадь» не отличались. Разница здесь не
    косметическая — докстринг `merge_faces` площадь прямо отвергает: узкая полоса
    обгоняет настоящее лицо по площади, а пикселей на отпечаток в ней меньше
    (4000 против 3600 при меньшей стороне 40 против 60).
    """
    kvadrat = lico((0, 0, 60, 60), "insight")       # сторона 60, площадь 3600
    polosa = lico((0, 0, 100, 40), "yunet")         # сторона 40, площадь 4000
    assert polosa.size < kvadrat.size
    assert iou(kvadrat, polosa) > 0.5               # это одно лицо: 2400/3600
    merged = merge_faces([kvadrat], [polosa])
    assert len(merged) == 1
    assert merged[0].box == (0.0, 0.0, 60.0, 60.0)
    assert merged[0] is kvadrat


# --- вложенные рамки сворачиваются в одном детекторе тоже (задача 7, замечание 4) ---
#
# `kept = list(primary)` означало, что secondary проверяется на коллизии только со
# вставшими рядом лицами, а primary между собой не сверялся никогда. Два таких лица
# одного детектора — не выдумка: детекторные NMS давят рамки по IoU, а не по
# пересечению относительно меньшей рамки, и вложенная рамка спокойно проходит.


def test_vlozhennye_ramki_vnutri_primary_svorachivayutsya() -> None:
    """Мелкая рамка внутри крупной у одного и того же детектора — это одно лицо."""
    melkoe = lico((45, 45, 65, 65), "insight")
    krupnoe = lico((0, 0, 100, 100), "insight")
    merged = merge_faces([melkoe, krupnoe], [])
    assert [f.box for f in merged] == [(0.0, 0.0, 100.0, 100.0)]
    # тот же результат, если крупная рамка стоит в списке первой
    assert [f.box for f in merge_faces([krupnoe, melkoe], [])] == [(0.0, 0.0, 100.0, 100.0)]


def test_kandidat_ne_povtoryaet_ostavlennoe_lico() -> None:
    """Случай из замечания: кандидат сталкивается и с мелкой, и с крупной рамкой.

    Старый `hit = next(...)` брал первую совпавшую, ставил туда кандидата и оставлял
    крупную рамку нетронутой: то же лицо выживало дважды с пересечением 1.0, а
    отпечаток мелкой рамки исчезал молча.
    """
    melkoe = lico((45, 45, 65, 65), "insight")
    krupnoe = lico((0, 0, 100, 100), "insight")
    kandidat = lico((10, 10, 80, 80), "yunet")
    merged = merge_faces([melkoe, krupnoe], [kandidat])
    assert [f.box for f in merged] == [(0.0, 0.0, 100.0, 100.0)]
    assert merged[0] is krupnoe


def test_vse_lichnye_vyzivshie_ramki_ne_sovpadayut() -> None:
    """Инвариант слияния: после `merge_faces` любых двух наборов совпавших рамок нет.

    Перебор на сетке координат, где вложенные рамки встречаются сплошь и рядом:
    он ловит и «внутри одного детектора», и «кандидат столкнулся с двумя сразу» —
    обе дыры, которые точечные тесты выше пропускают по отдельности.
    """
    rng = np.random.default_rng(7)
    for _ in range(200):
        faces: list[Face] = []
        for det in ("insight", "yunet"):
            for _ in range(6):
                x1, y1 = (int(v) for v in rng.integers(0, 300, size=2))
                w, h = (int(v) for v in rng.integers(5, 120, size=2))
                faces.append(lico((x1, y1, x1 + w, y1 + h), det))
        merged = merge_faces(faces[:6], faces[6:])
        assert merged, "на случайных рамках слияние не должно терять всё подчистую"
        for i, a in enumerate(merged):
            for b in merged[i + 1:]:
                assert iou(a, b) <= 0.5, (a.box, b.box)
        # порядок — по левому верхнему углу; сравниваем рамки, а не лица: `==` на двух
        # разных `Face` бросает ValueError из-за поля `embedding` (см. tests/test_face.py)
        boxes = [f.box for f in merged]
        assert boxes == sorted(boxes, key=lambda b: (b[0], b[1]))


# --- BothEngine -------------------------------------------------------------------


class FakeEngine:
    def __init__(self, name, faces):
        self.name = name
        self._faces = faces
        self.loaded = False

    def load(self):
        self.loaded = True

    def detect(self, image):
        return list(self._faces)


def test_obasha_gruzit_ooba_i_daeet_obedinenie() -> None:
    a = FakeEngine("insight", [lico((0, 0, 40, 40), "insight")])
    b = FakeEngine("yunet", [lico((200, 200, 260, 260), "yunet")])
    eng = BothEngine(b, a)
    eng.load()
    assert a.loaded and b.loaded
    got = eng.detect(np.zeros((10, 10, 3), dtype=np.uint8))
    assert {f.detector for f in got} == {"insight", "yunet"}


class Inspektiruemyj:
    """Движок, который пишет в общий журнал порядок вызовов и что ему передали.

    `sprosil_model` у юнета заведомо False: если BothEngine придёт к нему за
    распознавателем вместо того чтобы отдать свой, тест это поймает.
    """

    def __init__(self, name: str, faces: Sequence[Face], journal: list[str],
                 recognition: Any = None, sprosil_model: bool = False) -> None:
        self.name = name
        self._faces = list(faces)
        self._journal = journal
        self._recognition = recognition
        self._sprosil_model = sprosil_model
        self.loaded = False
        self.adopted: list[Any] = []

    @property
    def recognition_model(self) -> Any:
        if not self._sprosil_model:
            raise AssertionError(f"{self.name}: распознаватель запрошен там, где его нет")
        return self._recognition

    def load(self) -> None:
        self._journal.append(self.name)
        self.loaded = True

    def adopt_recognition(self, model: Any) -> None:
        self.adopted.append(model)

    def detect(self, image: np.ndarray) -> list[Face]:
        return list(self._faces)


def test_obasha_gruzit_insight_pervoy_i_peredayet_razpoznavanie_yunetu() -> None:
    """Одна и та же arcface на 174 МБ читается один раз: образ отдаётся юнету."""
    zhurnal: list[str] = []
    model = object()
    yunet = Inspektiruemyj("yunet", [], zhurnal)
    insight = Inspektiruemyj("insight", [], zhurnal, recognition=model,
                             sprosil_model=True)
    BothEngine(yunet, insight).load()
    assert zhurnal == ["insight", "yunet"]      # buffalo_l первой, юнет второй
    assert yunet.adopted == [model]             # ровно этот объект, без второй загрузки


def test_obasha_ne_padayet_esli_peredavat_nechego() -> None:
    """Движок без распознавателя (старый FakeEngine) — штатный случай, не ошибка."""
    yunet = FakeEngine("yunet", [])
    insight = FakeEngine("insight", [])
    BothEngine(yunet, insight).load()           # ни AttributeError, ни TypeError
    assert yunet.loaded and insight.loaded


def test_povtornyy_load_ne_chitaet_modeli_zanovo() -> None:
    """`load()` вызывают повторно: рабочий поток задачи 9 зовёт его на каждый прогон.

    До правки второй `load()` был гарантированным провалом: `insight.load()` заново
    собирал `FaceAnalysis` (13–21 с), юнет после этого принимал `adopt_recognition()`
    уже загруженным и бросал RuntimeError. Второе сканирование в одной сессии умирало
    с сообщением, которое не-программист не может ни понять, ни применить.
    """
    zhurnal: list[str] = []
    model = object()
    yunet = Inspektiruemyj("yunet", [], zhurnal)
    insight = Inspektiruemyj("insight", [], zhurnal, recognition=model, sprosil_model=True)
    eng = BothEngine(yunet, insight)
    eng.load()
    eng.load()
    eng.load()
    assert zhurnal == ["insight", "yunet"]      # ровно один проход, а не три
    assert yunet.adopted == [model]             # и распознаватель отдаётся один раз
    assert eng.detect(np.zeros((10, 10, 3), dtype=np.uint8)) == []   # движок живой


def test_povtornyy_load_na_zivom_yunete_ne_padayet(tmp_path: Path,
                                                   monkeypatch: MonkeyPatch) -> None:
    """То же самое на настоящем `YunetEngine` — он и был источником падения.

    Фиктивный движок выше фиксирует порядок вызовов, а здесь важна именно связка:
    живой юнет после `load()` отказывается принимать распознаватель, и второй `load()`
    у `BothEngine` обязан до этого отказа вообще не доходить. Весов не читаем:
    детектор и распознаватель подставлены.
    """
    import core.yunet as yunet_mod
    from core.yunet import YunetEngine

    monkeypatch.setattr(yunet_mod.cv2, "FaceDetectorYN_create", lambda *a, **k: object())
    monkeypatch.setattr(yunet_mod, "_load_recognition", lambda: object())
    model_path = tmp_path / "yunet.onnx"
    model_path.write_bytes(b"")

    class Insight:
        name = "insight"

        def __init__(self) -> None:
            self.zagruzki = 0
            self.model = object()

        @property
        def recognition_model(self) -> Any:
            return self.model

        def load(self) -> None:
            self.zagruzki += 1

        def detect(self, image: np.ndarray) -> list[Face]:
            return []

    insight = Insight()
    eng = BothEngine(YunetEngine(model_path), insight)
    eng.load()
    eng.load()                        # до правки: RuntimeError «юнет уже загружен»
    assert insight.zagruzki == 1      # и 13–21 с на второй FaceAnalysis не тратятся


def test_obasha_skadyvayut_otsortirovannye_oba_putya() -> None:
    """Режим «Оба» обязан отдать сводке потери обоих путей одним числом.

    Дубли, которые сворачивает `merge_faces`, сюда не входят: это то же лицо,
    найденное дважды, а не потерянные данные.
    """
    class Dvizhok:
        name = "fake"

        def __init__(self, otsortirovano: int) -> None:
            self.otsortirovannye_lica = otsortirovano

        def load(self) -> None:
            pass

        def detect(self, image: np.ndarray) -> list[Face]:
            return []

    assert BothEngine(Dvizhok(2), Dvizhok(3)).otsortirovannye_lica == 5
    # чужой движок без счётчика — не ошибка, а «нечего доложить»
    assert BothEngine(FakeEngine("yunet", []),
                      FakeEngine("insight", [])).otsortirovannye_lica == 0


# --- фабрика ----------------------------------------------------------------------


def test_fabrika_znaet_tri_rezhima(tmp_path: Path) -> None:
    for mode in ("yunet", "insight", "both"):
        assert make_engine(mode, tmp_path / "net.onnx").name == mode
    with pytest.raises(ValueError):
        make_engine("ne_takoy", tmp_path / "net.onnx")
