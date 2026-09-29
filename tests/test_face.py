"""`Face` — валюта между модулями: проверка полей, размеров и круга через dict.

Круг через оглавление проверяется в обе стороны: непригоден не только отпечаток (NaN,
Inf, чужая ширина, нули), но и рамка. JSON перевозит `Infinity` честно, и лицо с такой
рамкой доезжало до отрисовки в сетке, где `int(inf * k)` роняло показ всего архива уже
после того, как разбор успешно кончился.

Отпечатки собираются синтетически (`np.full`, `np.zeros`), реальные фото из data/ и ref/
здесь не участвуют: это снимки конкретного ребёнка, они личные.
"""

import base64
import math
from dataclasses import FrozenInstanceError, replace
from typing import Any

import numpy as np
import pytest

from core.engine import Face, embeddings_of, prigoden_otpechatok


def lico(detector: str = "yunet", side: float = 100.0, value: float = 0.1) -> Face:
    return Face(
        box=(10.0, 20.0, 10.0 + side, 20.0 + side * 1.5),
        landmarks=tuple(float(i) for i in range(10)),
        embedding=np.full(512, value, dtype=np.float32),
        detector=detector,
    )


def test_box_i_razmer() -> None:
    f = lico()
    assert (f.x1, f.y1, f.x2, f.y2) == (10.0, 20.0, 110.0, 170.0)
    assert f.size == 100.0


def test_krug_cherez_dict_teryaet_nol_i_odin() -> None:
    f = lico()
    back = Face.from_dict(f.to_dict())
    assert back.detector == "yunet"
    assert back.box == f.box
    assert np.allclose(back.embedding, f.embedding)
    assert back.landmarks == f.landmarks


def test_embeddings_of_dast_matricu_po_strokaam() -> None:
    m = embeddings_of([lico(), lico()])
    assert m.shape == (2, 512)
    assert embeddings_of([]).shape == (0, 512)


def test_size_beret_menshuyu_storonu() -> None:
    """`size` — меньшая сторона рамки: по ней судят, крупное ли лицо в кадре."""
    face = Face(box=(0.0, 0.0, 200.0, 40.0), landmarks=None,
                embedding=np.zeros(512, dtype=np.float32), detector="yunet")
    assert face.size == 40.0


def test_zamorozeno_a_replace_dast_novoe_lico() -> None:
    """Задачи 5–8 меняют лицо только через `dataclasses.replace` — проверяю это здесь.

    Иначе «правка» рамки фильтром молча трогала бы объект, который уже лежит в кэше
    и в списке у интерфейса.
    """
    f = lico()
    with pytest.raises(FrozenInstanceError):
        setattr(f, "box", (0.0, 0.0, 1.0, 1.0))
    bigger = replace(f, box=(0.0, 0.0, 300.0, 300.0), detector="insight")
    assert bigger.size == 300.0 and bigger.detector == "insight"
    assert f.size == 100.0 and f.detector == "yunet"          # оригинал не тронут
    assert bigger.embedding is f.embedding                     # replace копирует ссылку


def test_krug_sohranyaet_float32_i_vse_znacheniya() -> None:
    """Кэш пишет JSON, а там float32 становится Python float — обратно должно вернуться
    ровно float32 и без потери единого значения. Иначе пороги сходства поплывут."""
    f = lico(value=0.37)
    back = Face.from_dict(f.to_dict())
    assert back.embedding.dtype == np.float32
    assert back.embedding.shape == (512,)
    assert np.array_equal(back.embedding, f.embedding)


def test_net_landmarkov_prohodit_cherez_krug() -> None:
    """Поле `landmarks` опциональное: отсутствующее обязано проходить круг как None,
    а не превращаться в пустой список или в нули. Так строят лица хелперы задач 7–9.

    Отпечаток здесь заведомо пригодный: `from_dict` непригодный не возвращает (см.
    `test_neprigodnyy_otpechatok_ne_prohodit_cherez_krug`), и нулевой вектор в этом
    круге участвовать уже не может.
    """
    b = Face(box=(0.0, 0.0, 10.0, 10.0), landmarks=None,
             embedding=np.full(512, 0.1, dtype=np.float32), detector="insight")
    assert b.to_dict()["landmarks"] is None
    assert Face.from_dict(b.to_dict()).landmarks is None


def test_embeddings_of_vsegda_float32() -> None:
    """Матрица уходит на косинусное сравнение: dtype и пустой случай зафиксированы."""
    assert embeddings_of([lico()]).dtype == np.float32
    empty = embeddings_of([])
    assert empty.dtype == np.float32 and empty.shape == (0, 512)


def test_sravnenie_lic_cherez_eq_ne_rabotaet() -> None:
    """Замер, а не вымысел: `==` на двух разных объектах Face падает ValueError.

    Причина — поле `embedding`: numpy сравнивает поэлементно и отдаёт массив, а dataclass
    не умеет свести его к одному True. `hash()` недоступен по той же причине (ndarray не
    хешируется). Поэтому «то же лицо или нет» проверяют по полям (`box`, `detector`) или
    по идентичности объекта, и в множества лица не кладут — так и в задачах 5–9 по плану.
    Тест стоит ровно для того, чтобы это поведение не приняли за сломанный кэш.
    """
    a, b = lico(), lico()
    assert a == a                                   # идентичность работает
    with pytest.raises(ValueError):
        _ = a == b                                  # равенство двух объектов — падение
    with pytest.raises(TypeError):
        hash(a)


def test_poly_lico_opisany_v_dict_bez_lishnego() -> None:
    """Словарь — контракт сериализации: четыре ключа, и ничего, кроме них.

    Отпечаток лежит в словаре base64 от тех же 2048 байт float32, а не списком из 512
    чисел: список в JSON весит на лицо 8–11 КБ, то есть на архиве в 20 000 снимков —
    сотни мегабайт вместо примерно ста.
    """
    data: dict[str, Any] = lico().to_dict()
    assert set(data) == {"box", "landmarks", "embedding", "detector"}
    raw = base64.b64decode(data["embedding"])
    assert len(raw) == 512 * 4                      # ровно те же байты массива
    assert len(np.frombuffer(raw, dtype=np.float32)) == 512
    assert len(data["landmarks"]) == 10


# --- пригодность отпечатка: одно правило на весь проект ----------------------------
#
# Вырожденный вектор — NaN, Inf, нулевой или не 512 чисел — не имеет права ни ронять
# сравнение, ни молча обнулять архив. Ответ на вопрос «годен ли отпечаток» живёт здесь,
# в `engine`, и его же применяют оба детектора, оглавление (`from_dict`) и `matcher`.
#
# До правки правило было написано один раз в `insight.py`, второй раз — внутри
# `yunet.detect`, и третьим, иным, его никто не проверял ни при чтении оглавления,
# ни при сравнении. Старые строки оглавления (приложение писало их без проверки) при
# этом приезжали целыми и роняли разбор уже на первом же фото.


def lico_s_otpechatkom(emb: np.ndarray, box: tuple[float, float, float, float] =
                       (0.0, 0.0, 50.0, 50.0)) -> Face:
    return Face(box=box, landmarks=None, embedding=emb, detector="insight")


# Ключи — латиницей: они попадают в имена параметров pytest, а кириллицу тот
# экранирует последовательностями вида «\u0448\u0438», и вывод прогонов нечитаем.
NEPRIGODNYE = {
    "nan": np.full(512, np.nan, dtype=np.float32),
    "inf": np.full(512, np.inf, dtype=np.float32),
    "nuli": np.zeros(512, dtype=np.float32),
    "shirina_256": np.full(256, 0.1, dtype=np.float32),
    "ne_vektor": np.ones((512, 1), dtype=np.float32),
}


def test_prigoden_tolko_konechnyy_polnometrazhnyy_nenulevoy() -> None:
    assert prigoden_otpechatok(np.full(512, 0.1, dtype=np.float32))
    assert prigoden_otpechatok(lico().embedding)
    for imya, emb in NEPRIGODNYE.items():
        assert not prigoden_otpechatok(emb), imya


def test_predikat_ne_brosaet_na_chuzhih_dannyh() -> None:
    """Предикат отвечает на вопрос, а не бросает: его зовут на строках из кэша."""
    for podrostok in (None, "не массив", [], object()):
        assert prigoden_otpechatok(podrostok) is False


# Случай «ne_vektor» здесь не параметризован: base64 хранит плоские байты float32,
# и колонка (512, 1) после tobytes() обратно разворачивается в ровный вектор из 512
# чисел — то есть в пригодный. Форма сериализацией не переживается, проверять нечего.
@pytest.mark.parametrize("imya", ["nan", "inf", "nuli", "shirina_256"])
def test_neprigodnyy_otpechatok_ne_prohodit_cherez_krug(imya: str) -> None:
    """`from_dict` на битом отпечатке — ValueError, а не лицо с нулевым сходством.

    Оглавление ловит ValueError и означает этим «снимок разберётся заново»: ни падения
    рабочего потока, ни отравленной строки, которая ездила бы по архиву дальше.
    """
    sdelano = lico_s_otpechatkom(NEPRIGODNYE[imya]).to_dict()
    with pytest.raises(ValueError):
        Face.from_dict(sdelano)


def test_prigodnyy_otpechatok_prohodit_krug_cifra_v_cifru() -> None:
    """Проверка не должна портить годное лицо: те же байты, что и на входе."""
    ishodnoe = lico(value=0.37)
    back = Face.from_dict(ishodnoe.to_dict())
    assert back.embedding.dtype == np.float32
    assert np.array_equal(back.embedding, ishodnoe.embedding)


def test_embeddings_of_propuskaet_neprigodnye_stroki() -> None:
    """Матрица сравнения собирается только из того, что можно сравнить.

    Один битый вектор не обнуляет весь снимок: строки пригодных лиц остаются на месте,
    и `matcher` получает честный выбор из того, что реально найдено.
    """
    m = embeddings_of([lico(), lico_s_otpechatkom(NEPRIGODNYE["nan"]),
                       lico_s_otpechatkom(NEPRIGODNYE["shirina_256"])])
    assert m.shape == (1, 512)
    assert m.dtype == np.float32
    assert embeddings_of([lico_s_otpechatkom(NEPRIGODNYE["nan"])]).shape == (0, 512)


def test_pravilo_odno_na_vse_puti() -> None:
    """Копии правила в путях детекторов запрещены: и там, и там — функция из `engine`."""
    import core.insight as insight_mod
    import core.yunet as yunet_mod

    assert insight_mod.prigoden_otpechatok is prigoden_otpechatok
    assert yunet_mod.prigoden_otpechatok is prigoden_otpechatok


def test_min_face_px_definirovana_odin_raz() -> None:
    """Минимальный размер лица живёт в `engine`, а не по копии у каждого движка.

    Проверка по исходнику, а не «равно 20»: `20 is 20` в Python всегда истина, поэтому
    вторую копию константы сравнение значений не поймало бы вообще. А расходятся копии
    именно так: кто-то один правит число у себя, и на одном архиве «Быстрее» и «Точнее»
    начинают видеть разное лицо.
    """
    import re
    from pathlib import Path

    koren = Path(__file__).resolve().parents[1] / "src" / "core"
    definirov = [imya for imya in ("engine.py", "yunet.py", "insight.py", "both.py")
                 if re.search(r"^MIN_FACE_PX\s*=",
                              (koren / imya).read_text(encoding="utf-8"), re.MULTILINE)]
    assert definirov == ["engine.py"], f"у константы появилось второе место: {definirov}"

    import core.engine as engine_mod
    import core.insight as insight_mod
    import core.yunet as yunet_mod

    assert engine_mod.MIN_FACE_PX == 20
    # импортируется, а не переопределяется: имя в чужом модуле живёт, но определено
    # ровно один раз — это и проверяется выше по исходнику
    assert yunet_mod.MIN_FACE_PX == engine_mod.MIN_FACE_PX
    assert insight_mod.MIN_FACE_PX == engine_mod.MIN_FACE_PX


def test_obazhdy_puti_filteruut_po_odnomu_razmeru() -> None:
    """Значение по умолчанию у обоих движков — та же константа, а не своё число.

    Иначе «правило одно» остаётся словами: движок с дефолтом 20 и движок с дефолтом 40
    дают одинаковые тесты на подставных строках и разный архив на живом.
    """
    import inspect

    import core.engine as engine_mod
    from core.insight import InsightEngine
    from core.yunet import YunetEngine

    for klass in (YunetEngine, InsightEngine):
        parametr = inspect.signature(klass.__init__).parameters["min_face"]
        assert parametr.default == engine_mod.MIN_FACE_PX, klass.__name__


# --- рамка из оглавления: то же правило конечности, что и у отпечатка -----------------
#
# Проверка отпечатка была, проверки рамки — не было. Зря: JSON честно перевозит
# `Infinity`, и лицо с такой рамкой приезжало из оглавления целым. Жило оно ровно до
# карточки в сетке, где `int(inf * k)` бросает OverflowError — уже в потоке интерфейса,
# после успешно законченного разбора папки. Одна битая строка обрывала показ всего
# архива вместо того, чтобы разбрать один снимок заново.


@pytest.mark.parametrize("box", [
    pytest.param((0.0, 0.0, float("inf"), float("inf")), id="inf"),      # из JSON
    pytest.param((0.0, 0.0, float("nan"), 50.0), id="nan"),
    pytest.param((float("-inf"), 0.0, 50.0, 50.0), id="-inf"),
    pytest.param((0.0, 0.0, 50.0), id="tri-chisla"),
    pytest.param((0.0, 0.0, 50.0, 50.0, 0.0), id="pyat-chisel"),
])
def test_bitaya_ramka_ne_prohodit_cherez_krug(box: tuple) -> None:
    """`from_dict` на битой рамке — ValueError, а не лицо, которое нарисует себя колом."""
    sdelano = lico_s_otpechatkom(np.full(512, 0.1, dtype=np.float32), box=box).to_dict()
    with pytest.raises(ValueError):
        Face.from_dict(sdelano)


def test_celevaya_ramka_prohodit_krug_bez_izmenenij() -> None:
    """Проверка не имеет права подпортить годную рамку: те же четыре числа на выходе."""
    ishodnoe = lico()
    back = Face.from_dict(ishodnoe.to_dict())
    assert back.box == ishodnoe.box
    assert all(math.isfinite(v) for v in back.box)


def test_int_ot_bitoy_ramki_eto_imenno_padenie_a_ne_nol() -> None:
    """Почему рамка опасна именно в потоке интерфейса: `int(inf)` — OverflowError.

    Это про строку `int(v * k)` в `ui/results_view._ramka`, где координаты рамки
    пересчитываются в пиксели миниатюры. Поймать такое как «не число» нельзя, поэтому
    оно не должно доходить до отрисовки: тест держит сам факт опасности.
    """
    with pytest.raises(OverflowError):
        int(float("inf") * 0.09)
