"""Чистая логика окна эталона: карточки, отметки и сборка эталона без Qt.

Обещание модуля одно, и от него зависит весь поиск: **ответ не должен зависеть от
того, куда человек ткнул мышью первым.** Прежний порядок «anchor = то, что
подтвердили первым» в сетке с галочками превратил бы эталон в историю движений
курсора: те же самые карточки, отмеченные в другом порядке, давали бы другой поиск.

Тесты без QApplication и без личных снимков: `Face` собирается в память из коротких
numpy-векторов единичной длины, файлы — это `Path`-заглушки, с диска не читается
ничего. Мок ровно один, и он подменяет недостижимую ветку (см. последний тест).

С задачи T14 к обещаниям добавилось третье, и именно оно сейчас спорное: **отмеченное
лицо участвует в поиске, галочка человека — единственное решение.** Число сходства с
главным лицом больше не решает «брать или не брать», оно решает «предупредить или нет»,
и называется в ответе полем `slabo`. В `otvergnutye` остаётся ровно один случай — отпечаток,
которым физически нечем считать, и все проверки этой кучки собраны вокруг непригодных
векторов.
"""

from __future__ import annotations

import dataclasses
import inspect
import itertools
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pytest

from core import etalon
from core.engine import Face, prigoden_otpechatok
from core.etalon import Itog, Kartochka, odinochnye_otmetki, sobrat_etalon
from core.matcher import REF_MIN_SIM, Reference, make_reference, score_photo

A, B, C, D = Path("f1.jpg"), Path("f2.jpg"), Path("f3.jpg"), Path("f4.jpg")
NET = Path("nety-takogo-fajla.jpg")

E1 = [1.0] + [0.0] * 511
E2 = [0.0, 1.0] + [0.0] * 510


def vektor_blizkogo(x: float) -> list[float]:
    """Вектор с косинусом `x` к E1: `[x, sqrt(1 - x*x), 0, ...]`.

    Так задают сходство числом, а не «на глаз по вектору»: тест порога обязан знать,
    что легло ровно на границе, а что — на волосок ниже.
    """
    v = np.zeros(512, dtype=np.float32)
    v[0] = x
    v[1] = float(np.sqrt(1 - x * x))
    return v.tolist()


def ramka(storona: float) -> tuple[float, float, float, float]:
    """Квадратная рамка со стороной `storona`: `Face.size` даёт ровно это число."""
    return (0.0, 0.0, storona, storona)


def lico(emb: Sequence[float], box: tuple[float, float, float, float] = ramka(50.0)) -> Face:
    v = np.array(emb, dtype=np.float32)
    return Face(box=box, landmarks=None, embedding=v / np.linalg.norm(v), detector="insight")


def nan_lico(box: tuple[float, float, float, float] = ramka(10.0)) -> Face:
    return Face(box=box, landmarks=None, embedding=np.full(512, np.nan, dtype=np.float32),
                detector="insight")


def nulevoe_lico(box: tuple[float, float, float, float] = ramka(10.0)) -> Face:
    return Face(box=box, landmarks=None, embedding=np.zeros(512, dtype=np.float32),
                detector="insight")


def shirokoe_lico(box: tuple[float, float, float, float] = ramka(10.0)) -> Face:
    """«Вектор из другой модели»: 256 чисел вместо 512.

    Самое опасное из непригодных: `lic.embedding @ anchor.embedding` на нём бросает
    ValueError прямо из BLAS, а не молча даёт мусор.
    """
    return Face(box=box, landmarks=None, embedding=np.full(256, 0.1, dtype=np.float32),
                detector="insight")


NEPRIGODNYE = (nan_lico, nulevoe_lico, shirokoe_lico)


# --- сравнение двух `Itog`: почему не просто `==` -----------------------------------


def trebovat_odinakovy(pervyj: Itog, vtoroj: Itog) -> None:
    """Два `Itog`, собранные из одних и тех же лиц, обязаны совпасть во всём.

    Обычное `pervyj == vtoroj` здесь не годится и даже не проходит: внутри `Reference`
    лежит numpy-массив, а numpy на `==` отвечает массив булев, а не одно `True` (то же
    самое зафиксировано тестом для `Face` в `core/engine.py`). Карточки же сравниваются
    честно: лица в них — одни и те же объекты, и python на этом экономит, не доходя до
    сравнения векторов.
    """
    assert pervyj.karty == vtoroj.karty
    assert pervyj.extra == vtoroj.extra
    assert pervyj.otvergnutye == vtoroj.otvergnutye
    assert (pervyj.anchor is None) == (vtoroj.anchor is None)
    if pervyj.anchor is not None:
        assert pervyj.anchor.lic is vtoroj.anchor.lic
    if pervyj.reference is None:
        assert vtoroj.reference is None
    else:
        assert vtoroj.reference is not None
        assert np.array_equal(pervyj.reference.embeddings, vtoroj.reference.embeddings)


# --- пустой вход и снимок без лиц ----------------------------------------------------


def test_pustoy_vhod_dayet_pustye_karty_i_bez_etalona() -> None:
    itog = sobrat_etalon({}, frozenset())
    assert itog.karty == ()
    assert itog.anchor is None
    assert itog.extra == ()
    assert itog.otvergnutye == ()
    assert itog.reference is None
    assert odinochnye_otmetki({}) == frozenset()


def test_fayl_bez_lic_dayet_odnu_kartochku_s_lico_none() -> None:
    """Карточка «лицо не нашлось» обязана появляться: иначе снимок молча исчезает из
    окна, и человек не отличит «на фото никого нет» от «файл не прочитался»."""
    itog = sobrat_etalon({A: []}, frozenset())
    assert len(itog.karty) == 1
    kartochka = itog.karty[0]
    assert kartochka.put == A
    assert kartochka.nomer == 0
    assert kartochka.lic is None
    assert kartochka.otmecheno is False
    assert kartochka.v_poisk is False
    assert kartochka.slabo is False
    assert kartochka.procent is None


# --- одиночные отметки ---------------------------------------------------------------


def test_odinochnoe_lico_dae_otmetku_i_stanovit_anchor() -> None:
    """Один файл — одно лицо: человек об этом не просил, но так велено.

    Дальше весь путь проходит через `sobrat_etalon`, и ответ обязан быть готов к поиску
    сразу: эталон из одного лица. Числа на карточке при этом нет — сравнивать это лицо
    не с чем, а «100 %» без собеседника человек прочитал бы как «стопроцентное совпадение
    с самим собой», то есть как ответ на вопрос, которого он не задавал.
    """
    edinoe = lico(E1, box=ramka(120.0))
    lica = {A: [edinoe]}
    otmetki = odinochnye_otmetki(lica)
    assert otmetki == frozenset({(A, 0)})

    itog = sobrat_etalon(lica, otmetki)
    assert itog.anchor is itog.karty[0]
    assert itog.anchor.otmecheno is True
    assert itog.anchor.procent is None
    assert itog.anchor.v_poisk is True
    assert itog.anchor.slabo is False, "без остальных лиц предупреждать не о чем"
    assert itog.extra == ()
    assert itog.otvergnutye == ()
    assert itog.reference is not None
    assert itog.reference.count == 1


def test_tri_lica_na_snimke_ne_dayot_otmetki_vovse() -> None:
    """Угадывать, кто из троих ребёнок, приложение не имеет права: ни одной отметки.

    Файл без лиц и файл с двумя лицами тоже не отмечают себя сами.
    """
    lica = {A: [lico(E1), lico(E2), lico(vektor_blizkogo(0.5))],
            B: [],
            C: [lico(E1)],
            D: [lico(E1), lico(E2)]}
    assert odinochnye_otmetki(lica) == frozenset({(C, 0)})


# --- КЛЮЧЕВОЕ: независимость от порядка кликов ---------------------------------------


def test_anchor_ne_zavisit_ot_poryadka_otmetok() -> None:
    """Главный тест задачи: один и тот же набор отметок, разная очередь вставки —
    один и тот же `Itog` целиком, а не только anchor.

    Перебраны все шесть порядков трёх отметок: реализация, которая выбирает «первую из
    множества», прошла бы этот тест только по случайности.
    """
    melkoe = lico(E1, box=ramka(40.0))
    srednee = lico(vektor_blizkogo(0.6), box=ramka(90.0))
    krupnoe = lico(vektor_blizkogo(0.45), box=ramka(300.0))
    lica = {A: [melkoe], B: [srednee], C: [krupnoe]}
    pary = [(A, 0), (B, 0), (C, 0)]

    itogi = [sobrat_etalon(lica, frozenset(poryadok)) for poryadok in itertools.permutations(pary)]
    for другой in itogi[1:]:
        trebovat_odinakovy(itogi[0], другой)

    itog = itogi[0]
    assert itog.anchor.lic is krupnoe                       # крупнейшее, а не «первое по клику»
    assert [(k.put, k.nomer) for k in itog.karty] == [(A, 1), (B, 1), (C, 1)]
    assert itog.reference is not None
    assert itog.reference.count == 1 + len(itog.extra)


def test_anchor_ne_zavisit_ot_poryadka_faylov_v_slovare() -> None:
    """Anchor выбирается по размеру, а не по позиции в плоском списке: крупнейшее лицо
    стоит последним — и всё равно становится эталонным.

    Порядок самих карточек при этом остаётся порядком словаря: это другое правило, и
    оно не должно влиять на выбор anchor.
    """
    melkoe = lico(E1, box=ramka(60.0))
    krupnoe = lico(vektor_blizkogo(0.5), box=ramka(600.0))
    pervyj = sobrat_etalon({A: [melkoe], B: [krupnoe]}, frozenset({(A, 0), (B, 0)}))
    vtoroj = sobrat_etalon({B: [krupnoe], A: [melkoe]}, frozenset({(A, 0), (B, 0)}))

    assert pervyj.anchor.lic is krupnoe
    assert vtoroj.anchor.lic is krupnoe
    assert [(k.put, k.nomer) for k in pervyj.karty] == [(A, 1), (B, 1)]
    assert [(k.put, k.nomer) for k in vtoroj.karty] == [(B, 1), (A, 1)]


def test_pri_ravenstve_razmerov_anchor_pervyy_v_ploskom_poryadke() -> None:
    """При равных размерах правило одно и то же всегда: первое в плоском порядке.

    Отметки, переданные задом наперёд, ничего не меняют — «первое» считается по
    карточкам, а не по истории мыши.
    """
    odinakovo = ramka(200.0)
    pervoe = lico(E1, box=odinakovo)
    vtoroe = lico(E2, box=odinakovo)
    lica = {A: [pervoe, vtoroe], B: [lico(E1, box=odinakovo)]}

    itog = sobrat_etalon(lica, frozenset({(A, 0), (A, 1)}))
    obratno = sobrat_etalon(lica, frozenset([(A, 1), (A, 0)]))
    assert itog.anchor.lic is pervoe
    trebovat_odinakovy(itog, obratno)


def test_neotmechennoe_krupnoe_lico_ne_presledyaet_anchor() -> None:
    """Anchor — только из отмеченных: огромное неотмеченное лицо чужого человека не
    обязано утаскивать эталон на себя."""
    chuzhoe = lico(E2, box=ramka(900.0))
    moe = lico(E1, box=ramka(60.0))
    itog = sobrat_etalon({A: [chuzhoe, moe]}, frozenset({(A, 1)}))
    assert itog.anchor.lic is moe
    assert itog.karty[0].otmecheno is False
    assert itog.karty[0].procent is None
    assert itog.karty[0].v_poisk is False
    assert itog.karty[0].slabo is False


# --- число сходства предупреждает, но не запрещает -------------------------------------


def test_otmechennoe_lico_huzhe_poroga_slabosti_vse_ravno_idet_v_poisk() -> None:
    """Решение T14 (отменяет R-1): отмеченное лицо похожее на главное меньше чем на 40 %
    ВСЁ РАВНО участвует в поиске. Галочка человека — единственное решение.

    Прежний молчаливый отказ внутри `make_reference` заменён предупреждением: карточка
    остаётся отмеченной, получает свой процент, попадает в `extra` и в `reference`, а
    число ниже `porog_slabosti` зажигает признак `slabo`, из которого виджет соберёт
    слова «похоже слабо». В `otvergnutye` такое лицо больше не лежит.
    """
    anchor = lico(E1, box=ramka(300.0))
    otshepenelec = lico(vektor_blizkogo(0.39), box=ramka(100.0))
    itog = sobrat_etalon({A: [anchor], B: [otshepenelec]}, frozenset({(A, 0), (B, 0)}))

    kartochka = itog.karty[1]
    assert kartochka.otmecheno is True
    assert kartochka.v_poisk is True
    assert kartochka.slabo is True, "низкое число обязано быть замечено, а не проигнорировано"
    assert kartochka.procent == 39
    assert itog.karty[0].slabo is True, \
        "прежний «главный» получал 100 % и молчание, хотя был вторым человеком в кадре"
    assert itog.otvergnutye == ()
    assert itog.extra == (kartochka,)
    assert itog.reference is not None
    assert itog.reference.count == 2, "отмеченное лицо не доехало до эталона"


def test_slaboe_lico_menjaet_otvet_nastoyashhego_sravneniya() -> None:
    """Участие в поиске проверяется результатом, а не флагом: снимок, похожий ТОЛЬКО на
    слабое лицо, находится именно по нему.

    Это тот самый замер, который спорил с надписью «по нему ищем»: при одном лице в
    эталоне снимок даёт 39 %, при двух — 100 %. Сравнение ведёт настоящий `score_photo`,
    не подменённый, — тот же код, что считает живой прогон архива.
    """
    anchor = lico(E1, box=ramka(300.0))
    slaboe = lico(vektor_blizkogo(0.39), box=ramka(100.0))
    itog = sobrat_etalon({A: [anchor], B: [slaboe]}, frozenset({(A, 0), (B, 0)}))
    naidennoe_lico = lico(vektor_blizkogo(0.39), box=ramka(80.0))

    po_dvum_licam = score_photo([naidennoe_lico], itog.reference)
    po_odnomu_anchoru = score_photo([naidennoe_lico], make_reference(anchor, []))

    assert itog.reference.count == 2
    assert po_odnomu_anchoru.percent == 39, "замер-то не тот: один anchor обязан давать 39 %"
    assert po_dvum_licam.percent == 100
    assert po_dvum_licam.face is naidennoe_lico


def test_granica_poroga_slabosti_reshaet_toliko_preduprezhdenie() -> None:
    """Сравнение `>=` (то есть предупреждение при `sim < porog_slabosti`) — как в
    `make_reference`: ровно 0.40 предупреждения не даёт, а на волосок выше порога — даёт.

    В поиск теперь идут оба лица, поэтому граница перестала быть решающей для состава
    эталона и осталась решающей только для слова «похоже слабо». Границу меряют дважды:
    бытовым числом 0.40 и порогом, равным сходству ДО АБСОРБУ, — float32(0.4) как double
    больше литерала 0.40, и первая проверка отличает `<` от `<=` лишь случайно.
    """
    anchor = lico(E1, box=ramka(300.0))
    granica = lico(vektor_blizkogo(0.40), box=ramka(100.0))
    lica = {A: [anchor], B: [granica]}
    otmetki = frozenset({(A, 0), (B, 0)})

    itog = sobrat_etalon(lica, otmetki)
    kartochka = itog.karty[1]
    assert kartochka.procent == 40
    assert kartochka.v_poisk is True
    assert kartochka.slabo is False, "ровно на границе предупреждения нет"
    assert itog.extra == (kartochka,)
    assert itog.otvergnutye == ()
    assert itog.reference is not None
    assert itog.reference.count == 2

    tochnoe_shodstvo = float(granica.embedding @ anchor.embedding)
    # порог ровно равен сходству -> не предупреждаем (`>=`); на волосок выше -> предупреждаем
    assert sobrat_etalon(lica, otmetki, porog_slabosti=tochnoe_shodstvo).karty[1].slabo is False
    nad_volosok = float(np.nextafter(tochnoe_shodstvo, 2.0))
    zavershennyj = sobrat_etalon(lica, otmetki, porog_slabosti=nad_volosok).karty[1]
    assert zavershennyj.slabo is True
    assert zavershennyj.v_poisk is True, "порог слабости не властен над составом поиска"


def test_porog_slabosti_kak_parametr_pri_09_preduprezhdaet_pochti_vse() -> None:
    """Порог слабости — параметр, а не зашитое число: при 0.9 предупреждение получают
    все, кроме самых похожих, но в поиске остаются все годные.

    Прежняя версия этого теста требовала `v_poisk` и непустого `otvergnutye` — то есть
    молча отбирала отмеченные лица. Теперь параметр меняет только признак `slabo`.
    """
    svoe_a = lico(E1, box=ramka(400.0))
    svoe_b = lico(vektor_blizkogo(0.9), box=ramka(300.0))
    lishnee = lico(E2, box=ramka(100.0))          # 0 к `svoe_a`, 0.44 к `svoe_b`
    lica = {A: [svoe_a], B: [svoe_b], C: [lishnee]}
    otmetki = frozenset({(A, 0), (B, 0), (C, 0)})

    zhestko = sobrat_etalon(lica, otmetki, porog_slabosti=0.9)
    assert [k.v_poisk for k in zhestko.karty] == [True, True, True]
    assert [k.slabo for k in zhestko.karty] == [False, False, True]
    assert zhestko.otvergnutye == ()
    assert zhestko.reference is not None
    assert zhestko.reference.count == 3

    obichno = sobrat_etalon(lica, otmetki)
    assert [k.v_poisk for k in obichno.karty] == [True, True, True]
    assert [k.slabo for k in obichno.karty] == [False, False, False], \
        "при обычном пороге лишнее лицо всё ещё похоже на одно из своих"
    assert len(obichno.extra) == 2


def test_otmechennoe_lico_ne_teraetsja_v_sbore_etalona() -> None:
    """Окно не имеет права обещать одно, а `make_reference` делать другое.

    Прежде согласованность держалась на том, что одно и то же число уходило и в признак
    `v_poisk`, и в отбор кандидатов. С T14 отбора нет вовсе: сборка эталона вызывается с
    `min_sim=None`, и лицо с 39 % доезжает до эталона, а не исчезает в отборе молча.
    """
    anchor = lico(E1, box=ramka(400.0))
    a39 = lico(vektor_blizkogo(0.39), box=ramka(100.0))
    itog = sobrat_etalon({A: [anchor], B: [a39]}, frozenset({(A, 0), (B, 0)}))

    assert itog.karty[1].v_poisk is True
    assert itog.extra == (itog.karty[1],)
    assert itog.reference is not None
    assert itog.reference.count == 2
    assert itog.otvergnutye == ()


def test_okno_zovet_sborku_etalona_bez_otscheta(monkeypatch: pytest.MonkeyPatch) -> None:
    """Устройство, а не только итог: `make_reference` вызывают с `min_sim=None`.

    Один ассерт на `reference.count` прошёл бы и на случайной совпавшей паре чисел.
    Здесь подменяют сборку, пишут чем её позвали, и внутри зовут настоящую — чтобы
    провал не притворился успехом пустым ответом.
    """
    pozvannoe: list = []
    nastoyashhij = etalon.make_reference

    def zapisat(anchor, candidates, **kwargs):
        pozvannoe.append(kwargs.get("min_sim", "дефолт модуля matcher"))
        return nastoyashhij(anchor, candidates, **kwargs)

    monkeypatch.setattr(etalon, "make_reference", zapisat)
    itog = sobrat_etalon(
        {A: [lico(E1, box=ramka(300.0)), lico(vektor_blizkogo(0.39), box=ramka(100.0))]},
        frozenset({(A, 0), (A, 1)}))

    assert pozvannoe == [None]
    assert itog.reference is not None
    assert itog.reference.count == 2


def test_porog_slabosti_zhivet_v_matchere_odin_vladelec() -> None:
    """Два владельца одного числа однажды разойдутся: копия `0.40` в этом модуле
    запрещена — порог слабости по умолчанию обязан быть тем же объектом, что и
    `REF_MIN_SIM` в `core.matcher`."""
    parametry = inspect.signature(sobrat_etalon).parameters
    assert parametry["porog_slabosti"].default is REF_MIN_SIM
    assert etalon.REF_MIN_SIM is REF_MIN_SIM


# --- непригодные отпечатки ------------------------------------------------------------


@pytest.mark.parametrize("stroit_neprigodnoe", NEPRIGODNYE, ids=("nan", "nuli", "shirina_256"))
def test_krupnoe_neprigodnoe_lico_ne_stanovitsya_anchorom(
        stroit_neprigodnoe: Callable[..., Face]) -> None:
    """`make_reference` на непригодном anchor бросает ValueError — окно упало бы на
    пустом месте. Anchor переезжает на следующее крупнейшее годное лицо.

    Мелкое годное лицо стоит ПЕРВЫМ в плоском порядке: реализация «первое годное»
    выбрала бы его, а правило — «крупнейшее годное». Битое при этом не исчезает: оно
    остаётся карточкой с отметкой, и именно оно — единственный житель `otvergnutye`
    (слова вида «не ищем, сравнить нечем»), а его процент — 0, потому что считать
    нечем. Мелкое ГОДНОЕ лицо при этом уходит в поиск, хотя сходство с anchor — 0.
    """
    melkoe = lico(E2, box=ramka(100.0))                       # годное, но мелкое
    ploho = stroit_neprigodnoe(box=ramka(500.0))              # крупнейшее из отмеченных
    godnoe = lico(E1, box=ramka(200.0))
    lica = {A: [melkoe], B: [ploho], C: [godnoe]}
    itog = sobrat_etalon(lica, frozenset({(A, 0), (B, 0), (C, 0)}))

    assert itog.anchor.lic is godnoe
    assert itog.anchor.procent == 0 and itog.anchor.slabo is True, \
        "крупнейшее лицо больше не «главное»: с двумя непохожими лицами оба получают предупреждение"
    # в эталоне два лица: anchor и мелкое годное; за бортом одно — считать нечем
    assert itog.reference is not None
    assert itog.reference.count == 2
    assert itog.extra == (itog.karty[0],)
    assert itog.karty[0].lic is melkoe
    assert itog.karty[0].otmecheno is True
    assert itog.karty[0].v_poisk is True
    assert itog.karty[0].slabo is True
    assert itog.karty[0].procent == 0
    assert itog.karty[1].lic is ploho
    assert itog.karty[1].otmecheno is True
    assert itog.karty[1].v_poisk is False
    assert itog.karty[1].slabo is False, "не в поиске — значит не «слабо», а «нечем»"
    assert itog.karty[1].procent is None
    assert itog.otvergnutye == (itog.karty[1],)


def test_otvergnutye_nepusto_tolko_pri_neprigodnom_otpechatke() -> None:
    """Единственное, что теперь выкидывает отмеченное лицо из поиска, — непригодный
    отпечаток. Низкое число сходства больше ничего не выкидывает.

    Проверка в обе стороны: без битых лиц `otvergnutye` пуст, сколько бы слабых ни
    отметили, а `reference.count` равен числу ГОДНЫХ отмеченных — то есть инвариант
    «окно обещает и делает одно и то же» держится без всякого отбора.
    """
    anchor = lico(E1, box=ramka(400.0))
    srednee = lico(vektor_blizkogo(0.5), box=ramka(300.0))
    slaboe = lico(vektor_blizkogo(0.39), box=ramka(200.0))
    chuzhoe = np.zeros(512, dtype=np.float32)
    chuzhoe[2] = 1.0                              # не похожа ни на одно из остальных
    sovsem_chuzhoe = lico(chuzhoe.tolist(), box=ramka(100.0))
    bez_otbora = sobrat_etalon({A: [anchor], B: [srednee, slaboe], C: [sovsem_chuzhoe]},
                               frozenset({(A, 0), (B, 0), (B, 1), (C, 0)}))
    assert bez_otbora.otvergnutye == ()
    assert [k.v_poisk for k in bez_otbora.karty] == [True, True, True, True]
    # предупреждение теперь у одного лица, которое не похоже НИ НА ОДНО отмеченное,
    # а не у всех, кто не дотянул до произвольно выбранного «главного»
    assert [k.slabo for k in bez_otbora.karty] == [False, False, False, True]
    assert len(bez_otbora.extra) == 3
    assert bez_otbora.reference is not None
    assert bez_otbora.reference.count == 4, "count обязан равняться числу годных отмеченных"
    assert bez_otbora.reference.count == 1 + len(bez_otbora.extra)

    s_bitym = sobrat_etalon({A: [anchor], B: [slaboe, nan_lico(box=ramka(150.0))]},
                            frozenset({(A, 0), (B, 0), (B, 1)}))
    assert len(s_bitym.otvergnutye) == 1
    assert s_bitym.otvergnutye[0].lic is not None
    assert not prigoden_otpechatok(s_bitym.otvergnutye[0].lic.embedding)
    assert s_bitym.reference.count == 2                    # anchor + слабое, но годное


def test_vse_otmechennye_neprigodny_anchor_net_etalona_net_padeniya_net() -> None:
    """Ни одного годного лица — `anchor` и `reference` пустые, исключений нет, а
    `otvergnutye` называет все отмеченные: человеку есть что показать."""
    lica = {A: [nan_lico(box=ramka(500.0))],
            B: [nulevoe_lico(box=ramka(300.0))],
            C: [shirokoe_lico(box=ramka(100.0))]}
    itog = sobrat_etalon(lica, frozenset({(A, 0), (B, 0), (C, 0)}))

    assert itog.anchor is None
    assert itog.reference is None
    assert itog.extra == ()
    assert len(itog.karty) == 3
    assert len(itog.otvergnutye) == 3
    assert all(k.procent is None for k in itog.karty)


def test_lico_chuzhoy_shiriny_ne_ronjaet_pereschot() -> None:
    """Отмеченное лицо «из другой модели» (256 чисел) не роняет пересчёт: умножение
    векторов чужой ширины бросает ValueError, поэтому его сходство — 0."""
    anchor = lico(E1, box=ramka(300.0))
    drugaya_model = shirokoe_lico(box=ramka(100.0))
    itog = sobrat_etalon({A: [anchor], B: [drugaya_model]}, frozenset({(A, 0), (B, 0)}))

    assert itog.karty[1].procent is None
    assert itog.karty[1].v_poisk is False
    assert itog.otvergnutye == (itog.karty[1],)


# --- отметки, у которых нет лица ------------------------------------------------------


def test_otmetka_bez_litsa_molcha_ignoriruetsya() -> None:
    """Виджет живёт в мире, где список лиц мог перестроиться между кликом и
    пересчётом: несуществующий файл, индекс 99, индекс в файле без лиц — всё молча
    мимо, без исключения и без фантомной карточки."""
    edinoe = lico(E1)
    lica = {A: [edinoe], B: []}
    itog = sobrat_etalon(lica, frozenset({(NET, 0), (A, 99), (B, 0), (A, 1)}))

    assert [(k.put, k.nomer) for k in itog.karty] == [(A, 1), (B, 0)]
    assert all(k.otmecheno is False for k in itog.karty)
    assert itog.anchor is None
    assert itog.reference is None
    assert itog.otvergnutye == ()


# --- порядок карточек и неотмеченные ---------------------------------------------------


def test_poryadok_kartochek_ploskiy_i_ne_zavisit_ot_otmetok() -> None:
    """Файлы — в порядке словаря, внутри файла — по номеру лица; файл без лиц даёт
    карточку с номером 0 в конце своего файла.

    Отметки, переданные в обратном порядке, на этот список не влияют вовсе.
    """
    a1, a2 = lico(E1, box=ramka(100.0)), lico(E2, box=ramka(90.0))
    b1 = lico(vektor_blizkogo(0.5), box=ramka(80.0))
    lica = {A: [a1, a2], B: [b1], C: []}

    pryamo = sobrat_etalon(lica, frozenset([(A, 0), (A, 1), (B, 0)]))
    obratno = sobrat_etalon(lica, frozenset([(B, 0), (A, 1), (A, 0)]))

    assert [(k.put, k.nomer) for k in pryamo.karty] == [(A, 1), (A, 2), (B, 1), (C, 0)]
    assert [(k.put, k.nomer) for k in obratno.karty] == [(A, 1), (A, 2), (B, 1), (C, 0)]
    trebovat_odinakovy(pryamo, obratno)


def test_ne_edinichnyy_otpechatok_ne_dast_chisla_vyshe_stota() -> None:
    """Косинус делится на нормы: «400 %» на карточке появиться не может.

    Годный отпечаток не обязан быть единичной длины: `prigoden_otpechatok` требует
    конечности, ширины 512 и ненулевой нормы, а строку вдвое длиннее вполне может
    привезти оглавление, записанное старой версией. Скалярное произведение такого
    вектора с соседним — 4.0, и без деления на нормы карточка показала бы «400 %».
    """
    ne_edinichnyj = np.array(E1, dtype=np.float32) * 2.0
    krupnoe = Face(box=ramka(300.0), landmarks=None, embedding=ne_edinichnyj,
                   detector="insight")
    itog = sobrat_etalon({A: [krupnoe], B: [lico(E1, box=ramka(100.0))]},
                         frozenset({(A, 0), (B, 0)}))

    assert [k.procent for k in itog.karty] == [100, 100]
    assert all(k.procent <= 100 for k in itog.karty)
    assert itog.reference is not None


def test_neotmechennye_karty_procent_nul() -> None:
    """Неотмеченная карточка не показывает чужое число: человек ещё не решил, кто
    ребёнок, а процент уже кричит «сходство 80 %».

    `slabo` у неё тоже `False`: предупреждение касается только тех, кого человек сам
    позвал в поиск, а не всех подряд.
    """
    krupnoe = lico(E1, box=ramka(400.0))
    melkoe = lico(vektor_blizkogo(0.8), box=ramka(50.0))
    chuzhoe = lico(E2, box=ramka(50.0))
    lica = {A: [krupnoe, melkoe], B: [chuzhoe]}
    itog = sobrat_etalon(lica, frozenset({(A, 0)}))

    assert [k.procent for k in itog.karty] == [None, None, None], \
        "отмечено одно лицо — сравнивать его не с чем"
    assert [k.v_poisk for k in itog.karty] == [True, False, False]
    assert [k.slabo for k in itog.karty] == [False, False, False]
    assert itog.otvergnutye == ()


def test_otvergnutye_vsegda_podmnozhestvo_otmechennyh_bez_poiska() -> None:
    """Инвариант для виджета: подсветить словами «не ищем» можно только то, что человек
    отметил сам, и только то, у чего отпечаток не годен к сравнению.

    С T14 в эту кучку больше не попадает ничего похожего «ниже числа»: отмеченное и
    годное лицо всегда в поиске. Проверка держит обе стороны — слабое сходство лежит в
    `extra`, а битой отпечаток лежит в `otvergnutye`, и сумма не теряет ни одной отметки.
    """
    anchor = lico(E1, box=ramka(400.0))
    a39 = lico(vektor_blizkogo(0.39), box=ramka(300.0))
    a90 = lico(vektor_blizkogo(0.9), box=ramka(200.0))
    nan_ploho = nan_lico(box=ramka(100.0))
    lica = {A: [anchor], B: [a39, a90], C: [nan_ploho], D: []}
    itog = sobrat_etalon(lica, frozenset({(A, 0), (B, 0), (B, 1), (C, 0), (D, 0)}))

    for kartochka in itog.otvergnutye:
        assert kartochka.otmecheno is True
        assert kartochka.v_poisk is False
        assert not prigoden_otpechatok(kartochka.lic.embedding), \
            "в отвергнутые попало годное лицо — значит отбор вернулся"
    # `a39` слабая только относительно «главного» лица, а рядом есть `a90`, похожая на
    # неё на 75 %, — предупреждения нет, и в поиске она всё равно
    assert itog.karty[1].lic is a39 and itog.karty[1].v_poisk is True
    assert itog.karty[1].procent == 75 and not itog.karty[1].slabo
    assert itog.karty[2].lic is a90 and itog.karty[2].v_poisk is True
    assert itog.otvergnutye == (itog.karty[3],)
    otmechennye = [k for k in itog.karty if k.otmecheno]
    # `set(kartochki)` не собрать: внутри карточки лежит `Face` с numpy-вектором, а
    # numpy не хешируется. Идентичность сверяется по `id` — ровно так же, как виджет
    # сверяет карточки между пересчётами.
    assert {id(k) for k in itog.otvergnutye} <= {id(k) for k in otmechennye}
    assert len(itog.otvergnutye) + len(itog.extra) + 1 == len(otmechennye)
    assert itog.reference is not None
    assert itog.reference.count == 1 + len(itog.extra)
    assert itog.reference.count == len([k for k in otmechennye if k.lic is not None
                                        and prigoden_otpechatok(k.lic.embedding)])


# --- согласованность результата --------------------------------------------------------


def test_karty_i_etalon_ukazyvayut_na_odny_te_zhe_obekty() -> None:
    """Виджет сверяет карточки по идентичности: `anchor`, `extra` и `otvergnutye`
    обязаны быть те же самые объекты, что лежат в `karty`, а не их отдельные копии.

    Заодно тест фиксирует, КТО где лежит после T14: лицо, похожее на 20 %, едет в
    `extra` (оно в поиске), а `otvergnutye` отдаётся непригодному отпечатку.
    """
    krupnoe = lico(E1, box=ramka(400.0))
    dopolnitelnoe = lico(vektor_blizkogo(0.8), box=ramka(100.0))
    otshepenelec = lico(vektor_blizkogo(0.2), box=ramka(80.0))
    ploho = nan_lico(box=ramka(60.0))
    lica = {A: [krupnoe], B: [dopolnitelnoe], C: [otshepenelec], D: [ploho]}
    itog = sobrat_etalon(lica, frozenset({(A, 0), (B, 0), (C, 0), (D, 0)}))

    assert any(k is itog.anchor for k in itog.karty)
    assert all(any(k is dob for k in itog.karty) for dob in itog.extra)
    assert all(any(k is otk for k in itog.karty) for otk in itog.otvergnutye)
    assert [k.lic for k in itog.extra] == [dopolnitelnoe, otshepenelec]
    assert [k.lic for k in itog.otvergnutye] == [ploho]
    assert itog.reference.count == 1 + len(itog.extra)


def test_kartochka_i_itog_neizmenyaemy() -> None:
    """Замороженные dataclass: пересчёт окна не должен иметь возможности подправить
    уже показанное число."""
    lica = {A: [lico(E1)]}
    otmetki = frozenset({(A, 0)})
    kartochka = sobrat_etalon(lica, otmetki).karty[0]
    itog = sobrat_etalon(lica, otmetki)

    with pytest.raises(dataclasses.FrozenInstanceError):
        kartochka.otmecheno = False        # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        itog.anchor = None                 # type: ignore[misc]


def test_polya_sootvetstvuyut_soglasovannomu_tipu() -> None:
    """Имена и порядок полей заданы спецификацией: на них опирается виджет из T3.

    `slabo` добавлено задачей T14: виджету больше неоткуда взять слово «похоже слабо»,
    если ответ его не несёт, а придумывать его на месте значит завести второе владение
    тем же числом.
    """
    assert [f.name for f in dataclasses.fields(Kartochka)] == [
        "put", "nomer", "lic", "otmecheno", "procent", "v_poisk", "slabo"]
    assert [f.name for f in dataclasses.fields(Itog)] == [
        "karty", "anchor", "extra", "otvergnutye", "reference"]
    assert list(inspect.signature(sobrat_etalon).parameters) == [
        "lica_po_fajlam", "otmetki", "porog_slabosti"]


def test_valueerror_iz_sborki_etalona_ne_vyodit_naruzhu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Единственный мок в файле сторожит недостижимую ветку.

    `make_reference` бросает ValueError только на непригодном anchor, а anchor тут
    выбирается исключительно из годных лиц, — то есть в жизни исключения быть не может.
    Но обещание «окно не падает» должно быть проверено, а не принято на веру: подменяем
    именно сборку эталона и требуем тихий провал — «эталона нет», а не сбой всего окна.
    """
    def brosat(*args: object, **kwargs: object) -> Reference:
        raise ValueError("отпечаток эталонного лица непригоден к сравнению")

    monkeypatch.setattr(etalon, "make_reference", brosat)
    itog = sobrat_etalon(
        {A: [lico(E1, box=ramka(400.0)), lico(vektor_blizkogo(0.9), box=ramka(100.0))]},
        frozenset({(A, 0), (A, 1)}))

    assert itog.reference is None
    assert itog.anchor is None
    assert itog.extra == ()
    assert len(itog.karty) == 2
    assert all(k.otmecheno and not k.v_poisk for k in itog.karty)
    assert {id(k) for k in itog.otvergnutye} == {id(k) for k in itog.karty}


# --- число каждого лица решают его собственные собеседники, а не одно «главное» --------


def _blizkaya_k_E2(x: float) -> list[float]:
    """Вектор с косинусом `x` к E2 и нулём к E1: два «маминых» лица близки друг к другу
    и не похожи на детское."""
    v = np.zeros(512, dtype=np.float32)
    v[1] = x
    v[2] = float(np.sqrt(1 - x * x))
    return v.tolist()


def test_chislo_kazhdogo_lica_reshayut_ego_sobstvenniki_a_ne_odno_glavnoe() -> None:
    """Ни одно лицо не получает «100 %» и звание главного за то, что оно крупнее.

    Замер на данных пользователя: для «мама» они принесли те же 16 снимков, и самым
    крупным отмеченным лицом в них оказался спящий ребёнок. Прежнее правило сделало его
    «главным · по нему ищем», а все мамины лица получили «похоже слабо» относительно
    детского — то есть окно утверждало обратное тому, что человек в него принёс.
    """
    malish = lico(E1, box=ramka(600.0))                 # крупнейшее: по-старому стало бы «главным»
    mama_a = lico(E2, box=ramka(120.0))
    mama_b = lico(_blizkaya_k_E2(0.9), box=ramka(120.0))
    itog = sobrat_etalon({A: [malish], B: [mama_a], C: [mama_b]},
                         frozenset({(A, 0), (B, 0), (C, 0)}))
    po_litsam = {k.lic.box: k for k in itog.karty}

    assert po_litsam[malish.box].procent == 0, "детское лицо не похоже ни на одно мамино"
    assert po_litsam[mama_a.box].procent == 90 and po_litsam[mama_b.box].procent == 90
    assert po_litsam[malish.box].slabo and not po_litsam[mama_a.box].slabo
    assert {k.procent for k in itog.karty} != {100}, "кто-то назван главным по правке размера"


def test_odinom_lice_sravnit_nechego_i_chisla_net() -> None:
    """Единственное отмеченное лицо не получает «100 %»: сравнивать его не с чем.

    Прежнее «100 %» было не ответом, а артефактом — anchor сравнивался сам с собой.
    """
    odinochka = lico(E1, box=ramka(300.0))
    itog = sobrat_etalon({A: [odinochka]}, frozenset({(A, 0)}))
    assert itog.karty[0].procent is None and itog.karty[0].slabo is False
    assert itog.reference.count == 1, "в поиск лицо при этом всё равно идёт"


def test_v_kartochke_ne_ostalos_mesta_dlya_glavnogo_lica() -> None:
    """Понятия «главное лицо» в карточке больше нет — ни флага, ни подписи.

    Отметка человека — единственное решение (T14), и все отмеченные лица равноправны.
    Пока в данных живёт флаг `je_etalon`, какой-нибудь экран обязательно снова напишет
    «по нему ищем» про то, что человек не выбирал.
    """
    assert "je_etalon" not in {f.name for f in dataclasses.fields(Kartochka)}


# --- автоотметка не имеет права решать за человека (задача T22) -------------------------


def test_odinochnoe_lico_ne_otmechaetsya_esli_ego_uzhe_ishut() -> None:
    """Лицо, закреплённое за другим человеком, само себе отметку не ставит.

    Пользователь принёс для мамы те же 16 снимков, что и для ребёнка, и приложение
    автоматически отметило спящего ребёнка: «при добавлении мамы ты делаешь фото ребёнка
    главным». Автоотметка «на снимке одно лицо» была заказана человеком для случая «я
    приношу фото своего ребёнка», но там, где он уже сказал, кто на этом снимке, она
    решает против него.
    """
    a, b = A, B
    lica = {a: [lico(E1)], b: [lico(E2)]}
    assert odinochnye_otmetki(lica) == frozenset({(a, 0), (b, 0)})
    assert odinochnye_otmetki(lica, zanyatye=frozenset({(a, 0)})) == frozenset({(b, 0)})


def test_gruppovoe_lico_ne_bylo_otmecheno_i_ne_stanovitsya() -> None:
    """Снимок с несколькими лицами автоотметки не получал и не получает — занят он или нет.

    Угадывать, кто из троих в кадре искомый человек, приложение не имеет права ни в
    каком режиме; второй аргумент ничего в этой ветке менять не должен.
    """
    lica = {A: [lico(E1), lico(E2)]}
    assert odinochnye_otmetki(lica) == frozenset()
    assert odinochnye_otmetki(lica, zanyatye=frozenset({(A, 0), (A, 1)})) == frozenset()
