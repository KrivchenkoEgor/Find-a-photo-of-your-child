"""Эталон от подтверждённого фото и процент сходства без порога внутри ядра.

Тесты чистые: отпечатки — короткие numpy-векторы единичной длины, никаких моделей
и никаких личных фото из data/ и ref/.
"""

import inspect
from typing import Callable, Sequence

import numpy as np
import pytest

from core.engine import Face
from core.matcher import Reference, ScoredPhoto, make_reference, percent_of, score_photo


def lico(emb: Sequence[float], box: tuple[float, float, float, float] = (0.0, 0.0, 50.0, 50.0)) -> Face:
    v = np.array(emb, dtype=np.float32)
    return Face(box=box, landmarks=None, embedding=v / np.linalg.norm(v), detector="insight")


E1 = [1.0] + [0.0] * 511
E2 = [0.0, 1.0] + [0.0] * 510
E3 = [0.9, 0.1] + [0.0] * 510          # близко к E1


def test_reference_klyuchuet_chuzhie_lica() -> None:
    anchor = lico(E1)
    ref = make_reference(anchor, [anchor, lico(E2), lico(E3)], min_sim=0.40)
    assert ref.count == 2               # E2 не похож на anchor — отброшен


def test_anchor_vsegda_v_reference() -> None:
    anchor = lico(E1)
    assert make_reference(anchor, [anchor, lico(E2)]).count == 1


def test_beret_maximum_po_licam_i_po_etalonam() -> None:
    ref = make_reference(lico(E1), [lico(E1), lico(E3)])
    scored = score_photo([lico(E2), lico(E3)], ref)
    assert scored.similarity == 1.0     # E3 == один из эталонов
    assert scored.face.box == (0.0, 0.0, 50.0, 50.0)


def test_esli_lic_net_nulya_i_net_lica() -> None:
    ref = make_reference(lico(E1), [lico(E1)])
    scored = score_photo([], ref)
    assert scored.similarity == 0.0 and scored.percent == 0 and scored.face is None


def test_procent_okruglyaet() -> None:
    assert percent_of(0.3847) == 38
    assert percent_of(0.6749) == 67
    assert percent_of(-0.2) == 0        # отрицательное сходство не показываем минусом


def test_hudshee_podtverzhdennoe_foto_prohodit() -> None:
    """39.2% — худшее подтверждённое глазами фото; на пороге 0.38 оно обязано быть «похожим»."""
    ref = make_reference(lico(E1), [lico(E1)])
    near = np.zeros(512, dtype=np.float32)
    near[0], near[1] = 0.392, np.sqrt(1 - 0.392 ** 2)
    assert score_photo([lico(near.tolist())], ref).percent >= 38


# --- максимум считается по обоим измерениям ---------------------------------------


def test_lico_vybiraetsya_po_maximumu_po_licam() -> None:
    """Из двух лиц фото представляет то, что похоже: `face` — максимум по лицам.

    Рамки заведомо разные: ассерт на `box` из листинга плана не различал бы лица,
    потому что обе рамки там одинаковые по умолчанию.
    """
    ref = make_reference(lico(E1), [lico(E1)])
    chuzhoe = lico(E2, box=(0.0, 0.0, 10.0, 10.0))
    rodnoe = lico(E3, box=(300.0, 200.0, 400.0, 300.0))
    scored = score_photo([chuzhoe, rodnoe], ref)
    assert scored.face is rodnoe
    assert scored.similarity == pytest.approx(0.9 / float(np.sqrt(0.82)), abs=1e-6)
    assert scored.percent == 99


def test_maksimum_beret_i_po_vtoroy_stroke_etalona() -> None:
    """Лицо похоже не на anchor, а на добавленный эталон — сходство считается по нему.

    Если бы `score_photo` сравнивал только с anchor, это фото получило бы 0.994 вместо
    1.0: второй строки эталона в расчёте как бы не существовало.
    """
    ref = make_reference(lico(E1), [lico(E3)])   # эталон: anchor E1 + добавленное лицо E3
    assert ref.count == 2
    tolko_anchor = score_photo([lico(E3)], make_reference(lico(E1), []))
    scored = score_photo([lico(E3)], ref)
    assert scored.similarity == pytest.approx(1.0, abs=1e-6)   # попало ровно в свою строку
    assert scored.similarity > tolko_anchor.similarity
    assert scored.percent == 100


def test_porog_v_yadre_ne_zhivet() -> None:
    """Специфика требует: у оценки нет порога, ползунок должен двигаться мгновенно."""
    params = inspect.signature(score_photo).parameters
    assert list(params) == ["faces", "ref"]
    assert all(p.default is inspect.Parameter.empty for p in params.values())
    # порог есть только у сборки эталона, и он называется явно
    assert inspect.signature(make_reference).parameters["min_sim"].default == 0.40


def test_lico_rovno_na_poroge_ostaetsya_v_etalone() -> None:
    """Сравнение `>=`, а не `>`: ровно пороговое лицо остаётся в эталоне."""
    anchor = lico(E1)
    granica = np.zeros(512, dtype=np.float32)
    granica[0], granica[1] = 0.40, np.sqrt(1 - 0.40 ** 2)
    assert make_reference(anchor, [anchor, lico(granica.tolist())], min_sim=0.40).count == 2
    chut_nizhe = np.zeros(512, dtype=np.float32)
    chut_nizhe[0], chut_nizhe[1] = 0.39, np.sqrt(1 - 0.39 ** 2)
    assert make_reference(anchor, [anchor, lico(chut_nizhe.tolist())], min_sim=0.40).count == 1


def test_pustoy_etalon_ne_ronyaet_ocenku() -> None:
    """Защита для вызывающего кода: пустой эталон даёт 0, а не деление на ноль."""
    empty = Reference(embeddings=np.zeros((0, 512), dtype=np.float32))
    scored = score_photo([lico(E1)], empty)
    assert isinstance(scored, ScoredPhoto)
    assert scored.percent == 0 and scored.face is None


# --- непригодный отпечаток: «этого лица нет», а не падение и не обнулённый архив ----
#
# Вектор с NaN или не 512 чисел приезжает из оглавления, записанного до того, как
# отпечатки начали проверять при чтении (`Face.from_dict`), — то есть из реального
# файла пользователя. Для оценки фото это означает ровно одно: лицо в сравнении не
# участвует. До правки `argmax` выбирал именно NaN-строку, `round(nan)` бросал
# ValueError, и прогон обрывался на первом же таком снимке; отпечаток чужой ширины
# давал `matmul: size 256 is different from 512` тем же местом.


def nan_lico(box: tuple[float, float, float, float] = (0.0, 0.0, 10.0, 10.0)) -> Face:
    return Face(box=box, landmarks=None, embedding=np.full(512, np.nan, dtype=np.float32),
                detector="insight")


def nulevoe_lico(box: tuple[float, float, float, float] = (0.0, 0.0, 10.0, 10.0)) -> Face:
    return Face(box=box, landmarks=None, embedding=np.zeros(512, dtype=np.float32),
                detector="insight")


def shirokoe_lico(box: tuple[float, float, float, float] = (0.0, 0.0, 10.0, 10.0)) -> Face:
    """«Вектор из другой модели»: 256 чисел вместо 512."""
    return Face(box=box, landmarks=None, embedding=np.full(256, 0.1, dtype=np.float32),
                detector="insight")


NEPRIGODNYE_LICA = (nan_lico, nulevoe_lico, shirokoe_lico)


# --- номер строки эталона: по КАКОМУ выбранному лицу нашёлся снимок -------------------
#
# Задача T15. Человек отмечает не одно лицо ребёнка, а несколько примеров — и среди
# них может стоять брат или взрослый. Одно число «45 %» на карточке не отвечает на
# вопрос «кто на этом снимке»: ребёнок, похожий неуверенно, или совсем другой человек,
# угаданный по второму отмеченному лицу. Отвечает номер строки эталона, и он уже
# посчитан: `max(axis=1)` выбирает лучшее лицо, а какой строкой достигнут максимум,
# теряется. Второй `argmax` по тому же массиву стоит микросекунды — при эталоне из 16
# лиц сравнение всех 591 лица архива занимает 4,9 мс против 5,8 мс при одном лице, а
# всё долгое время (чтение файла и детекция) платится один раз за снимок.


def test_vtoraya_stroka_etalona_dajet_ref_lico_dva() -> None:
    """Максимум по второй строке обязан быть назван второй, а не первой.

    Векторы подобраны так, что лицо ближе ко второй строке, чем к первой: к anchor
    косинус 0.994, к добавленному лицу 1.0. Если бы номер строки не считался вовсе или
    считался «всегда 1», тест падал бы — при одинаковых числах он ничего не различает.
    """
    ref = make_reference(lico(E1), [lico(E3)])      # строка 1 = E1, строка 2 = E3
    assert ref.count == 2
    scored = score_photo([lico(E3)], ref)
    assert scored.similarity == pytest.approx(1.0, abs=1e-6)
    assert scored.ref_lico == 2, f"максимум по второй строке, а ответ: {scored.ref_lico}"


def test_pervaya_stroka_etalona_dajet_ref_lico_odin() -> None:
    """Лицо, похожее на anchor сильнее, чем на добавленное, обязано остаться первой
    строкой: иначе подпись «похоже на лицо 2» встанет на главном лице."""
    ref = make_reference(lico(E1), [lico(E3)])
    scored = score_photo([lico(E1)], ref)
    assert scored.similarity == pytest.approx(1.0, abs=1e-6)
    assert scored.ref_lico == 1


def test_ref_lico_ne_menjaet_chislo_na_kartochke() -> None:
    """Номер строки — только подпись. Проценты и лицо на снимке обязаны остаться ровно
    теми же, что и до правки: любое расхождение значило бы, что ради подписки кто-то
    перестал сравнивать со второй строкой, а число стало бы заниженным."""
    ref = make_reference(lico(E1), [lico(E3)])
    tolko_anchor = score_photo([lico(E3)], make_reference(lico(E1), []))
    s_etalom = score_photo([lico(E3)], ref)
    assert s_etalom.percent == 100 and s_etalom.percent > tolko_anchor.percent
    assert s_etalom.face is not None and s_etalom.similarity == pytest.approx(1.0)


def test_pustoj_etalon_i_foto_bez_lic_dayut_nul() -> None:
    """`ref_lico == 0` — честное «не считали»: пустой эталон и снимок без лиц.

    Ноль здесь не «первая строка» и не «последняя»: у вызывающего он обязан означать
    ровно одно — ответа нет, и подпись на карточке показывать нечем.
    """
    empty = Reference(embeddings=np.zeros((0, 512), dtype=np.float32))
    assert score_photo([lico(E1)], empty).ref_lico == 0
    ref = make_reference(lico(E1), [lico(E1)])
    assert score_photo([], ref).ref_lico == 0
    assert score_photo([nan_lico()], ref).ref_lico == 0


def test_bitaya_stroka_etalona_ne_vyigryvaet_nomery() -> None:
    """Номер строки берётся по тем же очищенным числам, что и максимум.

    `argmax` по строке с NaN возвращает именно NaN-индекс (NaN сравним как угодно), и
    наивный второй `argmax` указал бы на отравленную первую строку там, где настоящий
    максимум лежит на второй. Для человека это значило бы «похоже на лицо 1» про лицо,
    которым сравнивать нечем.
    """
    otravlen = Reference(embeddings=np.stack([np.full(512, np.nan, dtype=np.float32),
                                              np.array(E1, dtype=np.float32)]))
    scored = score_photo([lico(E1)], otravlen)
    assert scored.percent == 100
    assert scored.ref_lico == 2, f"номер строки указал на непригодный отпечаток: {scored}"


def test_scored_photo_imaet_pole_po_umolchaniyu() -> None:
    """Вызовующий код (оглавление, старые тесты) собирает `ScoredPhoto` и без номера:
    поле обязано иметь значение по умолчанию, иначе импорт ядра рассыпется всюду."""
    bez_nomera = ScoredPhoto(percent=50, similarity=0.5, face=None)
    assert bez_nomera.ref_lico == 0
    assert ScoredPhoto(50, 0.5, None, 2).ref_lico == 2


@pytest.mark.parametrize("stroit_neprigodnoe", NEPRIGODNYE_LICA,
                         ids=("nan", "nuli", "shirina_256"))
def test_neprigodnoe_lico_ne_meshaet_rodnomu(
        stroit_neprigodnoe: Callable[[], Face]) -> None:
    """Порядок в списке не важен: битое лицо не выигрывает максимум и не роняет фото.

    Проверены оба места из замечания ревью — `[nan_face, good_face]` и
    `[good_face, nan_face]`: раньше `argmax` выбирал NaN-индекс при любом из них.
    """
    ref = make_reference(lico(E1), [lico(E1)])
    rodnoe = lico(E1, box=(300.0, 200.0, 400.0, 300.0))
    ploho = stroit_neprigodnoe()
    for poryadok in ([ploho, rodnoe], [rodnoe, ploho]):
        scored = score_photo(poryadok, ref)
        assert scored.face is rodnoe                                # не битое лицо
        assert scored.percent == 100 and scored.similarity == pytest.approx(1.0)


def test_esli_vse_lica_neprigodnye_daut_tot_zhe_nul() -> None:
    """Ни одного пригодного лица — ответ как у пустого фото, без исключения."""
    ref = make_reference(lico(E1), [lico(E1)])
    scored = score_photo([nan_lico(), nulevoe_lico(), shirokoe_lico()], ref)
    assert scored.similarity == 0.0 and scored.percent == 0 and scored.face is None


def test_etalon_s_bitoj_stroki_ne_ronyaet_ocenku() -> None:
    """`Reference` строят и вручную: отравленная строка эталона не выигрывает максимум.

    Пригодные строки эталона продолжают считаться — то есть одно испорченное число
    не обнуляет результат всему архиву.
    """
    otravlen = Reference(embeddings=np.stack([np.full(512, np.nan, dtype=np.float32),
                                              np.array(E1, dtype=np.float32)]))
    scored = score_photo([lico(E1)], otravlen)
    assert scored.percent == 100 and scored.face is not None
    bez_etalona = score_photo([lico(E1)], Reference(
        embeddings=np.full((1, 512), np.nan, dtype=np.float32)))
    assert bez_etalona.percent == 0 and bez_etalona.face is None


def test_percent_of_na_nan_i_beskonechnosti_ne_padayet() -> None:
    """`round(nan)` бросает ValueError, `round(inf)` — OverflowError. Проценты так
    не роняют показ: непригодное число это 0. Правило округления не тронуто."""
    assert percent_of(float("nan")) == 0
    assert percent_of(float("inf")) == 0
    assert percent_of(float("-inf")) == 0
    assert percent_of(0.3847) == 38 and percent_of(0.6749) == 67 and percent_of(-0.2) == 0


def test_neprigodnye_kandidaty_ne_popadayut_v_etalon() -> None:
    """Сборка эталона на битом кандидате не роняет matmul и не раздувает эталон."""
    anchor = lico(E1)
    ref = make_reference(anchor, [anchor, nan_lico(), shirokoe_lico(), lico(E3)])
    assert ref.count == 2                                   # anchor + похожее E3
    assert score_photo([lico(E3)], ref).similarity == pytest.approx(1.0, abs=1e-6)


def test_min_sim_none_vykluchaet_otbor_i_beret_vse_godnye() -> None:
    """`min_sim=None` — «не отбирать вовсе»: в эталон едет любое годное лицо.

    Этим способом окно эталона зовёт сборку с задачи T14. Человек поставил галочку, и
    лицо, похожее на главное хоть на 0 %, обязано участвовать в поиске: молча не попасть
    в эталон — тот самый обман, из-за которого правка и началась. Непригодный отпечаток
    не участвует ни при каких настройках, потому что считать им нечем.

    Второй и третий ассерты не косметика: они показывают, что результат изменился именно
    из-за `None`, а не потому, что отсекаемый вектор случайно оказался похожим.
    """
    anchor = lico(E1)
    chuzhoe = lico(E2)                      # косинус 0 к anchor: прежний отбор его выбрасывал
    spryamoe = lico(E3)
    ref = make_reference(anchor, [anchor, chuzhoe, spryamoe, nan_lico(), shirokoe_lico()],
                         min_sim=None)
    assert ref.count == 3                                   # anchor + оба годных кандидата
    assert make_reference(anchor, [chuzhoe], min_sim=0.40).count == 1
    assert make_reference(anchor, [chuzhoe], min_sim=None).count == 2
    assert make_reference(anchor, [], min_sim=None).count == 1


@pytest.mark.parametrize("stroit_neprigodnoe", NEPRIGODNYE_LICA,
                         ids=("nan", "nuli", "shirina_256"))
def test_neprigodnyy_anchor_oshibka_a_ne_tihoe_nul(
        stroit_neprigodnoe: Callable[[], Face]) -> None:
    """Непригодный anchor до правки давал пустой эталон и 0 % на всём архиве молча.

    «Приложение потеряло моего ребёнка» без единого сообщения — худший исход для
    пользователя, который не программист. Здесь единственный случай, где громкая
    ошибка честнее тихого нуля; к оглавлению это не относится: `Face.from_dict` такое
    лицо просто не вернёт, и фото разберётся заново.
    """
    with pytest.raises(ValueError, match="непригоден"):
        make_reference(stroit_neprigodnoe(), [lico(E1)])


# --- похожесть каждого лица снимка, а не только максимум (задача T19) -------------------


def test_kazhdoe_lico_snimka_poluchaet_svoy_percent() -> None:
    """У каждого лица обязано быть своё число: ребёнок на фото может стоять дважды.

    Замер на архиве пользователя: на групповом снимке из 24 лиц ребёнок нашёлся два
    раза — лица 22 и 24, по 65 % каждый. `score_photo` возвращал один максимум,
    второе совпадение пропало, и человек увидел рамку на одном из двух своих детей.
    """
    ref = make_reference(lico(E1), [lico(E1), lico(E3)])
    a, b, c = (lico(E2, (0.0, 0.0, 50.0, 50.0)), lico(E3, (60.0, 0.0, 110.0, 50.0)),
               lico(E2, (120.0, 0.0, 170.0, 50.0)))
    assert score_photo([a, b, c], ref).po_licam == ((1, 11), (2, 100), (3, 11))


def test_dva_lico_odnogo_cheloveka_dayut_ramku_na_oboh() -> None:
    """Два лица одного ребёнка дают два одинаковых числа — рамка встанет на обоих."""
    ref = make_reference(lico(E1), [lico(E1)])
    odinakovie = [lico(E1, (0.0, 0.0, 50.0, 50.0)), lico(E1, (60.0, 0.0, 110.0, 50.0))]
    assert score_photo(odinakovie, ref).po_licam == ((1, 100), (2, 100))


def test_chislo_lica_v_snyatom_spiske_a_ne_v_ochishchennom() -> None:
    """Битое лицо не сдвигает номера: рамка должна попасть в то же лицо, что и в кадре.

    `prigodnye_lica` выкидывает непригодный отпечаток, и если считать номера по
    очищенному списку, второе лицо уехало бы на место первого — рамка легла бы мимо
    ребёнка, что для родителя хуже отсутствующей рамки.
    """
    ref = make_reference(lico(E1), [lico(E1)])
    good = lico(E1, (300.0, 200.0, 400.0, 300.0))
    assert score_photo([nan_lico(), good], ref).po_licam == ((2, 100),)
    assert score_photo([good, nan_lico()], ref).po_licam == ((1, 100),)


def test_bez_lic_i_bez_etalona_spisk_pustyh_a_ne_othet() -> None:
    """Ни лиц, ни эталона — пустой список, а не `None` и не падение: вызывающий
    перебирает его без проверок."""
    ref = make_reference(lico(E1), [lico(E1)])
    assert score_photo([], ref).po_licam == ()
    assert score_photo([lico(E1)], Reference(np.zeros((0, 512), dtype=np.float32))
                       ).po_licam == ()
