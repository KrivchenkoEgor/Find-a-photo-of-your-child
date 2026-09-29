"""Как нарисовать лицо карточкой: геометрия кропа и конвертация в QPixmap.

Модуль `ui/face_picker.py` спрашивать «какое лицо искать» перестал — это делает окно
эталона (`ui/reference_dialog.py`, свои тесты в `tests/test_ui_reference_dialog.py`).
Здесь остались четыре функции рисования, и у них есть второй потребитель — сетка
результатов, поэтому проверяются они сами по себе, без всякого диалога.

Что охраняют эти тесты — два места, где разметка карточек ломается молча:

* кроп режется по границе кадра и обязан оставаться квадратом нужной стороны: срез за
  край массива numpy обрезает тихо, и карточка выглядит целой, пока в ней не окажется
  полоса с противоположной стороны снимка;
* длина строки пикселей для `Format_RGB888` кратна четырём байтам: у честного BGR-кадра
  она `3 * ширина` и не кратна при каждой второй ширине — ровно такой бывает снимок из
  телефона. На этой сборке Qt кривую строку проходит молча, поэтому проверяем НЕ глазами,
  а побайтовым совпадением массива и тишиной в stderr.

Тесты синтетические: картинка — `np.zeros` с белыми прямоугольниками, лица — `Face` с
нулевым отпечатком. Ни реальных снимков ребёнка, ни моделей.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from core.engine import Face                     # noqa: E402
from ui.face_picker import crops_from            # noqa: E402


def lico(box: tuple[float, float, float, float]) -> Face:
    return Face(box=box, landmarks=None, embedding=np.zeros(512, dtype=np.float32),
                detector="insight")


def kadr(vysota: int = 400, shirina: int = 400,
         belyj: tuple[tuple[int, int], tuple[int, int]] | None = None) -> np.ndarray:
    """Чёрный кадр в три канала; `belyj` = (строки, столбцы) белого прямоугольника."""
    img = np.zeros((vysota, shirina, 3), dtype=np.uint8)
    if belyj is not None:
        slajsy, stolbcy = belyj
        img[slajsy[0]:slajsy[1], stolbcy[0]:stolbcy[1]] = 255
    return img


def dva_lica() -> list[Face]:
    return [lico((0, 0, 40, 40)), lico((100, 100, 300, 300))]


# --- кроп ---------------------------------------------------------------------------


def test_krop_rezaet_po_ramke_i_kvadratirovannyy() -> None:
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[100:200, 50:150] = 255
    crop = crops_from(img, [lico((50.0, 100.0, 150.0, 200.0))])[0]
    assert crop.ndim == 3 and crop.shape[0] == crop.shape[1]
    assert crop.max() == 255                     # белый квадрат попал в кроп
    assert crop.mean() > 0                       # и кроп не чёрный


def test_lico_u_kraya_daeot_krop_bez_poteri_vidimogo() -> None:
    """Лицо вылезает за правый нижний угол кадра. Обрезанный по полю кроп обязан остаться
    содержательным: человек смотрит на то же лицо, что нашёл разбор, а не на чёрный квадрат."""
    img = kadr(100, 100, ((85, 100), (85, 100)))
    crop = crops_from(img, [lico((90.0, 90.0, 300.0, 300.0))])[0]
    assert crop.shape == (160, 160, 3)
    assert crop.max() == 255, "видимая часть лица пропала из кропа"
    assert crop.mean() > 0


def test_lico_v_jugle_daeot_krop_u_kraya_a_ne_pustotu() -> None:
    """Лицо в левом верхнем углу: рамка с запасом упирается сразу в две границы кадра."""
    img = kadr(200, 200, ((0, 20), (0, 20)))
    crop = crops_from(img, [lico((0.0, 0.0, 30.0, 30.0))])[0]
    assert crop.shape == (160, 160, 3)
    assert crop.max() == 255


def test_kvadraty_vsegda_zadannoj_storony() -> None:
    """Карточки обязаны быть одинаковыми по стороне, иначе ряд разъедется."""
    for storona in (64, 160, 200):
        img = kadr(200, 200, ((50, 150), (50, 150)))
        crop = crops_from(img, [lico((50.0, 50.0, 150.0, 150.0))], side=storona)[0]
        assert crop.shape == (storona, storona, 3), storona


def test_vysek_ne_vyhodit_za_granicy_kadra() -> None:
    """Граничные рамки: высокая, отрицательная, в углу, нулевой размера — кроп обязан
    остаться квадратом нужной стороны и не обратиться в пустоту."""
    korobki = [(0.0, 0.0, 10.0, 10.0), (99.0, 99.0, 100.0, 100.0),
               (-20.0, -30.0, 5.0, 6.0), (10.0, 10.0, 30.0, 200.0),
               (50.0, 50.0, 50.0, 50.0)]                        # лицо нулевого размера
    for korobka in korobki:
        img = kadr(300, 100, ((100, 110), (18, 22)))
        crop = crops_from(img, [lico(korobka)])[0]
        assert crop.shape == (160, 160, 3), korobka
        assert crop.size > 0, korobka


def test_vytyanutoe_lico_rezetsja_po_centru_a_ne_ot_kraya() -> None:
    """Вытянутое лицо: квадрат берём по центру обрезанной рамки, а не от её верхнего края.

    На узком высоком снимке (лицо 20x190 px в кадре 100x300) расширение рамки даёт
    прямоугольник 96x266. Если обрезать его «до левого верхнего угла», в карточку попадёт
    только верхняя четверть лица, а белая метка в его середине — строки 100..110 —
    останется за кадром.
    """
    img = kadr(300, 100, ((100, 110), (18, 22)))
    crop = crops_from(img, [lico((10.0, 10.0, 30.0, 200.0))])[0]
    assert crop.max() == 255, "середина лица осталась за кадром карточки"


def test_lico_polnostju_za_kadrom_daeot_zaglushku() -> None:
    """Целиком вне кадра лица быть не должно, но и падать нельзя: отдаём квадрат-заглушку,
    карточка с номером остаётся на месте и окно не обрывается."""
    crop = crops_from(kadr(100, 100), [lico((500.0, 500.0, 600.0, 600.0))])[0]
    assert crop.shape == (160, 160, 3)
    assert crop.size > 0


def test_ramka_kropa_vsegda_vnutri_kadra_i_kvadrat() -> None:
    """Прямая проверка геометрии, а не только следствия: рамка, которой `crops_from` режет
    кадр, обязана целиком лежать внутри и быть квадратной. Иначе `image[verx:..., lev:...]`
    тихо уедет за границу массива, и на одном снимке человек увидит лицо, а на другом —
    полосу из противоположного края.

    Именно здесь живёт гарантия границы кадра: проверка «кроп получился непустой» её не
    даёт — срез за край массива numpy обрезает молча, и карточка выглядит целой.

    Перебираем рамки, которые реально выдают движки на групповом фото: впритык к углу,
    с отрицательными координатами, больше всего кадра и целиком вне кадра.
    """
    from ui.face_picker import _ramka_kadra

    korobki = [(0.0, 0.0, 10.0, 10.0), (99.0, 99.0, 100.0, 100.0),
               (-20.0, -30.0, 5.0, 6.0), (500.0, 500.0, 600.0, 600.0),
               (-500.0, -500.0, -10.0, -10.0), (0.0, 0.0, 5000.0, 5000.0),
               (10.0, 10.0, 30.0, 200.0), (50.0, 50.0, 50.0, 50.0)]
    vysota, shirina = 300, 100
    for korobka in korobki:
        verx, lev, storona = _ramka_kadra(korobka, vysota, shirina)
        assert storona >= 0, korobka
        assert lev >= 0 and verx >= 0, korobka
        assert lev + storona <= shirina, korobka
        assert verx + storona <= vysota, korobka


def test_lico_s_otricatelnymi_koordinatami() -> None:
    """Такую рамку даёт сам поиск лиц: `core.yunet.face_from_row` строит box из (x, y, w, h)
    и по кадру не обрезает, поэтому у лица у края x или y бывают отрицательными. Без
    `max(0, ...)` numpy прочитал бы срез с конца массива: в карточку легла бы противоположная
    сторона снимка, и человек отмечал бы не то лицо, глядя на то же фото."""
    img = kadr(300, 300, ((0, 8), (0, 8)))
    crop = crops_from(img, [lico((-40.0, -40.0, 10.0, 10.0))])[0]
    assert crop.shape == (160, 160, 3)
    assert crop.max() == 255, "видимая часть лица улетела из кропа"


def test_pustyashka_lic_daeot_pustyashku_kropov() -> None:
    assert crops_from(kadr(50, 50), []) == []


# --- QPixmap ------------------------------------------------------------------------


def test_kartinka_v_qpixmap_ne_teryaet_chislya(qapp) -> None:
    """Конвертация в QPixmap не теряет ни размеров, ни значений пикселей — в том числе на
    ширине, у которой строка не кратна четырём байтам: такой бывает реальный снимок.

    `qapp` здесь обязателен не для красоты: без QApplication создание QPixmap завершает
    процесс целиком (qFatal), и тест не упал бы, а обрёл бы весь прогон.
    """
    from PySide6.QtGui import QImage

    from ui.face_picker import face_to_qpixmap

    for shirina in (160, 5, 701):
        img = kadr(9, shirina)
        img[:, :, 0] = 11
        img[:, :, 1] = 22
        img[:, :, 2] = 33
        pm = face_to_qpixmap(img)
        assert (pm.width(), pm.height()) == (shirina, 9), shirina
        back = pm.toImage().convertToFormat(QImage.Format.Format_RGB888)
        arr = np.frombuffer(bytes(back.constBits()), dtype=np.uint8).reshape(
            9, back.bytesPerLine())[:, :3 * shirina].reshape(9, shirina, 3)
        assert np.array_equal(arr, img[:, :, ::-1]), shirina   # Qt ждёт RGB, у нас BGR


def test_nomer_na_kadre_razlichaet_odinakovie_kropy(qapp) -> None:
    """На снимке, сделанном в темноте, все кропы чёрные и одинаковые. Номер, нарисованный
    прямо на кадре, — единственное, чем человек различит «первое» и «второе».

    Проверка байтами, а не «на глаз»: если `naklej_nomer` перестанет рисовать (сломается
    шрифт, кружок уедет за край), карточки снова станут неразличимыми, и никакой обзор
    разметки этого не заметит.
    """
    from PySide6.QtCore import QSize

    from ui.face_picker import face_to_qpixmap, naklej_nomer

    osnovnoj = kadr(160, 160)
    pikseli = {}
    for nomer in (1, 2, 3):
        mass = face_to_qpixmap(naklej_nomer(osnovnoj, nomer)).toImage()
        pikseli[nomer] = bytes(mass.constBits())
    assert len(set(pikseli.values())) == 3, "карточки с разными номерами неразличимы"
    # размер тот же: номер не имеет права менять геометрию карточки и разъезжать ряд
    assert face_to_qpixmap(naklej_nomer(osnovnoj, 12)).toImage().size() == QSize(160, 160)


# --- тишина Qt -------------------------------------------------------------------------


def shumnoe(stroka: str) -> bool:
    """Служебный шум самой offscreen-платформы Qt, к нашему коду отношения не имеет."""
    priznaki = ("This plugin does not support propagateSizeHints()",
                "qt.qpa.fonts: Populating font family aliases")
    return any(priznak in stroka for priznak in priznaki)


def test_qt_ne_pishet_nichego_v_stderr(qapp, capfd: pytest.CaptureFixture[str]) -> None:
    """Вывод тестов обязан быть чистым: Qt на битом формате пикселей или на виджете без
    приложения не бросает исключение, а пишет в stderr — а без QApplication и вовсе
    завершает процесс (проверено: весь прогон обрывался молча). Так что «тесты зелёные»
    про тишину ничего не говорят, и смотрим на строку состояния процесса.

    Меряем тот путь, где Qt обычно и ворчит: строка пикселей, не кратная четырём байтам
    (нечётная ширина снимка), и показ сетки карточек.

    Поблажка одна, и она названа поимённо: служебные строки самой offscreen-платформы
    (см. `shumnoe`) — ей некуда передать размер окна и нечем заплатить за алиасы шрифтов.
    Они приходят один раз на процесс и зависят от порядка тестов; всё остальное — признак
    бага, и чтобы фильтр не проглотил настоящую жалобу, границы предиката проверяет
    `test_shumnoe_ne_opravdyvaet_nashu_oshibku`.
    """
    from PySide6.QtCore import QSize
    from PySide6.QtWidgets import QListWidget

    from ui.face_picker import face_to_qpixmap, naklej_nomer

    def setka(kardry: list) -> QListWidget:
        """Та же сетка в IconMode, что строит окно эталона: кривой пиксель виден на показе."""
        from PySide6.QtWidgets import QListWidgetItem

        spisok = QListWidget()
        spisok.setViewMode(QListWidget.ViewMode.IconMode)
        spisok.setIconSize(QSize(160, 160))
        for nomer, kadr_ in enumerate(kardry, start=1):
            spisok.addItem(QListWidgetItem(face_to_qpixmap(naklej_nomer(kadr_, nomer)),
                                           f"лицо {nomer}"))
        return spisok

    # Разогрев: offscreen-платформа при первом же касании шрифтов и окна печатает свои
    # служебные строки. Они приходят ОДИН раз на процесс и зависят от порядка тестов,
    # поэтому замер начинаем после них, а не до.
    razogrev = setka([kadr(9, shirina) for shirina in (5, 6, 160, 701)])
    razogrev.show()
    razogrev.close()
    _ = capfd.readouterr()

    # дальше — то, что действительно должно молчать
    face_to_qpixmap(kadr(9, 7))
    postrannie = setka([kadr(9, 7), kadr(3, 3)])
    postrannie.show()
    postrannie.close()
    err = capfd.readouterr().err
    lishnee = [strok for strok in err.splitlines() if strok.strip() and not shumnoe(strok)]
    assert lishnee == [], f"Qt пожаловался в stderr: {lishnee}"


def test_shumnoe_ne_opravdyvaet_nashu_oshibku() -> None:
    """Фильтр служебных сообщений не должен проглатывать настоящую жалобу Qt: проверяем
    границы предиката текстом, а не «на веру». Что Qt пишет в stderr и что capfd это видит —
    доказано на этом же модуле: `qt.qpa.fonts: Populating font family aliases` попал в
    «Captured stderr call» упавшего теста."""
    assert shumnoe("qt.qpa.gui: This plugin does not support propagateSizeHints()")
    assert shumnoe("qt.qpa.fonts: Populating font family aliases took 83 ms.")
    assert not shumnoe("QImage::convertToFormat: unsupported image type")
    assert not shumnoe("QPainter::begin: Paint device returned engine == 0")
    assert not shumnoe("")
