"""Сетка результатов: миниатюра с рамкой лица, процент и галочка.

Тест проверяет пять вещей, на которых эта сетка однажды уже спотыкалась:

1. кучки режутся по тому числу, которое человек видит на карточке, — и тем же правилом,
   что и `report.py`, иначе отчёт скажет «похоже», а карточка уедет в «слабое сходство»;
2. снимок, который не показался, никуда не исчезает: он остаётся в списке с прямым
   текстом «не читается» — молчаливый ноль в этом проекте хуже ошибки;
3. миниатюра собирается через `face_to_qpixmap`, а не своим `QImage` из буфера numpy:
   запрет на свою сборку держит проверка исходника, а попиксельное сравнение держит
   сам факт «экран = массив» на кадре с невыровненной строкой (подробно — в тесте);
4. массовая отметка и пересчёт не плодят ни лишних сигналов, ни призраков в раскладке;
5. первая отрисовка находки не декодирует снимки целиком и не читает их все подряд:
   кадр берётся быстрым путём (`Image.draft`), по одному на тик событий, а рамка при
   этом лежит ровно на лице — иначе «не виснет» означало бы «врёт про находку»;
6. слов, запрещённых спецификацией, в сетке нет — обход собирает тексты и до, и после
   заполнения, потому что подпись кучки появляется только после;
7. сетка строит не больше `MAKSIMUM_KARTOTSEK` карточек и поднимает потолок только по
   явному нажатию: две тысячи виджетов — это минуты и сотни мегабайт, а скрытое обязано
   быть названо в подписи числом, а не промолчано;
8. число в пояснении под сеткой живое: оно обязано повторять положение ползунка, а не
   цифру, когда-то зашитую в текст (скриншот человека: ползунок на 38 %, подпись про 45 %).
9. карточка рисует рамку на каждом найденном лице и ни на одном лишнем: одна рамка на
   фото, где ребёнок стоит дважды, читалась как «второго потеряли», а двадцать рамок —
   как «нашли всех».

Ни реальных фото, ни моделей: снимки рисуются `PIL` в `tmp_path`, кэш проверяется
подменой `load_photo` в самом модуле.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

pytest.importorskip("PySide6")

from core.matcher import ScoredPhoto                       # noqa: E402
from core.engine import Face                               # noqa: E402
from core.report import ResultRow, Sovpadenie, build_rows              # noqa: E402
from ui.face_picker import (MIN_RAMKA_PX, ZELENYJ, RamkaLica,  # noqa: E402
                            ramka_na_kadr)
from ui.results_view import (DOBAVLYAET_KARTOTSEK, MAKSIMUM_KARTOTSEK,  # noqa: E402
                             ResultsView, annotated_thumbnail, THUMB)

if TYPE_CHECKING:                        # подсказки для редактора, импорта в рантайме нет
    from PySide6.QtWidgets import QWidget

# Стем, а не слово целиком: ловим и «порог», и «порога», и «детектора».
STEM = ("порог", "уверенност", "embedd", "косинус", "детектор")
# «пол» — отдельной строкой и только как целое слово: как подстрока он сидит в «ползунке».
CELYE_SLOVA = (r"\bпол(а|у|ом|е)?\b", r"\bвозраст(а|у|е|ом)?\b")


@pytest.fixture
def view(qapp: object) -> Iterator[ResultsView]:
    """Сетка, которая после теста перестаёт что-либо читать.

    `hide()` здесь не косметика: очередь миниатюр живёт на `QTimer`, а таймер скрытого
    окна молчит. Без этого шага виджет, показанный одним тестом, дочитывал бы свои
    карточки во время следующего — и счётчик чтений враньём обвинил бы не тот тест.
    """
    setka = ResultsView()
    yield setka
    setka.hide()
    setka.deleteLater()


def zdat_kadry(qapp: object, view: ResultsView, ms: int = 8000) -> None:
    """Дать очереди миниатюр откачаться, крутя события вместо `sleep`.

    Кадры приходят по одному за тик — ровно для того, чтобы окно отвечало на клики.
    Тесту нужен тот же механизм: ждём, пока очередь опустеет, и попутно прокручиваем
    события, иначе карточки не доедут никогда.
    """
    konec = time.monotonic() + ms / 1000.0
    while view._ochered and time.monotonic() < konec:
        qapp.processEvents()          # type: ignore[union-attr]
        time.sleep(0.002)
    qapp.processEvents()              # type: ignore[union-attr]


def sroki_zhanra(percents: list[int] | None = None) -> list[ResultRow]:
    """Готовые строки результата. Пути несуществующие: делёж на кучки к диску не обращается."""
    znanii = [62, 55, 40, 30, 12] if percents is None else percents
    return [ResultRow(Path(f"/ф{i}.jpg"), percent=p, faces=1,
                      box=(0.0, 0.0, 10.0, 15.0), matched=p >= 38)
            for i, p in enumerate(znanii)]


def vse_galochki(vidzhet: "QWidget") -> list:
    """Видимые галочки сетки. Скрытых призраков тест видеть не должен (см. ниже)."""
    from PySide6.QtWidgets import QCheckBox

    return [c for c in vidzhet.findChildren(QCheckBox) if c.isVisibleTo(vidzhet)]


# --- контракт из плана --------------------------------------------------------------


def test_delit_na_kuchki_po_porogu(view) -> None:
    view.show_rows(sroki_zhanra(), threshold=0.38)
    assert view.counts == (3, 2)                   # 3 «похоже», 2 «слабое сходство»


def test_otmecheny_tolko_pohozhie(view) -> None:
    view.show_rows(sroki_zhanra(), threshold=0.38)
    assert len(view.checked) == 3


def test_knopka_otmetit_vse_pohozhie_dobavlyaet_slaboe(view) -> None:
    """«Отметить все» обязана отметить ВСЁ найденное, включая свёрнутую кучку.

    Иначе кнопка делает не то, что на ней написано: человек жмёт «отметить все», получает
    три из пяти и узнаёт о пропаже только в папке результатов. Поэтому кучка сначала
    разворачивается, и только потом ставятся галочки — отмеченное обязательно видно.
    """
    view.show_rows(sroki_zhanra(), threshold=0.38)
    view.check_all()
    assert view.checked == [Path(f"/ф{i}.jpg") for i in range(5)]
    assert view.counts == (3, 2), "разворот кучки не имеет права менять делёж"


def test_snyat_vse_galochek(view) -> None:
    view.show_rows(sroki_zhanra(), threshold=0.38)
    view.uncheck_all()
    assert view.checked == []


def test_miniatyura_bez_fayla_dast_None(tmp_path: Path) -> None:
    assert annotated_thumbnail(tmp_path / "net.jpg", [RamkaLica((0, 0, 10, 10), 1)], 2400) is None


def test_miniatyura_s_ramkoy(tmp_path: Path) -> None:
    from PIL import Image

    p = tmp_path / "a.jpg"
    Image.new("RGB", (800, 600), (9, 9, 9)).save(p, "JPEG")
    with_box = annotated_thumbnail(p, [RamkaLica((100.0, 100.0, 300.0, 300.0), 1)], 2400)
    plain = annotated_thumbnail(p, [], 2400)
    assert max(with_box.shape[:2]) == 220
    assert (with_box == np.array([0, 255, 0])).any()      # зелёная рамка нарисована
    assert not (plain == np.array([0, 255, 0])).any()
    # `cv2.rectangle` рисует на месте: если вторая миниатюра того же файла вернула бы
    # тот же массив, на ней осталась бы чужая рамка
    assert (plain == np.array([9, 9, 9])).all()


def test_miniatyura_ne_menjaet_sosednyaya_ramka(tmp_path: Path) -> None:
    """Два вызова для одного файла с разными рамками не заражают друг друга."""
    from PIL import Image

    p = tmp_path / "b.jpg"
    Image.new("RGB", (400, 400), (9, 9, 9)).save(p, "JPEG")
    a = annotated_thumbnail(p, [RamkaLica((50.0, 50.0, 150.0, 150.0), 1)], 2400)
    b = annotated_thumbnail(p, [RamkaLica((200.0, 200.0, 300.0, 300.0), 1)], 2400)
    assert a is not b
    zelenyj = np.array([0, 255, 0])
    assert (a == zelenyj).any() and (b == zelenyj).any()
    assert np.argwhere(a == zelenyj[0]).tolist() != np.argwhere(b == zelenyj[0]).tolist()


def test_degenerirovannyj_kadr_ne_ronyaet_setku(tmp_path: Path, qapp) -> None:
    """Полоска в 1 пиксель высотой — легальный ответ `load_photo` (задача 3 уменьшает по
    длинной стороне). `int(1 * k)` даёт 0, а `cv2.resize` на нулевой стороне бросает
    assertion: сетка рушилась бы на панорамном снимке целиком, а не на одной карточке."""
    from PIL import Image
    from PySide6.QtWidgets import QLabel

    p = tmp_path / "polosa.jpg"
    Image.new("RGB", (2000, 1), (12, 30, 40)).save(p, "JPEG")
    mini = annotated_thumbnail(p, [RamkaLica((0.0, 0.0, 500.0, 1.0), 1)], 2400)
    assert mini is not None and min(mini.shape[:2]) >= 1
    # и до экрана эта карточка должна дожить: на полоске плашка процента не рисуется,
    # иначе чёрный прямоугольник съел бы весь снимок
    vidzet = ResultsView()
    vidzet.show_rows([ResultRow(p, percent=44, faces=1, box=(0.0, 0.0, 500.0, 1.0),
                                matched=True)], threshold=0.38)
    kartinki = [et for et in vidzet.findChildren(QLabel)
                if et.pixmap() is not None and not et.pixmap().isNull()]
    assert len(kartinki) == 1
    chernye = 0
    for x in range(0, 60):
        c = kartinki[0].pixmap().toImage().pixelColor(x, 0).getRgb()[:3]
        if max(c) < 12:                       # именно чёрная плашка, а не зелёная рамка
            chernye += 1
    assert chernye == 0, "плашка процента легла на всю полоску"


# --- делёж совпадает с отчётом -------------------------------------------------------


@pytest.mark.parametrize("procenty, naporog", [
    ([56, 55, 54], 0.56),        # ловушка float: 0.56 * 100 == 56.00000000000001
    ([55, 54, 53], 0.55),
    ([28, 27], 0.28),
    ([40, 39, 38], 0.38),
    ([30, 29], 0.30),
])
def test_kuchki_rezhutsya_kak_otchyot(view, procenty, naporog) -> None:
    """Строки строит настоящий `build_rows` — тот же, что питает report.txt.

    `build_rows` сравнивает округлённый процент с `matcher.percent_of(threshold)`. Сетка
    обязана резать кучки по тому же правилу: голое `percent >= threshold * 100` на 0.56
    даёт `56 >= 56.00000000000001 == False`, и карточка с числом «56%» уезжает в «слабое
    сходство», а отчёт рядом печатает её как «похоже». Разъезд человек видит как
    противоречие между двумя частями одного экрана.
    """
    lico = Face(box=(0.0, 0.0, 10.0, 15.0), landmarks=None,
                embedding=np.eye(512, dtype=np.float32)[0], detector="insight")
    fajn = {Path(f"/ф{i}.jpg"): [lico] for i in range(len(procenty))}
    ocenki = {p: ScoredPhoto(percent=procenty[i], similarity=procenty[i] / 100.0, face=lico)
              for i, p in enumerate(fajn)}
    sroki = build_rows(fajn, [ ocenki], naporog)

    view.show_rows(sroki, threshold=naporog)
    n_pohozhe = sum(1 for r in sroki if r.matched)
    assert view.counts == (n_pohozhe, len(sroki) - n_pohozhe)
    assert len(vse_galochki(view)) == n_pohozhe


def test_otmetka_soglasovana_s_kuchkoj(view) -> None:
    """Галочка ставится по делёжу этой сетки, а не по флагу `matched` в строке.

    Флаг посчитан на пороге, который был у вызывающего минуту назад. Отметить карточку,
    которая лежит в «слабом сходстве», — значит сказать человеку «моё» и «не уверен»
    в одном месте экрана.
    """
    sroki = sroki_zhanra([62, 55, 40, 30, 12])      # matched=True у всех >=38
    assert any(r.matched and r.percent < 56 for r in sroki), "флаг специально устарел"
    view.show_rows(sroki, threshold=0.56)
    assert [str(p) for p in view.checked] == ["/ф0.jpg"]
    assert sum(1 for _, c in view._boxes if c.isChecked()) == 1


def test_skrytoe_ne_kopiruetsja_i_govoritsja_slovoami(view) -> None:
    """В `checked` только то, что на экране, и об этом сказано словами в подписи."""
    view.show_rows(sroki_zhanra(), threshold=0.38)
    view.check_all()                                 # разворачивает и отмечает все пять
    assert len(view.checked) == 5
    view.toggle_weak.click()                         # сворачиваем обратно
    assert view.checked == [Path(f"/ф{i}.jpg") for i in range(3)]
    assert "скрыт" in view.summary.text(), "о спрятанных галочках надо сказать словами"


def test_galochki_pomnyatsja_pri_perekljuchenii_kuchki(view) -> None:
    """Отметил слабое, свернул, развернул — отметка на месте, а не пропала молча."""
    view.show_rows(sroki_zhanra(), threshold=0.38)
    view.toggle_weak.click()                          # 5 карточек
    view._boxes[3][1].setChecked(True)                # ф3 (30%) — из слабого сходства
    view.toggle_weak.click()                          # свернули
    assert Path("/ф3.jpg") not in view.checked
    view.toggle_weak.click()                          # развернули
    assert Path("/ф3.jpg") in view.checked


def test_odin_signal_na_massovuju_otmetku(view) -> None:
    """Массовая отметка не обязана гнать по сигналу на каждую карточку: на архиве в
    несколько сотен фото это сотни включений кнопки копирования подряд."""
    view.show_rows(sroki_zhanra([62] * 40), threshold=0.38)
    prinjato: list = []
    view.selection_changed.connect(lambda spiski: prinjato.append(len(spiski)))
    view.check_all()
    assert len(prinjato) == 1 and prinjato[0] == 40


def test_odin_signal_na_klik_cheloveka(view) -> None:
    view.show_rows(sroki_zhanra(), threshold=0.38)
    prinjato: list = []
    view.selection_changed.connect(lambda spiski: prinjato.append(list(spiski)))
    view._boxes[0][1].click()
    assert len(prinjato) == 1 and Path("/ф0.jpg") not in prinjato[0]


# --- содержимое карточки -------------------------------------------------------------


def test_setka_ne_derzhit_zastarjeleshih_kletok(view) -> None:
    """Пересчёт пересоздаёт карточки. Старые обязаны исчезнуть с экрана, а не остаться
    призраками под новыми: снятый с раскладки виджет Qt сам не прячется."""
    view.show_rows(sroki_zhanra(), threshold=0.38)
    view.show_rows(sroki_zhanra([62]), threshold=0.38)
    assert len(vse_galochki(view)) == 1
    assert view._grid.count() == 1


def test_kolony_idut_za_shirinoj_okna(view, qapp) -> None:
    """Число карточек в ряду считается по ширине, а не заковано в четвёрку.

    Замер живого окна (см. отчёт): четыре колонки по 220 px в окне 1120 px давали
    горизонтальную полосу прокрутки, часть карточек уезжала за край, и человек читал
    это как «нашлось меньше».

    `show()` здесь обязателен: Qt не раскладывает скрытый виджет, и до показа ширина
    области прокрутки — случайные 640 px вместо настоящих.
    """
    sroki = sroki_zhanra([62] * 12)
    view.show()
    view.resize(1400, 700)
    qapp.processEvents()
    shirina = view._scroll.viewport().width()
    assert shirina > 1000, f"окно не разложено: viewport {shirina} px"
    view.show_rows(sroki, threshold=0.38)
    kolonki = view._kolonok_v_rjadu
    assert 1 < kolonki <= 8, f"широкое окно дало {kolonki} колонок"
    assert view._grid.columnCount() == kolonki
    # главный инвариант: ряду некуда вылезать за край
    assert view._grid.minimumSize().width() <= shirina + 1
    assert view._scroll.horizontalScrollBar().maximum() == 0, "полоса прокрутки вернулась"

    view.resize(320, 700)
    qapp.processEvents()
    assert view._kolonok_v_rjadu == 1, "в узком окне карточки не встают в ряд"
    assert view._scroll.horizontalScrollBar().maximum() == 0


def test_perekos_okna_perekadyvaet_ryady(view, qapp) -> None:
    """Сетка перекладывается на resize, а не ждёт следующего `show_rows`: человек
    развернул окно в пол-экрана и хочет видеть больше карточек сразу."""
    view.show()
    view.resize(560, 700)
    qapp.processEvents()
    view.show_rows(sroki_zhanra([62] * 12), threshold=0.38)
    uzko = view._kolonok_v_rjadu
    view.resize(1400, 700)
    qapp.processEvents()
    assert view._kolonok_v_rjadu > uzko
    assert len(vse_galochki(view)) == 12, "перекладка не имеет права терять карточки"


def test_nechitaemyj_fajn_ostaetsya_v_spiske(view, tmp_path: Path) -> None:
    """Файл, который не открылся, не исчезает: карточка остаётся и говорит об этом
    прямым текстом — молчаливого нуля здесь не бывает."""
    from PySide6.QtWidgets import QLabel

    net = tmp_path / "net.jpg"
    view.show_rows([ResultRow(net, percent=70, faces=2, box=None, matched=True)],
                   threshold=0.38)
    assert len(vse_galochki(view)) == 1
    nadpisi = [et.text() for et in view.findChildren(QLabel) if et.text()]
    assert any("не читается" in t for t in nadpisi), nadpisi


def test_kartochka_govorit_chislo_i_imya(view, tmp_path: Path) -> None:
    """На карточке человек обязан увидеть процент и имя файла — по имени он пойдёт искать
    снимок в своей папке. Число лиц на снимке — в подсказке."""
    p = tmp_path / "IMG_0001.jpg"
    p.write_bytes(b"ne-kartinka")
    view.show_rows([ResultRow(p, percent=51, faces=3, box=None, matched=True)],
                   threshold=0.38)
    podpisi = [c.text() for c in vse_galochki(view)]
    assert any("51%" in t and "IMG_0001.jpg" in t for t in podpisi), podpisi
    # число лиц на снимке — в подсказке карточки: наводишь на картинку, а не на галочку
    assert any("лиц на снимке: 3" in t for t in vse_teksty(view))


# --- миниатюра в Qt -------------------------------------------------------------------


def test_setka_risujaet_miniaturu_bes_sdviga(view, tmp_path: Path) -> None:
    """Картинка на экране совпадает с массивом попиксельно на кадре с НЕвыРОВНЕННОЙ
    строкой (3 * 101 = 303 байта).

    Это тот самый случай, ради которого сетка импортирует `face_to_qpixmap`, а не собирает
    `QImage(rgb.data, w, h, 3*w, Format_RGB888)` своей строкой: для `Format_RGB888` Qt
    требует длину строки, кратную четырём байтам, а у честного BGR-кадра там `3 * ширина`,
    и снимок с телефона ровно такой и есть. Честно о мере: на этой сборке PySide6
    (6.11.2) кривая строка проходит молча и рисует правильно — замерено, см. отчёт, —
    поэтому запрет на свою сборку держит тест по исходнику модуля ниже, а этот тест
    держит сам факт: то, что видит человек, побайтно равно тому, что посчитал OpenCV.
    На платформе, где Qt проигнорирует `bytesPerLine`, именно он покажет перекошенное
    лицо без единой ошибки в логе.
    """
    from PIL import Image
    from PySide6.QtWidgets import QLabel

    # 101 px ширины при длинной стороне 220 — кадр, который уменьшать не нужно, и
    # ровно такой, у которого строка пикселей (3 * 101 = 303 байта) не кратна четырём.
    p = tmp_path / "portret.jpg"
    kadr = np.zeros((220, 101, 3), dtype=np.uint8)
    kadr[:110, :50] = 0                    # чёрный верх-лево
    kadr[:110, 50:] = 255                  # белый верх-право
    kadr[110:, :50] = 255                  # белый низ-лево
    kadr[110:, 50:] = 0                    # чёрный низ-право
    Image.fromarray(kadr).save(p, "JPEG", quality=100)

    ozhidanie = annotated_thumbnail(p, [], 2400)
    assert ozhidanie is not None
    vysota, shirina = ozhidanie.shape[:2]
    assert (vysota, shirina) == (220, 101)
    assert (3 * shirina) % 4 != 0, f"тест не имеет смысла на строке в {3 * shirina} байт"

    view.show_rows([ResultRow(p, percent=70, faces=1, box=None, matched=True)],
                   threshold=0.38)
    kartinki = [et for et in view.findChildren(QLabel)
                if et.pixmap() is not None and not et.pixmap().isNull()]
    assert len(kartinki) == 1
    na_ekrane = kartinki[0].pixmap().toImage()
    assert (na_ekrane.width(), na_ekrane.height()) == (shirina, vysota)
    sdvinalis = 0
    # Допуск большой нарочно: JPEG quality=100 даёт звон у самой границы контраста на
    # единицы, а съехавший ряд меняет 0 на 255. Меряем сдвиг, а не артефакты сжатия.
    # Первые строки пропускаем: там плашка процента, она рисуется поверх кадра (см.
    # `test_chislo_narisovano_na_samoj_kartochke`).
    for y in range(28, vysota - 2, 7):
        for x in range(2, shirina - 2, 7):
            pokaz = na_ekrane.pixelColor(x, y).getRgb()[0]
            massiv = int(ozhidanie[y][x][0])
            if abs(pokaz - massiv) > 40:
                sdvinalis += 1
    assert sdvinalis == 0, f"{sdvinalis} точек съехало — строки выровнены неверно"


def test_chislo_narisovano_na_samoj_kartochke(view, tmp_path: Path) -> None:
    """Процент обязан быть на снимке, а не только в подписи под ним.

    Живой прогон окна (см. отчёт) показал: полоса результатов на ноутбуке 13" начинается
    ниже сгиба, подпись под карточкой остаётся за краем — и человек крутил бы колесо
    мыши ради каждого числа, то есть ради главного признака решения.
    """
    from PIL import Image
    from PySide6.QtWidgets import QLabel

    p = tmp_path / "r.jpg"
    Image.new("RGB", (300, 300), (90, 90, 90)).save(p, "JPEG")
    view.show_rows([ResultRow(p, percent=62, faces=1, box=None, matched=True)],
                   threshold=0.38)
    kartinka = [et for et in view.findChildren(QLabel)
                if et.pixmap() is not None and not et.pixmap().isNull()]
    assert len(kartinka) == 1
    na_ekrane = kartinka[0].pixmap().toImage()
    ugol = [(x, y) for y in range(0, 26) for x in range(0, 60)]
    belye = sum(1 for x, y in ugol if na_ekrane.pixelColor(x, y).red() > 200)
    chernye = sum(1 for x, y in ugol if na_ekrane.pixelColor(x, y).red() < 40)
    assert chernye > 200, "тёмной плашки под числом нет"
    assert belye > 20, "белых пикселей цифры нет"
    # и при этом вне угла кадр остался собой
    assert abs(na_ekrane.pixelColor(150, 150).red() - 90) <= 6


def test_ramka_poverh_plashki_chisla(view, tmp_path: Path) -> None:
    """Порядок слоёв: лицо важнее числа. Ребёнок на снимке вполне может сидеть в левом
    верхнем углу, и плашка процента не имеет права съесть рамку, по которой человек
    проверяет находку."""
    from PIL import Image
    from PySide6.QtWidgets import QLabel

    p = tmp_path / "ugol.jpg"
    Image.new("RGB", (300, 300), (90, 90, 90)).save(p, "JPEG")
    view.show_rows([ResultRow(p, percent=62, faces=1,
                              box=(2.0, 2.0, 120.0, 120.0), matched=True)],
                   threshold=0.38)
    na_ekrane = [et for et in view.findChildren(QLabel)
                 if et.pixmap() is not None and not et.pixmap().isNull()][0].pixmap().toImage()
    # Считаем зелёное ровно внутри плашки (0..40, 0..26): если порядок слоёв обратный,
    # рамка в этом углу исчезнет, а за пределами плашки она осталась бы видна — и тест
    # молча стал бы бесполезным.
    zelenye_v_plashke = sum(1 for y in range(0, 26) for x in range(0, 40)
                            if na_ekrane.pixelColor(x, y).green() > 200
                            and na_ekrane.pixelColor(x, y).red() < 120)
    assert zelenye_v_plashke > 20, "рамка лица в углу съедена плашкой процента"


# --- рамка ТОЛЬКО на найденных лицах (задача T19) ---------------------------------------
#
# Зелёная рамка в этом приложении значит «вот тут найденный ребёнок» — так ей пишет и
# подпись под сеткой. Значит обязана стоять на каждом лице выше ползунка (ребёнок может
# стоять в кадре дважды) и ни на одном другом: тех, кого не искали, красить нельзя.

# Три «лица» в координатах кадра 600 px. Снимок ровно этого разрешения, поэтому
# коэффициент рамки = 220/600 и тест не пересчитывает чужой масштаб.
TROE_LICA = ((20.0, 200.0, 120.0, 300.0),
             (240.0, 200.0, 340.0, 300.0),
             (460.0, 200.0, 560.0, 300.0))
_K = THUMB / 600


def _cvela_kadra(kadr: np.ndarray, box) -> set[tuple[int, int, int]]:
    """Какие цвета стоят внутри прямоугольника лица на массиве миниатюры."""
    x1, y1, x2, y2 = (int(v * _K) for v in box)
    kusok = kadr[max(0, y1):y2 + 1, max(0, x1):x2 + 1].reshape(-1, 3)
    return {tuple(int(c) for c in piksel) for piksel in kusok}


def _cvela_ekrana(iz, box) -> set[tuple[int, int, int]]:
    """То же самое по pixmap'у карточки. Буфер в numpy не тянем: stride здесь не для чего
    ловить, а обход `pixelColor` не зависит от выравнивания строки."""
    x1, y1, x2, y2 = (int(v * _K) for v in box)
    naideno = set()
    for y in range(max(0, y1), min(iz.height(), y2 + 1)):
        for x in range(max(0, x1), min(iz.width(), x2 + 1)):
            naideno.add(iz.pixelColor(x, y).getRgb()[:3])
    return naideno


def test_miniatyura_risyuet_tolko_perechislennye_lica(tmp_path: Path) -> None:
    """Рамки рисуют по присланному списку: что в списке, то и обведено.

    Проверка на чистой функции, а не на виджете: её вывод обязан сходиться с экраном
    байт в байт, и именно ей сетка обещает человеку зелёную рамку на находку.
    """
    from PIL import Image

    p = tmp_path / "troe.png"
    Image.new("RGB", (600, 600), (90, 90, 90)).save(p)      # PNG: сжатие не размоет линии

    a, b, c = TROE_LICA
    kadr = annotated_thumbnail(p, [RamkaLica(b, 2)], 600)
    assert kadr is not None
    assert ZELENYJ in _cvela_kadra(kadr, b)
    for bok in (a, c):
        assert ZELENYJ not in _cvela_kadra(kadr, bok), \
            f"обвели лицо {bok}, которого в списке нет"


def test_kartochka_risyuet_ramku_na_kazhdom_najdennom_lice(view, tmp_path: Path) -> None:
    """Ребёнок в кадре дважды — две рамки. Замер архива: лица 22 и 24 по 65 %.

    До правки `score_photo` возвращал один максимум, второе совпадение пропадало молча,
    и человек видел одну рамку там, где его детей двое.
    """
    from PIL import Image
    from PySide6.QtWidgets import QLabel

    p = tmp_path / "dvoe.png"
    Image.new("RGB", (600, 600), (90, 90, 90)).save(p)
    a, b, c = TROE_LICA
    view.show_rows([ResultRow(p, percent=65, faces=3, box=b, matched=True,
                              vse_lica=(a, b, c), lice=2, sovpadeniya=(Sovpadenie(1, 2, 65), Sovpadenie(1, 3, 65)))],
                   threshold=0.38, max_dim=600)
    na_ekrane = [et for et in view.findChildren(QLabel)
                 if et.pixmap() is not None and not et.pixmap().isNull()][0].pixmap().toImage()
    assert ZELENYJ in _cvela_ekrana(na_ekrane, b), "у первого совпадения нет рамки"
    assert ZELENYJ in _cvela_ekrana(na_ekrane, c), "второе совпадение потерялось"
    assert ZELENYJ not in _cvela_ekrana(na_ekrane, a), "помечен тот, кого не искали"


def test_kartochka_bez_sovpadenij_ne_obvodyaet_nichego(view, tmp_path: Path) -> None:
    """Ни одного лица выше ползунка — на карточке нет ни одной рамки.

    Соблазн обвести «хотя бы самое похожее» здесь и есть та самая путаница: зелёная рамка
    значит «найдено», а не «из этих двадцати четырёх вот это чуть более похоже». Лучшее
    лицо человек увидит в окне просмотра — там для этого есть его квадрат.
    """
    from PIL import Image
    from PySide6.QtWidgets import QLabel

    p = tmp_path / "slaboe.png"
    Image.new("RGB", (600, 600), (90, 90, 90)).save(p)
    a, b, c = TROE_LICA
    view.show_rows([ResultRow(p, percent=30, faces=3, box=b, matched=False,
                              vse_lica=(a, b, c), lice=2, sovpadeniya=())],
                   threshold=0.38, max_dim=600)
    view.toggle_weak.click()            # слабое сходство по умолчанию свёрнуто
    na_ekrane = [et for et in view.findChildren(QLabel)
                 if et.pixmap() is not None and not et.pixmap().isNull()][0].pixmap().toImage()
    for bok in (a, b, c):
        assert ZELENYJ not in _cvela_ekrana(na_ekrane, bok), bok


def test_stroka_bez_spiska_sovpadenij_risyuet_svoyo_ramku(view, tmp_path: Path) -> None:
    """Строку собрали не `build_rows` — рамку лучшего лица она терять не обязана.

    Так конструируют строку тесты и ручные вызовы: `box` и `matched` есть, а списка
    совпадений нет. Молча пустая миниатюра означала бы «находка пропала с экрана».
    """
    from PIL import Image
    from PySide6.QtWidgets import QLabel

    p = tmp_path / "odno.png"
    Image.new("RGB", (600, 600), (90, 90, 90)).save(p)
    view.show_rows([ResultRow(p, percent=62, faces=1, box=TROE_LICA[1], matched=True)],
                   threshold=0.38, max_dim=600)
    na_ekrane = [et for et in view.findChildren(QLabel)
                 if et.pixmap() is not None and not et.pixmap().isNull()][0].pixmap().toImage()
    assert ZELENYJ in _cvela_ekrana(na_ekrane, TROE_LICA[1]), \
        "единственная рамка исчезла вместе со списком совпадений"


def test_setka_ne_sobiraet_qimage_iz_bufera_sama() -> None:
    """Самопроверка по итогам ревью задачи 13: перенос `QImage(..., 3*w, Format_RGB888)`
    в новый модуль — это ровно тот баг со stride, который там уже починили."""
    ishodnik = Path(__file__).resolve().parents[1] / "src" / "ui" / "results_view.py"
    tekst = ishodnik.read_text(encoding="utf-8")
    assert "QImage(" not in tekst, "миниатюры собирает face_to_qpixmap, а не свой QImage"
    assert "face_to_qpixmap" in tekst


def test_miniatyura_ne_chitaet_fajn_povtorno(view, tmp_path: Path, monkeypatch,
                                             qapp) -> None:
    """Ползунок человек водит по двадцати значениям за прогон. Если каждое движение снова
    декодирует сотни снимков, «мгновенный пересчёт» превращается в подвисание окна:
    базовая миниатюра кэшируется, а рамка дорисовывается на копии.

    Кэш обязан переживать и смену похожести (одни и те же файлы), и промах по файлу:
    нечитаемый снимок не должен раз в пересчёт лезть на диск за новым отказом.

    Считается `_kadr_miniatury` — единственная функция модуля, которая трогает диск.
    `load_photo` для этого не годится: миниатюры JPEG она вообще не вызывает (быстрый
    декод через `Image.draft` дешевле), а смысл теста именно в «один файл — одно
    чтение».
    """
    from PIL import Image

    import ui.results_view as modul

    puti = []
    for i in range(3):
        p = tmp_path / f"ф{i}.jpg"
        Image.new("RGB", (300, 300), (20 + i, 20, 20)).save(p, "JPEG")
        puti.append(p)
    sroki = [ResultRow(p, percent=60, faces=1, box=(10.0, 10.0, 90.0, 90.0), matched=True)
             for p in puti]
    sroki.append(ResultRow(tmp_path / "net.jpg", percent=50, faces=1, box=None,
                           matched=True))

    vzivy: list = []
    nastoyashhij = modul._kadr_miniatury

    def schetchik(path, *a, **kw):
        vzivy.append(Path(path))
        return nastoyashhij(path, *a, **kw)

    monkeypatch.setattr(modul, "_kadr_miniatury", schetchik)
    view.show()                              # скрытая очередь не читает — см. `_zapatit_ochered`
    view.show_rows(sroki, threshold=0.38)
    zdat_kadry(qapp, view)
    pervyj_raund = len(vzivy)
    for procent in (0.40, 0.42, 0.44, 0.46):
        view.show_rows(sroki, threshold=procent)
        zdat_kadry(qapp, view)
    assert pervyj_raund == 4, "читается ровно один раз на файл, включая битый"
    assert len(vzivy) == 4, f"после кэша файлы открывались ещё {len(vzivy) - 4} раз"


# --- потолок сетки: карточка — это виджет, и их не может быть сколько угодно ---------------
#
# Замер на этом же архиве, развёрнутый до синтетики: 3000 карточек ≈ 350 мс + 772 МБ RSS,
# 6000 ≈ 1,04 с + 859 МБ, и это только сборка виджетов, до единого прочитанного снимка.
# Плюс ~160 мс очереди декодировки на карточку. Поэтому сетка обязана стоять на потолке
# и обязана говорить об этом словами — молча показать триста из двух тысяч было бы
# цензурой, а «приложение нашло 300» вместо «показало 300 из 2000» было бы враньём.


def tysyachi_strok(n: int, imya: str = "ф") -> list[ResultRow]:
    """`n` строк на несуществующих путях. Диск тут не читается: делёж и сборка виджетов
    к нему не обращаются, а нечитаемый файл сетка честно показывает заглушкой."""
    return [ResultRow(Path(f"/{imya}{i}.jpg"), percent=62, faces=1,
                      box=(0.0, 0.0, 10.0, 15.0), matched=True) for i in range(n)]


def test_setka_stroit_ne_bolshe_potolka_kartotek(view) -> None:
    """Две тысячи строк — это не две тысячи виджетов. И про скрытое сказано числом."""
    view.show_rows(tysyachi_strok(2000), threshold=0.38)
    assert len(view._boxes) == MAKSIMUM_KARTOTSEK
    assert len(vse_galochki(view)) == MAKSIMUM_KARTOTSEK
    # Числа кучок считаются по всей находке: потолок про показ, а не про результат.
    assert view.counts == (2000, 0), "потолок не имеет права врать про число находок"
    # Очередь декодировки закрыта тем же потолком: экран куплен, а не архив в целом.
    assert len(view._ochered) <= MAKSIMUM_KARTOTSEK
    assert f"Показано: {MAKSIMUM_KARTOTSEK} из 2000" in view.summary.text()
    assert not view.show_more_btn.isHidden(), "скрытое обязано быть названо кнопкой"
    # и новые подписи проверяются на запрещённые слова вместе с остальными
    proveryu_na_zaprety(vse_teksty(view) + [view.summary.text(),
                                            view.show_more_btn.text()])


def test_potolok_snimaet_tolko_nazhatie_knopki(view) -> None:
    """Ни пересчёт похожести, ни повторный показ, ни «отметить все» потолок не поднимают:
    автоматическое сняние было бы тем же зависшим окном, только реже."""
    sroki = tysyachi_strok(2000)
    view.show_rows(sroki, threshold=0.38)
    bylo = len(view._boxes)
    view.show_rows(sroki, threshold=0.40)        # пересчёт, который делает окно
    assert len(view._boxes) == bylo, "пересчёт не имеет права дорисовывать виджеты"
    view.check_all()                             # массовая отметка разворачивает кучку
    assert len(view._boxes) == bylo, "отметка не имеет права рисовать виджеты пачкой"
    assert "Отмечено, но скрыто" in view.summary.text()
    view.show_more_btn.click()
    assert len(view._boxes) == bylo + DOBAVLYAET_KARTOTSEK


def test_potolok_derzhitsja_poka_nabor_tot_zhe(view) -> None:
    """Человек нажал «показать ещё» и подвинул ползунок — скрытое обратно не прячется.
    Снимает потолок только новый набор файлов, то есть новый разбор."""
    sroki = tysyachi_strok(1000)
    view.show_rows(sroki, threshold=0.38)
    view.show_more_btn.click()
    posle_knopki = len(view._boxes)
    assert posle_knopki == 2 * MAKSIMUM_KARTOTSEK
    view.show_rows(sroki, threshold=0.44)
    assert len(view._boxes) == posle_knopki, "пересчёт молча вернул прежний потолок"
    view.show_rows(tysyachi_strok(1000, imya="drugoe"), threshold=0.44)
    assert len(view._boxes) == MAKSIMUM_KARTOTSEK, "новый архив обязан начаться с потолка"


def test_knopka_ishchet_kogda_skrytogo_ne_ostalos(view) -> None:
    """Кнопка обещает ровно столько, сколько прибавит (меньше потолка — остаток), и
    исчезает, когда скрывать стало нечего: «показать ещё 0 фото» было бы обещанием
    без содержания, а подпись про скрытое — враньём."""
    view.show_rows(tysyachi_strok(500), threshold=0.38)
    ostatok = 500 - MAKSIMUM_KARTOTSEK
    assert view.show_more_btn.text() == f"показать ещё {ostatok} фото"
    view.show_more_btn.click()
    assert len(view._boxes) == 500
    assert view.show_more_btn.isHidden()
    assert "Показано:" not in view.summary.text()


# --- первая отрисовка: полного кадра в потоке интерфейса быть не может ----------------

# Снимок заведомо длиннее `max_dim`: только так видно, каким кадрам оплачена миниатюра.
RAZMER_SNIMKA = (3000, 2250)
# «Лицо» на нём — тёмный прямоугольник в пикселях ИСХОДНИКА.
LITSO_RODNOE = (1000, 750, 1500, 1200)
# Те же четыре числа в кадре 2400 px — именно в такой системе движок возвращает рамку.
LITSO_V_KADRE_2400 = (800.0, 600.0, 1200.0, 960.0)


def narisovat_lico(tmp_path: Path, imja: str = "krupnoe.jpg") -> Path:
    """Настоящий JPEG, где «лицо» — тёмный прямоугольник в координатах исходника.

    Размер выбран нарочно больше `max_dim` (3000 px против 2400): на таком снимке
    быстрый декод читает 1/8 кадра, и проверка рамки перестаёт быть тавтологией.
    Тёмное пятно — независимый ориентир: рамка обязана лечь ровно на него.
    """
    from PIL import Image

    shirina, vysota = RAZMER_SNIMKA
    kadr = np.full((vysota, shirina, 3), 200, dtype=np.uint8)
    x1, y1, x2, y2 = LITSO_RODNOE
    kadr[y1:y2, x1:x2] = 20
    put = tmp_path / imja
    Image.fromarray(kadr).save(put, "JPEG", quality=95)
    return put


def ramka_na_kadre(kadr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Границы зелёной рамки на массиве миниатюры. None — рамки нет."""
    zelenyj = np.array([0, 255, 0])
    maska = (kadr == zelenyj).all(axis=2)
    if not maska.any():
        return None
    y, x = np.where(maska)
    return int(x.min()), int(y.min()), int(x.max()), int(y.max())


def zelennye_tochki(iz) -> list[tuple[int, int]]:
    """Зелёные пиксели QImage. Шаг 2 берёт линию толщиной в 2 px гарантированно.

    Массив из pixmap не вытаскиваем: `toImage()` возвращает Format_RGB32 с выравниванием
    строки, и возня со stride здесь ровным счётом нечего ловить — важны координаты
    рамки, а их даёт обход `pixelColor`.
    """
    tocki: list[tuple[int, int]] = []
    for y in range(0, iz.height(), 2):
        for x in range(0, iz.width(), 2):
            if iz.pixelColor(x, y).getRgb()[:3] == (0, 255, 0):
                tocki.append((x, y))
    return tocki


def test_bystyj_dekod_stavit_ramku_na_to_mesto(tmp_path: Path) -> None:
    """Рамка обязана совпасть с лицом и на снимке, прочитанном для карточки быстрее.

    Координаты лица приходят в пикселях кадра `max_dim` (2400 px), а кадр для карточки
    теперь читается в 1/8 исходника (375 px). Если коэффициент взять от того кадра,
    который реально декодировали, рамка уедет вчетверо и ляжет мимо лица — для родителя
    это хуже отсутствующей рамки: он решит, что нашёлся не тот человек. Тёмный
    прямоугольник на снимке — независимый ориентир, и тест меряет расстояние до него,
    а не «картинка вообще какая-то есть».
    """
    put = narisovat_lico(tmp_path)

    kadr = annotated_thumbnail(put, [RamkaLica(LITSO_V_KADRE_2400, 1)], 2400)
    assert kadr is not None
    ramka = ramka_na_kadre(kadr)
    assert ramka is not None, "зелёной рамки на миниатюре нет вовсе"
    # где на самой миниатюре лежит лицо: длинная сторона 220 px из 3000 px исходника
    pokazatel = 220 / RAZMER_SNIMKA[0]
    ozhidanie = tuple(int(v * pokazatel) for v in LITSO_RODNOE)
    rashozhdeniya = [abs(a - b) for a, b in zip(ramka, ozhidanie)]
    assert max(rashozhdeniya) <= 4, (
        f"рамка уехала на {rashozhdeniya} px: ожидалось {ozhidanie}, получено {ramka}")


def test_melkoe_lico_risuet_ramku_kotoruyu_vidno() -> None:
    """Рамка — указатель, и на мелком лице она обязана оставаться указателем.

    Замер на архиве проекта: на 55 из 147 фото меньшая сторона рамки после пересчёта
    в миниатюру выходила 5…13 px. Человек это читает как «рамки нет» — а подпись под
    сеткой обещает ровно обратное: «что верно, а что нет — решать вам по зелёной рамке
    лица». Обещание, которое интерфейс не может выполнить на трети снимков, хуже
    отсутствия обещания.

    Проверка числами, а не «на глаз по картинке»: считаем границы зелёных пикселей и
    требуем, чтобы меньшая сторона рамки была не меньше MIN_RAMKA_PX.
    """
    kadr = np.zeros((THUMB, THUMB, 3), dtype=np.uint8)
    box = (100.0, 80.0, 104.0, 86.0)          # лицо 4x6 px — ровно тот случай с архива

    ramka = ramka_na_kadre(ramka_na_kadr(kadr, box, 1.0))

    assert ramka is not None, "мелкое лицо осталось без рамки вовсе"
    shirina = ramka[2] - ramka[0] + 1
    vysota = ramka[3] - ramka[1] + 1
    assert min(shirina, vysota) >= MIN_RAMKA_PX, \
        f"рамка {shirina}x{vysota} px — на экране это точка, а не рамка"
    # Растягиваем вокруг ЦЕНТРА лица: рамка показывает, где искать, и съехать в сторону
    # значит соврать про место не меньше, чем про размер.
    tsentr_lica = ((100.0 + 104.0) / 2, (80.0 + 86.0) / 2)
    tsentr_ramki = ((ramka[0] + ramka[2]) / 2, (ramka[1] + ramka[3]) / 2)
    rashozhdenie = max(abs(a - b) for a, b in zip(tsentr_lica, tsentr_ramki))
    assert rashozhdenie <= 2, f"центр рамки уехал на {rashozhdenie} px от центра лица"


def test_krupnoe_lico_ramka_ne_rastyagivaet() -> None:
    """Нижняя граница не имеет права трогать крупные лица: на них рамка — размер лица,
    и раздуть её означало бы врать про то, насколько крупно ребёнок в кадре."""
    kadr = np.zeros((THUMB, THUMB, 3), dtype=np.uint8)
    box = (50.0, 50.0, 150.0, 150.0)

    ramka = ramka_na_kadre(ramka_na_kadr(kadr, box, 1.0))

    assert ramka is not None
    assert max(abs(a - b) for a, b in zip(ramka, (50, 50, 150, 150))) <= 3, \
        f"рамка крупного лица уехала: {ramka}"


def test_melkoe_lico_u_kraja_ne_vylezhaet_za_kadr() -> None:
    """Расширение до минимума у самого края кадра не должно ни падать, ни рисовать
    за массивом: `cv2.rectangle` режет сам, и рамка остаётся частичной — это честно."""
    kadr = np.zeros((THUMB, THUMB, 3), dtype=np.uint8)
    box = (1.0, 1.0, 4.0, 4.0)

    ramka = ramka_na_kadre(ramka_na_kadr(kadr, box, 1.0))

    assert ramka is not None, "лицо у края кадра осталось без рамки"
    assert ramka[0] >= 0 and ramka[1] >= 0, f"рамка вылезла за начало кадра: {ramka}"
    assert ramka[2] < THUMB and ramka[3] < THUMB, f"рамка вылезла за конец кадра: {ramka}"


def test_pervyj_ekran_ne_chitaet_polny_kadr(view, tmp_path: Path, monkeypatch) -> None:
    """Первая отрисовка находки не имеет права декодировать снимки целиком.

    Замер на архиве проекта (131 снимок по 24 МП): `load_photo(path, 2400)` — 403 мс на
    файл, из них 176 мс на уменьшение LANCZOS до 2400 px, которое миниатюре в 220 px не
    нужно вовсе. В потоке интерфейса это 12 секунд стоящего окна на 50 совпадениях и
    48 секунд на 200 — а спецификация, раздел 7, требует «без вида зависшего окна».
    Здесь факт проверяется прямо: полного чтения за первую отрисовку нет вообще, а
    карточка всё равно приходит с рамкой ровно на лице.
    """
    from PySide6.QtWidgets import QLabel

    import ui.results_view as modul

    put = narisovat_lico(tmp_path)
    polnoe_chtenie: list = []

    def shpion(*a, **kw):
        polnoe_chtenie.append(a)
        return None

    monkeypatch.setattr(modul, "load_photo", shpion)
    view.show_rows([ResultRow(put, percent=62, faces=1, box=LITSO_V_KADRE_2400,
                              matched=True)], threshold=0.38)
    assert polnoe_chtenie == [], "миниатюру оплатили полной декодировкой кадра"

    kartinki = [et for et in view.findChildren(QLabel)
                if et.pixmap() is not None and not et.pixmap().isNull()]
    assert len(kartinki) == 1, "карточка без кадра"
    tocki = zelennye_tochki(kartinki[0].pixmap().toImage())
    assert tocki, "на экране зелёной рамки нет"
    # лицо 500x450 px в кадре 2400 -> 46x41 px на миниатюре, левый верхний угол ~73, 55
    for x, y in tocki:
        assert 68 <= x <= 121 and 50 <= y <= 102, f"рамка уехала за лицо: ({x}, {y})"


def test_na_pervom_ekrane_chitaetsya_odin_kadr_a_ostalnye_ocheredju(
        view, tmp_path: Path, monkeypatch, qapp) -> None:
    """`show_rows` на десятке находок читает ОДИН кадр, остальные доходят очередью.

    Кэш базовых миниатюр спасает второй и следующий пересчёт, но не первый: экран, на
    котором человек ждёт результат, собирается из того, что ещё ничего не читано.
    Очередь отдаёт по кадру за тик событий — окно успевает отвечать на клики между
    карточками, а не стоит двадцать секунд подряд.
    """
    from PySide6.QtWidgets import QLabel

    import ui.results_view as modul

    ishodnik = narisovat_lico(tmp_path)
    schitannye: list = []
    nastoyashhij = modul._kadr_miniatury

    def schetchik(path, *a, **kw):
        schitannye.append(Path(path))
        return nastoyashhij(path, *a, **kw)

    monkeypatch.setattr(modul, "_kadr_miniatury", schetchik)
    # Очередь читает только то, что видно (см. `_zapatit_ochered`), поэтому окно показано:
    # скрытая сетка миниатюры догоняет в `showEvent`, а не в пустоту.
    view.show()
    sroki = []
    for i in range(12):
        p = tmp_path / f"kadr{i}.jpg"
        p.write_bytes(ishodnik.read_bytes())
        sroki.append(ResultRow(p, percent=60, faces=1, box=LITSO_V_KADRE_2400,
                              matched=True))

    view.show_rows(sroki, threshold=0.38)
    assert len(schitannye) == 1, f"первый экран прочитал кадров: {len(schitannye)}, ждём один"
    assert len(view._ochered) == 11

    zdat_kadry(qapp, view)
    assert list(view._ochered) == [] and len(schitannye) == 12
    kartinki = [et for et in view.findChildren(QLabel)
                if et.pixmap() is not None and not et.pixmap().isNull()]
    assert len(kartinki) == 12, "очередь обязана довести все карточки до кадра"
    # каждая карточка очереди — с рамкой ровно на лице: «не виснет» не имеет права
    # означать «а, ладно, рамка куда-нибудь съедет»
    for et in kartinki:
        tocki = zelennye_tochki(et.pixmap().toImage())
        assert tocki, "у карточки из очереди нет рамки"
        for x, y in tocki:
            assert 68 <= x <= 121 and 50 <= y <= 102, f"рамка уехала за лицо: ({x}, {y})"
    # пересчёт после этого уже ничего не читает: кадр лежит в кэше
    view.show_rows(sroki, threshold=0.40)
    assert len(schitannye) == 12, "пересчёт обязан брать кадры из кэша"


# --- слова, которых в интерфейсе быть не может -----------------------------------------


def vse_teksty(vidzhet: "QWidget") -> list[str]:
    """Все тексты виджета, которые человек может увидеть: надписи, кнопки, пункты
    списков, всплывающие подсказки и доступные имена."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QComboBox

    teksty = [vidzhet.windowTitle(), vidzhet.toolTip(), vidzhet.accessibleName()]
    for docha in vidzhet.findChildren(object):
        zavidnye_imena = ("text", "toolTip", "placeholderText", "accessibleName",
                          "statusTip", "whatsThis")
        for imja in zavidnye_imena:
            poluchit = getattr(docha, imja, None)
            if poluchit is None:
                continue
            try:
                znachenie = poluchit()
            except TypeError:                    # метод с обязательным аргументом
                continue
            if isinstance(znachenie, str):
                teksty.append(znachenie)
        if isinstance(docha, QComboBox):
            for i in range(docha.count()):
                teksty.append(docha.itemText(i))
                for rol in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
                    znachenie = docha.itemData(i, rol)
                    if isinstance(znachenie, str):
                        teksty.append(znachenie)
    return [t for t in teksty if t]


def proveryu_na_zaprety(teksty: list[str]) -> None:
    assert len(teksty) > 4, f"нечего проверять, нашли только {teksty}"
    for tekst in teksty:
        niz = tekst.lower()
        for stem in STEM:
            assert stem not in niz, f"«{stem}» в тексте: {tekst!r}"
        for vzorec in CELYE_SLOVA:
            assert not re.search(vzorec, niz), f"запрещённое слово в тексте: {tekst!r}"


def test_v_pustoj_setke_net_zapreshhennyh_slov(view) -> None:
    proveryu_na_zaprety(vse_teksty(view))


def test_v_zapolnjennoj_setke_net_zapreshhennyh_slov(view) -> None:
    """Подписи кучек живут только в заполненном видежете: пустую сетку проверять мало."""
    view.show_rows(sroki_zhanra(), threshold=0.38)
    view.toggle_weak.click()
    imena = vse_teksty(view) + [view.summary.text(), view.toggle_weak.text()]
    proveryu_na_zaprety(imena)


def test_setka_obyasnyaet_chto_znachaet_chislo(view) -> None:
    """Родитель, увидевший 45 %, обязан понять, что это удача, а не промах.

    Измеренный разрос настоящих совпадений — 39…67 % (спецификация, раздел 2), и без
    этой строки 45 % читается как «приложение почти ничего не нашло». Строка обязана
    называть ЧИСЛО ПОЛЗУНКА: прежняя версия зашивала 45 % в текст константы и врала
    человеку с ползунком на 38 % (живой скриншот, задача T2).
    """
    view.show_rows(sroki_zhanra(), threshold=0.45)
    pojasnenie = view.hint.text()
    assert "39" in pojasnenie and "67" in pojasnenie, "нет измеренного диапазона"
    assert "45" in pojasnenie, "нет числа, на котором стоит ползунок"
    niz = pojasnenie.lower()
    for opisanie in ("похож", "не промах", "решать"):
        assert opisanie in niz, f"в пояснении нет слова «{opisanie}»"


# --- живое число в пояснении ------------------------------------------------------------
#
# Пояснение под сеткой единственное в интерфейсе, где повторяется число настройки. Пока
# оно было константой, окно на скриншоте человека утверждало «так что 45 % — не промах
# поиска», а ползунок стоял на 38 %. Число, зашитое в текст, — обещание, которое экран
# не держит ни при одной настройке, поэтому подпись обязана пересчитываться на каждый
# `show_rows` (окно зовёт его на каждое движение ползунка из `rebuild_scores`).


@pytest.mark.parametrize("naporog, procent", [(0.38, 38), (0.45, 45), (0.52, 52),
                                              (0.56, 56)])
def test_pojasnenie_nazyvaet_tot_porog_kotoryj_stoit_na_polsunke(view, naporog,
                                                                 procent) -> None:
    """Число в пояснении = то, на чём стоит ползунок, округлённое ровно как на карточке.

    0.56 в списке нарочно: `round(0.56 * 100)` — это 56, а голое `0.56 * 100` даёт
    56.00000000000001. Подпись обязана называть то же число, что печатают карточка и
    report.txt, иначе человек видит два разных объяснения одного и того же делёжа.
    """
    view.show_rows(sroki_zhanra(), threshold=naporog)
    tekst = view.hint.text()
    assert re.search(rf"от {procent} %", tekst), \
        f"в пояснении нет текущего числа {procent}: {tekst}"


def test_chislo_v_pojasnenii_menjaetsja_vmeste_s_polsunkom(view) -> None:
    """Два положения ползунка — два разных текста. Ровно это и сломано в константе.

    Отдельно проверяем, что прежняя цифра 45 не остаётся в подписи при 38 %: иначе
    «живая» строка могла бы просто допечатывать новое число к старому.
    """
    view.show_rows(sroki_zhanra(), threshold=0.38)
    pervoe = view.hint.text()
    view.show_rows(sroki_zhanra(), threshold=0.45)
    vtoroe = view.hint.text()
    assert pervoe != vtoroe, "подпись не меняется вместе с ползунком"
    assert "38" in pervoe and "45" in vtoroe
    assert "45" not in pervoe, f"в подписи при 38 % осталось зашитое 45: {pervoe}"


def test_pojasnenie_pri_raznyh_porogah_bez_zapreshhennyh_slov(view) -> None:
    """Свой обход словаря спецификации: обход в `test_ui_settings` смотрит только на
    `SettingsBar`, а пояснение живёт в `ResultsView` и меняется на глазах.

    Проверяем на двух порогах, чтобы под запрет попали и шаблон, и подставленное в него
    число.
    """
    for naporog in (0.38, 0.52):
        view.show_rows(sroki_zhanra(), threshold=naporog)
        view.toggle_weak.click()
        teksti = vse_teksty(view) + [view.summary.text()]
        assert any("похоже" in t for t in teksti), "пояснение пропало из сетки"
        proveryu_na_zaprety(teksti)


def test_chuzhoj_porog_ne_ronyaet_i_ne_vret_pojasnenie(view) -> None:
    """Защита на будущее: подпись строится из числа, которое прислало окно.

    `porog` приезжает в `show_rows` извне и однажды может приехать не числом. Падать
    сетка не обязана, но и врать про «от 0 %» хуже падения: при чужом значении подпись
    остаётся прежней.
    """
    from ui.results_view import _procent_pojasnenija

    assert _procent_pojasnenija(0.38) == 38
    assert _procent_pojasnenija(0.56) == 56, "округление не по правилу карточки"
    for chuzhoe in (None, "не число", float("nan"), float("inf")):
        assert _procent_pojasnenija(chuzhoe) is None, f"{chuzhoe!r} дало не None"

    view.show_rows(sroki_zhanra(), threshold=0.38)
    bylo = view.hint.text()
    view._threshold = None
    view._obnovit_pojasnenie()
    assert view.hint.text() == bylo, "подпись перечеркнулась на чужом значении"


def test_podpis_kuchki_govorit_chto_dast_najim(view) -> None:
    """Спецификация, раздел 6: «ещё 7 фото, похоже слабее — нажмите, чтобы посмотреть»."""
    view.show_rows(sroki_zhanra(), threshold=0.38)
    assert "ещё 2 фото" in view.toggle_weak.text()
    assert "нажмите" in view.toggle_weak.text()
    view.toggle_weak.click()
    assert "скрыть" in view.toggle_weak.text()


def test_pustaja_kuchka_ne_zvaet_nachat_po_pustomu(view) -> None:
    """Кнопка «показать слабое сходство» без слабого сходства — обещание, за которым
    ничего нет: её не нажимают."""
    view.show_rows(sroki_zhanra([62, 60]), threshold=0.38)
    assert view.toggle_weak.isEnabled() is False
    assert "0" in view.summary.text()


# --- двойной клик и отметка извне -------------------------------------------------------

from PySide6.QtCore import Qt                             # noqa: E402
from PySide6.QtTest import QTest                          # noqa: E402


def klik_dva_raza(po: "QWidget") -> None:
    """Двойной клик НАСТОЯЩЕЙ мышью по центру виджета.

    Урок T3: записать состояние в модель — не то же самое, что кликнуть. Здесь проверяется
    именно путь мыши от карточки до сигнала.
    """
    QTest.mouseDClick(po, Qt.MouseButton.LeftButton,
                      Qt.KeyboardModifier.NoModifier, po.rect().center())


def test_dvojnyj_klik_po_kartochke_prosit_otkryt_foto(view) -> None:
    """Двойной клик по карточке — просьба открыть фото целиком, а не молчание."""
    view.show_rows(sroki_zhanra(), threshold=0.38)
    prosby: list = []
    view.prosyat_otkryt.connect(lambda put: prosby.append(put))

    klik_dva_raza(view._kletki[Path("/ф0.jpg")])
    assert prosby == [Path("/ф0.jpg")]


def test_dvojnyj_klik_po_samomu_snimku_toze_otkryvaet(view) -> None:
    """Человек дважды бьёт по фотографии, а не по серому полю вокруг неё. Метка снимка
    обязана отдавать событие карточке, а не проглатывать его."""
    view.show_rows(sroki_zhanra(), threshold=0.38)
    prosby: list = []
    view.prosyat_otkryt.connect(lambda put: prosby.append(put))

    klik_dva_raza(view._kartinki[Path("/ф1.jpg")])
    assert prosby == [Path("/ф1.jpg")]


def test_odinarnyj_klik_foto_ne_otkryvaet(view) -> None:
    """Одиночный клик — это ещё не просьба: он же не означает «покажи крупно». Двойной
    клик обязан быть единственным жестом открытия."""
    view.show_rows(sroki_zhanra(), threshold=0.38)
    prosby: list = []
    view.prosyat_otkryt.connect(lambda put: prosby.append(put))

    QTest.mouseClick(view._kletki[Path("/ф0.jpg")], Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier,
                     view._kletki[Path("/ф0.jpg")].rect().center())
    assert prosby == []


def test_otmetka_snaruji_vidna_v_setke_i_dovedena_do_kopira(view) -> None:
    """Отметить карточку может не только её собственная галочка: окно просмотра просит
    то же самое, и сетка обязана показать это и пересчитать список на копирование.

    Без синхронизации в обе стороны «копировать это фото» в окне и галочка в списке
    были бы двумя независимыми ответами на один вопрос.
    """
    sroki = sroki_zhanra()
    view.show_rows(sroki, threshold=0.38)
    put = Path("/ф2.jpg")
    ishodnoe = view.otmecheno(put)

    otrazheniya: list = []
    view.selection_changed.connect(lambda spiski: otrazheniya.append(list(view.checked)))

    view.otmetit_snaruji(put, not ishodnoe)

    assert view.otmecheno(put) is (not ishodnoe)
    galochka = dict(view._boxes)[put]
    assert galochka.isChecked() is (not ishodnoe), "галочка на экране осталась прежней"
    assert (put in view.checked) is (not ishodnoe)
    assert otrazheniya, "сетка не сообщила, что список копирования изменился"


def test_otmetit_snaruji_ne_shumit_pri_pustosmene(view) -> None:
    """Повторная установка того же состояния не имеет права высылать `selection_changed`:
    окно просмотра при открытии синхронизируется всегда, и каждое такое открытие
    добавляло бы ложное «изменилось»."""
    view.show_rows(sroki_zhanra(), threshold=0.38)
    put = Path("/ф0.jpg")
    bylo = view.otmecheno(put)
    otrazheniya: list = []
    view.selection_changed.connect(lambda _: otrazheniya.append(1))

    view.otmetit_snaruji(put, bylo)
    assert otrazheniya == []


# --- вторая строка карточки: по какому выбранному лицу нашёлся снимок (T15) --------------
#
# Человек отмечает не одно лицо ребёнка, а несколько примеров — среди них может стоять
# брат или взрослый. Одно число «45 %» не отвечает на вопрос «кто на этом снимке»:
# ребёнок, похожий неуверенно, или совсем другой человек. Отвечает подпись под
# карточкой, и она обязана быть и на главном лице тоже: отсутствующую подпись человек
# читает как «приложение не знает», а не как «сработало главное лицо» — тот же класс
# дефекта, что рамка, которой не видно.


IMENA_LJUDJ = ["ребёнок", "мама", "бабушка"]

# Подпись под карточкой отвечает на вопрос «КТО из искомых людей на этом снимке».
# Прежняя («похоже на лицо 2») называла отметку человека, а не человека: с двумя
# искомыми она не говорила ничего полезного, а решение об уборке снимка принимается
# именно по тому, чьё это фото.


def sroki_s_ljudmi(percents: list[int], nomera: list[int]) -> list[ResultRow]:
    """Карточки, в каждой из которых совпал свой человек.

    Номера задаются отдельно от процентов: подпись обязана зависеть от того, КТО совпал,
    а не от числа на карточке, и одним и тем же вектором входных данных это не проверить.
    """
    return [ResultRow(Path(f"/ф{i}.jpg"), percent=p, faces=1,
                      box=(0.0, 0.0, 10.0, 15.0), matched=p >= 38, chelovek=j,
                      sovpadeniya=(Sovpadenie(j, 1, p),) if j else ())
            for i, (p, j) in enumerate(zip(percents, nomera))]


def vtorye_stroki(vidzhet: "QWidget") -> list[str]:
    """Видимые подписи людей, собранные с экрана, а не из модели виджета."""
    from PySide6.QtWidgets import QLabel

    return sorted(m.text() for m in vidzhet.findChildren(QLabel)
                  if any(imja in m.text() for imja in IMENA_LJUDJ))


def test_vtora_stroka_est_u_kazhdoj_nahodki(view) -> None:
    """Подпись есть у каждой находки, и в ней ровно тот, кто совпал.

    Разные люди — разные подписи: если виджет начнёт брать номер не из строки
    результата, а из порядка карточек, ассерт поймает это на первой же паре.
    """
    view.show_rows(sroki_s_ljudmi([70, 45], [1, 2]), threshold=0.38,
                   imena_ljudj=IMENA_LJUDJ)
    assert vtorye_stroki(view) == ["мама 45 %", "ребёнок 70 %"]
    assert {put: m.text() for put, m in view._podpisi.items()} == {
        Path("/ф0.jpg"): "ребёнок 70 %", Path("/ф1.jpg"): "мама 45 %"}


def test_dva_cheloveka_na_odnom_snimke_pisutsya_obojane(view) -> None:
    """Кадр, где нашли и ребёнка, и маму, подписан обоими — с числом каждого.

    Один общий максимум прятал второго: снимок выглядел найденным «на 100 %», и было
    не понять, чьё это сто.
    """
    row = ResultRow(Path("/oba.jpg"), percent=100, faces=2, box=(0.0, 0.0, 10.0, 15.0),
                    matched=True, chelovek=1,
                    vse_lica=((0., 0., 10., 15.), (20., 20., 30., 35.)), lice=1,
                    sovpadeniya=(Sovpadenie(1, 1, 100), Sovpadenie(2, 2, 61)))
    view.show_rows([row], threshold=0.38, imena_ljudj=IMENA_LJUDJ)
    assert view._podpisi[Path("/oba.jpg")].text() == "ребёнок 100 % · мама 61 %"


@pytest.mark.parametrize("nomer, imya", [(1, "ребёнок"), (2, "мама"), (3, "бабушка")])
def test_podpis_sootvetstvuet_peredannomu_cheloveku(view, nomer, imya) -> None:
    """Текст карточки называет РОВНО того человека, который дал максимум.

    Подпись «мама» там, где нашёлся ребёнок, — это не неточность формулировки, а
    человек, убирающий из папки результатов верный снимок.
    """
    view.show_rows(sroki_s_ljudmi([62], [nomer]), threshold=0.38, imena_ljudj=IMENA_LJUDJ)
    assert view._podpisi[Path("/ф0.jpg")].text() == f"{imya} 62 %"


def test_pri_odnom_cheloveke_vtoroj_stroki_net(view) -> None:
    """Один искомый человек — второй строки нет: имя под числом повторяет заголовок шага.

    Со списком из одного имени карточка выглядит ровно как до задачи T20, а без имён —
    тем более.
    """
    view.show_rows(sroki_s_ljudmi([70, 45], [1, 1]), threshold=0.38)
    assert view._podpisi == {}

    view.show_rows(sroki_s_ljudmi([70], [1]), threshold=0.38, imena_ljudj=["ребёнок"])
    assert view._podpisi == {}


def test_neizvestnyj_chelovek_ne_dajet_podpisi(view) -> None:
    """Номер человека вне списка и снимок без совпадений остаются без второй строки.

    Выдуманное имя здесь хуже молчания: карточка с 0 % и «мама 40 %» убеждает человека,
    что приложение кого-то нашло.
    """
    view.show_rows(sroki_s_ljudmi([50, 45, 40], [0, 9, 3]), threshold=0.38,
                   imena_ljudj=IMENA_LJUDJ)
    assert {put: m.text() for put, m in view._podpisi.items()} == {
        Path("/ф2.jpg"): "бабушка 40 %"}
    assert len(vse_galochki(view)) == 3, "карточки из-за подписи не должны исчезать"


def test_klik_po_knopke_kuchki_ne_terjaet_vtoruju_stroku(view, qapp) -> None:
    """Перестройка сетки обязана принести подписи с собой (урок T3 — проверка мышью).

    Разворот слабой кучки, «отметить все» и «показать ещё» возвращаются в `show_rows`
    вторично, уже изнутри виджета. Если внутренний вызов забудет передать имена,
    человек увидит подписи ровно до первого нажатия — и решит, что приложение их
    потеряло.
    """
    view.show_rows(sroki_s_ljudmi([70, 30], [1, 2]), threshold=0.38,
                   imena_ljudj=IMENA_LJUDJ)
    assert vtorye_stroki(view) == ["ребёнок 70 %"], "до нажатия видно не то"

    QTest.mouseClick(view.toggle_weak, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, view.toggle_weak.rect().center())
    assert vtorye_stroki(view) == ["мама 30 %", "ребёнок 70 %"]

    QTest.mouseClick(view.check_all_btn, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, view.check_all_btn.rect().center())
    assert vtorye_stroki(view) == ["мама 30 %", "ребёнок 70 %"], \
        "массовая отметка стёрла подписи"


def klik_po_galochke(galochka: "QWidget") -> None:
    """Настоящий клик ПО САМОМУ квадратику, а не в центр галочки.

    Урок R-9: `QCheckBox::hitButton` — это прямоугольник индикатора, и нажатие в текст
    ничего не переключает. Проверить это «кликом» в центре значит получить зелёный тест
    о несуществующем жесте.
    """
    from PySide6.QtCore import QPoint                  # noqa: E402

    QTest.mouseClick(galochka, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     QPoint(4, galochka.height() // 2))


def test_klik_po_galochke_sohranyaet_nazvannogo_cheloveka(view) -> None:
    """Отметка карточки не имеет права переписать, КТО на ней найден: отметка решает
    вопрос «копировать или нет», а подпись — вопрос «чьё это фото».
    """
    view.show_rows(sroki_s_ljudmi([70, 45], [1, 2]), threshold=0.38,
                   imena_ljudj=IMENA_LJUDJ)
    galochka = dict(view._boxes)[Path("/ф1.jpg")]
    assert galochka.isChecked() is True, \
        "стартовое состояние не то — тест бы ничего не проверил"

    klik_po_galochke(galochka)

    assert galochka.isChecked() is False
    assert view._podpisi[Path("/ф1.jpg")].text() == "мама 45 %"
    assert "мама 45 %" in galochka.toolTip(), galochka.toolTip()


def test_v_setke_s_podpisami_net_zapreshhennyh_slov(view) -> None:
    """Имена людей — новые тексты на экране, и запрещённый словарь спецификации
    обходится именно по заполненной сетке: пустая их не показывает."""
    view.show_rows(sroki_s_ljudmi([70, 45], [1, 2]), threshold=0.38,
                   imena_ljudj=IMENA_LJUDJ)
    proveryu_na_zaprety(vse_teksty(view) + [view.summary.text(), view.toggle_weak.text()])
