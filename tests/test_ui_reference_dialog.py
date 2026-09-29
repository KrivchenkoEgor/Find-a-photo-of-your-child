"""Окно эталона: человек приносит несколько фото и отмечает, какие лица искать.

`ui.reference_dialog` только рисует, а считает в `core.etalon`. Тесты поэтому следят за
четырьмя вещами, которые рвутся именно на этом шве:

1. **Ни один файл не имеет права пропасть.** Добавил двадцать — на экране двадцать
   карточек или двадцать честных заглушек: «лицо не найдено», «фото не разобралось»,
   «разбор остановлен». Молча исчезнувший снимок неотличим от «приложение не нашло».
2. **Отказ назван, и он единственный.** С задачи T14 в поиске участвует ЛЮБОЕ отмеченное
   лицо: галочка человека — единственное решение, а число сходства только предупреждает
   («тоже ищем · похоже слабо»). Слова «не ищем» остаются за одним случаем — лицом
   нечем считать; он получает «сравнить нечем», но НЕ получает «0 %»: прочесть такое
   число как «нисколько не похоже» значит отправить человека искать другое фото там,
   где дело в качестве снимка.
3. **Ответ не зависит от мыши.** Anchor — самое крупное годное из отмеченных: порядок
   кликов, порядок вставки отметок и перетаскивание карточек в сетке не меняют эталон.
4. **Поток внутри модального окна.** Закрытие крестиком или «Отменой» посреди разбора
   не роняет приложение: поток остановлен и дождался, оглавление под живым потоком не
   закрывается (база принадлежит главному окну, а не диалогу).

Разбор подменён фиктивным потоком по образцу `tests/test_ui_main_window.py`: настоящий
`ScanWorker` — это `QThread`, и одно его создание уже означало бы, что тест полез в
модели. Снимки рисуются в `tmp_path` (белые квадраты по координатам лиц), `data/` и
`ref/` не открываются: там фотографии конкретного ребёнка.

Нумерация здесь та, что видит человек: карточка лица номерует с единицы
(`Kartochka.nomer`), заглушка файла без лиц и неразобравшегося файла — номер 0. Отметки
внутри диалога живут индексами лица в списке файла (с нуля): их выдаёт
`core.etalon.odinochnye_otmetki`.
"""

from __future__ import annotations

import ast
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PIL import Image                                        # noqa: E402
from PySide6.QtCore import QObject, QPoint, Qt, Signal        # noqa: E402

from core.engine import Face                                 # noqa: E402
from core.worker import ScanStats                            # noqa: E402

# Стем, а не слово: ловим и «порог», и «порога», и «детектора». Список — объединение двух
# существующих в проекте запретов (здесь и в `tests/test_ui_face_picker.py`).
STEM = ("порог", "уверенност", "embedd", "косинус", "детектор", "возраст")
CELYE_SLOVA = (r"\bпол(а|у|ом|е)?\b", r"\bpx\b", r"\bпиксел")

# Сколько крутить события: кадры для карточек доходят очередью по одному за 15 мс.
OZHDANIE_MS = 900

# Семь имён для теста «добавил двадцать, семь не читается».
PLOHIE_IMENA = ("c.jpg", "d.jpg", "e.jpg", "f.jpg", "g.jpg", "h.jpg", "i.jpg")


def srazu(qapp, ms: int = OZHDANIE_MS) -> None:
    """Дать очереди миниатюр отработать, крутя события, а не спать молча."""
    konec = time.monotonic() + ms / 1000.0
    while time.monotonic() < konec:
        qapp.processEvents()
        time.sleep(0.005)


def pokazat(dlg, qapp, ms: int = OZHDANIE_MS) -> None:
    """Показать окно и дождаться очереди кадров: таймер читает только видимому окну."""
    dlg.show()
    srazu(qapp, ms)


# --- синтетические лица и снимки ------------------------------------------------------


def otpechatok(bliz: float = 1.0) -> np.ndarray:
    """Единичный вектор с косинусом `bliz` к первому базисному.

    Сходство задаётся числом, а не «на глаз по вектору»: тест границы обязан знать, что
    легло ровно на неё, а что рядом.
    """
    v = np.zeros(512, dtype=np.float32)
    v[0] = bliz
    v[1] = float(np.sqrt(max(0.0, 1.0 - bliz * bliz)))
    return v


def lico(storona: float = 60.0, bliz: float = 1.0,
         smeshchenie: tuple[float, float] = (0.0, 0.0)) -> Face:
    """Лицо со стороной рамки `storona` и сходством `bliz` с первым лицом снимка."""
    lev, verh = smeshchenie
    return Face(box=(lev, verh, lev + storona, verh + storona), landmarks=None,
                embedding=otpechatok(bliz), detector="insight")


def chuzhoe_lico(storona: float = 80.0,
                 smeshchenie: tuple[float, float] = (0.0, 100.0)) -> Face:
    """Лицо, не похожее НИ НА ОДНО другое: вектор вдоль третьей оси.

    Двумерная параметризация `otpechatok` не даёт вектор, ортогональный сразу двум
    другим, а проверять нужно именно «не похоже на остальных» — предупреждение про
    чужое лицо, затесавшееся в отмеченный набор.
    """
    v = np.zeros(512, dtype=np.float32)
    v[2] = 1.0
    lev, verh = smeshchenie
    return Face(box=(lev, verh, lev + storona, verh + storona), landmarks=None,
                embedding=v, detector="insight")


def slomannoe_lico(storona: float = 60.0,
                   smeshchenie: tuple[float, float] = (0.0, 0.0)) -> Face:
    """Лицо с непригодным отпечатком (одни нули): сравнить его нечем.

    `core.etalon` отдаёт такое как `procent=0, v_poisk=False, slabo=False`, и виджет
    обязан прочитать это как «сравнить нечем · не ищем», а не как «похоже на 0 %» и не
    как «похоже слабо»: первое говорит о качестве снимка, второе о сходстве, и это
    разные решения человека.
    """
    lev, verh = smeshchenie
    return Face(box=(lev, verh, lev + storona, verh + storona), landmarks=None,
                embedding=np.zeros(512, dtype=np.float32), detector="insight")


def snimok(tmp_path: Path, imya: str, lica: Sequence[Face], storona: int = 300) -> Path:
    """Рисует JPEG 300x300 с белым квадратом в каждой рамке лица.

    Кадр настоящий, потому что карточки диалог берёт с диска (`load_photo` + `crops_from`):
    поддельный файл означал бы, что путь отрисовки карточки не проверен ничем.
    """
    put = tmp_path / imya
    kadr = Image.new("RGB", (storona, storona), (10, 10, 10))
    for lic in lica:
        x1, y1, x2, y2 = (int(round(v)) for v in lic.box)
        for x in range(max(0, x1), min(storona, x2)):
            for y in range(max(0, y1), min(storona, y2)):
                kadr.putpixel((x, y), (255, 255, 255))
    kadr.save(put, "JPEG")
    return put


def gniloj_fajl(tmp_path: Path, imya: str = "a.heic") -> Path:
    """Файл с расширением фото и мусором внутри: `load_photo` ответит None."""
    put = tmp_path / imya
    put.write_bytes(b"eto ne jpeg sovsem")
    return put


# --- фиктивный рабочий поток ----------------------------------------------------------


class FalsivyjPotok(QObject):
    """Замена `ScanWorker` внутри диалога: фиксирует сам факт создания и не трогает модели.

    Сигналы настоящие (класс — `QObject`), иначе тест шёл бы по пути «вызвали слот
    руками», и потеря строки `worker.done.connect(...)` осталась бы незамеченной.
    """

    sozdannye: list = []

    progress = Signal(int, int, str)
    done = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, files, engine, cache, max_dim, parent=None) -> None:
        super().__init__(parent)
        FalsivyjPotok.sozdannye.append(self)
        self.files, self.engine, self.cache, self.max_dim = (
            list(files), engine, cache, max_dim)
        self.stop_zyvali = 0
        self.zanimaet = False            # ответ `isRunning`
        self.otsvet_na_wait = True       # ответ `wait`: поток встал за отпущенное время
        self.prosili_wait: list[int] = []

    def start(self) -> None:
        pass

    def isRunning(self) -> bool:                                     # noqa: N802 (имя Qt)
        return self.zanimaet

    def request_stop(self) -> None:
        self.stop_zyvali += 1

    def wait(self, ms: int) -> bool:
        self.prosili_wait.append(ms)
        return self.otsvet_na_wait


class GluhojCache:
    """Оглавление-заглушка: диалог обязан отдать его потоку и НЕ закрывать сам."""

    def __init__(self) -> None:
        self.zakrytij = 0

    def close(self) -> None:
        self.zakrytij += 1


# --- сборка диалога и помощники -------------------------------------------------------


@pytest.fixture
def potoki(monkeypatch) -> type:
    """Никакого настоящего разбора во всех тестах файла."""
    FalsivyjPotok.sozdannye = []
    import ui.reference_dialog as modul
    monkeypatch.setattr(modul, "ScanWorker", FalsivyjPotok)
    return FalsivyjPotok


def novyj_dialog(kachestvo: int = 2400,
                 chuzhie: frozenset = frozenset()):
    from ui.reference_dialog import ReferenceDialog

    return ReferenceDialog(SimpleNamespace(name="glushka", loaded=True), GluhojCache(),
                           kachestvo, "/tmp", chuzhie=chuzhie)


@pytest.fixture
def okno(qapp, potoki):
    """Окно эталона, ещё не открытое: разбор подменён, база — заглушка.

    В конце закрываем: таймер очереди кадров живёт на диалоге, и незакрытое окно
    дочитывало бы файлы прошлого теста внутри следующего — подменный `load_photo`
    следующего теста записывал бы не свои чтения (проверено на деле).
    """
    dlg = novyj_dialog()
    yield dlg
    dlg.close()
    dlg.deleteLater()


def prinesi(dlg, monkeypatch, fajly: Sequence[Path]) -> None:
    """Человек нажал «Добавить фото…» и выделил эти файлы в системном диалоге."""
    import ui.reference_dialog as modul

    monkeypatch.setattr(modul.QFileDialog, "getOpenFileNames",
                        staticmethod(lambda *a, **k: ([str(f) for f in fajly], "")))
    dlg._dobavit_foto()


def gotovo(dlg, photos: dict, statistika: ScanStats | None = None) -> None:
    """Поток прислал ответ тем же путём, что и в живом приложении, — через `emit`."""
    dlg.worker.done.emit((photos, statistika or ScanStats(scanned=len(photos),
                                                           cached=0, seconds=0.5)))


def otsvetit(dlg, put: Path, nomer: int, znachenie: bool = True) -> None:
    """Отметить карточку ЗАПИСЬЮ В МОДЕЛЬ. Это НЕ клик: путь мыши здесь не проходит
    ни на йоту, и ровно поэтому диалог с нерабочей мышью оставался зелёным.
    Настоящий клик проверяет `mysh_po_tela_kartochki`.
    """
    element = dlg._karta(put, nomer)
    assert element is not None, f"карточки {put.name} номер {nomer} нет на экране"
    element.setCheckState(Qt.CheckState.Checked if znachenie else Qt.CheckState.Unchecked)


def otmetit_vse(dlg) -> None:
    """Отметить всё, что можно отметить, — в порядке карточек на экране."""
    for i in range(dlg.spisok.count()):
        element = dlg.spisok.item(i)
        if element.flags() & Qt.ItemFlag.ItemIsUserCheckable:
            element.setCheckState(Qt.CheckState.Checked)


def otpechatok_otveta(itog) -> tuple:
    """Слепок ответа: какие карточки, с какими числами и кто в поиске.

    Сравнение по полям, а не `itog == itog`: внутри `Reference` лежит numpy-матрица, и
    обычное равенство dataclass'ов бросает на ней ValueError.
    """
    return tuple((k.put.name, k.nomer, k.otmecheno, k.procent, k.v_poisk)
                 for k in itog.karty)


# --- сборка и запуск разбора --------------------------------------------------------------


def test_okno_sobiraetsja_bez_razbora_i_ok_vyklyucheno(okno) -> None:
    """Диалог не имеет права начинать разбор до того, как человек что-то принёс: одно
    создание потока здесь — это 290 МБ весов и висящее модальное окно."""
    assert okno.worker is None
    assert FalsivyjPotok.sozdannye == [], "поток создан до первого файла"
    assert okno.ok_button.isEnabled() is False
    assert okno.add_button.isEnabled() is True
    assert okno.itog is None
    assert okno.spisok.count() == 0
    assert "Добавьте фото" in okno.note.text()


def test_pervyj_fail_zapuskaet_tolko_svoj_potok(okno, potoki, monkeypatch,
                                                tmp_path) -> None:
    """Разбор принадлежит диалогу: свой поток, свои файлы, качество разбора — от главного
    окна (иначе оглавление мимо кэша на каждом файле уже разобранного архива)."""
    fajl = snimok(tmp_path, "a.jpg", [lico()])
    prinesi(okno, monkeypatch, [fajl])

    assert len(potoki.sozdannye) == 1
    potok = okno.worker
    assert potok is potoki.sozdannye[0]
    assert potok.files == [fajl]
    assert potok.max_dim == 2400, "качество разбора обязано прийти от окна"
    assert isinstance(potok.cache, GluhojCache), "диалог не имеет права открыть свою базу"
    assert okno.add_button.isEnabled() is False, "вторая партия посреди разбора"
    assert okno.ok_button.isEnabled() is False
    assert okno.stop_button.isEnabled() is True
    assert "разбираем 1" in okno.status_label.text()


def test_polosa_progressa_govorit_skolko_sdelano(okno, potoki, monkeypatch,
                                                 tmp_path) -> None:
    """Спецификация, раздел 6: прогресс в реальном времени — и в окне эталона тоже."""
    fajly = [snimok(tmp_path, imya, [lico()]) for imya in ("a.jpg", "b.jpg")]
    prinesi(okno, monkeypatch, fajly)
    okno.worker.progress.emit(1, 2, "a.jpg")

    assert (okno.progress.minimum(), okno.progress.value(), okno.progress.maximum()) == (
        0, 1, 2)
    assert "1 из 2" in okno.status_label.text()


def test_done_i_error_dovedeny_do_dialoga_cherez_signaly(okno, potoki, monkeypatch,
                                                         tmp_path) -> None:
    """Удаление строк `connect` оставило бы весь файл зелёным, если бы тесты дёргали
    слоты руками. Здесь ответ приходит только сигналом потока."""
    a = snimok(tmp_path, "a.jpg", [lico()])
    prinesi(okno, monkeypatch, [a])
    assert okno.spisok.count() == 0

    okno.worker.done.emit(({a: [lico()]}, ScanStats(scanned=1, cached=0, seconds=0.2)))
    assert okno.spisok.count() == 1
    okno.worker.progress.emit(1, 1, "a.jpg")
    assert okno.progress.value() == 1


# --- сетка карточек ------------------------------------------------------------------


def test_odinocnoe_lico_otmecheno_avtomaticheski_i_ono_zhe_etalon(okno, potoki,
                                                                  monkeypatch,
                                                                  tmp_path) -> None:
    """Файл с одним лицом размечен без человека (спецификация, раздел 6), и он же —
    самое крупное годное из отмеченных, то есть anchor со своим честным 100 %."""
    a = snimok(tmp_path, "a.jpg", [lico(60.0)])
    prinesi(okno, monkeypatch, [a])
    gotovo(okno, {a: [lico(60.0)]})

    assert okno.itog is not None and len(okno.itog.karty) == 1
    karta = okno.itog.karty[0]
    assert (karta.otmecheno, karta.v_poisk, karta.procent) == (True, True, None), \
        "единственному лицу не с чем сравнивать себя, и числа у него быть не должно"
    element = okno._karta(a, 1)
    assert element.checkState() == Qt.CheckState.Checked
    assert "ищем по этому лицу" in element.text()
    assert "%" not in element.text(), element.text()
    assert okno.ok_button.isEnabled() is True


def test_gruppovoe_lico_ne_otmechaetsya_ugadyvat_nelzya(okno, potoki, monkeypatch,
                                                        tmp_path) -> None:
    """Три лица на снимке — не отмечена ни одна карточка: приложение не имеет права
    решать за человека, кто из троих его ребёнок."""
    faces = [lico(60.0), lico(80.0, 0.9, (100.0, 0.0)), lico(40.0, 0.5, (0.0, 100.0))]
    a = snimok(tmp_path, "a.jpg", faces)
    prinesi(okno, monkeypatch, [a])
    gotovo(okno, {a: faces})

    assert [k.nomer for k in okno.itog.karty] == [1, 2, 3]
    assert [k.otmecheno for k in okno.itog.karty] == [False, False, False]
    assert [okno._karta(a, i).checkState() for i in (1, 2, 3)] == (
        [Qt.CheckState.Unchecked] * 3)
    assert okno.ok_button.isEnabled() is False
    assert "Отметьте" in okno.note.text()


def mysh_po_tela_kartochki(okno, element) -> None:
    """Клик НАСТОЯЩЕЙ мышью по телу карточки — туда, где человек видит лицо.

    Единственный способ поймать то, что пропустили все остальные проверки этого файла:
    `otsvetit` пишет состояние прямо в элемент (`setCheckState`), и такой вызов минует
    весь путь мыши. Диалог с отметкой-чекбоксом при `NoSelection` оставался полностью
    зелёным, пока живой человек не протыкал карточки и не получил ничего.
    """
    from PySide6.QtTest import QTest

    prost = okno.spisok.visualItemRect(element)
    assert not prost.isNull(), "карточка не видна на экране, кликать не по чему"
    QTest.mouseClick(okno.spisok.viewport(), Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, prost.center())
    okno.spisok.viewport().repaint()


def test_mysh_po_tela_kartochki_stavit_i_snimayet_otmetku(okno, potoki, monkeypatch,
                                                          tmp_path) -> None:
    """Клик по лицу обязан переключать отметку. Не по крошечному квадратику индикатора
    Qt, а по самой карточке: человек ищет ребёнка, смотрит на лицо и тычет в него.

    Снимок с двумя лицами взят не случайно: автоотметки здесь нет ни на одной карточке,
    и весь ответ зависит именно от жеста мышью. Это проверка живого пути, и заменить её
    вызовом `setCheckState` нельзя — `setCheckState` и есть то, что уже работало.
    """
    faces = [lico(60.0), lico(80.0, 0.9, (100.0, 0.0))]
    a = snimok(tmp_path, "a.jpg", faces)
    prinesi(okno, monkeypatch, [a])
    gotovo(okno, {a: faces})
    assert [okno._karta(a, n).checkState() for n in (1, 2)] == \
        [Qt.CheckState.Unchecked, Qt.CheckState.Unchecked], "лиц два, автоотметки быть не может"

    mysh_po_tela_kartochki(okno, okno._karta(a, 2))
    assert okno._karta(a, 2).checkState() == Qt.CheckState.Checked, \
        "клик по карточке не поставил отметку — человеком окно бесполезно"
    assert okno.itog.anchor.nomer == 2, "крупнейшее стало первой строкой эталона"
    assert okno.itog.karty[1].procent is None, \
        "отмечено одно лицо — сравнивать его не с чем, и числа нет"

    mysh_po_tela_kartochki(okno, okno._karta(a, 1))
    assert okno._karta(a, 1).checkState() == Qt.CheckState.Checked
    assert [k.procent for k in okno.itog.karty] == [90, 90], \
        "отметка есть, а пересчёта нет — экран врал бы про числа"

    mysh_po_tela_kartochki(okno, okno._karta(a, 2))
    assert okno._karta(a, 2).checkState() == Qt.CheckState.Unchecked, \
        "второй клик по отмеченной карточке не снимает отметку"
    assert okno.itog.anchor.nomer == 1, "эталон не переехал на оставшееся лицо"


def test_klik_po_inikatoru_galochki_pereklyuchaet_rovno_raz(okno, potoki, monkeypatch,
                                                            tmp_path) -> None:
    """Клик по самой галочке Qt переключает сам, и наш обработчик не имеет права
    переключить второй раз: иначе нажатие ровно туда, где нарисован признак отметки,
    выглядит как нажатие впустую.
    """
    a = snimok(tmp_path, "a.jpg", [lico()])
    prinesi(okno, monkeypatch, [a])
    gotovo(okno, {a: [lico()]})
    karta = okno._karta(a, 1)
    assert karta.checkState() == Qt.CheckState.Checked       # одиночное лицо

    from PySide6.QtTest import QTest

    prost = okno.spisok.visualItemRect(karta)
    QTest.mouseClick(okno.spisok.viewport(), Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, prost.topLeft() + QPoint(12, 16))
    okno.spisok.viewport().repaint()
    assert karta.checkState() == Qt.CheckState.Unchecked, \
        "клик по галочке переключил дважды — отметка вернулась на место"


def test_zaglushka_bez_lica_na_klik_ne_otvechaet(okno, potoki, monkeypatch,
                                                 tmp_path) -> None:
    """Клик по карточке «лицо не найдено» не имеет права ничего отмечать: галочки на
    ней нет, и отметка без лица обещала бы поиск по пустоте."""
    a = snimok(tmp_path, "a.jpg", [lico()])
    pustaya = snimok(tmp_path, "pustaya.jpg", [])
    prinesi(okno, monkeypatch, [a, pustaya])
    gotovo(okno, {a: [lico()], pustaya: []})

    zaglushka = okno._karta(pustaya, 0)
    assert zaglushka is not None
    mysh_po_tela_kartochki(okno, zaglushka)
    assert zaglushka.checkState() != Qt.CheckState.Checked
    assert okno.itog.anchor.nomer == 1, "клацанье по заглушке сдвинуло эталон"


def test_klik_po_kartochke_otmechaet_i_pereschityvaet(okno, potoki, monkeypatch,
                                                      tmp_path) -> None:
    """Отметка — это галочка на карточке, а не отдельная кнопка: между кликом человека
    и числом на карточке не должно быть второго действия."""
    faces = [lico(60.0), lico(80.0, 0.9, (100.0, 0.0))]
    a = snimok(tmp_path, "a.jpg", faces)
    prinesi(okno, monkeypatch, [a])
    gotovo(okno, {a: faces})

    otsvetit(okno, a, 2)                                  # крупнейшее
    otsvetit(okno, a, 1)                                  # и мелкое, похожее на 90 %

    itog = okno.itog
    assert [k.procent for k in itog.karty] == [90, 90]
    assert itog.anchor.nomer == 2, "anchor — крупнейшее, а не первое по клику"
    assert [k.v_poisk for k in itog.karty] == [True, True]
    assert okno.ok_button.isEnabled() is True


def test_klik_snimaet_otmetku_i_etalon_chistitsja(okno, potoki, monkeypatch,
                                                  tmp_path) -> None:
    a = snimok(tmp_path, "a.jpg", [lico()])
    prinesi(okno, monkeypatch, [a])
    gotovo(okno, {a: [lico()]})
    assert okno.ok_button.isEnabled() is True

    otsvetit(okno, a, 1, znachenie=False)

    assert okno.itog.karty[0].otmecheno is False
    assert okno.itog.reference is None
    assert okno.ok_button.isEnabled() is False, "ОК живо без единого лица в эталоне"
    assert "Отметьте" in okno.note.text()


def test_ploskiy_poryadok_kartochek_raven_poryadku_faylov(okno, potoki, monkeypatch,
                                                          tmp_path) -> None:
    """Карточки идут в том порядке, в котором человек приносил файлы, а не по алфавиту
    и не по размеру лица: он ждёт последний снимок рядом с собой."""
    a = snimok(tmp_path, "a.jpg", [lico(60.0), lico(50.0, 0.8, (100.0, 0.0))])
    b = snimok(tmp_path, "b.jpg", [lico(70.0)])
    prinesi(okno, monkeypatch, [b, a])                     # первыми принёс b
    gotovo(okno, {b: [lico(70.0)], a: [lico(60.0), lico(50.0, 0.8, (100.0, 0.0))]})

    assert [(k.put.name, k.nomer) for k in okno.itog.karty] == [
        ("b.jpg", 1), ("a.jpg", 1), ("a.jpg", 2)]
    assert [okno.spisok.item(i).data(Qt.ItemDataRole.UserRole) for i in range(3)] == [
        (b, 1), (a, 1), (a, 2)]


def test_vtoraja_partija_ne_zatiraet_pervuju(okno, potoki, monkeypatch, tmp_path) -> None:
    """«Добавить ещё фото этого же ребёнка» живёт внутри окна, и вторая партия сетку
    дополняет: прежнее окно принимало ровно один файл за раз."""
    a = snimok(tmp_path, "a.jpg", [lico(60.0)])
    b = snimok(tmp_path, "b.jpg", [lico(70.0)])
    prinesi(okno, monkeypatch, [a])
    gotovo(okno, {a: [lico(60.0)]})
    prinesi(okno, monkeypatch, [b])
    gotovo(okno, {b: [lico(70.0)]})

    assert list(okno.lica_po_fajlam) == [a, b]
    assert [(k.put.name, k.nomer) for k in okno.itog.karty] == [("a.jpg", 1), ("b.jpg", 1)]
    assert {k.otmecheno for k in okno.itog.karty} == {True}
    assert okno.itog.reference.count == 2


def test_snjataja_otmetka_ne_vernetsja_vtoroj_partiej(okno, potoki, monkeypatch,
                                                      tmp_path) -> None:
    """Автоотметка «лицо на снимке одно» касается НОВЫХ файлов. Распространить её на весь
    список значило бы перечеркнуть выбор человека: он снял галочку — она ожила бы."""
    a = snimok(tmp_path, "a.jpg", [lico(60.0)])
    b = snimok(tmp_path, "b.jpg", [lico(70.0)])
    prinesi(okno, monkeypatch, [a])
    gotovo(okno, {a: [lico(60.0)]})
    otsvetit(okno, a, 1, znachenie=False)
    assert okno.itog.karty[0].otmecheno is False

    prinesi(okno, monkeypatch, [b])
    gotovo(okno, {b: [lico(70.0)]})

    po_a = [k for k in okno.itog.karty if k.put.name == "a.jpg"][0]
    po_b = [k for k in okno.itog.karty if k.put.name == "b.jpg"][0]
    assert (po_a.otmecheno, po_b.otmecheno) == (False, True)


def test_povtornoe_priklyuchenie_togo_zhe_fajla_ne_dvoit_kartochku(okno, potoki,
                                                                   monkeypatch,
                                                                   tmp_path) -> None:
    a = snimok(tmp_path, "a.jpg", [lico(60.0)])
    prinesi(okno, monkeypatch, [a])
    gotovo(okno, {a: [lico(60.0)]})
    prinesi(okno, monkeypatch, [a])

    assert okno.spisok.count() == 1
    assert [k.put.name for k in okno.itog.karty] == ["a.jpg"]


# --- честные заглушки -----------------------------------------------------------------


def test_sem_neradobravshihsja_fajlov_dayut_sem_kartochek_a_ne_ischeznut(
        okno, potoki, monkeypatch, tmp_path) -> None:
    """Спецификация, раздел 7: неразобранный файл называется неразобранным. Молча
    исчезнувший снимок человек прочитал бы как «приложение не нашло»."""
    horoshie = [snimok(tmp_path, imya, [lico()]) for imya in ("a.jpg", "b.jpg")]
    gnilye = [gniloj_fajl(tmp_path, imya) for imya in PLOHIE_IMENA]
    prinesi(okno, monkeypatch, horoshie + gnilye)
    gotovo(okno, {put: [lico()] for put in horoshie},
           ScanStats(scanned=2, cached=0, seconds=1.0,
                     failures=[(str(put), "файл не читается: cannot identify")
                               for put in gnilye]))

    assert okno.spisok.count() == 9, "9 файлов принесли — 9 карточек, ни одной потерянной"
    for put in gnilye:
        element = okno._karta(put, 0)
        assert element is not None, f"{put.name} пропал из сетки"
        assert "не разобралось" in element.text()
        assert "лицо не найдено" not in element.text(), "две разные беды названы одинаково"
        assert element.checkState() == Qt.CheckState.Unchecked
        assert not element.flags() & Qt.ItemFlag.ItemIsUserCheckable, "отметить нечего"
    assert "не разобралось: 7" in okno.note.text()


def test_prichina_poteri_doezzhaet_v_podskazku(okno, potoki, monkeypatch, tmp_path) -> None:
    """Гнилой HEIC и обрезанный JPEG — две разные починки, и причина обязана быть
    прочитана, а не спрятана за общим словом."""
    gniloj = gniloj_fajl(tmp_path)
    prinesi(okno, monkeypatch, [gniloj])
    gotovo(okno, {}, ScanStats(failures=[(str(gniloj), "файл не читается: Truncated")]))

    assert "Truncated" in okno._karta(gniloj, 0).toolTip()


def test_fajl_bez_lic_otlichaetsja_ot_neradobravshegosja(okno, potoki, monkeypatch,
                                                         tmp_path) -> None:
    """«Лицо не найдено» — про снимок, который прочли и разобрали. Это ответ поиска,
    а не ошибка файла, и путать их нельзя (спецификация, раздел 7)."""
    chistoe = snimok(tmp_path, "a.jpg", [])
    prinesi(okno, monkeypatch, [chistoe])
    gotovo(okno, {chistoe: []})

    element = okno._karta(chistoe, 0)
    assert "лицо не найдено" in element.text()
    assert "не разобралось" not in element.text()
    assert not element.flags() & Qt.ItemFlag.ItemIsUserCheckable
    assert okno.ok_button.isEnabled() is False


def test_otmenennyj_razbor_ne_nazyvaet_nedosmotrennyj_snimok_ne_razobravsja(
        okno, potoki, monkeypatch, tmp_path) -> None:
    """«Остановить» — не «файл испорчен». Прежнее окно на пустом ответе после отмены
    советовало выбрать другое фото, хотя человек просто перестал ждать."""
    a = snimok(tmp_path, "a.jpg", [lico()])
    b = snimok(tmp_path, "b.jpg", [lico()])
    prinesi(okno, monkeypatch, [a, b])
    okno.stop_button.click()
    gotovo(okno, {a: [lico()]}, ScanStats(scanned=1, cached=0, seconds=0.4))

    hvost = okno._karta(b, 0)
    assert "останов" in hvost.text().lower(), hvost.text()
    assert "не разобралось" not in hvost.text()
    assert "не найдено" not in hvost.text()
    assert okno._karta(a, 1).checkState() == Qt.CheckState.Checked
    assert "остановлен" in okno.status_label.text()


def test_oshibka_potoka_ne_pustoj_otvet_a_nazvannaja_prichina(okno, potoki, monkeypatch,
                                                              tmp_path) -> None:
    """Поток различает «не поднялись веса» и «оглавление встало», и текст показывается
    как есть: собственный заголовок сделал бы из одного другое."""
    pokazanno: list = []
    monkeypatch.setattr(okno, "_warn", lambda text: pokazanno.append(text))
    a = snimok(tmp_path, "a.jpg", [lico()])
    prinesi(okno, monkeypatch, [a])
    okno.worker.error.emit("не удалось загрузить модели: нет файла весов.")

    assert pokazanno == ["не удалось загрузить модели: нет файла весов."]
    assert okno.add_button.isEnabled() is True
    assert okno.ok_button.isEnabled() is False
    assert "не удалось загрузить модели" in okno.status_label.text()
    assert okno.spisok.count() == 1, "файл обязан остаться карточкой"


# --- число на карточке предупреждает, а не отказывает ----------------------------------------


def dochitat_kadry(okno) -> None:
    """Дочитать очередь кадров так же, как её читает таймер.

    Синхронно в тесте читается один файл — тот же бюджет, что и на экране, — и карточка
    второго остаётся серой плиткой. Мерить цвет по такой плитке значит проверять не
    рамку, а очередь.
    """
    while okno._ochered_kadrov:
        okno._chitat_kadr(okno._ochered_kadrov.popleft())
    okno._pererisovat_kropy()


def cvet_po_krayu(okno, put: Path) -> tuple[int, int, int]:
    """Цвет левого края иконки карточки — то, что человек видит глазом."""
    from PySide6.QtCore import QSize

    element = okno._karta(put, 1)
    karta = element.icon().pixmap(QSize(okno.storona_kartochki, okno.storona_kartochki))
    assert not karta.isNull(), "иконки карточки нет вовсе"
    return karta.toImage().pixelColor(1, okno.storona_kartochki // 2).getRgb()[:3]


def test_ramka_na_kartochke_krasim_ishemoe_lico_zelenym(okno, potoki, monkeypatch,
                                                        tmp_path) -> None:
    """Цвет по краю снимка говорит, ищется это лицо или нет.

    Отмечено одно лицо из десяти, и по экрану это не найти: квадратик галочки в сетке
    карточек глазом не обнаруживается, а человек хочет видеть, по кому именно идёт
    поиск, не тыкая мышью. Зелёный край — ищем, красный — нет.
    """
    svoe = snimok(tmp_path, "svoe.jpg", [lico(60.0)])
    chuzhoe = snimok(tmp_path, "chuzhoe.jpg", [lico(60.0)])
    prinesi(okno, monkeypatch, [svoe, chuzhoe])
    gotovo(okno, {svoe: [lico(60.0)], chuzhoe: [lico(60.0)]})

    dochitat_kadry(okno)
    assert cvet_po_krayu(okno, svoe) == (0, 255, 0), "отмеченное лицо не зелёное"
    assert cvet_po_krayu(okno, chuzhoe) == (0, 255, 0), "отмеченное лицо не зелёное"

    mysh_po_tela_kartochki(okno, okno._karta(chuzhoe, 1))     # человек снимает галочку

    assert cvet_po_krayu(okno, chuzhoe) == (255, 0, 0), \
        "отметка снята, а рамка всё ещё говорит «ищем»"
    assert cvet_po_krayu(okno, svoe) == (0, 255, 0), "перекрасилась не та карточка"


def test_snjataya_otmetka_menjaet_ramku_obratno(okno, potoki, monkeypatch,
                                                tmp_path) -> None:
    """Рамка перекрашивается в тот же миг, что и галочка.

    Иначе единственный видимый признак решения остаётся вчерашним: человек снял
    отметку, а карточка всё ещё зелёная.
    """
    a = snimok(tmp_path, "a.jpg", [lico(60.0)])
    prinesi(okno, monkeypatch, [a])
    gotovo(okno, {a: [lico(60.0)]})
    assert cvet_po_krayu(okno, a) == (0, 255, 0), "автоотметка не покрасила рамку"

    mysh_po_tela_kartochki(okno, okno._karta(a, 1))

    assert cvet_po_krayu(okno, a) == (255, 0, 0), "отметку сняли, а рамка осталась"


def test_chuzhoe_lico_ne_poluchaet_avtootmetki(potoki, monkeypatch, tmp_path,
                                               qapp) -> None:
    """Файл, чьё лицо уже ищется другим человеком, открывается неотмеченным.

    Человек принёс для мамы те же 16 снимков, что и для ребёнка, и приложение само
    отметило спящего ребёнка: «при добавлении мамы ты делаешь фото ребёнка главным».
    Автоотметка была заказана для «я приношу фото своего ребёнка», и там, где человек
    уже сказал, кто на снимке, она обязана молчать — но снять отметку руками по-прежнему
    можно, и поставить её на чужое лицо тоже можно.
    """
    svoe = snimok(tmp_path, "svoe.jpg", [lico(60.0)])
    chuzhoe = snimok(tmp_path, "chuzhoe.jpg", [lico(60.0)])

    okno = novyj_dialog(chuzhie=frozenset({(chuzhoe, 0)}))
    try:
        prinesi(okno, monkeypatch, [svoe, chuzhoe])
        gotovo(okno, {svoe: [lico(60.0)], chuzhoe: [lico(60.0)]})

        assert okno._karta(svoe, 1).checkState() == Qt.CheckState.Checked, \
            "свободное одиночное лицо перестало отмечаться само"
        assert okno._karta(chuzhoe, 1).checkState() == Qt.CheckState.Unchecked, \
            "лицо другого человека отмечено без спроса"
    finally:
        okno.close()
        okno.deleteLater()


def test_slaboe_pohozhee_lico_gasnet_no_idet_v_poisk(okno, potoki, monkeypatch,
                                                     tmp_path) -> None:
    """Решение T14 (отменяет R-1) глазами человека: слабое лицо бледнее и подписано
    предупреждением, НО стоит в поиске. Снять галочку за человека — значит спрятать его
    выбор; не взять его в поиск — значит обещать одно, а делать другое.

    Прежняя версия этого теста требовала `[True, True, False]` и слов «в поиск не
    пойдёт» на третьей карточке.
    """
    faces = [lico(100.0), lico(90.0, 0.5, (100.0, 0.0)), chuzhoe_lico(80.0, (0.0, 100.0))]
    a = snimok(tmp_path, "a.jpg", faces)
    prinesi(okno, monkeypatch, [a])
    gotovo(okno, {a: faces})
    otmetit_vse(okno)

    karty = list(okno.itog.karty)
    assert [k.procent for k in karty] == [50, 50, 0]
    assert [k.v_poisk for k in karty] == [True, True, True], "отмеченное лицо не дошло до поиска"
    assert [k.slabo for k in karty] == [False, False, True]
    slabaja = okno._karta(a, 3)
    assert slabaja.checkState() == Qt.CheckState.Checked
    assert "ищем по этому лицу" in slabaja.text()
    assert "не похоже на остальных" in slabaja.text()
    assert "не пойдёт" not in slabaja.text(), slabaja.text()
    assert "0 %" in slabaja.text(), "число обязано быть видно, а не спрятано"
    assert slabaja.data(Qt.ItemDataRole.ForegroundRole) is not None, "не побледнело"
    assert okno._karta(a, 1).data(Qt.ItemDataRole.ForegroundRole) is None
    assert okno._karta(a, 2).data(Qt.ItemDataRole.ForegroundRole) is None, \
        "достойное число не имеет права выглядеть отвергнутым"
    assert okno.itog.reference.count == 3
    assert "лиц в поиске: 3 из 3" in okno.note.text(), okno.note.text()
    assert "не похожи на остальных: 1" in okno.note.text(), okno.note.text()
    assert "не пойдёт" not in okno.note.text(), okno.note.text()
    assert okno.ok_button.isEnabled() is True


def test_kazhdaya_kartochka_v_poiske_podpisana_slovom_a_ne_tolko_chislom(
        okno, potoki, monkeypatch, tmp_path) -> None:
    """Настоящий источник заблуждения «ищет по одному лицу»: подпись «по нему ищем»
    стояла ровно на одной карточке, а остальные восемь показывали голый процент.

    Здесь все четыре рода карточек видны в одной сетке и различаются словами:
    главное, участника с достойным числом, участника со слабым числом и непригодный
    отпечаток. Формулировка «тоже ищем» обязана стоять на КАЖДОЙ участвующей карточке.
    """
    faces = [lico(120.0, 1.0), lico(100.0, 0.7, (150.0, 0.0)),
             chuzhoe_lico(90.0, (0.0, 150.0)), slomannoe_lico(80.0, (150.0, 150.0))]
    a = snimok(tmp_path, "a.jpg", faces)
    prinesi(okno, monkeypatch, [a])
    gotovo(okno, {a: faces})
    otmetit_vse(okno)

    dostoinaja = okno._karta(a, 1).text()
    toze_dostoinaja = okno._karta(a, 2).text()
    slabaja = okno._karta(a, 3).text()
    nechem = okno._karta(a, 4).text()
    assert "ищем по этому лицу" in dostoinaja and "70" in dostoinaja, dostoinaja
    assert "ищем по этому лицу" in toze_dostoinaja and "70" in toze_dostoinaja, toze_dostoinaja
    assert "ищем по этому лицу" in slabaja and "не похоже на остальных" in slabaja, slabaja
    assert "сравнить нечем" in nechem and "не ищем" in nechem, nechem
    assert "0 %" not in nechem.split("\n"), nechem

    vseh_tekstov = [okno._karta(a, i).text() for i in (1, 2, 3, 4)]
    assert not [txt for txt in vseh_tekstov if "главное" in txt or "по нему ищем" in txt], \
        f"на карточках снова появилось «главное» лицо: {vseh_tekstov}"
    assert sum(1 for txt in vseh_tekstov if "ищем по этому лицу" in txt) == 3, \
        f"участников поиска подписали словами не на всех карточках: {vseh_tekstov}"
    assert not [txt for txt in vseh_tekstov if "не пойдёт" in txt], vseh_tekstov

    note = okno.note.text()
    assert "лиц в поиске: 3 из 4 отмеченных" in note, note
    assert "не похожи на остальных: 1" in note, note
    assert "не ищем: 1 — сравнить нечем" in note, note


def test_mysh_otmechennoe_slaboe_lico_edaet_v_poisk(okno, potoki, monkeypatch,
                                                    tmp_path) -> None:
    """Тот же вывод, полученный ЖЕСТОМ, а не записью в модель: человек кликает по мелкому
    лицу на групповом снимке и получает его в поиске.

    Помощник `otsvetit` пишет состояние прямо в элемент и минует весь путь мыши, а
    называть тестом про клик то, где клика не было, этот файл уже проходил (урок R-9).
    """
    faces = [lico(100.0), lico(60.0, 0.2, (100.0, 0.0))]
    a = snimok(tmp_path, "a.jpg", faces)
    prinesi(okno, monkeypatch, [a])
    gotovo(okno, {a: faces})
    assert [okno._karta(a, n).checkState() for n in (1, 2)] == \
        [Qt.CheckState.Unchecked] * 2, "лиц два, автоотметки быть не может"

    mysh_po_tela_kartochki(okno, okno._karta(a, 1))
    mysh_po_tela_kartochki(okno, okno._karta(a, 2))

    karta = okno.itog.karty[1]
    assert (karta.otmecheno, karta.v_poisk, karta.slabo, karta.procent) == (True, True, True, 20)
    assert okno.itog.karty[0].procent == 20, \
        "два непохожих лица предупреждаются оба: ни одно из них не «главное»"
    assert okno.itog.reference.count == 2
    podpis = okno._karta(a, 2).text()
    assert "ищем по этому лицу · не похоже на остальных" in podpis, podpis
    assert "не пойдёт" not in podpis


def test_neprigodnoe_lico_ne_pishet_nulevoj_procent(okno, potoki, monkeypatch,
                                                    tmp_path) -> None:
    """`procent=0` у лица с непригодным отпечатком значит «считать нечем», а не
    «нисколько не похоже». Подпись «0 %» на её месте — обман.

    Это единственный случай, когда отмеченное лицо не участвует: здесь и остаются слова
    отказа, но назвать причину должен не процент, а слова «сравнить нечем».
    """
    faces = [lico(100.0), slomannoe_lico(90.0, (100.0, 100.0))]
    a = snimok(tmp_path, "a.jpg", faces)
    prinesi(okno, monkeypatch, [a])
    gotovo(okno, {a: faces})
    otmetit_vse(okno)

    text = okno._karta(a, 2).text()
    stroki = text.split("\n")
    assert "0 %" not in stroki, stroki
    assert "похоже на 0" not in text
    assert "сравнить нечем" in text
    assert "не ищем" in text
    assert "не похоже на остальных" not in text, \
        "слабое и нечитаемое — два разных разговора"
    assert okno._karta(a, 2).checkState() == Qt.CheckState.Checked
    assert okno.itog.anchor.nomer == 1
    assert okno.itog.reference.count == 1
    assert "не ищем: 1 — сравнить нечем" in okno.note.text(), okno.note.text()


def test_vse_otmechennye_neprigodny_ok_vyklyucheno_i_skazano_pochemu(
        okno, potoki, monkeypatch, tmp_path) -> None:
    """Ни одного лица в эталоне — кнопка поиска мертва, и искать глазами причину
    человек не должен."""
    a = snimok(tmp_path, "a.jpg", [slomannoe_lico(100.0)])
    prinesi(okno, monkeypatch, [a])
    gotovo(okno, {a: [slomannoe_lico(100.0)]})

    assert okno.itog is not None and okno.itog.reference is None
    assert len(okno.itog.otvergnutye) == 1
    assert okno.ok_button.isEnabled() is False
    assert "сравнить нечем" in okno.note.text()
    assert "не ищем" in okno.note.text()
    assert "похоже меньше" not in okno.note.text(), \
        "число сходства больше не причина отказа"


def test_odinocnoe_neprigodnoe_lico_ne_propadaet(okno, potoki, monkeypatch,
                                                 tmp_path) -> None:
    """`odinochnye_otmetki` отмечает и битое одиночное лицо: оно приходит отвергнутым,
    и виджет обязан показать его отвергнутым, а не снять отметку молча."""
    a = snimok(tmp_path, "a.jpg", [slomannoe_lico(100.0)])
    prinesi(okno, monkeypatch, [a])
    gotovo(okno, {a: [slomannoe_lico(100.0)]})

    element = okno._karta(a, 1)
    assert element.checkState() == Qt.CheckState.Checked
    assert "не ищем" in element.text()
    assert "сравнить нечем" in element.text()


def test_podskazka_na_knopke_obeshchaet_vse_otmechennye(okno) -> None:
    """Подсказка на кнопке обязана обещать ровно то, что приложение делает.

    Прежняя фраза «в поиск пойдут отмеченные лица, похожие не меньше чем на 40 %»
    обещала отказ, которого больше нет: человек, отметивший лицо на 20 %, прочитал бы её
    как «его не возьмут» и снял бы свою же галочку. Теперь там объяснение предупреждения.
    """
    tekst = okno.ok_button.toolTip()
    niz = tekst.lower()
    assert "все отмеченные" in niz, tekst
    assert "не похоже на остальных" in niz, tekst
    assert "не меньше чем" not in niz, f"в подсказке вернулось обещание отбора: {tekst}"
    for otkaz in ("не пойд", "не пошло"):
        assert otkaz not in niz, tekst


# --- ответ не зависит от мыши ---------------------------------------------------------------


def test_otveta_ne_zavisit_ot_poryadka_klikov(okno, potoki, monkeypatch, tmp_path) -> None:
    """R-2: один и тот же набор отметок обязан давать один и тот же эталон, куда бы
    человек ни ткнул первым."""
    faces = [lico(60.0), lico(80.0, 0.9, (100.0, 0.0)), lico(40.0, 0.6, (0.0, 100.0))]
    a = snimok(tmp_path, "a.jpg", faces)
    prinesi(okno, monkeypatch, [a])
    gotovo(okno, {a: faces})

    for nomer in (1, 2, 3):
        otsvetit(okno, a, nomer)
    pervyj = otpechatok_otveta(okno.itog)
    anchor = (okno.itog.anchor.put.name, okno.itog.anchor.nomer)

    for nomer in (1, 2, 3):
        otsvetit(okno, a, nomer, znachenie=False)
    for nomer in (3, 2, 1):
        otsvetit(okno, a, nomer)

    assert otpechatok_otveta(okno.itog) == pervyj
    assert (okno.itog.anchor.put.name, okno.itog.anchor.nomer) == anchor


def test_perestanovka_otmetok_mnozhestvom_ne_menjaet_otvet(okno, potoki, monkeypatch,
                                                           tmp_path) -> None:
    """Отметки уходят в `core.etalon` множеством, а обход множества — лотерея: прогоняем
    оба порядка вставки одних и тех же пар."""
    faces = [lico(60.0), lico(60.0, 0.5, (100.0, 0.0))]      # равные размеры — хрупкий случай
    a = snimok(tmp_path, "a.jpg", faces)
    prinesi(okno, monkeypatch, [a])
    gotovo(okno, {a: faces})

    okno.otmetki.clear()
    okno.otmetki.update([(a, 0), (a, 1)])
    odin = otpechatok_otveta(okno._pereschislit())
    okno.otmetki.clear()
    okno.otmetki.update([(a, 1), (a, 0)])
    drugoj = otpechatok_otveta(okno._pereschislit())

    assert odin == drugoj
    assert okno.itog.anchor is okno.itog.karty[0], \
        "при равенстве размеров первой строкой эталона становится первое в плоском порядке"


def test_peretasyvanie_kartochki_myshyu_ne_menjaet_anchor(okno, potoki, monkeypatch,
                                                          tmp_path) -> None:
    """Карточки нельзя двигать, и даже если разметка сдвинулась, ответ считается по
    порядку файлов, а не по порядку строк на экране."""
    faces = [lico(60.0), lico(50.0, 0.8, (100.0, 0.0)), lico(40.0, 0.7, (0.0, 100.0))]
    a = snimok(tmp_path, "a.jpg", faces)
    b = snimok(tmp_path, "b.jpg", [lico(45.0)])
    prinesi(okno, monkeypatch, [a, b])
    gotovo(okno, {a: faces, b: [lico(45.0)]})
    otmetit_vse(okno)
    bylo = (okno.itog.anchor.put.name, okno.itog.anchor.nomer)
    poryadok_do = [okno.spisok.item(i).data(Qt.ItemDataRole.UserRole)
                   for i in range(okno.spisok.count())]

    element = okno.spisok.takeItem(0)
    okno.spisok.insertItem(okno.spisok.count(), element)      # верхнюю карточку — вниз
    okno._pereschislit()

    poryadok_posle = [okno.spisok.item(i).data(Qt.ItemDataRole.UserRole)
                      for i in range(okno.spisok.count())]
    assert poryadok_posle != poryadok_do, "тест не проверил ничего: карточки не сдвинулись"
    assert (okno.itog.anchor.put.name, okno.itog.anchor.nomer) == bylo
    assert otpechatok_otveta(okno.itog)[0][0] == "a.jpg", "порядок файлов поехал за экраном"


def test_kartochki_nelzya_peretaschit(okno) -> None:
    """Сдвинутая мышью карточка путает порядок, а порядок карточек — это соответствие
    карточки её лицу (замер на этой сборке Qt: в IconMode таскать можно по умолчанию)."""
    from PySide6.QtWidgets import QAbstractItemView, QListView

    assert okno.spisok.movement() == QListView.Movement.Static
    assert okno.spisok.dragDropMode() == QAbstractItemView.DragDropMode.NoDragDrop


# --- кропики лиц ----------------------------------------------------------------------


def test_krop_chitaetsja_v_tom_zhe_kachestve_chto_i_razbor(okno, potoki, monkeypatch,
                                                           tmp_path, qapp) -> None:
    """Кадры для карточек берутся с `max_dim` главного окна: рамка лица задана в пикселях
    ИМЕННО этого кадра, а оглавление ключуется парой (файл, качество). Расхождение — это
    кроп не с того лица и промах кэша на каждом файле уже разобранного архива."""
    import ui.reference_dialog as modul

    vyzovy: list = []
    nastoyashhij = modul.load_photo

    def spy(path, max_dim=None, **kwargs):
        vyzovy.append((Path(path), max_dim))
        return nastoyashhij(path, max_dim=max_dim)

    monkeypatch.setattr(modul, "load_photo", spy)
    a = snimok(tmp_path, "a.jpg", [lico(60.0)])
    b = snimok(tmp_path, "b.jpg", [lico(70.0)])
    prinesi(okno, monkeypatch, [a, b])
    gotovo(okno, {a: [lico(60.0)], b: [lico(70.0)]})
    pokazat(okno, qapp)

    assert {p for p, _ in vyzovy} == {a, b}, vyzovy
    assert {d for _, d in vyzovy} == {2400}, vyzovy


def test_kartochki_raznyh_lic_razlichajutsja_bajtami(okno, potoki, monkeypatch, tmp_path,
                                                     qapp) -> None:
    """На снимке, сделанном в темноте, все кропы одинаково чёрные и человек тычет
    вслепую. Номер прямо на кадре — единственный способ сказать «второе»."""
    faces = [lico(60.0), lico(80.0, 0.9, (100.0, 100.0))]
    a = snimok(tmp_path, "a.jpg", faces)
    prinesi(okno, monkeypatch, [a])
    gotovo(okno, {a: faces})
    pokazat(okno, qapp)

    storona = okno.storona_kartochki
    pikseli = [bytes(okno._karta(a, i).icon().pixmap(storona, storona).toImage().constBits())
               for i in (1, 2)]
    assert pikseli[0] != pikseli[1], "карточки неразличимы"
    assert okno._karta(a, 1).icon().pixmap(storona, storona).toImage().height() > 0


def test_nechitaemyj_kadr_ne_ronyaet_setku(okno, potoki, monkeypatch, tmp_path,
                                           qapp) -> None:
    """Файл разобрался (лица лежат в оглавлении), а кадр сегодня не читается: карточка
    обязана остаться на месте со своей отметкой и своим числом — только без фотографии."""
    lica = [lico(60.0)]
    a = gniloj_fajl(tmp_path, "a.jpg")
    prinesi(okno, monkeypatch, [a])
    gotovo(okno, {a: lica})
    pokazat(okno, qapp)

    element = okno._karta(a, 1)
    assert element is not None
    assert element.checkState() == Qt.CheckState.Checked
    assert "0 %" not in element.text().split("\n")
    assert "не читается" in element.toolTip()


# --- кнопки, состояния, закрытие -------------------------------------------------------


def test_knopki_podpisany_po_russki(okno) -> None:
    """Стандартные кнопки Qt отдаёт по-английски: для человека без английского это тупик
    прямо в диалоге."""
    teksty = {okno.add_button.text(), okno.ok_button.text(), okno.cancel_button.text(),
              okno.stop_button.text()}
    assert all(t for t in teksty), teksty
    assert "Отмена" in teksty
    assert any("Добавить фото" in t for t in teksty)
    assert any("разбор" in t.lower() for t in teksty), "кнопки остановки разбора нет"


def test_ok_vyklyucheno_poka_idet_razbor(okno, potoki, monkeypatch, tmp_path) -> None:
    """ОК посреди разбора — это эталон из половины лиц: человек не заметил бы, что
    половина карточек ещё не дошла."""
    a = snimok(tmp_path, "a.jpg", [lico()])
    b = snimok(tmp_path, "b.jpg", [lico()])
    prinesi(okno, monkeypatch, [a, b])

    assert okno.ok_button.isEnabled() is False
    assert "Разбираем" in okno.note.text()

    gotovo(okno, {a: [lico()], b: [lico()]})
    assert okno.ok_button.isEnabled() is True
    assert "Разбираем" not in okno.note.text()


def test_sprosit_posle_otmeny_dayet_nikogo(qapp, monkeypatch, tmp_path, potoki) -> None:
    from PySide6.QtWidgets import QDialog

    from ui.reference_dialog import ReferenceDialog

    monkeypatch.setattr(QDialog, "exec", lambda self: QDialog.DialogCode.Rejected)
    assert ReferenceDialog.sprosit(None, None, GluhojCache(), 2400, str(tmp_path)) is None


def test_sprosit_posle_ok_dayet_tot_zhe_itog_chto_videjen(qapp, monkeypatch, tmp_path,
                                                          potoki) -> None:
    """ОК переносит в главное окно ровно тот `Itog`, который человек видел на экране, —
    без второй сборки эталона на другом основании."""
    from PySide6.QtWidgets import QDialog

    from ui.reference_dialog import ReferenceDialog

    a = snimok(tmp_path, "a.jpg", [lico(60.0)])

    def simuliacija(self):
        self._prinjat_fajly([a])
        self.worker.done.emit(({a: [lico(60.0)]}, ScanStats(scanned=1, seconds=0.2)))
        QDialog.accept(self)
        return int(QDialog.DialogCode.Accepted)

    monkeypatch.setattr(ReferenceDialog, "exec", simuliacija)
    itog = ReferenceDialog.sprosit(None, None, GluhojCache(), 2400, str(tmp_path))

    assert itog is not None
    assert [(k.put.name, k.procent, k.v_poisk) for k in itog.karty] == [("a.jpg", None, True)]
    assert itog.reference.count == 1


def test_zakrytie_vo_vremya_razbora_zhdet_potok_i_ne_zakryvaet_bazu(okno, potoki,
                                                                    monkeypatch, tmp_path,
                                                                    qapp) -> None:
    """Те же грабли, что `main_window.closeEvent`: `wait` возвращает False ровно тогда,
    когда поток ещё жив. Закрыть окно эталона в этот миг — значит получить
    «QThread: Destroyed while thread is still running» и оглавление, закрытое под живым
    потоком (а база принадлежит главному окну)."""
    a = snimok(tmp_path, "a.jpg", [lico()])
    prinesi(okno, monkeypatch, [a])
    potok = okno.worker
    potok.zanimaet = True
    potok.otsvet_na_wait = False
    okno.show()
    srazu(qapp, 50)
    assert okno.isVisible()

    okno.reject()

    assert potok.stop_zyvali == 1, "поток не попросили остановиться"
    assert potok.prosili_wait, "у потока не спросили, встал ли он"
    assert okno.isVisible(), "окно закрылось под живым потоком"
    assert okno.cache.zakrytij == 0, "оглавление главного окна закрыто диалогом"

    potok.zanimaet = False
    potok.finished.emit()
    srazu(qapp, 50)
    assert okno.isVisible() is False, "поток встал, а окно осталось висеть"
    assert okno.cache.zakrytij == 0


def test_otmena_posredi_stojanogo_potoka_ne_ronyaet_prilozenie(okno, potoki, monkeypatch,
                                                               tmp_path, qapp) -> None:
    """Если поток встал за отпущенное время, «Отмена» закрывает окно сразу и наружу
    ничего не уходит."""
    a = snimok(tmp_path, "a.jpg", [lico()])
    prinesi(okno, monkeypatch, [a])
    potok = okno.worker
    potok.zanimaet = True
    potok.otsvet_na_wait = True
    okno.show()
    srazu(qapp, 50)

    okno.reject()

    assert potok.stop_zyvali == 1
    assert okno.isVisible() is False
    assert okno.itog is None


def test_vtoraja_otmena_ne_eshet_tri_sekundy(okno, potoki, monkeypatch, tmp_path,
                                             qapp) -> None:
    """Второе нажатие в ожидании потока не вешает модальное окно ещё на три секунды:
    отложенное закрытие уже назначено, человек ждёт ровно один раз."""
    from PySide6.QtGui import QCloseEvent

    a = snimok(tmp_path, "a.jpg", [lico()])
    prinesi(okno, monkeypatch, [a])
    potok = okno.worker
    potok.zanimaet = True
    potok.otsvet_na_wait = False
    okno.show()
    srazu(qapp, 50)

    okno.reject()
    sok = len(potok.prosili_wait)
    assert sok == 1
    okno.closeEvent(QCloseEvent())
    okno.reject()

    assert len(potok.prosili_wait) == sok, "ждём поток повторно — окно снова висит"
    assert okno.isVisible()


# --- слова, которых в интерфейсе быть не может ----------------------------------------


def vse_teksty(vidzhet) -> list[str]:
    """Все тексты, которые человек может увидеть в этом окне: надписи, кнопки,
    подсказки, заголовок и подписи карточек."""
    from PySide6.QtWidgets import QListWidget

    teksty = [vidzhet.windowTitle(), vidzhet.toolTip(), vidzhet.accessibleName()]
    for docha in vidzhet.findChildren(object):
        for imja in ("text", "toolTip", "placeholderText", "accessibleName", "statusTip",
                     "whatsThis"):
            poluchit = getattr(docha, imja, None)
            if poluchit is None:
                continue
            try:
                znachenie = poluchit()
            except TypeError:                    # метод с обязательным аргументом
                continue
            if isinstance(znachenie, str):
                teksty.append(znachenie)
        if isinstance(docha, QListWidget):
            for i in range(docha.count()):
                element = docha.item(i)
                teksty.append(element.text())
                if element.toolTip():
                    teksty.append(element.toolTip())
    return [t for t in teksty if t]


# Что обход обязан увидеть в каждом состоянии. Без этого списка «проверка прошла»
# означала бы «проверять было нечего»: у пустого диалога текстов ровно пять.
VIDIMO_V_SOSTOJANII: dict[str, tuple[str, ...]] = {
    "pustoe": ("Добавьте фото", "Добавить фото", "Отмена"),
    "razbor": ("разбираем", "Отменить разбор"),
    "otmechennoe": ("ищем по этому лицу", "Искать"),
    "slaboe_pohozhee": ("ищем по этому лицу", "не похоже на остальных",
                         "не похожи на остальных"),
    "neprigodnoe": ("сравнить нечем", "не ищем"),
    "bez_lic": ("лицо не найдено",),
    "ne_razobralsja": ("не разобралось",),
    "ostanovlennoe": ("останов",),
    "snyatye_otmetki": ("Отметьте",),
}


def dialog_v_sostojanii(tmp_path: Path, sostojanie: str, monkeypatch):
    """Окно, доведённое до конкретного состояния: каждая ветка подписей — свой литерал,
    и запрещённое слово обязано быть недопустимым во всех, а не в заметной.

    Состояние `slaboe_pohozhee` добавлено задачей T14: раньше «слабое» лицо было
    синонимом отказа, теперь это участник поиска с предупреждением, и его текст обязан
    попасть и в обход запрещённых слов.
    """
    papka = tmp_path / sostojanie
    papka.mkdir(exist_ok=True, parents=True)
    dlg = novyj_dialog()

    pary = [lico(100.0), slomannoe_lico(80.0, (100.0, 100.0))]
    if sostojanie == "slaboe_pohozhee":
        pary = [lico(100.0), lico(80.0, 0.2, (100.0, 100.0))]
    if sostojanie != "pustoe":
        horoshee = snimok(papka, "a.jpg", pary)
    chistoe = snimok(papka, "b.jpg", [])
    gniloj = gniloj_fajl(papka, "c.heic")

    if sostojanie == "pustoe":
        return dlg
    if sostojanie == "razbor":
        prinesi(dlg, monkeypatch, [horoshee])
        return dlg
    if sostojanie == "bez_lic":
        prinesi(dlg, monkeypatch, [chistoe])
        gotovo(dlg, {chistoe: []})
        return dlg
    if sostojanie == "ne_razobralsja":
        prinesi(dlg, monkeypatch, [gniloj])
        gotovo(dlg, {}, ScanStats(failures=[(str(gniloj), "файл не читается: broken image")]))
        return dlg
    if sostojanie == "ostanovlennoe":
        prinesi(dlg, monkeypatch, [horoshee, chistoe])
        dlg.stop_button.click()
        gotovo(dlg, {horoshee: pary})
        return dlg

    prinesi(dlg, monkeypatch, [horoshee])
    gotovo(dlg, {horoshee: pary})
    if sostojanie in ("otmechennoe", "slaboe_pohozhee"):
        otmetit_vse(dlg)
    elif sostojanie == "snyatye_otmetki":
        otmetit_vse(dlg)
        for i in range(dlg.spisok.count()):
            element = dlg.spisok.item(i)
            if element.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                element.setCheckState(Qt.CheckState.Unchecked)
    else:                                              # neprigodnoe: отмечено только битое
        otsvetit(dlg, horoshee, 2)
    return dlg


@pytest.mark.parametrize("sostojanie", list(VIDIMO_V_SOSTOJANII))
def test_v_tekstah_dialoga_net_zapreshhennyh_slov(qapp, potoki, monkeypatch, tmp_path,
                                                  sostojanie: str) -> None:
    dlg = dialog_v_sostojanii(tmp_path, sostojanie, monkeypatch)
    teksty = vse_teksty(dlg)
    assert len(teksty) > 6, f"нечего проверять: {teksty}"
    for tekst in teksty:
        niz = tekst.lower()
        for stem in STEM:
            assert stem not in niz, f"«{stem}» в тексте: {tekst!r}"
        for vzorec in CELYE_SLOVA:
            assert not re.search(vzorec, niz), f"запрещённое слово в тексте: {tekst!r}"


@pytest.mark.parametrize("sostojanie", list(VIDIMO_V_SOSTOJANII))
def test_perekhod_tekstov_lovit_vse_vidimoe(qapp, potoki, monkeypatch, tmp_path,
                                            sostojanie: str) -> None:
    """Сам обход бесполезен, если ничего не нашёл: подписи карточек, кнопки, статус
    разбора и заглушки обязаны в него попадать."""
    dlg = dialog_v_sostojanii(tmp_path, sostojanie, monkeypatch)
    teksty = vse_teksty(dlg)
    for ozhidaemoe in VIDIMO_V_SOSTOJANII[sostojanie]:
        assert any(ozhidaemoe.lower() in t.lower() for t in teksty), \
            f"обход не увидел «{ozhidaemoe}» в {sostojanie}: {teksty}"


def test_v_ishodnikah_okna_etalona_net_zapreshhennyh_slov() -> None:
    """Часть подписей живёт не в виджетах, а в строках исходника: обходом виджетов их не
    поймать. Проверяем и модуль отрисовки карточек, который диалог импортирует."""
    koren = Path(__file__).resolve().parents[1] / "src"
    rezultat: list = []
    for imya in ("ui/reference_dialog.py", "ui/face_picker.py"):
        derevo = ast.parse((koren / imya).read_text(encoding="utf-8"))
        docstriki = set()
        for uzel in ast.walk(derevo):
            if isinstance(uzel, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                telo = getattr(uzel, "body", [])
                if telo and isinstance(telo[0], ast.Expr) and \
                        isinstance(telo[0].value, ast.Constant) and \
                        isinstance(telo[0].value.value, str):
                    docstriki.add(id(telo[0].value))
        for uzel in ast.walk(derevo):
            if isinstance(uzel, ast.Constant) and isinstance(uzel.value, str) \
                    and id(uzel) not in docstriki:
                rezultat.append(uzel.value)
    assert len(rezultat) > 10, "скан не нашёл строк — проверка пустая"
    for tekst in rezultat:
        niz = tekst.lower()
        for stem in STEM:
            assert stem not in niz, f"«{stem}» в строке исходника: {tekst!r}"


# --- T12: двойной клик открывает просмотр ----------------------------------------------

def dvojnaya_mysh_po_kartochke(dlg, element) -> None:
    """Двойное нажатие мышью: пресс, релиз, пресс со счётчиком 2, релиз."""
    from PySide6.QtCore import QPoint, Qt as _Qt
    from PySide6.QtTest import QTest

    prost = dlg.spisok.visualItemRect(element)
    tochka = prost.center()
    QTest.mouseClick(dlg.spisok.viewport(), _Qt.MouseButton.LeftButton,
                     _Qt.KeyboardModifier.NoModifier, tochka)
    QTest.mouseDClick(dlg.spisok.viewport(), _Qt.MouseButton.LeftButton,
                      _Qt.KeyboardModifier.NoModifier, tochka)
    dlg.spisok.viewport().repaint()


def test_dvojnoj_klik_otkryvaet_prosmotr_etogo_zhe_snimka(okno, potoki, monkeypatch,
                                                          tmp_path) -> None:
    """Двойной клик по карточке лица — то же окно крупного кадра, что и в списке
    находок: человек решает «моё / не моё» одними глазами, и второй интерфейс для
    того же решения только запутает."""
    faces = [lico(60.0), lico(80.0, 0.9, (100.0, 0.0))]
    a = snimok(tmp_path, "a.jpg", faces)
    prinesi(okno, monkeypatch, [a])
    gotovo(okno, {a: faces})

    okno.spisok.dvojnoe_nazhatie.emit(okno._karta(a, 2))

    prosmotr = okno._okno_prosmotra
    assert prosmotr is not None, "окно просмотра не открылось"
    assert prosmotr.pokazyvaet == a
    assert not prosmotr.foto.pixmap().isNull()
    assert not prosmotr.krop.pixmap().isNull(), "лицо, по которому кликнули, не показано"


def test_v_prosmotre_etalona_net_galochki_kopirovaniya(okno, potoki, monkeypatch,
                                                       tmp_path) -> None:
    """Здесь решают, КОГО искать, а не что переносить. Галочка «копировать это фото»
    в этом контексте обещает действие, которого окно не делает."""
    a = snimok(tmp_path, "a.jpg", [lico()])
    prinesi(okno, monkeypatch, [a])
    gotovo(okno, {a: [lico()]})

    okno.spisok.dvojnoe_nazhatie.emit(okno._karta(a, 1))

    assert okno._okno_prosmotra.copy_check.isHidden()


def test_dvojnoj_klik_ne_menayet_otmetku(okno, potoki, monkeypatch, tmp_path) -> None:
    """Ловушка, о которую спотыкается любой интерфейс с обоими жестами: двойное
    нажатие начинается с ОДИНОЧНОГО пресса, и если пресс переключает отметку, то
    двойной клик открывает окно И переворачивает галочку. Человек этого не просил.

    Проверается итог, а не число вызовов: после двойного клика отметка обязана быть
    ровно такой, какой была до него.
    """
    faces = [lico(60.0), lico(80.0, 0.9, (100.0, 0.0))]
    a = snimok(tmp_path, "a.jpg", faces)
    prinesi(okno, monkeypatch, [a])
    gotovo(okno, {a: faces})
    karta = okno._karta(a, 1)
    bylo = karta.checkState()

    dvojnaya_mysh_po_kartochke(okno, karta)

    assert karta.checkState() == bylo, (
        f"двойной клик перевернул отметку: было {bylo.value()}, "
        f"стало {karta.checkState().value()}")
    assert okno._okno_prosmotra is not None


def test_odinochnyj_klik_prodol_zametit(okno, potoki, monkeypatch, tmp_path) -> None:
    """И обратная сторона той же правки: защитив двойной клик, нельзя потерять
    одиночный — именно он нужен, чтобы отмечать лица на групповом снимке."""
    faces = [lico(60.0), lico(80.0, 0.9, (100.0, 0.0))]
    a = snimok(tmp_path, "a.jpg", faces)
    prinesi(okno, monkeypatch, [a])
    gotovo(okno, {a: faces})
    karta = okno._karta(a, 1)
    assert karta.checkState().value == 0

    mysh_po_tela_kartochki(okno, karta)

    assert karta.checkState().value == 2, "одиночный клик перестал отмечать"


def test_zavershenie_dialoga_zakryvaet_prosmotr(okno, potoki, monkeypatch,
                                                tmp_path) -> None:
    """Крупный снимок принадлежит выбору: пережив диалог, он висел бы на экране с
    процентом от решения, которое человек уже подтвердил или отменил."""
    a = snimok(tmp_path, "a.jpg", [lico()])
    prinesi(okno, monkeypatch, [a])
    gotovo(okno, {a: [lico()]})
    okno.spisok.dvojnoe_nazhatie.emit(okno._karta(a, 1))
    prosmotr = okno._okno_prosmotra
    assert prosmotr.isVisible()

    okno.reject()

    assert not prosmotr.isVisible(), "просмотр пережил отмену выбора эталона"


# --- T13: повторное открытие возвращает прежний выбор -----------------------------------


def vse_kluchi(dlg):
    """(файл, номер лица) каждой карточки в том порядке, как они стоят на экране."""
    kluchi = [dlg.spisok.item(i).data(Qt.ItemDataRole.UserRole)
              for i in range(dlg.spisok.count())]
    return [(k[0], k[1]) for k in kluchi]


def vtoroe_okno(itog):
    """Новое окно эталона, открытое поверх прежнего выбора."""
    from ui.reference_dialog import ReferenceDialog

    return ReferenceDialog(SimpleNamespace(name="glushka", loaded=True), GluhojCache(),
                           2400, "/tmp", nahodno=itog)


def test_povtornoe_otkrytievozvraschaet_vse_kartochki(okno, potoki, monkeypatch,
                                                      tmp_path) -> None:
    """Поиск прошёл, человек снова жмёт «Выбрать фото ребёнка» — обязан увидеть тот же
    набор карточек, а не пустое окно.

    Прежний выбор никуда не делся: подпись главного окна продолжает считать «лиц 4,
    снимков 4». Пустое окно рядом с такой подписью — два экрана, которые рассказывают
    про один эталон разное.
    """
    a = snimok(tmp_path, "a.jpg", [lico(60.0), lico(80.0)])
    b = snimok(tmp_path, "b.jpg", [lico(70.0)])
    c = snimok(tmp_path, "c.jpg", [])
    prinesi(okno, monkeypatch, [a, b, c])
    gotovo(okno, {a: [lico(60.0), lico(80.0)], b: [lico(70.0)], c: []})
    otsvetit(okno, a, 2)
    pervyj_itog = okno.itog
    assert pervyj_itog is not None
    bylo = vse_kluchi(okno)

    vtoroe = vtoroe_okno(pervyj_itog)
    try:
        assert vse_kluchi(vtoroe) == bylo, (
            f"карточки не вернулись: было {bylo}, стало {vse_kluchi(vtoroe)}")
    finally:
        vtoroe.close()
        vtoroe.deleteLater()


def test_otmechennye_lica_vosstanavlivayutsya_otmechennymi(okno, potoki, monkeypatch,
                                                           tmp_path) -> None:
    """Отметки приезжают вместе с карточками: иначе человек увидит те же лица снятыми,
    а «Искать по этим лицам» станет неактивной там, где поиск уже идёт."""
    a = snimok(tmp_path, "a.jpg", [lico(60.0), lico(80.0)])
    prinesi(okno, monkeypatch, [a])
    gotovo(okno, {a: [lico(60.0), lico(80.0)]})
    otsvetit(okno, a, 1)
    otsvetit(okno, a, 2)
    itog = okno.itog
    bylo = [okno._karta(a, n).checkState() for n in (1, 2)]
    assert bylo == [Qt.CheckState.Checked, Qt.CheckState.Checked]

    vtoroe = vtoroe_okno(itog)
    try:
        stalo = [vtoroe._karta(a, n).checkState() for n in (1, 2)]
        assert stalo == bylo, f"отметки потерялись: {[s.value for s in stalo]}"
        assert vtoroe.ok_button.isEnabled() is True
        assert [k.procent for k in vtoroe.itog.karty] == [k.procent for k in itog.karty]
    finally:
        vtoroe.close()
        vtoroe.deleteLater()


def test_bez_naslediya_okno_pustoe_kak_i_resheno() -> None:
    """Первый запуск остаётся пустым: наследие не имеет права что-либо придумывать."""
    dlg = novyj_dialog()
    try:
        assert dlg.spisok.count() == 0
        assert dlg.itog is None or not dlg.itog.karty
    finally:
        dlg.close()
        dlg.deleteLater()



# --- T17: вклад каждого отмеченного лица в находки ---------------------------------------


def test_kartochka_govorit_skolko_snimkov_naideno_po_ee_licu(okno, potoki, monkeypatch,
                                                            tmp_path) -> None:
    """Против каждого отмеченного лица — сколько снимков нашлось именно по нему.

    Пользователь отменил молчаливый отказ фильтра, и цена отмены стала его задачей:
    если среди отмеченных затесалось лицо взрослого, по нему начнёт находиться взрослый,
    а выглядеть будет как находка ребёнка. Запрещать мы это не стали — показываем число,
    и человек снимает галочку сам.

    Замер на архиве: эталон из 12 отобранных лиц даёт 20 из 20 подтверждённых снимков и
    ноль чужих; эталон из 24 — те же 20 своих и 10 чужих. То есть фильтр не стоил ни
    одной находки, и это число должно быть видно на экране, а не только в моём отчёте.
    """
    a = snimok(tmp_path, "a.jpg", [lico(60.0), lico(80.0)])
    prinesi(okno, monkeypatch, [a])
    gotovo(okno, {a: [lico(60.0), lico(80.0)]})
    otsvetit(okno, a, 1)
    otsvetit(okno, a, 2)
    # крупнейшее (карточка 2) стало первой строкой эталона, но на экране это не подписано
    # ничем особым: с задачи T21 все отмеченные лица равноправны
    assert okno.itog.anchor.nomer == 2
    assert okno._karta(a, 2).text().endswith("ищем по этому лицу")

    okno.po_kazhdomu_licu = {1: 14, 2: 1}
    okno._obnovit_kartochki()

    p1 = okno._karta(a, 1).text()
    p2 = okno._karta(a, 2).text()
    assert "1 сним" in p1, f"вклада второго лица не видно: {p1!r}"
    assert "14 сним" in p2, f"вклада главного лица не видно: {p2!r}"


def test_lico_po_kotoromu_ne_nashlos_nichego_skazano_prymo(okno, potoki, monkeypatch,
                                                           tmp_path) -> None:
    """Ноль — это тоже ответ: «по нему не нашлось ни одного снимка». Пустое место
    человек прочитал бы как «приложение не знает», а не как «это лицо бесполезно»."""
    a = snimok(tmp_path, "a.jpg", [lico(60.0), lico(80.0)])
    prinesi(okno, monkeypatch, [a])
    gotovo(okno, {a: [lico(60.0), lico(80.0)]})
    otsvetit(okno, a, 1)
    otsvetit(okno, a, 2)

    okno.po_kazhdomu_licu = {1: 20, 2: 0}
    okno._obnovit_kartochki()

    assert "ни одного" in okno._karta(a, 1).text(), \
        "лицу, по которому не нашлось ничего, не сказано об этом"
    assert "ни одного" not in okno._karta(a, 2).text()


def test_bez_chisla_podpis_ne_vrjet(okno, potoki, monkeypatch, tmp_path) -> None:
    """До первого поиска чисел нет: окно эталона открывается и раньше. Врать
    «по нему найдено 0» там, где искать ещё не начинали, — значит отговорять
    человека от правильного лица."""
    a = snimok(tmp_path, "a.jpg", [lico()])
    prinesi(okno, monkeypatch, [a])
    gotovo(okno, {a: [lico()]})

    tekst = okno._karta(a, 1).text()
    assert "найдено" not in tekst, f"вклад там, где поиска не было: {tekst!r}"
