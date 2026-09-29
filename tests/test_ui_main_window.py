"""Главное окно: три шага, фоновый разбор и мгновенный пересчёт.

Файл проверяет не «кнопки на месте», а обещания, которые теряются именно на сборке
окна, потому что на этот раз за них отвечают уже разные модули ядра:

1. ползунок похожести НЕ имеет права запустить разбор папки — лица уже посчитаны,
   пересчёт идёт из памяти (`ScanWorker` не создаётся ни разу);
2. шторм сигналов при перетаскивании (до тридцати штук) сворачивается в один пересчёт и
   одну запись настроек на диск, а не записанное значение не теряется при закрытии;
3. `copy_photos` зовётся ТРЕМЯ позиционными аргументами, и причины потерь доезжают до
   report.txt вместе с числами отбраковки из `ScanStats`;
4. `Settings.save() == False` не выбрасывается молча;
5. `ValueError` от `make_reference` показывается человеком, а не тонет в слоте Qt;
6. сообщение рабочего потока показывается как есть, а сетка на ошибке не очищается;
7. в окне нет запрещённых спецификацией слов — ни в виджетах, ни в строках исходника, и
   число на карточке объяснено прямым текстом;
8. окно занятого разбора НЕ принимает правки полосы настроек: ни один выпадающий список
   посреди трёхминутного разбора не имеет права пересогласовать `scan_dim` с сеткой;
9. закрытие окна не закрывает оглавление под живым рабочим потоком;
10. отмена на шаге эталона не выдаётся за «лиц на фото нет»;
11. накопители потерь живут ровно один разбор и обнуляются чистым разбором;
12. сигнал рабочего потока доезжает до окна: `progress`, `error` и `done` идут через
    настоящие `Signal`, а не через вызов слота руками.

Модели не загружаются ни в одном тесте: разбор подменяется фиктивным рабочим потоком,
а готовый результат приходит в окно тем же путём, что и в живом приложении, — через
`emit` сигнала. `data/` и `ref/` не читаются: там снимки конкретного ребёнка, все файлы
рисуются в `tmp_path`.
"""

from __future__ import annotations

import ast
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, Signal                # noqa: E402

from core.engine import Face                       # noqa: E402
from core.etalon import Chelovek, Itog, Kartochka  # noqa: E402
from core.matcher import make_reference, percent_of      # noqa: E402
from core.worker import ScanStats                  # noqa: E402

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget

# Стем, а не слово целиком: ловим и «порог», и «порога», и «детектора».
STEM = ("порог", "уверенност", "embedd", "косинус", "детектор")
# «пол» — отдельной строкой и только как целое слово: как подстрока он сидит в «ползунке».
CELYE_SLOVA = (r"\bпол(а|у|ом|е)?\b", r"\bвозраст(а|у|е|ом)?\b")

# Сколько крутить события, чтобы гарантированно дождаться таймера дебаунса (он 150 мс).
OZHDANIE_MS = 600


def polozhit_etalon(window, anchor, extra=(), nazvanie="ребёнок"):
    """Положить готовый эталон в шаг «Кого ищем» — тем же путём, каким его приносит окно.

    Модальное окно эталона в этих тестах не открывается, но `Itog` обязан быть настоящим:
    по нему строка человека пишет счётчик «лиц N, снимков M», а вклад отметок считается
    номером строки. Файл-эталон фиктивный: он нужен только как подпись под «снимков».
    """
    karty = [Kartochka(put=Path(f"/эталон{i}.jpg"), nomer=1, lic=lic, otmecheno=True,
                       procent=None, v_poisk=True, slabo=False)
             for i, lic in enumerate((anchor, *extra))]
    itog = Itog(karty=tuple(karty), anchor=karty[0], extra=tuple(karty[1:]),
                otvergnutye=(), reference=make_reference(anchor, list(extra)))
    window.ljudi = [Chelovek(nazvanie=nazvanie, itog=itog)]
    window._perestroit_stroki_ljudej()
    return window.ljudi[0]


def lico(side=100.0, axis=0):
    """Единичный отпечаток вдоль выбранной оси: axis=0 совпадает с эталоном, axis=1 — нет."""
    v = np.zeros(512, dtype=np.float32)
    v[axis] = 1.0
    return Face(box=(0.0, 0.0, side, side * 1.5), landmarks=None,
                embedding=v, detector="insight")


class FalsivyjWorker(QObject):
    """Замена `ScanWorker`: фиксирует сам факт создания и не трогает модели.

    Проверка «ползунок не запускает разбор» возможна только так: настоящий `ScanWorker`
    — это `QThread`, и одно его создание уже означает, что окно надумало сканировать.

    Сигналы здесь настоящие (класс — `QObject`), а не заглушки с пустым `connect`.
    Прежняя подмена позволяла удалить все три строки `worker.progress.connect(...)` из
    окна, и весь файл тестов остался бы зелёным: каждый тест дергал слот напрямую и ни
    разу не проходил путь «поток испустил — окно отреагировало». Теперь не выйдет —
    см. `test_signaly_potoka_dovedeny_do_okna`.
    """

    sozdannye: list = []

    progress = Signal(int, int, str)     # та же сигнатура, что у `ScanWorker`
    done = Signal(object)
    error = Signal(str)
    finished = Signal()                  # есть и у настоящего `QThread`

    def __init__(self, files, engine, cache, max_dim, parent=None) -> None:
        super().__init__(parent)
        FalsivyjWorker.sozdannye.append(self)
        self.files, self.max_dim = list(files), max_dim
        self.stop_zyvali = False
        self.zanimaet = False            # ответ `isRunning` — для теста закрытия окна
        self.otsvet_na_wait = True       # ответ `wait`: поток встал за отпущенное время
        self.prosili_wait: list[int] = []

    def start(self) -> None:
        pass

    def isRunning(self) -> bool:                                 # noqa: N802 (имя Qt)
        return self.zanimaet

    def request_stop(self) -> None:
        self.stop_zyvali = True

    def wait(self, ms: int) -> bool:
        self.prosili_wait.append(ms)
        return self.otsvet_na_wait


@pytest.fixture
def bez_skana(monkeypatch) -> type:
    """Никакого настоящего разбора во всех тестах файла."""
    FalsivyjWorker.sozdannye = []
    import ui.main_window as modul
    monkeypatch.setattr(modul, "ScanWorker", FalsivyjWorker)
    return FalsivyjWorker


@pytest.fixture
def window(qapp, tmp_path):
    from ui.main_window import MainWindow
    from utils.config import Settings

    # cache_path — не придирка, а обязательная изоляция: по умолчанию оглавление живёт в
    # служебной папке пользователя, и тесты открывали бы живой кэш человека.
    return MainWindow(Settings(tmp_path / "s.ini"), cache_path=tmp_path / "c.sqlite3")


def srazu(qapp, ms: int = OZHDANIE_MS) -> None:
    """Дать таймерам отработать, крутя события, а не спать молча."""
    konec = time.monotonic() + ms / 1000.0
    while time.monotonic() < konec:
        qapp.processEvents()
        time.sleep(0.005)


# --- сборка ---------------------------------------------------------------------------


def test_okno_sobiraetsya_knopki_v_nachalnom_sostoyanii(window) -> None:
    assert window.scan_button.isEnabled() is False        # папка не выбрана
    assert window.copy_button.isEnabled() is False        # результатов нет
    assert "Где ищем" in window.step1.title()
    assert "Кого ищем" in window.step2.title()


def test_vybor_papki_vklyuchaet_razbor(window, tmp_path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"x")
    window.set_source_folders([tmp_path])
    assert window.scan_button.isEnabled()
    assert "1 фото" in window.source_status.text()


def test_papka_bez_foto_govoritsya_prymo(window, tmp_path) -> None:
    """Спецификация, раздел 7: «папка без фото — сказать прямо», а не выключить кнопку
    и молчать: человек решит, что приложение зависло."""
    (tmp_path / "zametki.txt").write_text("не фото", encoding="utf-8")
    window.set_source_folders([tmp_path])
    assert window.scan_button.isEnabled() is False
    assert "не найдено" in window.source_status.text()


def test_vlozhennye_papki_ne_dayut_dvojnogo_sheta(window, tmp_path) -> None:
    """Вложенная друг в друга пара папок давала бы каждый файл дважды (carry-over
    задачи 2): разбор платил бы моделями за один и тот же снимок."""
    vnutri = tmp_path / "arhiv" / "pod"
    vnutri.mkdir(parents=True)
    (vnutri / "a.jpg").write_bytes(b"x")
    (tmp_path / "arhiv" / "b.jpg").write_bytes(b"x")
    window.set_source_folders([tmp_path / "arhiv", vnutri])
    assert len(window.files) == 2, window.files


def test_dobavit_papku_rasshiryaet_spisok_a_ne_zamenyaet(window, tmp_path,
                                                         monkeypatch) -> None:
    """Спецификация, раздел 6: «выбор одной или нескольких папок».

    `getExistingDirectory` не умеет выделять несколько папок ни на одной платформе,
    поэтому их добавляет кнопка. Прежнее окно принимало ровно одну: вторая стирала
    первую, и человек, который ищет в «Фото» и «Камера», разбирал половину архива,
    глядя на честный счётчик найденного.
    """
    import ui.main_window as modul

    pervaya, vtoraya = tmp_path / "foto", tmp_path / "kamera"
    for papka, imya in ((pervaya, "a.jpg"), (vtoraya, "b.jpg")):
        papka.mkdir()
        (papka / imya).write_bytes(b"x")

    monkeypatch.setattr(modul.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(pervaya)))
    window._dobavit_papku()
    monkeypatch.setattr(modul.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(vtoraya)))
    window._dobavit_papku()

    assert window.folders == [pervaya, vtoraya]
    assert len(window.files) == 2
    assert "папок: 2" in window.source_status.text()
    assert str(pervaya) in window.source_list.text(), "первая папка пропала из вида"
    assert str(vtoraya) in window.source_list.text()

    # та же папка второй раз не удваивает ни список, ни число снимков
    window._dobavit_papku()
    assert window.folders == [pervaya, vtoraya] and len(window.files) == 2


def test_odna_knopka_papok_i_varianta_zamenit_net(window) -> None:
    """Двусмысленность, из-за которой кнопку убрали: «Выбрать папку» затирала то, что
    человек принёс кнопкой «Добавить папку», и отличить их можно было только после
    потери списка. Осталось одно действие — добавить; убрать всё — вторая кнопка.

    Тест держит это структурно: у окна ровно две кнопки шага 1, и ни одна не заменяет
    список. Прежняя схема ловилась бы словами на кнопке, а слова — косметика.
    """
    knopki = [window.add_folder_button, window.clear_folders_button]
    assert len(knopki) == 2
    assert not hasattr(window, "source_button"), \
        "кнопка-заменитель вернулась: она молча стирает добавленные папки"
    assert window.add_folder_button.text() == "Добавить папку с фото"

    # «Добавить» не имеет права затерять уже принесённое
    vtoraya = Path("/drugaia")
    window.folders = [Path("/pervaya")]
    window.files = [Path("/pervaya/a.jpg")]
    window.set_source_folders(window.folders + [vtoraya])
    assert window.folders == [Path("/pervaya"), vtoraya]


def test_ubrat_vse_chistit_spisok_papok_no_ne_tropaet_nahodok(window, tmp_path) -> None:
    """«Убрать все» относится к списку папок, а не к снятым лицам.

    Разбор стоит минуты, и стереть его находки кнопкой, которая называется «убрать
    папки», — значит заставить человека платить моделями заново из-за непонятного
    слова. Находки снимает только новый «Начать разбор папки».
    """
    arhiv = tmp_path / "arhiv"
    arhiv.mkdir()
    (arhiv / "a.jpg").write_bytes(b"x")
    window.set_source_folders([arhiv])
    window.photos = {arhiv / "a.jpg": [lico()]}
    assert window.clear_folders_button.isEnabled() is True

    window._ubrat_papki()

    assert window.folders == [] and window.files == []
    assert "папки не выбраны" in window.source_status.text()
    assert window.source_list.text() == ""
    assert window.scan_button.isEnabled() is False, "разбирать нечего, а кнопка жива"
    assert window.clear_folders_button.isEnabled() is False, "чистить уже нечего"
    assert window.settings_bar.estimate_label.text() == "", "оценка на пустом списке"
    # Сверка по путям, а не по лицам: `Face.__eq__` на numpy-векторе бросает ValueError
    # (это зафиксировано в docstring `core.engine`), и сравнение словарей с ними
    # развалилось бы в тесте, а не в приложении.
    assert list(window.photos) == [arhiv / "a.jpg"], "убрало найденные лица"


def test_ubrat_vse_pomnit_kuda_hodit_v_sledushij_raz(window, tmp_path) -> None:
    """Последняя папка остаётся в настройках: «убрать список» — не «забыть, где фото».

    Без этого следующий «Добавить папку с фото» открылся бы дома, а не там, где
    человек только что был, и поиск по трём папкам превратился бы в три перехода.
    """
    arhiv = tmp_path / "arhiv"
    arhiv.mkdir()
    window.set_source_folders([arhiv])
    window._ubrat_papki()
    assert window.settings.last_source == str(arhiv)


def test_otmena_dIALOGA_papku_ne_sosaet(window, tmp_path, monkeypatch) -> None:
    """Пустой ответ диалога — это «человек отменил», а не «больше не ищем нигде»."""
    import ui.main_window as modul

    arhiv = tmp_path / "arhiv"
    arhiv.mkdir()
    (arhiv / "a.jpg").write_bytes(b"x")
    window.set_source_folders([arhiv])
    monkeypatch.setattr(modul.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: ""))
    window._dobavit_papku()
    assert window.folders == [arhiv] and len(window.files) == 1
    assert window.clear_folders_button.isEnabled() is True, \
        "отменённый диалог почистил кнопку"


def test_razbora_net_papki_ne_menyayut(window, bez_skana, tmp_path) -> None:
    """Посреди разбора папки не добавляются и не убираются.

    Поток держит свой список файлов, а подпись, счётчик и оценка времени — нет:
    человек видел бы прогресс одного архива под заголовком другого.
    """
    arhiv = tmp_path / "arhiv"
    arhiv.mkdir()
    (arhiv / "a.jpg").write_bytes(b"x")
    window.set_source_folders([arhiv])
    window.start_scan()
    assert window.add_folder_button.isEnabled() is False
    assert window.clear_folders_button.isEnabled() is False
    assert window.scan_button.isEnabled() is False
    window.worker.done.emit(({}, ScanStats(scanned=1, cached=0, seconds=0.2)))
    assert window.add_folder_button.isEnabled() is True
    assert window.clear_folders_button.isEnabled() is True


def test_ocenka_vremeni_sootvetstvuet_rezhimu_i_arhivu(window, tmp_path) -> None:
    """Спецификация, раздел 6: «в подписи — минуты на текущем архиве».

    Человек выбирает способ поиска ради времени, и число обязано стоять рядом с
    выбором, меняться вместе с ним и исчезать, когда считать нечего: оценка «0 с»
    на невыбранной папке была бы обещанием мгновенного разбора.
    """
    arhiv = tmp_path / "arhiv"
    arhiv.mkdir()
    assert window.settings_bar.estimate_label.text() == "", "оценка до выбора папки"

    for i in range(100):
        (arhiv / f"f{i}.jpg").write_bytes(b"x")
    window.set_source_folders([arhiv])
    do_smery = window.settings_bar.estimate_label.text()
    assert "100 фото" in do_smery and do_smery.startswith("≈"), do_smery

    # «Оба» ищет двумя способами и потому дороже «Быстрее» ровно в измеренные разы
    window.settings_bar.set_engine("both")
    oba = window.settings_bar.estimate_label.text()
    window.settings_bar.set_engine("yunet")
    bystree = window.settings_bar.estimate_label.text()
    assert oba != bystree, "смена способа не меняет оценку"
    assert "3 мин" in oba, oba             # 100 x 1,9 с = 190 с
    assert "50 с" in bystree, bystree      # 100 x 0,5 с = 50 с, до минуты


def test_shirina_okna_ne_dreshit_polosu_nastroek(window) -> None:
    """Полосе настроек нужно ~970 px: если окно уже, Qt сожмёт подписи, и человек
    увидит «Насколько фото до…» вместо настройки."""
    assert window.minimumWidth() >= window.settings_bar.sizeHint().width()


# --- ползунок: пересчёт, а не разбор ---------------------------------------------------


def test_smena_poroga_pereschityivaet_bez_novogo_skana(window, qapp,
                                                       bez_skana) -> None:
    """Ядро теста: ползунок не имеет права запускать разбор папки заново."""
    window.photos = {Path("/svoe.jpg"): [lico(100, 0)], Path("/chuzhoe.jpg"): [lico(50, 1)]}
    polozhit_etalon(window, lico(100, 0))
    window.worker = None

    window.settings_bar.set_threshold(0.30)
    srazu(qapp)
    assert window.results.counts == (1, 1)      # своё похоже, чужое в слабом сходстве
    window.settings_bar.set_threshold(0.50)
    srazu(qapp)
    assert window.results.counts == (1, 1)      # своё 100 %, чужое 0 % — состав тот же
    window.settings_bar.set_threshold(0.60)     # потолок шкалы: 0.90 полоса не примет
    srazu(qapp)
    assert window.results.counts[0] == 1
    assert window.worker is None
    assert bez_skana.sozdannye == [], "ползунок создал рабочий поток — это разбор папки"


def test_debaun_odin_pereschet_i_odna_zapis_na_disk(window, qapp, bez_skana) -> None:
    """Тридцать сигналов при перетаскивании = один пересчёт и одна запись настроек.

    Без дебаунса каждое деление ползунка значило бы `QSettings.sync()` на диск плюс
    полную пересборку сетки: родитель видит тридцать миганий и слышит, как работает
    диск, — а итоговое число то же самое.
    """
    window.photos = {Path(f"/ф{i}.jpg"): [lico()] for i in range(5)}
    polozhit_etalon(window, lico())

    pereschety: list = []
    zapisi: list = []
    nastoyashhij_pereschet = window.rebuild_scores
    nastoyashhij_save = window.settings.save

    def schitat_pereschet() -> None:
        pereschety.append(1)
        nastoyashhij_pereschet()

    def schitat_zapis() -> bool:
        zapisi.append(1)
        return nastoyashhij_save()

    window.rebuild_scores = schitat_pereschet
    window.settings.save = schitat_zapis

    for procent in range(30, 60):                      # 30 значений подряд, как при drag
        window.settings_bar.set_threshold(procent / 100.0)
    assert pereschety == [], "пересчёт случился до того, как ползунок остановился"
    srazu(qapp)
    assert len(pereschety) == 1, f"пересчётов было {len(pereschety)}, ждём один"
    assert len(zapisi) == 1, f"настройки писались на диск {len(zapisi)} раз"
    assert window.settings.threshold == pytest.approx(0.59)


def test_pereschet_ne_sozdaet_dvizhok(window, qapp, bez_skana) -> None:
    """Смена похожести не трогает эталон и не собирает движок: `engine` остаётся None,
    пока человек не попросит новый разбор."""
    window.photos = {Path("/a.jpg"): [lico()]}
    polozhit_etalon(window, lico())
    window.settings_bar.set_threshold(0.45)
    srazu(qapp)
    assert window.engine is None
    assert bez_skana.sozdannye == []


# --- разбор и его сбои -------------------------------------------------------------------


def test_otmena_zovyvaet_request_stop(window) -> None:
    calls = []
    window.worker = type("W", (), {"request_stop": lambda self: calls.append(1)})()
    window._set_busy(True, "тест")               # без этого кнопка «Остановить» выключена,
    assert window.stop_button.isEnabled()        # и click() молча ничего бы не сделал
    window.stop_button.click()
    assert calls == [1]


SBVOI_MODELEJ = ("не удалось загрузить модели: нет файла весов. Распознавание не "
                 "начиналось: ни один снимок не разобран.")
SBVOI_OGLAVLENIYA = ("оглавление недоступно: на диске закончилось свободное место — "
                     "освободите его; уже снятые лица сохранены — 42 снимка в оглавлении.")


def test_sbboy_pokazyvaetsya_kak_est_i_setka_ne_chistitsya(window, qapp, monkeypatch,
                                                           bez_skana) -> None:
    """Рабочий поток сам различает этапы; окно не имеет права добавлять свой заголовок.

    `ModeliNeZagruzilis` и `OglavlenieNedostupno` — это два разных разговора с
    человеком: первому надо проверить путь к моделям, второму — освободить диск, и у
    второго уже снятое лежит в оглавлении. Собственный префикс («не удалось начать
    разбор…») сделал бы из одного другое, а очистка сетки уничтожила бы частичный
    результат, который именно в этом случае и сохранён.
    """
    window.photos = {Path("/a.jpg"): [lico()]}
    polozhit_etalon(window, lico())
    window.settings_bar.set_threshold(0.45)     # не 0.38: на умолчалке сигнала нет
    srazu(qapp)
    bylo = window.results.counts
    assert bylo[0] == 1, "пересчёт не случился — проверять «не очистили» нечего"

    pokazanno: list = []
    monkeypatch.setattr(window, "_warn", lambda text: pokazanno.append(text))
    window._on_scan_error(SBVOI_MODELEJ)
    window._on_scan_error(SBVOI_OGLAVLENIYA)

    assert pokazanno == [SBVOI_MODELEJ, SBVOI_OGLAVLENIYA], "текст сообщения переписан"
    assert window.results.counts == bylo, "сетку на ошибке очищать нельзя"
    for text in pokazanno:
        assert "не удалось начать" not in text.lower()


def test_sbboy_ne_ostavlyaet_okno_zanyatym(window, monkeypatch) -> None:
    monkeypatch.setattr(window, "_warn", lambda text: None)
    window._set_busy(True, "идёт разбор")
    window._on_scan_error("оглавление недоступно: диск не отвечает")
    assert window.stop_button.isEnabled() is False
    assert window._strok[0].knopka.isEnabled() is True


def test_chastichnyj_rezultat_doezzhaet_do_setki(window, qapp, bez_skana) -> None:
    """Отмена — не провал: присланный кусок разбирается и показывается как обычно, а
    число потерь видно в статусе."""
    window.photos = {}
    polozhit_etalon(window, lico())
    statistika = ScanStats(scanned=3, cached=1, seconds=4.0,
                           failures=[("/bityj.jpg", "файл не читается: cannot identify")],
                           dropped_faces={Path("/g.jpg"): 2}, cache_damage=1)
    window._on_scan_done(({Path("/a.jpg"): [lico()]}, statistika))
    srazu(qapp)
    assert Path("/a.jpg") in window.photos
    assert window.results.counts[0] == 1
    assert "не обработано: 1" in window.status_label.text()
    assert window._poteri == {"/bityj.jpg": "файл не читается: cannot identify"}
    assert window._otbroennye == {Path("/g.jpg"): 2}
    assert window._povrezhdeniya == 1


# --- шаг 2: эталон ------------------------------------------------------------------------
#
# Прежний маршрут «окно само разбирает один эталонный файл» (`_vybrat_foto`, `_na_etalon`,
# `_handle_reference_faces`, `_ask_face`, `more_button`) удалён целиком: выбор эталона живёт
# в окне `ui/reference_dialog.py`, и его собственные обещания — неразобранный файл, отмена
# разбора, «похоже слабо» вместо отказа и независимость от порядка кликов — проверяются там,
# в `tests/test_ui_reference_dialog.py`. Здесь остаётся шов: что окно передаёт диалогу, что
# делает с ответом и чего не трогает при отказе.


class FalsivyjDialog:
    """Замена `ReferenceDialog`: фиксирует, КАКИМ его позвали, и отдаёт заготовленный ответ.

    Настоящий диалог — модальный цикл событий со своим потоком; подменяя его здесь, мы
    проверяем именно договор «окно -> диалог -> окно»: качество разбора и база уходят те же
    самые, а отказ не имеет права ничего стереть.
    """

    vyzy: list = []
    otvet = None
    nasledie_peredanne = None
    vklad_peredann = None
    nazvanie_peredanno = None
    chuzhie_peredannye = frozenset()

    @classmethod
    def sprosit(cls, parent, engine, cache, max_dim, nachalo, nahodno=None,
                vklad=None, nazvanie=None, chuzhie=frozenset()):
        cls.vyzy.append({"parent": parent, "engine": engine, "cache": cache,
                         "max_dim": max_dim, "nachalo": nachalo, "nahodno": nahodno,
                         "vklad": vklad, "nazvanie": nazvanie,
                         "chuzhie": chuzhie})
        cls.nasledie_peredanne = nahodno
        cls.vklad_peredann = vklad
        cls.nazvanie_peredanno = nazvanie
        cls.chuzhie_peredannye = chuzhie
        return cls.otvet


@pytest.fixture
def dialog(monkeypatch) -> type:
    """Диалог эталона подменён во всех тестах этого файла: модалку тут не крутят."""
    FalsivyjDialog.vyzy = []
    FalsivyjDialog.otvet = None
    import ui.main_window as modul
    monkeypatch.setattr(modul, "ReferenceDialog", FalsivyjDialog)
    return FalsivyjDialog


def nulevoe_lico(side: float = 10.0) -> Face:
    """Лицо с непригодным отпечатком: эталона из него не собирается."""
    return Face(box=(0.0, 0.0, side, side), landmarks=None,
                embedding=np.zeros(512, dtype=np.float32), detector="insight")


def itog_iz(lica_po_fajlam: dict, otmetki=None):
    """Настоящий `Itog` из настоящей арифметики `core.etalon`.

    Собирать `Itog` руками было бы подделкой договора: окно обязано получить ровно тот
    объект, который диалог показывает человеку, со всеми `anchor`/`extra`/`otvergnutye`.
    """
    from core.etalon import odinochnye_otmetki, sobrat_etalon
    return sobrat_etalon(lica_po_fajlam,
                         otmetki if otmetki is not None
                         else odinochnye_otmetki(lica_po_fajlam))


def lica_po_fajlam(*lic: Face) -> dict:
    """По одному лицу на снимок: файл на лицо, порядок словаря = порядок карточек."""
    return {Path(f"/etalon{i}.jpg"): [l] for i, l in enumerate(lic, start=1)}


def arhiv_s_dvumja_snimkami(window) -> None:
    """Разобранный архив из одного своего и одного чужого снимка, эталон не выбран.

    `folders` рядом обязателен: «Убрать все» и «Начать разбор» живут от непустого списка
    папок, и без этой строки тест рисовал бы окно с файлами, но без папок — состояние,
    которого в приложении не бывает.
    """
    window.photos = {Path("/svoe.jpg"): [lico(100, 0)], Path("/chuzhoe.jpg"): [lico(50, 1)]}
    window.files = list(window.photos)
    window.folders = [Path("/arhiv")]
    window.scan_button.setEnabled(True)


def test_odna_knopka_etalona_vtoraya_ubezhala_vnutr(window) -> None:
    """Кнопка «Добавить ещё фото этого же ребёнка» переехала внутрь окна эталона: там
    вторая партия не просит человека проходить выбор заново. Снаружи — одно действие."""
    assert not hasattr(window, "more_button"), \
        "вторая кнопка вернулась: снаружи она принимает ровно один файл за раз"
    # Одно действие на человека: имя живёт в своём поле, кнопка зовёт окно, и второй
    # кнопки «добавить ещё фото» снаружи больше нет.
    assert window._strok[0].imja.text() == "ребёнок"
    assert window._strok[0].knopka.text() == "Выбрать фото: ребёнок"


def test_knopka_etalona_zovet_dialog_a_ne_skana(window, dialog,
                                                 bez_skana) -> None:
    """ОКНО больше не разбирает эталонный файл само: у диалога свой поток.

    Проверяется и то, ЧТО уходит диалогу: та же база оглавления и то же качество разбора.
    Своё число означало бы промах кэша на каждом файле уже разобранного архива.
    """
    window.scan_dim = 1200
    window.settings.last_source = "/kuda_smotret"
    window.worker = None

    window._strok[0].knopka.click()

    assert len(dialog.vyzy) == 1, "диалог не спросили"
    vyzyv = dialog.vyzy[0]
    assert vyzyv["parent"] is window
    assert vyzyv["cache"] is window.cache, "диалог открыл бы свою базу вместо этой"
    assert vyzyv["max_dim"] == 1200, "качество разбора обязано прийти от окна"
    assert vyzyv["nachalo"] == "/kuda_smotret"
    assert bez_skana.sozdannye == [], "окно запустило свой разбор ради эталона"


def test_otmena_dialoga_ne_tropaet_pokazannogo(window, dialog, monkeypatch,
                                               bez_skana) -> None:
    """Отказ от диалога — не «эталон сброшен»: прежний ответ обязан остаться на экране.

    Пока человек выбирал фото, минуты разбора архива никуда не делись, и стереть их
    кнопкой «Отмена» значило бы заставить платить моделями повторно.
    """
    arhiv_s_dvumja_snimkami(window)
    polozhit_etalon(window, lico(100, 0))
    window.settings_bar.set_threshold(0.30)
    window.rebuild_scores()
    bylo_foto = window.photos
    bylo_etalon = window.ljudi[0].reference
    bylo_chislo = window.results.counts
    bylo_podpisi = window._strok[0].podpis.text()
    monkeypatch.setattr(window, "_warn", lambda text: None)
    dialog.otvet = None                                       # человек нажал «Отмена»

    window._strok[0].knopka.click()

    assert window.photos == bylo_foto
    assert window.ljudi[0].reference is bylo_etalon
    assert window.results.counts == bylo_chislo
    assert window._strok[0].podpis.text() == bylo_podpisi
    assert bez_skana.sozdannye == [], "отменённый диалог запустил разбор"


def test_itog_dialoga_napolnjaet_etalon_bez_novogo_razbora(window, dialog,
                              monkeypatch, bez_skana) -> None:
    """Главная договорённость раунда: выбор переносится, сетка пересчитывается СРАЗУ,
    нового разбора папки нет."""
    arhiv_s_dvumja_snimkami(window)
    itog = itog_iz(lica_po_fajlam(lico(100, 0), lico(90, 0)))
    dialog.otvet = itog
    schitano: list = []
    nastoyashhij_pereschet = window.rebuild_scores

    def schitat_pereschet() -> None:
        schitano.append(1)
        nastoyashhij_pereschet()

    monkeypatch.setattr(window, "rebuild_scores", schitat_pereschet)

    window._strok[0].knopka.click()

    assert window.ljudi[0].itog.anchor.lic is itog.anchor.lic
    assert window.ljudi[0].itog.extra == itog.extra
    assert window.ljudi[0].reference is itog.reference
    assert itog.reference.count == 2
    assert schitano == [1], "сетку не пересчитали — человек видит проценты прошлого эталона"
    assert window.results.counts[0] == 1
    assert bez_skana.sozdannye == [], "выбор эталона вызвал разбор архива"


def test_podpis_etalona_govorit_pro_slabyh_pohozhih(window, dialog) -> None:
    """Решение T14 глазами: слабое лицо участвует в поиске, и подпись считает его рядом
    с остальными, но называет отдельно — иначе человек решит, что его галочку не взяли.

    Прежняя строка заканчивалась словами «не пошло N» про этих же людей: лицо с 0 %
    сходства теперь в поиске, и число в подписи обязано это показывать.
    """
    arhiv_s_dvumja_snimkami(window)
    itog = itog_iz(lica_po_fajlam(lico(100, 0), lico(90, 1)))     # косинус 0, но лицо годное
    assert len(itog.otvergnutye) == 0, "отмеченное годное лицо снова куда-то отвергли"
    # предупреждение теперь у ОБОИХ: два непохожих лица не дают ни одному из них
    # права называться «главным» и молчать
    assert [k.slabo for k in itog.karty] == [True, True]
    dialog.otvet = itog

    window._strok[0].knopka.click()

    tekst = window._strok[0].podpis.text()
    assert "лиц 2, снимков 2" in tekst, tekst
    assert "лиц 2" in tekst and "снимков 2" in tekst, tekst
    assert "не похожи на остальных 2" in tekst, \
        f"про слабое сходство не сказано: {tekst}"
    for otkaz in ("не пошло", "не пойдёт"):
        assert otkaz not in tekst, tekst
    assert itog.reference.count == 2


def test_podpis_etalona_govorit_pro_lico_kotoroe_ne_ishem(window, dialog) -> None:
    """Единственный настоящий отказ — непригодный отпечаток, и он назван причиной.

    «Не ищем N» без слов «сравнить нечем» человек прочитал бы как «приложение само
    выбросило моё отмеченное лицо» — ровно то подозрение, из-за которого T14 и начали.
    """
    arhiv_s_dvumja_snimkami(window)
    itog = itog_iz(lica_po_fajlam(lico(100, 0), nulevoe_lico()))
    assert len(itog.otvergnutye) == 1, "тест потерял предмет: показывать нечего"
    dialog.otvet = itog

    window._strok[0].knopka.click()

    tekst = window._strok[0].podpis.text()
    assert "лиц 1" in tekst, tekst
    assert "не ищем 1 — сравнить нечем" in tekst, tekst
    assert "слабо похожих" not in tekst, tekst


def test_podpis_etallona_soglasovana_s_chislom(window, dialog) -> None:
    """«ищем по 1 лицам» — сломанное согласование, и родитель споткнётся о него раньше,
    чем поймёт смысл строки. Форма «лиц 1» верна при любом числе (то же правило, что в
    шапке report.txt)."""
    arhiv_s_dvumja_snimkami(window)
    dialog.otvet = itog_iz(lica_po_fajlam(lico(100, 0)))

    window._strok[0].knopka.click()

    tekst = window._strok[0].podpis.text()
    assert "лицам" not in tekst, f"сломанное согласование: {tekst!r}"
    assert "лиц 1" in tekst and "снимков 1" in tekst, tekst
    assert "слабо похожих" not in tekst, f"пустое число в подписи: {tekst!r}"


def test_bezetalonnyj_otvet_dialoga_ne_stiraet_prezhnij(window, dialog,
                                                        monkeypatch) -> None:
    """ОК без эталона диалог не отдаёт, но и на чужую ошибку окно не имеет права молчать
    дважды: прежний эталон остаётся, а отсутствие нового называется словами."""
    arhiv_s_dvumja_snimkami(window)
    bylo = polozhit_etalon(window, lico(100, 0)).itog
    pokazanno: list = []
    monkeypatch.setattr(window, "_warn", lambda text: pokazanno.append(text))
    dialog.otvet = itog_iz(lica_po_fajlam(nulevoe_lico()))

    window._strok[0].knopka.click()

    assert window.ljudi[0].itog is bylo, "прежний эталон стёрт отказом"
    assert pokazanno, "окно промолчало там, где у человека не собрался эталон"
    tekst = pokazanno[0].lower()
    assert "сравнить нечем" in tekst, pokazanno
    for otkaz in ("не пошло", "не пойдёт"):
        assert otkaz not in tekst, \
            f"подписан отказ, которого больше нет: {pokazanno[0]}"
    assert "прежний" in tekst, f"не сказано, что старый эталон в силе: {pokazanno[0]}"


def test_etalon_iz_dialoga_ne_pishet_v_nakopitel_poter(window, dialog,
                                                       monkeypatch) -> None:
    """Потери архива доезжают до report.txt и после того, как человек выбирал эталон.

    Прежний путь разбирал эталонный файл потоком ОКНА, и его нули могли перечеркнуть
    потери папки — ради этого и жил признак `progon_papki`. Разбор переехал в диалог, и
    окно в свой накопитель не пишет вообще. Сам отчёт проверяется ниже:
    `test_kopirovanie_dostavlyaet_fajly_i_pishet_poteri`.
    """
    arhiv_s_dvumja_snimkami(window)
    window._poteri = {"/bityj.jpg": "файл не читается: cannot identify"}
    window._otbroennye = {Path("/g.jpg"): 2}
    window._povrezhdeniya = 3

    def popal(_photos, _stats) -> None:
        pytest.fail("окно свело потери эталонного разбора к нулю")

    monkeypatch.setattr(window, "_zapisi_poteri", popal)
    dialog.otvet = itog_iz(lica_po_fajlam(lico(100, 0)))

    window._strok[0].knopka.click()

    assert window._poteri == {"/bityj.jpg": "файл не читается: cannot identify"}
    assert window._otbroennye == {Path("/g.jpg"): 2}
    assert window._povrezhdeniya == 3, "счётчик починки оглавления тронут чужим разбором"


def test_poka_dialog_etalona_otkryt_novyj_razbor_ne_nachinetsja(window, bez_skana,
                                                                monkeypatch,
                                                                tmp_path) -> None:
    """Двух живых разборов не бывает — и держится это не только модальностью.

    Модальное окно не отдаёт главному клики мышью, но `start_scan` умеет приехать и из
    кода (таймер полосы настроек, повторный слот). Пока окно эталона открыто, отказывает
    `_mozhno_nachat`, и оглавление остаётся у одного потока. Проверка — на НАСТОЯЩЕМ
    диалоге: подменён только его цикл событий, поэтому отказ проверяется в тот миг, когда
    окно эталона действительно живо.
    """
    from PySide6.QtWidgets import QDialog

    arhiv = tmp_path / "arhiv"
    arhiv.mkdir()
    (arhiv / "a.jpg").write_bytes(b"x")
    window.set_source_folders([arhiv])
    pokazanno: list = []
    monkeypatch.setattr(window, "_warn", lambda text: pokazanno.append(text))
    svidetelstva: list = []

    def vo_vremya_otkrytogo_dialoga(self):
        svidetelstva.append(window._mozhno_nachat())
        window.start_scan()
        svidetelstva.append(list(bez_skana.sozdannye))
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", vo_vremya_otkrytogo_dialoga)

    window._strok[0].knopka.click()

    assert svidetelstva[0] is False, "пока открыт диалог эталона, окно пустило новый разбор"
    assert svidetelstva[1] == [], "второй рабочий поток создан поверх открытого диалога"
    assert pokazanno and "эталона" in pokazanno[0].lower(), pokazanno


def test_zanyatoe_okno_ne_otkryvaet_dialog_etalona(window, dialog, monkeypatch,
                                                   bez_skana) -> None:
    """Просить фото ребёнка, пока архив разбирается, — это второй поток к той же базе во
    время чужого разбора.

    Кнопка в эту минуту выключена, но слот живёт и вызывается из теста (и из горячих
    клавиш), поэтому отказ стоит и в слоте.
    """
    monkeypatch.setattr(window, "_warn", lambda text: None)
    window.files = [Path("/a.jpg")]
    window.start_scan()

    window._strok[0].knopka.click()

    assert dialog.vyzy == [], "диалог открыт, пока окно занято"
    assert len(bez_skana.sozdannye) == 1, "выбор эталона добавил второй поток"


# --- согласование с полосой -----------------------------------------------------------------


def test_razreshenie_iz_fajla_soglasuetsya_s_polosoj(tmp_path, qapp) -> None:
    """Файл настроек пропускает любое число от 600 до 6000, полоса предлагает два. Окно
    обязано взять то, что показывает полоса (1800 -> 2400), и записать обратно: иначе
    первое же движение ползунка увидит «качество разбора изменилось» и попросит
    перебрать весь архив — при каждом запуске, потому что в файл ляжет 1800.
    """
    from ui.main_window import MainWindow
    from utils.config import Settings

    s = Settings(tmp_path / "s.ini")
    s.engine, s.max_dim = "yunet", 1800
    assert s.save()
    win = MainWindow(s, cache_path=tmp_path / "c.sqlite3")
    assert win.settings_bar.max_dim == 2400
    assert win.scan_dim == 2400, "окно помнит сырое 1800 — будет ложный запрос перескана"
    assert win.scan_mode == "yunet" == win.settings_bar.engine
    assert Settings(tmp_path / "s.ini").max_dim == 2400, "в файл не легла исправленная пара"


def test_otkaz_ot_novogo_razbora_otkatyvaet_polosu(window, monkeypatch, qapp,
                                                   bez_skana) -> None:
    """Человек сказал «нет» новому разбору — полоса возвращается к прежнему значению, и
    никакой поток не создаётся. Иначе она бы показывала 1200 при разобранном 2400."""
    window.photos = {Path("/a.jpg"): [lico()]}
    polozhit_etalon(window, lico())
    window.files = [Path("/a.jpg")]
    monkeypatch.setattr(window, "_ask_rescan", lambda mode, max_dim: False)
    window.settings_bar.set_max_dim(1200)
    srazu(qapp)
    assert window.scan_dim == 2400
    assert window.settings_bar.max_dim == 2400, "полосу не откатили — она врёт о разборе"
    assert bez_skana.sozdannye == []


def test_otkaz_ot_novogo_razbora_ne_menjaet_fajn_nastroek(window, monkeypatch,
                                                          bez_skana) -> None:
    """Отказался от перескана — на диске остаётся тот способ, которым реально разбирал.

    Иначе файл настроек разошёлся бы с показанным результатом, и следующий запуск начал
    бы искать другим способом, чем человек отменил.
    """
    monkeypatch.setattr(window, "_ask_rescan", lambda mode, max_dim: False)
    window.photos = {Path("/a.jpg"): [lico()]}
    window.files = [Path("/a.jpg")]
    window.settings_bar.set_engine("insight")
    window._primenit_nastrojki()
    from utils.config import Settings
    assert Settings(window.settings.path).engine == "both"


def test_soglasie_na_novyj_razbor_zapuskaet_skany(window, monkeypatch, bez_skana) -> None:
    """Качество разбора меняет отпечатки: после согласия старый набор лиц недействителен,
    и окно обязано повести человека за новым разбором, а не молча пересчитывать старое.

    Второй проверяемый факт — способ поиска доезжает до файла настроек: без этого
    человек выбрал бы «Быстрее», окно бы его запомнило, а следующий запуск молча
    вернулся бы к «Оба».
    """
    monkeypatch.setattr(window, "_ask_rescan", lambda mode, max_dim: True)
    window.files = [Path("/a.jpg")]
    window.photos = {Path("/a.jpg"): [lico()]}    # папка уже разобрана
    polozhit_etalon(window, lico())
    window.settings_bar.set_engine("insight")
    window._primenit_nastrojki()
    from utils.config import Settings
    assert Settings(window.settings.path).engine == "insight", "настройки не дописались"
    assert window.scan_mode == "insight"
    assert len(bez_skana.sozdannye) == 1, "смена режима обязана привести разбор в движение"
    assert window.worker is bez_skana.sozdannye[0]
    assert window.photos == {}, "новый разбор обязан снять старые находки"
    assert window.engine is not None and window.engine.name == "insight", \
        "смена режима — смена движка: старый объект остаться не должен"


def test_rezhim_do_pervogo_razbora_tolko_zapominaetsya(window, bez_skana) -> None:
    """До первого разбора смена способа поиска не имеет права запускать скан: человек
    ещё не нажимал «Начать разбор папки», а минуты и 290 МБ уже пошли бы."""
    window.files = [Path("/a.jpg")]
    window.settings_bar.set_engine("insight")
    window._primenit_nastrojki()
    assert window.scan_mode == "insight"
    assert bez_skana.sozdannye == []
    assert window.worker is None


def test_zanyatoe_okno_ne_stiraet_pokazannoe(window, monkeypatch, bez_skana) -> None:
    """Второй клик «Начать разбор» посреди прогона не имеет права оставить человека с
    пустой сеткой: `_run` отказывает молча, а очистка находок до этой отказа уже
    уничтожила бы результат первого разбора."""
    najt = {Path("/a.jpg"): [lico()], Path("/b.jpg"): [lico()]}
    window.photos = dict(najt)
    polozhit_etalon(window, lico())
    window.files = list(najt)
    window.rebuild_scores()
    assert window.results.counts[0] == 2
    pokazanno: list = []
    monkeypatch.setattr(window, "_warn", lambda text: pokazanno.append(text))
    window._set_busy(True, "разбираем папку")
    window.start_scan()
    assert window.photos == najt, "находки стёрты, хотя новый разбор не начинался"
    assert pokazanno == ["разбор уже идёт, дождитесь или нажмите «Остановить»"]
    assert bez_skana.sozdannye == []


# --- занятость: полоса настроек посреди разбора ----------------------------------------------


def test_polosa_nastroek_zablokirovana_posredi_razbora(window, bez_skana,
                                                       monkeypatch) -> None:
    """Пока разбирается папка, настройку не двигают мышью.

    Разбор архива — минуты, и всё это время кнопка «Начать разбор» выключена, а полоса
    настроек была жива: человек успевал переключить «качество разбора», окно молча
    записывало новое число себе в `scan_dim` и в файл настроек, а новый разбор при этом
    НЕ начинался. Сетка в конце получала рамки старого разрешения на кадрах нового —
    зелёная рамка уезжала с лица. См. `test_dvizhenie_polosy_posredi_razbora...`.
    """
    monkeypatch.setattr(window, "_warn", lambda text: None)
    window.files = [Path("/a.jpg")]
    assert window.settings_bar.isEnabled() is True
    window.start_scan()
    assert window.settings_bar.isEnabled() is False
    assert window.settings_bar._engine_box.isEnabled() is False
    assert window.settings_bar._dim_box.isEnabled() is False
    assert window.settings_bar.clear_button.isEnabled() is False

    gotovo = bez_skana.sozdannye[0]
    gotovo.done.emit(({}, ScanStats(scanned=1, cached=0, seconds=1.0)))
    assert window.settings_bar.isEnabled() is True, "разбор кончился, а полоса осталась закрытой"


def test_dvizhenie_polosy_posredi_razbora_nichego_ne_kommitit(window, bez_skana,
                                                             monkeypatch, qapp) -> None:
    """Программный ход полосой посреди разбора не имеет права ничего изменить.

    `setEnabled(False)` отсекает мышь, но не `setCurrentIndex`: сигнал `changed` может
    прийти и из кода, и из события, вставшего в очередь за миг до блокировки. Окно
    обязано отказаться применять настройку ДО того, как тронуты `scan_mode`, `scan_dim`
    и файл настроек, — иначе `rebuild_scores` отдаст сетке новый `scan_dim` поверх
    кадров, разобранных в старом, и рамка ляжет мимо лица.
    """
    from utils.config import Settings

    monkeypatch.setattr(window, "_warn", lambda text: None)
    window.files = [Path("/a.jpg")]
    window.start_scan()
    potok = bez_skana.sozdannye[0]
    dvizhok = window.engine
    assert dvizhok is not None and dvizhok.name == "both"

    window.settings_bar.set_max_dim(1200)
    window.settings_bar.set_engine("insight")
    srazu(qapp)                                   # дебаунс сработал ПОСРЕДИ разбора

    assert window.scan_dim == 2400, "новое качество записано в идущий разбор"
    assert window.scan_mode == "both"
    assert potok.max_dim == 2400, "разбор идёт в одном качестве, а окно помнит другое"
    assert window.engine is dvizhok, "движок сброшен посреди разбора"
    s = Settings(window.settings.path)
    assert s.max_dim == 2400 and s.engine == "both", "настройки записаны посреди разбора"
    assert len(bez_skana.sozdannye) == 1, "смена настройки запустила второй разбор"
    # Полосу НЕ откатываем: её текущий вид и есть просьба человека, а откатить её —
    # значит молча выкинуть то, о чём он попросил за миг до старта разбора.
    assert window.settings_bar.max_dim == 1200 and window.settings_bar.engine == "insight"
    # ...и по окончании разбора окно спрашивает, а не хранит обмолчку
    sprashivali: list = []
    monkeypatch.setattr(window, "_ask_rescan",
                        lambda rezhim, razmer: sprashivali.append((rezhim, razmer)) or False)
    potok.done.emit(({Path("/a.jpg"): [lico()]}, ScanStats(scanned=1, cached=0, seconds=1.0)))
    srazu(qapp)
    assert sprashivali == [("insight", 1200)], "просьбу, поданную посреди разбора, выкинули"
    assert window.scan_dim == 2400 and window.scan_mode == "both", "отказ не откатил полосу"
    assert window.settings_bar.max_dim == 2400, "после отказа полоса врёт о разборе"
    assert len(bez_skana.sozdannye) == 1, "отказ обязан отменить и новый разбор"


def test_chistka_oglavleniya_posredi_razbora_otkazana(window, bez_skana,
                                                      monkeypatch) -> None:
    """«Очистить оглавление» посреди разбора — это «разбор вот-вот соврёт про потери».

    Рабочий поток держит соединение с базой и пишет в неё каждый снимок. Очистка от
    человека, нажавшего кнопку в ту же секунду, оставляла пустой показ и сообщение
    «оглавление очищено: удалено записей N», а `done` возвращал на экран весь архив
    поверх этого обещания — с числом потерь от уже удалённого оглавления.
    """
    pokazanno: list = []
    monkeypatch.setattr(window, "_warn", lambda text: pokazanno.append(text))
    window.files = [Path("/a.jpg")]
    window.start_scan()
    window.photos = {Path("/a.jpg"): [lico()]}
    window._clear_cache()
    assert window.photos != {}, "находки стёрты под живым потоком"
    assert pokazanno and "разбор" in pokazanno[0]


# --- сигнал рабочего потока доезжает до окна -----------------------------------------------


def test_signaly_potoka_dovedeny_do_okna_progress(window, bez_skana, monkeypatch) -> None:
    """Спецификация, раздел 6: «обработано 47 из 131».

    Единственное место, где эта строка появляется, — слот `_on_progress`, а приходит он
    сигналом `progress`. Пока подменённый поток имел заглушку вместо сигнала, ни один
    тест по этому пути не ходил, и удаление `connect` не заметил бы ни один тест.
    """
    monkeypatch.setattr(window, "_warn", lambda text: None)
    window.files = [Path("/a.jpg")]
    window.start_scan()
    assert window.worker is bez_skana.sozdannye[0]

    window.worker.progress.emit(47, 131, "IMG_0047.jpg")
    assert window.progress.maximum() == 131
    assert window.progress.value() == 47
    assert window.status_label.text() == "обработано 47 из 131: IMG_0047.jpg"


def test_signaly_potoka_dovedeny_do_okna_oshibka(window, bez_skana, monkeypatch) -> None:
    """`error` наружу = то же сообщение слово в слово, занятость снята, сетка цела."""
    window.photos = {Path("/a.jpg"): [lico()]}
    polozhit_etalon(window, lico())
    window.files = [Path("/a.jpg")]
    window.rebuild_scores()
    bylo = window.results.counts
    pokazanno: list = []
    monkeypatch.setattr(window, "_warn", lambda text: pokazanno.append(text))

    window.worker = None
    window.start_scan()
    window.worker.error.emit(SBVOI_OGLAVLENIYA)

    assert pokazanno == [SBVOI_OGLAVLENIYA], "сообщение потока переписано окном"
    assert window.results.counts == bylo, "сетку на ошибке очищать нельзя"
    assert window.stop_button.isEnabled() is False
    assert window._strok[0].knopka.isEnabled() is True


def test_signaly_potoka_dovedeny_do_okna_gotovoj_rezultat(window, bez_skana,
                                                          monkeypatch) -> None:
    """`done` наружу = находки в памяти, находки в сетке и снятая занятость."""
    monkeypatch.setattr(window, "_warn", lambda text: None)
    window.photos = {}
    polozhit_etalon(window, lico())
    window.files = [Path("/a.jpg")]
    window.start_scan()
    window.worker.done.emit(({Path("/a.jpg"): [lico()]},
                             ScanStats(scanned=1, cached=0, seconds=2.0)))
    assert Path("/a.jpg") in window.photos
    assert window.results.counts[0] == 1
    assert window.stop_button.isEnabled() is False


# --- накопители потерь живут ровно один разбор ------------------------------------------------


def test_povtornoe_chtenie_snimka_snimaet_otbroshennye_lica(window,
                                                             bez_skana) -> None:
    """Лицо, отбракованное однажды, не имеет права всплывать в каждом следующем отчёте.

    `failures` вычищается по пути снимка, который на этом прогоне разобрался, а
    `dropped_faces` в прежнем коде только добавлялся: «лиц отброшено: 2» кочевало из
    report.txt в report.txt, даже когда снимок перечитали и всё в нём в порядке.
    """
    put = Path("/g.jpg")
    window.files = [put]
    window._on_scan_done(({put: [lico()]},
                          ScanStats(scanned=1, cached=0, seconds=1.0,
                                    dropped_faces={put: 2}, cache_damage=3)))
    assert window._otbroennye == {put: 2}

    window._on_scan_done(({put: [lico()]}, ScanStats(scanned=1, cached=0, seconds=0.4)))
    assert window._otbroennye == {}, "снимок перечитан чисто, а отчёт всё ещё винит его"


def test_teplyj_progon_prinosit_poteri_iz_oglavleniya(window,
                                                                bez_skana) -> None:
    """Тёплый прогон (всё из оглавления) обязан принести те же потери, а не ноль.

    Прежняя связка «окно чистит запись по каждому присланному снимку» + «модели на
    тёплом прогоне не считали потерь» давала второй запуск с пустым отчётом о потерях:
    «потерь нет: ни один файл не пропущен» об архив, где лица уже теряли. Число теперь
    приходит вместе со снимком из оглавления, и окно обязано его не выбросить.

    Отдельно проверяется порядок «почистили — добавили»: снимок, который на этом
    прогоне разобрался ЧИСТО, всё равно обязан выйти из потерь (тест выше), иначе
    первая assertion-строка превратилась бы в вечное «лиц отброшено».
    """
    put = Path("/g.jpg")
    chistyj = Path("/chistyj.jpg")
    window.files = [put, chistyj]
    window._on_scan_done(({put: [lico()]},
                          ScanStats(scanned=1, cached=0, seconds=1.0,
                                    dropped_faces={put: 2})))
    assert window._otbroennye == {put: 2}

    # второй запуск: оба снимка легли из оглавления, потерь там 2 и 0
    window._on_scan_done(({put: [lico()], chistyj: [lico()]},
                          ScanStats(scanned=0, cached=2, seconds=0.1,
                                    dropped_faces={put: 2})))
    assert window._otbroennye == {put: 2}, "тёплый прогон стёр потери оглавления"

    # снимок, который на этом прогоне перечитан заново и чисто, из потерь выходит
    window._on_scan_done(({put: [lico()], chistyj: [lico()]},
                          ScanStats(scanned=1, cached=1, seconds=0.5)))
    assert window._otbroennye == {}


def test_chistyj_razbor_papki_snimaet_povrezhdeniya_oglavleniya(window,
                                                                bez_skana) -> None:
    """Числа потерь относятся к разбору, который их принёс, а не ко всей жизни окна.

    Счётчик повреждённых строк оглавления — про починку: после первого же пересчёта он
    обязан замолчать, иначе через неделю человек прочитает в отчёте про беду, которой
    давно нет.
    """
    put = Path("/g.jpg")
    window.files = [put]
    window.start_scan()
    window.worker.done.emit(({}, ScanStats(scanned=1, cached=0, seconds=1.0,
                                           cache_damage=4)))
    assert window._povrezhdeniya == 4
    window.start_scan()
    window.worker.done.emit(({}, ScanStats(scanned=1, cached=0, seconds=0.5)))
    assert window._povrezhdeniya == 0, "чистый разбор обязан обнулить старую починку"


def test_chistyj_progon_snimaet_poteri_predydushhego_okna(window, bez_skana) -> None:
    """Единственный прогон, который пишет в накопитель, — прогон папки (Spec, раздел 6).

    Раньше здесь стоял тест на «догляд за одним эталонным снимком»: он не имел права
    стирать потери архива, и для этого у `_run` жил признак `progon_papki`. Эталон
    разбирает СВОЙ поток внутри диалога (`ui/reference_dialog.py`), и окно о его потерях
    не узнаёт вовсе — ветка `progon_papki=False` удалена вместе с признаком.

    Проверяем то, что осталось: прогон папки перекрывает `cache_damage` числом ЭТОГО
    прогона, а не копит сумму за жизнь окна, и чистый разбор не оставляет старых потерь.
    Два `_on_scan_done` подряд без `start_scan` — не фантазия, а то, как накопитель
    выглядел бы изнутри: `start_scan` обнуляет счётчик сам, и без этой пары тест не
    отличал бы «перекрываю» от «прибавляю».
    """
    put = Path("/g.jpg")
    window.files = [put]
    window._on_scan_done(({put: [lico()]}, ScanStats(
        scanned=1, cached=0, seconds=1.0, failures=[("/starsij.jpg", "файл не читается")],
        dropped_faces={put: 2}, cache_damage=5)))
    assert window._poteri == {"/starsij.jpg": "файл не читается"}
    assert window._povrezhdeniya == 5

    window._on_scan_done(({put: [lico()]}, ScanStats(scanned=1, cached=0, seconds=0.6,
                                                      cache_damage=7)))
    assert window._povrezhdeniya == 7, "счётчик починки обязан перекрываться, а не копиться"

    window.start_scan()
    window.worker.done.emit(({put: [lico()]}, ScanStats(scanned=1, cached=1, seconds=0.3)))
    assert window._poteri == {}, "перечитанный снимок обязан выйти из потерь"
    assert window._povrezhdeniya == 0, "чистый прогон обнуляет старую починку"


# --- закрытие окна -------------------------------------------------------------------------------


def test_zakrytie_poka_potok_zhiv_ne_zakryvaet_oglavlenie(window, bez_skana,
                                                          monkeypatch) -> None:
    """Закрытие окна с живым рабочим потоком: `wait` имел результат, но не имел цены.

    Модели читаются 13–21 с, и `scan_photos` во время чтения флаг отмены не проверяет,
    поэтому поток переживает три секунды ожидания почти всегда. Прежнее `closeEvent`
    закрывало соединение с базой не глядя, следующий `cache.get` в потоке падал с
    `ProgrammingError: Cannot operate on a closed database`, и человек видел
    «оглавление недоступно» в окне, которое уже закрыто.
    """
    from PySide6.QtGui import QCloseEvent

    zakryto: list = []
    monkeypatch.setattr(window.cache, "close", lambda: zakryto.append(1))
    monkeypatch.setattr(window, "_warn", lambda text: None)
    window.files = [Path("/a.jpg")]
    window.start_scan()
    potok = window.worker
    potok.zanimaet = True
    potok.otsvet_na_wait = False

    sobytie = QCloseEvent()
    window.closeEvent(sobytie)
    assert sobytie.isAccepted() is False, "окно закрыто под живым потоком"
    assert potok.stop_zyvali is True, "поток не попросили остановиться"
    assert potok.prosili_wait, "у потока не спросили, встал ли он"
    assert zakryto == [], "база оглавления закрыта, пока поток к ней обращается"
    assert "останав" in window.status_label.text().lower(), window.status_label.text()

    # поток встал — окно закрывается само и только тогда освобождает оглавление
    window.show()
    potok.zanimaet = False
    potok.finished.emit()
    assert zakryto == [1], "оглавление так и не закрылось после остановки потока"


def test_povtornoe_zakrytie_ne_eshet_tri_sekundy(window, bez_skana, monkeypatch) -> None:
    """Второе «закрыть» в ожидании потока не вешает интерфейс ещё на три секунды.

    Нетерпеливый человек нажимает Cmd+W снова, а `wait(3000)` в потоке интерфейса —
    это ровно тот вид зависшего окна, которого спецификация (раздел 7) не допускает.
    Отложенное закрытие уже назначено, поэтому второй заход только отменяет событие.
    """
    from PySide6.QtGui import QCloseEvent

    zakryto: list = []
    monkeypatch.setattr(window.cache, "close", lambda: zakryto.append(1))
    monkeypatch.setattr(window, "_warn", lambda text: None)
    window.files = [Path("/a.jpg")]
    window.start_scan()
    potok = window.worker
    potok.zanimaet = True
    potok.otsvet_na_wait = False

    window.closeEvent(QCloseEvent())
    sok = len(potok.prosili_wait)
    assert sok == 1
    vtoree = QCloseEvent()
    window.closeEvent(vtoree)
    assert vtoree.isAccepted() is False
    assert len(potok.prosili_wait) == sok, "ждём поток повторно — окно снова висит"
    assert zakryto == []


def test_zakrytie_svobodnogo_okna_zakryvaet_oglavlenie(window, monkeypatch) -> None:
    """Окно без разбора закрывается сразу: ждать нечего, база освобождается."""
    from PySide6.QtGui import QCloseEvent

    zakryto: list = []
    monkeypatch.setattr(window.cache, "close", lambda: zakryto.append(1))
    sobytie = QCloseEvent()
    window.closeEvent(sobytie)
    assert sobytie.isAccepted() is True
    assert zakryto == [1]
    # второй заход (окно уже закрывают повторно) не имеет права падать на закрытой базе
    window.closeEvent(QCloseEvent())
    assert zakryto == [1]


def test_zakrytie_sbrosivaet_ozhidajuschij_debaun(window) -> None:
    """Число, выставленное за миг до закрытия, не имеет права вернуться на место.

    Полоса шлёт изменение, а окно применяет его через 150 мс (шторм перетаскивания).
    Человек успел отпустить мышь и закрыть окно — в файле оставалось старое число, и
    на следующем запуске он видел бы «своё» значение, сдвинутое назад.
    """
    from PySide6.QtGui import QCloseEvent

    from utils.config import Settings

    window.settings_bar.set_threshold(0.52)
    assert window.settings.threshold == pytest.approx(0.38), "дебаунс обязан ещё молчать"

    window.closeEvent(QCloseEvent())

    assert window.settings.threshold == pytest.approx(0.52)
    assert Settings(window.settings.path).threshold == pytest.approx(0.52)


def test_okno_derzhit_modulejnuju_ssylku_do_zavershenija_potoka() -> None:
    """`src/main.py` держит окно на уровне модуля: сборщик не имеет права его съесть.

    Окно — родитель рабочего потока. Если единственный указатель на него — локальная
    переменная `main()`, объект умирает вместе с возвратом из функции, а вместе с ним
    и `QThread`, который ещё крутится: «QThread: Destroyed while thread is still
    running» в консоли и недописанное оглавление в папке пользователя.
    """
    import ast

    ishodnik = Path(__file__).resolve().parents[1] / "src" / "main.py"
    derevo = ast.parse(ishodnik.read_text(encoding="utf-8"))
    tochka_vhoda = next(u for u in derevo.body
                        if isinstance(u, ast.FunctionDef) and u.name == "main")
    obeyavlennye = {im for u in ast.walk(tochka_vhoda)
                    if isinstance(u, ast.Global) for im in u.names}
    assert obeyavlennye, "окно в main() держится локальной переменной — поток переживёт окно"
    prisvoennye = {t.id for u in ast.walk(tochka_vhoda) if isinstance(u, ast.Assign)
                   for t in u.targets if isinstance(t, ast.Name)}
    assert obeyavlennye & prisvoennye, "модульная ссылка объявлена, но не присвоена"


# --- вывод: копирование и отчёт ---------------------------------------------------------------

# Виджеты, которые обязаны гаснуть, когда окно занято: всё, чем можно что-то запустить
# или изменить. `stop_button` в этом списке особенная: она жива только посреди разбора.
AKTIVNYE_VIDEJETY = ("add_folder_button", "clear_folders_button", "scan_button",
                     "add_person_button", "stop_button", "copy_button",
                     "settings_bar", "results")


def _sostojanie_vseh(window) -> dict:
    """Жив ли каждый из виджетов списка прямо сейчас.

    Кнопка выбора фото живёт не на окне, а в строке человека, и её имена меняются вместе
    с числом людей: добавляем её в снимок отдельно, иначе «окно гасит все кнопки»
    проверялось бы без главной кнопки шага 2.
    """
    sostojanie = {im: getattr(window, im).isEnabled() for im in AKTIVNYE_VIDEJETY}
    sostojanie["knopka_cheloveka"] = window._strok[0].knopka.isEnabled()
    return sostojanie


def _okno_gotovo_k_kopirovaniyu(window, monkeypatch, kuda: Path) -> None:
    """Два отмеченных снимка и выбранная папка результатов — всё, чего ждёт `copy_marked`.

    Разбор не запускается: находки кладутся в `photos` руками, а `copy_photos` каждый
    тест подменяет сам, потому что именно в момент этого вызова окно и занято.

    `_set_busy(False, "")` в конце — то самое состояние, в которое окно приходит после
    настоящего `done`: все виджеты включены по одному признаку занятости, и тест сверяет
    «до» и «после» копирования с настоящим снимком, а не с наполовину собранным окном.
    """
    import ui.main_window as modul

    monkeypatch.setattr(modul.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(kuda)))
    window.photos = {Path("/a.jpg"): [lico()], Path("/b.jpg"): [lico()]}
    # Список файлов — как после настоящего выбора папки: иначе «Начать разбор» выключено
    # и в свободном окне, и тест путал бы «окно занято» с «окну нечего начинать».
    # `folders` рядом обязателен: «Убрать все» живёт при непустом списке папок, и без
    # этой строки помощник рисовал бы окно с файлами, но без папок, — состояние,
    # которого в приложении не бывает.
    window.folders = [Path("/arhiv")]
    window.files = list(window.photos)
    polozhit_etalon(window, lico())
    window.settings_bar.set_threshold(0.30)
    window.rebuild_scores()
    window.results.check_all()
    window._set_busy(False, "")
    assert window.results.checked == [Path("/a.jpg"), Path("/b.jpg")]


def test_kopirovanie_gasit_vse_knopki_i_govorit_chto_proishodit(window, tmp_path,
                                                                monkeypatch) -> None:
    """`copy_photos` идёт в потоке интерфейса и на сотне телефонных снимков занимает
    минуты. В эту минуту ни один виджет не имеет права выглядеть живым, а строка
    состояния обязана сказать, что именно происходит и куда (спецификация, раздел 7 —
    «без вида зависшего окна»)."""
    import ui.main_window as modul

    pokaz: dict = {}

    def sprosit_v_moment_kopirovaniya(fajly, dest, poteri):
        pokaz.update(_sostojanie_vseh(window))
        pokaz["status"] = window.status_label.text()
        return [(f, Path(dest) / f.name) for f in fajly]

    _okno_gotovo_k_kopirovaniyu(window, monkeypatch, tmp_path / "kuda")
    do = _sostojanie_vseh(window)
    monkeypatch.setattr(modul, "copy_photos", sprosit_v_moment_kopirovaniya)
    window.copy_marked()

    assert do["knopka_cheloveka"] and do["copy_button"], "тест ловит только живое окно"
    zivye = [im for im in AKTIVNYE_VIDEJETY if pokaz[im]]
    assert zivye == [], f"окно притворяется живым: {zivye}"
    assert pokaz["status"].startswith("копируем 2 фото"), pokaz["status"]
    assert "kuda" in pokaz["status"], "не сказано, куда летят снимки"
    # и после отпускает ровно то, что было живым до
    posle = _sostojanie_vseh(window)
    assert {im: posle[im] for im in do} == do, f"окно не вернулось: {posle}"
    assert not posle["stop_button"], "кнопка «Остановить» ожила без рабочего потока"
    assert window._zanyat is False and window._kopiruetsya is False


def test_zanyatost_vyhozhit_na_ekran_do_blokirovki(window, tmp_path,
                                                   monkeypatch) -> None:
    """Строка «копируем…» обязана ДОЙТИ до экрана, а не встать в очередь за блокирующим
    вызовом: без принудительной отрисовки человек прочитал бы её уже после того, как
    всё скопировалось, — то есть ровно тогда, когда она бесполезна."""
    import ui.main_window as modul

    poryadok: list = []
    monkeypatch.setattr(window, "_otrisovat_okno",
                        lambda: poryadok.append("otrisovka"))
    monkeypatch.setattr(modul, "copy_photos",
                        lambda f, d, p: poryadok.append("kopirovanie") or [])
    _okno_gotovo_k_kopirovaniyu(window, monkeypatch, tmp_path / "kuda")
    window.copy_marked()

    assert poryadok == ["otrisovka", "kopirovanie"], \
        f"отрисовка занятого окна идёт после блокировки: {poryadok}"


def test_neudavsheesja_kopirovanie_otpuskaet_okno_do_soobshchenija(
        window, tmp_path, monkeypatch) -> None:
    """Причина промаха показывается ОТПУЩЕННОМУ окну. Модальное окно внутри занятого
    — это просьба решить что-то, когда кнопки ещё выключены, и держится это на `finally`:
    без неё `OSError` оставил бы окно серым навсегда."""
    import ui.main_window as modul

    sostojanie_v_warn: dict = {}

    def pomestit(_text):
        sostojanie_v_warn.update(_sostojanie_vseh(window))

    def otazhe(fajly, dest, poteri):
        raise OSError(30, "Read-only file system")

    _okno_gotovo_k_kopirovaniyu(window, monkeypatch, tmp_path / "kuda")
    monkeypatch.setattr(modul, "copy_photos", otazhe)
    monkeypatch.setattr(window, "_warn", pomestit)
    window.copy_marked()

    assert sostojanie_v_warn, "о промахе копирования не сказали"
    zhivy = [im for im in AKTIVNYE_VIDEJETY
             if im != "stop_button" and not sostojanie_v_warn[im]]
    assert zhivy == [], f"окно показало причину занятым: {zhivy}"
    assert not sostojanie_v_warn["stop_button"]
    assert window._zanyat is False


def test_kopirovanie_ne_obeshchaet_knopku_ostanovit(window, monkeypatch) -> None:
    """Занятость занятостью рознь: посреди разбора человек вправе попросить поток встать,
    посреди копирования такой кнопки на экране нет, и обещать её — значит отправлять
    человека её искать."""
    pokazanno: list = []
    monkeypatch.setattr(window, "_warn", lambda text: pokazanno.append(text))

    window._set_busy(True, "копируем 2 фото…", kopirovanie=True)
    assert window._mozhno_nachat() is False
    window._set_busy(False, "")

    window._set_busy(True, "разбираем папку")
    assert window._mozhno_nachat() is False
    window._set_busy(False, "")

    assert len(pokazanno) == 2, pokazanno
    assert "Остановить" not in pokazanno[0], pokazanno[0]
    assert "копирован" in pokazanno[0], pokazanno[0]
    assert "Остановить" in pokazanno[1], pokazanno[1]
    # и свободное окно разбор начать пустит
    assert window._mozhno_nachat() is True
    assert len(pokazanno) == 2, "свободное окно получило отказ"


def test_podgotovka_rezultata_dlya_kopirovaniya(window, qapp) -> None:
    window.photos = {Path("/a.jpg"): [lico()]}
    polozhit_etalon(window, lico())
    window.settings_bar.set_threshold(0.30)
    srazu(qapp)
    window.results.check_all()
    assert window.results.checked == [Path("/a.jpg")]


def test_kopirovanie_dostavlyaet_fajly_i_pishet_poteri(window, tmp_path, monkeypatch,
                                                      qapp) -> None:
    """Сквозная проверка вывода: три позиционных аргумента `copy_photos`, настоящие
    файлы, причины потерь и счётчики отбраковки внутри report.txt.

    Без `poteri` отчёт печатал бы «скопировано 2» и молчал про третий файл, а без
    `dropped_faces`/`cache_damage` честно писал бы «нет данных» в живом приложении,
    хотя эти числа в `ScanStats` есть.
    """
    from PIL import Image

    import ui.main_window as modul

    arhiv = tmp_path / "arhiv"
    arhiv.mkdir()
    for imya in ("a.jpg", "b.jpg", "c.jpg"):
        Image.new("RGB", (60, 60), (7, 7, 7)).save(arhiv / imya, "JPEG")
    najt = {arhiv / imya: [lico()] for imya in ("a.jpg", "b.jpg", "c.jpg")}
    window.set_source_folders([arhiv])
    window.photos = najt
    polozhit_etalon(window, lico())
    statistika = ScanStats(
        scanned=3, cached=0, seconds=1.0,
        failures=[(str(arhiv / "d.jpg"), "файл не читается: cannot identify image file")],
        dropped_faces={arhiv / "b.jpg": 2}, cache_damage=1)
    window._on_scan_done((najt, statistika))
    srazu(qapp)
    window.results.check_all()

    kuda = tmp_path / "kuda"
    monkeypatch.setattr(modul.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(kuda)))
    window.copy_marked()

    assert len(list(kuda.glob("*.jpg"))) == 3, list(kuda.iterdir())
    otchyot = (kuda / "report.txt").read_text(encoding="utf-8")
    assert "лиц отброшено: 2" in otchyot, "число отбракованных лиц не доехало до отчёта"
    assert "строк оглавления удалено: 1" in otchyot
    assert "нет данных" not in otchyot, "живое приложение не имеет права печатать «нет данных»"
    assert "файл не читается" in otchyot
    assert "скопировано 3 фото" in window.status_label.text()
    assert (kuda / "results.csv").exists()


def test_kopirovanie_peredaet_treti_argument_i_chitaet_poteri(window, tmp_path,
                                                              monkeypatch) -> None:
    """Двухаргументный вызов `copy_photos` — TypeError. Окно зовёт тремя ПОЗИЦИОННЫМИ
    аргументами и складывает накопитель потерь в отчёт и в статус."""
    import ui.main_window as modul

    zvonki: list = []

    def shpion(*args, **kwargs):
        zvonki.append((len(args), len(kwargs)))
        args[2].append((Path("/net.jpg"), "копирование не удалось: [Errno 28]"))
        return [(Path("/a.jpg"), Path("/kuda/a.jpg"))]

    monkeypatch.setattr(modul, "copy_photos", shpion)
    monkeypatch.setattr(modul.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp_path / "kuda")))
    window.photos = {Path("/a.jpg"): [lico()]}
    polozhit_etalon(window, lico())
    window.settings_bar.set_threshold(0.30)
    window.rebuild_scores()
    window.results.check_all()
    window.copy_marked()

    assert zvonki == [(3, 0)], f"аргументы ушли не тремя позиционными: {zvonki}"
    assert "не скопировано 1" in window.status_label.text()
    otchyot = (tmp_path / "kuda" / "report.txt").read_text(encoding="utf-8")
    assert "копирование не удалось" in otchyot, "причина потери пропала из отчёта"


def test_readonly_papka_kopirovaniya_ne_ronyaet_okno(window, tmp_path,
                                                     monkeypatch) -> None:
    """`copy_photos` бросает OSError, если папка результатов не создалась вовсе
    (read-only том, сетевой диск отвалился). Ловим именно `OSError`: `PermissionError`
    — лишь один подвид, а `NotADirectoryError` и `OSError(30)` — другие."""
    import ui.main_window as modul

    def otazhe(*args, **kwargs):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(modul, "copy_photos", otazhe)
    monkeypatch.setattr(modul.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp_path / "kuda")))
    pokazanno: list = []
    monkeypatch.setattr(window, "_warn", lambda text: pokazanno.append(text))
    window.photos = {Path("/a.jpg"): [lico()]}
    polozhit_etalon(window, lico())
    window.settings_bar.set_threshold(0.30)
    window.rebuild_scores()
    window.results.check_all()
    window.copy_marked()                          # без try/except здесь было бы падение

    assert pokazanno and "Read-only" in pokazanno[0]
    assert not (tmp_path / "kuda" / "report.txt").exists()


def test_perezapisi_nastroek_net_no_okno_govorit(window, monkeypatch) -> None:
    """`Settings.save()` возвращает bool, и False нельзя выбросить: человек видит свою
    настройку на экране и узнаёт о промахе только в следующем запуске."""
    papka = window.settings.path.parent
    (papka / "a.jpg").write_bytes(b"x")
    monkeypatch.setattr(window.settings, "save", lambda: False)
    soobshcheniya: list = []
    monkeypatch.setattr(window, "_warn", lambda text: soobshcheniya.append(text))

    window.set_source_folders([papka])

    assert soobshcheniya and "настройк" in soobshcheniya[0].lower()
    assert "не сохраняются" in window.status_label.text()
    # модальное окно на каждое движение мыши — наказание; предупреждаем один раз,
    # а в строку состояния пишем каждый промах
    window.set_source_folders([papka])
    assert len(soobshcheniya) == 1
    assert "не сохраняются" in window.status_label.text()


def test_chistka_oglavleniya_sbrasyuet_nakoplennoe(window) -> None:
    """«Очистить оглавление» обещает начать с нуля: значит и накопленные за прогон
    потери, и показанные результаты обнуляются вместе с базой."""
    window.photos = {Path("/a.jpg"): [lico()]}
    window._poteri["/a.jpg"] = "файл не читается: прочее"
    window._otbroennye[Path("/a.jpg")] = 1
    window._povrezhdeniya = 2
    window._clear_cache()
    assert window.photos == {} and window._poteri == {} and window._otbroennye == {}
    assert window._povrezhdeniya == 0
    assert "оглавление" in window.status_label.text()


def test_neudavshajasja_chistka_ne_chistit_pokazannoe(window, monkeypatch) -> None:
    """База ответила отказом — значит оглавление НЕ чисто, и показанное трогать нельзя.

    Без обёртки исключение утонуло бы в слоте Qt: кнопка молча не сделала бы ничего, а
    человек решил бы, что чистка прошла.
    """
    def otazhe() -> int:
        raise OSError("database is locked")

    monkeypatch.setattr(window.cache, "clear", otazhe)
    window.photos = {Path("/a.jpg"): [lico()]}
    pokazanno: list = []
    monkeypatch.setattr(window, "_warn", lambda text: pokazanno.append(text))
    window._clear_cache()
    assert pokazanno and "не очистилось" in pokazanno[0]
    assert window.photos != {}, "находки стёрты, хотя оглавление осталось целым"


def test_nezapisannyj_otchyot_ne_otricayet_kopirovanie(window, tmp_path,
                                                       monkeypatch) -> None:
    """Отчёт не записался, а снимки уже в папке: сказать надо ОБА факта.

    Иначе человек решит, что не сделал ничего, и повторит копирование поверх — а это
    вторая пачка тех же файлов в его семейной папке.
    """
    import ui.main_window as modul

    def otazhe(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(modul, "write_report", otazhe)
    monkeypatch.setattr(modul.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp_path / "kuda")))
    pokazanno: list = []
    monkeypatch.setattr(window, "_warn", lambda text: pokazanno.append(text))
    a = tmp_path / "a.jpg"
    a.write_bytes(b"x")                       # настоящий файл: копирование должно удаться
    window.photos = {a: [lico()]}
    polozhit_etalon(window, lico())
    window.settings_bar.set_threshold(0.30)
    window.rebuild_scores()
    window.results.check_all()
    window.copy_marked()
    assert pokazanno and "фото скопированы" in pokazanno[0]
    assert "скопировано 1 фото" in window.status_label.text()
    assert not (tmp_path / "kuda" / "results.csv").exists()


def test_nezapisannye_nastrojki_vidny_v_itoge(window, tmp_path, monkeypatch) -> None:
    """Последняя строка статуса не должна перекрывать сообщение о незаписанных
    настройках: `status_label` один, а молчание здесь — обещание, которое не выполнили."""
    import ui.main_window as modul

    monkeypatch.setattr(window.settings, "save", lambda: False)
    monkeypatch.setattr(modul.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp_path / "kuda")))
    monkeypatch.setattr(window, "_warn", lambda text: None)
    a = tmp_path / "a.jpg"
    a.write_bytes(b"x")
    window.photos = {a: [lico()]}
    polozhit_etalon(window, lico())
    window.settings_bar.set_threshold(0.30)
    window.rebuild_scores()
    window.results.check_all()
    window.copy_marked()
    itog = window.status_label.text()
    assert "настройки не сохраняются" in itog, itog
    assert "скопировано 1 фото" in itog, itog


def test_novyj_razbor_ne_nasleduet_starye_poteri(window, bez_skana) -> None:
    """Смена папки и новый разбор обнуляют накопитель: отчёт о второй папке не должен
    рассказывать о потерях первой."""
    window._poteri["/starsij.jpg"] = "файл не читается: прочее"
    window.photos = {Path("/starsoe.jpg"): [lico()]}
    window.files = [Path("/novoe.jpg")]
    window.start_scan()
    assert window._poteri == {} and window.photos == {}
    assert len(bez_skana.sozdannye) == 1


def test_knopka_kopirovaniya_zanimaet_pri_razbore(window, bez_skana, qapp) -> None:
    """Копировать посреди разбора нельзя: отчёт противоречил бы скопированному.

    Воспроизведение проверки ревьюера: `start_scan` снимает `self.photos`, а карточки
    и галочки остаются на экране. Прежнее окно не гасило кнопку, и нажатие давало
    report.txt, где рядом стояли «Скопировано: 1» и «фото в отчёте: 0 / ни одного
    разобранного фото» — то есть отчёт отрицал то, что человек уже видел в папке
    результатов.

    Проверяются оба замка и оба возврата: занятость гасит кнопку, свободный прогон с
    отметками включает её снова, а `rebuild_scores` и `selection_changed` не имеют
    права включить её, пока поток жив.
    """
    a = Path("/a.jpg")
    window.files = [a]
    polozhit_etalon(window, lico())
    window.settings_bar.set_threshold(0.30)
    window._on_scan_done(({a: [lico()]}, ScanStats(scanned=1, cached=0, seconds=0.2)))
    srazu(qapp)
    window.results.check_all()
    srazu(qapp)
    assert window.copy_button.isEnabled(), "копировать готовое — можно"

    window.start_scan()
    assert window.copy_button.isEnabled() is False, "кнопка жива посреди разбора"
    assert window.results.checked == [], "старые галочки пережили смену данных"

    # ни одна из трёх точек, где кнопка включается, не имеет права включить её занятой
    window.results._na_galochke(a, True)
    assert window.copy_button.isEnabled() is False
    window.rebuild_scores()
    assert window.copy_button.isEnabled() is False

    window.worker.done.emit(({a: [lico()]}, ScanStats(scanned=1, cached=0, seconds=0.2)))
    srazu(qapp)
    assert window.copy_button.isEnabled() is True, "после разбора кнопка не вернулась"


def test_report_posle_razbora_ne_vraet_pro_skopirovannoe(window, bez_skana, tmp_path,
                                                         monkeypatch, qapp) -> None:
    """Отчёт строится по тем же данным, что и копирование, — а не по пустой памяти.

    Сквозная проверка того же дефекта: если бы окно всё-таки пустил копирование
    вслепую, в report.txt оказалось бы «фото в отчёте: 0» рядом с реальной пачкой
    файлов. Здесь проверяем, что после разбора обе части сходятся числом.
    """
    from PIL import Image

    import ui.main_window as modul

    arhiv = tmp_path / "arhiv"
    arhiv.mkdir()
    a = arhiv / "a.jpg"
    Image.new("RGB", (60, 60), (7, 7, 7)).save(a, "JPEG")
    window.files = [a]
    polozhit_etalon(window, lico())
    window.settings_bar.set_threshold(0.30)
    window._on_scan_done(({a: [lico()]}, ScanStats(scanned=1, cached=0, seconds=0.2)))
    srazu(qapp)
    window.results.check_all()
    monkeypatch.setattr(modul.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *ar, **kw: str(tmp_path / "kuda")))
    window.copy_marked()
    otchyot = (tmp_path / "kuda" / "report.txt").read_text(encoding="utf-8")
    assert "ни одного разобранного фото" not in otchyot
    assert "фото в отчёте: 1" in otchyot
    assert "скопировано 1 фото" in window.status_label.text()


# --- слова, которых в интерфейсе быть не может -------------------------------------------------


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
    assert len(teksty) > 10, f"нечего проверять, нашли только {teksty}"
    for tekst in teksty:
        niz = tekst.lower()
        for stem in STEM:
            assert stem not in niz, f"«{stem}» в тексте: {tekst!r}"
        for vzorec in CELYE_SLOVA:
            assert not re.search(vzorec, niz), f"запрещённое слово в тексте: {tekst!r}"


def stroki_iz_ishodnika(*imena: str) -> list[str]:
    """Строковые литералы модулей, кроме docstring: из них собраны подписи, подсказки и
    тексты диалогов, которые ещё не стали виджетами."""
    koren = Path(__file__).resolve().parents[1] / "src"
    rezultat: list = []
    for imya in imena:
        derevo = ast.parse((koren / imya).read_text(encoding="utf-8"))
        docstriki = set()
        for uzel in ast.walk(derevo):
            if isinstance(uzel, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                telo = getattr(uzel, "body", [])
                if telo and isinstance(telo[0], ast.Expr) and \
                        isinstance(telo[0].value, ast.Constant) and \
                        isinstance(telo[0].value.value, str):
                    docstriki.add(id(telo[0].value))
        for uzel in ast.walk(derevo):
            if isinstance(uzel, ast.Constant) and isinstance(uzel.value, str) \
                    and id(uzel) not in docstriki:
                rezultat.append(uzel.value)
    return rezultat


def test_v_okne_net_zapreshhennyh_slov(window) -> None:
    proveryu_na_zaprety(vse_teksty(window))


def test_v_okne_s_rezultatami_net_zapreshhennyh_slov(window, qapp, tmp_path) -> None:
    """Пустое окно проверяем и заполненное: подпись кучки, статус разбора и объяснение
    числа появляются только после работы."""
    arhiv = tmp_path / "arhiv"
    arhiv.mkdir()
    (arhiv / "a.jpg").write_bytes(b"x")
    window.set_source_folders([arhiv])
    papki = {arhiv / "a.jpg": [lico(100, 0)], arhiv / "b.jpg": [lico(50, 1)]}
    polozhit_etalon(window, lico(100, 0))
    window._on_scan_done((papki, ScanStats(scanned=2, cached=1, seconds=3.0)))
    srazu(qapp)
    window.results.toggle_weak.click()
    teksty = (vse_teksty(window) + [window.status_label.text(),
                                    window.source_status.text(),
                                    window._strok[0].podpis.text()])
    proveryu_na_zaprety(teksty)
    vmeste = " ".join(teksty)
    assert "39" in vmeste and "67" in vmeste, "окно не объясняет измеренный разрос"


def test_v_ishodnikah_okna_net_zapreshhennyh_slov() -> None:
    """Диалог «Нужен новый разбор» и текст неудачного копирования живут не в виджетах,
    а в строках исходника: обходом виджетов их не поймать."""
    proveryu_na_zaprety(stroki_iz_ishodnika("ui/main_window.py", "ui/results_view.py",
                                            "ui/reference_dialog.py", "main.py"))


def test_obiasnenie_chisla_vidno_srazu_s_pervogo_ekrana(window) -> None:
    """Объяснение числа обязано быть видно до всякого разбора: именно оно не даёт
    прочитать низкий процент как провал, и сделано оно по измеренным 39…67 %.

    Число внутри подписи берётся из настройки, а не зашито в текст. Прежняя версия этого
    теста проверяла `"45" in podskazka` и тем самым была соучастницей дефекта: она
    охраняла константу, нарисованную в исходнике, а не согласованность с ползунком.
    """
    podskazka = " ".join(vse_teksty(window))
    tekushhij = str(percent_of(window.settings_bar.threshold))
    assert "39" in podskazka and "67" in podskazka, "окно не объясняет измеренный разброс"
    assert tekushhij in podskazka, \
        f"в пояснении нет текущего числа ползунка {tekushhij} — подпись врёт о настройке"
    assert "45" not in podskazka or tekushhij == "45", \
        "в пояснение вернулась зашитая четвёрка-пятёрка вместо значения из ползунка"
    niz = podskazka.lower()
    for opisanie in ("похож", "не промах", "решать"):
        assert opisanie in niz, f"в пояснении нет слова «{opisanie}»"


# --- точка входа -------------------------------------------------------------------------------


def test_tochka_vhoda_est_i_ne_zapiskaet_pri_importe() -> None:
    """`src/main.py` при импорте не создаёт окно и не запускает цикл событий:
    `app.exec()` начинается только внутри `main()`."""
    import main as tochka_vhoda

    assert callable(tochka_vhoda.main)
    # `from __future__ import annotations` оставляет аннотацию строкой — принимаем обе формы
    assert "int" in str(tochka_vhoda.main.__annotations__.get("return"))


# --- окно просмотра по двойному клику ----------------------------------------------------


def narisovat_foto(tmp_path, imja: str = "svoe.jpg") -> Path:
    """Настоящий читаемый JPEG: окно просмотра обязано открыть именно файл, а не путь."""
    from PIL import Image

    kadr = np.full((600, 800, 3), 210, dtype=np.uint8)
    kadr[200:360, 300:420] = 20
    put = tmp_path / imja
    Image.fromarray(kadr).save(put, "JPEG", quality=95)
    return put


def arhiv_s_fajlami(window, tmp_path) -> Path:
    """Разобранный архив, где снимки реально лежат на диске, и выбранный эталон."""
    pervyj = narisovat_foto(tmp_path, "svoe.jpg")
    vtoroj = narisovat_foto(tmp_path, "chuzhoe.jpg")
    window.photos = {pervyj: [lico(100, 0)], vtoroj: [lico(50, 1)]}
    window.files = list(window.photos)
    window.folders = [tmp_path]
    polozhit_etalon(window, lico(100, 0))
    window.rebuild_scores()
    return pervyj


def test_dvojnyj_klik_otkryvaet_prosmotr_s_lice(qapp, tmp_path) -> None:
    """Двойной клик по карточке открывает окно с этим фото, его рамкой и отметкой.

    Проверяется связка целиком: сигнал сетки -> окно -> прочитанный кадр. Отдельно
    сетка и отдельно окно это не ловят.
    """
    from ui.main_window import MainWindow
    from utils.config import Settings

    okno = MainWindow(Settings(tmp_path / "s.ini"), cache_path=tmp_path / "c.sqlite3")
    pervyj = arhiv_s_fajlami(okno, tmp_path)

    okno.results.prosyat_otkryt.emit(pervyj)

    prosmotr = okno._okno_prosmotra
    assert prosmotr is not None, "окно просмотра не открылось"
    assert prosmotr.pokazyvaet == pervyj
    assert not prosmotr.foto.pixmap().isNull(), "фото в окне не показалось"
    assert not prosmotr.krop.pixmap().isNull(), "крупного лица в окне нет"
    okno.close()


def imena_strok_v_raskladke(window) -> list[str]:
    """Имена людей в том порядке, в каком строки лежат в раскладке шага 2."""
    from PySide6.QtWidgets import QLineEdit

    imena = []
    for indeks in range(window._pole_ljudj.count()):
        videzhet = window._pole_ljudj.itemAt(indeks).widget()
        if videzhet is window.add_person_button:
            continue
        imena.append(videzhet.findChild(QLineEdit).text())
    return imena


def test_okno_vtorogo_cheloveka_otkryvaetsya_bez_chuzhogo_naslediya(
        window, dialog) -> None:
    """У второго человека своё наследие: пустое, пока он ничего не выбрал.

    Диалог эталона один на всех, и чужой прежний ответ внутри него — это второй поиск,
    выросший из первого: «главное фото» считается по снимкам другого человека, и
    проценты на карточках перестают означать то, что человек думает. Проверено руками:
    добавили ребёнка, открыли «мама» — внутри лежали снимки ребёнка.
    """
    polozhit_etalon(window, lico(100, 0))
    window._dobavit_cheloveka("мама")

    window._strok[1].knopka.click()

    assert dialog.nasledie_peredanne is None, "второму человеку принесли эталон ребёнка"
    assert dialog.nazvanie_peredanno == "мама"


def test_vtoromu_cheloveku_okno_peredaet_lica_pervogo(window, dialog) -> None:
    """Главное окно говорит диалогу, какие лица уже заняты другим поиском.

    Без этого диалог не может отличить «человек принёс снимки своего ребёнка» от
    «человек принёс те же снимки для мамы» и ставит автоотметку там, где человек уже
    всё решил (задача T22).
    """
    pervyj = polozhit_etalon(window, lico(100, 0))
    okno_stroka = window._strok[0]
    window._dobavit_cheloveka("мама")

    window._strok[1].knopka.click()

    assert dialog.chuzhie_peredannye == frozenset({(okno_stroka.chelovek.itog.anchor.put, 0)}), \
        dialog.chuzhie_peredannye
    assert pervyj.nazvanie == "ребёнок"


def test_sam_sebe_chelovek_zanyatyh_lic_ne_stavit() -> None:
    """Свои же лица человеку не «чужие»: повторно открытое окно помнит его выбор.

    Иначе второй заход в ту же строку обнулил бы автоотметки, и человек увидел бы
    пустую сетку там, где он уже отметил четыре лица.
    """
    from ui.main_window import MainWindow
    from utils.config import Settings

    okno = MainWindow(Settings(), cache_path=Path("net.sqlite3"))
    try:
        chelovek = okno.ljudi[0]
        assert okno._chuzhie_otmetki(chelovek) == frozenset()
    finally:
        okno.close()


def test_otvet_vtorogo_cheloveka_ne_tropaet_pervogo(window, dialog) -> None:
    """Ответ заезжает ровно в того человека, чью кнопку нажали.

    Прежний эталон первого обязан остаться на месте: окна у людей общие, а данные — нет.
    """
    pervyj = polozhit_etalon(window, lico(100, 0))
    bylo_pervogo = pervyj.itog
    window._dobavit_cheloveka("мама")
    dialog.otvet = itog_iz(lica_po_fajlam(lico(120, 2)))

    window._strok[1].knopka.click()

    assert window.ljudi[1].itog is dialog.otvet
    assert window.ljudi[0].itog is bylo_pervogo, "выбор ребёнка перезаписан"


def test_stroki_ljudej_idut_v_poryadke_spiska_a_ne_naoborot(window) -> None:
    """Строка номер N на экране обязана быть человеком номер N, а не первым сверху.

    Проверено руками: после «Добавить ещё одного» порядок виджетов оказался перевёрнут
    относительно списка, и переименованная строка уехала не туда, где лежал выбор фото.
    Человек вводит «мама», открывает окно с заголовком «мама» — и видит там снимки
    ребёнка, потому что это была та же строка, только названная другим именем.
    """
    window._dobavit_cheloveka("мама")
    window._dobavit_cheloveka("бабушка")

    assert imena_strok_v_raskladke(window) == ["ребёнок", "мама", "бабушка"]


def test_knopka_nazyvaet_cheloveka_foto_kotorogo_vybiraesh(window) -> None:
    """На кнопке написано, ЧЬИ фото она просит, и имя обновляется вместе с полем.

    Две одинаковые кнопки «Выбрать фото…» рядом не объясняют, в какую строку что
    приедет, а после переименования строки старая подпись врёт.
    """
    from PySide6.QtTest import QTest
    from PySide6.QtCore import Qt

    window._dobavit_cheloveka()
    assert window._strok[1].knopka.text() == "Выбрать фото: человек 2"

    imja = window._strok[1].imja
    # `QTest.keyClicks` на кириллице рвётся внутри Qt (ASSERT "false" в qasciikey.cpp:
    # у символа нет кода клавиши), поэтому поле заполняется `setText` — тот же сигнал
    # `textChanged`, что и при наборе. Мышь этим тестом не проверяется, её проверяют
    # тесты сетки и окна эталона.
    imja.setText("мама")
    imja.editingFinished.emit()

    assert window._strok[1].knopka.text() == "Выбрать фото: мама"
    assert window.ljudi[1].nazvanie == "мама", "имя уехало не в того человека"


def test_kartochka_i_okno_risyut_ramku_na_kazhdom_najdennom_lice(qapp, tmp_path) -> None:
    """Один снимок, два места показа: рамка на каждом найденном лице и ни на одном лишнем.

    Замер на архиве пользователя: на групповом фото из 24 лиц ребёнок стоит дважды —
    лица 22 и 24, по 65 %. Приложение обводило одно, человек видел одну рамку там, где
    его детей двое. Проверка идёт по пикселю карточки и по подписи окна: список
    совпадений мог доехать до строки результата и не доехать до рисования.
    """
    from PIL import Image
    from PySide6.QtWidgets import QLabel

    from ui.face_picker import ZELENYJ
    from ui.main_window import MainWindow
    from ui.results_view import THUMB
    from utils.config import Settings

    def otpechatok(axis: int) -> np.ndarray:
        v = np.zeros(512, dtype=np.float32)
        v[axis] = 1.0
        return v

    # PNG, а не JPEG: точный цвет линии рамочки сжатие съедает, а проверять нужно именно
    # цвет — зелёный значит «найдено».
    put = tmp_path / "dvoe.png"
    Image.fromarray(np.full((600, 800, 3), 210, dtype=np.uint8)).save(put)

    okno = MainWindow(Settings(tmp_path / "s.ini"), cache_path=tmp_path / "c.sqlite3")
    chuzhoj = Face(box=(60.0, 120.0, 200.0, 260.0), landmarks=None,
                   embedding=otpechatok(1), detector="insight")
    pervoe = Face(box=(500.0, 300.0, 640.0, 440.0), landmarks=None,
                  embedding=otpechatok(0), detector="insight")
    vtoroe = Face(box=(500.0, 60.0, 640.0, 200.0), landmarks=None,
                  embedding=otpechatok(0), detector="insight")
    okno.photos = {put: [chuzhoj, pervoe, vtoroe]}     # один и тот же ребёнок дважды
    okno.files = [put]
    okno.folders = [tmp_path]
    polozhit_etalon(okno, pervoe)
    polozhit_etalon(okno, pervoe, [vtoroe])
    okno.rebuild_scores()

    k = THUMB / min(800, okno.scan_dim)
    kartinka = [et for et in okno.results.findChildren(QLabel)
                if et.pixmap() is not None and not et.pixmap().isNull()][0].pixmap().toImage()

    def cveta_vnutri(box) -> set[tuple[int, int, int]]:
        x1, y1, x2, y2 = (int(v * k) for v in box)
        return {kartinka.pixelColor(x, y).getRgb()[:3]
                for y in range(max(0, y1), min(kartinka.height(), y2 + 1))
                for x in range(max(0, x1), min(kartinka.width(), x2 + 1))}

    assert ZELENYJ in cveta_vnutri(pervoe.box), "у первого совпадения нет рамки"
    assert ZELENYJ in cveta_vnutri(vtoroe.box), "второе совпадение потерялось"
    assert ZELENYJ not in cveta_vnutri(chuzhoj.box), "обвели того, кого не искали"

    okno.results.prosyat_otkryt.emit(put)
    prosmotr = okno._okno_prosmotra
    assert prosmotr is not None, "окно просмотра не открылось"
    # «из 3» — про кадр, а не про число рамок: совпадений два, а лиц в кадре три.
    assert "лицо 2 из 3" in prosmotr.slovo.text(), prosmotr.slovo.text()
    assert prosmotr.pokazyvaemoe_lice == 2
    assert len(prosmotr._ramki) == 2, "окно обвело не то число лиц, что карточка"
    okno.close()


def test_galochka_prosmotra_otmechaet_kartochku_v_setke(qapp, tmp_path) -> None:
    """Владелец отметки — сетка. Решение из окна обязано вернуться в неё галочкой,
    иначе «Скопировать отмеченные» унесёт не то, что человек отмечал."""
    from ui.main_window import MainWindow
    from utils.config import Settings

    okno = MainWindow(Settings(tmp_path / "s.ini"), cache_path=tmp_path / "c.sqlite3")
    pervyj = arhiv_s_fajlami(okno, tmp_path)
    okno.results.prosyat_otkryt.emit(pervyj)
    prosmotr = okno._okno_prosmotra
    bylo = okno.results.otmecheno(pervyj)

    prosmotr.copy_check.setChecked(not bylo)

    assert okno.results.otmecheno(pervyj) is (not bylo)
    assert dict(okno.results._boxes)[pervyj].isChecked() is (not bylo)
    okno.close()


def test_otmetka_v_setke_peredajotsja_v_otkrytoe_okno(qapp, tmp_path) -> None:
    """Обратная сторона: снял галочку в списке — открытое окно обязано показать то же.

    Иначе человек снимет отметку, а окно продолжит утверждать обратное, и следующее
    «копировать это фото» вернёт то, от чего он только что отказался.
    """
    from ui.main_window import MainWindow
    from utils.config import Settings

    okno = MainWindow(Settings(tmp_path / "s.ini"), cache_path=tmp_path / "c.sqlite3")
    pervyj = arhiv_s_fajlami(okno, tmp_path)
    okno.results.prosyat_otkryt.emit(pervyj)
    prosmotr = okno._okno_prosmotra

    okno.results.otmetit_snaruji(pervyj, not prosmotr.copy_check.isChecked())

    assert prosmotr.copy_check.isChecked() is okno.results.otmecheno(pervyj)
    okno.close()


def test_prosmotr_ne_otkryvaetsja_posle_nachala_novogo_razbora(window, bez_skana,
                                                               tmp_path) -> None:
    """Посреди разбора карточки на экране относятся к прошлому набору: открыть по
    двойному клику фото, которого в новых данных нет, — значит показать проценты
    вчерашнего эталона под сегодняшней подписью."""
    pervyj = arhiv_s_fajlami(window, tmp_path)
    window.start_scan()

    window.results.prosyat_otkryt.emit(pervyj)

    assert window._okno_prosmotra is None, "окно просмотра открылось занятому окну"
    assert bez_skana.sozdannye, "разбор обязан идти"


def test_okno_prosmotra_zakryvaetsja_vmeste_s_glavnym(window, tmp_path) -> None:
    """Просмотр — дитя главного окна: закрытие родителя не оставляет на экране окно,
    которое больше nobody не контролирует."""
    pervyj = arhiv_s_fajlami(window, tmp_path)
    window.results.prosyat_otkryt.emit(pervyj)
    prosmotr = window._okno_prosmotra
    assert prosmotr is not None

    window.close()

    assert prosmotr.isHidden() or not prosmotr.isVisible()


# --- T11: стрелки и процент в окне просмотра --------------------------------------------


def lico_s_pohozhestju(storona: float, bliz: float) -> Face:
    """Лицо с косинусом `bliz` к эталону (вектор axis=0).

    Единичные векторы по осям дают только 100 % и 0 % — на них порядок находок не
    построить, а именно порядок здесь и проверяется.
    """
    v = np.zeros(512, dtype=np.float32)
    v[0] = bliz
    v[1] = float(np.sqrt(max(0.0, 1.0 - bliz * bliz)))
    return Face(box=(0.0, 0.0, storona, storona * 1.5), landmarks=None,
                embedding=v, detector="insight")


def arhiv_tri_fajla(window, tmp_path) -> list:
    """Три реальных снимка с процентами 62, 40 и 12. Порядок в списке — по убыванию
    похожести, как его показывает сетка."""
    from PIL import Image

    puti = []
    for imja in ("a.jpg", "b.jpg", "c.jpg"):
        kadr = np.full((600, 800, 3), 210, dtype=np.uint8)
        kadr[200:360, 300:420] = 20
        put = tmp_path / imja
        Image.fromarray(kadr).save(put, "JPEG", quality=95)
        puti.append(put)
    window.photos = {puti[0]: [lico_s_pohozhestju(400, 0.62)],
                     puti[1]: [lico_s_pohozhestju(300, 0.40)],
                     puti[2]: [lico_s_pohozhestju(200, 0.12)]}
    window.files = list(window.photos)
    window.folders = [tmp_path]
    polozhit_etalon(window, lico(100, 0))
    window.rebuild_scores()
    return puti


def test_strelka_vpered_pokazyvaet_sledujushchee_foto_v_tom_zhe_okne(window, tmp_path) -> None:
    """Порядок листания обязан совпадать с порядком на экране: «вперёд» — это к менее
    похожему фото, а не к случайному из словаря."""
    puti = arhiv_tri_fajla(window, tmp_path)
    window.results.prosyat_otkryt.emit(puti[0])
    pervoe = window._okno_prosmotra
    assert pervoe.pokazyvaet == puti[0]

    pervoe.prosyat_perehod.emit(+1)

    assert window._okno_prosmotra is pervoe, "открылось второе окно вместо перелистывания"
    assert pervoe.pokazyvaet == puti[1]
    assert pervoe.copy_check.isChecked() is window.results.otmecheno(puti[1])


def test_strelka_nazad_vozvraschaet_okno_na_shag(window, tmp_path) -> None:
    puti = arhiv_tri_fajla(window, tmp_path)
    window.results.prosyat_otkryt.emit(puti[1])
    okno = window._okno_prosmotra

    okno.prosyat_perehod.emit(-1)

    assert okno.pokazyvaet == puti[0]


def test_na_kraine_spiska_peremeshcheniya_net(window, tmp_path) -> None:
    """За первым и последним фото ходить некуда. Кнопки гаснет на основании позиции,
    которую даёт сетка, а не догадки окна."""
    puti = arhiv_tri_fajla(window, tmp_path)
    window.results.prosyat_otkryt.emit(puti[0])
    okno = window._okno_prosmotra
    assert okno.prev_button.isEnabled() is False
    assert okno.next_button.isEnabled() is True

    window.results.prosyat_otkryt.emit(puti[-1])
    assert okno.next_button.isEnabled() is False
    assert okno.prev_button.isEnabled() is True

    # за краем не происходит ничего: шаг читается знаком, а «вперёд с последнего» —
    # это ровно то, что клавиша сделает по ошибке, и окно обязано остаться на месте
    poslednij = okno.pokazyvaet
    okno.prosyat_perehod.emit(+1)
    assert okno.pokazyvaet == poslednij


def test_procent_viden_v_okne_prosmotra(window, tmp_path) -> None:
    """Число, по которому человек и отбирает снимки, обязано быть перед глазами и в
    крупном кадре, а не оставаться на карточке под окном."""
    puti = arhiv_tri_fajla(window, tmp_path)
    window.results.prosyat_otkryt.emit(puti[0])
    tekst = window._okno_prosmotra.slovo.text()
    assert "62" in tekst, f"процента нет на форме: {tekst!r}"
    assert "1" in tekst and "3" in tekst, f"позиции нет на форме: {tekst!r}"


def test_nachalo_novogo_razbora_zakryvaet_prosmotr(window, bez_skana, tmp_path) -> None:
    """Новый разбор снимает с экрана прежние проценты. Окно просмотра с кадром
    вчерашнего поиска и живой галочкой от исчезнувшего списка — это обман в двух
    местах сразу, поэтому оно закрывается, а не висит."""
    puti = arhiv_tri_fajla(window, tmp_path)
    window.results.prosyat_otkryt.emit(puti[0])
    okno = window._okno_prosmotra
    assert okno.isVisible()

    window.start_scan()

    assert not okno.isVisible(), "просмотр пережил разбор, который обнулил список"


# --- T13: повторное открытие окна эталона возвращает прежний выбор -----------------


def test_otmena_povtornogo_okna_ne_tropaet_etalon(window, dialog) -> None:
    """Открыл, посмотрел, нажал «Отмена» — поиск продолжается по прежним лицам, и в
    следующий раз окно покажет ровно то же самое.

    Иначе «вспомнить, что я тогда отметил» стоило бы сброса всего найденного.
    """
    arhiv_s_dvumja_snimkami(window)
    itog = itog_iz(lica_po_fajlam(lico(100, 0), lico(90, 0)))
    dialog.otvet = itog
    window._strok[0].knopka.click()
    assert window.ljudi[0].itog is itog

    schet_do = window.ljudi[0].reference.count
    dialog.otvet = None                     # человек жмёт «Отмена»
    window._strok[0].knopka.click()

    assert window.ljudi[0].reference.count == schet_do
    assert window.ljudi[0].itog is itog, "наследие стёрто отменой"


def test_novoe_okno_poluchaet_nasledie_ot_okna(window, dialog) -> None:
    """Главное окно обязано ПЕРЕДАТЬ прежний ответ в диалог. Без этого аргумента
    диалог физически не знает, что восстанавливать, — и открывается пустым."""
    arhiv_s_dvumja_snimkami(window)
    itog = itog_iz(lica_po_fajlam(lico(100, 0), lico(90, 0)))
    dialog.otvet = itog
    window._strok[0].knopka.click()

    window._strok[0].knopka.click()

    assert dialog.nasledie_peredanne is itog


# --- T15: по какому именно выбранному лицу нашёлся снимок -------------------------------
#
# Пользователь отмечает не одно лицо ребёнка, а несколько примеров — и среди них может
# быть другой человек. Одно число на карточке эти два случая не различает, а решение
# «убрать снимок из папки результатов» человек принимает именно по нему. Подпись
# проходит три места: карточку сетки, окно просмотра и отчёт, и собирает имя строки из
# ОДНОЙ функции главного окна — иначе номер и подпись разъедутся при первой же
# перестановке отметок.

from PySide6.QtCore import Qt                             # noqa: E402
from PySide6.QtTest import QTest                          # noqa: E402


def arhiv_s_dvumja_ljudmi(window, tmp_path) -> dict:
    """Два человека в поиске и три снимка: оба, только ребёнок, только мама.

    Один общий эталон на все отмеченные лица давал один максимум на кадр, и «100 %»
    не отвечало, кто именно в кадре. Здесь у каждого свой отпечаток: ребёнок — вдоль
    оси 0, мама — вдоль оси 2, и их сходство с чужим лицом ровно нулевое.
    """
    oba = narisovat_foto(tmp_path, "oba.jpg")
    tolko_rebenok = narisovat_foto(tmp_path, "tolko_rebenok.jpg")
    tolko_mama = narisovat_foto(tmp_path, "tolko_mama.jpg")
    malish, mama = lico(100, 0), lico(120, 2)
    window.photos = {oba: [malish, mama], tolko_rebenok: [malish], tolko_mama: [mama]}
    window.files = list(window.photos)
    window.folders = [tmp_path]
    polozhit_etalon(window, malish, nazvanie="ребёнок")
    window.ljudi.append(Chelovek(nazvanie="мама",
                                 itog=Itog(karty=(), anchor=None, extra=(),
                                           otvergnutye=(), reference=make_reference(mama, []))))
    window._perestroit_stroki_ljudej()
    window.rebuild_scores()
    return {"oba": oba, "rebenok": tolko_rebenok, "mama": tolko_mama}


def test_kartochka_nazyvaet_kogo_iz_iskemyh_nashli(window, qapp, tmp_path) -> None:
    """Два человека в поиске — карточка говорит, кто из них в кадре, и каждым числом.

    Прежняя подпись («похоже на лицо 2») называла отметку, а не человека: при двух
    искомых она не отвечала на вопрос, чьё это фото, и решение об уборке снимка
    принималось вслепую.
    """
    snimki = arhiv_s_dvumja_ljudmi(window, tmp_path)
    podpisi = {put: metka.text() for put, metka in window.results._podpisi.items()}
    assert podpisi[snimki["oba"]] == "ребёнок 100 % · мама 100 %", podpisi
    assert podpisi[snimki["rebenok"]] == "ребёнок 100 %", podpisi
    assert podpisi[snimki["mama"]] == "мама 100 %", podpisi


def test_chislo_na_kartochke_ne_perekrivaet_vtorogo_cheloveka(
        window, qapp, tmp_path) -> None:
    from ui.results_view import ramki_stroki

    """Слабое совпадение одного человека не прячется за сильным у другого.

    Один общий максимум на кадр означал бы, что на снимке, где ребёнок узнаётся
    плохо, а мама отлично, человек увидел бы одно число и не понял, что ребёнка там
    всё-таки нашли.
    """
    oba = arhiv_s_dvumja_ljudmi(window, tmp_path)["oba"]
    row = next(r for r in window.results.itogi_prosmotra() if r.path == oba)
    po_ljudjam = {s.chelovek: s.percent for s in row.sovpadeniya}
    assert po_ljudjam == {1: 100, 2: 100}, po_ljudjam
    assert len(ramki_stroki(row)) == 2, "каждому найденному — своя рамка"


def test_dvojnyj_klik_govorit_kto_najden(window, qapp, tmp_path) -> None:
    """Крупный кадр подписывает тем же именем, что и карточка.

    Проверка жестом: окно просмотра получает ответ из сетки, и своё мнение об эталоне
    ему заводить не на чем.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from ui.results_view import ramki_stroki

    snimki = arhiv_s_dvumja_ljudmi(window, tmp_path)
    kletka = window.results._kletki[snimki["mama"]]

    QTest.mouseDClick(kletka, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                      kletka.rect().center())

    prosmotr = window._okno_prosmotra
    assert prosmotr is not None and prosmotr.pokazyvaet == snimki["mama"]
    tekst = prosmotr.slovo.text()
    assert "найден: мама" in tekst, f"в окне не сказано, кого нашли: {tekst!r}"
    assert "ребёнок" not in tekst, f"окно приписало снимок не тому человеку: {tekst!r}"
    window.close()


def test_dvojnyj_klik_vozvraschaet_prosmotr_na_pered(window, qapp, tmp_path) -> None:
    """Жест целиком: окно уехало за главное — двойной клик по карточке возвращает его.

    Проверено мышью, а не вызовом `pokazat`: жалоба человека была в том, что после
    клика по главному окну просмотр оказывается под ним, и следующий двойной клик не
    показывает ничего. Данные при этом менялись, так что сетка «не видит» промаха.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    snimki = arhiv_s_dvumja_ljudmi(window, tmp_path)
    window.show()
    kletka = window.results._kletki[snimki["mama"]]
    QTest.mouseDClick(kletka, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                      kletka.rect().center())
    prosmotr = window._okno_prosmotra
    assert prosmotr is not None
    qapp.processEvents()
    assert qapp.activeWindow() is prosmotr, "первый клик не открыл окно на перед"

    # Человек кликнул по главному окну — просмотр остался за ним.
    window.activateWindow()
    qapp.processEvents()
    assert qapp.activeWindow() is window, "главное окно не забрало активость — тест слепой"

    kletka2 = window.results._kletki[snimki["rebenok"]]
    QTest.mouseDClick(kletka2, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                      kletka2.rect().center())
    qapp.processEvents()
    assert prosmotr.pokazyvaet == snimki["rebenok"], prosmotr.pokazyvaet
    assert qapp.activeWindow() is prosmotr, "новое фото показали, но окно осталось за главным"
    window.close()


def test_kopirovanie_pishet_kogo_nashli_v_obe_chasti_otchyota(
        window, qapp, tmp_path, monkeypatch) -> None:
    """report.txt и results.csv называют того же человека, что и карточка.

    Отчёт человек читает дома, в папке результатов, уже без экрана: если имя живёт
    только на карточке, через час он не восстановит, чьё это фото.
    """
    import ui.main_window as modul

    arhiv_s_dvumja_ljudmi(window, tmp_path)
    window.results.check_all()
    kuda = tmp_path / "kuda"
    monkeypatch.setattr(modul.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(kuda)))

    window.copy_marked()

    otchyot = (kuda / "report.txt").read_text(encoding="utf-8")
    assert "кто из искомых людей найден" in otchyot, "колонка не объяснена заголовком"
    linii = [ln for ln in otchyot.splitlines() if "tolko_mama.jpg" in ln]
    assert linii and "мама 100 %" in linii[0], linii
    assert "Кого ищем: ребёнок (1 лиц), мама (1 лиц)" in otchyot, otchyot.splitlines()[:8]
    tablica = (kuda / "results.csv").read_text(encoding="utf-8-sig")
    assert "кто найден и насколько похож" in tablica.splitlines()[0], tablica.splitlines()[0]
    assert "мама 100 %" in tablica


def test_pri_odnom_cheloveke_imya_ne_povtoryaetsya(window, qapp, tmp_path) -> None:
    """Эталон из одного лица: ни второй строки в сетке, ни «найдено по» в окне, ни
    колонки в отчёте.

    При одном выбранном лице любая такая подпись отвечает на вопрос, которого не было,
    поэтому прежнее поведение сохраняется целиком.
    """
    pervyj = arhiv_s_fajlami(window, tmp_path)
    assert window.ljudi[0].reference.count == 1

    assert window.results._podpisi == {}
    window.results.prosyat_otkryt.emit(pervyj)
    assert "найдено по" not in window._okno_prosmotra.slovo.text()
    window.close()


# --- T16: в окне просмотра видны ВСЕ лица снимка -----------------------------------------


def test_prosmotr_pokazyvaet_vse_lica_snimka_a_ne_odno(tmp_path, qapp) -> None:
    """Снимок, где взрослых и детей несколько, обязан показать все найденные лица,
    а не только то, что дало совпадение.

    Находка пользователя: на фото, где ребёнок и взрослый сидят вместе, окно подписало
    «лицо 1 из 1», хотя детектор нашёл там два лица (157 и 170 px — проверено по
    оглавлению архива). Причина была не в детекции: окно получало ровно одну рамку,
    ту, что дала максимум. Подпись «1 из 1» при этом звучала как «приложение не увидело
    ребёнка», и проверить это было нечем.

    Отличить найденное лицо от ненайденного человек может только по числу лиц в подписи.
    """
    from PIL import Image

    from ui.main_window import MainWindow
    from utils.config import Settings

    kadr = np.full((600, 800, 3), 210, dtype=np.uint8)
    kadr[100:260, 100:260] = 20          # «лицо 1»
    kadr[300:470, 500:670] = 30          # «лицо 2», чуть темнее
    put = tmp_path / "vmeste.jpg"
    Image.fromarray(kadr).save(put, "JPEG", quality=95)

    okno = MainWindow(Settings(tmp_path / "s.ini"), cache_path=tmp_path / "c.sqlite3")
    try:
        pervoe, vtoroe = lico(160, 0), lico(170, 1)
        okno.photos = {put: [pervoe, vtoroe]}
        okno.files = [put]
        okno.folders = [tmp_path]
        polozhit_etalon(okno, vtoroe)
        okno.rebuild_scores()
        row = okno.results.itogi_prosmotra()[0]
        assert row.faces == 2, "на снимке два лица, тест потерял предмет"

        okno.results.prosyat_otkryt.emit(put)

        podpis = okno._okno_prosmotra.slovo.text()
        assert "лицо 2 из 2" in podpis, (
            f"окно показывает одно лицо там, где их два: {podpis!r}")
    finally:
        okno.close()


def test_okno_etalona_poluchaet_vklad_kazhdoj_stroki_etalo(tmp_path, qapp) -> None:
    """Главное окно обязано передать в диалог, сколько снимков нашлось по каждому
    выбранному лицу. Без этого диалог не может сказать «по нему найдено 14» и человек
    остаётся один на один с ценой решения, которое он принял сам.

    Замер на архиве пользователя: эталон из 12 отобранных лиц — 20 из 20 подтверждённых
    снимков и ноль чужих; эталон из 24 — те же 20 своих и 10 чужих. Фильтр ничего не
    стоил, и снят он был осознанно, но число должно быть на экране.
    """
    from PIL import Image

    from ui.main_window import MainWindow
    from utils.config import Settings

    kadr = np.full((600, 800, 3), 210, dtype=np.uint8)
    kadr[100:260, 100:260] = 20
    put = tmp_path / "a.jpg"
    Image.fromarray(kadr).save(put, "JPEG", quality=95)

    okno = MainWindow(Settings(tmp_path / "s.ini"), cache_path=tmp_path / "c.sqlite3")
    try:
        assert okno._vklad_strok_etalona(okno.ljudi[0]) == {}, "до поиска чисел быть не должно"

        pervoe, vtoroe = lico(160, 0), lico(170, 1)
        okno.photos = {put: [pervoe, vtoroe]}
        okno.files = [put]
        okno.folders = [tmp_path]
        polozhit_etalon(okno, pervoe)
        okno.rebuild_scores()

        vklad = okno._vklad_strok_etalona(okno.ljudi[0])
        assert vklad == {1: 1}, f"вклад считается не по строкам эталона: {vklad}"
    finally:
        okno.close()
