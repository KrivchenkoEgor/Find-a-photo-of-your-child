"""Окно выбора фото ребёнка: несколько снимков сразу, сетка лиц и число на каждом.

Шаг «Кого ищем» был самым одиноким местом приложения: одна кнопка, один файл за раз, и
чтобы принести четыре снимка ребёнка, человек четыре раза проходил диалог выбора. Здесь
он проходит его один: `QFileDialog.getOpenFileNames`, сколько угодно файлов, лица
находятся сами, и вся сетка — одна страница.

Почему это отдельное модальное окно, а не ещё одна строка в главном.

* Отметка лица и её следствие (какой эталон получится) должны быть видны рядом. В главном
  окне между «отметил» и «что нашлось» лежали бы минуты разбора архива.
* Пока человек выбирает фото, прежний эталон и прежние проценты обязаны оставаться на
  экране нетронутыми. Отмена модалки — единственный способ это гарантировать: у окна,
  которое ничего не приняло, просто нет нового ответа.

Почему диалог владеет СВОИМ `ScanWorker`, а не просит главное окно разобрать файлы.

* `FaceCache` потокобезопасен (`core/cache.py:83-84`: `check_same_thread=False` плюс
  собственный `threading.Lock`), второй поток к той же базе законен;
* диалог модальный, а `main_window._mozhno_nachat` дополнительно отказывает, пока окно
  эталона открыто (`main_window._etalon_otkryt`), — двух живых разборов не бывает, и
  проверяется это тестом, а не на слово;
* отдать разбор окну означало бы вручить его машину продолжений (`_posle_skana`) окну,
  которое живёт во вложенном цикле событий: два владельца одного признака занятости.

Качество разбора (`max_dim`) приходит из главного окна, а не выбирается здесь: оглавление
ключуется парой (файл, наносекунды, размер, движок, `max_dim`). Со своим числом диалог
миновал бы кэш на каждом файле уже разобранного архива — минуты работы впустую, и человек
видел бы это как «программа зависла». Тот же `max_dim` используется для кадров карточек:
рамка лица задана в пикселях именно того кадра.

Всё, что считается, живёт в `core.etalon`: карточки, проценты, «кто крупней» и кому
полагается предупреждение. Виджет только рисует ответ и пересылает туда набор отметок.
Соблазн пересчитать сходство на месте велик ровно до первого расхождения с арифметикой
этого модуля — именно из-за него его и пишут заново, а не дописывают в окно.

Пять обещаний, за которыми здесь следит тест.

1. **Ни один файл не пропадает.** Разобрался — карточка лиц; прочитан и без лиц — «лицо не
   найдено»; не прочитан — «фото не разобралось» с причиной в подсказке; разбор
   остановили — «разбор остановлен». Это четыре разных разговора с человеком, и у каждого
   своё слово.
2. **Отмеченное лицо участвует в поиске, и каждая участвующая карточка говорит об этом
   одними и теми же словами (решения T14 и T21).** Подпись «по нему ищем» стояла ровно на
   одной карточке, а восемь других показывали голый процент, — и человек вынес из экрана
   вывод «я выбрал два лица, а ищет всё ещё по одному». Исправление сначала поставило на
   это место слово «главное» — и оказалось, что «главным» оказывалось самое крупное лицо
   набора: когда для мамы принесли те же снимки, что и для ребёнка, главным назван спящий
   ребёнок. Теперь у всех участников одна подпись — «ищем по этому лицу», а число рядом
   показывает сходство с другим отмеченным лицом ЭТОГО ЖЕ человека.
3. **Отказ назван, и он один.** Слова «не ищем» остаются за единственным случаем, где
   галочку человека не выполняют: отпечаток непригоден, сравнивать им нечем. Такое лицо
   получает «сравнить нечем», но НЕ получает «0 %»: там `procent=0` значит «считать
   нечем», и прочесть это как «нисколько не похоже» — значит отправить человека искать
   другое фото там, где дело в качестве снимка.
4. **Ответ не зависит от мыши.** Карточки нельзя таскать, а отметки уходят в `core.etalon`
   множеством: первой строкой эталона становится самое крупное годное из отмеченных, а не
   то, куда ткнули первым. Для человека это не «главное лицо» — на экране все отмеченные
   равноправны, и ни одна карточка не получает число за сам размер.
5. **Поток переживает окно, а не наоборот.** `wait()` возвращает False ровно тогда, когда
   поток ещё жив, и в эту минуту закрытие отменяется и повторяется на `finished` — те же
   грабли, что чинит `main_window.closeEvent`. Оглавление этот диалог НЕ закрывает: база
   принадлежит главному окну и обязана пережить это окно.

Модели здесь не загружаются: `ScanWorker` поднимает их сам (`worker.obespechit_zagruzku`),
а очередь кадров читает с диска по одному снимку за тик событий.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Sequence

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QIcon
from PySide6.QtWidgets import (QAbstractItemView, QDialog, QDialogButtonBox, QFileDialog,
                              QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QListView,
                              QMessageBox, QProgressBar, QPushButton, QVBoxLayout, QWidget)

from core.cache import FaceCache
from core.engine import Face, FaceEngine, prigoden_otpechatok
from core.etalon import Itog, Kartochka, odinochnye_otmetki, sobrat_etalon
from core.matcher import REF_MIN_SIM, percent_of
from core.scanner import load_photo
from core.worker import ScanWorker
from ui.face_picker import (RAMKA_KRASNYJ, RamkaLica, STORONA_KARTOKI, ZELENYJ,
                            crops_from, face_to_qpixmap, naklej_nomer, obvesti)
from ui.photo_view import PhotoViewDialog

# Расширения совпадают с `scanner.PHOTO_SUFFIXES`. Константа живёт здесь, а не в
# `main_window`: единственное место, где у человека просят выбрать снимок, — это это окно.
FILTER_FOTO = "Фото (*.jpg *.jpeg *.png *.heic *.heif *.webp)"

# Сколько секунд ждём свой поток при закрытии окна. Дальше ждать смысла нет: поток
# отменён, закрытие отложено и повторится на `finished` (см. `main_window`).
OZHDANIE_ZAKRYTIJA_MS = 3000

# Кадры для карточек читаются в потоке интерфейса, поэтому по одному за тик событий:
# двадцать снимков с телефона — это секунды, а спецификация (раздел 7) не допускает вида
# зависшего окна. Первый кадр читается сразу: до него окно выглядело бы пустым ровно там,
# где человек ждёт лицо. Очередь, как и в `results_view`, работает только по видимому
# окну — и по той же причине: невидимому окну некуда спешить, а диск горит.
OGRESH_KADRA_MS = 15
SINHRONNO_KADROV = 1
PLITKA = 128                       # серый цвет плитки «кадр ещё не дошёл»

# Бледная карточка: предупреждение «похоже слабо» и единственный отказ «сравнить нечем».
# Оттенок не косметика: отметка обязана остаться видимой (её ставил человек), но не
# спорить вниманием с подтверждённой. С задачи T14 бледность — НЕ знак отказа: такие
# лица тоже участвуют в поиске, и слово «тоже ищем» стоит на них рядом с числом.
CVET_POVERYAVSHEJ = QColor(125, 125, 125)

# Число, ниже которого отмеченное лицо подсвечивают бледнее и подписывают
# предупреждением. В поиск оно при этом идёт ровно так же: с задачи T14 галочка человека
# — единственное решение, а число только просит перепроверить глазами. Второго владельца
# не заведено: это `REF_MIN_SIM` из `core.matcher`, тем же числом пользуется
# `core.etalon` и ровно тем же зовёт сборку эталона.
PROCENT_SLABO = percent_of(REF_MIN_SIM)

# --- тексты -------------------------------------------------------------------------

SLOVO_NET_LIC = "лицо не найдено"
SLOVO_NE_RAZOBRALOS = "фото не разобралось"
SLOVO_OSTANOVLEN = "разбор остановлен"
SLOVO_NE_CHITAETSYA = "кадр не читается — лицо не показать"
# Все отмеченные лица равноправны (T14) и подписаны одними словами: ни «главного», ни
# «тоже» на карточках больше нет. Прежнее «главное · по нему ищем» стояло на самом
# крупном лице набора, и когда человек приносил для мамы те же снимки, что и для
# ребёнка, главным оказывался спящий ребёнок — экран утверждал обратное тому, что
# человек в него принёс (задача T21).
SLOVO_ISSKEM = "ищем по этому лицу"
SLOVO_SLABO = "не похоже на остальных"
SLOVO_NECHEM = "сравнить нечем"
SLOVO_NE_ISSKEM = "не ищем"

# Вклад лица в находки: то, что отменило молчаливый отказ фильтра (T14) и не
# отменяет решения человека. Число отвечает на вопрос «что даст, если я это
# сниму», и снимает с приложения ответственность за выбор, который он больше
# не запрещает.
FRAZA_VKLAD = "по нему найдено {n} снимков"
FRAZA_VKLAD_ODIN = "по нему найден 1 снимок"
FRAZA_VKLAD_NOL = "по нему не нашлось ни одного снимка"

TEKST_PUSTO = ("Добавьте фото этого человека — можно сразу несколько. Приложение само "
               "найдёт на них лица и покажет каждое отдельной карточкой.")
TEKST_RAZBOR = "Разбираем фото… Кнопка поиска оживёт, когда закончим."
TEKST_BEZ_OTMETOK = ("Отметьте лица этого человека — кликните карточку. Сами отмечаются "
                     "только те снимки, где лицо одно и о котором вы ещё ничего не сказали")
TEKST_VSE_NEPRIGODNY = ("отмеченных лиц: {n}, но не ищем ни одного: сравнить нечем — "
                        "отметьте другое лицо или добавьте фото, где этот человек "
                        "виден крупнее и не в профиль")
# Подпись собирается из фраз-частей и склеивается точкой с запятой: каждую часть можно
# назвать отдельно, и ни одна не начинается с маленькой буквы посреди текста.
FRAZA_V_POISKE = "лиц в поиске: {poisk} из {vsego} отмеченных"
FRAZA_SLABO = "не похожи на остальных: {n}"
FRAZA_NECHEM = "не ищем: {n} — сравнить нечем"
FRAZA_NE_RAZOBRALOS = "не разобралось: {n} — причина в подсказке карточки"
FRAZA_OSTANOVLEN = "разбор остановлен, не досмотрено фото: {n}"

PODSKAZKA_KARTE = ("кликните карточку, чтобы отметить или снять отметку; число на "
                   "отмеченной карточке — насколько это лицо похоже на самое крупное из "
                   "отмеченных")
PODSKAZKA_OK = ("в поиск берутся все отмеченные лица — решает только ваша галочка. Число "
                f"показывает, насколько лицо похоже на ДРУГИЕ отмеченные лица этого "
                f"человека; меньше {PROCENT_SLABO} % — предупреждение «не похоже на "
                "остальных»: такое лицо всё равно участвует в поиске, но перепроверьте "
                "глазами, что на снимке правда искомый человек. Одно чужое лицо в эталоне "
                "начинает находить по всему архиву чужого, и отличить его потом нельзя")
PODSKAZKA_DOBAVIT = ("можно выделить несколько снимков сразу — со скольких бы человек ни "
                     "принёс, эталон собирается один")
PODSKAZKA_STOP = "прекратить разбор на текущем фото; уже найденные лица останутся в сетке"


class _SpisokLits(QListWidget):
    """Список карточек, где нажатие КУДА УГОДНО по карточке переключает отметку.

    Qt сам переключает галочку, только если нажатие пришло в прямоугольник индикатора.
    Замер на этой сборке, offscreen, обходом всего прямоугольника карточки: переключают
    **81 точка из 9880** — квадратик 16x16 в левом верхнем углу карточки 190x208.
    Остальные 9799 не делают ничего заметного: выделение в этом списке намеренно
    выключено, и карточка даже не синеет.

    При этом интерфейс обещает обратное в двух местах: «кликните карточку, чтобы отметить
    или снять отметку» и «Отметьте лица ребёнка — кликните карточку». Живой человек так и
    сделал: тыкал в лица на групповом снимке и получил молчание.

    Пресс перехватывается и НЕ отдаётся базовому классу: отдать его и переключить самим
    значило бы переключить дважды. Вариант с фильтром событий на `viewport()` тоже
    пробовался и убит: фильтр переживает смерть диалога на один ход цикла событий, и
    прокрутка событий из тестов главного окна доставляла нажатие уже удалённому виду —
    сегфолт на ровном месте в чужом файле.

    Собственный сигнал, а не вызываемая функция: виджет не знает, кто отвечает за отметки,
    и не держит ссылку на окно.
    """

    nazhatie_po_kartochke = Signal(object)
    dvojnoe_nazhatie = Signal(object)

    def _element_pod(self, sobytie):
        if sobytie.button() != Qt.MouseButton.LeftButton:
            return None
        return self.itemAt(sobytie.position().toPoint())

    def mouseDoubleClickEvent(self, sobytie) -> None:   # noqa: N802 (имя Qt)
        """Двойное нажатие — просьба показать снимок целиком, а не «отметить дважды».

        Первый пресс двойного клика Qt всё равно считает одиночным, и отметка на нём
        перевернулась бы. Возвращает её назад принимающая сторона — см.
        `_na_dvojnom_klike`: только она знает, каким состояние было до нажатия.
        """
        element = self._element_pod(sobytie)
        if element is not None:
            self.dvojnoe_nazhatie.emit(element)
            return
        super().mouseDoubleClickEvent(sobytie)

    def mousePressEvent(self, sobytie) -> None:      # noqa: N802 (имя Qt)
        # Отличить первый пресс двойного клика от одиночного здесь нельзя: Qt шлёт на
        # `clickCount` отдельное событие dblclick, а не второй пресс, и `clickCount` в
        # этой сборке PySide6 наружу не открыт. Поэтому пресс всегда переключает отметку,
        # а двойной клик откатывает её назад — см. `_na_dvojnom_klike`.
        element = self._element_pod(sobytie)
        if element is not None and element.flags() & Qt.ItemFlag.ItemIsUserCheckable:
            self.nazhatie_po_kartochke.emit(element)
            return
        super().mousePressEvent(sobytie)


class ReferenceDialog(QDialog):
    """«Кого ищем» целиком: сколько угодно фото, отметка лиц, число на каждой карточке."""

    def __init__(self, engine: FaceEngine, cache: FaceCache, max_dim: int,
                 nachalo: str, parent: QWidget | None = None,
                 nahodno: Itog | None = None,
                 vklad: dict[int, int] | None = None,
                 nazvanie: str | None = None,
                 chuzhie: frozenset[tuple[Path, int]] = frozenset()) -> None:
        super().__init__(parent)
        # Имя человека в заголовке: окно открывается по очереди для каждого из поиска, и
        # «выберите фото» без имени не отвечает на вопрос, ФОТО КОГО именно просят.
        self.setWindowTitle(f"Кого ищем: {nazvanie}" if nazvanie else "Выбор фото ребёнка")
        self.setModal(True)

        self.engine = engine
        self.cache = cache
        self._max_dim = int(max_dim)
        self._nachalo = str(nachalo or "") or str(Path.home())

        # Состояние, которое переживает нажатие «Добавить фото…» столько раз, сколько
        # человек его нажал. `lica_po_fajlam` копит файлы и задаёт плоский порядок
        # карточек; `otmetki` — что отметил человек, поверх автоотметок «лицо одно».
        # Лица, которые человек уже отдал другому поиску. Нужны ровно для одного:
        # автоотметка «на снимке одно лицо» не имеет права ставить галочку там, где
        # человек уже сказал, кто на этом снимке (задача T22).
        self._chuzhie = frozenset(chuzhie)
        self.lica_po_fajlam: dict[Path, list[Face]] = {}
        self.vse_fajly: list[Path] = []
        self.otmetki: set[tuple[Path, int]] = set()
        self.po_kazhdomu_licu = dict(vklad or {})
        # Сколько снимков нашлось по каждой СТРОКЕ эталона (1 = главное лицо).
        # Пусто до первого поиска: врать «найдено 0» там, где искать ещё не
        # начинали, значит отговорять от правильного лица.
        self.po_kazhdomu_licu: dict[int, int] = {}
        self._stroka_karty: dict[int, int] = {}
        self._poteri: dict[Path, str] = {}
        self._itog: Itog | None = None

        self.worker: ScanWorker | None = None
        # Окно крупного снимка и состояние отметки ДО последнего нажатия:
        # первый пресс двойного клика переворачивает галочку, и откатывать её
        # надо именно здесь, где это случилось.
        self._okno_prosmotra: PhotoViewDialog | None = None
        self._sostoyanie_pered_nazhatiem: tuple | None = None
        self._tekushhaya_partija: list[Path] = []
        # Признак «разбор идёт» ведёт окно, а не поток: `isRunning()` ложно и до `start()`,
        # и в зазоре между `run()` и `finished` — спросить поток значит один раз увидеть
        # живое окно с серыми кнопками, а другой раз с живыми.
        self._razbor_idyot = False
        self._ostanovlen = False
        self._obshaya_beda: str | None = None
        self._ostanovka_zaprosheyna = False
        self._zakrytie_otlozheno = False
        self._czel_zakrytiya = ""

        self.storona_kartochki = STORONA_KARTOKI
        self._kropy: dict[Path, list[np.ndarray] | None] = {}
        # Последнее слово `sobrat_etalon` по ключу карточки: нужно рамке на иконке,
        # которую перерисовывает и очередь кадров, и пересчёт после клика.
        self._karty_po_klucham: dict = {}
        self._ochered_kadrov: deque[Path] = deque()
        self._sinhr_kadrov = SINHRONNO_KADROV
        self._item_po_kluchu: dict[tuple[Path, int], QListWidgetItem] = {}
        self._idem_perestroyku = False

        self._build_ui()

        # Таймер — ДО `connect` ниже по коду: обработчик трогает `_taymer_kadrov`, и
        # висящий после конструктор означал бы AttributeError на сигнале во время сборки.
        self._taymer_kadrov = QTimer(self)
        self._taymer_kadrov.setInterval(OGRESH_KADRA_MS)
        self._taymer_kadrov.timeout.connect(self._dostavit_kadr)

        self.spisok.itemChanged.connect(self._na_otmetke)
        # Нажатие куда угодно по карточке, а не только в квадратик галочки: см.
        # `_SpisokLits`.
        self.spisok.nazhatie_po_kartochke.connect(self._po_nazhatiyu)
        self.spisok.dvojnoe_nazhatie.connect(self._na_dvojnom_klike)
        if nahodno is not None:
            self._vosstanovit(nahodno)
        self._obnovit_sostojanie()
        self.resize(780, 620)

    # --------------------------------------------------------------- наследие
    def _vosstanovit(self, nahodno: Itog) -> None:
        """Вернуть прежний выбор: те же файлы, те же лица, те же отметки.

        Ответ `Itog` — единственный документ, который главное окно хранит о выборе: в нём
        есть и файл, и номер лица, и отметка, и отвергнутые. Собирать состояние по
        `anchor`/`extra` было бы потерей половины картины — там нет ни путей файлов, ни
        неотмеченных лиц, ни снимков без лиц.

        Порядок карточек восстанавливается по порядку `karty`, а не по порядку ключей
        словаря: плоский порядок — это и есть договор с человеком о том, какое лицо
        «первое слева», и от него зависит, какое из отмеченных станет эталоном.

        Снимки лиц при этом читаются заново, через обычную очередь: отпечатки приезжают
        из ответа, а пикселей карточки в нём нет.
        """
        if not nahodno.karty:
            return
        for karta in nahodno.karty:
            lica = self.lica_po_fajlam.setdefault(karta.put, [])
            if karta.lic is not None:
                lica.append(karta.lic)
            if karta.put not in self.vse_fajly:
                self.vse_fajly.append(karta.put)
            if karta.otmecheno and karta.lic is not None:
                self.otmetki.add((karta.put, karta.nomer - 1))
        self._perestroit_setku()

    # --------------------------------------------------------------- сборка UI
    def _build_ui(self) -> None:
        self.add_button = QPushButton("Добавить фото…")
        self.add_button.setToolTip(PODSKAZKA_DOBAVIT)
        self.add_button.clicked.connect(self._dobavit_foto)
        self.stop_button = QPushButton("Отменить разбор")
        self.stop_button.setToolTip(PODSKAZKA_STOP)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._ostanovit_razbor)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        self.spisok = _SpisokLits()
        self.spisok.setViewMode(QListWidget.ViewMode.IconMode)
        self.spisok.setIconSize(QSize(self.storona_kartochki, self.storona_kartochki))
        self.spisok.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.spisok.setUniformItemSizes(True)
        self.spisok.setGridSize(QSize(self.storona_kartochki + 60,
                                      self.storona_kartochki + 124))
        self.spisok.setSpacing(12)
        self.spisok.setWordWrap(True)
        # Карточки нельзя ни таскать, ни выделять: галочка — единственное состояние
        # карточки, а синий квадрат Qt рядом с ней только спорил бы за внимание. Сдвинутая
        # мышью карточка путает порядок, а порядок карточек — это соответствие лица ему.
        self.spisok.setMovement(QListView.Movement.Static)
        self.spisok.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop)
        self.spisok.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.spisok.setToolTip(PODSKAZKA_KARTE)

        self.note = QLabel(TEKST_PUSTO)
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color: gray")

        boksy = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                 | QDialogButtonBox.StandardButton.Cancel)
        # Подписи ставим сами: стандартные кнопки Qt отдаёт по-английски, и для человека
        # без английского это тупик прямо в диалоге.
        self.ok_button = boksy.button(QDialogButtonBox.StandardButton.Ok)
        self.ok_button.setText("Искать по этим лицам")
        self.ok_button.setToolTip(PODSKAZKA_OK)
        self.ok_button.setEnabled(False)
        self.cancel_button = boksy.button(QDialogButtonBox.StandardButton.Cancel)
        self.cancel_button.setText("Отмена")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        odna_stroka = QHBoxLayout()
        odna_stroka.addWidget(self.add_button)
        odna_stroka.addWidget(self.stop_button)
        odna_stroka.addStretch(1)
        layout = QVBoxLayout(self)
        layout.addLayout(odna_stroka)
        layout.addWidget(self.progress)
        layout.addWidget(self.status_label)
        layout.addWidget(self.spisok, 1)
        layout.addWidget(self.note)
        layout.addWidget(boksy)

    # --------------------------------------------------------------- наружу
    @property
    def itog(self) -> Itog | None:
        """Последний пересчёт: ровно то, что человек видит на экране."""
        return self._itog

    @classmethod
    def sprosit(cls, parent: QWidget | None, engine: FaceEngine, cache: FaceCache,
                max_dim: int, nachalo: str, nahodno: Itog | None = None,
                vklad: dict[int, int] | None = None,
                nazvanie: str | None = None,
                chuzhie: frozenset[tuple[Path, int]] = frozenset()) -> Itog | None:
        """Модально спросить эталон. None — отмена или закрытие крестиком.

        Отмена обязана вернуть именно None, а не пустой `Itog`: главному окну нужно
        отличить «человек передумал, прежний эталон в силе» от «эталона больше нет».

        `nahodno` — прежний ответ, если он был. Без него окно открывается пустым и
        человек, который зашёл посмотреть, что он тогда отметил, видит пустоту там, где
        поиск уже идёт по четырём лицам.

        `nazvanie` — кого сейчас ищем: окно одно на всех, и без имени в заголовке
        вторая партия фото уехала бы не тому человеку.

        `chuzhie` — лица, уже отмеченные за другим человеком: автоотметка «на снимке
        одно лицо» их пропускает. Без этого тот же набор снимков, принесённый для мамы,
        сам себя отмечал как ребёнка.
        """
        dlg = cls(engine, cache, max_dim, nachalo, parent, nahodno=nahodno,
                  vklad=vklad, nazvanie=nazvanie, chuzhie=chuzhie)
        if int(dlg.exec()) != int(QDialog.DialogCode.Accepted):
            return None
        return dlg.itog

    # --------------------------------------------------------------- файлы
    def _dobavit_foto(self) -> None:
        """Сколько угодно файлов за раз — то, ради чего это окно и существует.

        Повторы снимаются на входе: тот же файл второй раз удвоил бы и карточки, и цену
        разбора, а отличить «я его уже приносил» человеку по сетке невозможно.
        """
        if self._potok_zanimaetsja():
            return                       # кнопка выключена, но слот живёт и из теста
        imena, _ = QFileDialog.getOpenFileNames(
            self, "Фото ребёнка — можно выбрать несколько", self._nachalo, FILTER_FOTO)
        novye = [put for put in dict.fromkeys(Path(s) for s in imena)
                 if put not in self.vse_fajly]
        if not novye:
            return
        self._nachalo = str(novye[0].parent)
        self._prinjat_fajly(novye)

    def _prinjat_fajly(self, fajly: Sequence[Path]) -> None:
        """Файлы приняты и уходят на разбор.

        Сетка здесь не перестраивается: карточки новой партии появляются с ответом
        потока, а старые за это время обязаны остаться на экране.
        """
        self.vse_fajly.extend(fajly)
        self._razbirat(list(fajly))

    def _razbirat(self, fajly: list[Path]) -> None:
        self._tekushhaya_partija = list(fajly)
        self._ostanovlen = False
        self._obshaya_beda = None
        self._ostanovka_zaprosheyna = False
        self._razbor_idyot = True
        self.progress.setMaximum(max(1, len(fajly)))
        self.progress.setValue(0)
        self.status_label.setText(f"разбираем {len(fajly)} фото…")
        self.worker = ScanWorker(fajly, self._ensure_engine(), self.cache, self._max_dim,
                                 parent=self)
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self._on_scan_done)
        self.worker.error.connect(self._on_scan_error)
        self._obnovit_sostojanie()
        self.worker.start()

    def _ensure_engine(self) -> FaceEngine:
        """Движок берётся у главного окна и создаётся только если его ещё нет.

        `make_engine` ничего с диска не читает (модели поднимаются в потоке), но второй
        объект означал бы вторую загрузку весов на того же человека.
        """
        if self.engine is None:
            from core.engine import make_engine
            from utils.config import buffalo_l_dir, yunet_model_path
            self.engine = make_engine("both", yunet_model_path(), buffalo_l_dir())
        return self.engine

    def _ostanovit_razbor(self) -> None:
        """Просьба встать после текущего фото.

        Отмена — не «лиц нет»: файлы, которые не досмотрены, после `done` называются
        остановленными, а не испорченными (см. `_slovo_zaglushki`).
        """
        if self.worker is not None:
            self.worker.request_stop()
            self._ostanovlen = True
            self.stop_button.setEnabled(False)
            self.status_label.setText("останавливаю после текущего фото…")

    def _on_progress(self, sdelano: int, vsego: int, imya: str) -> None:
        self.progress.setMaximum(max(1, vsego))
        self.progress.setValue(sdelano)
        self.status_label.setText(f"обработано {sdelano} из {vsego}: {imya}")

    # --------------------------------------------------------------- ответ потока
    def _on_scan_done(self, payload: object) -> None:
        photos, statistika = payload                        # type: ignore[misc]
        prishedshie = [put for put in self._tekushhaya_partija if put in photos]
        for put in prishedshie:
            self.lica_po_fajlam[put] = photos[put]
        # Автоотметка «лицо на снимке одно» — только про НОВЫЕ файлы. Прогнать её по всему
        # списку значило бы вернуть галочку, которую человек только что снял руками.
        self.otmetki.update(odinochnye_otmetki({put: photos[put] for put in prishedshie},
                                               zanyatye=self._chuzhie))
        prichiny = dict(statistika.failures)
        for put in self._tekushhaya_partija:
            if put in photos:
                self._poteri.pop(put, None)       # перечитался — причины больше нет
            elif str(put) in prichiny:
                self._poteri[put] = str(prichiny[str(put)])
        # Итог разбора называется, а не молчит: «разбор остановлен» и «разобрано N» — это
        # два разных продолжения для человека, и первое он обязан увидеть строкой.
        vsego = len(self._tekushhaya_partija)
        self.status_label.setText(
            (f"разбор остановлен: досмотрено {len(photos)} фото из {vsego}"
             if self._ostanovlen else f"разобрано фото: {len(photos)}"))
        self._konets_potoka()

    def _on_scan_error(self, soobshchenie: str) -> None:
        """Текст рабочего потока показывается КАК ЕСТЬ (см. `main_window._on_scan_error`).

        «Не поднялись веса» и «оглавление встало» — это два разных разговора и два разных
        действия человека; собственный заголовок здесь сделал бы из одного другое. Файлы
        партии остаются карточками: они не разобрались, и сказать надо именно это.
        """
        self._obshaya_beda = soobshchenie
        for put in self._tekushhaya_partija:
            self._poteri.setdefault(put, soobshchenie)
        self._konets_potoka()
        self.status_label.setText(soobshchenie)
        self._warn(soobshchenie)

    def _konets_potoka(self) -> None:
        self._tekushhaya_partija = []
        self._razbor_idyot = False
        self._perestroit_setku()
        self._obnovit_sostojanie()

    # --------------------------------------------------------------- сетка карточек
    def _pereschislit(self) -> Itog:
        """Единственная точка, откуда зовут арифметику.

        Отметки уходят `frozenset`-ом и решают только состав, а не порядок: anchor
        выбирается по размеру и плоскому порядку файлов. Тот же набор отметок обязан
        давать тот же эталон — за этим следит отдельный тест.
        """
        self._itog = sobrat_etalon(self.lica_po_fajlam, frozenset(self.otmetki))
        # Порядок строк эталона — договор `make_reference`: 1 = anchor, дальше `extra`
        # в его порядке. Держим соответствие здесь, в единственном месте: если заведём
        # второй пересчёт, он разъедется с тем, что уже посчитано в сетке результатов.
        self._stroka_karty = {id(self._itog.anchor): 1} if self._itog.anchor else {}
        self._stroka_karty.update({id(k): i + 2
                                   for i, k in enumerate(self._itog.extra)})
        return self._itog

    def _perestroit_setku(self) -> None:
        """Собрать сетку заново: порядок карточек задает порядок файлов, а не мышь."""
        self._pereschislit()
        gruppy = self._karty_po_puti()
        self._idem_perestroyku = True
        try:
            self.spisok.clear()
            self._item_po_kluchu = {}
            for put in self.vse_fajly:
                karty = gruppy.get(put)
                if karty is None:                     # файла нет в ответе потока
                    self._dobavit_zaglushku(put)
                    continue
                for karta in karty:
                    if karta.lic is None:             # прочитан, но лиц нет
                        self._dobavit_zaglushku(put, karta)
                    else:
                        self._dobavit_kartochku(karta, len(karty))
        finally:
            self._idem_perestroyku = False
        self._obnovit_kartochki()
        self._nakanune_kadrov()
        self._obnovit_itogi()

    def _karty_po_puti(self) -> dict[Path, list[Kartochka]]:
        """Карточки `Itog`, сгруппированные по файлу; плоский порядок сохранён."""
        gruppy: dict[Path, list[Kartochka]] = {}
        if self._itog is not None:
            for karta in self._itog.karty:
                gruppy.setdefault(karta.put, []).append(karta)
        return gruppy

    def _kluch(self, put: Path, nomer: int) -> tuple[Path, int]:
        return (put, nomer)

    def _dobavit_kartochku(self, karta: Kartochka, vsego_v_fajle: int) -> None:
        element = self._novyj_element(karta.put, karta.nomer, markiruetsja=True)
        self._pokazat_krop(element, karta.put, karta.nomer)

    def _dobavit_zaglushku(self, put: Path, karta: Kartochka | None = None) -> None:
        """Карточка файла без лиц и файла, который не разобрался, — видимы обе.

        У заглушки нет лица, значит и галочки на ней быть не может: отметить нечего, а
        живая галочка без лица обещала бы поиск по пустоте.
        """
        element = self._novyj_element(put, karta.nomer if karta is not None else 0,
                                      markiruetsja=False)
        self._pokazat_krop(element, put, 0)

    def _novyj_element(self, put: Path, nomer: int,
                       markiruetsja: bool) -> QListWidgetItem:
        flags = Qt.ItemFlag.ItemIsEnabled
        if markiruetsja:
            flags |= Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable
        element = QListWidgetItem()
        element.setData(Qt.ItemDataRole.UserRole, self._kluch(put, nomer))
        element.setFlags(flags)
        if markiruetsja:
            element.setCheckState(Qt.CheckState.Unchecked)
        self.spisok.addItem(element)
        self._item_po_kluchu[self._kluch(put, nomer)] = element
        return element

    def _pokazat_krop(self, element: QListWidgetItem, put: Path, nomer: int) -> None:
        """Иконка карточки: кроп лица с номером или серая плитка, пока кадр не дошёл.

        Номер рисуется прямо на кадре: подпись под карточкой человек не читает, когда
        карточек десять, а «второе слева» надо тыкнуть (см. `face_picker.naklej_nomer`).
        У карточки-заглушки номера нет — на файле без лиц показывать нечего.
        """
        storona = self.storona_kartochki
        kropy = self._kropy.get(put)
        kadr = kropy[nomer - 1] if kropy and 1 <= nomer <= len(kropy) else None
        if kadr is None:
            kadr = np.full((storona, storona, 3), PLITKA, dtype=np.uint8)
            element.setIcon(QIcon(face_to_qpixmap(kadr)))
            return
        # Рамка по краю — про то, ищется это лицо или нет. Галочку в углу карточки
        # человек на сетке из двадцати снимков не находит и не может сказать, по какому
        # лицу из десяти идёт поиск (задача T23).
        tsvet = ZELENYJ if self._ishetsya(self._kluch(put, nomer)) else RAMKA_KRASNYJ
        element.setIcon(QIcon(face_to_qpixmap(naklej_nomer(obvesti(kadr, tsvet), nomer))))

    def _ishetsya(self, kluch) -> bool:
        """Отмечено это лицо и годно к сравнению — то есть по нему ищут."""
        karta = self._karty_po_klucham.get(kluch)
        return bool(karta is not None and karta.v_poisk)

    # --- очередь кадров --------------------------------------------------------------
    def _nakanune_kadrov(self) -> None:
        """Файлы, чьи кропы ещё не прочитаны, в очередь; один читается сразу.

        Иконки перерисовываются послЕ синхронного чтения: карточка создаётся без кадра, и
        без этого шага первый же снимок остался бы серой плиткой до первого тика таймера.
        """
        self._ochered_kadrov = deque(
            put for put in self.lica_po_fajlam
            if put not in self._kropy and self.lica_po_fajlam[put])
        self._sinhr_kadrov = SINHRONNO_KADROV
        while self._ochered_kadrov and self._sinhr_kadrov > 0:
            self._sinhr_kadrov -= 1
            self._chitat_kadr(self._ochered_kadrov.popleft())
        self._pererisovat_kropy()
        self._zapatit_ochered()

    def _pererisovat_kropy(self) -> None:
        """Разложить готовые кропы по карточкам (и вернуть плитку там, где кадра ещё нет)."""
        for (put, nomer), element in self._item_po_kluchu.items():
            self._pokazat_krop(element, put, nomer)

    def _chitat_kadr(self, put: Path) -> None:
        """Кропы одного файла; None в ответе — кадр сегодня не читается.

        Файл мог разобраться вчера и не открыться сегодня (диск отвалился, файл правили).
        Это не повод ронять окно: карточка остаётся с отметкой и числом, только без
        фотографии, и говорит об этом словами.
        """
        lica = self.lica_po_fajlam.get(put) or []
        if not lica:
            self._kropy[put] = []
            return
        try:
            image = load_photo(put, max_dim=self._max_dim)
        except Exception:                                  # noqa: BLE001 — не ронять окно
            image = None
        if image is None:
            self._kropy[put] = None
            return
        try:
            self._kropy[put] = crops_from(image, lica)
        except Exception:                                  # noqa: BLE001 — чужой формат кадра
            self._kropy[put] = None

    def _zapatit_ochered(self) -> None:
        """Пустить очередь, если читать есть что и если на это смотрят.

        Невидимому окну кадры не нужны: диалог переживает чужое окно, свёрнутое или
        закрытое человеком, и читать в эту минуту архив значило бы жечь диск про запас.
        `showEvent` догонит очередь, как только сетку покажут (тот же порядок, что в
        `results_view`).
        """
        if self._ochered_kadrov and self.spisok.isVisible():
            if not self._taymer_kadrov.isActive():
                self._taymer_kadrov.start()
        else:
            self._taymer_kadrov.stop()

    def _dostavit_kadr(self) -> None:
        """Один кадр за тик событий: между двумя чтениями окно успевает ответить на клик."""
        if not self._ochered_kadrov:
            self._taymer_kadrov.stop()
            return
        put = self._ochered_kadrov.popleft()
        self._chitat_kadr(put)
        for (p, nomer), element in self._item_po_kluchu.items():
            if p == put:
                self._pokazat_krop(element, put, nomer)
        self._obnovit_kartochki()                          # подсказка про дочитанный кадр
        self._zapatit_ochered()

    # --------------------------------------------------------------- подписи
    def _obnovit_kartochki(self) -> None:
        """Подписи, галочки и «погасание» существующих карточек по новому `Itog`.

        Карточки не пересоздаются: перестройка сетки на каждый клик сбрасывала бы
        прокрутку, и человек работал бы с экраном, который уезжает из-под курсора.
        """
        if self._itog is None:
            return
        po_klucham = {self._kluch(k.put, k.nomer): k for k in self._itog.karty}
        self._karty_po_klucham = po_klucham
        vsego_po_puti = {put: len(karty) for put, karty in self._karty_po_puti().items()}
        self._idem_perestroyku = True
        try:
            for kluch, element in self._item_po_kluchu.items():
                karta = po_klucham.get(kluch)
                if karta is None or karta.lic is None:
                    self._opodpit_zaglushku(element, kluch[0])
                    continue
                element.setCheckState(Qt.CheckState.Checked if karta.otmecheno
                                      else Qt.CheckState.Unchecked)
                self._opodpit_lico(element, karta, vsego_po_puti.get(karta.put, 1))
        finally:
            self._idem_perestroyku = False

    def _opodpit_lico(self, element: QListWidgetItem, karta: Kartochka,
                      vsego_v_fajle: int) -> None:
        """Что написано на карточке лица.

        Неотмеченной карточке число не показывают: до отметки сравнивать ещё не с чем, а
        «0 %» на её месте человек прочитал бы как «не похоже».

        Подпись участия стоит на КАЖДОЙ отмеченной карточке, а не только на anchor:
        ровно этого на экране и не хватало, чтобы человек решил, будто поиск идёт по
        одному лицу (задача T14). Слов без числа остаётся одна ветка — непригодный
        отпечаток: нуль там значит не «не похоже», а «не чем считать».
        """
        stroki = [karta.put.name, f"лицо {karta.nomer} из {vsego_v_fajle}"]
        goden = karta.lic is not None and prigoden_otpechatok(karta.lic.embedding)
        if karta.otmecheno:
            if not goden:
                stroki += [f"{SLOVO_NECHEM} · {SLOVO_NE_ISSKEM}"]
            else:
                # Числа может не быть: лицо единственное, сравнивать его не с чем.
                chislo = None if karta.procent is None else f"{karta.procent} %"
                znak = f"{SLOVO_ISSKEM} · {SLOVO_SLABO}" if karta.slabo else SLOVO_ISSKEM
                stroki += [chast for chast in (chislo, znak) if chast]
            vklad = self._vklad(karta)
            if vklad is not None:
                stroki.append(vklad)
        element.setText("\n".join(stroki))
        element.setToolTip(self._podskazka_lica(karta, goden, vsego_v_fajle))
        # Рамка на иконке обязана перекраситься в тот же миг, что и галочка: иначе
        # единственный видимый признак решения остаётся вчерашним.
        self._pokazat_krop(element, karta.put, karta.nomer)
        poveryalas = bool(karta.otmecheno) and (not karta.v_poisk or karta.slabo)
        element.setData(Qt.ItemDataRole.ForegroundRole,
                        QBrush(CVET_POVERYAVSHEJ) if poveryalas else None)

    def _vklad(self, karta: Kartochka) -> str | None:
        """Сколько снимков нашлось по этому лицу. None — чисел ещё нет.

        Считается по строкам эталона, а не по карточкам: карточка может быть отмечена,
        но в эталон не попасть (непригодный отпечаток), и тогда вклада у неё нет.
        """
        if not self.po_kazhdomu_licu:
            return None
        stroka = self._stroka_karty.get(id(karta))
        if stroka is None:
            return None
        n = self.po_kazhdomu_licu.get(stroka, 0)
        if n == 0:
            return FRAZA_VKLAD_NOL
        if n == 1:
            return FRAZA_VKLAD_ODIN
        return FRAZA_VKLAD.format(n=n)

    def _podskazka_lica(self, karta: Kartochka, goden: bool, vsego_v_fajle: int) -> str:
        """Полная фраза в подсказке: здесь число названо словами, а не только цифрой."""
        osnova = f"{karta.put}\nлицо {karta.nomer} из {vsego_v_fajle} на этом снимке"
        if self._kropy.get(karta.put) is None:
            osnova += f"\n{SLOVO_NE_CHITAETSYA}"
        if not karta.otmecheno:
            return osnova + "\nне отмечено — в поиске не участвует"
        if not goden:
            return (osnova + f"\nотмечено, но {SLOVO_NECHEM}: лицо снято слишком мелко или "
                    f"не в фокусе, поэтому {SLOVO_NE_ISSKEM}. Отметьте снимок, где этот "
                    "человек виден крупнее")
        if karta.procent is None:
            return (osnova + f"\nотмечено, {SLOVO_ISSKEM}. Сравнить это лицо с другими "
                    "отмеченными нечем: оно на отметках единственное")
        if karta.slabo:
            return (osnova + f"\nотмечено, на {karta.procent} % похоже на остальные "
                            f"отмеченные лица — {SLOVO_SLABO}. В поиске оно участвует "
                            "наравне с остальными, но перепроверьте: возможно, это "
                            "другой человек и его надо отмечать отдельной строкой")
        return (osnova + f"\nотмечено, на {karta.procent} % похоже на остальные "
                        f"отмеченные лица — {SLOVO_ISSKEM}")

    def _opodpit_zaglushku(self, element: QListWidgetItem, put: Path) -> None:
        element.setText("\n".join([put.name, self._slovo_zaglushki(put)]))
        element.setToolTip(self._podskazka_zaglushki(put))
        element.setData(Qt.ItemDataRole.ForegroundRole, QBrush(CVET_POVERYAVSHEJ))

    def _slovo_zaglushki(self, put: Path) -> str:
        if put in self.lica_po_fajlam:
            return SLOVO_NET_LIC
        if self._ostanovlen and put not in self._poteri:
            return SLOVO_OSTANOVLEN
        return SLOVO_NE_RAZOBRALOS

    def _podskazka_zaglushki(self, put: Path) -> str:
        """Неразобранный файл обязан назвать причину: «файл не читается» и «оглавление
        перестало отвечать» — это две разные починки, а не одна и та же беда."""
        if put in self.lica_po_fajlam:
            return f"{put}\n{SLOVO_NET_LIC} — снимок прочитан, лиц на нём нет"
        if self._ostanovlen and put not in self._poteri:
            return (f"{put}\n{SLOVO_OSTANOVLEN} — разбор остановлен, этот снимок не "
                    "досмотрен. Добавьте его снова или дождитесь конца разбора")
        prichina = self._poteri.get(put) or self._obshaya_beda
        if prichina:
            return f"{put}\n{SLOVO_NE_RAZOBRALOS}: {prichina}"
        return f"{put}\n{SLOVO_NE_RAZOBRALOS}"

    # --------------------------------------------------------------- отметки
    def _na_otmetke(self, element: QListWidgetItem) -> None:
        """Единственная точка, куда приходит смена отметки — чья бы она ни была.

        Пересчёт — весь ответ целиком, а не правка одной карточки: отметка меняет anchor,
        а с ним числа на всех остальных, и половинчатый пересчёт оставил бы на экране
        вчерашние проценты.
        """
        if self._idem_perestroyku:
            return
        kluch = element.data(Qt.ItemDataRole.UserRole)
        if not isinstance(kluch, tuple) or len(kluch) != 2:
            return
        put, nomer = kluch
        para = (put, int(nomer) - 1)                  # отметка — индекс лица в списке файла
        if element.checkState() == Qt.CheckState.Checked:
            self.otmetki.add(para)
        else:
            self.otmetki.discard(para)
        self._pereschislit()
        self._obnovit_kartochki()
        self._obnovit_itogi()

    def _po_nazhatiyu(self, element: QListWidgetItem) -> None:
        """Отметить или снять карточку по нажатию куда угодно по её телу.

        Единственный вход для жеста — сигнал `_SpisokLits`, который уже отобрал карточки,
        годные под отметку: на заглушке «лицо не найдено» отмечать нечего, и нажатие по
        ней до сюда не доходит.

        Прежнее состояние запоминается нарочно: первый пресс двойного клика Qt считает
        одиночным, и отметка перевернулась бы до того, как станет ясно, что это был
        двойной клик.
        """
        self._sostoyanie_pered_nazhatiem = (element, element.checkState())
        element.setCheckState(Qt.CheckState.Unchecked
                              if element.checkState() == Qt.CheckState.Checked
                              else Qt.CheckState.Checked)

    def _na_dvojnom_klike(self, element: QListWidgetItem) -> None:
        """Двойной клик по карточке: вернуть отметку как было и показать снимок целиком.

        Без отката жест делал две вещи, а просили одну: окно открылось, а галочка
        незаметно перевернулась. Закрыто окно — человек видит, что лицо, которое он
        только что отмечал, снято, и решает, что программа его не слушается.

        Откатывается по идентичности объекта, а не по пути: карточки при пересчёте не
        пересоздаются (см. `_obnovit_kartochki`), поэтому объект — единственный честный
        признак «та же самая карточка».
        """
        zapomnennoe = self._sostoyanie_pered_nazhatiem
        if zapomnennoe is not None and zapomnennoe[0] is element:
            self._sostoyanie_pered_nazhatiem = None
            element.setCheckState(zapomnennoe[1])
        self._otkryt_prosmotr(element)

    def _otkryt_prosmotr(self, element: QListWidgetItem) -> None:
        """Показать снимок в том же окне, что и список находок.

        Рамками покрываются ВСЕ лица этого снимка и каждое подписывается номером — тем
        же, что на карточке. Одна рамка здесь не отвечает на вопрос, который человек и
        задаёт: «какое из трёх лиц — мой ребёнок».

        Галочки копирования нет: в этом окне решают, кого искать, а не что переносить.
        """
        kluch = element.data(Qt.ItemDataRole.UserRole)
        if not isinstance(kluch, tuple) or len(kluch) != 2:
            return
        put, nomer = kluch
        karta = self._karta_iz_itoga(put, nomer)
        if karta is None or karta.lic is None:
            return
        sroki = [k for k in self._itog.karty if k.put == put and k.lic is not None]
        ramki = [RamkaLica(k.lic.box, k.nomer) for k in sroki]
        poziciya = next((i + 1 for i, k in enumerate(self._itog.karty)
                         if k.put == put and k.nomer == nomer), 0)
        if poziciya == 0:
            return
        if self._okno_prosmotra is None:
            self._okno_prosmotra = PhotoViewDialog(self)
            self._okno_prosmotra.prosyat_perehod.connect(self._perehod_prosmotra)
        self._okno_prosmotra.pokazat(put, ramki, nomer, self._max_dim, karta.procent,
                               poziciya, len(self._itog.karty))

    def _karta_iz_itoga(self, put: Path, nomer: int) -> Kartochka | None:
        """Карточка `Itog` по файлу и номеру лица. None — ответа ещё нет."""
        if self._itog is None:
            return None
        return next((k for k in self._itog.karty
                     if k.put == put and k.nomer == nomer), None)

    def _perehod_prosmotra(self, shag: int) -> None:
        """Стрелки в окне просмотра идут по тому же списку карточек, что и сетка.

        Здесь «следующее фото» — это следующее ЛИЦО: на групповом снимке их несколько,
        и листать именно по лицам, когда выбираешь, кого искать, естественнее, чем
        перескакивать через снимок целиком.
        """
        if self._itog is None or not self._itog.karty:
            return
        tekushhij = self._okno_prosmotra.pokazyvaet
        nomer = next((i for i, k in enumerate(self._itog.karty)
                      if k.put == tekushhij and k.nomer == self._okno_prosmotra.pokazyvaemoe_lice), None)
        if nomer is None:
            return
        sosednij = nomer + (1 if shag > 0 else -1)
        if not 0 <= sosednij < len(self._itog.karty):
            return
        karta = self._itog.karty[sosednij]
        if karta.lic is None:
            return
        self._otkryt_prosmotr(self._karta(karta.put, karta.nomer))

    def _karta(self, put: Path, nomer: int) -> QListWidgetItem | None:
        """Карточка файла `put` с номером `nomer` (с единицы; 0 — заглушка всего файла)."""
        return self._item_po_kluchu.get(self._kluch(put, nomer))

    # --------------------------------------------------------------- состояние
    def _potok_zanimaetsja(self) -> bool:
        """Идёт ли разбор СЕЙЧАС. Отвечает признак окна, а не поток — см. `__init__`."""
        return self._razbor_idyot

    def _obnovit_sostojanie(self) -> None:
        """Кнопки — из одного места: посреди разбора не добавляют и не подтверждают."""
        zanyato = self._potok_zanimaetsja()
        self.add_button.setEnabled(not zanyato)
        self.stop_button.setEnabled(zanyato)
        self._obnovit_itogi()

    def _obnovit_itogi(self) -> None:
        """Слова про то, что уйдёт в поиск, и жива ли кнопка.

        ОК мертва ровно в трёх случаях: нет файлов, нет отметок, у всех отмеченных
        отпечаток непригоден. Каждый назван, а не молчит: «кнопка не нажимается» без
        объяснения человек читает как сломанное приложение.

        Прежняя сводка заканчивалась словами «не пойдёт: 7 — похоже меньше чем на 40 %» и
        тем самым называла отказом то, что отказом больше не является: все отмеченные и
        годные лица участвуют в поиске, а низкое число теперь предупреждение, и живёт оно
        в своей части фразы.
        """
        if self._potok_zanimaetsja():
            self.note.setText(TEKST_RAZBOR)
            self.ok_button.setEnabled(False)
            return
        itog = self._itog
        if not self.vse_fajly or itog is None:
            self.note.setText(TEKST_PUSTO)
            self.ok_button.setEnabled(False)
            return

        otmechennye = [k for k in itog.karty if k.otmecheno]
        if itog.reference is None:
            # Причина мёртвой кнопки называется одна, а не дважды: «ни одного» уже говорит
            # и про число отметок, и про то, что не ищем ровно всё.
            osnova = (TEKST_BEZ_OTMETOK if not otmechennye
                      else TEKST_VSE_NEPRIGODNY.format(n=len(otmechennye)))
            self.note.setText("; ".join([osnova] + self._hvost_o_fajlah()) + ".")
            self.ok_button.setEnabled(False)
            return
        slabyh = sum(1 for karta in itog.karty if karta.slabo)
        chasti = [FRAZA_V_POISKE.format(poisk=itog.reference.count,
                                        vsego=len(otmechennye))]
        if slabyh:
            chasti.append(FRAZA_SLABO.format(n=slabyh))
        if itog.otvergnutye:
            chasti.append(FRAZA_NECHEM.format(n=len(itog.otvergnutye)))
        self.note.setText("; ".join(chasti + self._hvost_o_fajlah()) + ".")
        self.ok_button.setEnabled(True)

    def _hvost_o_fajlah(self) -> list[str]:
        """Фразы про файлы, которые человек принёс, но которые не дошли до разбора.

        Остановленный хвост и неразобранный файл — это разные беды: первую человек
        выбрал сам, вторую ему чинить. Одно слово на две ситуации отправило бы человека
        починкой не той (спецификация, раздел 7).
        """
        ne_razobralos = sum(1 for put in self.vse_fajly if put not in self.lica_po_fajlam)
        if not ne_razobralos:
            return []
        if self._ostanovlen:
            return [FRAZA_OSTANOVLEN.format(n=ne_razobralos)]
        return [FRAZA_NE_RAZOBRALOS.format(n=ne_razobralos)]

    # --------------------------------------------------------------- закрытие
    def done(self, rezultat: int) -> None:
        """Любое завершение диалога — и ОК, и отмена — уносит с собой окно просмотра.

        Крупный снимок принадлежит выбору лица. Пережив диалог, он остался бы висеть на
        экране с рамкой и процентом от выбора, который человек уже подтвердил или
        отменил, и следующим двойным кликом его было бы не отличить от нового.
        """
        self._zakryt_prosmotr()
        super().done(rezultat)

    def _zakryt_prosmotr(self) -> None:
        if self._okno_prosmotra is not None:
            self._okno_prosmotra.close()

    def accept(self) -> None:
        """ОК переносит выбор и закрывает окно: нового разбора архива не будет.

        Пересчёт здесь обязателен: наружу обязан уйти ровно тот ответ, который человек
        видел на экране, а не его версия до последнего движения мыши.
        """
        if self._potok_zanimaetsja():
            self._warn("разбор ещё идёт — дождитесь или нажмите «Отменить разбор»")
            return
        itog = self._pereschislit()
        if itog.reference is None:
            self._obnovit_itogi()
            return
        super().accept()

    def reject(self) -> None:
        """Отмена: главное окно остаётся со своим прежним эталоном и своими процентами."""
        if not self._potok_zavershit():
            self._czel_zakrytiya = "reject"
            return
        super().reject()

    def closeEvent(self, event) -> None:                       # noqa: N802 (имя Qt)
        if not self._potok_zavershit():
            self._czel_zakrytiya = "close"
            event.ignore()
            return
        self._taymer_kadrov.stop()
        self._zakryt_prosmotr()
        super().closeEvent(event)

    def showEvent(self, event) -> None:                        # noqa: N802 (имя Qt)
        """Окно показали — очередь кадров, стоявшая до этого, догоняет с первого тика."""
        super().showEvent(event)
        self._zapatit_ochered()

    def hideEvent(self, event) -> None:                        # noqa: N802 (имя Qt)
        """Окно убрали — читать нечего: очередь кадров глушится, данные остаются."""
        self._taymer_kadrov.stop()
        super().hideEvent(event)

    def _potok_zavershit(self) -> bool:
        """Остановить свой поток и ответить, можно ли закрывать окно.

        `wait()` возвращает False ровно тогда, когда поток ещё жив. Закрыть окно в эту
        минуту значит убить родителя `QThread`: человек увидел бы «QThread: Destroyed
        while thread is still running», а недописанная строка оглавления осталась бы
        потерянной. Поэтому просим встать, ЖДЁМ и спрашиваем результат; не дождался —
        отменяем закрытие и возвращаемся к нему на `finished`.

        Оглавление здесь НЕ закрывается: база принадлежит главному окну и обязана
        пережить это диалоговое.
        """
        if self._zakrytie_otlozheno:
            return False          # отложенное закрытие уже назначено — второй раз не ждём
        potok = self.worker
        if potok is None or not potok.isRunning():
            return True
        if not self._ostanovka_zaprosheyna:
            potok.request_stop()
            self._ostanovka_zaprosheyna = True
            self.status_label.setText("останавливаю разбор окна выбора фото…")
        if potok.wait(OZHDANIE_ZAKRYTIJA_MS):
            return True
        self._zakrytie_otlozheno = True
        potok.finished.connect(self._zakrytsja_posle_potoka)
        return False

    def _zakrytsja_posle_potoka(self) -> None:
        """Поток встал — доделываем то закрытие, которое человек уже попросил."""
        if not self._zakrytie_otlozheno:
            return
        self._zakrytie_otlozheno = False
        czel, self._czel_zakrytiya = self._czel_zakrytiya, ""
        if czel == "close":
            self.close()
        else:
            self.reject()

    def _warn(self, text: str) -> None:
        QMessageBox.warning(self, "Внимание", text)
