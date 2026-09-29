"""Одно окно, три шага сверху вниз. Тяжёлая работа — в `ScanWorker`.

Исключение из этой фразы ровно одно, и оно объявлено: копирование отмеченных снимков
идёт в потоке интерфейса и на сотнях файлов занимает минуты. Почему так — пункт
**Копирование гасит окно заранее** ниже; молчать про это заголовок не имеет права.

Смена похожести пересчитывает проценты из памяти: лица уже разобраны и лежат в
оглавлении, нового разбора нет (спецификация, раздел 6). Поэтому здесь два правила,
которые легко нарушить именно на сборке окна.

**Ползунок не запускает скан.** Единственная точка, откуда стартует фоновая работа этого
окна, — `_run`, и попадает в неё только кнопка «Начать разбор папки» (смена качества
разбора просит тот же прогон). Выбор эталона у окна разбора ничего не просит: у окна
эталона свой поток (`ui/reference_dialog.py`), а здесь остаётся перенос готового ответа.
Пересчёт (`rebuild_scores`) не создаёт ни рабочего потока, ни движка: тесты следят за
этим через подменённый `ScanWorker`, а не «на глаз».

**Шторм сигналов сворачивается в одно действие.** При перетаскивании полоса шлёт
`changed` на каждое деление — до тридцати штук за жест, и каждое в прежнем виде значило
`QSettings.sync()` на диск плюс полную пересборку сетки. Человек видел бы тридцать
миганий и слышал бы диск, а итог тот же. Теперь сигнал только перезапускает таймер на
`DEBAUN_MS`, и когда значение остановилось, происходит ОДНО применение: один пересчёт и
одна запись настроек.

Что здесь ещё решено явно.

* **Копирование гасит окно заранее.** `copy_photos` — цикл `shutil.copy2` без callbacks
  отмены и без прогресса, и рабочий поток он бы получил, не получив ни отмены, ни
  процентов: то есть ещё одно состояние окна ценой в ноль новых возможностей. Цена
  настоящая — минуты занятого потока интерфейса на сотне телефонных снимков, и против
  неё работает честный занятый вид: до первого байта копирования окно пишет в строку
  состояния «копируем N фото в …», отрисовывает это и выключает решительно всё, что
  можно выключить, — папки, полосу настроек, сетку с галочками, обе кнопки запуска и саму
  кнопку копирования. Спецификация (раздел 7) держит «без вида зависшего окна», и вид
  занятого окна — это не «серые кнопки», а обещание, которое окно может выполнить.
* **Сообщение рабочего потока показывается как есть.** `worker` различает «не удалось
  загрузить модели…» и «оглавление недоступно…, уже снятые лица сохранены» — это два
  разных разговора с человеком и два разных его действия. Свой заголовок здесь сделал бы
  из одного другое. На ошибке сетка НЕ очищается: частичный результат в силе.
* **Накопитель потерь живёт ровно один поиск.** `ScanStats` приходит на каждый прогон
  свой, а отчёт пишется один раз и позже — поэтому потери копятся по путям (`_poteri`,
  `_otbroennye`) и в `report.txt` уходят те же числа, что видит разбор. Оба счётчика по
  путям чистятся, когда снимок разобрался на этом прогоне: иначе одно неудачное лицо
  кочевало бы из отчёта в отчёт вечно.
* **Занятое окно не принимает настроек.** Посреди разбора полоса гаснет целиком, а
  просьба, поданная за миг до старта, применяется, когда поток встанет, — а не пишет в
  окно и в файл новое качество разбора поверх лиц, снятых в старом (тогда рамка лица
  уезжает с миниатюры).
* **Отмена — это не «лиц нет».** Пустой ответ после «Остановить» — это «мы сами перестали
  смотреть», а не промах поиска. Фраза спецификации (раздел 7) «на этом фото лицо не
  найдено» остаётся за снимком, который разобрали и где лиц и правда нет; неразобранный
  файл называется неразобранным. То же обещание на своих файлах держит окно эталона.
* **Шаг «Кого ищем» — один модальный разговор.** Кнопка на шаге одна, и она открывает окно
  эталона: сколько снимков принести, какие лица отметить и что из этого пойдёт в поиск —
  всё там, рядом с числами. Отказ от окна не трогает показанное, а ответ переезжает в
  `anchor`/`extra`/`reference` без нового разбора архива. Прежнее окно принимало один файл
  за раз, а лица, похожие меньше чем на 40 %, молча не попадали в эталон; с задачи T14
  в поиск идёт любое отмеченное лицо, а число только предупреждает.
* **Отказ от нового разбора откатывает полосу.** Иначе она показывала бы 1200 px при
  разобранном архиве в 2400 px, и следующее «согласие» человека уехало бы по ложному
  основанию.
* **Каждый искомый человек оценивает кадр сам, и карточка называет, КТО найден.**
  Человек ищет ребёнка и, например, маму: один общий эталон на все отмеченные лица давал
  один общий максимум, и «61 %» не отвечало на вопрос, чьё это фото — сильное совпадение
  одного человека перекрывало слабое у другого. Имена живут в списке людей шага 2, а
  сетка, окно просмотра и отчёт берут их у строки результата (задача T20).
* **Окно ждёт свой поток.** `worker.wait(...)` возвращает `False`, и раньше этот ответ
  выбрасывался: оглавление закрывалось под живым потоком. Теперь закрытие отменяется и
  повторяется на `finished`, а окно живёт на модульной ссылке из `main.py`.
* **Незаписанное число дописывается при закрытии.** Дебаунс — обещание «применю через
  150 мс», а не «применю, если доживу».
* **Число объяснено на экране до всякого разбора.** Измеренная полоса настоящих
  совпадений — 39…67 %, и без этой строки 45 % читается как провал.
* **Папок может быть несколько** — спецификация, раздел 6: «выбор одной или нескольких
  папок». `QFileDialog.getExistingDirectory` не выделяет несколько ни на одной
  платформе, поэтому папки добавляет кнопка, а список лежит под строкой выбора:
  человек должен видеть, где именно он ищет, а не доверять счётчику. Повторы
  снимаются на входе — вложенная пара папок иначе удваивала бы и снимки, и цену разбора.
  Кнопок ровно две и обе делают одно: **«Добавить папку с фото»** дополняет список,
  **«Убрать все»** очищает его. Прежде рядом стояла «Выбрать папку с фото», которая
  *замещала* список целиком, и это была ловушка, а не выбор: человек приносил «Фото»,
  потом «Камеру», потом жмёт «Выбрать» — и половина архива исчезает без единого слова.
  Отличить дополняющее действие от замещающего по названию нельзя, поэтому замещающего
  в окне нет вовсе, а редкое «начать с чистого листа» стоит два нажатия.

Модели в этом модуле не загружаются: `make_engine` только собирает объект, а 13–21 с
начинаются внутри рабочего потока (см. `worker.obespechit_zagruzku`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import (QApplication, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QProgressBar, QPushButton,
                               QVBoxLayout, QWidget)

from core.cache import FaceCache
from core.etalon import Chelovek, Itog
from core.copier import copy_photos
from core.engine import Face, FaceEngine, make_engine
from core.matcher import percent_of, score_photo
from core.report import ResultRow, build_rows, write_csv, write_report
from core.scanner import find_photos
from core.worker import ScanStats, ScanWorker
from ui.photo_view import PhotoViewDialog
from ui.reference_dialog import ReferenceDialog
from ui.results_view import ResultsView, ramki_stroki
from ui.settings import SettingsBar
from utils.config import Settings, buffalo_l_dir, cache_file, yunet_model_path

# Единственная задержка между «человек водит ползунок» и «пересчёт + запись настроек».
# 150 мс — жест не замедляет, а тридцать пересчётов в нём схлопываются в один.
DEBAUN_MS = 150

# Сколько секунд ждём рабочий поток при закрытии окна. Больше — не имеет смысла: если за
# это время поток не встал, ждём его ОТДЕЛЬНОГО закрытия (`closeEvent` отменяется и
# повторяется на `finished`), а не держим человека перед мёртвым окном.
OZHDANIE_ZAKRYTIYA_MS = 3000

MODEL_HINT = ("готовим модели распознавания: при первом запуске скачиваем около 290 МБ, "
              "это минуту-две")

# Как подписывается первая строка эталона — то лицо, которое человек подтвердил сам.
@dataclass
class _Stroka:
    """Один человек в шаге 2: его виджеты, чтобы гасить их и подписывать по ответу окна."""

    chelovek: Chelovek
    videzhety: QWidget
    imja: QLineEdit
    knopka: QPushButton
    podpis: QLabel
    ubrat: QPushButton


class MainWindow(QWidget):
    """Три шага в одном окне: где ищем, кого ищем, что показало."""

    def __init__(self, settings: Settings | None = None, *,
                 cache_path: Path | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Поиск фото ребёнка")

        # `cache_path` — не украшение: по умолчанию оглавление лежит в служебной папке
        # пользователя, и окно, собранное в тесте или в двух запущенных копиях, лезло бы
        # в живой кэш человека.
        self.settings = settings if settings is not None else Settings()
        self.cache = FaceCache(Path(cache_path) if cache_path is not None else cache_file())

        self.engine: FaceEngine | None = None
        self.worker: ScanWorker | None = None
        self._posle_skana: Callable[[], None] | None = None

        self.folders: list[Path] = []
        self.files: list[Path] = []
        self.photos: dict[Path, list[Face]] = {}
        # Кого ищем: сколько строк в шаге 2, столько и независимых поисков. Один смешанный
        # эталон на всех давал один общий максимум на кадр — и «61 %» не отвечало, кто в
        # кадре: ребёнок или взрослый, отмеченный вторым примером (задача T20).
        self.ljudi: list[Chelovek] = []
        self._strok: list[_Stroka] = []

        # Накопитель потерь: приходит с каждого прогона и переживает несколько прогонов
        # одного поиска — но не всю жизнь окна: `_zapisi_poteri` чистит запись по каждому
        # снимку, который на этом прогоне разобрался, а новый разбор папки обнуляет всё
        # (см. `start_scan`).
        self._poteri: dict[str, str] = {}
        self._otbroennye: dict[Path, int] = {}
        self._povrezhdeniya: int = 0
        self._zanyat = False
        # Открыто ли окно выбора эталона. Пока оно открыто, главный окно не начинает СВОЙ
        # разбор: два потока к одной базе — это не гонка за блокировку, а два ответа про
        # один архив на одном экране. Модальность мышь сюда не пустит, а `start_scan` из
        # кода (таймер полосы настроек) — пустит, поэтому замок стоит и здесь.
        self._etalon_otkryt = False
        # Ответ окна эталона с прошлого подтверждения: по нему окно восстанавливает
        # карточки и отметки при повторном открытии. None — пока ничего не подтверждали.
        # Окно просмотра одно на все двойные клики: несколько открытых кадров человек
        # путал бы, а закрывать его на каждом следующем — значило бы терять то, что он
        # только что разглядывал.
        self._okno_prosmotra: PhotoViewDialog | None = None
        # Разновидность занятости: посреди копирования рабочего потока нет, и кнопки
        # «Остановить» на экране тоже нет — обещать её в тексте было бы враньём.
        self._kopiruetsya = False
        self._o_perezapisi_skazano = False
        # Отмена, запрошенная человеком, и просьба настройки, поданная посреди разбора.
        self._otmenen = False
        self._nastrojki_ozhidayut = False
        # Закрытие окна, которое не состоялось из-за живого рабочего потока.
        self._zakrytie_otlozheno = False
        self._oglavlenie_zakryto = False

        self.scan_mode = self.settings.engine
        self.scan_dim = self.settings.max_dim

        self._build_ui()

        self.settings_bar.set_engine(self.settings.engine)
        self.settings_bar.set_max_dim(self.settings.max_dim)
        # Полоса возвращает БЛИЖАЙШЕЕ разрешённое значение (1800 -> 2400), а строки выше
        # записали в scan_dim сырое 1800 из настроек. Без этого согласования первое же движение
        # ползунка увидело бы «разрешение поменялось» и потребовало перебрать весь архив —
        # при каждом запуске, потому что в файл так и легло бы 1800.
        self.scan_mode = self.settings_bar.engine
        self.scan_dim = self.settings_bar.max_dim
        self.settings.engine, self.settings.max_dim = self.scan_mode, self.scan_dim
        self._sohranit_nastrojki(modalno=False)      # окно ещё не показано
        self.settings_bar.set_threshold(self.settings.threshold)

        # Таймер — ДО connect: обработчик `changed` трогает `_taymer`, и висящий ниже
        # конструктор означал бы AttributeError на сигнале, пришедшем во время сборки окна.
        self._taymer = QTimer(self)
        self._taymer.setSingleShot(True)
        self._taymer.setInterval(DEBAUN_MS)
        self._taymer.timeout.connect(self._primenit_nastrojki)

        # connect — ПОСЛЕ восстановления значений: см. docstring `ui/settings.py`
        self.settings_bar.changed.connect(self._nastrojka_izmenilas)
        self.settings_bar.clear_cache.connect(self._clear_cache)
        self.results.selection_changed.connect(self._obnovit_knopku_kopirovaniya)
        # Двойной клик по карточке — окно с крупным снимком; отметка при этом живёт в
        # сетке, а окно лишь показывает и просит (см. `ui.photo_view`).
        self.results.prosyat_otkryt.connect(self._otkryt_prosmotr)
        self.results.selection_changed.connect(self._sinkhronizirat_prosmotr)

        # ~970 px: если окно уже, полоса настроек сжимается в «Насколько фото до…».
        self.setMinimumWidth(max(900, self.settings_bar.sizeHint().width() + 24))
        self.resize(max(1120, self.minimumWidth()), 820)

    # --------------------------------------------------------------- сборка UI
    def _build_ui(self) -> None:
        self.step1 = QGroupBox("Где ищем")
        # Одна кнопка добавления вместо пары «выбрать / добавить»: см. модульный
        # docstring, пункт про папки. Замещающая кнопка стирала принесённое молча.
        self.add_folder_button = QPushButton("Добавить папку с фото")
        self.add_folder_button.clicked.connect(self._dobavit_papku)
        self.clear_folders_button = QPushButton("Убрать все")
        self.clear_folders_button.setEnabled(False)
        self.clear_folders_button.clicked.connect(self._ubrat_papki)
        self.source_status = QLabel("папки не выбраны")
        self.source_list = QLabel("")
        self.source_list.setWordWrap(True)
        self.source_list.setStyleSheet("color: gray")
        self.scan_button = QPushButton("Начать разбор папки")
        self.scan_button.setEnabled(False)
        self.scan_button.clicked.connect(self.start_scan)
        self.stop_button = QPushButton("Остановить")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_requested)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.status_label = QLabel("")
        odna_stroka = QHBoxLayout()
        odna_stroka.addWidget(self.add_folder_button)
        odna_stroka.addWidget(self.clear_folders_button)
        odna_stroka.addWidget(self.source_status, 1)
        odna_stroka.addWidget(self.scan_button)
        odna_stroka.addWidget(self.stop_button)
        lay1 = QVBoxLayout(self.step1)
        lay1.addLayout(odna_stroka)
        lay1.addWidget(self.source_list)
        lay1.addWidget(self.progress)
        lay1.addWidget(self.status_label)

        self.step2 = QGroupBox("Кого ищем")
        # По строке на человека: имя, кнопка выбора его фото, подписанный счётчик и
        # «Убрать». Сколько снимков человек принесёт и какие лица отметит — разговор окна
        # эталона с ним, а не трёх кнопок главного окна.
        self._pole_ljudj = QVBoxLayout()
        self.add_person_button = QPushButton("Добавить ещё одного")
        self.add_person_button.clicked.connect(lambda _=False: self._dobavit_cheloveka())
        self._pole_ljudj.addWidget(self.add_person_button)
        QVBoxLayout(self.step2).addLayout(self._pole_ljudj)
        self._dobavit_cheloveka("ребёнок")     # первый поиск есть всегда

        self.step3 = QGroupBox("Что показало")
        self.settings_bar = SettingsBar()
        self.results = ResultsView()
        self.copy_button = QPushButton("Скопировать отмеченные")
        self.copy_button.setEnabled(False)
        self.copy_button.clicked.connect(self.copy_marked)
        stroka3 = QHBoxLayout()
        stroka3.addWidget(self.copy_button)
        stroka3.addStretch(1)
        lay3 = QVBoxLayout(self.step3)
        lay3.addWidget(self.settings_bar)
        lay3.addWidget(self.results, 1)
        lay3.addLayout(stroka3)

        koren = QVBoxLayout(self)
        koren.addWidget(self.step1)
        koren.addWidget(self.step2)
        koren.addWidget(self.step3, 1)

    def _obnovit_knopku_kopirovaniya(self, *spiski: object) -> None:
        """Жива ли кнопка «Скопировать отмеченные» — одно место, три вызова.

        Посреди разбора копировать нельзя, и это не забота о прогресс-баре: `start_scan`
        снимает `self.photos`, а карточки в сетке остаются и галочки у них остаются.
        Прежнее окно позволяло нажать кнопку в этот момент, и `copy_marked` строил
        отчёт по пустому `self.photos`: рядом с «Скопировано: 1» в report.txt
        писались «фото в отчёте: 0» и «ни одного разобранного фото». Копирование при
        этом проходило успешно — то есть отчёт противоречил тому, что уже лежало в
        папке результатов, и человек читал это как «скопировалось не всё».

        Сигнал `selection_changed` приходит со списком, `rebuild_scores` зовёт нас без
        аргументов: число отмеченных спрашиваем у сетки, а у аргументов только проверяем,
        что сигнал был.
        """
        self.copy_button.setEnabled(not self._zanyat and bool(self.results.checked))

    # --------------------------------------------------------------- шаг 1
    def set_source_folders(self, folders: list[Path]) -> None:
        """Папки, где ищем. Счётчик фото — сразу, без разбора: человек должен увидеть,
        что приложение нашло что искать, ещё до минут ожидания.

        Список папок — это список, а не одна папка (спецификация, раздел 6): семейный
        архив обычно лежит в двух-трёх местах — «Фото», «Камера», «Отпуск». Порядок
        сохраняется, порядок же и дедуплицируется: та же папка дважды из двух кнопок
        вернула бы каждый снимок дважды, а `find_photos` об этом молчит.
        """
        # dict.fromkeys — дедупликация с сохранением порядка: вложенная друг в друга
        # пара папок вернула бы каждый снимок дважды (задача 2).
        self.folders = list(dict.fromkeys(Path(f) for f in folders))
        self.files = list(dict.fromkeys(find_photos(self.folders)))
        if self.files:
            # «фото» не склоняется, поэтому «найдено 21 фото» верно при любом числе;
            # «папок» — существительным вперёд, как в отчёте: так не ломается согласование.
            self.source_status.setText(
                f"найдено {len(self.files)} фото, папок: {len(self.folders)}")
        elif self.folders:
            self.source_status.setText(
                "в выбранных папках фото не найдено — выберите другую или зайдите "
                "внутрь: фото могут лежать в подпапке")
        else:
            self.source_status.setText("папки не выбраны")
        self.source_list.setText(self._podbork_papok())
        self.source_list.setToolTip("\n".join(str(f) for f in self.folders))
        self.scan_button.setEnabled(bool(self.files) and not self._zanyat)
        # «Убрать все» живёт при непустом списке. Окно занято — кнопка гаснет по
        # другому признаку (см. `_set_busy`), и здесь её нельзя ни включать, ни выключать:
        # иначе пустой список посреди разбора вернул бы кнопку живому окну.
        if not self._zanyat:
            self.clear_folders_button.setEnabled(bool(self.folders))
        # Оценка времени разбора живёт у полосы настроек: там способ поиска, а цена
        # именно у него. Число известно уже сейчас — модели для этого не нужны.
        self.settings_bar.set_archive_size(len(self.files))
        if self.folders:
            # Пустой список НЕ затирает remembered путь: «убрать все» — это про список
            # папок, а не про «забыть, где лежат фото». Иначе следующий «Добавить
            # папку с фото» открылся бы дома, а не там, где человек только что был.
            self.settings.last_source = str(self.folders[-1])
            self._sohranit_nastrojki()

    def _podbork_papok(self) -> str:
        """Что человек читает под списком папок: все выбранные, без повторов.

        Полный путь показан целиком: половинку имени вида «…/Фото» человек отличит от
        «…/Фото grandma» только глазами, а перепутать папки — это разобрать не то и
        ждать вдвое дольше.
        """
        if not self.folders:
            return ""
        return "ищем в папках: " + " | ".join(str(f) for f in self.folders)

    def _dobavit_papku(self) -> None:
        """Добавить ещё одну папку к уже выбранным — спецификация, раздел 6.

        Единственная точка, откуда в окно попадает папка. Она только дополняет:
        замещающей кнопки в интерфейсе нет намеренно (см. модульный docstring).

        Начало диалога — последняя добавленная папка: архив обычно не разбросан по
        диску, а лежит соседними папками, и второй заход человек делает рядом.
        """
        nachalo = str(self.folders[-1]) if self.folders else (
            self.settings.last_source or str(Path.home()))
        papka = QFileDialog.getExistingDirectory(self, "Добавить папку с фото", nachalo)
        if papka:
            self.set_source_folders(self.folders + [Path(papka)])

    def _ubrat_papki(self) -> None:
        """Очистить список папок. Снятые лица и найденные фото не трогаются.

        Кнопка называется «убрать», а не «забыть», и обязана держать слово: разбор
        стоит минуты, и стереть его результат из-за непонятного глагола значило бы
        заставить человека платить моделями повторно. Находки снимает только новый
        «Начать разбор папки» — это единственное место, где окно объявляет прежний
        ответ недействительным.
        """
        if self._zanyat:
            # Посреди разбора список уже разминулся с потоком, и менять его нельзя.
            # Кнопка в эту минуту выключена, но слот живёт и вызывается из тестов.
            return
        self.set_source_folders([])

    def start_scan(self) -> None:
        """Разобрать текущий список файлов. Прежние находки снимаем: они от другой папки
        или от другого качества разбора, и держать их в сетке — значит показывать то, чего
        человек уже не просил.

        Проверка «разбор уже идёт» — ДО очистки. Иначе второй клик по кнопке (или смена
        режима посреди прогона) стёр бы показанные результаты, а нового разбора человек
        бы не получил: `_run` отказывает молча.
        """
        if not self._mozhno_nachat():
            return
        # Открытое окно просмотра относится к прежнему набору находок: его проценты
        # больше нечем подтвердить, а галочка «копировать это фото» ссылается на список,
        # который сейчас обнулится.
        if self._okno_prosmotra is not None:
            self._okno_prosmotra.close()
        self.photos = {}
        # Снять галочки с сетки, которую человек ещё видит. Карточки на время разбора
        # остаются на месте — иначе ошибка потока оставила бы пустой экран там, где
        # раньше что-то лежало, — но копировать по ним становится нечего: под ними уже
        # нет лиц, а значит нечего и класть в отчёт. Это второй замок против
        # противоречия «Скопировано: 1» рядом с «фото в отчёте: 0» (см. `_set_busy`).
        self.results.uncheck_all()
        self._poteri = {}
        self._otbroennye = {}
        self._povrezhdeniya = 0
        self._run(self.files, then=None, note="разбираем папку")

    def _mozhno_nachat(self) -> bool:
        """Свободны ли мы для нового разбора — и сказать это человеку, если нет.

        Копирование занимает тот же признак `_zanyat`, но просить при нём «нажать
        «Остановить»» нельзя: такой кнопки в эту минуту на экране нет, и человек искал
        бы её по всему окну.
        """
        if self._etalon_otkryt:
            # Модальное окно не отдаёт этому окну клики мышью, но `start_scan` может
            # приехать и из кода — из таймера полосы настроек, например.
            self._warn("окно выбора эталона ещё открыто — сначала закройте его")
            return False
        if self._zanyat:
            if self._kopiruetsya:
                self._warn("копирование уже идёт — дождитесь, пока снимки дойдут "
                           "до папки результатов")
            else:
                self._warn("разбор уже идёт, дождитесь или нажмите «Остановить»")
            return False
        return True

    def _run(self, files: list[Path], then: Callable[[], None] | None,
             note: str) -> None:
        """Единственная точка запуска фоновой работы окна: разбор выбранной папки.

        Прогон папки — это исчерпывающий ответ про весь архив, поэтому его потери и
        ложатся в отчёт целиком. Единственный другой разбор в приложении — файлы окна
        эталона — живёт СВОИМ потоком в `ui/reference_dialog.py` и в накопитель окна не
        пишет вовсе: раньше ради него здесь жил признак `progon_papki`, и его нули могли
        перечеркнуть потери архива.
        """
        if not self._mozhno_nachat():
            return                        # единственная точка выхода занятого окна
        self._posle_skana = then
        self._otmenen = False
        self.progress.setMaximum(max(1, len(files)))
        self.progress.setValue(0)
        self.worker = ScanWorker(files, self._ensure_engine(), self.cache,
                                 self.scan_dim, parent=self)
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self._on_scan_done)
        self.worker.error.connect(self._on_scan_error)
        self._set_busy(True, note)
        if not getattr(self.engine, "loaded", False):
            # Слово про модели уместно ровно один раз и ровно там, где веса ещё не читаны:
            # иначе человек ждёт скачивания 290 МБ на втором, третьем и десятом прогоне.
            self.status_label.setText(f"{note}; {MODEL_HINT}")
        self.worker.start()

    def _ensure_engine(self) -> FaceEngine:
        if self.engine is None:
            self.engine = make_engine(self.scan_mode, yunet_model_path(),
                                      buffalo_l_dir())
        return self.engine

    def _on_progress(self, sdelano: int, vsego: int, imya: str) -> None:
        self.progress.setMaximum(max(1, vsego))
        self.progress.setValue(sdelano)
        self.status_label.setText(f"обработано {sdelano} из {vsego}: {imya}")

    def _on_scan_error(self, soobshchenie: str) -> None:
        """Показать текст КАК ЕСТЬ и не трогать показанное.

        Рабочий поток уже различает этапы и пишет по-русски, что именно не удалось:
        «не удалось загрузить модели…» (ни один снимок не тронут) против «оглавление
        недоступно…, уже снятые лица сохранены» (кое-что уже снято и лежит в
        базе). Свой
        заголовок здесь наврал бы человеку, по какому пути идти, а очистка сетки
        уничтожила бы частичный результат, который в этом случае и есть вся награда.
        """
        # Продолжение (выбор эталона) аннулируется здесь же: снимок, на который оно
        # рассчитывало, не разобран, а отложенное продолжение выстрелило бы на следующем
        # прогоне и объяснило бы человеку результат другого разбора.
        self._posle_skana = None
        self._otmenen = False
        self._set_busy(False, "")
        self._warn(soobshchenie)

    def _on_scan_done(self, payload: object) -> None:
        photos, statistika = payload                      # type: ignore[misc]
        self.photos.update(photos)
        self._zapisi_poteri(photos, statistika)
        self._set_busy(False, "")
        otmenen = self._otmenen
        self._otmenen = False
        self.status_label.setText(
            ("разбор остановлен — " if otmenen else "") +
            f"разобрано: {statistika.scanned}, взято из оглавления: {statistika.cached}, "
            f"не обработано: {len(self._poteri)}, время: {statistika.seconds:.0f} с")
        if otmenen and self._posle_skana is not None:
            # Отмена — не провал и не «лиц на фото нет». Пустой ответ после нажатия
            # «Остановить» прежнее окно читало как промах поиска и совал человеку
            # модальное окно «выберите другое фото», спецификация (раздел 7) держит эту
            # фразу за снимком, на котором лица и правда нет. Продолжение не запускаем,
            # сделанное показываем как есть.
            self._posle_skana = None
            self.status_label.setText("разбор остановлен — эталон не выбран, "
                                      "на экране ничего нет")
        if self._posle_skana is not None:
            then, self._posle_skana = self._posle_skana, None
            then()
        else:
            self.rebuild_scores()

    def _zapisi_poteri(self, photos: dict, statistika: ScanStats) -> None:
        """Свести числа прогона в накопитель окна.

        Потери — по пути файла, а не списком: второй прогон того же снимка обязан
        затерять старую беду, а не копить её. Снимок, который на этот раз разобрался,
        убирается из списка потерь прямо здесь, иначе отчёт рассказал бы о файле,
        который приложение уже прочитало. Так чистятся ОБА счётчика по путям: и
        «не обработано», и «лиц отброшено» — иначе одна неудачная отбраковка кочевала бы
        из report.txt в report.txt вечно (`dropped_faces` приходит только по тем файлам,
        где что-то отбраковали, а снять его обязан любой перечитанный снимок).

        Счётчик повреждённых строк оглавления путями не привязан, и для него «свежий
        прогон» — это прогон папки: он ПЕРЕКРЫВАЕТ число, а не копит его. Через `_run`
        окно сегодня ходит только ради папки, поэтому суммировать больше не с чем;
        вернётся второй вид прогона — вернётся и признак, с тестом на него.

        Чистка по каждому присланному снимку безопасна ровно потому, что снимок из
        оглавления приезжает СО СВОИМ числом потерь: оно записано в саму строку
        оглавления (`cache.put(..., dropped=)`) и приходит в `dropped_faces` на тёплом
        прогоне. Без этого поля второй запуск папки печатал бы «потерь нет» об архив,
        где лица теряли на первом, холодном прогоне, — модели-то к нему больше не
        заглядывали и сказать про него нечего.
        """
        for put in photos:
            self._poteri.pop(str(put), None)
            self._otbroennye.pop(put, None)
        self._poteri.update(statistika.failures)
        self._otbroennye.update(statistika.dropped_faces)
        self._povrezhdeniya = statistika.cache_damage

    # --------------------------------------------------------------- шаг 2
    def _dobavit_cheloveka(self, nazvanie: str | None = None) -> Chelovek:
        """Одна строка в шаге «Кого ищем» = один независимый поиск.

        Имя по умолчанию — «ребёнок» для первого поиска: приложение существует ради
        поиска ребёнка, и большинство людей так и начнут. Для второго имя нейтральное:
        угадать, кто это (мама, брат, подруга), приложение не может, а выдумывать «взрослый
        2» значило бы подсказывать человеку вывод о том, кого он отметил.
        """
        chelovek = Chelovek(nazvanie or f"человек {len(self.ljudi) + 1}")
        self.ljudi.append(chelovek)
        self._perestroit_stroki_ljudej()
        return chelovek

    def _perestroit_stroki_ljudej(self) -> None:
        """Собрать шаг «Кого ищем» из списка людей. Один источник — `self.ljudi`.

        Пересборка вместо вставки по индексу держится на одном: виджеты и данные не
        могут разойтись, если виджеты всегда строятся по данным. Этим же путём идут и
        тесты, которым нужен готовый эталон без модального окна.
        """
        for stroka in self._strok:
            self._pole_ljudj.removeWidget(stroka.videzhety)
            stroka.videzhety.setParent(None)
            stroka.videzhety.deleteLater()
        self._strok = [self._sozdat_stroku(c) for c in self.ljudi]
        for nomer, stroka in enumerate(self._strok):
            self._podpisat_stroku(stroka)
            # Строка N встаёт на место N: вставка всем в один и тот же индекс
            # переворачивала порядок, и переименованная строка уезжала не туда, где
            # лежал выбор фото этого человека.
            self._pole_ljudj.insertWidget(nomer, stroka.videzhety)
        self._obnovit_knopki_ubral_i_zapusk()

    def _sozdat_stroku(self, chelovek: Chelovek) -> _Stroka:
        videzhety = QWidget()
        maket = QHBoxLayout(videzhety)
        maket.setContentsMargins(0, 0, 0, 0)

        imja = QLineEdit()
        imja.setPlaceholderText("как назвать")
        imja.setFixedWidth(150)

        knopka = QPushButton()
        knopka.clicked.connect(lambda _=False, c=chelovek: self._vybrat_etalon(c))

        podpis = QLabel("фото не выбрано")
        ubrat = QPushButton("Убрать")
        ubrat.clicked.connect(lambda _=False, c=chelovek: self._ubrat_cheloveka(c))

        maket.addWidget(imja)
        maket.addWidget(knopka)
        maket.addWidget(podpis, 1)
        maket.addWidget(ubrat)

        stroka = _Stroka(chelovek, videzhety, imja, knopka, podpis, ubrat)
        imja.textChanged.connect(lambda tekst, s=stroka: self._na_imya(s, tekst))
        # Пересчёт — по завершении правки, а не по каждому нажатию клавиши: имя уезжает в
        # подписи карточек, и пересобирать три сотни виджетов на каждый символ значило бы
        # подвешивать окно ради текста, который человек ещё не дописал.
        imja.editingFinished.connect(self.rebuild_scores)
        # Текст ставится ПОСЛЕ подключения: имя приходит из того же места, что и правка
        # человеком, и подпись кнопки не может отстать от поля.
        imja.setText(chelovek.nazvanie)
        return stroka

    def _podpisat_stroku(self, stroka: _Stroka) -> None:
        """Что написано в строке человека: либо счётчик эталона, либо «фото не выбрано»."""
        itog = stroka.chelovek.itog
        if itog is None or itog.reference is None or itog.anchor is None:
            stroka.podpis.setText("фото не выбрано")
            return
        snimov = len({karta.put for karta in (itog.anchor, *itog.extra)})
        slabyh = sum(1 for karta in itog.karty if karta.slabo)
        tekst = f"лиц {itog.reference.count}, снимков {snimov}"
        if slabyh:
            tekst += f", не похожи на остальных {slabyh}"
        if itog.otvergnutye:
            tekst += f", не ищем {len(itog.otvergnutye)} — сравнить нечем"
        stroka.podpis.setText(tekst)

    def _ubrat_cheloveka(self, chelovek: Chelovek) -> None:
        """Убрать человека и его строку. Разобранные лица архива при этом остаются.

        Проценты пересчитываются без него: то, что он давал, перестаёт быть основанием
        показать фото, но минут разбора архива это не отменяет.
        """
        if self._zanyat:
            self._warn("посреди разбора список людей не меняется — дождитесь конца "
                       "или нажмите «Остановить»")
            return
        for nomer, stroka in enumerate(self._strok):
            if stroka.chelovek is chelovek:
                self.ljudi.pop(nomer)
                self._strok.pop(nomer)
                stroka.videzhety.setParent(None)
                stroka.videzhety.deleteLater()
                break
        self._obnovit_knopki_ubral_i_zapusk()
        self.rebuild_scores()

    def _obnovit_knopki_ubral_i_zapusk(self) -> None:
        """«Убрать» живёт только пока людей больше одного, иначе убирать нечего."""
        for stroka in self._strok:
            stroka.ubrat.setEnabled(len(self._strok) > 1)

    def _na_imya(self, stroka: _Stroka, tekst: str) -> None:
        """Имя человека: оно же — на кнопке, в подписях карточек и в отчёте.

        Кнопка обязана говорить, ЧЬИ фото она просит. Две одинаковые «Выбрать фото…»
        рядом не отличимы, а после переименования строки старая подпись врёт: человек
        открывает окно «мама» и видит в нём снимки ребёнка.
        """
        stroka.chelovek.nazvanie = tekst.strip() or stroka.chelovek.nazvanie
        stroka.knopka.setText(f"Выбрать фото: {stroka.chelovek.nazvanie}")

    def _chuzhie_otmetki(self, chelovek: Chelovek) -> frozenset[tuple[Path, int]]:
        """Какие лица человек уже отдал ДРУГИМ поискам.

        Нужны одному месту — автоотметке «на снимке ровно одно лицо». Приносит тот же
        набор файлов для мамы: детское лицо с единственного кадра было помечено само,
        и окно «мама» открывалось с ребёнком в первом ряду отмеченных (задача T22).
        """
        zanyato: set[tuple[Path, int]] = set()
        for drugoj in self.ljudi:
            if drugoj is chelovek or drugoj.itog is None:
                continue
            zanyato.update((karta.put, karta.nomer - 1) for karta in drugoj.itog.karty
                           if karta.otmecheno)
        return frozenset(zanyato)

    def _aktyvnye_ljudi(self) -> list[Chelovek]:
        """Те, кого реально ищем: у человека выбран эталон. Порядок = порядок строк.

        Номер человека в строке результата (`ResultRow.chelovek`) считается именно по
        этому списку, и имя для подписи берётся отсюда же: два разных порядка дали бы
        карточке «похоже на человека 2» про того, кого человек зовёт мамой.
        """
        return [c for c in self.ljudi if c.reference is not None]

    def _vybrat_etalon(self, chelovek: Chelovek) -> None:
        """Шаг «Кого ищем» целиком: окно эталона, перенос выбора, мгновенный пересчёт.

        Своего разбора здесь нет: окно эталона разбирает принесённые файлы СВОИМ потоком,
        а главное окно получает готовый `Itog` и переклеивает эталон в памяти. Лица архива
        уже сняты и лежат в оглавлении, поэтому проценты пересчитываются за доли секунды,
        и нового «разбора папки» человек не ждёт.

        Отказ от окна — не «эталон сброшен». Прежние `anchor`, `extra` и показанные проценты
        остаются в силе: минуты разбора архива никуда не делись, и стереть их кнопкой
        «Отмена» значило бы заставить человека платить моделями повторно.
        """
        if not self._mozhno_nachat():
            return
        self._etalon_otkryt = True
        try:
            itog = ReferenceDialog.sprosit(self, self._ensure_engine(), self.cache,
                                           self.scan_dim, self.settings.last_source,
                                           nahodno=chelovek.itog,
                                           vklad=self._vklad_strok_etalona(chelovek),
                                           nazvanie=chelovek.nazvanie,
                                           chuzhie=self._chuzhie_otmetki(chelovek))
        finally:
            # Флаг снимается и на исключении: окно, запершее человека «в незакрытом диалоге»,
            # не смогло бы больше начать ни одного разбора.
            self._etalon_otkryt = False
        if itog is None:
            return
        if itog.reference is None or itog.anchor is None:
            # Диалог не держит кнопку «Искать» живой без эталона, так что сюда приезжают
            # только чужие данные. Молча стереть показанное было бы хуже честного «всё по-старому».
            # Причина здесь ровно одна — непригодный отпечаток (с T14 низкое число сходства
            # в поиск не мешает), и назвать её «лица не похожи» значит отправить человека
            # искать другое фото там, где дело в качестве снимка.
            self._warn("эталон не собрался: сравнить нечем — отмеченные лица сняты "
                       "слишком мелко или не в фокусе. Отметьте фото, где ребёнок виден "
                       "крупнее и не в профиль; прежний эталон остался в силе")
            return
        # Ответ хранится целиком, а не как `anchor` + `extra`: в `Itog` есть файл каждого
        # лица, неотмеченные карточки и снимки без лиц. Только так окно открывается
        # повторно таким, каким человек его оставил.
        chelovek.itog = itog
        self._obnovit_podpis_etalona(chelovek)

    def _obnovit_podpis_etalona(self, chelovek: Chelovek) -> None:
        """Что человек читает в своей строке — честный счётчик, а не утешение.

        `лиц` — сколько отпечатков этого человека участвует в сравнении, `снимков` — с
        скольких файлов они взяты: два лица с одного группового снимка это один снимок и
        два лица. Назвать снимками лица значило бы потом не объяснить человеку число в
        отчёте.

        Дальше два числа про отмеченные лица, и означают они разное (T14): «не похожи на
        остальных» — участвуют в поиске, но их стоит перепроверить глазами, а «не ищем» —
        только про непригодный отпечаток: таким вектором нечем считать. Прежняя строка заканчивалась
        словами «не пошло N» про первых: человек считал, что его галочку отбросили, и делал
        вывод, что приложение ищет по одному лицу.
        """
        stroka = next(s for s in self._strok if s.chelovek is chelovek)
        self._podpisat_stroku(stroka)
        self.rebuild_scores()

    # --------------------------------------------------------------- шаг 3
    def rebuild_scores(self) -> None:
        """Только пересчёт процентов: лица уже в памяти, нового разбора нет.

        Тяжёлой работе здесь взяться нечему — ни `ScanWorker`, ни движок не создаются,
        ни оглавление не открывается. Это и проверяют тесты через подменённый `ScanWorker`.

        Каждый активный человек оценивает кадр ОТДЕЛЬНО, и сетка получает столько
        словарей, сколько людей: иначе «61 %» остаётся вопросом «это кто?».
        """
        lydi = self._aktyvnye_ljudi()
        if not lydi or not self.photos:
            return
        sroki = self._stroki_dlya_ljudej(lydi)
        self.results.show_rows(sroki, self.settings_bar.threshold, self.scan_dim,
                               imena_ljudj=[c.nazvanie for c in lydi])
        self._obnovit_knopku_kopirovaniya()

    def _stroki_dlya_ljudej(self, lydi: list[Chelovek]) -> list[ResultRow]:
        """Оценить разобранные кадры по каждому человеку и собрать строки результата."""
        ocenki = [{put: score_photo(lica, c.reference) for put, lica in self.photos.items()}
                  for c in lydi]
        return build_rows(self.photos, ocenki, self.settings_bar.threshold)

    def _nastrojka_izmenilas(self, _podobnost: float, _rezhim: str, _razmer: int) -> None:
        """Полоса шлёт это на каждое деление. Сами значения берём не из аргументов, а из
        полосы в момент срабатывания таймера: они к тому времени уже последние."""
        self._taymer.start()

    def _primenit_nastrojki(self) -> None:
        """Одно применение: пересчёт и одна запись настроек после остановки ползунка."""
        if self._zanyat:
            # Разбор идёт — не применяем ВООБЩЕ ничего: ни `scan_mode`, ни `scan_dim`, ни
            # файл настроек. Прежнее окно записывало новое качество посреди прогона,
            # новый разбор при этом не начинался (`start_scan` отказывает занятому), а
            # `rebuild_scores` отдавал сетке `self.scan_dim` поверх кадров, снятых в
            # старом, — и зелёная рамка уезжала с лица. Полосу при этом не откатываем:
            # просьба человека живёт в виджете и будет применена, как только поток
            # встанет (см. `_set_busy`), — молча выкинуть её было бы хуже.
            self._nastrojki_ozhidayut = True
            return
        podobnost = self.settings_bar.threshold
        rezhim = self.settings_bar.engine
        razmer = self.settings_bar.max_dim
        self.settings.threshold = podobnost
        if rezhim == self.scan_mode and razmer == self.scan_dim:
            self._sohranit_nastrojki()
            self.rebuild_scores()
            return
        bylo_razobrano = bool(self.photos)
        if bylo_razobrano and not self._ask_rescan(rezhim, razmer):
            # Отказ = настройки не менялись: на диске остаётся то, чем реально разбирал.
            self.settings_bar.set_engine(self.scan_mode)
            self.settings_bar.set_max_dim(self.scan_dim)
            self._sohranit_nastrojki()
            return
        self.scan_mode, self.scan_dim = rezhim, razmer
        # Способ поиска и качество разбора едут в файл вместе с решением: без этих двух
        # строк человек выбрал бы «Быстрее», окно бы его запомнило, а следующий запуск
        # молча вернулся бы к «Оба».
        self.settings.engine, self.settings.max_dim = rezhim, razmer
        self._sohranit_nastrojki()
        self.engine = None                       # новый режим — новый движок
        # До первого разбора запускать скан по смене настройки нельзя: человек ещё не
        # нажимал «Начать разбор», а минуты и 290 МБ уже пошли.
        if bylo_razobrano and self.files:
            self.start_scan()

    def _ask_rescan(self, rezhim: str, razmer: int) -> bool:
        chto = []
        if rezhim != self.scan_mode:
            chto.append("способ поиска лиц")
        if razmer != self.scan_dim:
            chto.append("качество разбора (оно меняет отпечатки)")
        otvet = QMessageBox.question(
            self, "Нужен новый разбор",
            f"Вы изменили {', '.join(chto)}. Папку придётся разобрать заново — "
            f"это минуты, но не секунды. Начать сейчас?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return otvet == QMessageBox.Yes

    def _clear_cache(self) -> None:
        """Удалить оглавление и всё показанное по нему. Сами фото не трогаются.

        Сам вызов к базе обёрнут, потому что из слота Qt необработанное исключение не
        долетает: человек нажал «Очистить оглавление», а кнопка молча ничего не сделала
        бы — и он решил бы, что оглавление чисто, хотя это не так.

        Посреди разбора чистка запрещена вторым слоем, хотя кнопка в это время и так
        выключена: поток пишет в эту же базу каждый снимок, а `done` вернул бы на экран
        весь архив поверх обещания «начать с нуля».
        """
        if self._zanyat:
            self._warn("оглавление не очистить: разбор уже идёт — дождитесь или "
                       "нажмите «Остановить»")
            return
        try:
            n = self.cache.clear()
        except Exception as exc:               # noqa: BLE001 — база отвечает чем угодно
            self._warn(f"оглавление не очистилось: {exc}")
            return
        self.photos = {}
        self._poteri = {}
        self._otbroennye = {}
        self._povrezhdeniya = 0
        # Показать старые карточки после обещания «начать с нуля» — значит соврать
        # дважды: галочки остались, а данных под ними больше нет.
        self.results.show_rows([], self.settings_bar.threshold, self.scan_dim)
        self.status_label.setText(f"оглавление очищено: удалено записей {n}")

    # --------------------------------------------------------------- просмотр фото
    def _otkryt_prosmotr(self, put: Path) -> None:
        """Двойной клик по карточке: показать этот снимок крупно.

        Посреди разбора и копирования окно не открывает: карточки на экране остались от
        прошлого набора, а `photos` уже пуст, и рамка либо взялась неоткуда, либо
        относится к другому эталону. Молча открыть враньё хуже, чем не отреагировать:
        человек решит, что фото не найдено, хотя дело в состоянии окна.

        Файла нет в разобранных — тоже тихо: карточка под потолком сетки могла остаться
        от прогона, который уже сняли с экрана.
        """
        if self._zanyat or put not in self.photos:
            return
        if self._okno_prosmotra is None:
            self._okno_prosmotra = PhotoViewDialog(self)
            self._okno_prosmotra.razresheno.connect(self._razresheno_prosmotra)
            self._okno_prosmotra.prosyat_perehod.connect(self._perehod_prosmotra)
        self._pokazat_v_prosmotre(put)

    def _vklad_strok_etalona(self, chelovek: Chelovek) -> dict[int, int]:
        """Сколько снимков на экране нашлось по каждой строке эталона.

        Нужно окну эталона, чтобы сказать человеку «по нему найдено 1 снимок» против
        «по нему найдено 14». Сняв молчаливый отказ фильтра (T14), приложение оставило
        человеку и цену решения: лицо другого человека в эталоне начинает находить его
        по всему архиву, и выглядит это как находка ребёнка. Запрещать мы это не стали,
        поэтому показываем число.

        Считается по кучке «похоже» (`matched`), а не по всему архиву: это ровно те
        карточки, которые человек видит под счётчиком «Похоже: 32», и другое число
        разошлось бы с экраном.

        Пустой ответ до первого разбора — не ошибка, а единственно честное состояние:
        чисел ещё нет, и подписывать «найдено 0» было бы отговоркой от верного лица.
        """
        lydi = self._aktyvnye_ljudi()
        if chelovek not in lydi or not self.photos:
            return {}
        nomer = lydi.index(chelovek) + 1
        vklad: dict[int, int] = {}
        for row in self.results.itogi_prosmotra():
            if row.matched and row.ref_lico and row.chelovek == nomer:
                vklad[row.ref_lico] = vklad.get(row.ref_lico, 0) + 1
        return vklad

    def _pokazat_v_prosmotre(self, put: Path) -> None:
        """Передать окну всё, что он показывает, но сам не знает.

        Порядок списка, позицию в нём, процент и рамку знает только сетка: у неё строки
        результата, у неё же и отметки. Окно получает ответ и не заводит свою копию —
        иначе два ответа на вопрос «какое фото следующее» разъедутся при первом же
        пересчёте.
        """
        sroki = self.results.itogi_prosmotra()
        nomer = next((i + 1 for i, r in enumerate(sroki) if r.path == put), 0)
        if nomer == 0:
            return                     # карточка осталась от снятого с экрана набора
        row = sroki[nomer - 1]
        # Рамки берёт тот же владелец, что и карточка, — `ramki_stroki` от строки
        # результата. На групповом фото из 24 лиц ребёнок стоит дважды, и обвести нужно
        # оба совпадения, а не одно самое похожее (задача T19). Остальные 22 лица не
        # обводятся ничем: человек их не искал, а зелёная рамка значит «найдено».
        # «лицо N из M» при этом говорит про весь кадр, а не про число рамок, — M приходит
        # числом из строки.
        ramki = ramki_stroki(row)
        # Кого именно нашли — имя из строки поиска: «найдено по лицу 9» человеку ничего
        # не говорит, а «найден: мама» отвечает на заданный вопрос. Спрашиваем сетку, а не
        # читаем `self.ljudi` напрямую: порядок активных людей знает она, потому что по
        # нему же подписаны карточки.
        self._okno_prosmotra.pokazat(put, ramki, row.lice or None, self.scan_dim,
                                     row.percent, nomer, len(sroki),
                                     kopiruetsya=self.results.otmecheno(put),
                                     vsego_lic=row.faces,
                                     imya_cheloveka=self.results.imya_cheloveka(row))

    def _perehod_prosmotra(self, shag: int) -> None:
        """Стрелка или клавиша: шаг по списку находок в этом же окне.

        Край списка проверяется здесь, а не в окне: у окна нет права решать, есть ли
        куда идти, — порядок принадлежит сетке. За краем не происходит ничего, и это
        верно: живая клавиша, которая молчит на краю, лучше молчаливого прыжка в начало.
        """
        sroki = self.results.itogi_prosmotra()
        tekushhij = self._okno_prosmotra.pokazyvaet
        nomer = next((i for i, r in enumerate(sroki) if r.path == tekushhij), None)
        if nomer is None:
            return
        sosednij = nomer + (1 if shag > 0 else -1)
        if not 0 <= sosednij < len(sroki):
            return
        self._pokazat_v_prosmotre(sroki[sosednij].path)

    def _razresheno_prosmotra(self, put: Path, sostojanie: bool) -> None:
        """Человек решил в окне просмотра — решение уходит в сетку, единственный
        владелец отметки. Обратно в окно его доставит `_sinkhronizirat_prosmotr`."""
        self.results.otmetit_snaruji(put, sostojanie)

    def _sinkhronizirat_prosmotr(self, _spiski: object = None) -> None:
        """Сетка изменила отметки — открытое окно обязано показать то же самое.

        Вызывается на тот же `selection_changed`, что и включение кнопки копирования:
        оба отвечают на один вопрос «что сейчас отмечено», и разносить их по разным
        сигналам значило бы однажды рассинхронизировать экран с списком на перенос.
        """
        if self._okno_prosmotra is None:
            return
        pokazyvaemoe = self._okno_prosmotra.pokazyvaet
        if pokazyvaemoe is not None:
            self._okno_prosmotra.sinkhronizirat(pokazyvaemoe,
                                          self.results.otmecheno(pokazyvaemoe))

    # --------------------------------------------------------------- вывод
    def copy_marked(self) -> None:
        """Скопировать отмеченные и написать рядом report.txt и results.csv.

        Копирование идёт в ЭТОМ потоке и занимает его целиком (см. модульный docstring,
        пункт про занятое окно), поэтому окно заранее гасит все виджеты и говорит, что
        происходит: несколько сотен телефонных снимков — это минуты, и всё это время
        окно не имеет права притворяться живым (спецификация, раздел 7 — «без вида
        зависшего окна»).
        """
        fajly = self.results.checked
        if not fajly:
            self._warn("отметьте хотя бы одно фото")
            return
        lydi = self._aktyvnye_ljudi()
        if not lydi:
            self._warn("сначала выберите фото хотя бы одного человека — шаге «Кого ищем»")
            return
        nachalo = self.settings.last_dest or str(Path.home())
        kuda = QFileDialog.getExistingDirectory(self, "Куда скопировать", nachalo)
        if not kuda:
            return
        cel = Path(kuda)
        # `poteri` обязателен: без него «скопировано 0 фото» осталось бы без причины,
        # а это ровно то молчание, против которого весь модуль.
        poteri: list[tuple[Path, str]] = []
        sbroj: OSError | None = None
        self._set_busy(True, f"копируем {len(fajly)} фото в {cel}…", kopirovanie=True)
        self._otrisovat_okno()
        try:
            pari = copy_photos(fajly, cel, poteri)
        except OSError as exc:
            # Папка не создалась вовсе: read-only том, отвалившийся сетевой диск,
            # NotADirectoryError. Это `OSError`, а не только `PermissionError`.
            sbroj, pari = exc, []
        finally:
            # Впереди `_warn` — модальное окно. Показывать его занятому окну значило бы
            # просить человека решить что-то, пока кнопки всё ещё выключены, поэтому
            # занятость снимается ДО любого разговора.
            self._set_busy(False, "", kopirovanie=True)
        if sbroj is not None:
            self._warn(f"не удалось сохранить фото в {cel}: {sbroj}")
            return
        self.settings.last_dest = str(cel)
        papku_zapomnil = self._sohranit_nastrojki()
        sroki = self._stroki_dlya_ljudej(lydi)
        nastrojki = {
            "Кого ищем": ", ".join(f"{c.nazvanie} ({c.litsa} лиц)" for c in lydi),
            "Способ поиска лиц": self.settings_bar.current_engine_label(),
            "Качество разбора": f"{self.scan_dim} px",
            "Насколько фото должно быть похоже":
                f"{percent_of(self.settings_bar.threshold)}%",
        }
        itog = f"скопировано {len(pari)} фото в {cel}"
        # Отчёт отвечает на тот же вопрос, что и карточка: дома, в папке результатов,
        # экрана нет, а решение «чьё это лицо» человек принимает по-прежнему.
        try:
            write_report(cel / "report.txt", settings=nastrojki, rows=sroki,
                         failures=[*self._poteri.items(),
                                   *((str(put), pochemu) for put, pochemu in poteri)],
                         copied=pari,
                         dropped_faces=self._otbroennye,
                         cache_damage=self._povrezhdeniya,
                         imena_ljudj=[c.nazvanie for c in lydi])
            write_csv(cel / "results.csv", sroki,
                      imena_ljudj=[c.nazvanie for c in lydi])
        except OSError as exc:
            # Снимки уже легли в папку — сказать об этом важно, иначе человек решит,
            # что не сделал ничего, и повторит копирование поверх.
            self._warn(f"фото скопированы, а отчёт записать не удалось: {exc}")
            self.status_label.setText(itog + ", отчёт не записался")
            return
        if poteri:
            itog += f", не скопировано {len(poteri)} — причины в report.txt"
        if not papku_zapomnil:
            # Не отдельной строкой: `status_label` один, и последнее сообщение убивал бы
            # предыдущее — человек видел бы успех там, где настройка не сохранилась.
            itog += "; папку результатов не запомнил — настройки не сохраняются"
        self.status_label.setText(itog)

    # --------------------------------------------------------------- служебное
    def _set_busy(self, zanyato: bool, note: str, kopirovanie: bool = False) -> None:
        """Одно занятое окно на два режима: фоновый разбор и блокирующее копирование.

        `kopirovanie` — не второй флажок для красоты, а честный ответ на «что именно
        сейчас нельзя». Посреди разбора у окна есть рабочий поток, которого можно
        попросить встать, поэтому «Остановить» жива, а сетка доступна: человек листает
        то, что уже нашлось. Посреди копирования потока нет и просить нечего — кнопка
        отмены была бы пустой обещанкой, а живые галочки в сетке ввели бы в обман:
        список для переноса уже составлен, и снять отметку под копированием — значит
        разминуться с тем, что реально легло в папку результатов.

        Всё остальное занято одинаково, и держать два почти одинаковых списка виджетов
        значило бы однажды развести их руками.
        """
        # Единственный источник правды про «что именно нельзя» — поле `_kopiruetsya`:
        # оно истинно ТОЛЬКО на занятом окне копирования. На освобождении (`zanyato=False`)
        # оно обязано быть ложным, иначе сетка осталась бы мёртвой и после копирования.
        kopiruetsya = zanyato and kopirovanie
        self._zanyat = zanyato
        self._kopiruetsya = kopiruetsya
        self.scan_button.setEnabled(not zanyato and bool(self.files))
        # Смену папок окно тоже не принимает: `self.files` под живым разбором поменялся бы
        # так, что человек смотрел бы прогресс одного архива в подписи о другом. Поток
        # держит свой список, но подпись и оценка времени — нет.
        self.add_folder_button.setEnabled(not zanyato)
        # «Убрать все» включается по своему признаку, а не по непустому списку: на
        # освобождении окна список мог остаться пустым, и наивный `not zanyato` вернул
        # бы живую кнопку «убрать» туда, где убирать нечего.
        self.clear_folders_button.setEnabled(not zanyato and bool(self.folders))
        for stroka in self._strok:
            stroka.knopka.setEnabled(not zanyato)
        self.add_person_button.setEnabled(not zanyato)
        self.stop_button.setEnabled(zanyato and not kopiruetsya)
        # Сетка гасится ТОЛЬКО на копировании — см. docstring. `not kopiruetsya`, а не
        # `not kopirovanie`: освобождение окно зовёт с тем же флагом, и наивное чтение
        # оставило бы сетку мёртвой до конца жизни окна.
        self.results.setEnabled(not kopiruetsya)
        # Кнопка копирования — часть того же списка. Отдельно и обязательно: снимки
        # летят в папку результатов с прошлого поиска, а `self.photos` уже пуст, и
        # отчёт вышел бы противоречием (см. `_obnovit_knopku_kopirovaniya`).
        self._obnovit_knopku_kopirovaniya()
        # Полоса настроек — это тоже кнопка запуска, только притворившаяся: она меняет
        # качество разбора, а посреди прогона это «сетке показывать одно разрешение, а
        # лицам быть снятыми в другом». Гасим всю полосу разом — и ползунок, и два
        # списка, и «Очистить оглавление».
        self.settings_bar.setEnabled(not zanyato)
        if not zanyato and self._nastrojki_ozhidayut:
            # Просьбу, поданную за миг до старта разбора, не выкидываем: применяем теперь,
            # с тем же вопросом о новом разборе, который человек и ожидал увидеть.
            self._nastrojki_ozhidayut = False
            self._taymer.start()
        if note:
            self.status_label.setText(note)

    def _otrisovat_okno(self) -> None:
        """Довести занятость до экрана ДО того, как поток интерфейса уходит в работу.

        Без этого шага строка «копируем 200 фото…» осталась бы в очереди событий и
        показалась бы уже после копирования: блокирующий вызов не отдаёт поток интерфейса
        ни на секунду, и окна с баннером «не отвечает» — ровно про это.

        Пользовательский ввод на время отрисовки исключён: клик по ещё живой кнопке в
        этот миг уронил бы второй прогон копирования поверх первого. Таймер дебаунза за
        это же время успеет сработать, но `_primenit_nastrojki` занятость уже видит и
        просьбу отложит.
        """
        QApplication.processEvents(
            QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

    def stop_requested(self) -> None:
        """Просьба остановиться. Отмена завершает текущее фото и сохраняет сделанное.

        Флаг `_otmenen` нужен окну, а не потоку: поток честно пришлёт `done` с тем, что
        успел, и по пустому ответу окно не должно решать, что на фото нет лиц.
        """
        if self.worker is not None:
            self.worker.request_stop()
            self._otmenen = True
            self.status_label.setText("останавливаю после текущего фото…")

    def _sohranit_nastrojki(self, modalno: bool = True) -> bool:
        """Записать настройки и НЕ молчать про промах.

        `Settings.save()` возвращает False, когда на диск легло не всё: том только для
        чтения, полное место, подменённый файл. Человек видит свою настройку на экране и
        узнаёт о пропаже только в следующем запуске, когда всё вернётся к умолчалкам.
        Модальным окном бить по каждому движению мыши — наказание, поэтому в диалог оно
        идёт один раз, а в строку состояния при каждом промахе.
        """
        sohranilos = self.settings.save()
        if not sohranilos:
            self.status_label.setText(
                f"настройки не сохраняются: файл {self.settings.path} недоступен для записи")
            if modalno and not self._o_perezapisi_skazano:
                self._o_perezapisi_skazano = True
                self._warn("настройки не сохраняются: файл "
                           f"{self.settings.path} недоступен для записи. Приложение "
                           "работает, но после перезапуска вернётся к значениям по "
                           "умолчанию.")
        return sohranilos

    def _warn(self, text: str) -> None:
        QMessageBox.warning(self, "Внимание", text)

    def closeEvent(self, event) -> None:      # noqa: N802 (имя Qt)
        """Закрыть окно так, чтобы рабочий поток не остался с мёртвой базой.

        `worker.wait(3000)` стоял здесь и раньше, но результатом никто не
        интересовался, а `wait` возвращает `False` ровно тогда, когда поток ещё жив.
        Модели читаются 13–21 с, и `scan_photos` во время чтения флаг отмены не
        проверяет вовсе, так что поток переживает эти три секунды почти всегда.
        Оглавление закрывалось под ним, следующий `cache.get` в рабочем потоке падал с
        `ProgrammingError: Cannot operate on a closed database`, и человек получал
        «оглавление недоступно» модальным окном в уже закрытом окне.

        Поэтому: ждём, СПРАШИВАЕМ результат и, если поток ещё жив, отменяем закрытие.
        Окно переживает поток не только ради базы: `ScanWorker` — ребёнок окна, и если
        окно умрёт вместе с ним, человек увидит «QThread: Destroyed while thread is
        still running», а недописанное оглавление останется незакрытым.
        """
        self._sbrosit_debaun_pri_zakrytii()
        potok = self.worker
        if self._zakrytie_otlozheno:
            # Человек жмёт «закрыть» второй раз, пока мы ждём поток: снова вешать три
            # секунды на поток интерфейса незачем, отложенное закрытие уже назначено.
            event.ignore()
            return
        if potok is not None and potok.isRunning():
            potok.request_stop()
            if not potok.wait(OZHDANIE_ZAKRYTIYA_MS):
                event.ignore()
                self._zakrytie_otlozheno = True
                self.status_label.setText("останавливаю разбор: окно закроется само, как "
                                          "только поток доделает текущий снимок")
                potok.finished.connect(self._zakrytsja_posle_potoka)
                return
        self._osvobodit_oglavlenie()
        # Просмотр не модальный, а значит переживает родителя сам: без явного закрытия
        # на экране остаётся кадр, которым больше никто не управляет, с живой галочкой
        # «копировать это фото» от исчезнувшего списка.
        if self._okno_prosmotra is not None:
            self._okno_prosmotra.close()
        super().closeEvent(event)

    def _zakrytsja_posle_potoka(self) -> None:
        """Поток встал — закрываемся всерьёз: событие пройдёт заново, теперь уже чисто."""
        if self._zakrytie_otlozheno:
            self._zakrytie_otlozheno = False
            self.close()

    def _osvobodit_oglavlenie(self) -> None:
        """Отпустить соединение с оглавлением — ровно один раз за жизнь окна.

        Сюда приходят дважды (сейчас и после остановки потока), а исключение из
        `closeEvent` человек увидел бы как непонимающую кнопку.
        """
        if self._oglavlenie_zakryto:
            return
        self._oglavlenie_zakryto = True
        try:
            self.cache.close()
        except Exception:                      # noqa: BLE001 — уходить — не искать правых
            pass

    def _sbrosit_debaun_pri_zakrytii(self) -> None:
        """Дописать похожесть, если человек отпустил ползунок и тут же закрыл окно.

        Дебаунс — это обещание «применю через 150 мс», а не «применю, если доживу»: без
        этого сброса последние полсекунды работы полоской молча возвращались к старому
        числу при следующем запуске. Способ поиска и качество разбора при закрытии НЕ
        применяются: они требуют нового разбора, а устраивать его под закрытие окна —
        значит оставить модальный вопрос, на который уже некому ответить.
        """
        if not self._taymer.isActive():
            return
        self._taymer.stop()
        if self._zanyat:
            return                            # посреди разбора не коммитим ничего
        self.settings.threshold = self.settings_bar.threshold
        self._sohranit_nastrojki()
