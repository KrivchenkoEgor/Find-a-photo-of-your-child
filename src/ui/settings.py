"""Полоса настроек над результатами. Всё применяется сразу, кнопки «Применить» нет.

Три вещи, которые человек решает сам и охотно:

* насколько фото должно быть похоже — ползунок 30..60 %;
* каким способом искать лица — «Быстрее», «Точнее» или «Оба»;
* с каким качеством разбирать папку — «внимательное» или «обычное».

Рядом со способом поиска стоит четвёртое, что человек решает по этой полосе, но не
настройкой, а числом: «≈ 4 мин на 131 фото» (спецификация, раздел 6 — «в подписи —
минуты на текущем архиве»). Окно передаёт count снимков сразу после выбора папок через
`set_archive_size`, до всякого разбора: цена настройки обязана быть видна ДО нажатия
«Начать разбор папки», а не после трёх минут ожидания. Пока папка не выбрана — подписи
нет: считать нечего, а «0 с» читалось бы обещанием.

Спецификация (раздел 6) держит настройки над сеткой результатов и требует, чтобы смена
похожести пересчитывала список на месте: лица уже в памяти, нового разбора нет, человек
водит ползунок — и кучка «Похоже» наполняется. Поэтому сигнал `changed` идёт на каждое
изменение, а не по кнопке.

Про две строки — это сам спецификация, раздел 6: «Полоса в две строки: в первой подпись
ползунка, ползунок и число, во второй — способ поиска, качество разбора и "Очистить
оглавление"». Исторически здесь стоял один `QHBoxLayout`, и живое окно показало вместо
подписи ползунка «Насколько фото должно бы»: при нехватке ширины Qt делит дефицит по всем
виджетам строки, а обрезанная подпись — это настройка, которую человек не прочитал
(скриншот, задача T2). Для человека не изменилось ничего: те же три настройки над теми же
результатами, те же минуты рядом со способом поиска, и `main_window.py` по-прежнему берёт
ширину окна из `sizeHint()` полосы — после переноса она только меньше.

Почему здесь нет слов «порог», «детектор», «уверенность». Это не косметика: человек,
который ищет фото своего ребёнка, не знает, что такое косинусное расстояние, и слово
«порог» ему не говорит ни о чём. Спецификация прямо запрещает этот словарь в интерфейсе,
а `tests/test_ui_settings.py` обходит все надписи и подсказки виджета — так запрет
перестаёт быть пожеланием. Вместо названия настройки подсказка говорит, что изменится для
человека: «влево — больше фото, но чаще чужих».

Про порядок строк в `__init__`. Задача 15 собирает окно так: создаёт полосу, восстанавливает
сохранённые значения и только потом подключает `changed`. Чтобы восстановление не выглядело
как изменение настройки, все `setValue`/`setCurrentIndex` стоят ДО `connect`. Порядок
держится на четырёх строках, и тест проверяет его отдельно: переставить `connect` выше
`setValue` — это час отладки «окно само начало пересчитывать архив до выбора папки».
"""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QPushButton, QSlider,
                               QVBoxLayout, QWidget)

from utils.config import (DEFAULT_ENGINE, DEFAULT_MAX_DIM, DEFAULT_THRESHOLD,
                          THRESHOLD_RANGE)

# Ключи режимов — те же, что принимает `core.engine.make_engine`, и порядок совпадает с
# `config.ENGINE_KEYS`: подпись находится по номеру строки, путать их нельзя.
ENGINES: tuple[tuple[str, str], ...] = (("both", "Оба"), ("yunet", "Быстрее"),
                                        ("insight", "Точнее"))

# Две длины длинной стороны снимка. Выбор не между двумя числами, а между двумя
# последствиями: 2400 находит мелкие лица на групповом фото, 1200 разбирает папку заметно
# быстрее, но стоит 3–7 процентных пунктов сходства (замеры — docs/bench-2026-09-28).
QUALITY: tuple[tuple[int, str], ...] = ((2400, "внимательное (точнее)"),
                                        (1200, "обычное (быстрее)"))

# Измеренный факт, который объясняет шкалу лучше любого описания устройства настройки: на
# подтверждённых фото настоящего ребёнка похожесть поднималась максимум до 39,2 %,
# а выше 67 % не бывало ни разу — даже когда ребёнок на снимке точно есть.
PODSKAZKA_POROGA = "выше 67% не бывает даже когда ребёнок на фото точно есть"

# Что делает настройка для человека — формулировка спецификации: «влево = больше фото,
# вправо = только явные».
PODSKAZKA_POLSUNKA = ("влево — больше фото в списке, но среди них чаще чужие; "
                      "вправо — только очень похожие")

PODSKAZKA_REZHIMA = ("«Быстрее» ищет одним способом и не всегда замечает ребёнка на "
                     "групповом фото; «Оба» ищет двумя и берёт лучший ответ — надёжнее, "
                     "но разбор дольше")

PODSKAZKA_KACHESTVA = ("«Внимательное» разбирает фото крупно и находит мелкие лица; "
                       "«обычное» быстрее, но часть лиц не увидит. Смена качества требует "
                       "нового разбора папки")

PODSKAZKA_CHISTKI = ("Приложение забудет, какие фото оно уже разобрало. Следующий разбор "
                     "прочитает каждый снимок заново — это минуты. Сами фото не удаляются "
                     "и не меняются")

PODSKAZKA_CHASTOTA = ("Число справа — насколько фото должно быть похоже на выбранное "
                      "фото ребёнка, в процентах")

# Сколько секунд разбор стоит на одном снимке, измерено на архиве проекта: 131 снимок
# по 24 Мп, 2400 px, один поток, пустая память разбора (см. docs/bench-2026-09-28 и
# спецификацию, раздел 2). Спецификация требует «в подписи — минуты на текущем
# архиве»: человек выбирает способ поиска не по названию, а по тому, сколько он ждёт.
SEKUND_NA_FOTO: dict[str, float] = {"both": 1.9, "insight": 1.0, "yunet": 0.5}

PODSKAZKA_OCENKI = ("Столько занимает ПЕРВЫЙ разбор: каждый снимок читается моделями. "
                    "Повторный запуск той же папки заметно быстрее — лица уже посчитаны "
                    "и лежат в памяти приложения, а кнопка «Забыть разобранные фото» "
                    "возвращает разбор к этому числу.")


def _mnohozhechestvo_seconds(sec: float) -> str:
    """Длительность по-русски: «45 с», «2 мин», «1 ч 5 мин».

    Секундомер человеку полезнее точности до десятичных: на 20-минутном разборе цифра
    «19,7» читается как обещание, а разница в полторы минуты — как обман.
    """
    vsekgo = max(0, int(round(sec)))
    if vsekgo < 60:
        return f"{vsekgo} с"
    minimal = int(round(vsekgo / 60))
    if minimal < 60:
        return f"{minimal} мин"
    chasy, ostatok = divmod(minimal, 60)
    return f"{chasy} ч {ostatok} мин" if ostatok else f"{chasy} ч"


class SettingsBar(QWidget):
    """Две строки над результатами: ползунок похожести, режим поиска, качество разбора."""

    changed = Signal(float, str, int)      # похожесть, режим, качество разбора
    clear_cache = Signal()                 # «Очистить оглавление», спецификация раздел 5

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Настройки поиска")

        # --- ползунок похожести: внутри целые проценты, наружу отдаём float ---
        self.threshold_label = QLabel("Насколько фото должно быть похоже")
        self.threshold_label.setToolTip(PODSKAZKA_POLSUNKA)
        # Подпись настройки не имеет права превращаться в огрызок. `minimumSizeHint`
        # QLabel тут не спасает: при дефиците ширины Qt его не соблюдает, а вот явный
        # `minimumWidth` соблюдает (замер offscreen, см. `tests/test_ui_settings.py`).
        # Основное лечение — ниже, в раскладке: метке отдана строка целиком.
        self.threshold_label.setMinimumWidth(self.threshold_label.sizeHint().width())
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(_procent(THRESHOLD_RANGE[0]), _procent(THRESHOLD_RANGE[1]))
        self._slider.setSingleStep(1)
        self._slider.setToolTip(PODSKAZKA_POLSUNKA)
        self.value_label = QLabel(f"{_procent(DEFAULT_THRESHOLD)}%")
        self.value_label.setToolTip(PODSKAZKA_CHASTOTA)
        self._slider.setValue(_procent(DEFAULT_THRESHOLD))
        # connect — после setValue: см. модульный docstring
        self._slider.valueChanged.connect(self._na_polsunke)

        # --- режим поиска ---
        self._engine_box = QComboBox()
        for kluch, podpis in ENGINES:
            self._engine_box.addItem(podpis, kluch)
        self._engine_box.setToolTip(PODSKAZKA_REZHIMA)
        self._engine_box.setCurrentIndex(_nomer([k for k, _ in ENGINES], DEFAULT_ENGINE))
        self._engine_box.currentIndexChanged.connect(self._na_rezhime)

        # --- качество разбора ---
        self._dim_box = QComboBox()
        for znachenie, podpis in QUALITY:
            self._dim_box.addItem(podpis, znachenie)
        self._dim_box.setToolTip(PODSKAZKA_KACHESTVA)
        self._dim_box.setCurrentIndex(_blizhe([z for z, _ in QUALITY], DEFAULT_MAX_DIM))
        self._dim_box.currentIndexChanged.connect(self._na_kachestve)

        # Оценка времени разбора — подпись рядом со способом поиска (спецификация,
        # раздел 6: «в подписи — минуты на текущем архиве»). Пустая, пока папка не
        # выбрана: выдумывать число нечем, а врать про «минуты» на пустом месте
        # хуже, чем молчать.
        self.estimate_label = QLabel("")
        self.estimate_label.setStyleSheet("color: gray")
        self.estimate_label.setToolTip(PODSKAZKA_OCENKI)
        self._foto_v_arhive = 0

        self.clear_button = QPushButton("Забыть разобранные фото")
        self.clear_button.setToolTip(PODSKAZKA_CHISTKI)
        self.clear_button.clicked.connect(self.clear_cache.emit)

        hint = QLabel(PODSKAZKA_POROGA)
        hint.setStyleSheet("color: gray")

        # --- раскладка: две строки, а не одна ---------------------------------------
        #
        # В одной `QHBoxLayout` всем настройкам метка ползунка получала ровно то, что
        # оставалось после двух `QComboBox`, кнопки и подписи минут, — и живой экран
        # показывал «Насколько фото должно бы». Разводим по строкам: рядом с меткой
        # остаются только ползунок и его число, конкурентов за пиксели больше нет.
        # Порядок `setValue` -> `connect` при переносе не менялся: см. модульный docstring.
        row_polsunka = QHBoxLayout()
        row_polsunka.addWidget(self.threshold_label)
        row_polsunka.addWidget(self._slider)
        row_polsunka.addWidget(self.value_label)

        row_rezhimov = QHBoxLayout()
        row_rezhimov.addWidget(QLabel("поиск:"))
        row_rezhimov.addWidget(self._engine_box)
        row_rezhimov.addWidget(self.estimate_label)
        row_rezhimov.addWidget(QLabel("качество разбора:"))
        row_rezhimov.addWidget(self._dim_box)
        row_rezhimov.addWidget(self.clear_button)

        layout = QVBoxLayout(self)
        layout.addLayout(row_polsunka)
        layout.addLayout(row_rezhimov)
        layout.addWidget(hint)

    # --- текущие значения ------------------------------------------------------
    @property
    def threshold(self) -> float:
        """Похожесть от 0.30 до 0.60 — ровно то число, которого ждёт сравнение."""
        return self._slider.value() / 100.0

    @property
    def engine(self) -> str:
        """Ключ способа поиска: 'both' | 'yunet' | 'insight' — его принимает фабрика движков."""
        return str(self._engine_box.currentData())

    @property
    def max_dim(self) -> int:
        """Длинная сторона снимка в пикселях, с которой фото уходит на разбор."""
        return int(self._dim_box.currentData())

    def engine_label(self, index: int) -> str:
        """Подпись способа поиска по его номеру."""
        return self._engine_box.itemText(index)

    def current_engine_label(self) -> str:
        """Надпись выбранного способа — её печатаем в отчёте."""
        return self._engine_box.currentText()

    # --- оценка времени разбора --------------------------------------------------
    def set_archive_size(self, n_foto: int) -> None:
        """Сколько фото найдено в выбранных папках — отсюда считается ожидание.

        Полоса не ходит в файловую систему: число приносит окно сразу после выбора
        папок, ещё до всякого разбора. Человек видит цену настройки ДО того, как
        нажал «Начать разбор папки», — и может выбрать «Быстрее», не гадая.
        """
        try:
            self._foto_v_arhive = max(0, int(n_foto))
        except (TypeError, ValueError):
            self._foto_v_arhive = 0
        self._obnovit_ocenku()

    def _obnovit_ocenku(self) -> None:
        """Подпись у способа поиска: «≈4 мин на 131 фото», а при нуле — пусто.

        Число честна называть оценкой, а не обещанием: оно снято на одном архиве с
        24-мегапиксельными снимками, и телефонная папка с кадрами поменьше разбирается
        быстрее. Поэтому «≈», а не «примерно столько и ждите».
        """
        if not self._foto_v_arhive:
            self.estimate_label.setText("")
            return
        sek = SEKUND_NA_FOTO.get(self.engine, SEKUND_NA_FOTO["both"])
        self.estimate_label.setText(
            f"≈ {_mnohozhechestvo_seconds(sek * self._foto_v_arhive)} "
            f"на {self._foto_v_arhive} фото")

    # --- установка значений: восстановление из настроек и тесты -----------------
    def set_threshold(self, value: float) -> None:
        """Ставит похожесть на ползунок, за шкалу не выпускает.

        Округляем, а не отбрасываем дробь: у `int(0.57 * 100)` получается 56, и
        сохранённая настройка на каждом следующем запуске уезжала бы на процент вниз.
        """
        self._slider.setValue(_procent(_zazhim(value)))

    def set_engine(self, key: str) -> None:
        """Способ поиска из файла настроек.

        Чужой ключ — возврат к «Оба», а не исключение: строку в ini-файл мог дописать
        человек, и из-за опечатки в слове приложение не обязано терять окно целиком.
        """
        self._engine_box.setCurrentIndex(_nomer([k for k, _ in ENGINES], str(key)))

    def set_max_dim(self, value: int) -> None:
        """Качество разбора.

        Полоса предлагает два варианта, а `Settings.max_dim` пропускает любое число от 600
        до 6000 — ставим ближайший предложенный. Иначе значение 1800 из подправленного
        руками файла уронило бы сборку окна, и человек увидел бы только пустоту.
        """
        try:
            shirina = int(value)
        except (TypeError, ValueError):
            return
        self._dim_box.setCurrentIndex(_blizhe([z for z, _ in QUALITY], shirina))

    # --- обработчики: каждый меняет свою настройку и зовёт _izvestit ------------
    def _na_polsunke(self, procent: int) -> None:
        self.value_label.setText(f"{procent}%")
        self._izvestit()

    def _na_rezhime(self, _indeks: int) -> None:
        # Способ поиска меняет и ожидание: «Быстрее» — это ровно про время, и подпись
        # обязана перестроиться вместе с выбором, а не остаться от прежнего способа.
        self._obnovit_ocenku()
        self._izvestit()

    def _na_kachestve(self, _indeks: int) -> None:
        self._izvestit()

    def _izvestit(self) -> None:
        """Единственная точка, откуда наружу уходит `changed`.

        Один вызов вместо трёх копий: сигнал обязан нести актуальную тройку значений, а
        размывать её по трём обработчикам — значит однажды рассинхронизировать экран с тем,
        что реально ушло на сравнение.
        """
        self.changed.emit(self.threshold, self.engine, self.max_dim)


# --- вспомогательные функции -------------------------------------------------------


def _procent(pohozhest: float) -> int:
    """Похожесть 0.38 -> проценты 38. Только с округлением, см. `set_threshold`."""
    return round(float(pohozhest) * 100)


def _zazhim(value: float) -> float:
    """Держим похожесть внутри шкалы: наружу не уйдёт ни 0.05, ни 0.99, ни NaN.

    Значение приезжает из ini-файла, где оно могло стать чем угодно. `Settings` своё
    проверит, но полоса не должна полагаться на то, что её всегда вызывают извне
    воспитанным кодом: на экране край шкалы честнее молчаливого 5 %.
    """
    niz, verh = THRESHOLD_RANGE
    try:
        znachenie = float(value)
    except (TypeError, ValueError):
        return DEFAULT_THRESHOLD
    if znachenie != znachenie:                 # NaN не равен самому себе — так его видно
        return DEFAULT_THRESHOLD
    return max(niz, min(verh, znachenie))


def _nomer(kluchi: Sequence[str], nuzhny: str) -> int:
    """Номер строки списка с нужным ключом; чужой ключ — первая строка (умолчалка)."""
    return kluchi.index(nuzhny) if nuzhny in kluchi else 0


def _blizhe(varianty: Sequence[int], sprashivaet: int) -> int:
    """Номер ближайшего предложенного варианта.

    `min` при равенстве берёт первый: на 1800 px это «внимательное», то есть вариант,
    который человек уже видел на экране, — менять настройку без смены настройки нельзя.
    """
    return min(range(len(varianty)), key=lambda i: abs(varianty[i] - sprashivaet))
