"""Сетка найденного. Процент и рамка лица — чтобы человек решил сам.

Разброс подтверждённых совпадений 39…67 %, поэтому фото ниже настроенного числа не
выбрасываются: они ложатся в свёрнутую кучку «слабое сходство», которую человек может
развернуть и где стоит явная подпись — сколько там и что будет, если нажать
(спецификация, раздел 6).

Семь решений, которые здесь не косметика.

**Отмечено равно видно.** Галочки снимаются со скрытой кучки: в `checked` попадает
только то, что человек видел глазами. Прятать отметку и молча копировать снимок — значит
решать за человека по тому, чего он не видел. Состояние каждой галочки при этом
запоминается по пути файла, поэтому разворот кучки возвращает отметки, а не обнуляет их,
а скрытая с галочкой строка названа числом в подписи.

**Экран имеет потолок.** Карточка — это виджет, а виджет стоит и времени, и памяти: на
`MAKSIMUM_KARTOTSEK` сетка останавливается и ждёт явного «показать ещё». Молча показать
триста из двух тысяч было бы цензурой, поэтому скрытое названо числом в подписи под
сеткой, а снять потолок может только кнопка рядом с ней. Замер — у константы.

**Число объяснено рядом.** Карточка с «45 %» без пояснения читается как провал:
измеренная полоса настоящих совпадений — 39…67 % (спецификация, раздел 2). Поэтому под
подписью живёт строка, которая говорит это прямым текстом, и тест проверяет
её наличие, а не только отсутствие запрещённых слов. Само число продублировано на
кадре: подпись под рядом на ноутбуке 13" остаётся за сгибом, а число — ровно тот
признак, по которому человек решает.

Про живое число в этой строке — это сам спецификации раздел 6: подпись «обязана брать
текущее значение ползунка из настройки, а не хранить замер в тексте». Когда-то здесь
сидела константа с зашитым «так что 45 % — не промах поиска», и на скриншоте живого окна
человек с ползунком на 38 % читал про 45 %. Число в тексте, которое ни от чего не
зависит, — обещание, которое экран не держит ни при одной настройке, поэтому строка
теперь строится функцией `podskaz_chisla` от текущего порога и обновляется на каждый
`show_rows`: окно зовёт его с каждого движения ползунка.

**Миниатюру собирает `face_to_qpixmap`.** Своя сборка из буфера numpy с длиной строки
`3 * ширина` для формата Qt `RGB888` не выровнена по 4 байта — см. docstring той функции
в `ui/face_picker.py`. На этой сборке PySide6 кривая строка проходит молча, а на платформе,
где Qt проигнорирует `bytesPerLine`, даёт съехавшие ряды; сетка проверяет и то, что
миниатюру собирает именно `face_to_qpixmap`, и то, что экран равен массиву.

**Кэш базовых миниатюр.** Ползунок похожести человек водит по двадцати значениям за раз,
и без кэша каждое движение заново декодировало бы сотни снимков с диска: «мгновенный
пересчёт» превратился бы в подвисание окна. Рамка дорисовывается на копии кадра, поэтому
кэшированный оригинал остаётся чистым.

**Первый экран не стоит.** Кэш спасает второй и следующий пересчёт, но не первый: экран с
находками собирается из того, что ещё ничего не читано. Замер на архиве проекта (131
снимок по 24 МП): `load_photo(path, 2400)` — 403 мс на файл, и 176 мс из них — уменьшение
до 2400 px, которое миниатюре в 220 px не нужно вовсе. На пятидесяти совпадениях это
двадцать секунд стоящего окна, а спецификация (раздел 7) требует «без вида зависшего
окна». Поэтому кадр для карточки читается быстрым путём (`Image.draft` для JPEG умеет
сразу 1/8 снимка — 162 мс против 403 мс), и за один вход в `show_rows` читается РОВНО
ОДИН кадр: остальные доходят очередью по одному за тик событий, сверху вниз, то есть в
том порядке, в котором человек их и читает. Окно между карточками отвечает на клики.

**Масштаб рамки не зависит от способа чтения.** Координаты лица приходят в пикселях кадра
`max_dim`, и коэффициент считается от длинной стороны ИМЕННО этого кадра — она берётся из
заголовка файла, без декодировки. Быстрый путь даёт тот же кадр, только мягче: зелёная
рамка лежит на лице и при 1/8, и при полном чтении. Рамка не по месту хуже отсутствующей
— родитель решит, что нашёлся чужой человек.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QCheckBox, QGridLayout, QHBoxLayout, QLabel,
                               QPushButton, QScrollArea, QVBoxLayout, QWidget)

from core.matcher import percent_of
from core.report import ResultRow
from core.scanner import load_photo
from ui.face_picker import RamkaLica, face_to_qpixmap, ramki_na_kadr

THUMB = 220                 # сторона миниатюры в пикселях
COLUMNS = 4                 # карточек в ряду, пока ширина окна не известна
MAKSIMUM_KOLONOK = 8          # потолок: двадцать подписей в ряд не читаются
PLITKA = 128                # серый цвет плитки «кадр ещё не дошёл»

# Цвет, толщина и нижняя граница размера рамки лица переехали в `face_picker` вместе с
# правилом: окно просмотра обязано рисовать тот же прямоугольник, что и сетка.

# Сколько базовых миниатюр держать в памяти. Двести кадров по 220 px — это ~30 МБ;
# дальше кэш вытесняет самые старые. FIFO, а не LRU: обход сетки и так идёт сверху вниз,
# а учёт попаданий обошёлся бы дороже, чем экономит пару лишних декодировок.
_MINIATUR_PREDEL = 200

# Кадр, которым платим за миниатюру в 220 px: двойной запас на плашку числа, на
# округление и на разворот кучки, разрыва между ним и 220 px глаз не видит.
ZAPAS_DEKODIROVANIYA = 2

# Сколько кадров первый экран имеет права прочитать на себе. Ноль означал бы, что
# даже единственная найденная карточка приходит пустой; больше одного — возврат к
# «окно стоит, пока дочитается ряд».
SINHRONNO_KADROV = 1

# Пауза между карточками очереди. Не «ещё одна задержка», а гарантия того, что между
# двумя декодировками поток интерфейса успевает отрисовать окно и ответить на клик.
OGRESH_KADRA_MS = 15

# Сколько карточек сетка имеет право построить за один раз. Потолок — не экономия
# красоты, а цена виджета: замер на этом же архиве, развёрнутый до синтетики, дал
# 3000 карточек ≈ 350 мс + 772 МБ, 6000 ≈ 1,04 с + 859 МБ RSS, и это только сборка
# виджетов без единого прочитанного снимка. Плюс ~160 мс очереди на карточку: на
# десяти тысячах находок одна прокрутка съела бы 25 минут.
#
# Ограничение обязано быть видимой частью интерфейса, а не молчаливой цензурой:
# скрытое считается в подписи «отмечено, но скрыто», и кнопка «показать ещё»
# поднимает потолок ровно на столько же, на сколько он установлен.
MAKSIMUM_KARTOTSEK = 300
DOBAVLYAET_KARTOTSEK = 300

PODSKAZKA_ESHCHE = ("Показать следующие карточки. Сетка намеренно не строит сразу "
                    "тысячи снимков: окно с ними замирает, а память кончается.")

# «Фото» не склоняется, поэтому такая форма верна и при одном снимке, и при трёхстах:
# существительное по числу здесь ломалось бы на каждом шаге кнопки.
KNOPKA_ESHCHE = "показать ещё {n} фото"

# Подпись под сеткой, когда показана не вся находка. Слово «скрыто» в ней обязательное:
# человек обязан отличить «приложение не нашло» от «приложение нашло, но не показал».
POTOLOK_V_PODPISI = (" Показано: {pokazano} из {vsego}. Остальные — по кнопке "
                     "«показать ещё».")


# Шаблон пояснения под сеткой. `{procent}` — единственное живое число в подписи: вёрстка
# текста менялась один раз, а цифра обязана меняться на каждом движении ползунка.
SHABLON_POJASENIJA = (
    "Число на карточке — насколько это фото похоже на выбранное лицо. На снимках того же "
    "ребёнка оно держится в пределах 39–67 %, поэтому низкое число — не промах поиска, а "
    "обычное совпадение. Сейчас в «похоже» попадает всё от {procent} %; что верно, а что "
    "нет — решать вам по зелёной рамке лица.")


def podskaz_chisla(procent: int) -> str:
    """Пояснение к числу на карточке с текущим значением ползунка.

    Функция вместо константы: зашитое в текст число врёт при любом другом положении
    настройки (было «так что 45 % — не промах поиска» при ползунке на 38 %). Новая
    формулировка верна при любом пороге и заодно объясняет, что ползунок делает.
    """
    return SHABLON_POJASENIJA.format(procent=int(procent))


def _procent_pojasnenija(porog: object) -> int | None:
    """Порог в проценты для пояснения; None — если приехало не число.

    Правило округления не своё: `core.matcher.percent_of` — единственный владелец
    перевода «сходство -> проценты», и подпись обязана называть ровно то число, которое
    печатает карточка и отчёт. Непригодное значение (None, строка, NaN, Inf) даёт None:
    сетка не падает, а подпись остаётся прежней — врать «от 0 %» хуже, чем молчать.
    """
    try:
        znachenie = float(porog)                       # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not bool(np.isfinite(znachenie)):
        return None
    return percent_of(znachenie)


PUSTAYA_SETKA = ("Здесь появятся найденные фото. Сначала выберите папку с фото "
                 "и фото ребёнка.")

ZAGLUISHKA = "снимок не читается"
KUCHKA_PUSTO = "слабое сходство не найдено"
KUCHKA_VIDNO = "скрыть слабое сходство"

# Вторая строка карточки: ПО КАКОМУ из выбранных лиц нашёлся этот снимок. Человек
# отмечает несколько примеров, и среди них может стоять брат или взрослый; по одному
# числу «45 %» эти два случая не различить, а различать их обязан человек, который
# решает, что уносить в папку результатов (задача T15).
#
# Подпись ставится и на главном лице тоже: отсутствующую строку человек читает как
# «приложение не знает», а не как «сработало главное лицо». Это тот же класс дефекта,
# что рамка, которая есть, но её не видно, — обещание без подтверждения.
# Стиль второй строки: тот же серый, что у остальных пояснений, чтобы карточка не
# превращалась в два равноправных заголовка.
STIL_PODPISI_STROKI = "color: gray"


def _kto_i_skolko(row: ResultRow, imena: Sequence[str]) -> str:
    """Вторая строка карточки: «ребёнок 61 % · мама 45 %» — кто в кадре и насколько.

    Прежняя подпись («похоже на лицо 3») называла отметку человека, а не человека: при
    двух искомых «лицо 3» не отвечало на вопрос, чьё это фото. Пустая строка — если в
    кадре не нашлось никого из искомых: карточка остаётся с одним числом, и это честно.
    """
    if len(imena) < 2:
        # Один искомый человек: имя под числом повторяет заголовок шага «Кого ищем» и
        # отвечает на вопрос, которого человек не задавал. Со вторым оно становится ответом.
        return ""
    lydi: dict[int, int] = {}
    for sovp in row.sovpadeniya:
        lydi[sovp.chelovek] = max(lydi.get(sovp.chelovek, 0), sovp.percent)
    return " · ".join(f"{imena[n - 1]} {procent} %"
                      for n, procent in sorted(lydi.items()) if 1 <= n <= len(imena))



def _rodnaja_dlina(path: Path) -> int | None:
    """Длинная сторона исходника по заголовку файла, без единого декодированного пикселя.

    Нужна ровно одна длина: `load_photo` уменьшает снимок до `max_dim`, и рамка лица
    пришла в пикселях именно того кадра. Заголовок читается микросекунды, а без этой
    длины коэффициент для рамки пришлось бы брать от кадра, который мы реально
    декодировали, — и рамка уехала бы в разы.
    """
    try:
        with Image.open(path) as im:
            # длинная сторона устойчива к повороту: `exif_transpose` меняет стороны
            # местами, а `max` от этого не меняется
            return int(max(im.size))
    except Exception:                        # noqa: BLE001 — нечитаемый файл не повод падать
        return None


def _bytryj_kadr(path: Path, storona: int) -> np.ndarray | None:
    """Кадр для миниатюры без полной декодировки снимка. None — быстрым путём нельзя.

    `Image.draft` просит у декодера JPEG сразу уменьшенный вариант (1/2, 1/4 или 1/8), и
    чтение обходится в разы дешевле: замер на архиве проекта — 162 мс против 403 мс на
    `load_photo(path, 2400)`. PNG, WEBP и HEIC уменьшения при чтении не умеют, и для них
    быстрый путь честно отвечает «не могу»: вызывающий прочтёт через `load_photo`, а не
    подставит в карточку чёрный квадрат.
    """
    try:
        with Image.open(path) as im:
            if im.format != "JPEG":
                return None
            im.draft("RGB", (storona, storona))
            # ориентация из EXIF обязательна: без поворота лицо уехало бы в другой угол
            povernutyj = ImageOps.exif_transpose(im).convert("RGB")
    except Exception:                        # noqa: BLE001 — решает вызывающий
        return None
    # Pillow отдаёт RGB, а `_ramka` и `face_to_qpixmap` работают с BGR — как `_to_bgr`
    return np.ascontiguousarray(np.asarray(povernutyj)[:, :, ::-1])


def _kadr_miniatury(path: Path, max_dim: int,
                    side: int) -> tuple[np.ndarray, float] | None:
    """Уменьшенный кадр для карточки и коэффициент для рамки. None — файл не открылся.

    Рамка лица задана в пикселях кадра размера `max_dim` (в нём лицо и искали), поэтому
    миниатюру делаем из того же кадра: читаем `load_photo(path, max_dim)` и режем. Иначе
    масштаб рамки пришлось бы где-то хранить, а так он однозначен.

    Единственная оговорка — КАКИМ чтением. Быстрый путь (`_bytryj_kadr`) даёт тот же
    кадр, только мягче, и потому коэффициент для рамки считается не от прочитанных
    пикселей, а от длинной стороны кадра-эталона: `side / etalon`. Заодно и уменьшать мы
    просим не до `max_dim`, а до стороны, которая нужна карточке: переход 6000 -> 2400 px
    для миниатюры в 220 px — это треть времени чтения и ноль пользы.

    `int(1 * k)` от вырожденного кадра (1 px по короткой стороне — легальный ответ
    `load_photo`, задача 3 уменьшает только длинную) даёт ноль, а `cv2.resize` на нулевой
    стороне бросает assertion. Поэтому обе стороны держим не меньше пикселя: одна
    нечитаемая панорама не имеет права ронять всю сетку.
    """
    rodalnaya = _rodnaja_dlina(path)
    if not rodalnaya:
        return None
    etalon = max(1, min(rodalnaya, int(max_dim)))     # кадр, в котором пришла рамка
    storona_chteniya = max(side * ZAPAS_DEKODIROVANIYA, side + 1)
    dekodirovannyj = _bytryj_kadr(path, storona_chteniya)
    if dekodirovannyj is None:
        dekodirovannyj = load_photo(path, max_dim=storona_chteniya)
    if dekodirovannyj is None:
        return None
    vysota, shirina = dekodirovannyj.shape[:2]
    if vysota < 1 or shirina < 1:
        return None
    # Длинная сторона миниатюры ставится РАВНОЙ `side`, а не `int(side * k)`: от обрезки
    # дробного произведения длинная сторона выходила 219 px при `side` 220, а коэффициент
    # для рамки при этом делил на 220 — то есть рамка систематически ехала на пиксель.
    dlina = max(vysota, shirina)
    korotkaya = max(1, round(min(vysota, shirina) * side / dlina))
    novye_dimensii = (korotkaya, side) if vysota >= shirina else (side, korotkaya)
    try:
        umenshennoe = cv2.resize(dekodirovannyj, novye_dimensii,
                                 interpolation=cv2.INTER_AREA)
    except cv2.error:                        # чужой формат кадра — заглушка, не падение
        return None
    return umenshennoe, side / etalon


def ramki_stroki(row: ResultRow) -> list[RamkaLica]:
    """Рамки карточки: каждое найденное лицо и его номер в кадре.

    Только совпадения. Зелёная рамка значит «вот тут ребёнок» — так ей пишет и подпись
    под сеткой, и человек решает «моё / не моё» по ней. На групповом фото из 24 лиц
    искали двоих: обвести остальных двадцать — значит показать «найдены все», и ровно за
    это правку и попросили (задача T19).

    Строка без `sovpadeniya`, но с живым `box` собрана не `build_rows` (тесты, ручные
    вызовы): рисуем хотя бы её. Молча потерять рамку, которую карточка рисовала всегда,
    — значит спрятать находку с экрана.
    """
    lica = sorted({s.lice for s in row.sovpadeniya})
    if lica:
        # Одно лицо, похожее на двух искомых сразу, получает ОДНУ рамку: рамок в кадре
        # столько, сколько лиц, а не сколько пар «лицо × человек».
        return [RamkaLica(row.vse_lica[n - 1], n) for n in lica if 1 <= n <= len(row.vse_lica)]
    if row.box is not None and row.matched:
        return [RamkaLica(row.box, 1)]
    return []


def annotated_thumbnail(path: Path, ramki: Sequence[RamkaLica], max_dim: int,
                        side: int = THUMB) -> np.ndarray | None:
    """Миниатюра с рамками на перечисленных лицах. None — если файл не читается.

    Список рамок приходит извне (`ramki_stroki`), то есть из строки результата: превью,
    окно находок и окно эталона рисуют одни и те же рамки одними и теми же правилами.

    Функция без состояния и без кэша: каждый вызов сам читает файл. Кэш нужен `ResultsView`
    между пересчётами (см. `_kadr_dlya_kartochki`), а этому контракту — нет: миниатюра из
    теста и миниатюра на экране обязаны сходиться байт в байт. Читается тем же
    `_kadr_miniatury`, что и сетка, — иначе «байт в байт» было бы неправдой.

    Координаты рамок приходят в пикселях кадра `max_dim`, поэтому умножаются на
    `side / etalon` — тот самый коэффициент, который перевёл бы кадр-эталон в эту
    миниатюру, каким бы чтением миниатюра ни была получена (см. `_kadr_miniatury`).

    Тяжёлая работа здесь идёт в потоке вызывающего: функцию зовут и из теста, и из
    очереди миниатюр (`ResultsView._dostavit_kadr`), и с диска она берёт ровно один кадр.
    """
    para = _kadr_miniatury(path, max_dim, side)
    if para is None:
        return None
    kadr, k = para
    return ramki_na_kadr(kadr, ramki, k)


def _znak_procenta(kadr: np.ndarray, procent: int) -> np.ndarray:
    """Число прямо на снимке, в левом верхнем углу. Возвращает копию.

    Подпись под карточкой видно только тогда, когда ряд целиком влез в окно: на
    ноутбуке 13" полоса результатов начинается ниже сгиба, и за каждым числом человек
    крутил бы колесо мыши. А число — ровно тот признак, по которому решают «моё / не
    моё», поэтому он обязан быть на самой карточке, как дата на снимке в галерее
    телефона. Приём тот же, что в `face_picker.naklej_nomer`: тёмная плашка и белый
    текст читаются и на светлом, и на тёмном кадре.
    """
    risunok = kadr.copy()
    vysota, shirina = risunok.shape[:2]
    if vysota < 60 or shirina < 60:
        # Узкая полоска (вырожденный кадр, 1 px по короткой стороне): плашка съела бы
        # весь снимок, а число от этого только пропало. Число остаётся в подписи.
        return risunok
    sherift, masshtab = cv2.FONT_HERSHEY_SIMPLEX, max(0.45, min(vysota, shirina) / THUMB)
    tolshchina = max(1, int(round(masshtab * 2)))
    tekst = f"{procent}%"
    (shir_t, vys_t), _ = cv2.getTextSize(tekst, sherift, masshtab, tolshchina)
    otstup = 5
    cv2.rectangle(risunok, (0, 0), (otstup * 2 + shir_t, otstup * 2 + vys_t),
                  (0, 0, 0), cv2.FILLED)
    cv2.putText(risunok, tekst, (otstup, otstup + vys_t), sherift, masshtab,
                (255, 255, 255), tolshchina, cv2.LINE_AA)
    return risunok


class KartochkaResultatov(QWidget):
    """Одна карточка сетки: снимок, подпись, галочка — и двойной клик как просьба открыть.

    Двойной клик ловится здесь, а не фильтром событий на чужом виджете: фильтр
    переживает смерть карточки на один ход цикла событий, и в этом проекте это уже
    стоило сегфолта (задача T3). Подкласс отдаёт событие наружу сигналом и ничего не
    знает о том, кто его слушает.

    Одиночный клик намеренно не обработан: он не значит «покажи крупно», и открывать
    окно на каждое наведение курсора значило бы воевать с собственным интерфейсом.
    """

    dvojnyj_klik = Signal()

    def mouseDoubleClickEvent(self, sobytie) -> None:   # noqa: N802 (имя Qt)
        if sobytie.button() == Qt.MouseButton.LeftButton:
            self.dvojnyj_klik.emit()
            return
        super().mouseDoubleClickEvent(sobytie)


class ResultsView(QWidget):
    """Карточки с галочками: «похоже» сверху, «слабое сходство» — свёрнутой кучкой."""

    selection_changed = Signal(list)     # наружу — список видимых отмеченных путей
    prosyat_otkryt = Signal(object)      # Path: человек дважды кликнул по карточке

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Найденные фото")

        self._rows: list[ResultRow] = []
        self._boxes: list[tuple[Path, QCheckBox]] = []
        # Карточки по путям: двойной клик по любой из них обязан открыть именно это
        # фото, а не «то, что сейчас под рукой».
        self._kletki: dict[Path, "KartochkaResultatov"] = {}
        # Подписи «похоже на …» по путям: вторая строка карточки. Живёт отдельно от
        # галочек, потому что отвечает на другой вопрос — не «копировать ли», а «кто на
        # снимке».
        self._podpisi: dict[Path, QLabel] = {}
        # Имена строк эталона приходят СНАРУЖИ (см. `show_rows`): виджет не знает ни
        # что такое эталон, ни тем более `core.etalon`, и импорт туда за подписями
        # означал бы второй владелец выбора лиц в модуле, который только рисует.
        self._imena_ljudj: list[str] = []
        # Память галочек по пути файла: пересчёт и сворачивание кучки не имеют права
        # решать за человека, что он отмечал.
        self._otmetki: dict[Path, bool] = {}
        self._miniatury: dict[tuple[str, int], "tuple[np.ndarray, float] | None"] = {}
        self._threshold = 0.38
        self._max_dim = 2400
        self._weak_visible = False
        self._counts: tuple[int, int] = (0, 0)
        self._massovyj = False           # массовая отметка: сигнал один, а не на карточку
        self._pokazannye: list[tuple[ResultRow, bool]] = []
        self._kolonok_v_rjadu = COLUMNS
        # Потолок сетки и память о том, для какого набора находок он поднят. См.
        # `MAKSIMUM_KARTOTSEK` и `show_rows`.
        self._lid = MAKSIMUM_KARTOTSEK
        self._puti_pod_ekranom: frozenset[Path] = frozenset()
        self._za_potolkom = 0

        # ОЧЕРЕДЬ миниатюр: пути, кадр которых ещё не прочитан, в порядке показа. Карточка
        # на месте с первой же секунды, а снимки подтягиваются по одному за тик событий —
        # см. модульный docstring («первый экран не стоит»).
        self._ochered: deque[Path] = deque()
        self._kartinki: dict[Path, QLabel] = {}
        self._riad_po_puti: dict[Path, ResultRow] = {}
        self._sinhr_kadrov = SINHRONNO_KADROV
        self._taymer_kadrov = QTimer(self)
        self._taymer_kadrov.setInterval(OGRESH_KADRA_MS)
        self._taymer_kadrov.timeout.connect(self._dostavit_kadr)

        self.summary = QLabel(PUSTAYA_SETKA)
        self.summary.setWordWrap(True)

        self.hint = QLabel("")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color: gray")
        # Подпись стартует с того порога, который по умолчанию стоит на ползунке: сетка
        # до первого `show_rows` обязана уже объяснять число, а не молчать.
        self._obnovit_pojasnenie()

        self.toggle_weak = QPushButton(KUCHKA_PUSTO)
        self.toggle_weak.setCheckable(True)
        self.toggle_weak.setEnabled(False)
        self.toggle_weak.clicked.connect(self._na_kuchke)

        # Кнопка «показать ещё» — единственный виджет, которому разрешено поднять потолок.
        # Прячется она, как только скрывать стало нечего: живая кнопка «показать ещё
        # 0 фото» была бы обещанием без содержания.
        self.show_more_btn = QPushButton()
        self.show_more_btn.setToolTip(PODSKAZKA_ESHCHE)
        self.show_more_btn.clicked.connect(self._na_eshche)
        self.show_more_btn.setVisible(False)

        self.check_all_btn = QPushButton("Отметить все")
        self.check_all_btn.clicked.connect(self.check_all)
        self.uncheck_all_btn = QPushButton("Снять все")
        self.uncheck_all_btn.clicked.connect(self.uncheck_all)

        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(self._grid_host)

        row = QHBoxLayout()
        row.addWidget(self.toggle_weak)
        row.addWidget(self.show_more_btn)
        row.addWidget(self.check_all_btn)
        row.addWidget(self.uncheck_all_btn)
        row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(self.summary)
        layout.addWidget(self.hint)
        layout.addLayout(row)
        layout.addWidget(self._scroll, 1)

    # --- заполнение ------------------------------------------------------------------
    def show_rows(self, rows: Sequence[ResultRow], threshold: float,
                  max_dim: int = 2400, imena_ljudj: Sequence[str] = ()) -> None:
        """Показать строки результата. Разбора эта функция не запускает: лица к моменту
        вызова уже посчитаны и лежат в памяти. С диска берётся не больше одного кадра
        (см. `SINHRONNO_KADROV`), остальные дочитывает очередь по кадру за тик.

        `imena_ljudj` — как человек назвал искомых людей, номер в списке = номер человека
        в `ResultRow.chelovek`. Пришли списком и без импорта эталона: сетка рисует ответ
        чужого выбора, а не выбирает сама.

        Список помнится виджетом и передаётся всем вторичным показам (`_na_eshche`,
        `check_all`, `_na_kuchke`): внутренний вызов без имён уронил бы вторую строку со
        всех карточек после первого же нажатия, и человек прочитал бы это как «приложение
        разучилось подписывать».

        Потолок `MAKSIMUM_KARTOTSEK` снимается ТОЛЬКО нажатием «показать ещё» и держится
        ровно до тех пор, пока под сеткой тот же набор файлов. Пересчёт похожести и
        разворот кучки — это те же самые находки, и молча вернуть прежний потолок
        значило бы снова спрятать то, о чём человек уже попросил. Новый разбор папки
        даёт другие файлы — с них потолок начинается заново.
        """
        if max_dim != self._max_dim:
            # Миниатюры другого качества разбора — другие кадры; старые держать в памяти
            # смысла нет: к этому разрешению окно вернётся не раньше нового скана.
            self._miniatury.clear()
        self._max_dim = max_dim
        self._threshold = threshold
        self._imena_ljudj = list(imena_ljudj)
        # Подпись про число успевает за ползунком на каждом показе: окно пересобирает
        # сетку из `rebuild_scores` на каждое движение, а прежний текст с чужим числом
        # человек читал бы ровно до следующего касания настройки.
        self._obnovit_pojasnenie()
        self._rows = list(rows)
        puti = frozenset(r.path for r in self._rows)
        if puti != self._puti_pod_ekranom:
            self._puti_pod_ekranom = puti
            self._lid = MAKSIMUM_KARTOTSEK
        # Отметки прежних находок, которых в этом списке больше нет, вычищаем: иначе
        # «отмечено, но скрыто» считалось бы по архиву прошлого запуска.
        nynesie = {r.path for r in self._rows}
        self._otmetki = {p: z for p, z in self._otmetki.items() if p in nynesie}
        # Округление, а не голое сравнение с `threshold * 100`: на 0.56 это
        # 56.00000000000001, и карточка с числом «56%» уехала бы в слабое сходство,
        # а report.txt рядом напечатал бы её как «похоже». Правило берётся у
        # `matcher.percent_of` — того же владельца, что у числа на карточке и у делёжа
        # в `build_rows`; три руки с одним округлением однажды разъезжаются.
        porog = percent_of(threshold)
        silnye = [r for r in self._rows if r.percent >= porog]
        slabye = [r for r in self._rows if r.percent < porog]
        self._counts = (len(silnye), len(slabye))
        # Пара (строка, «похоже» ли она) — единственный источник галочки для карточки:
        # делёж считается ЗДЕСЬ один раз, и `_cell` уже не пересчитывает его второй раз
        # по другому основанию. Порядок — «похоже» сверху, слабое сходство под ним.
        poka = [(r, True) for r in silnye]
        if self._weak_visible:
            poka += [(r, False) for r in slabye]
        self._napolnit(poka)

    def _napolnit(self, poka_zhem: Sequence[tuple[ResultRow, bool]]) -> None:
        """Разложить карточки и проверить фактом, что они влезли.

        Формула `_kolonok` — прикидка по ширине; настоящее слово имеет сама раскладка:
        если после раскладки её минимальная ширина больше области прокрутки, ряд не влез
        бы, и Qt дал бы горизонтальную полосу прокрутки — карточки за краем человек
        читает как «нашлось меньше». Тогда колонок становится на одну меньше.

        Бюджет синхронного чтения (`_sinhr_kadrov`) ставится ЗДЕСЬ, а не в `_razlozhit`:
        второй и третий проход цикла — это та же самая отрисовка, и тратить на неё ещё
        один кадр было бы удвоением цены первого экрана.

        Потолок числа карточек (`MAKSIMUM_KARTOTSEK`) режется тоже ЗДЕСЬ: очередь
        декодировки наполняет `_razlozhit`, поэтому одно нажатие «показать ещё» не имеет
        права поставить в очередь больше кадров, чем влезло на экран.
        """
        # Потолок снимается здесь, а не в `_razlozhit`: цикл перекладки ниже зовёт
        # `_razlozhit` до четырёх раз, и обрезка внутри него резала бы список от
        # раскладки к раскладке. Обрезанный хвост — это ровно то число, которое
        # `_itogi` назовёт словами, и очередь декодировки закрывает только показанное.
        self._pokazannye = list(poka_zhem)          # весь намеренный показ, с хвостом
        self._za_potolkom = max(0, len(poka_zhem) - self._lid)
        na_ekrane = poka_zhem[:self._lid]
        self._sinhr_kadrov = SINHRONNO_KADROV
        kolonki = self._kolonok()
        while True:
            self._razlozhit(na_ekrane, kolonki)
            if kolonki <= 1:
                break
            ne_vlezlo = self._grid.minimumSize().width() - self._scroll.viewport().width()
            if ne_vlezlo <= 1:
                break
            kolonki -= 1
        self._itogi()
        self.selection_changed.emit(self.checked)
        self._zapatit_ochered()

    def _razlozhit(self, poka_zhem: Sequence[tuple[ResultRow, bool]],
                   kolonki: int) -> None:
        """Собрать сетку из ПРИСЛАННОГО списка. Обрезка по потолку — забота `_napolnit`.

        Здесь список укорачиваться не имеет права: `_napolnit` зовёт эту функцию
        по нескольку раз на одном и том же показе, пока ряды не влезут в ширину.
        """
        self._ochistit_setku()
        self._boxes = []
        self._kletki = {}
        self._kartinki = {}
        self._riad_po_puti = {}
        self._podpisi = {}
        self._ochered = deque()
        self._kolonok_v_rjadu = kolonki
        for indeks, para in enumerate(poka_zhem):
            self._grid.addWidget(self._cell(*para), indeks // kolonki, indeks % kolonki)

    def _na_eshche(self) -> None:
        """Поднять потолок ещё на `DOBAVLYAET_KARTOTSEK`. Единственный способ это сделать.

        Автоматически — по прокрутке, по таймеру, «по числу строк» — потолок не снимается
        никогда: иначе он перестал бы быть потолком, а цена виджетов никуда бы не делась.
        """
        self._lid += DOBAVLYAET_KARTOTSEK
        self.show_rows(self._rows, self._threshold, self._max_dim,
                       imena_ljudj=self._imena_ljudj)

    def _kolonok(self) -> int:
        """Сколько карточек в ряду — по фактической ширине окна.

        Четыре колонки по 220 px — это около 1000 px содержимого, и в окне обычной
        ширины Qt показывал из-за этого горизонтальную полосу прокрутки: часть карточек
        уезжала за край, а человек читал это как «нашлось меньше». Считаем по месту и
        держим потолок — на широком экране ряд из двадцати подписей не читается тоже.
        """
        na_kartochku = THUMB + 40                  # кадр + поля ячейки с подписью галочки
        shirina = self._scroll.viewport().width()
        if shirina <= 0:
            return COLUMNS                         # виджет ещё не разложен: старое число
        return max(1, min(MAKSIMUM_KOLONOK, shirina // na_kartochku))

    def resizeEvent(self, event) -> None:          # noqa: N802 (имя Qt)
        """Окно растянули или свернули — ряды перекладываем, пока карточки не поехали
        по горизонтали. Перекладка стоит только смены числа колонок, а не каждого пикселя:
        `show_rows` на каждое движение мыши перебирал бы все галочки заново."""
        super().resizeEvent(event)
        if self._pokazannye and self._kolonok() != self._kolonok_v_rjadu:
            self._napolnit(self._pokazannye)

    def _ochistit_setku(self) -> None:
        """Снять карточки. `hide()` здесь не косметика: виджет, убранный из раскладки,
        Qt не прячет — до `deleteLater()` он дорисовался бы призраком под новыми."""
        while self._grid.count():
            kletka = self._grid.takeAt(0).widget()
            if kletka is not None:
                kletka.hide()
                kletka.setParent(None)
                kletka.deleteLater()

    def _cell(self, row: ResultRow, pohozh: bool) -> QWidget:
        """Одна карточка: миниатюра с рамкой лица и галочка с процентом и именем файла.

        `pohozh` — ответ на вопрос «лежит ли эта строка в кучке „похоже“», и приходит он
        из `show_rows`, а не пересчитывается здесь. Второй счёт того же делёжа по второму
        основанию — это будущий конфликт: карточка в «слабом сходстве» с галочкой,
        поставленной по чужому числу.

        Кадр берётся из кэша; если его ещё нет, карточка выходит сразу с серой плиткой
        и числом, а снимок дописывается в очередь. Ждать чтения здесь нельзя: на
        десятке находок это секунды стоящего окна (см. модульный docstring).
        """
        kletka = KartochkaResultatov()
        kletka.dvojnyj_klik.connect(lambda put=row.path: self.prosyat_otkryt.emit(put))
        v = QVBoxLayout(kletka)

        kartinka = QLabel()
        kartinka.setFixedSize(THUMB, THUMB)
        kartinka.setAlignment(Qt.AlignmentFlag.AlignCenter)
        kartinka.setToolTip(f"{row.path}\nлиц на снимке: {row.faces}")
        self._kartinki[row.path] = kartinka
        self._riad_po_puti[row.path] = row
        if (str(row.path), self._max_dim) in self._miniatury or self._sinhr_kadrov > 0:
            self._sinhr_kadrov -= 1
            self._pokazat_kadr(row, kartinka)
        else:
            self._plitka_zaezd(kartinka, row)
            self._ochered.append(row.path)

        kto = _kto_i_skolko(row, self._imena_ljudj)
        check = QCheckBox(f"{row.percent}% — {row.path.name}")
        check.setToolTip(self._podskazka_galochki(row, kto))
        # Отметка — по делёжу ЭТОЙ сетки (`show_rows`), а не по флагу `matched` в строке:
        # флаг посчитан вызывающим, и галочка в кучке «слабое сходство» противоречила бы
        # подписи самой кучки. Пересчитывать тот же делёж вторично здесь значило бы иметь
        # два основания для одного решения, поэтому карточке приходит готовый ответ.
        bylo = self._otmetki.get(row.path, pohozh)
        check.setChecked(bylo)
        self._otmetki[row.path] = bylo
        # connect — ПОСЛЕ setChecked: иначе каждая карточка на сборке выстрелит наружу
        check.toggled.connect(lambda sostojanie, put=row.path:
                              self._na_galochke(put, sostojanie))

        v.addWidget(kartinka)
        v.addWidget(check)
        podpis = self._podpis_stroki(row, kto)
        if podpis is not None:
            v.addWidget(podpis)
        self._boxes.append((row.path, check))
        self._kletki[row.path] = kletka
        return kletka

    def _podpis_stroki(self, row: ResultRow, kto: str) -> QLabel | None:
        """Вторая строка карточки: кто из искомых людей в этом кадре и насколько.

        Пустая строка — если не нашлось никого (снимок в кучке слабого сходства), и тогда
        карточка остаётся с одним числом: врать именем человека она не имеет права.

        Подпись ставится отдельной меткой, а не дописывается в текст галочки: галочка
        остаётся единственной подписью, на которую человек нажимает, и втиснуть в неё
        второй смысл значило бы увеличить мишень и потерять её на узком экране.
        """
        if not kto:
            return None
        metka = QLabel(kto)
        metka.setWordWrap(True)
        metka.setStyleSheet(STIL_PODPISI_STROKI)
        metka.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._podpisi[row.path] = metka
        return metka

    def _podskazka_galochki(self, row: ResultRow, kto: str) -> str:
        """Подсказка под галочкой: число и, если кто-то найден, кто именно."""
        osnova = f"насколько фото похоже на выбранное лицо: {row.percent}%"
        return osnova if not kto else f"{osnova} · {kto}"

    # --- очередь миниатюр ------------------------------------------------------------
    def _plitka_zaezd(self, kartinka: QLabel, row: ResultRow) -> None:
        """Плитка «кадр ещё не дошёл»: карточка на месте, снимка ещё нет.

        Число рисуется и на плитке — тем же способом, что и на готовом кадре, поэтому
        карточка при приходе снимка не меняет ни размер, ни подпись: человек решает по
        числу и видит, куда смотреть, ещё до того, как дочитался пиксель. Никакого
        текста тут нет нарочно: «не читается» про этот файл сказать нельзя — его ещё
        не читали.
        """
        kadr = _znak_procenta(np.full((THUMB, THUMB, 3), PLITKA, dtype=np.uint8),
                              row.percent)
        kartinka.setPixmap(face_to_qpixmap(kadr))
        kartinka.setStyleSheet("")

    def _pokazat_kadr(self, row: ResultRow, kartinka: QLabel) -> None:
        """Нарисовать карточку по состоянию кэша: кадр, плитка или честная заглушка."""
        kadr = self._kadr_dlya_kartochki(row.path)
        if kadr is None:
            # Снимок остаётся в списке: он найден, его можно отметить, и он доедет до
            # отчёта. Пропать бесследно из сетки — значит соврать про число находок.
            kartinka.setText(ZAGLUISHKA)
            kartinka.setWordWrap(True)
            kartinka.setStyleSheet("color: gray; background: palette(mid);")
            return
        kadr, k = kadr
        kadr = _znak_procenta(kadr, row.percent)
        ramki = ramki_stroki(row)
        if ramki:
            # Рамки поверх плашки: лицо — то, по чему решают, и закрывать его числом
            # нельзя, даже если ребёнок сидит ровно в левом верхнем углу кадра.
            ramki_na_kadr(kadr, ramki, k)
        kartinka.setText("")
        kartinka.setStyleSheet("")
        kartinka.setPixmap(face_to_qpixmap(kadr))

    def _zapatit_ochered(self) -> None:
        """Пустить таймер, если есть что читать и если на это смотрят.

        Невидимому окну миниатюры не нужны: карточки, которых никто не видит (свёрнутое
        приложение, окно под другим), только выжигают диск и процессор. `showEvent`
        догонит очередь, как только сетку покажут.
        """
        if self._ochered and self.isVisible():
            if not self._taymer_kadrov.isActive():
                self._taymer_kadrov.start()
        else:
            self._taymer_kadrov.stop()

    def showEvent(self, event) -> None:      # noqa: N802 (имя Qt)
        """Окно показали — очередь, стоявшая до этого, догоняет с первого кадра."""
        super().showEvent(event)
        self._zapatit_ochered()

    def hideEvent(self, event) -> None:      # noqa: N802 (имя Qt)
        """Окно убрали — читать нечего: таймер глушится, а очередь остаётся на месте."""
        super().hideEvent(event)
        self._taymer_kadrov.stop()

    def _dostavit_kadr(self) -> None:
        """Один кадр за тик событий, сверху вниз — в том порядке, в каком человек читает.

        «Один» — не скромность, а единственная гарантия того, что между двумя
        декодировками окно успеет отрисоваться и ответить на клик: очередь двигается
        шагом таймера, а не одним блоком на весь экран.

        Путь берётся до чтения и сверяется после: за время очереди человека могли
        заинтересовать другие карточки (`show_rows`, разворот кучки, смена похожести), и
        рисовать кадр в карточку, которой уже нет, нельзя.
        """
        while self._ochered:
            put = self._ochered.popleft()
            kartinka = self._kartinki.get(put)
            row = self._riad_po_puti.get(put)
            if kartinka is None or row is None:
                continue                     # эту карточку уже переложили — кадр не нужен
            self._pokazat_kadr(row, kartinka)
            return
        self._taymer_kadrov.stop()

    def _kadr_dlya_kartochki(self, path: Path) -> "tuple[np.ndarray, float] | None":
        """СВЕЖАЯ КОПИЯ базового кадра из кэша и коэффициент для рамки.

        Нечитаемый файл кэшируется как None: за новым отказом на диск тоже нельзя ходить
        на каждый пересчёт.
        """
        kluch = (str(path), self._max_dim)
        if kluch not in self._miniatury:
            if len(self._miniatury) >= _MINIATUR_PREDEL:
                self._miniatury.pop(next(iter(self._miniatury)))
            self._miniatury[kluch] = _kadr_miniatury(path, self._max_dim, THUMB)
        para = self._miniatury[kluch]
        # копия обязательна: и `_znak_procenta`, и `_ramka` пишут в массив, а оригинал
        # лежит в кэше для следующих пересчётов и для других карточек этого же файла
        return (para[0].copy(), para[1]) if para is not None else None

    # --- подписи ------------------------------------------------------------------------
    def _obnovit_pojasnenie(self) -> None:
        """Строка под сеткой: та же цифра, которая стоит на ползунке похожести.

        Единственная точка, откуда подпись обновляется. Держать её одной нужно затем,
        чтобы число в тексте не разъехалось с числом на карточках: `show_rows` зовёт
        пересчёт кучек, `_na_eshche` и `check_all` возвращаются в него же, и подпись
        обязана успевать за каждым из этих путей.

        Чужое значение порога — молчаливый отказ: сетка не падает, а под ней остаётся
        прежняя строка вместо вранья про «от 0 %».
        """
        procent = _procent_pojasnenija(self._threshold)
        if procent is None:
            return
        self.hint.setText(podskaz_chisla(procent))

    def _itogi(self) -> None:
        """Что написано под сеткой. Числа — существительным вперёд («Похоже: 3»), как в
        отчёте: такая форма верна при любом числе, а «3 фото найдено» ломается на «1».

        Числа кучок считаются по ВСЕЙ находке, а не по показанному: «Похоже: 1200» и
        «Показано: 300 из 1200» — это два разных факта, и человек обязан видеть оба.
        Молчаливый ноль хуже ошибки ровно так же, как молчаливая цензура.
        """
        n_silnyh, n_slabyh = self._counts
        vidimye = {put for put, _ in self._boxes}
        n_otmecheno = len(self.checked)
        skrytye_s_galochkoj = [put for put, znak in self._otmetki.items()
                               if znak and put not in vidimye]
        tekst = (f"Похоже: {n_silnyh}. Слабое сходство: {n_slabyh}. "
                 f"Отмечено к копированию: {n_otmecheno}.")
        if self._za_potolkom:
            pokazaemoe = len(self._boxes) + self._za_potolkom
            tekst += POTOLOK_V_PODPISI.format(pokazano=len(self._boxes),
                                              vsego=pokazaemoe)
        if skrytye_s_galochkoj:
            # Прятать отметку и молчать нельзя: человек решил, а копирование не увидело.
            tekst += (f" Отмечено, но скрыто: {len(skrytye_s_galochkoj)} — "
                      "их не скопируют.")
        self.summary.setText(tekst)

        # Кнопка «показать ещё» живёт ровно столько, сколько есть что показывать: висящая
        # на пустом месте кнопка была бы вторым обещанием без содержания.
        if self._za_potolkom:
            dobavit = min(DOBAVLYAET_KARTOTSEK, self._za_potolkom)
            self.show_more_btn.setText(KNOPKA_ESHCHE.format(n=dobavit))
        self.show_more_btn.setVisible(bool(self._za_potolkom))

        if n_slabyh == 0:
            self.toggle_weak.setText(KUCHKA_PUSTO)
            self.toggle_weak.setEnabled(False)
            self.toggle_weak.setChecked(False)
            self._weak_visible = False
        else:
            self.toggle_weak.setEnabled(True)
            self.toggle_weak.setText(
                KUCHKA_VIDNO if self._weak_visible else
                f"ещё {n_slabyh} фото, похоже слабее — нажмите, чтобы посмотреть")

    # --- состояние -----------------------------------------------------------------------
    @property
    def counts(self) -> tuple[int, int]:
        """(похоже, слабое сходство) по последнему `show_rows` — независимо от того,
        развернута ли кучка: делёж от показа не зависит."""
        return self._counts

    @property
    def checked(self) -> list[Path]:
        """Отмеченные ВИДИМЫЕ карточки. Порядок — как в сетке, то есть по убыванию
        процента: он же уходит в `copy_photos` и в отчёт."""
        return [put for put, check in self._boxes if check.isChecked()]

    def itogi_prosmotra(self) -> list[ResultRow]:
        """Весь список находок в том порядке, в котором его показывает сетка.

        Нужен окну просмотра, чтобы «вперёд» означало «к менее похожему фото», а не
        «к случайному из словаря». Возвращается копия списка: окно не имеет права
        переставить находки и обнаружить потом, что карточки поехали.

        Порядок — полный, включая свёрнутую кучку слабого сходства и карточки под
        потолком сетки. Иначе стрелка упрётся в край видимой части и человек решит, что
        фото больше нет, хотя оно просто скрыто другой кнопкой.
        """
        return list(self._rows)

    def imya_cheloveka(self, row: ResultRow) -> str | None:
        """Кого из искомых людей нашла эта строка. None — никого или имён не передавали.

        Окно просмотра спрашивает сетку, а не заводит своё мнение: имена принадлежат
        главному окну и человек правит их, не закрывая просмотр.
        """
        imena = self._imena_ljudj
        if len(imena) < 2:
            return None                      # см. `_kto_i_skolko`: один человек — не ответ
        return imena[row.chelovek - 1] if 1 <= row.chelovek <= len(imena) else None

    def otmecheno(self, put: Path) -> bool:
        """Отмечено ли это фото. Единственный ответ на вопрос, у которого не должно быть
        двух источников: окно просмотра спрашивает сетку, а не заводит своё мнение."""
        return bool(self._otmetki.get(put, False))

    def otmetit_snaruji(self, put: Path, sostojanie: bool) -> None:
        """Отметить карточку не её собственной галочкой — так просит окно просмотра.

        Обновляются ОБА места: память отметок и видимая галочка. Оставить галочку
        прежней значило бы показать человеку на экране одно, а в список копирования
        положить другое.

        Тот же путь, что и щелчок по галочке, — через `setChecked`, а не прямая запись в
        `_otmetki`: иначе сигнал `toggled` не случится и `_na_galochke` не пересчитает
        подпись с числом отмеченных.
        """
        bylo = self.otmecheno(put)
        self._otmetki[put] = sostojanie
        for path, check in self._boxes:
            if path == put:
                if check.isChecked() != sostojanie:
                    check.setChecked(sostojanie)      # сам зовёт _na_galochke и шлёт сигнал
                return
        # Карточки на экране нет: она под потолком сетки или в свёрнутой кучке. Отметка
        # в памяти при этом честная, и сообщить наружу всё равно надо.
        if bylo != sostojanie:
            self.selection_changed.emit(self.checked)

    def check_all(self) -> None:
        """Отметить всё найденное, включая свёрнутую кучку.

        Кучка разворачивается первой: кнопка обещает «все», а «все», чего не видно,
        быть не может — снять лишнюю отметку с того, что не показано, человек не сможет.

        Потолок сетки (`MAKSIMUM_KARTOTSEK`) эта кнопка НЕ снимает: разметить две тысячи
        карточек — значит построить две тысячи виджетов, ровно ту цену, из-за которой
        потолок и заведён. Отмечен поэтому весь найденный список, а в `checked` уходят
        отмеченные ПОКАЗАННЫЕ: подпись под сеткой называет оба числа и прямо говорит про
        хвост — «отмечено, но скрыто», и их не скопируют, пока их не покажут.
        """
        if self._counts[1] and not self._weak_visible:
            self._weak_visible = True
            self.toggle_weak.setChecked(True)      # clicked не дёргается, перестройка одна
            self.show_rows(self._rows, self._threshold, self._max_dim,
                       imena_ljudj=self._imena_ljudj)
        self._set_vse(True)

    def uncheck_all(self) -> None:
        self._set_vse(False)

    def _set_vse(self, sostojanie: bool) -> None:
        self._massovyj = True
        try:
            for _, check in self._boxes:
                check.setChecked(sostojanie)       # сигнал глушим: он один на всю операцию
        finally:
            self._massovyj = False
        # «Снять все» снимает и спрятанное: кнопка обещает все, а не видимые.
        self._otmetki.update({r.path: sostojanie for r in self._rows})
        # Подпись пересчитывается здесь, а не ждёт следующего показа: массовая отметка
        # сетку не перекладывает, а без этого человек видел бы «Отмечено: 300» там, где
        # он только что отметил две тысячи, и про 1700 скрытых галочек она бы молчала.
        self._itogi()
        self.selection_changed.emit(self.checked)

    def _na_galochke(self, put: Path, sostojanie: bool) -> None:
        self._otmetki[put] = sostojanie
        if not self._massovyj:
            self.selection_changed.emit(self.checked)

    def _na_kuchke(self) -> None:
        self._weak_visible = self.toggle_weak.isChecked()
        self.show_rows(self._rows, self._threshold, self._max_dim,
                       imena_ljudj=self._imena_ljudj)
