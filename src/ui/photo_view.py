"""Окно просмотра одного фото: снимок целиком, лицо отдельно, решение о копировании.

Зачем оно нужно. Миниатюра в сетке — 220 px на весь снимок, и ребёнок на общем плане
занимает в ней десяток пикселей. Число «38 %» на такой карточке — не ответ, а вопрос:
«точно мой?». Ответ даёт только крупный кадр. Поэтому здесь два вида одного и того же
лица: рамка на всём снимке (видно, КТО ещё в кадре и куда смотреть) и квадрат лица в
320 px (видно, ЧТО это за ребёнок).

Окно не модальное. Человек сверяет фото друг с другом, и закрывать просмотр ради
следующего кадра — значит терять то, что только что увидел. Поэтому окно одно, и
`pokazat` подменяет в нём содержимое.

Галочка «копировать» принадлежит СЕТКЕ, а не окну. Единственный владелец отметки —
`ResultsView._otmetki`, и окно лишь показывает её состояние и просит перемену. Завести
здесь свою копию значило бы однажды получить «в отчёт ушло одно, на экране другое» —
тот самый разъезд, против которого в этом проекте уже написаны тесты.

Чтение снимка — единственная тяжёлая работа, и оно идёт в потоке интерфейса по явному
двойному клику: 0,4 с на 24-мегапиксельный кадр. Это одно действие по просьбе, а не
сотня файлов подряд, — но и его достаточно, чтобы окно выглядело мёртвым, поэтому
«читаем…» рисуется ДО чтения, а не после (см. `showEvent`).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QHBoxLayout, QLabel,
                               QPushButton, QVBoxLayout, QWidget)

from core.scanner import load_photo
from ui.face_picker import RamkaLica, face_to_qpixmap, krop_po_ramke, ramki_na_kadr

# Сторона крупного квадрата лица. Вдвое больше карточки выбора эталона (160 px): здесь
# задача не «указать на лицо», а «узнать ребёнка», и на 160 px узнавание не гарантировано.
STORONA_KROP = 320

CHITAEM = "читаем снимок…"
NET_LICA = "лицо на этом фото не найдено — смотрите снимок целиком"
PUSTO = "двойной клик по фото в списке откроет его здесь"

# Подпись над кадром: где мы в списке и что нашёл поиск. Процент обязан быть на самой
# форме, а не только на карточке под миниатюрой: человек смотрит сюда, когда решает.
# Позиция «N из M» — чтобы стрелки не были прыжком в неизвестность.
FORMA_PODPISI = "{nomer} из {vsego}{po_lice} · похоже на {procent} %"

# «найдено: …» — КТО из искомых людей дал это число (задача T20). Одно «похоже на 61 %»
# не различает «нашли ребёнка» и «нашли маму, которую отметили вторым примером», а
# решение «класть это фото в папку или нет» принимается именно по этому.
NAJDEN = " · найден: {imya}"

# Доля экрана, которую занимает окно. Больше — человек упрётся в масштабирование,
# меньше — снова не разглядит.
DOLJA_EKRANA = 0.8


def kad_dlya_prosmotra(put: Path, ramki: Sequence[RamkaLica], lice: int | None,
                       max_dim: int) -> tuple[np.ndarray | None, np.ndarray | None,
                                              str | None]:
    """Снимок со ВСЕМИ лицами в рамках + крупный квадрат выбранного. Третий элемент —
    причина отказа.

    Рамки на всех лицах, а не на одном, нужны окну выбора эталона: там человек решает
    «какое из трёх лиц — мой ребёнок», и одна рамка на этот вопрос не отвечает.

    Кадр читается ровно тем же путём, что и для моделей (`load_photo(put, max_dim)`), и
    это не совпадение: координаты рамок пришли в пикселях именно этого кадра, так что
    масштаб равен единице. Любое другое чтение (быстрое, через `Image.draft`) сдвинуло бы
    рамки с лиц, а для окна, которое существует ради «разглядеть», это худший из
    возможных дефектов.

    `lice=None` или номера, которого нет в `ramki`, — квадрат не вырезается: между кликом
    и пересчётом список лиц мог перестроиться. Снимок с рамками при этом показывается:
    «нет кропа» не значит «нет фото».

    Нечитаемый файл отвечает причиной с именем файла внутри: «фото не читается» без пути
    человек не проверит, а «нет лица» и «нет файла» — это два разных его действия.
    """
    kadr = load_photo(put, max_dim=max_dim)
    if kadr is None:
        return None, None, f"{put} — снимок не читается"
    s_ramkami = kadr
    if ramki:
        s_ramkami = ramki_na_kadr(kadr.copy(), ramki)
    vybrannoe = next((r for r in ramki if r.nomer == lice), None)
    krop = krop_po_ramke(kadr, vybrannoe.box, STORONA_KROP) if vybrannoe else None
    return s_ramkami, krop, None


def nadpisi_bez_lica() -> str:
    """Что окно говорит о снимке, где лицо не нашлось."""
    return NET_LICA


class _FotoPole(QLabel):
    """Метка снимка, которая сама говорит, когда изменился её размер.

    Без этого сигнала снимок остаётся крупнее отпущенного места и обрезается: при
    первой раскладке Qt отдаёт метке размер ПОСЛЕ того, как окно уже вписало в неё кадр
    по прежним 292 px, и пере-вписывать некому. Проверено тестом: pixmap 618 px в метке
    292 px.
    """

    razmer_izmenilsja = Signal()

    def resizeEvent(self, sobytie) -> None:      # noqa: N802 (имя Qt)
        super().resizeEvent(sobytie)
        self.razmer_izmenilsja.emit()


class PhotoViewDialog(QDialog):
    """Один кадр крупно: фото целиком, лицо отдельно, галочка копирования."""

    razresheno = Signal(object, bool)     # Path и решение: копировать это фото или нет
    prosyat_perehod = Signal(int)         # -1 назад по списку, +1 вперёд

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Фото целиком")
        self._put: Path | None = None
        self._ramki: Sequence[RamkaLica] = ()
        self._lice: int | None = None
        self._max_dim: int = 2400
        self._podpis = ""
        self._ishodnyj: QPixmap | None = None
        # Признак «галочку меняем мы сами» нужен, чтобы программная установка состояния
        # не выслала наружу ложное «человек решил». В отличие от аналогичного признака в
        # окне эталона, живёт он строго внутри одного вызова: ставится и снимается здесь
        # же, поэтому пережить первый клик человека не может по построению.
        self._menyaem_sami = False

        self.slovo = QLabel(PUSTO)
        self.slovo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.slovo.setStyleSheet("color: gray")

        self.foto = _FotoPole()
        self.foto.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.foto.setMinimumSize(200, 150)
        self.foto.razmer_izmenilsja.connect(self._vpisat)

        self.krop = QLabel()
        self.krop.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.krop.setFixedWidth(STORONA_KROP)
        self.krop.hide()

        sfera = QHBoxLayout()
        sfera.addWidget(self.foto, 1)
        sfera.addWidget(self.krop)

        self.copy_check = QCheckBox("копировать это фото")
        self.copy_check.setToolTip(
            "то же самое, что галочка под карточкой в списке: отметка живёт в списке, "
            "а окно её только показывает")
        self.copy_check.toggled.connect(self._na_galochke)

        # Стрелки листают список находок, не закрывая окно: сравнивать кадры, каждый
        # раз теряя предыдущий, нельзя.
        self.prev_button = QPushButton("← предыдущее фото")
        self.prev_button.setToolTip("показать фото, которое стоит в списке левее этого")
        self.prev_button.clicked.connect(lambda: self.prosyat_perehod.emit(-1))

        self.next_button = QPushButton("следующее фото →")
        self.next_button.setToolTip("показать фото, которое стоит в списке правее этого")
        self.next_button.clicked.connect(lambda: self.prosyat_perehod.emit(+1))

        self.close_button = QPushButton("Закрыть")
        self.close_button.clicked.connect(self.close)

        niz = QHBoxLayout()
        niz.addWidget(self.copy_check)
        niz.addStretch(1)
        niz.addWidget(self.prev_button)
        niz.addWidget(self.next_button)
        niz.addWidget(self.close_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.slovo)
        layout.addLayout(sfera, 1)
        layout.addLayout(niz)
        self._razmerit_po_ekranu()

    # --------------------------------------------------------------- содержимое
    def pokazat(self, put: Path, ramki: Sequence[RamkaLica], lice: int | None,
                max_dim: int, procent: int, poziciya: int, vsego: int,
                kopiruetsya: bool | None = None,
                imya_cheloveka: str | None = None,
                vsego_lic: int | None = None) -> None:
        """Показать это фото. Окно можно вызывать повторно — оно одно на все просмотры.

        `nomer`/`vsego` приходят извне, а не считаются здесь: порядок списка находок
        принадлежит сетке, и своя копия этого порядка разъехалась бы с тем, что человек
        видит под окном.

        `imya_cheloveka` — как человек назвал того, кого нашёл на этом снимке (`None` —
        подписывать нечего: на кадре не угадал никто). Окно получает слово и не знает ни
        что такое эталон, ни сколько в нём людей: эти ответы принадлежат сетке и главному
        окну.

        `vsego_lic` — сколько лиц нашёл разбор в этом кадре. Это НЕ длина `ramki`: с
        задачи T19 рамкой помечены только совпадения, и на групповом фото их две из
        двадцати четырёх. Без числа окно считало бы «из» по рамкам и сказало бы
        «лицо 22 из 2» — то есть ровно то, чего человек не спрашивал.
        """
        self._put = put
        self._ramki = ramki
        self._lice = lice
        self._max_dim = max_dim
        self.setWindowTitle(put.name)
        po_lice = NAJDEN.format(imya=imya_cheloveka) if imya_cheloveka else ""
        self._podpis = FORMA_PODPISI.format(nomer=poziciya, vsego=vsego, po_lice=po_lice,
                                            procent=procent)
        if lice:
            v_kadre = vsego_lic if vsego_lic is not None else len(ramki)
            self._podpis = f"лицо {lice} из {v_kadre} · " + self._podpis
        self.prev_button.setEnabled(poziciya > 1)
        self.next_button.setEnabled(poziciya < vsego)
        # Галочка копирования принадлежит списку находок. В окне выбора эталона
        # копировать нечего: там человек решает, кого искать, и живая галочка
        # обещала бы действие, которого окно не делает.
        self.copy_check.setVisible(kopiruetsya is not None)
        if kopiruetsya is not None:
            self._postavit_galochku(kopiruetsya)
        self._ishodnyj = None
        self.krop.hide()
        self.foto.clear()
        self.slovo.setText(f"{self._podpis} · {CHITAEM}")
        self._v_pered()
        # Синхронная отрисовка ДО чтения. Чтение кадра — блокирующий вызов на ~0,4 с, и
        # без этого шага человек увидел бы окно, не успевшее нарисовать даже собственную
        # надпись «читаем…», — то есть ровно «вид зависшего окна».
        #
        # `repaint`, а не `processEvents`: второй переочередывает события изнутри
        # собственное showing-событие, и на этой сборке это стоило сегфолта в первом
        # же тесте. `repaint` перерисовывает сразу и никуда не заходит.
        self.repaint()
        self._nachitat()

    def _v_pered(self) -> None:
        """Поднять окно на перед, даже если оно уже открыто и стоит за другим.

        `show()` на видимом окне не делает ничего, а менять содержимое окна, которое
        человек не видит, — значит оставить его с пустым экраном после двойного клика.
        `activateWindow()` поднимает окно на macOS, `raise_()` нужен Windows, где
        активация не всегда вытаскивает окно из-за соседних.
        """
        if not self.isVisible():
            self.show()
        self.raise_()
        self.activateWindow()

    def sinkhronizirat(self, put: Path, otmecheno: bool) -> None:
        """Сетка изменила отметку — окно обязано показать то же самое.

        Иначе человек снимет галочку в списке, а открытое окно будет утверждать
        обратное, и следующее «копировать это фото» вернёт то, от чего он только что
        отказался.
        """
        if put == self._put:
            self._postavit_galochku(otmecheno)

    @property
    def pokazyvaemoe_lice(self) -> int | None:
        """Номер лица, чей квадрат сейчас показан. None — квадрата нет."""
        return self._lice

    @property
    def pokazyvaet(self) -> Path | None:
        """Какой файл сейчас в окне. None — окно ещё ничего не показывало."""
        return self._put

    def keyPressEvent(self, sobytie) -> None:         # noqa: N802 (имя Qt)
        """Клавиши «стрелка влево/вправо» — тот же жест, что кнопки.

        Листать сотню находок мышью к кнопкам утомительно, а рука и так на клавиатуре
        после двойного клика. Край списка обрабатывается так же, как край кнопок:
        `prosyat_perehod` уходит наружу, а решение «есть ли туда ход» принимает сетка —
        здесь можно и промазать, а два источника правды про границы разъедутся.
        """
        if sobytie.key() == Qt.Key.Key_Left:
            self.prosyat_perehod.emit(-1)
            return
        if sobytie.key() == Qt.Key.Key_Right:
            self.prosyat_perehod.emit(+1)
            return
        super().keyPressEvent(sobytie)

    # --------------------------------------------------------------- служебное
    def _nachitat(self) -> None:
        if self._put is None:
            return
        photo, krop, beda = kad_dlya_prosmotra(self._put, self._ramki, self._lice,
                                                     self._max_dim)
        if beda is not None:
            self.slovo.setText(beda)
            self.foto.clear()
            self.krop.hide()
            return
        self._ishodnyj = face_to_qpixmap(photo)
        self._vpisat()
        if krop is None:
            self.slovo.setText(f"{self._podpis} · {NET_LICA}")
            self.krop.hide()
        else:
            self.slovo.setText(self._podpis)
            self.krop.setPixmap(face_to_qpixmap(krop))
            self.krop.show()

    def _vpisat(self) -> None:
        """Вписать снимок в текущую ширину и высоту, сохранив пропорции.

        `KeepAspectRatio` здесь не косметика: обрезанное фото для вопроса «мой ребёнок
        или нет» бесполезно, а растянутое без пропорций узнавание ломает.
        """
        if self._ishodnyj is None or self._ishodnyj.isNull():
            return
        self.foto.setPixmap(self._ishodnyj.scaled(
            self.foto.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, sobytie) -> None:    # noqa: N802 (имя Qt)
        super().resizeEvent(sobytie)
        self._vpisat()

    def _postavit_galochku(self, znachenie: bool) -> None:
        self._menyaem_sami = True
        try:
            self.copy_check.setChecked(znachenie)
        finally:
            self._menyaem_sami = False

    def _na_galochke(self, znachenie: bool) -> None:
        if self._menyaem_sami or self._put is None:
            return
        self.razresheno.emit(self._put, znachenie)

    def _razmerit_po_ekranu(self) -> None:
        ekran = QApplication.primaryScreen()
        if ekran is None:
            self.resize(1100, 800)             # тестовый offscreen-запуск без экрана
            return
        dostupnyj = ekran.availableGeometry()
        self.resize(int(dostupnyj.width() * DOLJA_EKRANA),
                    int(dostupnyj.height() * DOLJA_EKRANA))
