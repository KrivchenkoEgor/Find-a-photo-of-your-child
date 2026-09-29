"""Полоса настроек: слайдер похожести, режим поиска, качество разбора.

Тест проверяет три вещи, на которых уже не один проект спотыкался:
1) человек видит понятные слова, а не «порог» и «детектор» (прямой запрет спецификации);
2) сигнал `changed` не выстреливает, пока окно ещё достраивается (задача 15 подключается
   к нему позже, так что выстрел на сборке уходит в никуда);
3) значения из файла настроек, которых полоса не предлагает списком, не роняют приложение.

Ни реальных фото, ни моделей: виджет собирается из воздуха и ничего не читает.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:                        # подсказки для редактора, импорта в рантайме нет
    from PySide6.QtWidgets import QWidget
    from ui.settings import SettingsBar

pytest.importorskip("PySide6")


@pytest.fixture
def bar(qapp: object) -> "SettingsBar":
    from ui.settings import SettingsBar

    return SettingsBar()


# --- контракт из брифа ------------------------------------------------------------


def test_polzunok_daet_desyatichnye_i_shlet_bez_poroga(bar) -> None:
    assert bar.threshold == pytest.approx(0.38)
    assert bar.engine == "both"
    assert bar.max_dim == 2400
    assert "порог" not in bar.threshold_label.text().lower()


def test_diapazon_polzunka_030_060(bar) -> None:
    bar.set_threshold(0.30)
    assert bar.threshold == pytest.approx(0.30)
    bar.set_threshold(0.60)
    assert bar.threshold == pytest.approx(0.60)
    bar.set_threshold(0.99)                 # за пределы не выпускаем
    assert bar.threshold == pytest.approx(0.60)
    bar.set_threshold(-5.0)
    assert bar.threshold == pytest.approx(0.30)


def test_signal_changed_idet_pri_izmenenii(bar) -> None:
    got = []
    bar.changed.connect(lambda t, e, d: got.append((round(t, 2), e, d)))
    bar.set_threshold(0.45)
    bar.set_engine("yunet")
    bar.set_max_dim(1200)
    assert got[-1] == (0.45, "yunet", 1200)


def test_rezhimy_nazyvajutsja_prosto(bar) -> None:
    labels = {bar.engine_label(i) for i in range(3)}
    assert labels == {"Быстрее", "Точнее", "Оба"}


def test_knopka_chistki_oglavlenija_daeet_signal(bar) -> None:
    got = []
    bar.clear_cache.connect(lambda: got.append(1))
    bar.clear_button.click()
    assert got == [1]


# --- целое внутри, дробь снаружи ---------------------------------------------------


def test_polzunok_celyj_vnutri_i_desyatichnyj_snaruzi(bar) -> None:
    """Внутри ползунок — целые 30..60: у Qt нет дробных значений. Снаружи — обычный float,
    который ждёт `matcher`.

    Округлять на этом переходе надо `round`, а не `int`: у `int(0.57 * 100)` получается 56,
    и сохранённая настройка на следующем запуске уезжает на процент влево.
    """
    polzunok = bar._slider
    assert (polzunok.minimum(), polzunok.maximum()) == (30, 60)
    assert polzunok.singleStep() == 1
    for procent in range(30, 61):
        bar.set_threshold(procent / 100.0)
        assert polzunok.value() == procent, procent
        assert bar.threshold == pytest.approx(procent / 100.0), procent
        assert bar.value_label.text() == f"{procent}%", procent


def test_podpis_u_polsunka_pokazyvaet_tot_zhe_procent(bar) -> None:
    """Число справа от ползунка обязано совпадать с `threshold`: человек судит о настройке
    по тому, что видит на экране, а отчёт печатает ровно это число."""
    bar.set_threshold(0.52)
    assert bar.value_label.text() == "52%"
    assert round(bar.threshold * 100) == 52


# --- раскладка: подпись ползунка обязана читаться целиком ---------------------------
#
# Живой скриншот окна: метка показалась как «Насколько фото должно бы». Дело не в
# шрифте и не в размере окна: ВСЕ настройки стояли в одном `QHBoxLayout`, и при
# нехватке ширины Qt размазывает остаток по всем виджетам строки сразу — досталось и
# метке, у которой `minimumSizeHint` ровно равен её `sizeHint`. Отсечка в начале строки
# режет и саму подпись. Поэтому здесь две проверки, и обе ловят именно эту регрессию.
#
# 1. геометрическая — полосу РЕАЛЬНО сжало ниже её собственной нужной ширины (как в
#    живом окне, которое тянут влево), а метка всё равно не уже своего `sizeHint`;
# 2. структурная — рядом с меткой в строке не осталось ни списков, ни кнопок: на любом
#    шрифте и при любой локализации конкурентов за пиксели у них больше нет.
#
# Геометрическая проверка без сжатия в контейнере бессмысленна: Qt отдаёт виджету его
# minimumSizeHint, пока есть чем поделиться, — на полосе, разложенной в собственной
# минимальной ширине, тест зелёный и до правки.


def _stroke_vidzheta(bar, vidzhet: "QWidget"):
    """Строка (`QHBoxLayout`) раскладки, в которой лежит виджет, или None.

    Ищем по объекту виджета, а не по тексту: подпись правят, а контракт раскладки — нет.
    """
    stolbec = bar.layout()
    for i in range(stolbec.count()):
        element = stolbec.itemAt(i)
        stroke = element.layout() if element is not None else None
        if stroke is None:                     # виджет лежит в столбце напрямую
            continue
        for j in range(stroke.count()):
            v = stroke.itemAt(j).widget()
            if v is vidzhet:
                return stroke
    return None


def _vidzhety_stroki(stroke) -> list:
    return [stroke.itemAt(i).widget() for i in range(stroke.count())
            if stroke.itemAt(i).widget() is not None]


def test_metka_i_polzunok_stojat_v_stroke_bez_knopok_i_spiskov(bar) -> None:
    """Метка похожести делит строку только с ползунком и его числом.

    Списки способов поиска и кнопка очистки стоят в другой строке: именно они съедали
    ширину, из-за которой метку и обрезало. Тест структурный, потому что он не зависит
    ни от шрифта, ни от платформы — его нельзя «починить» растягиванием окна.
    """
    from PySide6.QtWidgets import QComboBox, QPushButton

    stroke = _stroke_vidzheta(bar, bar.threshold_label)
    assert stroke is not None, "метка не в строке — проверить нечего"
    lezhat = _vidzhety_stroki(stroke)
    assert bar._slider in lezhat and bar.value_label in lezhat, \
        "метка уехала от ползунка: подпись и число человек сопоставляет глазами"
    konkurenti = [type(w).__name__ for w in lezhat if isinstance(w, (QComboBox, QPushButton))]
    assert konkurenti == [], f"в строке метки теснятся виджеты: {konkurenti}"

    # а настройки-конкуренты живут отдельной строкой
    for korobka in (bar._engine_box, bar._dim_box, bar.clear_button):
        chuzhaja = _stroke_vidzheta(bar, korobka)
        assert chuzhaja is not None, "у способа поиска нет своей строки"
        assert chuzhaja is not stroke, f"{type(korobka).__name__} вернулся в строку метки"


def test_metka_pohozhesti_ne_rezhetsja_v_stesnennom_okne(qapp) -> None:
    """Полосу сжали за пределы её собственных планов — слово «похоже» обязано остаться.

    Проверять «метку не обрезали» на полосе, разложенной в удобной ей ширине, бесполезно:
    Qt не отдаст виджету меньше `minimumSizeHint`, пока есть чем поделиться, и тест был
    бы зелёным при сломанной однострочной раскладке. Обрезка рождается ровно там, где
    строке НЕ ХВАТАЕТ ширины: окно тянут влево, дефицит делится на всех, и подписи
    достаётся остаток. Значит тест обязан начать с того, что дефицит создать, — для
    этого полосу кладут в контейнер заведомо уже её `sizeHint`.
    """
    from PySide6.QtWidgets import QVBoxLayout, QWidget
    from ui.settings import SettingsBar

    DEFICIT = 200          # px: на столько полосе «не хватает» её же желаемой ширины
    bar = SettingsBar()
    kontejner = QWidget()
    QVBoxLayout(kontejner).addWidget(bar)
    kontejner.show()
    qapp.processEvents()
    kontejner.setFixedWidth(bar.sizeHint().width() - DEFICIT)
    qapp.processEvents()

    metka = bar.threshold_label
    # сам факт сжатия: без него проверка ниже могла бы ничего не мерить
    assert bar.width() < bar.sizeHint().width(), \
        f"полосу не сжало ({bar.width()} из {bar.sizeHint().width()}) — тест бесполезен"
    assert bar.width() < bar.minimumSizeHint().width(), \
        f"дефицита ширины нет ({bar.width()} >= {bar.minimumSizeHint().width()}), обрезку не ловим"
    assert metka.width() >= metka.sizeHint().width(), (
        f"метку обрезало до {metka.width()} px при нужных {metka.sizeHint().width()} — "
        "человек видит огрызок вместо настройки")
    kontejner.close()


# --- во время сборки сигнала быть не может ------------------------------------------


def test_pri_sobranii_vidzheta_signal_ne_izverenii(qapp) -> None:
    """Задача 15 собирает окно в таком порядке: `SettingsBar()`, потом `set_engine(...)`,
    `set_max_dim(...)`, `set_threshold(...)` — это возврат сохранённых значений, — и только
    затем `changed.connect(...)`.

    Порядок держится на четырёх строках внутри `__init__`, и проверять его надо двумя
    способами: по отдельности каждый даёт ложное «всё хорошо».

    1. перехват `_izvestit` — единственной точки выхода `changed`. Замечает саму попытку
       уведомить, даже когда виджет не достроен: на полусборке `self.engine` падает на
       `_engine_box`, PySide6 проглатывает исключение из слота, и до `emit` дело не доходит;
    2. шпион на сигнале класса. Замечает уже сам `emit` — если `changed` дёрнут напрямую,
       когда все атрибуты уже на месте.

    Оба детектора пристёгнуты к конкретным именам, и в этом их слепое пятно: рефакторинг,
    который переименовывает `_izvestit`, отцепляет перехват молча. Тест при этом остаётся
    зелёным над сломанным порядком строк — первый детектор не вызывается вообще, а второй
    ничего не видит, потому что исключение из слота Qt глотает и до `emit` не доходит.
    Закрываем пятно с двух сторон: наличие этих методов проверяется до сборки виджета,
    а шпион ставится ещё и на три обработчика — вызов обработчика виден всегда, и
    угадывать имя метода, который делает `emit`, для этого не нужно.
    """
    from ui.settings import SettingsBar

    # Имена, к которым пристёгнуты перехватчики. Переименовали метод — правьте и здесь;
    # ослепнуть без ведома человека тесту не даст именно эта проверка.
    for imja in ("_izvestit", "_na_polsunke", "_na_rezhime", "_na_kachestve"):
        assert imja in vars(SettingsBar), \
            f"в SettingsBar нет метода {imja}: перехватчик отцеплён, тест ослеп"

    popytki: list = []
    zapis: list = []

    class Sledopyt(SettingsBar):
        """Ловим уведомление до того, как оно дотронется до недостающих атрибутов.

        Три обработчика плюс `_izvestit`: `setValue`/`setCurrentIndex` летят в свой
        обработчик, так что вызов любого из них на полусборке и есть та самая ошибка
        порядка — независимо от того, как внутри называется метод финального `emit`.
        """

        def _na_polsunke(self, _procent: int) -> None:
            popytki.append("polsunok")

        def _na_rezhime(self, _indeks: int) -> None:
            popytki.append("rezhim")

        def _na_kachestve(self, _indeks: int) -> None:
            popytki.append("kachestvo")

        def _izvestit(self) -> None:      # перехват вместо реального emit
            popytki.append("izvestit")

    class Shpion:
        """Тот же интерфейс, что у сигнала, только `emit` пишет в список, а не рассылает."""

        def __init__(self, signal: object) -> None:
            self._signal = signal

        def emit(self, *args: object) -> None:
            zapis.append(args)

        def connect(self, *args: object, **kwargs: object) -> object:
            return self._signal.connect(*args, **kwargs)

        def disconnect(self, *args: object, **kwargs: object) -> object:
            return self._signal.disconnect(*args, **kwargs)

    nastojashhij = SettingsBar.changed
    SettingsBar.changed = Shpion(nastojashhij)
    try:
        Sledopyt()
    finally:
        SettingsBar.changed = nastojashhij

    assert popytki == [], "настройка изменилась до конца сборки виджета"
    assert zapis == [], f"changed выстрелил в конструкторе: {zapis}"

    # после сборки сигнал живой: сеттер обычного экземпляра доносит его наружу
    zhivyie = SettingsBar()
    polucheno: list = []
    zhivyie.changed.connect(lambda *a: polucheno.append(a))
    zhivyie.set_threshold(0.45)
    assert polucheno == [(0.45, "both", 2400)]


def test_povtornaja_ustanovka_ne_povtornyaet_signal(bar) -> None:
    """`set_threshold(0.45)` уже на установленном 0.45 не должен порождать второй пересчёт:
    задача 15 откатывает назад через эти же сеттеры, если человек отказался от разбора."""
    bar.set_threshold(0.45)
    got = []
    bar.changed.connect(lambda *a: got.append(a))
    bar.set_threshold(0.45)
    bar.set_engine("both")
    bar.set_max_dim(2400)
    assert got == []


# --- чужие значения из файла настроек -----------------------------------------------


def test_chuzhoj_max_dim_ne_ronaet_polosu(bar) -> None:
    """`Settings.max_dim` пропускает любое число от 600 до 6000, а полоса предлагает два
    варианта. Значение 1800 из подправленного руками файла не должно ронять окно на старте:
    ставим ближайший предложенный вариант и молчим, если он уже стоит."""
    got = []
    bar.changed.connect(lambda *a: got.append(a))
    bar.set_max_dim(1800)
    assert bar.max_dim == 2400               # ближайший из предложенных
    bar.set_max_dim(1900)
    assert got == [], "значение уже стояло — пересчёта быть не должно"
    bar.set_max_dim(1300)
    assert bar.max_dim == 1200
    bar.set_max_dim(600)
    assert bar.max_dim == 1200
    assert len(got) == 1 and got[0][1:] == ("both", 1200)   # ровно один пересчёт
    bar.set_max_dim(5000)
    assert bar.max_dim == 2400
    assert len(got) == 2


def test_neizvestnyj_rezhim_ne_ronaet_polosu(bar) -> None:
    """Ключ режима тоже приезжает из файла. Неизвестная строка — возврат к «Оба», а не
    StopIteration в сборке окна."""
    bar.set_engine("yunet")
    assert bar.engine == "yunet"
    bar.set_engine("расширенный")
    assert bar.engine == "both"
    assert bar.current_engine_label() == "Оба"


def test_vsem_kljucham_iz_config_est_podpisi(bar) -> None:
    """Каждому ключу движка из `config.ENGINE_KEYS` соответствует своя подпись: задача 15
    печатает `current_engine_label()` в отчёте, и подпись обязана быть человекочитаемой."""
    from utils.config import ENGINE_KEYS

    for indeks, kluch in enumerate(ENGINE_KEYS):
        bar.set_engine(kluch)
        assert bar.engine == kluch
        assert bar.current_engine_label() == bar.engine_label(indeks)
        assert bar.engine_label(indeks) in {"Быстрее", "Точнее", "Оба"}


# --- слова, которых в интерфейсе быть не может ---------------------------------------

# Стем, а не слово целиком: ловим и «порог», и «порога», и «детектора».
STEM = ("порог", "уверенност", "embedd", "косинус", "детектор")
# «пол» — отдельной строкой и только как целое слово: как подстрока он сидит в «ползунке».
CELYE_SLOVA = (r"\bпол(а|у|ом|е)?\b", r"\bвозраст(а|у|е|ом)?\b")


def vse_teksty(vidzhet: QWidget) -> list[str]:
    """Все тексты виджета, которые человек может увидеть: надписи, кнопки, пункты списков,
    всплывающие подсказки и доступные имена."""
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
                # подписанные тексты и подсказки отдельных строк: у QComboBox они живут
                # в ItemDataRole, а не в методе itemToolTip (его нет)
                for rol in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
                    znachenie = docha.itemData(i, rol)
                    if isinstance(znachenie, str):
                        teksty.append(znachenie)
    return [t for t in teksty if t]


def test_v_tekstah_vidzheta_net_zapreshhennyh_slov(bar) -> None:
    """Спецификация: в интерфейсе нет слов «порог», «уверенность», «embedding», «косинусное
    расстояние», «детектор», возраст, пол.

    Проверка обходит ВСЕ тексты виджета, включая подсказки, а не пару подписей, которые
    автор теста счёл главными: запрет на слово стоит в одном месте, а появиться оно может
    в любом новом подписанном элементе.
    """
    teksty = vse_teksty(bar)
    assert len(teksty) > 5, f"нечего проверять, нашли только {teksty}"
    for tekst in teksty:
        niz = tekst.lower()
        for stem in STEM:
            assert stem not in niz, f"«{stem}» в тексте: {tekst!r}"
        for vzorec in CELYE_SLOVA:
            assert not re.search(vzorec, niz), f"запрещённое слово в тексте: {tekst!r}"


def test_v_podpasyah_polosy_net_slovo_oglavlenie(bar) -> None:
    """Слово «оглавление» из интерфейса убрано — оно не читается как «кэш».

    Пользователь, тестирующий приложение руками, спросил, где кнопка удаления кэша, —
    и не нашёл её, хотя на экране она стояла и называлась «Очистить оглавление». Это не
    невнимательность: «оглавление» в русском — про книгу, и перенос на «список уже
    разобранных фото» делает человек только после объяснения. Кнопка, которую нельзя
    найти глазами, для ручного прогона бесполезна, а ручной прогон — единственный способ
    проверить, что поиск находит нужного ребёнка.

    Тест держит свойство, а не конкретную надпись: переименовывать кнопку ещё раз
    можно, возвращать книжный метафорический термин — нет.
    """
    for tekst in vse_teksty(bar):
        assert "оглавлен" not in tekst.lower(), f"книжное слово вернулось в подпись: {tekst!r}"


def test_knopka_chistki_nazyvaet_dejstvie_a_ne_strukturu(bar) -> None:
    """Кнопка обязана называть, что случится с фото пользователя, а не как устроено
    хранилище: «забыть разобранные фото» человек понимает до нажатия, «очистить
    оглавление» — только после."""
    nadpisi = " ".join(vse_teksty(bar)).lower()
    assert "забыть" in nadpisi, "кнопка чистки не названа понятным действием"
    assert "разобран" in nadpisi, "не сказано, что именно приложение забудет"
    # и обязательно снят страх: «забыть» не равно «удалить фото»
    assert "сами фото не удаляются" in nadpisi, \
        "кнопка «забыть» пугает удалением снимков — надо сказать обратное"


def test_perekhod_tekstov_lovit_vse_vidimoe(bar) -> None:
    """Сам обход бесполезен, если ничего не находит: проверяем, что в список попали
    конкретные подписи — заголовок ползунка, три режима, кнопка, подсказка и всплывающие
    объяснения. Без этого теста запрет на слова мог бы проверяться над пустым списком."""
    from ui.settings import PODSKAZKA_KACHESTVA, PODSKAZKA_POROGA, PODSKAZKA_REZHIMA

    teksty = vse_teksty(bar)
    for ozhidaemoe in ("Насколько фото должно быть похоже", "Оба", "Быстрее", "Точнее",
                       "внимательное (точнее)", "обычное (быстрее)", "Забыть разобранные фото",
                       PODSKAZKA_POROGA, PODSKAZKA_REZHIMA, PODSKAZKA_KACHESTVA, "влево"):
        assert any(ozhidaemoe in t for t in teksty), f"обход не увидел: {ozhidaemoe}"


def test_podskazka_soderzhit_izmerennyj_fakt(bar) -> None:
    """Подсказка рядом с ползунком обязана содержать измеренный факт из спецификации — именно
    он объясняет человеку, зачем вообще трогать настройку."""
    vse = " ".join(vse_teksty(bar))
    assert "выше 67% не бывает даже когда ребёнок на фото точно есть" in vse


def test_obiasnenie_nastrojki_govorit_chto_ona_delaet(bar) -> None:
    """Подсказка у ползунка объясняет следствие для человека («больше фото / только
    явные»), а не то, как устроена настройка внутри."""
    podskazka = bar._slider.toolTip().lower()
    assert "влево" in podskazka and "вправо" in podskazka
    assert len(podskazka) > 20, "подсказка пустая или бесполезная"


# --- оценка времени разбора (спецификация, раздел 6) --------------------------------


def test_ocenka_pustya_poka_arhiv_ne_izvesten(bar) -> None:
    """«≈ 0 с» на невыбранной папке было бы обещанием мгновенного разбора."""
    assert bar.estimate_label.text() == ""
    bar.set_archive_size(0)
    assert bar.estimate_label.text() == ""
    bar.set_archive_size(-5)               # чужое число не имеет права дать мусор
    assert bar.estimate_label.text() == ""


def test_ocenka_menyaetsya_vmeste_so_sposobom_poiska(bar) -> None:
    """Подпись «минуты на текущем архиве» стоит у способа поиска и обязана
    перестраиваться на его смене: иначе человек выбрал бы «Быстрее» и ждал бы
    столько же, сколько обещает «Оба»."""
    bar.set_archive_size(131)
    oba = bar.estimate_label.text()
    bar.set_engine("yunet")
    bystree = bar.estimate_label.text()
    bar.set_engine("insight")
    tochnoe = bar.estimate_label.text()
    assert len({oba, bystree, tochnoe}) == 3, f"оценка не различает способы: {oba}"
    assert "131 фото" in oba and oba.startswith("≈")
    # измеренный порядок дорогой: «Быстрее» < «Точнее» < «Оба»
    def v_sekundah(podpis: str) -> float:
        chasti = podpis.split()[1]
        chislo = float("".join(c for c in chasti if c.isdigit() or c == "."))
        if "ч" in chasti:
            return chislo * 3600
        return chislo * 60 if "мин" in chasti else chislo

    assert v_sekundah(bystree) < v_sekundah(tochnoe) < v_sekundah(oba)


def test_ocenka_rastet_s_arhivom_linejno(bar) -> None:
    """Удвоение архива удваивает ожидание: подпись читается как число снимков × цена."""
    bar.set_engine("yunet")
    bar.set_archive_size(10)               # 10 x 0,5 с = 5 с
    malyj = bar.estimate_label.text()
    bar.set_archive_size(400)              # 400 x 0,5 с = 200 с = 3 мин
    bolshoj = bar.estimate_label.text()
    assert malyj == "≈ 5 с на 10 фото", malyj
    assert bolshoj == "≈ 3 мин на 400 фото", bolshoj


def test_ocenka_skazana_kak_ocenka(bar) -> None:
    """Подсказка обязана объяснять, что это цена ПЕРВОГО разбора: повторный той же
    папки идёт в разы меньше, а молчаливое «минуты» человек проверил бы дважды.

    Проверяется свойство, а не слово. Прежняя версия искала в тексте «оглавлен» — и
    именно из-за этого книжное название хранилища и жило на экране: тест требовал
    термина там, где требовалось требование «скажи, что повтор дешевле».
    """
    podskazka = bar.estimate_label.toolTip().lower()
    assert "первый" in podskazka, "не сказано, что цена — у первого разбора"
    assert "повторн" in podskazka and "быстрее" in podskazka, \
        "не сказано, что повторный запуск дешевле — «минуты» читаются как приговор"


def test_polos_stroitja_bez_modelej_i_foto(qapp, tmp_path: Path) -> None:
    """Виджет конструируется без моделей и без единого файла на диске — это он и делает в
    первом кадре окна, до всякого выбора папки."""
    from ui.settings import SettingsBar

    do_polosa = list(tmp_path.iterdir())
    ponovo = SettingsBar()
    assert do_polosa == [] and list(tmp_path.iterdir()) == []
    assert ponovo.threshold == pytest.approx(0.38)
    assert ponovo.sizeHint().height() > 0, "полоса не влезает ни в одну раскладку"
