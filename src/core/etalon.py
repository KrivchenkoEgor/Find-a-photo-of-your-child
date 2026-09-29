"""Чистая арифметика окна эталона: отметки карточек -> то, что реально пойдёт в поиск.

Модуль отвечает на один вопрос: «человек отметил вот эти карточки — какой эталон и с
какими числами мы получаем?». Здесь нет ни одного импорта Qt: виджет (задача T3)
только рисует то, что вернул этот модуль, и пересылает сюда набор отметок. Пока
правила живут здесь, они остаются проверяемыми без окна — одноимённый тест собирает
`Face` в память и не открывает ни одного файла.

Три греха, из-за которых модуль написан заново, а не вынут из `main_window`.

**Порядок кликов не должен ничего решать.** Прежний anchor — «лицо, которое
подтвердили первым» — в сетке с мышью превращает эталон в историю движений курсора:
те же шесть карточек, отмеченные в другом порядке, давали бы другой поиск и другой
список находок. Здесь anchor — самое крупное годное из отмеченных (`Face.size`), а
при равенстве размеров — первое в плоском порядке карточек. Плоский порядок задаётся
словарём файлов, а не множеством отметок, ответ функции зависит только от состава
отметок.

**Отмеченное лицо участвует в поиске (решение T14, отменяет R-1).** Человек ставит
галочку, и никто не имеет права молча её не выполнить. Раньше лицо, похожее на anchor
меньше чем на `porog_slabosti`, в эталон не бралось: человек видел «лиц в поиске: 9 из
16 отмеченных» и делал вывод, что приложение ищет по одному лицу. Теперь в поиск
берётся всякое отмеченное лицо с годным отпечатком, а число ниже `porog_slabosti`
превратилось из запрета в предупреждение — признак `slabo`, из которого виджет
собирает слова «похоже слабо». Единственный настоящий отказ — непригодный отпечаток:
таким вектором нечем считать, и никакой выбор числа тут ни при чём.

**Отказ обязан быть назван.** `matcher.make_reference` молча выбрасывал добавленное лицо
по числу сходства; теперь окно зовёт его с `min_sim=None`, и молчать
больше негде. В `otvergnutye` остаётся ровно одна причина — «сравнить нечем», и она
обязана доехать до человека словами, а не исчезновением карточки. Годный отпечаток,
каким бы похожим он ни был, теперь не выпадает из поиска никогда.

Порог предупреждения не имеет здесь второго владельца: `porog_slabosti` по умолчанию —
тот же объект, что и `REF_MIN_SIM` в `core.matcher`. Заведя копию числа, окно однажды
начало бы подсвечивать бледностью не те карточки, о каких говорит подсказка.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .engine import Face, embeddings_of, prigoden_otpechatok
from .matcher import REF_MIN_SIM, Reference, make_reference, percent_of


@dataclass(frozen=True)
class Kartochka:
    """Один снимок с одним лицом — ровно как его видит человек в сетке.

    `nomer` считается с единицы, потому что так же его подписывает интерфейс; у
    снимка без лиц номер 0 и `lic is None` — карточка обязана существовать, иначе
    файл молча исчезнет и «лица нет» станет неотличим от «файл не прочитался».

    `slabo` добавлено задачей T14 и существует потому, что виджет не считает сходство
    сам: без этого поля словам «похоже слабо» пришлось бы рождаться из числа прямо в
    отрисовке, и второе владение правилом появилось бы ровно там, где его ловили весь
    раунд. У anchor и у неотмеченных карточек `slabo` всегда `False`.

    Заморожена намеренно: пересчёт окна не должен иметь возможности подправить уже
    показанное число.
    """

    put: Path                 # файл, с которого это лицо
    nomer: int                # номер лица внутри файла, начиная с 1
    lic: Face | None          # None — на снимке лицо не найдено
    otmecheno: bool           # человек отметил карточку
    # Наибольшее сходство с ДРУГИМ отмеченным лицом этого же человека, в процентах.
    # `None` — сравнивать не с чем: лицо в наборе одно (или человек его не отмечал).
    # Никакого «100 %» за то, что лицо крупнее остальных, здесь больше нет (задача T21),
    # и «0 %» на пустом месте тоже: ноль без собеседника человек читает как «не похоже».
    procent: int | None
    v_poisk: bool             # отмечено и отпечаток годен — участвует в сравнении
    slabo: bool               # в поиске, но не похоже на остальных — предупреждение


@dataclass(frozen=True)
class Itog:
    """Результат пересчёта: все карточки плюс готовый к поиску эталон.

    `anchor`, `extra` и `otvergnutye` — те же самые объекты, что лежат в `karty`, а не
    их копии: виджет сверяет карточки по идентичности и подсвечивает их на месте.

    Держать в голове стоит одно: с задачи T14 `otvergnutye` — это НЕ «не похожее», а
    «нечем считать». Отмеченное и годное лицо всегда лежит в `extra` (или является
    `anchor`), каким бы низким ни оказалось его число, а `slabo` рядом с ним только
    просит человека перепроверить глазами. Инвариант `reference.count == 1 +
    len(extra)` связывает ответ с тем, что реально уйдёт в сравнение.
    """

    karty: tuple[Kartochka, ...]        # все карточки, в плоском порядке
    anchor: Kartochka | None
    extra: tuple[Kartochka, ...]        # отмеченные, годные и взяемые в поиск, кроме anchor
    otvergnutye: tuple[Kartochka, ...]  # отмеченные, но с непригодным отпечатком
    reference: Reference | None         # готовый эталон для score_photo


@dataclass
class Chelovek:
    """Один человек в списке «Кого ищем»: имя и ответ окна эталона про него.

    Изменяемый нарочно: имя человек правит прямо в строке главного окна, не пересобирая
    поиск, а `itog` приезжает из диалога целиком — только так повторно открытое окно
    показывает тот же выбор, каким человек его оставил.

    `reference` — единственное, что нужно поиску от этого человека. Держать его отдельным
    полем значило бы иметь два ответа на вопрос «кого ищем»: при правке эталона одно
    обязательно отстанет от другого.
    """

    nazvanie: str
    itog: Itog | None = None

    @property
    def reference(self) -> Reference | None:
        return self.itog.reference if self.itog is not None else None

    @property
    def litsa(self) -> int:
        """Сколько отпечатков этого человека участвует в сравнении. 0 — фото не выбрано."""
        return self.reference.count if self.reference is not None else 0


@dataclass(frozen=True)
class _Zagotovka:
    """Карточка до того, как известны anchor и проценты: файл, номер, лицо."""

    put: Path
    nomer: int
    lic: Face | None


def odinochnye_otmetki(
        lica_po_fajlam: dict[Path, list[Face]],
        zanyatye: frozenset[tuple[Path, int]] = frozenset()
) -> frozenset[tuple[Path, int]]:
    """Автоотметки «на снимке ровно одно лицо» — и только для лиц, о которых ещё не сказали.

    Возвращает пары (путь, индекс лица в списке этого файла). Файлы без лиц и файлы
    с несколькими лицами не дают отметки вовсе: угадывать, кто из троих искомый человек,
    приложение не имеет права.

    `zanyatye` — те же пары, уже отмеченные за ДРУГИМ человеком (задача T22). Без них
    один и тот же набор снимков, принесённый для мамы, автоматически отмечал спящего
    ребёнка: автоотметка была заказана для «я приношу фото своего ребёнка», а там, где
    человек уже сказал, кто на снимке, она решает против него.
    """
    return frozenset((put, 0) for put, lica in lica_po_fajlam.items()
                     if len(lica) == 1 and (put, 0) not in zanyatye)


def sobrat_etalon(lica_po_fajlam: dict[Path, list[Face]],
                  otmetki: frozenset[tuple[Path, int]],
                  porog_slabosti: float = REF_MIN_SIM) -> Itog:
    """Единственная точка входа: набор лиц + набор отметок -> карточки и эталон.

    Порядок действий выбран так, чтобы ни один шаг не зависел от порядка обхода
    `otmetki` (а `frozenset` обходит элементы как ему вздумается): сначала строится
    плоский каркас карточек, потом отметки превращаются в МНОЖЕСТВО позиций этого
    каркаса, и все решения — anchor, деление на «идёт/не идёт», порядок в `karty` —
    принимаются по возрастанию позиции. Отметки, которым не нашлось лица, отбрасываются
    здесь же и молча: между кликом и пересчётом список лиц мог перестроиться.

    `porog_slabosti` больше не решает, брать ли лицо в поиск, — он решает, предупреждать
    ли о нём. Сборка эталона вызывается с `min_sim=None`, и это не случайность, а то
    самое решение T14: молча отброшенная галочка человека была бы ровно тем обманом, из-за
    которого весь раунд и начался.
    """
    karkas, pozicii = _ploskiy_karkas(lica_po_fajlam)
    otmechennye = frozenset(p for p in (pozicii.get(otmetka) for otmetka in otmetki)
                            if p is not None)

    anchor = _vybrat_pervoy_stroki(karkas, otmechennye)
    v_poisk, otvergnutye = _razdelit_po_godnosti(karkas, otmechennye, anchor)
    # Число каждого лица меряется с самым похожим из ЕГО СОБЕСЕДНИКОВ по этому же
    # человеку, а не с одной «главной» фотографией.
    shodstva = _shodstva_s_chelem(karkas, v_poisk)

    reference: Reference | None = None
    if anchor is not None:
        try:
            reference = make_reference(
                _lic(karkas, anchor),
                [_lic(karkas, p) for p in sorted(v_poisk) if p != anchor],
                min_sim=None,
            )
        except ValueError:
            # Недостижимо: anchor выбирается только среди годных отпечатков, а
            # `make_reference` бросает ValueError ровно на непригодном anchor. Но
            # обещание «окно не падает» дороже правоты по построению: проваливаемся в
            # состояние «эталона нет» целиком, чтобы инварианты не врали друг другу.
            anchor = None
            v_poisk = frozenset()
            otvergnutye = otmechennye

    karty: list[Kartochka] = []
    for poziciya, zagotovka in enumerate(karkas):
        otmecheno = poziciya in otmechennye
        if not otmecheno:
            karty.append(Kartochka(put=zagotovka.put, nomer=zagotovka.nomer,
                                   lic=zagotovka.lic, otmecheno=False,
                                   procent=None, v_poisk=False, slabo=False))
            continue
        sim = shodstva.get(poziciya)
        v_poisk_zdes = poziciya in v_poisk
        karty.append(Kartochka(put=zagotovka.put, nomer=zagotovka.nomer, lic=zagotovka.lic,
                               otmecheno=True,
                               procent=None if sim is None else percent_of(sim),
                               v_poisk=v_poisk_zdes,
                               slabo=_slabo(v_poisk_zdes, sim, porog_slabosti)))

    return Itog(
        karty=tuple(karty),
        anchor=karty[anchor] if anchor is not None else None,
        extra=tuple(karty[p] for p in sorted(v_poisk) if p != anchor),
        otvergnutye=tuple(karty[p] for p in sorted(otvergnutye)),
        reference=reference,
    )


# --- внутреннее -----------------------------------------------------------------------


def _ploskiy_karkas(
        lica_po_fajlam: dict[Path, list[Face]],
) -> tuple[list[_Zagotovka], dict[tuple[Path, int], int]]:
    """Каркас карточек в плоском порядке и словарь «отметка -> позиция».

    Порядок задаётся словарём файлов (он сохраняет порядок вставки) и номером лица
    внутри файла — отметки в нём не участвуют, поэтому мышь на список не влияет.
    Файл без лиц получает одну карточку с `nomer=0` и `lic=None`.

    Отдельный словарь нужен для третьего правила: отметка без своего лица (индекс вне
    диапазона, файл, которого нет в словаре) должна быть проигнорирована, а не найти
    ближайшее похожее.
    """
    karkas: list[_Zagotovka] = []
    pozicii: dict[tuple[Path, int], int] = {}
    for put, lica in lica_po_fajlam.items():
        if not lica:
            karkas.append(_Zagotovka(put=put, nomer=0, lic=None))
            continue
        for indeks, lic in enumerate(lica):
            karkas.append(_Zagotovka(put=put, nomer=indeks + 1, lic=lic))
            pozicii[(put, indeks)] = len(karkas) - 1
    return karkas, pozicii


def _lic(karkas: list[_Zagotovka], poziciya: int) -> Face | None:
    """Лицо карточки по её позиции (короткая запись для читаемых вызовов)."""
    return karkas[poziciya].lic


def _shodstva_s_chelem(karkas: list[_Zagotovka],
                       pozicii: frozenset[int]) -> dict[int, float | None]:
    """Для каждого лица набора — наибольшее сходство с другим его лицом. `None` — не с чем.

    Это ответ на вопрос человека «то же это лицо, что и остальные, или я промахнулся
    галочкой», и он не зависит от того, какое из отмеченных лиц оказалось крупнейшим.
    Прежнее сравнение с одной «главной» фотографией в наборе для мамы выдавало
    предупреждение всем маминым лицам, потому что крупнейшим в принесённых снимках
    оказался спящий ребёнок.
    """
    poryadok = sorted(pozicii)
    if len(poryadok) < 2:
        return {poziciya: None for poziciya in poryadok}
    matrica = embeddings_of([_lic(karkas, p) for p in poryadok])
    # Делим на нормы: годный отпечаток не обязан быть единичной длины (`prigoden_otpechatok`
    # требует конечность, ширину и ненулевую норму), а оглавление старой версии вполне
    # может привезти строку вдвое длиннее. Без этого косинус превращается в 4.0, и на
    # карточке появляется «400 %».
    normy = np.linalg.norm(matrica, axis=1, keepdims=True)
    blizost = (matrica / normy) @ (matrica / normy).T
    return {poryadok[i]: max(float(blizost[i][j]) for j in range(len(poryadok)) if j != i)
            for i in range(len(poryadok))}


def _vybrat_pervoy_stroki(karkas: list[_Zagotovka], otmechennye: frozenset[int]) -> int | None:
    """Позиция крупнейшего ГОДНОГО из отмеченных — та, что станет первой строкой эталона.

    Выбор по размеру нужен только `make_reference`: ему обязана предшествовать одна
    конкретная строка. Для человека здесь нет ни «главного», ни «подтверждённого» лица:
    в поиск идут все отмеченные наравне (T14), и на экране это лицо ничем не помечено.

    Непригодное лицо anchor'ом не становится: `make_reference` бросает на нём
    ValueError, и окно падало бы прямо на пустом месте — на снимке, который человек
    отметил последним из шести.

    Сравнение строгое (`>`), поэтому первое встретившееся крупнейшее и остаётся:
    правило «первое в плоском порядке» выполняется само, без второго прохода.
    """
    luchshaya: int | None = None
    for poziciya in sorted(otmechennye):
        lic = _lic(karkas, poziciya)
        if lic is None or not prigoden_otpechatok(lic.embedding):
            continue
        if luchshaya is None or lic.size > _razmer(karkas, luchshaya):
            luchshaya = poziciya
    return luchshaya


def _razmer(karkas: list[_Zagotovka], poziciya: int) -> float:
    """Размер лица карточки; у карточки без лиц -1, то есть крупнейшей она не бывает.

    Ветка с `-1` — страховка: сегодня anchor выбирается только среди карточек с лицом,
    поэтому `None` сюда не доходит. Убери проверку — и вызов для карточки «лицо не
    найдено» кончился бы AttributeError вместо спокойного «эта карточка не крупней».
    """
    lic = _lic(karkas, poziciya)
    return -1.0 if lic is None else float(lic.size)


def _razdelit_po_godnosti(
        karkas: list[_Zagotovka], otmechennye: frozenset[int],
        anchor: int | None) -> tuple[frozenset[int], frozenset[int]]:
    """Разделение отмеченных на «идёт в поиск» и «не идёт» — по годности отпечатка.

    С задачи T14 это единственное деление, которое тут вообще есть: число сходства на
    состав поиска не влияет, влияет только «чем считать». Anchor в первой паре всегда,
    когда он есть: человек подтвердил именно это фото, и потерять его значит потерять
    ребёнка из поиска.

    Непригодное лицо попадает во вторую пару — и это не решение приложения, а физика:
    вектор с NaN или чужой шириной не даёт ни косинуса, ни строки эталона, поэтому
    сравнить такое лицо буквально нечем. Именно поэтому `otvergnutye` остаётся, хотя
    молчаливый отказ по числу из него убрали.

    Отметка без anchor (ни одного годного лица) не даёт поиска вовсе: вторая пара тогда
    включает все отмеченные лица, и виджет подписывает каждое.
    """
    if anchor is None:
        return frozenset(), frozenset(otmechennye)
    v_poisk = {anchor}
    otvergnutye: set[int] = set()
    for poziciya in otmechennye:
        if poziciya == anchor:
            continue
        lic = _lic(karkas, poziciya)
        if lic is None or not prigoden_otpechatok(lic.embedding):
            otvergnutye.add(poziciya)
        else:
            v_poisk.add(poziciya)
    return frozenset(v_poisk), frozenset(otvergnutye)


def _slabo(v_poisk: bool, sim: float | None, porog_slabosti: float) -> bool:
    """Нужно ли предупредить человека: «это лицо не похоже на остальные его же лица».

    Два ложных случая отсекаются обоими условиями: у карточки вне поиска беда
    называется другими словами («сравнить нечем»), а у единственного лица в наборе
    сравнивать нечего — предупреждать «не похоже на остальных» без остальных значит
    пугать человека пустотой. Сравнение строгое (`<`): ровно на границе предупреждения
    нет, как и прежний отбор по `>=`.
    """
    return bool(v_poisk and sim is not None and sim < porog_slabosti)

