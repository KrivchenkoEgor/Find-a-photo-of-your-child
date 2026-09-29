"""Отчёты: человекочитаемый report.txt и таблица для Excel.

Правило, из которого растёт весь модуль: молчаливый ноль хуже ошибки. Родитель,
увидевший в отчёте «похожих: 0», не может отличить «ребёнка в архиве нет» от
«приложение потеряло половину снимков». Поэтому отчёт печатает и результаты, и
потери: сколько файлов не прочиталось, сколько лиц отброшено как непригодные и
сколько строк оглавления пришлось удалить как повреждённые. Число, которое вызывающий
не передал, печатается как «нет данных», а не как ноль: ложный ноль в этом разделе
был бы ровно той ложью, ради которой он и заведён.

Порядок строк — забота отчёта, а не разбора. `scanner.find_photos` отдаёт файлы,
отсортированные по полному пути, поэтому снимки одной папки шли бы подряд, а отчёт
читают не по папкам: он сортирует сам — по убыванию похожести, а при равных числах —
по имени файла. Порядок задаёт одна функция `_kluch`, и её же вызывают `build_rows`,
`write_report` и `write_csv`, чтобы сетка, report.txt и CSV не расходились.

Четыре детали, которые стоят места в коде.

1. Текст потери сворачивается в одну строку. Причина из `ScanStats.failures`
   продолжается сообщением библиотеки, и в этом сообщении бывают переводы строк:
   одна такая строка разорвала бы вёрстку отчёта, а в CSV разорвала бы запись.

2. Файлы группируются по причине, а не перечисляются. Pillow вставляет имя снимка в
   текст ошибки, так что двести одинаковых бед дали бы двести разных строк.
   Группировка режет причину по русской приставке («файл не читается») — она
   устойчива, а хвост от библиотеки бывает каким угодно. CSV от группировки свободен
   сознательно: он остаётся плоским и машинным, а потери живут в текстовом отчёте.

3. Подписи чисел ставят существительное перед числом («файлов: 21»), как в таблицах.
   Такая форма верна при любом числе, а «21 файлов» было бы ошибкой, и заметил бы
   человек её раньше самого текста.

4. Колонка «на какое из выбранных лиц похож снимок» печатается только при двух и больше
   выбранных лицах и всегда последней. Человек отмечает несколько примеров — среди них
   может стоять брат или взрослый, — и по одному «45 %» не понять, нашли ребёнка
   неуверенно или нашли ДРУГОГО. При одном же лице колонка молчит: одно и то же число
   в каждой строке отвечает на вопрос, которого не было, а сдвинутый молча столбец
   рассогласовал бы старые таблицы, где «рамка» лежала правее.

Модуль ничего не знает о Qt, моделях и рабочем потоке: на импорт он тянет только
numpy и два модуля ядра, чтобы отчёт строился и в тесте, и в пакетном прогоне без
окна. Отдельно поэтому сюда приходят числа потерь, а не объект `ScanStats`.
"""

from __future__ import annotations

import csv
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .engine import Face
from .matcher import ScoredPhoto, percent_of

_PRIMER = 3           # сколько имён файлов показать под одной причиной
_DIAGNOSTIK = 5       # сколько разных хвостов причины перечислить под заголовком
_SEMEN = 8            # сколько заголовков-семей печатать, остальные — счётчиком
_DLINA_DIAGNOSTIKI = 220   # длиннее — обрезаем: трейсировку показывать целиком нельзя


@dataclass(frozen=True)
class Sovpadenie:
    """Одно найденное лицо одного человека на одном снимке.

    Три числа, и все три нужны одновременно: `chelovek` отвечает на «КТО из искомых тут»,
    `lice` — на «ГДЕ в кадре» (номер лица в списке разбора, с единицы), `percent` — на
    «НАСКОЛЬКО похоже». Прежняя строка держала один общий максимум на смешанный список
    лиц, и в кадре, где ребёнок обнимает маму, было не понять, кого именно нашли.
    """

    chelovek: int
    lice: int
    percent: int


@dataclass(frozen=True)
class ResultRow:
    """Одна строка результата: снимок, его похожесть и все лица, которые нашёл разбор.

    `vse_lica` — рамки всех лиц в том порядке, в котором их вернул разбор; `lice` —
    позиция того лица, которое дало это число (с единицы; 0 — совпавшего лица нет);
    `sovpadeniya` — лица не ниже порога по каждому искомому человеку: ровно то, что
    обводится рамкой, и ровно то, что карточка называет числами. Один список на две
    поверхности: карточку сетки и окно просмотра (задачи T18, T19 и T20).
    """

    path: Path
    percent: int
    faces: int
    box: tuple[float, float, float, float] | None
    matched: bool
    # Номер строки эталона, давший максимум (с единицы); 0 — «не считали». Нужен затем,
    # чтобы в отчёте было видно, ПО КАКОМУ из выбранных лиц нашёлся снимок: человек
    # отмечает несколько примеров, и среди них может стоять другой человек. Одно число
    # «45 %» эти два случая не различает, а подпись различает (задача T15).
    ref_lico: int = 0
    vse_lica: tuple[tuple[float, float, float, float], ...] = ()
    lice: int = 0
    sovpadeniya: tuple[Sovpadenie, ...] = ()
    # Кто из искомых людей дал это число (номер в списке поиска, с единицы); 0 — никто
    # не оценивал этот снимок. Без него «61 %» остаётся вопросом «это кто?».
    chelovek: int = 0


NET_OTVETA = "никто из искомых не найден"

# Заголовок колонки в CSV — словами и через «лицо»: человек открывает таблицу в Excel и
# должен понять её, не спрашивая, что значат номера.
ZAGOLOVOK_KOLONKI = "кто найден и насколько похож"

# Строка-пояснение над таблицей в report.txt. Пояснение обязано быть: без него колонка
# «лицо 2» — это загадка, а не ответ.
POJASENIE_KOLONKI = "  кто из искомых людей найден на этом снимке и с каким числом"


def _kluch(row: ResultRow) -> tuple[int, str, str]:
    """Порядок отчёта: похожесть убывающая, затем имя файла, затем полный путь.

    Имя без учёта регистра: архив с iPhone лежит вперемешку `IMG_` и `img_`, и по
    регистру имена разъезжались бы. Полный путь в конце только разрешает ничью двух
    одинаковых имён из разных папок — иначе два запуска дали бы два разных отчёта.
    """
    return (-row.percent, row.path.name.casefold(), str(row.path))


def _po_poryadku(rows: Sequence[ResultRow]) -> list[ResultRow]:
    """Один источник порядка для сетки, report.txt и CSV."""
    return sorted(rows, key=_kluch)


def build_rows(photos: dict[Path, list[Face]],
               ocenki: Sequence[Mapping[Path, ScoredPhoto]],
               threshold: float) -> list[ResultRow]:
    """Плоские строки по убыванию похожести. Порог только помечает, что «похоже».

    `ocenki` — ПО ОДНОМУ словарю на каждого искомой человек, в том же порядке, в каком
    люди перечислены в поиске (задача T20). Один смешанный словарь на всех давал один
    общий максимум на кадр, и «61 %» не отвечало на вопрос, кто именно в кадре: ребёнок
    или тот взрослый, которого отметили вторым примером.

    Порог сравнивается с округлённым процентом, а не с исходным сходством: сетка
    результатов (задача 14) режет кучки по тому числу, которое человек видит на
    карточке. Сверять `similarity >= threshold` значило бы разойтись с галочкой на
    37.99%: на экране 38%, в отчёте «не похоже».

    Округление берёт у `matcher.percent_of` — у того же владельца, что у числа на
    карточке. Написанное здесь руками `round(threshold * 100)` расходится с сеткой на
    0.56 (это 56.00000000000001), и отчёт начинает называть «не похожим» снимок, у
    которого на экране стоит галочка. Два места с одним правилом однажды разъезжаются;
    одно не разъедется никогда.

    `photos` нужен, чтобы посчитать лица, но снимок, который есть в `photos` и не
    получил ни одной оценки, строку всё равно получает. Исчезнуть из отчёта бесследно
    фото не должно ни при каких промахах вызывающего кода.

    Тот же `photos` отдаёт строке список рамок. Номер лучшего лица ищется по кортежу
    координат, а не по объекту: `score_photo` возвращает лицо из этого же списка,
    поэтому координаты — честный признак, а вызывающий код свободен собрать оценки из
    копии списка.
    """
    porog = percent_of(threshold)        # то же число, что делит карточки на кучки
    razobrali = {put for o in ocenki for put in o}
    relevant = list(razobrali) + [p for p in photos if p not in razobrali]
    out = []
    for p in relevant:
        lica = photos.get(p, ())
        # Порядок людей задан списком `ocenki`, а не порядком файлов: номер человека
        # уходит в карточку и в отчёт, и он обязан означать одно и то же на всех снимках.
        ocenki_po_ljudjam = [(nomer, o[p]) for nomer, o in enumerate(ocenki, 1) if p in o]
        luchshaya = max((para for para in ocenki_po_ljudjam if para[1] is not None),
                        key=lambda para: para[1].percent, default=(0, None))
        nomer_cheloveka, s = luchshaya
        ramka_luchshego = s.face.box if (s is not None and s.face is not None) else None
        sovpadeniya = tuple(sorted(
            (Sovpadenie(chelovek, lice, procent)
             for chelovek, oc in ocenki_po_ljudjam
             for lice, procent in (oc.po_licam if oc is not None else ())
             if procent >= porog),
            key=lambda x: (x.lice, -x.percent, x.chelovek)))
        out.append(ResultRow(
            path=p,
            percent=s.percent if s is not None else 0,
            faces=len(lica),
            box=ramka_luchshego,
            matched=s is not None and s.percent >= porog,
            ref_lico=s.ref_lico if s is not None else 0,
            vse_lica=tuple(lic.box for lic in lica),
            lice=next((i + 1 for i, lic in enumerate(lica)
                       if lic.box == ramka_luchshego), 0),
            sovpadeniya=sovpadeniya,
            chelovek=nomer_cheloveka,
        ))
    return _po_poryadku(out)


def _kolonka_ljudj(imena_ljudj: Sequence[str] | None) -> list[str] | None:
    """Имена искомых людей, если колонка имеет смысл, иначе `None`.

    Порог смысла — один человек: «ребёнок 61 %» под каждой строкой отчёта говорит ровно
    то же, что заголовок раздела, и это шум. Со вторым человеком колонка становится
    ответом на вопрос, ради которого её и заводят: чьё это фото.

    Пустой список и `None` значат одно и то же («имён не передали»), и оба дают
    отсутствие колонки: вызывающий не обязан различать два вида «не знаю».
    """
    if not imena_ljudj or len(imena_ljudj) < 2:
        return None
    return list(imena_ljudj)


def _kto_nayden(row: ResultRow, imena: Sequence[str] | None) -> str:
    """Что написать в колонке: «ребёнок 61 %, мама 45 %» или честное «никто».

    Строка без совпадений — это не «мы не знаем», а «ни один искомый человек сюда не
    попал»: снимок лежит в слабом сходстве, и пустое место в таблице человек прочитал бы
    как испоренный документ. Поэтому здесь всегда слово.
    """
    if not imena:
        return NET_OTVETA
    lydi: dict[int, int] = {}
    for sovp in row.sovpadeniya:
        lydi[sovp.chelovek] = max(lydi.get(sovp.chelovek, 0), sovp.percent)
    bez = " · ".join(f"{imena[n - 1]} {procent} %"
                     for n, procent in sorted(lydi.items()) if 1 <= n <= len(imena))
    return bez or NET_OTVETA


def _stroka(text: Any) -> str:
    """Любой текст — в одну строку отчёта.

    Причина потери приходит из `ScanStats.failures` продолжением текста исключения,
    и в нём бывают переводы строк. Без сворачивания такая причина уезжает на
    несколько строк вперёд и сдвигает весь список.
    """
    return " ".join(str(text).split())


def _prichina(put: str, text: Any) -> str:
    """Причина потери в виде, годном для группировки: без имени конкретного файла.

    Диагностика библиотек обычно повторяет имя снимка («cannot identify image file
    'IMG_1234.HEIC'»), и если группировать по тексту целиком, двести одинаковых бед
    дали бы двести заголовков. Сначала снимаем полный путь, затем имя файла: рабочий
    поток подменяет путь именем, поэтому нужны оба варианта.

    Пустой путь — не «файл без имени», а «файла нет вовсе»: `Path("")` сворачивается в
    `.`, а `.` — самая жадная подстрока в тексте, и она сняла бы каждую точку в
    причине («OSError. Ошибка 28.» → «OSError Ошибка 28»). Теряется ровно то, ради
    чего причину и печатают. Поэтому `.` здесь пропускаем. До `worker` это не доходит
    (он всегда зовёт с настоящим путём), а приходит от вызывающего, у которого одна
    беда на весь прогон и назвать файл нечем: из интерфейса это
    `("", "копирование не удалось: …")`.
    """
    svod = _stroka(text)
    p = Path(str(put))
    for imya in sorted({str(p), p.name}, key=len, reverse=True):
        if imya and imya != ".":
            svod = svod.replace(imya, "")
    svod = svod.replace("''", "").replace('""', "")
    return _stroka(svod).rstrip(" :,.") or "без описания"


def _semya(prichina: str) -> tuple[str, str]:
    """Русская приставка причины и её хвост.

    Приставка — устойчивая часть: `worker` строит причины как «файл не читается: …»,
    «файл недоступен: …», «ошибка распознавания: …». Хвост принадлежит библиотеке и
    меняется от снимка к снимку; без хвоста заголовок остаётся одной приставкой.
    """
    semya, _, hvost = prichina.partition(": ")
    return (semya or "без описания"), _stroka(hvost)


@dataclass
class _Gruppa:
    """Одна семья причин: сколько файлов, чем именно и примеры для глаз."""

    semya: str
    vsego: int = 0
    diagnostiki: Counter[str] = field(default_factory=Counter)
    imena: list[str] = field(default_factory=list)


def _gruppy(failures: Sequence[tuple[str, str]]) -> list[_Gruppa]:
    """Пары (файл, причина) → семьи причин, крупные сверху.

    Имя в примеры берём только настоящее: у потери без пути — когда одна беда на весь
    прогон и назвать файл нечем — пустая строка в `примеры: ` читалась бы как
    испорченный отчёт, а не как «файл не назван».
    """
    po_semyam: dict[str, _Gruppa] = {}
    for put, text in failures:
        semya, hvost = _semya(_prichina(put, text))
        gruppa = po_semyam.get(semya)
        if gruppa is None:
            gruppa = po_semyam[semya] = _Gruppa(semya)
        gruppa.vsego += 1
        if hvost:
            gruppa.diagnostiki[hvost] += 1
        imya = Path(str(put)).name
        if imya:
            gruppa.imena.append(imya)
    for gruppa in po_semyam.values():
        gruppa.imena.sort(key=str.casefold)      # примеры всегда одни и те же
    return sorted(po_semyam.values(), key=lambda g: (-g.vsego, g.semya.casefold()))


def _stoki_poterej(failures: Sequence[tuple[str, str]]) -> list[str]:
    """Раздел «Не обработано»: заголовок на семью причин, число и до трёх имён.

    Список файлов здесь бесполезен: в архиве с кривым HEIC таких двести строк, и
    человек не дочитывает ни до чего. Заголовок говорит причину и сколько, примеры
    дают, что открыть глазами, а хвосты библиотек перечислены числом — с ними
    осмысленно идти в поддержку.
    """
    if not failures:
        return []
    linii = ["", "Не обработано", f"  всего файлов: {len(failures)}"]
    gruppy = _gruppy(failures)
    for gruppa in gruppy[:_SEMEN]:
        linii += ["", f"  {gruppa.semya} — файлов: {gruppa.vsego}"]
        hvosty = sorted(gruppa.diagnostiki.items(), key=lambda kv: (-kv[1], kv[0]))
        for hvost, schet in hvosty[:_DIAGNOSTIK]:
            mnogotochie = "" if len(hvost) <= _DLINA_DIAGNOSTIKI else " …"
            linii.append(f"    {hvost[:_DLINA_DIAGNOSTIKI]}{mnogotochie} — всего: {schet}")
        if len(hvosty) > _DIAGNOSTIK:
            linii.append(f"    иных диагностик: {len(hvosty) - _DIAGNOSTIK}")
        pokaz = gruppa.imena[:_PRIMER]
        if pokaz:
            linii.append(f"    примеры: {', '.join(pokaz)}")
        if gruppa.vsego > len(pokaz):
            linii.append(f"    и ещё файлов: {gruppa.vsego - len(pokaz)}")
    if len(gruppy) > _SEMEN:
        linii += ["", f"  и других причин: {len(gruppy) - _SEMEN}"]
    return linii


def _schet(znachenie: int | None) -> str:
    """Число или «нет данных».

    Надо отличить «приложение не отбросило ни одного лица» от «вызывающий не передал
    число»: ложный ноль — ровно та ложь, ради которой раздел потерь и заведён.
    """
    return "нет данных" if znachenie is None else str(znachenie)


def write_report(path: Path, *, settings: Mapping[str, str],
                 rows: Sequence[ResultRow],
                 failures: Sequence[tuple[str, str]],
                 copied: Sequence[tuple[Path, Path]],
                 dropped_faces: Mapping[Path | str, int] | None = None,
                 cache_damage: int | None = None,
                 imena_ljudj: Sequence[str] | None = None) -> Path:
    """Пишет report.txt и возвращает путь к нему.

    Аргументы потерь — продолжение `ScanStats`, а не сам объект: отчёт не должен
    тянуть рабочий поток, Qt и модели, иначе его нельзя ни построить в тесте, ни
    написать из пакетного прогона без окна. Вызывающий передаёт `stats.dropped_faces`
    и `stats.cache_damage` как есть — ключи словаря там Path, пересборка не нужна.

    `failures` — пары (путь, причина) из `ScanStats.failures`: причина может быть
    многострочной и повторять имя файла, и то и другое здесь исправляется.

    Пропущенное число потерь печатается как «нет данных», а не как ноль: отчёт,
    который молчит о том, чего не знает, неотличим от отчёта о удачном разборе.

    `imena_ljudj` — как человек назвал искомых людей, номер в списке = номер человека
    в списке = номер строки эталона. Колонка появляется только при двух и больше именах:
    при одном выбранном лице подпись стояла бы под каждой строкой и не отвечала бы ни на
    какой вопрос. Имена приходят СНАРУЖИ, а не считаются здесь: отчёт не знает, что
    такое эталон, и не должен тянуть `core.etalon` ради подписей.
    """
    lica = None if dropped_faces is None else sum(int(n) for n in dropped_faces.values())
    snimki = None if dropped_faces is None else len(dropped_faces)
    povrezhdeniya = None if cache_damage is None else max(0, int(cache_damage))
    poryadok = _po_poryadku(rows)
    pohozhie = sum(1 for r in poryadok if r.matched)

    linii: list[str] = [
        "Отчёт поиска фотографий",
        f"дата: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "Настройки",
    ]
    linii += [f"  {_stroka(k)}: {_stroka(v)}" for k, v in settings.items()] or ["  не записаны"]
    linii += [
        "",
        "Итог",
        f"  фото в отчёте: {len(poryadok)}",
        f"  фото с лицами: {sum(1 for r in poryadok if r.faces > 0)}",
        f"  похоже: {pohozhie}",
        f"  слабое сходство: {len(poryadok) - pohozhie}",
        f"  скопировано: {len(copied)}",
        "",
        "Потери — чтобы «похожих: 0» не читалось как «ребёнка в архиве нет»",
        f"  файлов не обработано: {len(failures)}",
        f"  лиц отброшено: {_schet(lica)}",
    ]
    if lica:
        linii.append("    лицо с непригодным отпечатком: сам снимок разобран и потерян "
                     "не был, но сравнить по нему нельзя")
    linii += [f"  снимков с отброшенным лицом: {_schet(snimki)}",
              f"  строк оглавления удалено: {_schet(povrezhdeniya)}"]
    if povrezhdeniya:
        linii.append("    оглавление починилось само: эти снимки перечитаны моделями "
                     "заново")
    if not failures and lica == 0 and povrezhdeniya == 0:
        linii.append("  потерь нет: ни один файл не пропущен, оглавление целое")

    linii += ["", "Найденные фото — по убыванию похожести, при равных числах по имени файла"]
    lydi = _kolonka_ljudj(imena_ljudj)
    if lydi:
        linii.append(POJASENIE_KOLONKI)
    if not poryadok:
        linii.append("  ни одного разобранного фото")
    else:
        powtornye = {n for n, c in Counter(r.path.name for r in poryadok).items() if c > 1}
        for r in poryadok:
            imya = r.path.name
            if imya in powtornye:
                # Порядок по имени ставит одноимённые снимки рядом: без папки их не
                # различить, а перепутать при переносе руками — проще всего.
                imya = f"{imya} [{r.path.parent.name}]"
            mark = "похоже" if r.matched else "слабое сходство"
            # Хвост строки — только когда отвечать есть на что: при одном искомом человеке
            # колонка молчит, и отчёт выглядит ровно как раньше.
            hvost = f" · {_kto_nayden(r, lydi)}" if lydi else ""
            linii.append(f"  {r.percent:>3}%  {imya:<24} лиц: {r.faces:<3} {mark}{hvost}")

    linii += _stoki_poterej(failures)

    if copied:
        # Порядок — как копировали: суффиксы `_1`, `_2` нарастают именно в нём, и
        # пересортированный список перестал бы объяснять, откуда взялось второе имя.
        linii += ["", "Скопировано в папку результатов", f"  файлов: {len(copied)}"]
        linii += [f"  {_stroka(s.name)} -> {_stroka(d)}" for s, d in copied]

    itog = Path(path)
    itog.parent.mkdir(parents=True, exist_ok=True)
    itog.write_text("\n".join(linii) + "\n", encoding="utf-8")
    return itog


def write_csv(path: Path, rows: Sequence[ResultRow],
              imena_ljudj: Sequence[str] | None = None) -> Path:
    """Плоская таблица для Excel: одна строка на снимок, без заголовков-группировок.

    `utf-8-sig` — из-за BOM: без него Excel открывает кириллицу кракозябрами, и
    человек решает, что испорчен файл, а не кодировка. Порядок тот же, что в
    report.txt (`_po_poryadku`), чтобы две части отчёта не начинались с разных фото.

    Столбец «на какое лицо похоже» — ВСЕГДА последний и только когда выбранных лиц
    больше одного. Прежние пять колонок при этом не сдвигаются ни на шаг: человек
    держит старые таблицы, открытый с новой версии CSV с переставленными столбцами
    означал бы «рамка» в колонке «похоже», а такой ошибки ему хватало бы и на одном
    файле.

    Потери сюда не приходят сознательно: у них форма «причина → список файлов», а
    в плоской таблице они стали бы колонкой с текстом в три строки. Их место в
    текстовом отчёте, где их читают глазами.

    Путь в таблице не правится: путь — это данные, и `csv` экранирует даже перевод
    строки внутри имени файла, а «исправленный» путь перестал бы указывать на снимок.
    """
    itog = Path(path)
    itog.parent.mkdir(parents=True, exist_ok=True)
    lydi = _kolonka_ljudj(imena_ljudj)
    zagolovki = ["файл", "процент", "лиц", "похоже", "рамка"]
    if lydi:
        zagolovki.append(ZAGOLOVOK_KOLONKI)
    with open(itog, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(zagolovki)
        for r in _po_poryadku(rows):
            zapis = [str(r.path), r.percent, r.faces, int(r.matched),
                     ";".join(f"{v:.0f}" for v in r.box) if r.box else ""]
            if lydi:
                zapis.append(_kto_nayden(r, lydi))
            writer.writerow(zapis)
    return itog
