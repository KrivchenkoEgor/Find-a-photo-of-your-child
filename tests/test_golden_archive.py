"""Боевая проверка ядра на реальном архиве: ровно то, что подтвердил человек.

Эталон истинности — не форма распределения процентов, а 20 файлов из 131, где человек
глазами подтвердил ребёнка. При 2400 px, режиме «Оба» и пороге 0.38 эти 20 должны
найтись все и без единого лишнего: таков замер `docs/bench-2026-09-28-detektory.md`
(«Версия 3», строка «порог 0.38 — 20/20, всего фото 20, непроверенных 0»). Если
приложение расходится с этим замером — это регрессия ядра, а не повод смягчать тест.

Почему тест отдельный и почему он медленный. Быстрые тесты проверяют правила на подставных
движках и синтетических кадрах. Здесь под проверкой вся цепочка целиком: декодировка
24-мегапиксельных снимков, уменьшение, оба детектора, оглавление, слияние лиц, эталон из
`ref/` и сравнение. Единственный честный способ это проверить — прогнать настоящий архив,
а это несколько минут работы.

Приватность. `data/` и `ref/` закрыты в `.gitignore`: снимки ребёнка не в репозитории.
Там же лежит и `arhiv.local.json` — путь к архиву, имя эталонного снимка и список
подтверждённых файлов: по одним только именам вместе с логином на GitHub можно
восстановить, чьи фото разбирают. Без этого файла (или без папок, на которые он указывает)
тест пропускается с причиной, в которой названо, что скопировать из `arhiv.example.json`, —
а не падает и не выдаёт ложное «пройдено» (спецификация, раздел 11). Тест читает архив
только через `find_photos` и `load_photo`, ничего в него не пишет и оглавление заводит
временное, в `tmp_path`.

Что проверяется, кроме самих находок:
  * оглавление реально экономит работу — второй прогон не обращается к моделям вовсе;
  * в режиме «Оба» модель распознавания ОДНА на оба пути (вторая копия w600k_r50 на
    174 МБ — утечка памяти, а не деталь реализации);
  * слияние лиц на групповых снимках: сколько рамок оно сворачивает, что это делает с
    оценкой и не прячет ли подтверждённое фото. Правило не меняем — измеряем.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pytest

from core import both as obasha
from core.cache import FaceCache
from core.engine import Face, make_engine
from core.matcher import (Reference, ScoredPhoto, make_reference, percent_of,
                          score_photo)
from core.report import ResultRow, build_rows
from core.scanner import PHOTO_SUFFIXES, find_photos, load_photo
from core.worker import ScanStats, obespechit_zagruzku, scan_photos
from utils.config import DEFAULT_ENGINE, DEFAULT_MAX_DIM, DEFAULT_THRESHOLD, yunet_model_path

# Пути от корня репозитория, а не от рабочей папки: `pytest tests/test_golden_archive.py`
# из подпапки иначе отчитается «нет папки data/» и тест мирно пропустится там, где его
# как раз и ждали. Пропуск вместо падения — худший из исходов для боевого теста.
#
# Ни папки, ни списка подтверждённых снимков здесь нет: они лежат в `arhiv.local.json`
# (образец — `arhiv.example.json`), который закрыт в `.gitignore`. Имена файлов семейного
# архива — такие же личные данные, как и сами снимки.
from zolotoj_arhiv import prichina_propuska, prochitat

ARKHIV = prochitat()
DATA = ARKHIV.photos if ARKHIV is not None else None
REF = ARKHIV.ref if ARKHIV is not None else None
ANCHOR = ARKHIV.anchor if ARKHIV is not None else ""
CONFIRMED = ARKHIV.confirmed if ARKHIV is not None else frozenset()
PRICHINA_PROPUSKA = prichina_propuska(ARKHIV)
NET_ARKHIVA = bool(PRICHINA_PROPUSKA)

# Настройки, на которых считался эталон истинности. Держим их отдельными константами,
# а не читаем `config.DEFAULT_*`: сдвиг умолчаний приложения не должен молча уносить
# и этот тест. Сверку с текущими умолчаниями делает `test_umolchaniya_prilozheniya_te_zhe`.
MAX_DIM = 2400
THRESHOLD = 0.38
ENGINE = "both"
# Округление берём у того же владельца правила, что и сетка, — иначе «схоже» в этом тесте
# и «схоже» на экране однажды начнут означать разные числа, и боевой прогон перестанет
# проверять то, что видит человек.
POROG = percent_of(THRESHOLD)

NADO_ARKHIV = pytest.mark.skipif(NET_ARKHIVA, reason=PRICHINA_PROPUSKA)


def test_prichina_bez_lokalnogo_fayla_nazyvaet_chto_skopirovat() -> None:
    """Свежий клон должен получить инструкцию, а не пустой «skipped».

    Пропуск боевого теста сам по себе тихий: человек запускает `pytest`, видит зелёный
    отчёт и решает, что ядро проверено. Причина обязана называть файл-образец.
    """
    from zolotoj_arhiv import prichina_propuska

    prichina = prichina_propuska(None)

    assert "arhiv.example.json" in prichina, prichina
    assert "arhiv.local.json" in prichina, prichina


def test_konfiguraciya_bez_polya_govorit_kakogo(tmp_path) -> None:
    """Половина полей — это ошибка с именем отсутствующего, а не `KeyError`.

    Файл пишет человек от руки. `KeyError: 'confirmed'` он не разберёт, а «нет полей
    confirmed» — разберёт.
    """
    import pytest as _pytest

    from zolotoj_arhiv import Arhiv

    with _pytest.raises(ValueError) as vozrazhenie:
        Arhiv({"photos": "a", "reference": "b", "anchor": "c.jpg"}, koren=tmp_path)

    assert "confirmed" in str(vozrazhenie.value), str(vozrazhenie.value)


def test_ukazannoj_papki_net_prichina_a_ne_padenie(tmp_path) -> None:
    """Нет папки, на которую указывает конфиг, — причина пропуска с самим путём.

    Путь в тексте нужен, чтобы человек понял: опечатка в конфиге или фото лежат в другом
    месте, — иначе он начнёт искать не там.
    """
    from zolotoj_arhiv import Arhiv, prichina_propuska

    arkhiv = Arhiv({"photos": "net-takoj", "reference": "ref", "anchor": "a.jpg",
                    "confirmed": ["a.jpg"]}, koren=tmp_path)

    prichina = prichina_propuska(arkhiv)

    assert "net-takoj" in prichina, prichina


def test_pustoj_spisok_podtverzhdenyh_ne_znaet_chego_zdat(tmp_path) -> None:
    """Пустой `confirmed` — это «сверять не с чем», а не «всё верно, ноль находок».

    Молча пройти с пустым золотым списком значило бы назвать проверкой то, где сравнение
    не выполнялось ни разу.
    """
    from zolotoj_arhiv import Arhiv, prichina_propuska

    (tmp_path / "data").mkdir()
    (tmp_path / "ref").mkdir()
    arkhiv = Arhiv({"photos": "data", "reference": "ref", "anchor": "a.jpg",
                    "confirmed": []}, koren=tmp_path)

    assert "confirmed" in prichina_propuska(arkhiv)


def test_umolchaniya_prilozheniya_te_zhe() -> None:
    """Золотые числа выверены по настройкам приложения по умолчанию.

    Тест быстрый и выполняется даже без архива — он про то, что константы прогона выше
    не уехали от `config.DEFAULT_*`. Сдвиг умолчаний означает смену рекомендаций, а тогда
    перемерять надо и спецификацию, и список подтверждённых фото, а не подправлять число
    здесь.
    """
    assert (DEFAULT_ENGINE, DEFAULT_MAX_DIM, DEFAULT_THRESHOLD) == (ENGINE, MAX_DIM, THRESHOLD)


@dataclass
class Zapis:
    """Один вызов слияния: что принесли оба детектора и что осталось после слияния.

    `oba_nabora` — лица двух путей БЕЗ слияния. Они нужны, чтобы посчитать, какой была бы
    оценка снимка, если бы правила слияния не существовало: разница и есть цена правила.
    """

    insight: int
    yunet: int
    slito: int
    ramki: tuple[tuple[float, float, float, float], ...]
    oba_nabora: list[Face] = field(repr=False, default_factory=list)

    @property
    def sastalogo(self) -> int:
        """Сколько лиц свернулось: столько рамок было до слияния и не стало после."""
        return self.insight + self.yunet - self.slito


@dataclass
class Sliyanie:
    """Итог измерения правила слияния на всём архиве. Только числа: правило не меняется."""

    foto: int                                  # снимков, где свернулось хотя бы одно лицо
    lica: int                                  # всего свёрнутых лиц по архиву
    snizhennyh: int                            # снимков, где лучшее число после слияния ниже
    max_ponizhenie_pp: int                     # самое сильное понижение, процентных пунктов
    snizhenie: list[tuple[str, int, int]]      # (файл, % со слиянием, % без слияния)
    svoi_vypali: list[str]                     # подтверждённые, сброшенные слиянием из «похоже»
    chuzhie_vypali: list[str]                  # то же среди неподтверждённых
    naydeny_bez_sliyaniya: set[str]            # кучка «похоже», если лица не сливать вовсе
    podmeny: list[tuple[str, int, int, str, int, bool]] = field(default_factory=list)
    # (файл, % без слияния, % со слиянием, детектор лидера, морда лидера, выжил ли лидер)

    @property
    def svoi_snizheny(self) -> list[tuple[str, int, int]]:
        """Подтверждённые фото, у которых слияние съело часть процента."""
        return [x for x in self.snizhenie if x[0] in CONFIRMED]

    @property
    def est_chto_merit(self) -> bool:
        """Правило вообще сработало на архиве? При нуле измерение было бы пустым.

        Нуль означал бы не «слияние безопасно», а то, что на этих снимках два детектора
        ни разу не посмотрели одно и то же лицо по-разному, — и про коллизию на групповых
        фото мы бы не узнали ничего.
        """
        return self.lica > 0


@dataclass
class Zoloto:
    """Всё, что измерил один боевой прогон ядра. Тесты ниже только читают эти числа."""

    engine: Any
    files: list[Path]
    photos: dict[Path, list[Face]]
    stats: ScanStats
    anchor: Face
    kandidaty: int
    ref: Reference
    scores: dict[Path, ScoredPhoto]
    rows: list[ResultRow]
    zapisi: list[Zapis]
    soglasovanie: int
    stats2: ScanStats
    photos2: dict[Path, list[Face]]
    k_modelyam_1: int
    k_modelyam_2: int
    lica_do_sliyaniya: int
    t_gruzka: float
    t_progon: float
    t_progon2: float

    @property
    def naydeny(self) -> set[str]:
        """Имена снимков в кучке «похоже» — по тому же флагу, что ставит сетка."""
        return {r.path.name for r in self.rows if r.matched}

    @property
    def lishnie(self) -> set[str]:
        return self.naydeny - CONFIRMED

    @property
    def propuscheny(self) -> set[str]:
        return CONFIRMED - self.naydeny

    def protsent(self, imya: str) -> int:
        """Похожесть снимка по имени файла — чтобы писать её в сообщении об ошибке."""
        for r in self.rows:
            if r.path.name == imya:
                return r.percent
        return -1

    def sliyanie(self) -> Sliyanie:
        """Разбор правила слияния по архиву: что оно свернуло и чего это стоило.

        Сравниваются два ответа на один вопрос: «как оценивает приложение» и «как
        оценивало бы оба набора лиц, если бы слияния не было». Тот же снимок, тот же
        эталон — разница ровно в одном шаге, и её видно по каждому файлу.
        """
        foto = lica = snizhennyh = 0
        max_dn = 0
        snizhenie: list[tuple[str, int, int]] = []
        svoi_vypali: list[str] = []
        chuzhie_vypali: list[str] = []
        naydeny_bez: set[str] = set()
        podmeny: list[tuple[str, int, int, str, int, bool]] = []
        for put, zapis in zip(self.files, self.zapisi):
            slitoe = self.scores[put]
            oboe = score_photo(zapis.oba_nabora, self.ref)
            if zapis.sastalogo:
                foto += 1
                lica += zapis.sastalogo
            if slitoe.percent < oboe.percent:
                snizhennyh += 1
                max_dn = max(max_dn, oboe.percent - slitoe.percent)
                snizhenie.append((put.name, slitoe.percent, oboe.percent))
            if oboe.percent >= POROG:
                naydeny_bez.add(put.name)
                if slitoe.percent < POROG:
                    (svoi_vypali if put.name in CONFIRMED else chuzhie_vypali).append(put.name)
            # «Подмена лидера» — тот самый схлоп, из-за которого правило тревожили на
            # групповом фото: без слияния отвечало лицо X, а после слияния его рамка
            # исчезает, и отвечает другое (обычно крупнее, но не обязательно нужное).
            if put.name in CONFIRMED and oboe.face is not None and zapis.sastalogo:
                podmeny.append((
                    put.name, oboe.percent, slitoe.percent, oboe.face.detector,
                    round(oboe.face.size), oboe.face.box in zapis.ramki,
                ))
        snizhenie.sort(key=lambda x: (x[2] - x[1], x[2]), reverse=True)
        return Sliyanie(foto, lica, snizhennyh, max_dn, snizhenie, svoi_vypali,
                        chuzhie_vypali, naydeny_bez, podmeny)

    def itog(self) -> str:
        """Сводка прогона числами — отсюда строки уезжают в спецификацию и в отчёт."""
        sli = self.sliyanie()
        naim = sorted((r for r in self.rows if r.matched), key=lambda r: -r.percent)
        svoi = [r.percent for r in naim if r.path.name in CONFIRMED]
        stroke = [
            "", "=" * 78,
            f"БОЕВОЙ ПРОГОН: {DATA} через ядро приложения",
            f"  снимков: {len(self.files)} | прошли модели: {self.stats.scanned} | "
            f"взято из оглавления: {self.stats.cached}",
            f"  не прочитались: {len(self.stats.failures)} | лиц отброшено как непригодные: "
            f"{sum(self.stats.dropped_faces.values())} | починено строк оглавления: "
            f"{self.stats.cache_damage}",
            f"  загрузка моделей: {self.t_gruzka:.1f} с | холодный прогон: "
            f"{self.t_progon:.1f} с ({self.t_progon / max(1, len(self.files)):.2f} с/снимок) | "
            f"второй прогон: {self.t_progon2:.1f} с",
            f"  обращений к моделям: прогон 1 — {self.k_modelyam_1}, "
            f"прогон 2 — {self.k_modelyam_2}",
            f"  лиц: до слияния {self.lica_do_sliyaniya}, после слияния "
            f"{sum(len(v) for v in self.photos.values())}",
            f"  эталон: лиц {self.ref.count} в образе из {self.kandidaty + 1} рассмотренных "
            f"в ref/ (anchor {ANCHOR}, морда {self.anchor.size:.0f} px)",
            f"  похоже на пороге {THRESHOLD:.2f}: {len(naim)} из {len(CONFIRMED)} "
            f"подтверждённых | лишних: {len(self.lishnie)} | "
            f"пропущено: {len(self.propuscheny)}",
            f"  слияние лиц: снимков со свёрнутыми рамками {sli.foto} из {len(self.files)} | "
            f"свёрнуто лиц {sli.lica} | оценка снижена на {sli.snizhennyh} снимках "
            f"(сильнее всего на {sli.max_ponizhenie_pp} п.п.) | ниже порога сбросило: "
            f"своих {len(sli.svoi_vypali)}, чужих {len(sli.chuzhie_vypali)}",
            f"  набор «похоже» без слияния лиц: {len(sli.naydeny_bez_sliyaniya)} фото | "
            f"разница с приложением: {sorted(sli.naydeny_bez_sliyaniya ^ self.naydeny)}",
        ]
        if svoi:
            stroke.append(f"  худшее подтверждённое: {min(svoi)}% | лучшее: {max(svoi)}%")
        else:
            stroke.append("  подтверждённых в кучке «похоже» нет")
        if sli.snizhenie:
            stroke.append("  сильнее всего слияние занизило: " + ", ".join(
                f"{f} {o}% → {s}% (−{o - s})" for f, s, o in sli.snizhenie[:5]))
        stroke += ["", f"  {'файл':16} {'%':>4} {'лиц':>4} {'морда, px':>10}"]
        for r in naim:
            lico = self.scores[r.path].face
            side = f"{lico.size:.0f}" if lico is not None else "—"
            znak = "" if r.path.name in CONFIRMED else "   ← НЕ ПОДТВЕРЖДЁН"
            stroke.append(f"  {r.path.name:16} {r.percent:>4} {r.faces:>4} {side:>10}{znak}")
        stroke.append("\n  слияние и подтверждённые фото (файл, % без слияния → % со слиянием, "
                      "лидер: детектор/морда, выжил ли лидер):")
        for imya, oboe, slitoe, det, side, vyzhil in sli.podmeny:
            stroke.append(f"    {imya:16} {oboe:>3}% → {slitoe:>3}%   лидер {det}/{side} px   "
                          f"{'выжил' if vyzhil else 'СХЛОПНУТ'}")
        stroke.append("=" * 78)
        return "\n".join(stroke)


@pytest.fixture(scope="module")
def zoloto(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Zoloto]:
    """Один полный боевой прогон на весь модуль: холодный скан, эталон, тёплый скан, замеры.

    В фикстуре, а не в тестах, — по прозаичной причине: холодный прогон архива стоит
    минуты, а проверяющих тестов несколько. Фикстура почти ничего не утверждает; она
    измеряет, чтобы падение было видно в конкретном тесте и с конкретным числом в сообщении.
    """
    if NET_ARKHIVA:
        pytest.skip(PRICHINA_PROPUSKA)

    files = find_photos([DATA])
    assert files, f"в {DATA} не нашлось ни одного снимка"
    assert len({p.name for p in files}) == len(files), \
        "в архиве два снимка с одинаковым именем: сравнение по именам было бы нечестным"

    kataloh = tmp_path_factory.mktemp("oglavlenie")
    cache = FaceCache(kataloh / "cache.sqlite3")
    engine = make_engine(ENGINE, yunet_model_path())

    # `merge_faces` — единственная точка, где видно оба набора лиц и их итог: один вызов
    # ровно на один `BothEngine.detect`, то есть на один проход обеих сетей. Обёртка
    # вызывает оригинал без изменений: правило слияния не трогается, оно измеряется.
    zapisi: list[Zapis] = []
    nastoyashchee_sliyanie = obasha.merge_faces

    def zamer(primary: list[Face], secondary: list[Face],
              overlap: float = obasha.OVERLAP) -> list[Face]:
        itog = nastoyashchee_sliyanie(primary, secondary, overlap)
        zapisi.append(Zapis(
            insight=len(primary), yunet=len(secondary), slito=len(itog),
            ramki=tuple(f.box for f in itog), oba_nabora=[*primary, *secondary],
        ))
        return itog

    obasha.merge_faces = zamer
    try:
        t0 = time.perf_counter()
        obespechit_zagruzku(engine)
        t_gruzka = time.perf_counter() - t0

        def progress(sdelano: int, vsego: int, put: Path) -> None:
            if sdelano % 20 == 0 or sdelano == vsego:
                print(f"  разбор {sdelano}/{vsego}: {put.name}", flush=True)

        t0 = time.perf_counter()
        photos, stats = scan_photos(files, engine, cache, max_dim=MAX_DIM,
                                    on_progress=progress)
        t_progon = time.perf_counter() - t0
        k_modelyam_1 = len(zapisi)
        lica_do_sliyaniya = sum(z.insight + z.yunet for z in zapisi[:k_modelyam_1])

        # Эталон: крупное лицо с anchor-снимка плюс похожие на него лица со всех ref/.
        #
        # ВНИМАНИЕ: это НЕ в точности путь приложения с задачи T14. Приложение берёт в
        # эталон всё, что отметил человек, а порог `REF_MIN_SIM` здесь оставлен нарочно:
        # этот тест — про истинность находок, и его второй половине («чужих нет») нужен
        # эталон, в который чужие не попали. Здесь он и отфильтрован.
        #
        # Замер, зафиксированный в docstring ниже: из 20 лиц, найденных в `ref/`, порог
        # 40 % пропускает 11, а остальные 9 — другие люди. То есть на этом архиве отказ
        # по порогу отсекал не ребёнка, а чужих. Это число стоит помнить, прежде чем
        # считать снятый фильтр безвредным.
        anchor: Face | None = None
        kandidaty: list[Face] = []
        ne_otkrylis: list[str] = []
        for put in sorted(REF.iterdir()):
            if put.suffix.lower() not in PHOTO_SUFFIXES:
                continue
            image = load_photo(put, max_dim=MAX_DIM)
            if image is None:
                ne_otkrylis.append(put.name)
                continue
            for lico in engine.detect(image):
                if put.name == ANCHOR:
                    if anchor is None or lico.size > anchor.size:
                        anchor = lico
                elif put.name != ANCHOR:
                    kandidaty.append(lico)
        assert anchor is not None, f"на {ANCHOR} лицо не найдено — проверять не на чём"
        ref = make_reference(anchor, kandidaty)

        scores = {put: score_photo(lica, ref) for put, lica in photos.items()}
        rows = build_rows(photos, [ scores], THRESHOLD)

        # Замеры слияния привязываем к снимкам: один вызов на один файл, в том же порядке,
        # в котором их обрабатывал `scan_photos`.
        soglasovanie = sum(
            1 for put, zapis in zip(files, zapisi[:k_modelyam_1])
            if zapis.ramki == tuple(f.box for f in photos.get(put, ()))
        )

        # Второй прогон по тому же оглавлению — он и есть проверка, что кэш экономит работу.
        zapisi_do = len(zapisi)
        t0 = time.perf_counter()
        photos2, stats2 = scan_photos(files, engine, cache, max_dim=MAX_DIM)
        t_progon2 = time.perf_counter() - t0
        k_modelyam_2 = len(zapisi) - zapisi_do
    finally:
        obasha.merge_faces = nastoyashchee_sliyanie
        cache.close()

    if ne_otkrylis:
        print(f"  эталонные фото, которые не открылись: {', '.join(ne_otkrylis)}")
    izm = Zoloto(
        engine=engine, files=files, photos=photos, stats=stats, anchor=anchor,
        kandidaty=len(kandidaty), ref=ref, scores=scores, rows=rows,
        zapisi=zapisi[:k_modelyam_1], soglasovanie=soglasovanie, stats2=stats2,
        photos2=photos2, k_modelyam_1=k_modelyam_1, k_modelyam_2=k_modelyam_2,
        lica_do_sliyaniya=lica_do_sliyaniya, t_gruzka=t_gruzka, t_progon=t_progon,
        t_progon2=t_progon2,
    )
    print(izm.itog(), flush=True)
    yield izm


# ---------------------------------------------------------------- целостность замера
@pytest.mark.slow
@NADO_ARKHIV
def test_arhiv_deystvitelno_proshel_skvoz_modeli(zoloto: Zoloto) -> None:
    """Прогон обязан быть настоящим: весь архив прошёл через модели, потерь нет.

    Без этой проверки остальные числа бессмысленны: половина папки «не прочиталась»
    выглядела бы как «лишних нет», а оглавление, заполненное предыдущим запуском, — как
    экономия моделей.
    """
    s = zoloto.stats
    assert s.failures == [], f"снимки не дошли до моделей: {s.failures[:3]}"
    assert s.cached == 0, "оглавление уже было заполнено — холодный прогон не состоялся"
    assert s.scanned == len(zoloto.files), \
        f"через модели прошло {s.scanned} из {len(zoloto.files)}"
    assert zoloto.k_modelyam_1 == len(zoloto.files)
    assert sum(s.dropped_faces.values()) == 0, \
        f"лица отброшены как непригодные: {list(s.dropped_faces.items())[:3]}"
    assert s.cache_damage == 0, "оглавление пришлось чинить посреди прогона"


# -------------------------------------------------------------------- 2. эталон
@pytest.mark.slow
@NADO_ARKHIV
def test_etallon_postroen_i_lic_dostatochno(zoloto: Zoloto) -> None:
    """Образ человека собирается от anchor-снимка и набирает не меньше 10 лиц.

    Замер 2026-09-28 на калибровочном прогоне с тем же anchor-файлом принял в образ
    11 лиц из 20 найденных в `ref/`: остальные оказались другими людьми.
    Меньше 10 — значит эталонный путь порвался: `ref/` не открылся, детектор на нём не
    сработал или поехал порог отбора кандидатов `REF_MIN_SIM`. С пустым образом разговор
    о находках начинается не с того места.
    """
    assert zoloto.ref.count >= 10, (
        f"мало эталонных лиц: {zoloto.ref.count} из {zoloto.kandidaty + 1} рассмотренных в ref/")


# ---------------------------------------------------------------- 1. золотой набор
@pytest.mark.slow
@NADO_ARKHIV
def test_vse_podtverzhdennye_naideny_i_lihnih_net(zoloto: Zoloto) -> None:
    """Главное обещание приложения: все 20 подтверждённых фото и ни одного лишнего.

    Делёж на «похоже»/«слабое сходство» берётся из флага `matched`, который ставит
    `build_rows`, — по тому же правилу сетка задачи 14 раскладывает карточки. Отдельно
    сверяем флаг с округлённым процентом: разойдутся — и тест перестанет описывать то,
    что видит человек.
    """
    assert all(r.matched == (r.percent >= POROG) for r in zoloto.rows), \
        "флаг «похоже» разошёлся с числом на карточке"
    assert not zoloto.propuscheny, (
        f"пропущено {len(zoloto.propuscheny)} подтверждённых фото: "
        f"{[(f, zoloto.protsent(f)) for f in sorted(zoloto.propuscheny)]}")
    assert not zoloto.lishnie, (
        f"лишние {len(zoloto.lishnie)} фото сверх подтверждения: "
        f"{[(f, zoloto.protsent(f)) for f in sorted(zoloto.lishnie)]}")
    assert zoloto.naydeny == CONFIRMED


# -------------------------------------------------------------------- 3. оглавление
@pytest.mark.slow
@NADO_ARKHIV
def test_kesh_pomoschaet_vtoroy_progon_bez_modely(zoloto: Zoloto) -> None:
    """Второй прогон той же папки не зовёт модели вовсе — и отвечает тем же числом.

    Экономия должна быть видимой: `scanned == 0`, `cached == столько-то снимков` и,
    главное, ноль обращений к движку. Плюс лица из оглавления дают тот же ответ, что и
    свежий разбор: иначе кэш был бы не ускорением, а тихой порчей результата.
    """
    assert zoloto.k_modelyam_2 == 0, \
        f"второй прогон снова читал фото моделями {zoloto.k_modelyam_2} раз"
    assert zoloto.stats2.scanned == 0
    assert zoloto.stats2.cached == len(zoloto.files)
    assert set(zoloto.photos2) == set(zoloto.photos)
    for put in zoloto.files:
        assert [f.box for f in zoloto.photos2[put]] == [f.box for f in zoloto.photos[put]], \
            f"оглавление вернуло другие рамки на {put.name}"
    te_zhe = build_rows(zoloto.photos2, [
        {put: score_photo(lica, zoloto.ref) for put, lica in zoloto.photos2.items()}], THRESHOLD)
    assert {r.path.name for r in te_zhe if r.matched} == zoloto.naydeny, \
        "результат второго прогона отличается от первого"


# ---------------------------------------------------------------- 4а. одна модель
@pytest.mark.slow
@NADO_ARKHIV
def test_razpoznavanie_v_rezhime_oba_odin_obekt(zoloto: Zoloto) -> None:
    """Режим «Оба» поднимает arcface w600k_r50 один раз, а не дважды по 174 МБ.

    Фиктивы задачи 7 фиксируют договорённость `adopt_recognition()` на подставных
    объектах. Здесь — живой стек: после настоящего `load()` распознаватель внутри
    `FaceAnalysis` и тот, которым юнет строит отпечатки, должны быть буквально одним
    объектом. Две копии — это и двойная память, и тихий риск: два пути считают векторы
    разными экземплярами одной сети, и сходство внутри архива теряет смысл.
    """
    ins = zoloto.engine._insight
    yun = zoloto.engine._yunet
    model_ins = ins.recognition_model
    model_yun = yun._recognition
    assert model_ins is not None, "у InsightEngine нет распознавателя: FaceAnalysis не поднят"
    assert model_yun is not None, "YunetEngine строит отпечатки неизвестно чем"
    assert model_ins is model_yun, (
        "в режиме «Оба» две копии модели распознавания: "
        f"{type(model_ins).__name__} id={id(model_ins)} против id={id(model_yun)}")


# ---------------------------------------------------------------- 4б. слияние лиц
@pytest.mark.slow
@NADO_ARKHIV
def test_sliyanie_lic_na_gruppovyh_foto_merno(zoloto: Zoloto) -> None:
    """Правило слияния на реальном архиве: сколько рамок сворачивает и чего это стоит.

    Коллизия правила известна: крупная рамка, лёгшая поверх двух непересекающихся мелких,
    оставляет одну — то есть двух разных людей на групповом фото сервис покажет как одного.
    Правило не трогаем, но снимаем числа, которых раньше не было: на скольких снимках оно
    сработало и что это сделало с оценкой. Обязательным остаётся одно — слияние не имеет
    права выбрасывать из «похоже» подтверждённое фото, ради которого всё и строится.

    Чужие фото сбросить ниже порога правило вправе (меньше мусора человеку), и это тоже
    видно числом, но не как ошибка, а как факт.
    """
    assert zoloto.soglasovanie == len(zoloto.files), (
        f"замеры слияния сопоставились лишь с {zoloto.soglasovanie} снимками из "
        f"{len(zoloto.files)} — привязка вызовов к файлам сломалась, числам верить нельзя")
    sli = zoloto.sliyanie()
    assert sli.est_chto_merit, \
        "на архиве не свернулось ни одной рамки: измерение пустое, правило не проверено"
    assert not sli.svoi_vypali, (
        f"слияние сбросило подтверждённые фото ниже порога: {sli.svoi_vypali}")
