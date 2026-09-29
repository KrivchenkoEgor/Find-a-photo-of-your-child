"""Отчёты: report.txt для человека и CSV для Excel.

Тесты синтетические: лица — numpy-векторы, пути — фиктивные, файлы пишутся только
в `tmp_path`. Модуль обязан оставаться лёгким на импорт: он не тянет ни Qt, ни
OpenCV, ни insightface, поэтому проверки порядка и группировки здесь недорогие.

За чем следит этот файл:
 — порядок строк задаёт отчёт, а не разбор: по убыванию похожести, при равных числах
   по имени файла (`find_photos` сортирует по полному пути, и снимки одной папки шли
   бы подряд);
 — текст потери сворачивается в одну строку и не рвёт вёрстку;
 — потери названы числами, а не промолчаны, и нуль тоже написан прямо;
 — CSV остаётся плоским и открывается в Excel с кириллицей.
"""

import csv
from pathlib import Path
from typing import Any

import numpy as np

from core.engine import Face
from core.matcher import ScoredPhoto
from core.report import ResultRow, Sovpadenie, build_rows, write_csv, write_report

NASTROJKI = {"Режим распознавания": "Оба",
             "Качество разбора": "внимательное",
             "Насколько фото должно быть похоже": "38%"}


def lico(box: tuple[float, float, float, float] = (0.0, 0.0, 50.0, 50.0)) -> Face:
    return Face(box=box, landmarks=None,
                embedding=np.zeros(512, dtype=np.float32), detector="insight")


def otchet(tmp_path: Path, **dannye: Any) -> str:
    """`write_report` с заполненными обязательными полями; возвращает текст отчёта."""
    danny: dict[str, Any] = {"settings": NASTROJKI, "rows": [], "failures": [],
                             "copied": []}
    danny.update(dannye)
    path = write_report(tmp_path / "report.txt", **danny)      # type: ignore[arg-type]
    return path.read_text(encoding="utf-8")


# -------------------------------------------------------------------- строки


def test_priznak_i_kuchka_sovpadayut_na_granice() -> None:
    """37.99% округляется в 38: и галочка, и кучка должны считать его похожим."""
    scores = {Path("/g.jpg"): ScoredPhoto(38, 0.3799, lico())}
    row = build_rows({Path("/g.jpg"): [lico()]}, [ scores], 0.38)[0]
    assert row.percent == 38 and row.matched is True


def test_stroki_soderzhat_puti_percent_i_priznak() -> None:
    photos = {Path("/a.jpg"): [lico()], Path("/b.jpg"): []}
    scores = {Path("/a.jpg"): ScoredPhoto(62, 0.62, lico()),
              Path("/b.jpg"): ScoredPhoto(0, 0.0, None)}
    rows = build_rows(photos, [ scores], threshold=0.38)
    assert [r.percent for r in rows] == [62, 0]
    assert rows[0].matched is True and rows[1].matched is False
    assert rows[0].box == (0.0, 0.0, 50.0, 50.0)


def test_snimok_bez_ocenki_ne_propadaet_iz_otcheta() -> None:
    """Фото разобрано и лежит в `photos`, но оценки не получило — строка обязана быть.

    Иначе отчёт соврёт ровно там, где вызывающий код допустил промах: снимок исчезнет
    из документа бесследно, и «похожих нет» станет неотличимо от «мы его потеряли».
    """
    rows = build_rows({Path("/a.jpg"): [lico()], Path("/c.jpg"): [lico()]}, [
                      {Path("/a.jpg"): ScoredPhoto(62, 0.62, lico())}], 0.38)
    assert [r.path.name for r in rows] == ["a.jpg", "c.jpg"]
    assert rows[1].percent == 0 and rows[1].matched is False and rows[1].faces == 1


def test_stroka_neset_vse_lica_snimka_v_poryadke_razbora() -> None:
    """Сколько лиц нашёл разбор — часть строки результата, а не чужая память окна.

    Превью в сетке рисовало рамку только по совпавшему лицу, а открытое окно — по всем,
    и человек читал одиночную рамку как «приложение не увидело ребёнка рядом со
    взрослым». Ответ «какие лица есть в кадре» обязан приехать вместе с числом похожести:
    тогда и карточка, и окно показывают одно и то же число рамок.

    Порядок — порядок разбора, с единицы, иначе номер лица на карточке не сойдётся с
    подписью окна «лицо 2 из 3».
    """
    a = lico((0.0, 0.0, 50.0, 50.0))
    b = lico((100.0, 100.0, 150.0, 150.0))
    c = lico((200.0, 0.0, 260.0, 70.0))
    row = build_rows({Path("/a.jpg"): [a, b, c]}, [
                     {Path("/a.jpg"): ScoredPhoto(62, 0.62, b)}], 0.38)[0]
    assert row.vse_lica == (a.box, b.box, c.box)
    assert row.lice == 2, "совпавшее лицо — его позиция в списке разбора"
    assert row.box == b.box


def test_bez_sovpadenija_litsa_snimka_vse_ravno_peredajutsja() -> None:
    """«В кадре пять лиц, ни одно не похоже» — ответ, а не пустота.

    Снимок без оценки остаётся в сетке с нулём, и рамки его лиц человек должен видеть:
    иначе карточка с «0 %» выглядит так, будто фото вообще не разбирали.
    """
    lica = [lico((0.0, 0.0, 50.0, 50.0)), lico((60.0, 60.0, 110.0, 110.0))]
    row = build_rows({Path("/a.jpg"): lica}, [ {}], 0.38)[0]
    assert row.vse_lica == (lica[0].box, lica[1].box)
    assert row.lice == 0 and row.box is None


def test_snimok_bez_lica_ne_imaet_spiska_ramok() -> None:
    """Где нет лиц, там нет и рамок: пустой список — честный ответ, а не забытый."""
    row = build_rows({Path("/a.jpg"): []}, [ {Path("/a.jpg"): ScoredPhoto(0, 0.0, None)}], 0.38)[0]
    assert row.vse_lica == () and row.lice == 0 and row.faces == 0


def test_stroka_neset_sovpadeniya_po_kazhdomu_cheloveku() -> None:
    """В одном кадре могут быть два разных человека из поиска — и это должно быть видно.

    Пользователь ищет ребёнка и маму. Прежний эталон держал их смешанным списком из 16
    лиц и один общий максимум: «похоже на 61 %» не отвечало на вопрос, КТО в кадре, а
    сильное совпадение одного человека перекрывало слабое у другого.
    """
    a = lico((0.0, 0.0, 50.0, 50.0))
    b = lico((60.0, 60.0, 110.0, 110.0))
    malysh = {Path("/a.jpg"): ScoredPhoto(61, 0.61, a, po_licam=((1, 61), (2, 12)))}
    mama = {Path("/a.jpg"): ScoredPhoto(45, 0.45, b, po_licam=((1, 15), (2, 45)))}
    row = build_rows({Path("/a.jpg"): [a, b]}, [malysh, mama], 0.38)[0]
    assert row.percent == 61, "число на карточке — лучший из людей, а не первый в списке"
    assert row.chelovek == 1 and row.lice == 1
    assert row.sovpadeniya == (Sovpadenie(1, 1, 61), Sovpadenie(2, 2, 45))


def test_porog_reshaet_koe_iz_lic_popadaet_v_sovpadeniya() -> None:
    """В список совпадений попадает только то, что не ниже ползунка, — по каждому человеку.

    Ползунок один на всех: поднимешь выше 45 % — и мама с этого снимка исчезнет, а ребёнок
    останется. Делёж берётся из того же округлённого числа, что человек видит на карточке.
    """
    a = lico((0.0, 0.0, 50.0, 50.0))
    b = lico((60.0, 60.0, 110.0, 110.0))
    malysh = {Path("/a.jpg"): ScoredPhoto(61, 0.61, a, po_licam=((1, 61), (2, 12)))}
    mama = {Path("/a.jpg"): ScoredPhoto(45, 0.45, b, po_licam=((1, 15), (2, 45)))}
    photos = {Path("/a.jpg"): [a, b]}
    assert build_rows(photos, [malysh, mama], 0.38)[0].sovpadeniya == \
        (Sovpadenie(1, 1, 61), Sovpadenie(2, 2, 45))
    assert build_rows(photos, [malysh, mama], 0.50)[0].sovpadeniya == (Sovpadenie(1, 1, 61),)
    assert build_rows(photos, [malysh, mama], 0.70)[0].sovpadeniya == ()


def test_dva_sovpadenija_odnogo_lica_ostajutsya_dvumya_zapismi() -> None:
    """Лицо, похожее на двух людей сразу, — два факта, а не один усреднённый.

    Пока это только данные: рамку на таком лице рисуют одну (лицо-то одно), а вот
    «кто именно» человеку обязаны сказать и в карточке, и в отчёте.
    """
    a = lico((0.0, 0.0, 50.0, 50.0))
    malysh = {Path("/a.jpg"): ScoredPhoto(61, 0.61, a, po_licam=((1, 61),))}
    mama = {Path("/a.jpg"): ScoredPhoto(58, 0.58, a, po_licam=((1, 58),))}
    row = build_rows({Path("/a.jpg"): [a]}, [malysh, mama], 0.38)[0]
    assert row.sovpadeniya == (Sovpadenie(1, 1, 61), Sovpadenie(2, 1, 58))
    assert row.percent == 61 and row.chelovek == 1


def test_pustoj_spisk_ljudej_dajet_stroku_bez_chisla_no_ne_bez_snimka() -> None:
    """Ни одного человека в поиске — снимок всё равно попадает в строку с нулём.

    Молча исчезнуть из отчёта фото не имеет права ни при каких промахах вызывающего кода
    (см. `test_snimok_bez_ocenki_ne_propadaet_iz_otcheta`).
    """
    row = build_rows({Path("/a.jpg"): [lico()]}, [], 0.38)[0]
    assert row.percent == 0 and row.sovpadeniya == () and row.chelovek == 0
    assert row.faces == 1 and row.vse_lica and row.matched is False


def test_chelovek_bez_etogo_snimka_ne_sdvigaet_nomerov() -> None:
    """Номер человека — позиция в списке поиска, а не индекс в своём словаре.

    Если человек ещё не разбирал этот файл (промах вызывающего или снятый с экрана
    набор), его номер не имеет права съехать на чужого: по этому числу карточка называет
    имя.
    """
    a = lico()
    malysh = {Path("/a.jpg"): ScoredPhoto(61, 0.61, a, po_licam=((1, 61),))}
    vtoroy: dict = {}
    tretiy = {Path("/a.jpg"): ScoredPhoto(40, 0.40, a, po_licam=((1, 40),))}
    row = build_rows({Path("/a.jpg"): [a]}, [malysh, vtoroy, tretiy], 0.38)[0]
    assert row.sovpadeniya == (Sovpadenie(1, 1, 61), Sovpadenie(3, 1, 40))
    assert row.chelovek == 1


def test_poryadok_po_imeni_snimka_a_ne_po_papke() -> None:
    """При равных числах имена идут по алфавиту, даже если снимки из разных папок."""
    iz_b = Path("/архив/B/aaa.jpg")        # по полному пути — второй
    iz_a = Path("/архив/A/zzz.jpg")        # по полному пути — первый
    scores = {iz_b: ScoredPhoto(50, 0.5, lico()), iz_a: ScoredPhoto(50, 0.5, lico())}
    rows = build_rows({iz_b: [lico()], iz_a: [lico()]}, [ scores], 0.38)
    assert [r.path.name for r in rows] == ["aaa.jpg", "zzz.jpg"]


def test_pohozhest_pervee_imeni() -> None:
    """Алфавит разрешает ничью при равных числах, но не спорит с похожестью."""
    a = Path("/y/aa.jpg")
    b = Path("/a/zz.jpg")
    scores = {a: ScoredPhoto(20, 0.2, lico()), b: ScoredPhoto(70, 0.7, None)}
    rows = build_rows({a: [lico()], b: []}, [ scores], 0.38)
    assert [r.percent for r in rows] == [70, 20]


def test_registr_imeni_ne_ryvjot_poryadok() -> None:
    """IMG_10.JPG и img_9.JPG из одного архива: порядок по имени без учёта регистра."""
    a = Path("/x/IMG_10.JPG")
    b = Path("/x/img_9.JPG")
    scores = {a: ScoredPhoto(50, 0.5, None), b: ScoredPhoto(50, 0.5, None)}
    rows = build_rows({a: [], b: []}, [ scores], 0.38)
    assert [r.path.name for r in rows] == ["IMG_10.JPG", "img_9.JPG"]


# ---------------------------------------------------------------- report.txt


def test_otchet_chitaetsya_i_soderzhit_nastrojki(tmp_path: Path) -> None:
    rows = build_rows({Path("/a.jpg"): [lico()]}, [
                      {Path("/a.jpg"): ScoredPhoto(62, 0.62, lico())}], 0.38)
    path = write_report(tmp_path / "report.txt",
                        settings=NASTROJKI,
                        rows=rows, failures=[("/битое.jpg", "файл не читается")],
                        copied=[(Path("/a.jpg"), tmp_path / "a.jpg")])
    text = path.read_text(encoding="utf-8")
    assert "Оба" in text and "38%" in text and "битое" in text and "a.jpg" in text


def test_otchet_peresortivaet_stroki_sam(tmp_path: Path) -> None:
    """Порядок — забота отчёта: строки пришли перемешанными, а печатаются по правилам."""
    iz_b = Path("/архив/B/aaa.jpg")
    iz_a = Path("/архив/A/zzz.jpg")
    rows = [ResultRow(iz_a, 50, 1, None, False), ResultRow(iz_b, 50, 1, None, False)]
    text = otchet(tmp_path, rows=rows)
    assert text.index("aaa.jpg") < text.index("zzz.jpg")


def test_odinakovaja_prichina_daeet_odin_zagolovok(tmp_path: Path) -> None:
    """Двадцать «файл не читается» — это один заголовок, а не двадцать строк.

    Pillow вставляет имя файла в текст диагностики, поэтому группировка режет причину
    по русской приставке: она устойчива, а хвост от библиотеки бывает всякий.
    """
    failures = [(f"/архив/IMG_{i}.HEIC",
                 f"файл не читается: UnidentifiedImageError: cannot identify "
                 f"image file 'IMG_{i}.HEIC'") for i in range(20)]
    text = otchet(tmp_path, failures=failures)
    zagolovki = [ln for ln in text.splitlines() if "файл не читается" in ln]
    assert len(zagolovki) == 1
    assert zagolovki[0].strip() == "файл не читается — файлов: 20"
    primer = [ln for ln in text.splitlines() if "IMG_0.HEIC" in ln]
    assert len(primer) == 1                            # имя показано ровно один раз
    assert "IMG_19.HEIC" not in text                   # остальные свёрнуты в счётчик
    assert "и ещё файлов: 17" in text


def test_drugoj_hvost_prichiny_ostaetsja_v_toj_zhe_seme(tmp_path: Path) -> None:
    """Тот же предлог, но другой хвост — та же семья: заголовок остаётся одним."""
    failures = [("/a.jpg", "файл не читается: UnidentifiedImageError: формат не опознан"),
                ("/b.jpg", "файл не читается: OSError: поток обрезан в 11 байтах"),
                ("/c.jpg", "файл недоступен: [Errno 2] No such file or directory: '/c.jpg'")]
    text = otchet(tmp_path, failures=failures)
    chitaetsya = [ln for ln in text.splitlines() if ln.strip().startswith("файл не читается")]
    nedostupen = [ln for ln in text.splitlines() if ln.strip().startswith("файл недоступен")]
    assert len(chitaetsya) == 1 and len(nedostupen) == 1
    assert "формат не опознан" in text and "поток обрезан" in text
    assert text.count("в 11 байтах") == 1
    assert "No such file or directory" in text          # хвост виден целиком
    assert text.count("c.jpg") == 1                     # имя файла — только в примерах


def test_mnogostrochnaja_prichina_ne_rvyot_vertyku(tmp_path: Path) -> None:
    """Причина с переводами строк обязана свернуться в одну строку отчёта.

    Хвост приходит текстом исключения библиотеки, и в нём бывает трейсировка: без
    сворачивания report.txt разъезжается, а в CSV такая строка разорвала бы запись.
    """
    mnogostrochnaja = ("файл не читается: UnidentifiedImageError: cannot identify "
                       "image file '/x/y.jpg'\nTraceback (most recent call last):\n"
                       "  File \"decoder.py\", line 1\n    raise err\n"
                       "OSError: поток оборвался")
    text = otchet(tmp_path, failures=[("/x/y.jpg", mnogostrochnaja)])
    for ln in text.splitlines():
        assert not ln.strip().startswith('File "decoder.py"')
        assert not ln.strip().startswith("raise err")
    odna = [ln for ln in text.splitlines() if "UnidentifiedImageError" in ln]
    assert len(odna) == 1
    assert "Traceback" in odna[0] and "поток оборвался" in odna[0]


def test_prichina_bez_puti_sohranyaet_tochki(tmp_path: Path) -> None:
    """Потеря без пути обязана печатать свою причину точками вперёд.

    `Path("")` сворачивается в `.`, а `.` — самая жадная подстрока в тексте: сняв её,
    `_prichina` съедала каждую точку в причине («OSError. Ошибка 28.» → «OSError
    Ошибка 28») и оставляла в примерах пустое имя. Сейчас до этого не доходит:
    `worker` всегда зовёт с настоящим путём, и `copy_photos` причину тоже отдаёт
    обязательно и с настоящим файлом. Страховка остаётся нужной, пока любой
    вызывающий может собрать пару потери руками, — `("", "копирование не
    удалось: …")` выглядит как естественная запись для того, у кого файла уже нет.
    """
    text = otchet(tmp_path, failures=[("", "копирование не удалось: OSError. Ошибка 28.")])
    assert "копирование не удалось — файлов: 1" in text
    assert "OSError. Ошибка 28" in text                  # точка внутри причины цела
    assert "OSError Ошибка" not in text                  # она не склеилась в одно слово
    assert "примеры: \n" not in text                     # пустого имени не показываем
    assert "и ещё файлов: 1" in text                     # число потерь осталось правдой


def test_pustoy_put_ne_vypolnyaet_nastoyashchie_primery(tmp_path: Path) -> None:
    """Смешанная пачка: у одной потери пути нет, а примеры берутся только настоящие.

    Раздел «Не обработано» читают глазами и идут с ним в поддержку, поэтому пустая
    строка в примерах — это не «нет данных», а вид сломанного отчёта.
    """
    odinakovye = "копирование не удалось: OSError. Ошибка 28."
    text = otchet(tmp_path, failures=[("", odinakovye),
                                      ("/архив/b.jpg", odinakovye)])
    assert "примеры: b.jpg" in text
    assert "копирование не удалось — файлов: 2" in text
    assert "и ещё файлов: 1" in text


def test_dlinnaja_diagnostika_obryvaetsja(tmp_path: Path) -> None:
    """Трейсировка на три экрана в отчёте не помещается: обрываем и ставим многоточие."""
    text = otchet(tmp_path, failures=[("/a.jpg", "ошибка распознавания: " + "слово " * 100)])
    odna = [ln for ln in text.splitlines() if "слово" in ln]
    assert len(odna) == 1 and "…" in odna[0] and len(odna[0]) < 300


def test_prichina_bez_hvosta_gruppiruetsja_celikom(tmp_path: Path) -> None:
    """Короткая причина без двоеточия — тоже одна группа, а не четыре строки."""
    text = otchet(tmp_path, failures=[(f"/p/{i}.jpg", "файл не читается")
                                      for i in range(4)])
    zagolovki = [ln for ln in text.splitlines()
                 if ln.strip().startswith("файл не читается")]
    assert len(zagolovki) == 1
    assert zagolovki[0].strip().endswith("файлов: 4")


def test_poteri_nazvany_chislami(tmp_path: Path) -> None:
    """Отчёт обязан назвать потери: молчаливый ноль хуже ошибки.

    Родитель смотрит на «похожих: 0» и не может отличить «ребёнка в архиве нет» от
    «приложение потеряло половину снимков». Здесь видны все три числа.
    """
    failures = [("/bitoe1.jpg", "файл не читается: формат не опознан"),
                ("/bitoe2.jpg", "файл не читается: формат не опознан")]
    text = otchet(tmp_path, failures=failures,
                  dropped_faces={Path("/a.jpg"): 2, Path("/b.jpg"): 1},
                  cache_damage=5)
    assert "файлов не обработано: 2" in text
    assert "лиц отброшено: 3" in text
    assert "снимков с отброшенным лицом: 2" in text
    assert "строк оглавления удалено: 5" in text


def test_net_poterej_i_oni_skazany_pryamo(tmp_path: Path) -> None:
    """Нули тоже пишутся: отсутствие строк неотличимо от забытого раздела."""
    text = otchet(tmp_path, rows=[ResultRow(Path("/a.jpg"), 62, 1, None, True)],
                  dropped_faces={}, cache_damage=0)
    assert "файлов не обработано: 0" in text
    assert "лиц отброшено: 0" in text
    assert "строк оглавления удалено: 0" in text
    assert "потерь нет" in text


def test_chislo_ne_peredano_v_otchete_etoviden(tmp_path: Path) -> None:
    """«Не передали» и «ноль» — разные утверждения, и отчёт не имеет права их смешивать.

    Ложный ноль хуже ошибки ровно в том месте, ради которого раздел потерь и заведён:
    человек прочитает «лиц отброшено: 0» и успокоится, хотя приложение просто ничего
    об этом не знает.
    """
    text = otchet(tmp_path)
    assert "лиц отброшено: нет данных" in text
    assert "снимков с отброшенным лицом: нет данных" in text
    assert "строк оглавления удалено: нет данных" in text
    assert "потерь нет" not in text                      # право на эту строку — за нулями


def test_itog_ne_nazyvaet_bezlicevie_snimki_licami(tmp_path: Path) -> None:
    """rows содержит и снимки без лиц — «всего фото с лицами: N» была бы ложь.

    Счётчики разделены: сколько фото попало в отчёт и сколько из них с лицами.
    """
    rows = [ResultRow(Path("/a.jpg"), 62, 2, None, True),
            ResultRow(Path("/b.jpg"), 0, 0, None, False)]
    text = otchet(tmp_path, rows=rows)
    assert "фото в отчёте: 2" in text
    assert "фото с лицами: 1" in text
    assert "похоже: 1" in text
    assert "слабое сходство: 1" in text


def test_odnoimennye_snimki_pokazany_s_papkoj(tmp_path: Path) -> None:
    """Порядок по имени ставит два IMG_1.JPG рядом — без папки их не различить."""
    a = Path("/архив/Отпуск/IMG_1.JPG")
    b = Path("/архив/Друг/IMG_1.JPG")
    rows = [ResultRow(a, 50, 1, None, False), ResultRow(b, 50, 1, None, False)]
    text = otchet(tmp_path, rows=rows)
    linii = [ln for ln in text.splitlines() if "IMG_1.JPG" in ln]
    assert len(linii) == 2
    assert "Друг" in linii[0] and "Отпуск" in linii[1]


def test_skopirovannye_faily_perechisleny(tmp_path: Path) -> None:
    text = otchet(tmp_path, copied=[(Path("/a.jpg"), tmp_path / "a_1.jpg")])
    assert "скопировано: 1" in text
    assert "a.jpg" in text and "a_1.jpg" in text


# ---------------------------------------------------------------- строка эталона (T15)
#
# Человек отмечает несколько лиц, и среди них может быть другой человек (брат,
# взрослый). Одно число «45 %» не отвечает на вопрос «кто на снимке»: ребёнок
# неуверенно или совсем другой. Отвечает номер строки эталона, и он обязан доехать до
# отчёта — колонкой, которая появляется ТОЛЬКО когда выбранных лиц больше одного: при
# одном лице в колонке у всех строк стояло бы одно и то же число, и это шум вместо
# ответа.


def test_stroka_ponosit_nomer_stroki_etalona() -> None:
    """`build_rows` переносит номер строки из `ScoredPhoto`, а не теряет его по пути:
    всё, что виден ответ, зависит от этой пересылки."""
    scores = {Path("/a.jpg"): ScoredPhoto(62, 0.62, lico(), 2),
              Path("/b.jpg"): ScoredPhoto(50, 0.5, lico(), 1),
              Path("/c.jpg"): ScoredPhoto(0, 0.0, None)}
    rows = build_rows({Path("/a.jpg"): [lico()], Path("/b.jpg"): [lico()],
                       Path("/c.jpg"): []}, [ scores], 0.38)
    po_imenam = {r.path.name: r for r in rows}
    assert po_imenam["a.jpg"].ref_lico == 2
    assert po_imenam["b.jpg"].ref_lico == 1
    # снимок без оценки и лицо без ответа на вопрос «по какому лицу» — это 0, а не 1
    assert po_imenam["c.jpg"].ref_lico == 0


def test_otchet_pishet_kogo_nashli_kogda_lydej_dvoe(tmp_path: Path) -> None:
    """В отчёте видно, КТО из искомых людей на этом снимке и с каким числом.

    Второй ассерт — про то, что имя доехало до нужной строки, а не появилось где
    попало: без сверки с именем файла отчёт мог подписать не тот снимок, и человек
    убрал бы из папки результатов верный.
    """
    malysh = {Path("/arhiv/svoy.jpg"): ScoredPhoto(70, 0.7, lico(), po_licam=((1, 70),)),
              Path("/arhiv/chuzhoe.jpg"): ScoredPhoto(20, 0.2, lico(), po_licam=((1, 20),))}
    mama = {Path("/arhiv/svoy.jpg"): ScoredPhoto(30, 0.3, lico(), po_licam=((1, 30),)),
            Path("/arhiv/chuzhoe.jpg"): ScoredPhoto(45, 0.45, lico(), po_licam=((1, 45),))}
    rows = build_rows({Path("/arhiv/svoy.jpg"): [lico()],
                       Path("/arhiv/chuzhoe.jpg"): [lico()]}, [malysh, mama], 0.38)
    text = otchet(tmp_path, rows=rows, imena_ljudj=["ребёнок", "мама"])
    linii = {ln.split()[1]: ln for ln in text.splitlines() if ".jpg" in ln and "%" in ln}
    assert "ребёнок 70 %" in linii["svoy.jpg"], linii["svoy.jpg"]
    assert "мама" not in linii["svoy.jpg"], "мама не дотянула до порога, но попала в отчёт"
    assert "мама 45 %" in linii["chuzhoe.jpg"], linii["chuzhoe.jpg"]
    assert "кто из искомых людей найден" in text, "колонка не объяснена заголовком"


def test_otchet_bez_vtorogo_lica_ne_pishet_kolonku(tmp_path: Path) -> None:
    """При эталоне из одного лица колонки нет вовсе — ни пустой, ни с однообразным
    «главное лицо» у каждой строки: это шум, который человек читает как ответ, хотя
    спрашивать было не о чем."""
    rows = [ResultRow(Path("/a.jpg"), 62, 1, None, True, sovpadeniya=(Sovpadenie(1, 1, 62),))]
    odin = otchet(tmp_path, rows=rows, imena_ljudj=["ребёнок"])
    nichego = otchet(tmp_path, rows=rows)
    for text in (odin, nichego):
        assert "ребёнок" not in text
        assert "кто из искомых людей найден" not in text
    assert "62" in odin, "проценты из-за колонки не должны исчезнуть вовсе"


def test_otchet_govorit_pro_neizvestnogo_cheloveka_vmesto_pustoty(tmp_path: Path) -> None:
    """Строка без совпадений обязана быть названа словом, а не пустым местом.

    Пусто в колонке человек прочитает как «отчёт испортился», а «никто из искомых не
    найден» — как «на этом фото наших нет». Молчаливый ноль хуже ошибки — то же
    правило, что держит раздел потерь. Номер старше списка людей тоже не имеет права
    придумать себе имя.
    """
    rows = [ResultRow(Path("/a.jpg"), 0, 0, None, False),
            ResultRow(Path("/b.jpg"), 40, 1, None, True,
                      sovpadeniya=(Sovpadenie(9, 1, 40),))]
    text = otchet(tmp_path, rows=rows, imena_ljudj=["ребёнок", "мама"])
    assert text.count("никто из искомых не найден") == 2, text


def test_csv_stavit_kolonku_v_konec_i_ne_sdigaet_poryadok(tmp_path: Path) -> None:
    """Новый столбец встает последним: прежние пять обязаны остаться на своих местах.

    Человек держит старые таблицы с этого места, и сдвинутая молча колонка — это
    «рамка» в столбце «похоже»: разбор вслепую, где виноваты данные, а не экран.
    """
    a = Path("/arhiv/a.jpg")
    b = Path("/arhiv/b.jpg")
    rows = [ResultRow(a, 70, 1, (1.0, 2.0, 30.0, 40.0), True,
                      sovpadeniya=(Sovpadenie(1, 1, 70),)),
            ResultRow(b, 45, 1, None, True, sovpadeniya=(Sovpadenie(2, 1, 45),))]
    with open(write_csv(tmp_path / "r.csv", rows, imena_ljudj=["ребёнок", "мама"]),
              encoding="utf-8-sig", newline="") as fh:
        got = list(csv.reader(fh))
    assert got[0] == ["файл", "процент", "лиц", "похоже", "рамка",
                      "кто найден и насколько похож"]
    assert all(len(zap) == 6 for zap in got), got
    assert got[1][5] == "ребёнок 70 %" and got[2][5] == "мама 45 %"


def test_csv_bez_kolonki_ostajetsja_kak_byl(tmp_path: Path) -> None:
    """Берём тот же вызов без имён: таблица обязана выйти ровно прежней ширины, иначе
    любой, кто открыл старый CSV, увидел бы сдвиг."""
    rows = [ResultRow(Path("/a.jpg"), 62, 1, None, True, ref_lico=1)]
    with open(write_csv(tmp_path / "r.csv", rows), encoding="utf-8-sig", newline="") as fh:
        got = list(csv.reader(fh))
    assert got[0] == ["файл", "процент", "лиц", "похоже", "рамка"]
    assert all(len(zap) == 5 for zap in got), got


def test_csv_nazyvaet_neizvestnogo_cheloveka(tmp_path: Path) -> None:
    """В машинной таблице пропуск так же назван словом, а не пустой ячейкой: пустую
    ячейку в Excel читают как «не заполнили», а не как «наших на снимке нет»."""
    rows = [ResultRow(Path("/a.jpg"), 0, 0, None, False)]
    with open(write_csv(tmp_path / "r.csv", rows, imena_ljudj=["ребёнок", "мама"]),
              encoding="utf-8-sig", newline="") as fh:
        got = list(csv.reader(fh))
    assert got[1][5] == "никто из искомых не найден"


# ----------------------------------------------------------------------- csv


def test_csv_s_kodirovkoj_dlja_excel(tmp_path: Path) -> None:
    rows = build_rows({Path("/a.jpg"): [lico()]}, [
                      {Path("/a.jpg"): ScoredPhoto(62, 0.62, lico())}], 0.38)
    path = write_csv(tmp_path / "r.csv", rows)
    assert path.read_bytes()[:3] == b"\xef\xbb\xbf"    # BOM: Excel откроет кириллицу
    with open(path, encoding="utf-8-sig") as fh:
        got = list(csv.DictReader(fh))
    assert got[0]["процент"] == "62" and got[0]["файл"].endswith("a.jpg")


def test_csv_ne_losest_imena_i_derzhit_zapis(tmp_path: Path) -> None:
    """Имя файла с переводом строки — данные, а не поломка: `csv` его экранирует.

    «Исправить» такое имя значило бы потерять путь: запись обязана остаться одной и
    указывать на тот же снимок.
    """
    p = Path("/архив/кривое\nимя.jpg")
    path = write_csv(tmp_path / "r.csv", [ResultRow(p, 44, 1, None, True)])
    with open(path, encoding="utf-8-sig", newline="") as fh:
        got = list(csv.reader(fh))
    assert len(got) == 2
    assert Path(got[1][0]) == p


def test_csv_ploskij_i_mashinnyj(tmp_path: Path) -> None:
    """CSV — машинная таблица: строка на снимок, без заголовков-группировок.

    Порядок тот же, что в report.txt, чтобы две части отчёта не начинались с разных
    фото и не заставляли сверять их глазами.
    """
    iz_b = Path("/архив/B/aaa.jpg")
    iz_a = Path("/архив/A/zzz.jpg")
    rows = [ResultRow(iz_a, 40, 1, (1.0, 2.0, 30.0, 40.0), True),
            ResultRow(iz_b, 40, 1, None, True)]
    with open(write_csv(tmp_path / "r.csv", rows), encoding="utf-8-sig", newline="") as fh:
        got = list(csv.reader(fh))
    assert got[0] == ["файл", "процент", "лиц", "похоже", "рамка"]
    assert [row[0] for row in got[1:]] == [str(iz_b), str(iz_a)]
    assert got[1][4] == "" and got[2][4] == "1;2;30;40"
    assert all(len(row) == 5 for row in got)
