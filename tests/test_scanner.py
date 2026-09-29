"""Поиск фото по папкам: рекурсия, форматы, служебные каталоги, штамп файла."""

from pathlib import Path

from core.scanner import file_stamp, find_photos


def _make(root: Path, *names: str) -> None:
    for n in names:
        p = root / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")


def test_hodit_rekursivno_i_filtraet_formaty(tmp_path) -> None:
    _make(tmp_path, "a.jpg", "b.HEIC", "c.txt", "d.mp4", "вложенно/e.png", "вложенно/f.webp")
    got = [p.relative_to(tmp_path).as_posix() for p in find_photos([tmp_path])]
    assert sorted(got) == ["a.jpg", "b.HEIC", "вложенно/e.png", "вложенно/f.webp"]


def test_propuskaet_sluzhebnye_papki_i_skrytye_fayly(tmp_path) -> None:
    _make(tmp_path, ".hidden.jpg", "Кэш/normalno.jpg", ".Trash/y.jpg",
          "__pycache__/z.jpg", "Photos/.DS_Store", "Photos/ok.jpg")
    got = {p.name for p in find_photos([tmp_path])}
    assert got == {"normalno.jpg", "ok.jpg"}


def test_propuskaet_nesushchestvuyushchuyu_papku(tmp_path) -> None:
    _make(tmp_path, "k.jpg")
    other = tmp_path / "другая"
    assert find_photos([tmp_path, other]) == [tmp_path / "k.jpg"]


def test_poryadok_deteminiran_i_ne_zavisit_ot_scandir(tmp_path) -> None:
    """Порядок нужен Task 9 и Task 15: прогресс и отчёт должны воспроизводиться.

    Требование — детерминированность (одинаковый результат всегда и при повторном
    вызове), а не алфавит имён: сортировка по полному пути держит файлы одной папки
    рядом, и так честнее для отчёта.
    """
    _make(tmp_path, "b.jpg", "a.jpg", "c/d.jpg", "c/a.jpg")
    got = find_photos([tmp_path])
    assert got == sorted(got)                       # порядок = порядок полных путей
    assert find_photos([tmp_path]) == got           # повторный вызов идентичен
    assert [p.name for p in got] == ["a.jpg", "b.jpg", "a.jpg", "d.jpg"]


def test_poryadok_pri_neskolkih_papkah_govorit_pravdu(tmp_path) -> None:
    """`got == sorted(got)` верно ровно для одной входной папки — и это не баг.

    Каждая папка обходится сама по себе и отдаётся отсортированной внутри себя, а
    несколько папок идут в том порядке, в котором их назвал человек. Глобальная
    сортировка стёрла бы это намерение: «Фото», потом «Камера», потом «Отпуск» —
    порядок, который человек узнаёт, а не алфавит, который ему ничего не говорит.

    Прежняя проверка в тесте выше держала только этот частный случай и на двух папках
    стала бы лгать, поэтому здесь формулировка настоящая: детерминированность,
    упорядоченность ВНУТРИ каждой папки и следование порядку аргументов.
    """
    foto = tmp_path / "Я"                # намеренно «позже» по алфавиту
    kamera = tmp_path / "А"
    _make(foto, "b.jpg", "a.jpg")
    _make(kamera, "c.jpg", "a.jpg")

    pervyj = find_photos([foto, kamera])
    vtoroj = find_photos([foto, kamera])
    assert pervyj == vtoroj, "повторный вызов дал другой порядок"

    granica = len(list(foto.rglob("*.jpg")))
    assert pervyj[:granica] == sorted(pervyj[:granica]), "внутри первой папки не sorted"
    assert pervyj[granica:] == sorted(pervyj[granica:]), "внутри второй не sorted"
    assert set(p.name for p in pervyj[:granica]) == {"a.jpg", "b.jpg"}
    assert set(p.name for p in pervyj[granica:]) == {"a.jpg", "c.jpg"}

    # порядок аргументов — порядок отдачи: смена местами меняет и результат
    obratnyj = find_photos([kamera, foto])
    assert obratnyj[granica:] == pervyj[:granica]
    # и именно первая папка идёт первой: глобальной сортировки здесь нет
    assert pervyj != obratnyj
    assert pervyj != sorted(pervyj), "папки всё-таки отсортировали по алфавиту"


def test_shtamp_menyaetsya_pri_perepisiivanii_v_tu_zhe_sekundu(tmp_path) -> None:
    """Ключ инвалидации кэша: перезапись файла в ту же секунду обязана менять штамп."""
    p = tmp_path / "f.jpg"
    p.write_bytes(b"x" * 10)
    stamp = file_stamp(p)
    assert isinstance(stamp, tuple) and len(stamp) == 2
    assert all(isinstance(v, int) for v in stamp)
    p.write_bytes(b"x" * 10)               # тот же размер, та же секунда
    assert file_stamp(p) != stamp, "штамп не различил перезапись — кэш протухнет молча"
    p.write_bytes(b"x" * 11)               # изменился размер
    assert file_stamp(p)[1] == 11
