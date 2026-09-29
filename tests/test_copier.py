"""Копирование отмеченных фото в папку результатов.

Обещание модуля одно, и оно дорогое для семейного архива: ничего существующего не
затирается. Тесты синтетические — файлы пишутся в `tmp_path`, личные снимки из data/
и ref/ не участвуют.

Второе обещание — за ним приходит человек с папкой на 900 снимков: один плохой файл
не стоит всей пачки. Поэтому здесь проверены и имя, которое не влезает в бюджет диска,
и кончившееся место в середине прогона, и снятые с файла права на чтение.

Третье обещание дороже первых двух: потеря обязана быть названа. Аккумулятор причин
передают теперь каждым вызовом, оборванный на ползаписи снимок не остаётся в папке
результатов без слова, а имя, которому не с чем сталкиваться, не платит резервом под
счётчик.
"""

import os
import shutil
import sys
from pathlib import Path

import pytest

from core import copier
from core.copier import copy_photos, unique_dest


def _bjudzet_imeni(papka: Path) -> int:
    """Сколько том вмещает в одно имя. 255 — общий предел, где спросить нельзя.

    Считаем его в тесте сами, а не берём из `_predel_imeni`: проверять бюджет тем же
    кодом, который его соблюдает, — значит ничего не проверить.
    """
    pathconf = getattr(os, "pathconf", None)
    if pathconf is None:
        return 255
    try:
        return int(pathconf(papka, "PC_NAME_MAX"))
    except (OSError, ValueError, NotImplementedError):
        return 255


def _foto(dir_: Path, name: str, content: bytes = b"x") -> Path:
    """Синтетический снимок нужного имени.

    Папку создаём здесь, а не в вызывающем тесте: тесты из брифа передавали
    `tmp_path / "in"`, которой на диске нет, и `write_bytes` падал ещё до проверки
    копирования.
    """
    dir_ = Path(dir_)
    dir_.mkdir(parents=True, exist_ok=True)
    p = dir_ / name
    p.write_bytes(content)
    return p


@pytest.fixture()
def poteri() -> list[tuple[Path, str]]:
    """Свежий аккумулятор причин.

    `copy_photos` без него уже не вызывается, поэтому это фикстура, а не строчка в
    каждом тесте: тестов, которым сама причина не интересна, большинство.
    """
    return []


def test_daeet_novoe_imya_pri_konflikte(tmp_path: Path) -> None:
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "a.jpg").write_bytes(b"old")
    got = unique_dest(dest, "a.jpg")
    assert got.name == "a_1.jpg"
    # Строка брифа ждала `a_1_1.jpg`, не заняв `a_1.jpg`: свободное имя ищется по
    # диску, а не по только что предложенному варианту, — иначе два вызова подряд
    # отдали бы одно и то же место под два разных снимка.
    (dest / "a_1.jpg").write_bytes(b"old")
    assert unique_dest(dest, "a_1.jpg").name == "a_1_1.jpg"
    assert unique_dest(dest, "a.jpg").name == "a_2.jpg"


def test_kopiruet_sohranyaya_imena_i_vozvraaet_pary(
        tmp_path: Path, poteri: list[tuple[Path, str]]) -> None:
    src = [_foto(tmp_path / "in", "1.jpg"), _foto(tmp_path / "in", "2.jpg")]
    dest = tmp_path / "out"
    pairs = copy_photos(src, dest, poteri)
    assert [d.name for _, d in pairs] == ["1.jpg", "2.jpg"]
    assert dest.joinpath("1.jpg").exists()


def test_ne_zatiraet_suschestvuyushchiy_fail(tmp_path: Path,
                                             poteri: list[tuple[Path, str]]) -> None:
    src = _foto(tmp_path, "vazhnoe.jpg", b"new")
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "vazhnoe.jpg").write_bytes(b"already here")
    copy_photos([src], dest, poteri)
    assert (dest / "vazhnoe.jpg").read_bytes() == b"already here"
    assert (dest / "vazhnoe_1.jpg").read_bytes() == b"new"


def test_odnoimennye_snimki_iz_raznyh_papok_legayut_dvumya_failami(
        tmp_path: Path, poteri: list[tuple[Path, str]]) -> None:
    """Один снимок из «Отпуск», другой из «Друг» — оба называются IMG_1.jpg.

    Разбор папки отдаёт их по полному пути и оба попадают в отметку. В папке
    результатов путь один, и второй обязан получить суффикс, а не съесть первый.
    """
    a = _foto(tmp_path / "otpusk", "IMG_1.jpg", b"pervyy")
    b = _foto(tmp_path / "drugie", "IMG_1.jpg", b"vtoroy")
    dest = tmp_path / "out"
    pairs = copy_photos([a, b], dest, poteri)
    assert [d.name for _, d in pairs] == ["IMG_1.jpg", "IMG_1_1.jpg"]
    assert (dest / "IMG_1.jpg").read_bytes() == b"pervyy"
    assert (dest / "IMG_1_1.jpg").read_bytes() == b"vtoroy"


def test_sozydaet_papku_resultatov(tmp_path: Path,
                                   poteri: list[tuple[Path, str]]) -> None:
    src = _foto(tmp_path / "in", "1.jpg")
    dest = tmp_path / "нет" / "такой" / "papki"
    copy_photos([src], dest, poteri)
    assert (dest / "1.jpg").exists()


def test_uscheznusshiy_fail_daet_para_menshe_i_ne_padaet(
        tmp_path: Path, poteri: list[tuple[Path, str]]) -> None:
    """Снимок удалили из архива между разбором и копированием — прогон не рвётся.

    Молчание здесь стоило бы доверия: пары меньше, чем отметили, и вызывающий обязан
    сравнивать числа (`len(files)` против `len(pairs)`), а не верить, что всё легло.
    """
    zhivoy = _foto(tmp_path / "in", "zhivoy.jpg")
    udalennyy = tmp_path / "in" / "uzhe-net.jpg"
    pairs = copy_photos([zhivoy, udalennyy], tmp_path / "out", poteri)
    assert [s.name for s, _ in pairs] == ["zhivoy.jpg"]
    assert [p.name for p, _ in poteri] == ["uzhe-net.jpg"]


# ------------------------------------------------- изоляция одной потери в пачке


@pytest.mark.skipif(sys.platform.startswith("win"),
                    reason="на Windows chmod снимает запись, а не чтение")
def test_odin_nechitaemyy_snimok_ne_steryaet_vsyo_ostalnoe(
        tmp_path: Path, poteri: list[tuple[Path, str]]) -> None:
    """Файл без прав на чтение в середине списка: остальные три обязаны доехать.

    До правки `shutil.copy2` бросал `PermissionError` наружу, цикл обрывался, и два
    снимка, стоявшие за гнилым, не копировались вообще — молча, без единой записи о
    причине. Ради одного пропавшего файла терять остальные двести нельзя.
    """
    zhivye = [_foto(tmp_path / "in", f"{i}.jpg", b"x" * i) for i in (1, 2)]
    gnooy = _foto(tmp_path / "in", "00-gnooy.jpg", b"ne otkroetsya")
    posle = _foto(tmp_path / "in", "3.jpg", b"yyy")
    os.chmod(gnooy, 0o000)
    dest = tmp_path / "out"
    try:
        pairs = copy_photos([*zhivye, gnooy, posle], dest, poteri)
    finally:
        os.chmod(gnooy, 0o644)
    assert [s.name for s, _ in pairs] == ["1.jpg", "2.jpg", "3.jpg"]
    assert sorted(p.name for p in dest.iterdir()) == ["1.jpg", "2.jpg", "3.jpg"]
    assert (dest / "3.jpg").read_bytes() == b"yyy"     # идёт после гнилого — живой
    assert [p.name for p, _ in poteri] == ["00-gnooy.jpg"]


def test_poteri_nazyvayut_fayl_i_prichinu(tmp_path: Path,
                                          poteri: list[tuple[Path, str]]) -> None:
    """`poteri` — канал вызывающего: не только сколько, но и почему.

    Число пар говорит «потеряли один», а человеку нужны имя и причина, чтобы понять,
    что делать. Паре (файл, причина) рады в разделе потерь `write_report`, где лежит
    ровно такой же кортеж.
    """
    zhivoy = _foto(tmp_path / "in", "1.jpg")
    net = tmp_path / "in" / "net.jpg"
    pairs = copy_photos([zhivoy, net], tmp_path / "out", poteri)
    assert [d.name for _, d in pairs] == ["1.jpg"]
    assert [p.name for p, _ in poteri] == ["net.jpg"]
    assert poteri[0][1].startswith("файл недоступен")


def test_bez_akkumulyatora_prichin_vyzyv_ne_prohodit(tmp_path: Path) -> None:
    """Двухаргументный вызов обязан падать `TypeError` на стенде, а не молчать в бою.

    Пока `poteri` был необязательным, канал причины жил только на честном слове
    вызывающего: `copy_photos(files, target)` возвращал пустой список, и человек
    видел «скопировано 0 фото» без единого слова о причине — ровно то молчание,
    против которого весь модуль. Забыть о параметре можно было только в вызове,
    которого никто не заметил бы; теперь его нет в сигнатуре.
    """
    src = _foto(tmp_path / "in", "1.jpg")
    with pytest.raises(TypeError, match="poteri"):
        copy_photos([src], tmp_path / "out")


class _KonchaetsyaMesto:
    """Заглушка `shutil` для проверки полного диска: второй снимок — и ENOSPC.

    Настоящий диск в тесте не заполнишь, а проверяется ровно это поведение: беда
    приходит в середине пачки и не утаскивает за собой ни скопированное, ни
    остальное.
    """

    def __init__(self) -> None:
        self.vyzy = 0

    def copy2(self, src, dst) -> None:
        self.vyzy += 1
        if self.vyzy > 1:
            raise OSError(28, "No space left on device", str(src))
        shutil.copy2(src, dst)


def test_polnyy_disk_v_seredine_progona_ostavlyaet_nachalo_pachki_i_prichinu(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        poteri: list[tuple[Path, str]]) -> None:
    """Место кончилось на втором снимке: первый уже на диске, причина названа.

    Здесь важно, что `except` висит над копированием, а не над прогоном: без него
    человек получил бы «процесс упал» и ни строчки в отчёте о том, что он потерял.
    """
    a, b, c = (_foto(tmp_path / "in", f"{i}.jpg") for i in (1, 2, 3))
    disk = _KonchaetsyaMesto()
    monkeypatch.setattr(copier, "shutil", disk)
    pairs = copy_photos([a, b, c], tmp_path / "out", poteri)
    assert [s.name for s, _ in pairs] == ["1.jpg"]
    assert [p.name for p, _ in poteri] == ["2.jpg", "3.jpg"]
    assert "No space left on device" in poteri[0][1]


def test_odin_fayl_ne_list_na_papku_resultatov(tmp_path: Path,
                                               poteri: list[tuple[Path, str]]) -> None:
    """Форма возврата не изменилась: список пар `Path`, и распаковывается как раньше.

    План задачи 15 зовёт `pairs = copy_photos(files, target, poteri)` и дальше кладёт
    результат в `copied=pairs`, в `len(pairs)` и в `for s, d in pairs`. Причина уезжает
    аргументом именно поэтому: третьим элементом кортежа она сломала бы каждый из
    этих трёх способов обратиться к парам.
    """
    src = _foto(tmp_path / "in", "1.jpg")
    pary = copy_photos([src], tmp_path / "out", poteri)
    assert isinstance(pary, list) and len(pary) == 1
    pervaya, vtoraya = pary[0]                     # пара — ровно два `Path`
    assert isinstance(pervaya, Path) and isinstance(vtoraya, Path)


# ------------------------------------------ оборванный на ползаписи снимок


class _PishetIObryvaetsya:
    """Пишет половину файла и падает с `ENOSPC` — точно как настоящий `copy2`.

    `shutil.copy2` копирует частями и за собой не убирает: после `ENOSPC` или `EIO`
    на диске остаётся усечённый снимок. Воспроизвести это на настоящем томе нельзя,
    а поведение нужно ровно то же: файл создан, ошибка пришла. Первый вызов отдаём
    настоящему `copy2`, чтобы заодно проверить живую часть пачки.
    """

    def __init__(self) -> None:
        self.vyzy = 0

    def copy2(self, src, dst) -> None:
        self.vyzy += 1
        if self.vyzy == 1:
            shutil.copy2(src, dst)
            return
        Path(dst).write_bytes(b"polovina_snimka")
        raise OSError(28, "No space left on device", str(src))


def test_posle_obrannogo_kopirovaniya_ogryzka_v_papke_ne_ostaetsya(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        poteri: list[tuple[Path, str]]) -> None:
    """Место кончилось на середине второго снимка: в папке результатов — только первый.

    До правки усечённый `2.jpg` оставался лежать в папке результатов: в `pairs` его
    нет, значит в разделе «Скопировано» отчёта он не значится, и человек находит его
    только глазами, пересматривая папку. Огрызок без слова о нём хуже упавшего
    прогона — прогон хотя бы говорит, что что-то не так.
    """
    a = _foto(tmp_path / "in", "1.jpg", b"celiy")
    b = _foto(tmp_path / "in", "2.jpg", b"x" * 5000)
    dest = tmp_path / "out"
    monkeypatch.setattr(copier, "shutil", _PishetIObryvaetsya())
    pairs = copy_photos([a, b], dest, poteri)
    assert [d.name for _, d in pairs] == ["1.jpg"]
    assert sorted(p.name for p in dest.iterdir()) == ["1.jpg"]   # огрыстка тут нет
    assert (dest / "1.jpg").read_bytes() == b"celiy"              # живой не задет
    assert [p.name for p, _ in poteri] == ["2.jpg"]
    assert "No space left on device" in poteri[0][1]


class _PishetIZakryvaetPapku:
    """Как предыдущая заглушка, но перед отказом перекрывает папке запись.

    Так проверяется вторая ветка уборки: убрать огрызок удаётся не всегда — том
    перемонтировался read-only или права отобрали вместе с ошибкой. Тогда о нём надо
    сказать прямо в причине, а не делать вид, что убрали.
    """

    def __init__(self, papka: Path) -> None:
        self.papka = papka

    def copy2(self, src, dst) -> None:
        Path(dst).write_bytes(b"polovina_snimka")
        os.chmod(self.papka, 0o500)
        raise OSError(28, "No space left on device", str(src))


@pytest.mark.skipif(sys.platform.startswith("win"),
                    reason="на Windows chmod не снимает права на удаление")
def test_neubrannyj_ogryzok_nazvan_v_prichine(tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch,
                                              poteri: list[tuple[Path, str]]) -> None:
    """Огрызок остался на диске — потеря обязана называть и это, а не только ENOSPC.

    Раздел потерь читают глазами и с него идут разбираться в папку: «не смог
    скопировать» и «не смог скопировать, и половине файла пришлось остаться у вас» —
    это разные инструкции человеку.
    """
    src = _foto(tmp_path / "in", "1.jpg", b"x" * 5000)
    dest = tmp_path / "out"
    dest.mkdir()
    monkeypatch.setattr(copier, "shutil", _PishetIZakryvaetPapku(dest))
    try:
        pairs = copy_photos([src], dest, poteri)
        assert pairs == []
        assert (dest / "1.jpg").exists()                        # убрать было нечем
        prichina = poteri[0][1]
        assert "No space left on device" in prichina
        assert "остался в папке" in prichina                    # и про огрызок сказано
    finally:
        os.chmod(dest, 0o755)


def test_sboy_na_imeni_ne_ostavlyaet_ogryzka_i_ne_padaet(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        poteri: list[tuple[Path, str]]) -> None:
    """Том отказал, ещё не отдав имя: убирать нечего, и прогон идёт дальше.

    Проверка сторожит разборку `try`: сбой `unique_dest` и сбой копирования лежат в
    разных ветках, и первая не имеет права лезть в `unlink` по несуществующему пути, —
    иначе изоляция одного файла обошлась бы удалением чужого снимка из папки
    результатов.
    """
    a, b = _foto(tmp_path / "in", "1.jpg"), _foto(tmp_path / "in", "2.jpg")
    dest = tmp_path / "out"

    def net_imeni(papka: Path, imya: str) -> Path:
        raise OSError(63, "File name too long", str(papka / imya))

    monkeypatch.setattr(copier, "unique_dest", net_imeni)
    pairs = copy_photos([a, b], dest, poteri)
    assert pairs == []
    assert [p.name for p, _ in poteri] == ["1.jpg", "2.jpg"]
    assert "File name too long" in poteri[0][1]
    assert sorted(p.name for p in dest.iterdir()) == []         # там пусто


# ------------------------------------------------------ имя, которое не влезает


def test_dlinnoe_kirillicheskoe_imya_s_konflikom_ne_padaet(tmp_path: Path) -> None:
    """Имя в 250 кириллических букв и занятое место: `unique_dest` обязан не упасть.

    До правки `Path.exists()` на кандидате `…_1.jpg` ловил `ENAMETOOLONG` — 3.11 не
    проглатывает эту ошибку, — и прогон вставал ещё до первого копирования. Имя
    возвращается урезанным по байтам, а не по символам: буквам нужно два байта, и
    предел наступает вдвое раньше, чем кажется на глаз.
    """
    dest = tmp_path / "out"
    dest.mkdir()
    dlinnoe = "ю" * 250 + ".jpg"
    pervoe = unique_dest(dest, dlinnoe)
    assert not pervoe.exists()
    pervoe.write_bytes(b"staroe")                  # занимаем ровно то, что он дал
    vtoroe = unique_dest(dest, dlinnoe)
    predel = _bjudzet_imeni(dest)
    assert vtoroe != pervoe
    assert len(os.fsencode(vtoroe.name)) <= predel
    assert len(os.fsencode(pervoe.name)) <= predel
    assert vtoroe.name.endswith(".jpg")            # суффикс цел: по нему и опознают
    assert vtoroe.name.startswith("ю")
    vtoroe.write_bytes(b"novoe")                   # том принимает выданное имя
    assert pervoe.read_bytes() == b"staroe"        # и ничего не затёрто


def test_dva_odnoimennyh_snimka_s_dlinnym_imenom_kopiruyutsja_oba(
        tmp_path: Path, poteri: list[tuple[Path, str]]) -> None:
    """Два одноимённых снимка с именем на пределе длины: оба копируются.

    Пробиваем не только `unique_dest`, а весь путь, которым идёт задача 15, — на
    реальном файле, а не на строке: имя из 255 латинских байт примет и APFS, и ext4,
    поэтому проверка не разъезжается по платформам.
    """
    dlinnoe = "a" * 251 + ".jpg"
    a = _foto(tmp_path / "odna", dlinnoe, b"pervyy")
    b = _foto(tmp_path / "drugaya", dlinnoe, b"vtoroy")
    dest = tmp_path / "out"
    pairs = copy_photos([a, b], dest, poteri)
    assert pairs and not poteri
    assert len({d for _, d in pairs}) == 2
    assert sorted(d.read_bytes() for _, d in pairs) == [b"pervyy", b"vtoroy"]
    assert all(d.is_file() for _, d in pairs)


def test_pervoe_imya_beryot_vsyu_meru_a_rezerv_platit_tolko_konflikt(
        tmp_path: Path) -> None:
    """Резерв под `_999` платит только то имя, которое столкнулось с занятым местом.

    Раньше он откусывался заранее, и снимок `aaaa…a.jpg` из 251 буквы — ровно бюджет
    тома — терял четыре буквы просто так, а кириллический из 251 буквы сжимался до 127
    символов: половина имени, которое том принял бы целиком. Столкновения не было,
    значит и счётчика, ради которого оставлен запас, не предвидится.
    """
    dest = tmp_path / "out"
    dest.mkdir()
    predel = _bjudzet_imeni(dest)
    rovno = "a" * (predel - len(".jpg")) + ".jpg"       # ровно `predel` байт
    pervoe = unique_dest(dest, rovno)
    assert pervoe.name == rovno                         # целое: сталкиваться не с чем
    assert len(os.fsencode(pervoe.name)) == predel
    pervoe.write_bytes(b"zanjato")
    vtoroe = unique_dest(dest, rovno)
    assert vtoroe.name.endswith("_1.jpg")               # а вот теперь счётчик платит
    osnova = vtoroe.name[: -len("_1.jpg")]
    assert len(os.fsencode(osnova)) < len(os.fsencode(rovno[: -len(".jpg")]))
    assert len(os.fsencode(osnova + "_999.jpg")) <= predel    # места хватит до 999-го
    vtoroe.write_bytes(b"novoe")                        # и том его принимает


def test_kirillicheskoe_imya_bez_konflika_beryot_ves_bjudzet_bajtami(
        tmp_path: Path) -> None:
    """`ю`*251 без конфликта: теряем только сколько требуют байты, и ни буквы сверх.

    Мера осталась байтовой — это наихудшая из тех, которыми измеряют томы: APFS
    считает символы и принимает 255 кириллических букв, NTFS — единицы UTF-16, и
    только ext4 считает байты. Имя, вписанное в байты, не откажет никто. Правка
    убирает лишь запас счётчика там, где счётчик не нужен: имя берёт бюджет до байта.
    """
    dest = tmp_path / "out"
    dest.mkdir()
    predel = _bjudzet_imeni(dest)
    got = unique_dest(dest, "ю" * 251 + ".jpg")
    assert got.name.endswith(".jpg")                    # суффикс не тронут
    assert len(os.fsencode(got.name)) <= predel
    assert len(os.fsencode(got.name + "ю")) > predel    # ещё буква — и бюджет кончился
    got.write_bytes(b"x")                               # том принимает выданное имя
    s_konfliktom = unique_dest(dest, "ю" * 251 + ".jpg")
    assert s_konfliktom != got
    assert len(os.fsencode(s_konfliktom.name)) <= predel    # и вариант со счётчиком влез
