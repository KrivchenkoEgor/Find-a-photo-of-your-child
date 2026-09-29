"""Пути служебных данных и сохранённые между запусками настройки.

Ни один тест не должен оставить след в настоящей служебной папке пользователя: за этим
следит барьер `_sluzhebnaya_papka_polzovatelya_v_tshine` — снимок состояния до теста и
сверка после. Настоящий путь при этом не создаётся: `app_dir()` — геттер без побочных
эффектов, иначе проверять «не тронули ли настоящее» было бы уже поздно.
"""

import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from utils.config import (APP_NAME, DEFAULT_ENGINE, DEFAULT_MAX_DIM, DEFAULT_THRESHOLD,
                          THRESHOLD_RANGE, Settings, app_dir, buffalo_l_dir, cache_file,
                          models_dir, settings_file, yunet_model_path)

# Настоящая служебная папка вычисляется при импорте модуля, до всяких подменок HOME.
_realnaya_papka = app_dir()


def _sostoyanie(papka: Path) -> tuple[bool, tuple[str, ...]]:
    """Есть ли папка и что в ней лежит. Сравнивается целиком, по именам."""
    if not papka.exists():
        return (False, ())
    return (True, tuple(sorted(p.name for p in papka.iterdir())))


@pytest.fixture(autouse=True)
def _sluzhebnaya_papka_polzovatelya_v_tshine() -> Iterator[None]:
    """Барьер: тест упадёт, если после него в папке реального пользователя что-то изменилось."""
    do = _sostoyanie(_realnaya_papka)
    yield
    posle = _sostoyanie(_realnaya_papka)
    assert posle == do, (
        f"тест тронул настоящую служебную папку {_realnaya_papka}: "
        f"было {do}, стало {posle}")


@pytest.fixture
def dom(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Домашняя папка подменой — и macOS/Linux (HOME), и Windows (USERPROFILE).

    Без USERPROFILE тест, который завтра пустят на Windows, молча написал бы в
    настоящий каталог пользователя, и барьер выше поймал бы это только постфактум.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("APPDATA", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    return tmp_path


def sohrani(**znachenija: Any) -> Settings:
    """`Settings` по умолчанию с записью на диск — так делает интерфейс после ползунка."""
    s = Settings()
    for klyuch, znachenie in znachenija.items():
        setattr(s, klyuch, znachenie)
    s.save()
    return s


# ------------------------------------------------------------------ умолчалки


def test_umolchalnie_iz_zmer(tmp_path: Path) -> None:
    s = Settings(tmp_path / "s.ini")
    assert s.threshold == 0.38          # худшее подтверждённое фото — 39.2%
    assert s.engine == "both"           # YuNet один теряет 1 фото из 20
    assert s.max_dim == 2400            # 1200 px стоит 3–7 процентных пунктов
    assert (DEFAULT_THRESHOLD, DEFAULT_ENGINE, DEFAULT_MAX_DIM) == (0.38, "both", 2400)
    assert s.last_source == "" and s.last_dest == ""    # пустой путь, а не строка "None"


def test_diapazon_polzunka_zadan_iz_mere() -> None:
    """Задача 12 рисует ползунок по этому коридору: выше 67% не бывает даже на своём фото."""
    assert THRESHOLD_RANGE == (0.30, 0.60)
    assert THRESHOLD_RANGE[0] <= DEFAULT_THRESHOLD <= THRESHOLD_RANGE[1]


# --------------------------------------------------------------- перенос значений


def test_znachenija_peredajutsja_mezhda_ekzemplyarami(tmp_path: Path) -> None:
    p = tmp_path / "s.ini"
    s = Settings(p)
    s.threshold, s.engine, s.max_dim, s.last_source = 0.45, "yunet", 1200, "/фото"
    s.save()
    again = Settings(p)
    assert (again.threshold, again.engine, again.max_dim, again.last_source) == \
           (0.45, "yunet", 1200, "/фото")


def test_poslednyaya_papka_resultatov_pomnitsya(tmp_path: Path) -> None:
    """Диалог копирования открывается в прошлый раз — для этого и нужен `last_dest`."""
    p = tmp_path / "s.ini"
    s = Settings(p)
    s.last_dest = "/фото/Отпуск 2024"
    s.save()
    assert Settings(p).last_dest == "/фото/Отпуск 2024"


def test_dva_fayla_ne_putayutsja_mezhdu_soboy(tmp_path: Path) -> None:
    """QSettings обязан уважать переданный путь, а не глобальное имя организации.

    Если собрать его конструктором «организация + приложение», тесты начнут читать и
    писать настройки реального пользователя и при этом оставаться зелёными.
    """
    a, b = tmp_path / "a.ini", tmp_path / "b.ini"
    pervyy = Settings(a)
    pervyy.threshold = 0.50
    pervyy.save()
    vtoroy = Settings(b)
    vtoroy.engine = "insight"
    vtoroy.save()
    assert Settings(b).threshold == DEFAULT_THRESHOLD
    assert Settings(a).threshold == 0.50
    assert Settings(a).engine == DEFAULT_ENGINE


# ---------------------------------------------------------------- сохранение


def test_save_vozvraaet_true_kogda_fayl_leg_na_disk(tmp_path: Path) -> None:
    p = tmp_path / "s.ini"
    s = Settings(p)
    s.threshold = 0.42
    assert s.save() is True
    assert p.is_file() and Settings(p).threshold == 0.42


def test_neudachnyy_save_vozvraaet_false_a_ne_molchit(
        tmp_path: Path) -> None:
    """`save()` обязан вернуть False, если на диск не легло ничего.

    `QSettings.sync()` исключений не бросает: Qt молча кладёт ошибку в `status()`, и
    без его проверки человек, передвинувший ползунок на съёмном диске, увидел бы своё
    число на экране и потерял бы его при следующем запуске — без единого слова.
    Родителем файла здесь служит обычный файл: каталог для `s.ini` Qt не сможет
    создать нигде, включая Windows, где `chmod` на права не влияет.
    """
    blokada = tmp_path / "sini.txt"
    blokada.write_text("это файл, а не папка", encoding="utf-8")
    s = Settings(blokada / "s.ini")
    s.threshold = 0.45
    s.max_dim = 1800
    assert s.save() is False
    assert not (blokada / "s.ini").exists()


# ------------------------------------------------------------------ битый файл

def test_slomaannyy_fail_ne_ronyaet_prilozhenie(tmp_path: Path) -> None:
    p = tmp_path / "s.ini"
    p.write_text("это не ini", encoding="utf-8")
    assert Settings(p).threshold == 0.38


def test_bitoe_znachenie_v_fayle_ne_ishet_na_sboy(tmp_path: Path) -> None:
    """Файл правят руками или записала старая версия: читаем или отбрасываем, но не падаем.

    Режим проверяется по трём допустимым значениям: `make_engine` на неизвестном
    бросает ValueError, и человек получил бы «разбор прервался» из-за строки, которую
    он и не выбирал. Размер разбора проверяется тем же приёмом: ноль обнулил бы
    изображение до 1 пикселя.
    """
    p = tmp_path / "s.ini"
    p.write_text("[General]\nthreshold=abc\nmax_dim=abc\nengine=42\n", encoding="utf-8")
    s = Settings(p)
    assert s.threshold == DEFAULT_THRESHOLD
    assert s.max_dim == DEFAULT_MAX_DIM
    assert s.engine == DEFAULT_ENGINE


def test_znachenie_za_predelami_ne_razojdetsja_s_polzunkom(tmp_path: Path) -> None:
    """Ползунок показывает 60%, а поиск идёт по 0.99 — расхождение, которое не объяснить.

    Значение из файла приводится к коридору настроек, поэтому число на экране и число,
    по которому режутся кучки «похоже»/«слабое сходство», всегда одни и те же.
    """
    p = tmp_path / "s.ini"
    p.write_text("[General]\nthreshold=0.99\nmax_dim=0\n", encoding="utf-8")
    s = Settings(p)
    assert s.threshold == THRESHOLD_RANGE[1]
    assert s.max_dim == DEFAULT_MAX_DIM


def test_spisok_vmesto_chisla_tozhe_ne_ishet(tmp_path: Path) -> None:
    """Qt толкует «0,5» как список строк: `float(['0', '5'])` бросает TypeError.

    Ветку `except TypeError` легко оставить мёртвым кодом — и тогда первая же
    перечисленная настройка уронит приложение на старте.
    """
    p = tmp_path / "s.ini"
    p.write_text("[General]\nthreshold=0,5\n", encoding="utf-8")
    assert Settings(p).threshold == DEFAULT_THRESHOLD


def test_nan_i_inf_v_fayle_ne_prohodit_dalse(tmp_path: Path) -> None:
    """`float('nan')` не бросает ничего, а `int(nan)` — уже ValueError.

    Строка `nan` или `inf` в файле — это не «не число», а число, с которым Python
    согласен до первого превращения в пиксели: проверка на конечность нужна именно
    здесь, а не в каждом потребителе настройки.
    """
    p = tmp_path / "s.ini"
    p.write_text("[General]\nthreshold=nan\nmax_dim=inf\n", encoding="utf-8")
    s = Settings(p)
    assert s.threshold == DEFAULT_THRESHOLD
    assert s.max_dim == DEFAULT_MAX_DIM


# -------------------------------------------------------------- расположение


def test_papki_prilozhenija_vnutri_sluzhebnoy(tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert app_dir().name == "FindChild"
    assert cache_file().suffix == ".sqlite3"
    assert cache_file().parent == app_dir()


def test_nastroyki_lezhat_v_sluzhebnoy_papke_a_ne_v_papke_s_foto(
        dom: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ни реестра, ни файлов в архиве: ini лежит в app_dir(), а вокруг пусто.

    «В семейном архиве не должно быть наших файлов» — это не про аккуратность: папку с
    фото ребёнка носят между дисками, и мусор внутри неё не прощают.
    """
    foto = dom / "архив фото"
    foto.mkdir()
    monkeypatch.chdir(foto)
    s = sohrani(threshold=0.42)
    assert s.path == settings_file() == app_dir() / "settings.ini"
    assert s.path.parent == app_dir()
    tekst = s.path.read_text(encoding="utf-8")
    assert "[General]" in tekst and "threshold=0.42" in tekst    # ini на диске, не реестр
    assert list(foto.iterdir()) == []                            # в папке с фото — пусто
    assert not list(dom.glob("*.ini"))                           # и в «доме» ничего нет


def test_znachenija_pomnyayutsja_cherez_novyy_ekzemplyar_bez_puti(dom: Path) -> None:
    sohrani(engine="insight")
    assert Settings().engine == "insight"
    assert Settings().path == settings_file()


def test_app_dir_nichego_ne_sozdaet(dom: Path) -> None:
    """Геттер пути без побочных эффектов.

    Пишущие создают папку сами: `Settings` — через Qt, `FaceCache` — через свой mkdir.
    Иначе один только вопрос «куда писать» плодил бы каталоги и у реального
    пользователя, и в тестовом «доме».
    """
    assert not app_dir().exists()
    assert app_dir() == dom / "Library" / "Application Support" / APP_NAME


def test_windowskaya_vetka_idet_v_appdata_a_ne_v_reestr(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows берёт %APPDATA%; та же ветка проверяется на macOS подменой платформы."""
    monkeypatch.setattr(sys, "platform", "win32", raising=False)
    appdata = tmp_path / "AppData"
    monkeypatch.setenv("APPDATA", str(appdata))
    assert app_dir() == appdata / APP_NAME
    assert settings_file().parent == app_dir()


def test_na_windows_bez_appdata_put_ne_propadaet(dom: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """%APPDATA% физически нет (служебный вход, портативный запуск) — путь всё равно есть.

    Откат на `%USERPROFILE%\\AppData\\Roaming` ведёт туда же, куда Windows подставляет
    APPDATA: настройка не должна уезжать в корень домашней папки, где её никто не ищет.
    """
    monkeypatch.setattr(sys, "platform", "win32", raising=False)
    monkeypatch.delenv("APPDATA", raising=False)
    assert app_dir() == dom / "AppData" / "Roaming" / APP_NAME


def test_linux_idet_v_xdg(dom: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(dom / "xdg"))
    assert app_dir() == dom / "xdg" / APP_NAME
    monkeypatch.delenv("XDG_DATA_HOME")
    assert app_dir() == dom / ".local" / "share" / APP_NAME


# ---------------------------------------------------------------------- модели


def test_papka_modeley_kotoruyu_proveryayut_pri_oshibke(dom: Path) -> None:
    """`worker` на сбое весов говорит «проверьте путь к моделям» — путь должен быть где-то.

    `models_dir()` повторяет правило insightface: корень `~/.insightface`, подкаталог
    `models`, в нём `buffalo_l` (~290 МБ). Импортировать ради этого insightface нельзя
    (тестам он запрещён), а правило стабильно между версиями библиотеки.
    """
    assert models_dir() == dom / ".insightface" / "models"
    assert models_dir().name == "models"
    assert models_dir().parent.name == ".insightface"


def test_models_dir_ne_sozdaet_papki(dom: Path) -> None:
    """Отсутствующие веса — как раз тот случай, когда путь и спрашивают.

    Геттер, который создаёт каталог, превратил бы «папки нет» в пустую папку, и
    следующий прогон потерял бы единственную зацепку для человека.
    """
    assert not models_dir().exists()
    assert not buffalo_l_dir().exists()


def test_buffalo_l_ozhidaetsja_v_keshe_insightface(dom: Path) -> None:
    """`worker` бросает «проверьте путь к моделям» — показать надо папку `buffalo_l`.

    Она лежит внутри `models_dir()`: пять onnx-файлов, и нужное приложение ищет
    recognition-модель именно там.
    """
    assert buffalo_l_dir() == models_dir() / "buffalo_l"
    assert buffalo_l_dir().name == "buffalo_l"
    assert buffalo_l_dir().parent == models_dir()


def test_v_sobrannom_prilozhenii_vesa_beryotsja_iz_arhiva(
        dom: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`.app` уезжает на другой Mac, где `~/.insightface` нет вообще.

    Если путь к весам смотрит только в кэш, на чужой машине приложение попросит
    интернет и 290 МБ, хотя все нужные веса лежат внутри самого бандла. Архив —
    первое, где их надо искать.
    """
    arhiv = tmp_path / "arhiv"
    (arhiv / "models" / "buffalo_l").mkdir(parents=True)
    monkeypatch.setattr(sys, "_MEIPASS", str(arhiv), raising=False)

    assert buffalo_l_dir() == arhiv / "models" / "buffalo_l"


def test_bezy_ves_v_arhive_keshe_ostaetsja_domom(
        dom: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Сборка без весов (и запуск из исходников) — прежний путь, ничего не выдумываем.

    Геттер обязан и не создавать папок: пустой каталог внутри архива превратил бы
    «весов нет» в «весы есть, но пустые», и человек чинил бы не то.
    """
    arhiv = tmp_path / "arhiv"
    (arhiv / "models").mkdir(parents=True)
    monkeypatch.setattr(sys, "_MEIPASS", str(arhiv), raising=False)

    assert buffalo_l_dir() == dom / ".insightface" / "models" / "buffalo_l"
    assert not (arhiv / "models" / "buffalo_l").exists()


def test_yunet_lezhit_v_paketie_a_ne_v_koshe_modeley(dom: Path) -> None:
    """YuNet весит 230 КБ и едет внутри приложения: `.insightface` ему не дом.

    Отсюда два пути на один вопрос «где модели»: пакет распознавания чинит
    пользователь, а YuNet починить можно только пересборкой. Врать про один путь на
    оба было бы дешевле в коде и дороже в переписке с поддержкой.
    """
    put = yunet_model_path()
    assert put.name == "face_detection_yunet_2023mar.onnx"
    assert put.parent.name == "models"
    assert put.parent.parent.name == "assets"
    assert put != models_dir() / put.name


def test_yunet_iz_repozitoriya_na_meste_v_rezhime_razrabotki() -> None:
    """В исходниках путь обязан вести в реальный файл, иначе разбор встанет без слов."""
    assert yunet_model_path().is_file(), f"нет файла модели: {yunet_model_path()}"


def test_yunet_ne_zhivyot_v_gitignore(tmp_path: Path) -> None:
    """Файл модели должен быть в РЕПОЗИТОРИИ, а не только на этом диске.

    Тест выше зелёный и при невключённом файле: он смотрит на рабочую папку. Первый же
    клон с GitHub получал бы «нет модели» на первой кнопке, и чинить это мог только
    человек, скачавший вес по ссылке, которой в прежнем тексте ошибки не было.

    Причина пряталась в `.gitignore`: общий запрет `*.onnx` и `models/` вычёркивал и
    230-килобайтный файл OpenCV. Спрашиваем не текст файла игнорирования, а самого git:
    только он знает, какое из правил пересилило остальные и в каком порядке.
    """
    koren = Path(__file__).resolve().parents[1]
    fayl = koren / "assets" / "models" / "face_detection_yunet_2023mar.onnx"
    proverka = subprocess.run(["git", "check-ignore", "-q", str(fayl)],
                              cwd=str(koren), capture_output=True)
    if proverka.returncode not in (0, 1):
        pytest.skip("git недоступен — проверять нечем")
    assert proverka.returncode == 1, (
        f"{fayl.name} в gitignore: со свежего клона приложение не разберёт ни одной "
        "папки; нужен `!assets/models/` и `!assets/models/*.onnx`")
    otsvet = subprocess.run(["git", "ls-files", "--error-unmatch",
                             str(fayl.relative_to(koren))],
                            cwd=str(koren), capture_output=True)
    assert otsvet.returncode == 0, f"{fayl.name} не добавлен в репозиторий"


def test_ikonka_lezhit_v_repozitorii_i_etо_nastoyashchij_icns(
        tmp_path: Path) -> None:
    """Без `.icns` PyInstaller не собирает `.app` вовсе — и это видно только на сборке.

    Общий запрет `*.png` в gitignore молча вычёркивал и мастер-файл иконки: на этой
    машине файл есть, а со свежего клона сборка падала бы с невнятным «Icon input file
    None not found». Спрашиваем git, а не текст `.gitignore`, — только он знает, какое
    из правил пересилило остальные.
    """
    koren = Path(__file__).resolve().parents[1]
    fayl = koren / "assets" / "icon.icns"
    master = koren / "assets" / "icon.png"

    assert fayl.is_file(), "нет assets/icon.icns: сборка .app встанет"
    assert fayl.read_bytes()[:4] == b"icns", "это не формат .icns"
    assert master.is_file(), "нет assets/icon.png: иконку нечем пересобрать"

    for proveryaemyj in (fayl, master):
        otsvet = subprocess.run(["git", "check-ignore", "-q", str(proveryaemyj)],
                                cwd=str(koren), capture_output=True)
        assert otsvet.returncode == 1, f"{proveryaemyj.name} в gitignore"


def test_sobrannoe_prilozhenie_chitaet_model_iz_arhiva(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """PyInstaller `--onefile` раскрывает ресурсы в `sys._MEIPASS` — путь смотрит туда."""
    bundle = tmp_path / "bundle"
    (bundle / "assets" / "models").mkdir(parents=True)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    assert yunet_model_path() == (bundle / "assets" / "models"
                                 / "face_detection_yunet_2023mar.onnx")
