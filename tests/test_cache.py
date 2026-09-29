"""Оглавление отпечатков: ключ из файла и настроек, устаревание, пустые фото.

Файлы создаются во временной папке теста (`tmp_path`), содержимое — синтетическое:
личные фото из data/ и ref/ в тестах не участвуют.
"""

import sqlite3
import threading
from pathlib import Path

import numpy as np
import pytest

from core.cache import PAYLOAD_VERSION, FaceCache
from core.engine import Face


@pytest.fixture
def photo(tmp_path: Path) -> Path:
    p = tmp_path / "foto.jpg"
    p.write_bytes(b"x" * 100)
    return p


def lico(detector: str = "insight", side: float = 100.0) -> Face:
    return Face(box=(0.0, 0.0, side, side), landmarks=None,
                embedding=np.ones(512, dtype=np.float32) / np.sqrt(512), detector=detector)


def test_posle_put_vozvraet_tot_zhe_sostav(tmp_path: Path, photo: Path) -> None:
    c = FaceCache(tmp_path / "cache.sqlite3")
    c.put(photo, (1000, 100), "both", 2400, [lico(), lico(side=50)])
    got = c.get(photo, (1000, 100), "both", 2400)
    assert got is not None and len(got) == 2
    assert [f.size for f in got] == [100.0, 50.0]
    assert np.allclose(got[0].embedding, lico().embedding)
    c.close()


def test_net_zapisi_dast_None(tmp_path: Path, photo: Path) -> None:
    c = FaceCache(tmp_path / "cache.sqlite3")
    assert c.get(photo, (1, 1), "both", 2400) is None
    c.close()


@pytest.mark.parametrize("stamp,engine,dim", [
    ((1001, 100), "both", 2400),      # файл изменён по времени
    ((1000, 101), "both", 2400),      # изменён размер
    ((1000, 100), "yunet", 2400),     # другой режим — другие отпечатки
    ((1000, 100), "both", 1200),      # другое разрешение — другие отпечатки
])
def test_ustarevaet_pri_lubom_izmenenii_klyucha(tmp_path: Path, photo: Path,
                                                stamp: tuple[int, int], engine: str,
                                                dim: int) -> None:
    c = FaceCache(tmp_path / "cache.sqlite3")
    c.put(photo, (1000, 100), "both", 2400, [lico()])
    assert c.get(photo, stamp, engine, dim) is None
    c.close()


def test_pustoy_spisok_lic_hranitsya_kak_pustoy(tmp_path: Path, photo: Path) -> None:
    """Фото без лиц тоже кэшируем: иначе каждый прогон заново его читаем."""
    c = FaceCache(tmp_path / "cache.sqlite3")
    c.put(photo, (1000, 100), "both", 2400, [])
    assert c.get(photo, (1000, 100), "both", 2400) == []
    c.close()


def test_otpechatok_v_keshe_kompakten(tmp_path: Path, photo: Path) -> None:
    """512 float32 = 2048 байт. base64 добавляет ~33%; JSON-список чисел — в 4-5 раз больше."""
    c = FaceCache(tmp_path / "cache.sqlite3")
    c.put(photo, (1000, 100), "both", 2400, [lico() for _ in range(5)])
    blob = c._db.execute("SELECT payload FROM faces").fetchone()[0]
    assert len(blob) < 5 * 3200, f"5 отпечатков весят {len(blob)} байт — слишком раздуто"
    c.close()


def test_bitaya_zapis_dast_None_i_ischeznet(tmp_path: Path, photo: Path) -> None:
    """Оглавление — восстанавливаемые данные. Битая строка не имеет права ронять разбор."""
    c = FaceCache(tmp_path / "cache.sqlite3")
    c.put(photo, (1000, 100), "both", 2400, [lico()])
    c._db.execute("UPDATE faces SET payload='{не json'")
    c._db.commit()
    assert c.get(photo, (1000, 100), "both", 2400) is None
    assert c._db.execute("SELECT COUNT(*) FROM faces").fetchone()[0] == 0, \
        "битая запись должна удалиться, а не висеть мусором"
    c.close()


def test_zapis_staroversii_ne_chitaetsya(tmp_path: Path, photo: Path) -> None:
    c = FaceCache(tmp_path / "cache.sqlite3")
    c.put(photo, (1000, 100), "both", 2400, [lico()])
    # Версия берётся из константы, а не из строки: тест проверяет «НЕ текущая версия»,
    # и плановое повышение `PAYLOAD_VERSION` не должно превращать его в молчаливый
    # промах, который ничего не подменил и потому всегда зелёный.
    c._db.execute("UPDATE faces SET payload=replace(payload, ?, '\"v\":99')",
                  (f'"v":{PAYLOAD_VERSION}',))
    c._db.commit()
    assert c.get(photo, (1000, 100), "both", 2400) is None
    c.close()


# --- счётчик повреждённых строк -----------------------------------------------------
#
# Для `get` битая строка и отсутствие записи — одно и то же: `None`. Разбору папки
# (задача 9) этого мало: отчёт (задача 10) обязан отличить «этого фото ещё не было
# в оглавлении» от «оглавление самопочинилось, строку пришлось выбросить». Иначе
# архив, где половина строк записана старой версией с NaN-отпечатками, выглядит
# как «всё хорошо», хотя на самом деле он пересчитывается заново на каждый прогон.


def test_bitaya_stroka_uchityvaetsya_a_proloh_net(tmp_path: Path, photo: Path) -> None:
    c = FaceCache(tmp_path / "cache.sqlite3")
    assert c.damage_count == 0                      # свежее оглавление чисто
    c.put(photo, (1000, 100), "both", 2400, [lico()])
    assert c.get(photo, (1000, 100), "both", 2400) is not None
    assert c.damage_count == 0, "попадание не считается повреждением"
    c._db.execute("UPDATE faces SET payload='{не json'")
    c._db.commit()
    assert c.get(photo, (1000, 100), "both", 2400) is None
    assert c.damage_count == 1
    assert c.get(photo, (1000, 100), "both", 2400) is None
    assert c.damage_count == 1, "промах по уже удалённой строке — не новое повреждение"
    c.close()


def test_zapis_staroversii_povrezhdeniem_ne_schitaetsya(tmp_path: Path,
                                                       photo: Path) -> None:
    """Устаревший формат — плановое пересканирование, а не испорченные данные.
    Считать такое повреждением значит соврать в отчёте про «оглавление починено»."""
    c = FaceCache(tmp_path / "cache.sqlite3")
    c.put(photo, (1000, 100), "both", 2400, [lico()])
    # Версия берётся из константы, а не из строки: тест проверяет «НЕ текущая версия»,
    # и плановое повышение `PAYLOAD_VERSION` не должно превращать его в молчаливый
    # промах, который ничего не подменил и потому всегда зелёный.
    c._db.execute("UPDATE faces SET payload=replace(payload, ?, '\"v\":99')",
                  (f'"v":{PAYLOAD_VERSION}',))
    c._db.commit()
    assert c.get(photo, (1000, 100), "both", 2400) is None
    assert c.damage_count == 0
    c.close()


# --- строки, записанные до проверки отпечатков -------------------------------------
#
# Оглавление живёт у пользователя на диске и переживает не одну версию приложения.
# Старые файлы записывали вектор с NaN, с Inf и чужой ширины молча — проверить их
# можно только при чтении. Такое попадание обязано означать «снимок разберётся заново»:
# ни падения рабочего потока посреди папки, ни отравленного лица, по нулям которого
# весь архив молча потерялся бы.


@pytest.mark.parametrize("imya", ["nan", "inf", "nuli", "shirina_256"])
def test_staroestrichnaya_stroka_s_neprigodnym_otpechatkom_dast_None_i_ischeznet(
        tmp_path: Path, photo: Path, imya: str) -> None:
    vektor = {"nan": np.full(512, np.nan, dtype=np.float32),
              "inf": np.full(512, np.inf, dtype=np.float32),
              "nuli": np.zeros(512, dtype=np.float32),
              "shirina_256": np.full(256, 0.1, dtype=np.float32)}[imya]
    bitoe = Face(box=(0.0, 0.0, 50.0, 50.0), landmarks=None, embedding=vektor,
                 detector="insight")
    c = FaceCache(tmp_path / "cache.sqlite3")
    c.put(photo, (1000, 100), "both", 2400, [bitoe, lico()])        # как писала старая версия
    assert c.get(photo, (1000, 100), "both", 2400) is None, imya
    assert c._db.execute("SELECT COUNT(*) FROM faces").fetchone()[0] == 0, \
        "непригодная строка должна удалиться, а не висеть мусором"
    c.put(photo, (1000, 100), "both", 2400, [lico()])               # переразбор чинится
    assert c.get(photo, (1000, 100), "both", 2400) is not None
    c.close()


def test_staroestrichnaya_stroka_s_bitoy_ramkoy_dast_None_i_ischeznet(
        tmp_path: Path, photo: Path) -> None:
    """Рамка с Inf приезжает из JSON целой, а лицо с ней роняет уже отрисовку.

    Оглавление переживает правку руками и старые версии приложения, поэтому битая
    рамка — такие же испорченные данные, как и битый вектор: строка удаляется, снимок
    разберётся заново, а `damage_count` считает её в отчёт про «оглавление починено».
    """
    bitoe = Face(box=(0.0, 0.0, float("inf"), float("inf")), landmarks=None,
                 embedding=np.ones(512, dtype=np.float32) / np.sqrt(512),
                 detector="insight")
    c = FaceCache(tmp_path / "cache.sqlite3")
    c.put(photo, (1000, 100), "both", 2400, [bitoe])       # JSON пишет Infinity честно
    assert c.get(photo, (1000, 100), "both", 2400) is None
    assert c.damage_count == 1, "битая рамка обязана считаться повреждением"
    assert c._db.execute("SELECT COUNT(*) FROM faces").fetchone()[0] == 0, \
        "непригодная строка должна удалиться, а не висеть мусором"
    c.put(photo, (1000, 100), "both", 2400, [lico()])      # переразбор чинится
    assert c.get(photo, (1000, 100), "both", 2400) is not None
    c.close()


def test_clear_udalyaet_vse_i_skazyvaet_skolko(tmp_path: Path, photo: Path) -> None:
    c = FaceCache(tmp_path / "cache.sqlite3")
    c.put(photo, (1000, 100), "both", 2400, [lico()])
    assert c.clear() == 1
    assert c.get(photo, (1000, 100), "both", 2400) is None
    c.close()


def test_pustoe_popadenie_ne_putaetsya_s_otkazom(tmp_path: Path, photo: Path) -> None:
    """Пустой список — это попадание в кэш. Проверять надо `is not None`, а не истинность:
    `if cached:` прочитал бы фото без лиц заново на каждом прогоне — ровно тот промах,
    которого оглавление и должно избегать.
    """
    c = FaceCache(tmp_path / "cache.sqlite3")
    c.put(photo, (1000, 100), "both", 2400, [])
    got = c.get(photo, (1000, 100), "both", 2400)
    assert got is not None
    assert not got                      # по истинности не отличить от промаха
    c.close()


def test_otchatok_vozvrahaetsya_float32_bez_poteri(tmp_path: Path, photo: Path) -> None:
    """Через SQLite и JSON отпечаток возвращается ровно тем же float32, цифра в цифру.

    Отпечаток собран через `linspace(..., dtype=np.float32)`, а не хелпером `lico()`:
    там `np.ones(512, float32) / np.sqrt(512)` — делитель float64, и по правилам numpy 2
    результат тоже float64. Контракт поля — float32, поэтому здесь ровно он и проверяется.
    """
    emb = np.linspace(-1.0, 1.0, 512, dtype=np.float32)
    face = Face(box=(0.0, 0.0, 100.0, 100.0), landmarks=None, embedding=emb, detector="insight")
    c = FaceCache(tmp_path / "cache.sqlite3")
    c.put(photo, (1000, 100), "both", 2400, [face])
    got = c.get(photo, (1000, 100), "both", 2400)
    assert got is not None
    assert got[0].embedding.dtype == np.float32
    assert np.array_equal(got[0].embedding, emb)
    c.close()


def test_chuzhoi_dtype_stanovitsya_float32(tmp_path: Path, photo: Path) -> None:
    """Если движок отдал float64, оглавление приводит его к float32, а не тянет дальше.

    Смысл: сравнение косинусами идёт по матрице из `embeddings_of`, она float32 —
    значит и в кэше должно лежать ровно то же, что увидит matcher, без тихого
    расхождения в последнем знаке между «до записи» и «после чтения».
    """
    face = lico()                                   # float64 — как раз случай хелпера брифа
    # Если хелпер починить под float32, этот тест обязан собрать свой float64-отпечаток:
    # он проверяет именно привод чужого dtype, а не удачное совпадение.
    assert face.embedding.dtype == np.float64
    expected = face.embedding.astype(np.float32)
    c = FaceCache(tmp_path / "cache.sqlite3")
    c.put(photo, (1000, 100), "both", 2400, [face])
    got = c.get(photo, (1000, 100), "both", 2400)
    assert got is not None
    assert np.array_equal(got[0].embedding, expected)
    assert got[0].embedding.dtype == np.float32
    c.close()


def test_stamp_iz_nanosekund_hranitsya_tochno(tmp_path: Path, photo: Path) -> None:
    """`file_stamp` отдаёт st_mtime_ns — это ~1,8 квинтиллиона, и ключ обязан вернуться целым.

    Плюс проверка на ±1 наносекунду: округление до секунд (st_mtime) молча вернуло бы
    устаревшие отпечатки для файла, перезаписанного в ту же секунду.
    """
    stamp = (1758940800123456789, 4_821_333)
    c = FaceCache(tmp_path / "cache.sqlite3")
    c.put(photo, stamp, "both", 2400, [lico()])
    assert c.get(photo, stamp, "both", 2400) is not None
    assert c.get(photo, (stamp[0] + 1, stamp[1]), "both", 2400) is None
    c.close()


def test_posle_pereotkrytiya_fayla_zapisi_ostayutsya(tmp_path: Path, photo: Path) -> None:
    """Смысл оглавления — переживать перезапуск приложения: закрыли, открыли, нашли."""
    way = tmp_path / "cache.sqlite3"
    first = FaceCache(way)
    first.put(photo, (1000, 100), "both", 2400, [lico(), lico(side=50)])
    first.close()
    second = FaceCache(way)
    got = second.get(photo, (1000, 100), "both", 2400)
    assert got is not None and [f.size for f in got] == [100.0, 50.0]
    assert second.clear() == 1
    second.close()


def test_povtornyy_put_zamenyaet_zapis_a_ne_dubit(tmp_path: Path, photo: Path) -> None:
    """Повторный снимок того же фото с тем же ключом обязан перезаписаться, а не упасть
    на PRIMARY KEY и не оставить две строки с разными лицами.
    """
    c = FaceCache(tmp_path / "cache.sqlite3")
    c.put(photo, (1000, 100), "both", 2400, [lico()])
    c.put(photo, (1000, 100), "both", 2400, [lico(), lico(side=50)])
    got = c.get(photo, (1000, 100), "both", 2400)
    assert got is not None and len(got) == 2
    assert c.clear() == 1               # строк в оглавлении по-прежнему одна
    c.close()


def test_dva_foto_hranitsya_otdelno(tmp_path: Path, photo: Path) -> None:
    """Путь — часть ключа: два снимка не должны перетирать друг друга."""
    other = tmp_path / "drugoe.jpg"
    other.write_bytes(b"y" * 100)
    c = FaceCache(tmp_path / "cache.sqlite3")
    c.put(photo, (1000, 100), "both", 2400, [lico()])
    c.put(other, (1000, 100), "both", 2400, [])
    got = c.get(photo, (1000, 100), "both", 2400)
    assert got is not None and len(got) == 1
    assert c.get(other, (1000, 100), "both", 2400) == []
    assert c.clear() == 2
    c.close()


# --- число отброшенных отпечатков в строке -----------------------------------------
#
# Отбраковку считает разбор, а снимок из оглавления моделями не читается вовсе. Если
# число потерь живёт только в сводке прогона, второй (тёплый) запуск печатает
# «потерь нет» об архив, где лица уже теряли. Поэтому оно лежит в строке.


def test_dropped_perezhivaet_pereotkrytie_fayla(tmp_path: Path, photo: Path) -> None:
    """Число потерь — часть записи об оглавлении, а не часть прогона."""
    way = tmp_path / "cache.sqlite3"
    first = FaceCache(way)
    first.put(photo, (1000, 100), "both", 2400, [lico()], dropped=2)
    first.close()
    second = FaceCache(way)
    got = second.get_s_poterjami(photo, (1000, 100), "both", 2400)
    assert got is not None and len(got[0]) == 1 and got[1] == 2
    second.close()


def test_bez_parametra_dropped_zapisyvaetsya_nol(tmp_path: Path, photo: Path) -> None:
    """Старый вызов без числа потерь обязан дать ноль, а не «нет данных».

    Ноль в строке — это «смотрели и ничего не потеряли», отсутствие поля — «не знаем».
    Отчёт различает их, и молча подменить второе на первое нельзя.
    """
    c = FaceCache(tmp_path / "cache.sqlite3")
    c.put(photo, (1000, 100), "both", 2400, [lico()])
    got = c.get_s_poterjami(photo, (1000, 100), "both", 2400)
    assert got is not None and got[1] == 0
    c.close()


def get_payload(c: FaceCache, photo: Path) -> str:
    import json
    (tekst,) = c._db.execute("SELECT payload FROM faces").fetchone()
    return tekst


def test_chislo_poter_ne_prychet_i_ne_otricatelsynoe(tmp_path: Path,
                                                     photo: Path) -> None:
    """Отрицательное, дробное, NaN и не-число в поле потерь превращаются в 0.

    Поле читается из файла, который пережил правку руками. Падать из-за него на весь
    архив нельзя, а `int(True) == 1` или `int(-5)` значили бы «потери были», которых
    никто не видел.
    """
    c = FaceCache(tmp_path / "cache.sqlite3")
    for znachenie, ozhidanie in ((-5, 0), (2.9, 2), (True, 0), ("много", 0),
                                 (float("nan"), 0), (float("inf"), 0), (None, 0)):
        c.put(photo, (1000, 100), "both", 2400, [lico()], dropped=znachenie)
        got = c.get_s_poterjami(photo, (1000, 100), "both", 2400)
        assert got is not None and got[1] == ozhidanie, znachenie
    c.close()


def test_stroka_bez_polya_dropped_chitaetsya_kak_nol(tmp_path: Path,
                                                     photo: Path) -> None:
    """Запись текущей версии, но без поля потерь — это 0, а не испорченная строка.

    Лица в ней пригодные, и выбрасывать её из-за отсутствия справки — значит заставить
    человека перечитывать снимок моделями впустую.
    """
    import json
    c = FaceCache(tmp_path / "cache.sqlite3")
    c.put(photo, (1000, 100), "both", 2400, [lico()], dropped=4)
    data = json.loads(get_payload(c, photo))
    del data["dropped"]
    c._db.execute("UPDATE faces SET payload=?", (json.dumps(data),))
    c._db.commit()
    got = c.get_s_poterjami(photo, (1000, 100), "both", 2400)
    assert got is not None and len(got[0]) == 1 and got[1] == 0
    assert c.damage_count == 0
    c.close()


def test_v_keshe_net_baytov_kartinki(tmp_path: Path, photo: Path) -> None:
    """Оглавление хранит числа, а не снимки: личные фото не должны попадать в базу.

    Проверяю и схему (нет BLOB-столбцов), и содержимое (payload — текст), и что размер
    файла на диске остаётся скромным: строка с 512-мерным отпечатком измеряется
    килобайтами, а не мегабайтами.
    """
    way = tmp_path / "cache.sqlite3"
    c = FaceCache(way)
    c.put(photo, (1000, 100), "both", 2400, [lico()])
    db = sqlite3.connect(str(way))
    try:
        declared = {row[1].upper() for row in db.execute("PRAGMA table_info(faces)")}
        assert "BLOB" not in declared
        (kind,) = db.execute("SELECT typeof(payload) FROM faces").fetchone()
        assert kind == "text"
    finally:
        db.close()
    assert way.stat().st_size < 100_000
    c.close()


def test_odin_kesh_iz_drugogo_potoka(tmp_path: Path, photo: Path) -> None:
    """Задача 9 создаст оглавление в потоке интерфейса, а читать будет рабочий QThread.

    По умолчанию sqlite3 запрещает использовать соединение из чужого потока и бросает
    ProgrammingError — тест фиксирует, что кэш этого не делает.
    """
    c = FaceCache(tmp_path / "cache.sqlite3")
    c.put(photo, (1000, 100), "both", 2400, [lico()])
    seen: list[Face] = []
    errors: list[Exception] = []

    def rabotat() -> None:
        try:
            got = c.get(photo, (1000, 100), "both", 2400)
            if got is None:
                raise AssertionError("чужой поток не увидел запись")
            seen.extend(got)
            c.put(tmp_path / "iz_potoka.jpg", (7, 7), "both", 2400, [lico(side=50)])
        except Exception as exc:            # собран и разобран в основном потоке
            errors.append(exc)

    thread = threading.Thread(target=rabotat)
    thread.start()
    thread.join()
    assert not errors, errors
    assert [f.size for f in seen] == [100.0]
    assert c.get(tmp_path / "iz_potoka.jpg", (7, 7), "both", 2400) is not None
    c.close()


def test_odnovremennaya_zapis_nichego_ne_teryaet(tmp_path: Path) -> None:
    """Разбор идёт в рабочем потоке, а сброс оглавления — в потоке интерфейса: кто-то
    обязательно лезет в базу одновременно.

    Замер вне теста (6 потоков по 400 операций, доступ к соединению без сериализации):
    5 падений `SystemError: error return without exception set` и 475 записей вместо
    2400 — оглавление молча теряет снятые фото. Здесь та же проверка, только в меньшем
    объёме (4 потока по 50 операций), чтобы тест оставался долей секунды.
    """
    face = lico()
    c = FaceCache(tmp_path / "cache.sqlite3")
    threads_n, each = 4, 50
    errors: list[Exception] = []

    def rabotat(seed: int) -> None:
        try:
            for i in range(each):
                way = tmp_path / f"p{seed}_{i}.jpg"
                c.put(way, (1000 + i, 100), "both", 2400, [face])
                if c.get(way, (1000 + i, 100), "both", 2400) is None:
                    errors.append(AssertionError(f"потеряна запись {seed}/{i}"))
        except Exception as exc:                # собран и разобран в основном потоке
            errors.append(exc)

    workers = [threading.Thread(target=rabotat, args=(seed,)) for seed in range(threads_n)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert not errors, errors
    assert c.clear() == threads_n * each        # ни одна строка не растворилась
    c.close()
