"""Окно просмотра фото по двойному клику.

Проверяется то, что ломается именно здесь:

1. **Рамка лица в полном размере совпадает с лицом.** Координаты приходят в пикселях
   кадра `max_dim`; если масштаб взять от другого чтения, рамка уедет и человек решит,
   что нашёлся не тот человек.
2. **Фото вписано в окно, а не обрезано.** Пропорции обязаны сохраниться при любом
   размере окна: обрезанное фото для «мой ребёнок или нет» бесполезно.
3. **Крупный квадрат лица — то же лицо, что в рамке**, а не первое лицо снимка.
4. **Галочка не имеет второго владельца.** Отметка живёт в сетке; окно показывает и
   просит. Разъезд = «копировать» не то, что отмечено.
5. **Нет лица — нет правой половины**, и словами сказано почему, а не показан чёрный
   квадрат.
6. **Нечитаемый файл не открывает окна** — причина называется.
9. **Двойной клик по карточке доходит до окна**, а не тонет в виджете: этот путь мыши
   нельзя проверить записью в модель (урок T3).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PIL import Image                                        # noqa: E402

from ui.photo_view import (STORONA_KROP, RamkaLica, kad_dlya_prosmotra,  # noqa: E402
                           nadpisi_bez_lica)


# --- синтетические снимки ---------------------------------------------------------------

RAZMER = (1200, 800)          # родной снимок: ширина, высота
MAX_DIM = 800                 # кадр, в пикселях которого приходит рамка лица
K_RODNOJ = RAZMER[0] / MAX_DIM  # во столько раз родной кадр крупнее кадра моделей

# Лицо в координатах КАДРА max_dim — ровно тот язык, на котором `ResultRow.box` говорит
# с интерфейсом. Родные пиксели сюда подставлять нельзя: рамка легла бы мимо лица, и
# тест ловил бы не сдвиг, а собственную путаницу в масштабах.
LITSO = (300.0, 150.0, 390.0, 270.0)


def narisovat(tmp_path: Path, imja: str = "a.jpg",
              box: tuple[float, float, float, float] | None = LITSO,
              razmer: tuple[int, int] = RAZMER) -> Path:
    """Снимок с тёмным прямоугольником там, где «лицо», на светлом фоне.

    Тёмная фигура — независимый ориентир: по её пикселям видно, куда легла рамка, без
    всякого «на глаз по картинке». Рисуем в родных координатах, поэтому рамку
    пересчитываем из кадра `max_dim` обратно.
    """
    kadr = np.full((razmer[1], razmer[0], 3), 220, dtype=np.uint8)
    if box is not None:
        x1, y1, x2, y2 = (int(v * K_RODNOJ) for v in box)
        kadr[y1:y2, x1:x2] = 20
    put = tmp_path / imja
    Image.fromarray(kadr).save(put, "JPEG", quality=95)
    return put


# --- чистая подготовка кадра (без Qt) ---------------------------------------------------


def test_kadr_i_krop_prichodjat_v_razmere_kadra_max_dim(tmp_path) -> None:
    """Длина кадровой стороны обязана равняться `max_dim`: рамка пришла в пикселях
    именно этого кадра, и любое другое чтение сдвинуло бы её с лица."""
    put = narisovat(tmp_path)
    photo, krop, beda = kad_dlya_prosmotra(put, [RamkaLica(LITSO, 1)], 1, MAX_DIM)

    assert beda is None
    assert max(photo.shape[:2]) == MAX_DIM, photo.shape
    assert krop.shape[0] == krop.shape[1] == STORONA_KROP


def test_ramka_lezhit_na_lice_a_ne_ryadom(tmp_path) -> None:
    """Рамка обязана лечь ровно на переданные координаты и ни на йоту правее или левее.

    Тёмное лицо в этом снимке занимает ровно те же пиксели, что и `LITSO`: сверяем
    прямоугольник зелёных пикселей с ожидаемым и с тем, где действительно лежит лицо.
    """
    put = narisovat(tmp_path, box=LITSO)
    photo, _, _ = kad_dlya_prosmotra(put, [RamkaLica(LITSO, 1)], 1, MAX_DIM)

    zelenyj = (photo[:, :, 1] > 200) & (photo[:, :, 0] < 80) & (photo[:, :, 2] < 80)
    ys, xs = np.where(zelenyj)
    assert len(xs) > 0, "рамки на кадре нет вовсе"
    gotovoe = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    ozhidanie = [int(v) for v in LITSO]
    slozhno = max(abs(a - b) for a, b in zip(ozhidanie, gotovoe))
    assert slozhno <= 3, f"рамка уехала: ожидалось {ozhidanie}, получено {gotovoe}"

    temnoe = (photo[:, :, :3].mean(axis=2) < 100)
    ty, tx = np.where(temnoe)
    litso = [int(tx.min()), int(ty.min()), int(tx.max()), int(ty.max())]
    # лицо в кадре max_dim = LITSO без округления до пикселя рамки
    assert max(abs(a - b) for a, b in zip(ozhidanie, litso)) <= 3, \
        f"рамка не на лице: рамка {ozhidanie}, лицо {litso}"


def test_krop_vyrezan_po_toy_zhe_ramke(tmp_path) -> None:
    """Крупный квадрат вырезан вокруг того же лица: тёмная фигура обязана занять его
    центр, иначе справа показывают чужое лицо."""
    put = narisovat(tmp_path, box=LITSO)
    _, krop, _ = kad_dlya_prosmotra(put, [RamkaLica(LITSO, 1)], 1, MAX_DIM)

    temnoe = krop.mean(axis=2) < 100
    ds = float(temnoe.mean())
    assert 0.25 < ds < 0.95, f"тёмное занимает {ds:.2f} квадрата — это не то лицо"
    ys, xs = np.where(temnoe)
    assert abs(float(xs.mean()) - STORONA_KROP / 2) < STORONA_KROP * 0.15, \
        "лицо не в центре квадрата — кроп взят с другого места"
    assert abs(float(ys.mean()) - STORONA_KROP / 2) < STORONA_KROP * 0.15


def test_bez_lica_foto_idet_a_kropa_net(tmp_path) -> None:
    """Фото без найденного лица показывать можно, а врать про лицо — нельзя: krop=None,
    и вызывающий прячет правую половину вместо чёрного квадрата."""
    put = narisovat(tmp_path, box=None)
    photo, krop, beda = kad_dlya_prosmotra(put, [], None, 800)

    assert beda is None and photo is not None
    assert krop is None


def test_nechitaemyj_fail_nazyvaet_prichinu(tmp_path) -> None:
    """Пустой файл — не «лицо не найдено», а «файл не читается»: два разных разговора
    и два разных действия человека."""
    put = tmp_path / "pustom.jpg"
    put.write_bytes(b"")
    photo, krop, beda = kad_dlya_prosmotra(put, [], None, 800)

    assert photo is None and krop is None
    assert beda and str(put) in beda, f"в причине нет имени файла: {beda}"


def test_otsutstvuyushchij_fail_ne_padayet(tmp_path) -> None:
    """Файла нет (переместили за время разбора) — ответ текстом, а не исключение:
    вызов идёт из слота Qt, и необработанное исключение было бы молчаливой кнопкой."""
    photo, krop, beda = kad_dlya_prosmotra(tmp_path / 'neta_kde.jpg', [], None, 800)
    assert photo is None and beda


def test_slova_bez_lica_ne_zaprescheny() -> None:
    """Подпись «лица нет» не содержит запрещённого интерфейсного словаря и не обещает
    то, чего не видит человек."""
    niz = nadpisi_bez_lica().lower()
    for stem in ("порог", "уверенност", "embedd", "косинус", "детектор"):
        assert stem not in niz, f"«{stem}» в тексте: {niz!r}"
    assert "не найдено" in niz, "не сказано, что лица нет"
    assert "смотрите снимок целиком" in niz, \
        "человеку не сказано, что делать со снимком без лица"


# --- окно: содержимое -----------------------------------------------------------------

from PySide6.QtCore import QPoint, Qt                    # noqa: E402
from PySide6.QtTest import QTest                         # noqa: E402
from PySide6.QtWidgets import QApplication               # noqa: E402

from ui.photo_view import PhotoViewDialog                # noqa: E402


@pytest.fixture
def okno(qapp) -> PhotoViewDialog:
    dlg = PhotoViewDialog()
    yield dlg
    dlg.close()
    dlg.deleteLater()


def test_posle_pokaza_est_foto_i_est_krop(okno, tmp_path) -> None:
    put = narisovat(tmp_path, box=LITSO)
    okno.pokazat(put, [RamkaLica(LITSO, 1)], 1, MAX_DIM, procent=62, poziciya=1, vsego=3, kopiruetsya=True)

    assert okno.pokazyvaet == put
    assert not okno.foto.pixmap().isNull(), "фото в окне не показалось"
    assert not okno.krop.pixmap().isNull(), "крупного лица в окне нет"
    assert okno.krop.isVisible()
    assert okno.copy_check.isChecked() is True


def test_podpis_govorit_skolko_lic_v_kadre_a_ne_skolko_obvedeno(okno, tmp_path) -> None:
    """«Лицо 22 из 24» — это про кадр, а не про число рамок.

    Рамка с задачи T19 стоит только на найденных лицах, и на групповом снимке их две из
    двадцати четырёх. Считать «из» по длине присланного списка — значит показать
    «лицо 22 из 2» и оставить человека без ответа, сколько людей в кадре.
    """
    put = narisovat(tmp_path, box=LITSO)
    vtoroe = (LITSO[0] + 400.0, LITSO[1], LITSO[2] + 400.0, LITSO[3])
    okno.pokazat(put, [RamkaLica(LITSO, 22), RamkaLica(vtoroe, 24)], 22, MAX_DIM,
                 procent=65, poziciya=2, vsego=131, kopiruetsya=True, vsego_lic=24)
    assert "лицо 22 из 24" in okno.slovo.text(), okno.slovo.text()


def test_bez_chisla_lic_v_kadre_okno_schitaet_po_chislu_ramok(okno,
                                                                     tmp_path) -> None:
    """`vsego_lic` не передали — окно считает по своим рамкам, как умело всегда.

    Так зовёт окно выбора эталона: там обведены все лица снимка, и длина списка —
    честное число лиц. Падать или врать из-за отсутствующего аргумента окно не должно.
    """
    put = narisovat(tmp_path, box=LITSO)
    okno.pokazat(put, [RamkaLica(LITSO, 1)], 1, MAX_DIM, procent=62, poziciya=1, vsego=3,
                 kopiruetsya=None)
    assert "лицо 1 из 1" in okno.slovo.text(), okno.slovo.text()


def test_proporcii_snimka_sohraneny(okno, tmp_path) -> None:
    """Вписанное фото обязано сохранить отношение сторон: обрезанный или растянутый
    снимок для «мой ребёнок или нет» не годится."""
    put = narisovat(tmp_path, box=LITSO)
    okno.pokazat(put, [RamkaLica(LITSO, 1)], 1, MAX_DIM, procent=62, poziciya=1, vsego=3, kopiruetsya=False)

    pokazannoe = okno.foto.pixmap().size()
    ozhidaemoe = RAZMER[0] / RAZMER[1]      # 1200x800 -> 3:2, каким кадр и пришёл
    fakt = pokazannoe.width() / pokazannoe.height()
    assert abs(fakt - ozhidaemoe) < 0.02, \
        f"пропорции съехали: {pokazannoe.width()}x{pokazannoe.height()} = {fakt:.3f}"
    assert pokazannoe.width() <= okno.foto.width() + 1, "фото шире отпущенного места"
    assert pokazannoe.height() <= okno.foto.height() + 1, "фото выше отпущенного места"


def test_bez_lica_pravaya_polovina_skryta_i_skazany_slova(okno, tmp_path) -> None:
    """Вместо лица нельзя показывать чёрный квадрат: человек прочитает его как «лицо
    не похоже», хотя лица там просто искали и не нашли."""
    put = narisovat(tmp_path, box=None)
    okno.pokazat(put, [], None, MAX_DIM, procent=12, poziciya=1, vsego=3, kopiruetsya=False)

    assert okno.krop.isHidden(), "пустой квадрат лица на месте"
    assert not okno.foto.pixmap().isNull(), "само фото тоже пропало"
    assert "не найдено" in okno.slovo.text().lower()


def test_nechitaemyj_fail_nazyvaet_prichinu_i_ne_pokazyvaet_kadr(okno, tmp_path) -> None:
    put = tmp_path / "pustom.jpg"
    put.write_bytes(b"")
    okno.pokazat(put, [RamkaLica(LITSO, 1)], 1, MAX_DIM, procent=62, poziciya=1, vsego=3, kopiruetsya=False)

    assert str(put) in okno.slovo.text(), f"в причине нет имени файла: {okno.slovo.text()}"
    assert okno.foto.pixmap().isNull()


# --- окно: галочка ---------------------------------------------------------------------


def test_galochka_otkryvaetsja_s_sostojaniem_setki_i_ne_shumit(okno, tmp_path) -> None:
    """Открытие окна с уже отмеченным фото НЕ имеет права выслать наружу «человек
    решил»: сетка получила бы обратно то, что только что сказала.

    Отдельно проверяется урок T3: защита от программной смены состояния обязана жить
    строго внутри одного вызова. Если сделать её признаком «уже трогали», она переживёт
    первый клик человека, и первое нажатие молча пропадёт.
    """
    put = narisovat(tmp_path, box=LITSO)
    polucheno: list = []
    okno.razresheno.connect(lambda p, z: polucheno.append((p, z)))

    okno.pokazat(put, [RamkaLica(LITSO, 1)], 1, MAX_DIM, procent=62, poziciya=1, vsego=3, kopiruetsya=True)
    assert polucheno == [], f"окно шумит при открытии: {polucheno}"

    QTest.mouseClick(okno.copy_check, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier,
                     okno.copy_check.rect().center())
    assert polucheno == [(put, False)], \
        f"первый клик по галочке не дошёл: {polucheno}"


def test_sinkhronizacia_bez_otveta(okno, tmp_path) -> None:
    """Сетка изменила отметку — окно показывает новое, но наружу ничего не шлёт:
    иначе изменение вернётся в сетку эхом и отметка закачается."""
    put = narisovat(tmp_path, box=LITSO)
    okno.pokazat(put, [RamkaLica(LITSO, 1)], 1, MAX_DIM, procent=62, poziciya=1, vsego=3, kopiruetsya=True)
    polucheno: list = []
    okno.razresheno.connect(lambda p, z: polucheno.append(z))

    okno.sinkhronizirat(put, False)
    assert okno.copy_check.isChecked() is False
    assert polucheno == []

    # чужой файл не имеет права переписать открытое окно
    okno.sinkhronizirat(Path("/drugoe.jpg"), True)
    assert okno.copy_check.isChecked() is False


def test_povtornyj_pokaz_zamenyaet_soderzhimoe(okno, tmp_path) -> None:
    """Одно окно на все просмотры: второй файл обязан вытеснить первый, а не накопить
    два кадра в одном QLabel."""
    pervoe = narisovat(tmp_path, "pervoe.jpg", box=LITSO)
    vtoroe = narisovat(tmp_path, "vtoroe.jpg", box=None)
    okno.pokazat(pervoe, [RamkaLica(LITSO, 1)], 1, MAX_DIM, procent=62, poziciya=1, vsego=3, kopiruetsya=True)
    okno.pokazat(vtoroe, [], None, MAX_DIM, procent=12, poziciya=2, vsego=3, kopiruetsya=False)

    assert okno.pokazyvaet == vtoroe
    assert okno.copy_check.isChecked() is False
    assert okno.krop.isHidden()
    assert vtoroe.name in okno.windowTitle()


# --- окно: навигация -------------------------------------------------------------------


def test_na_forme_est_procent_i_poziciya_v_spiske(okno, tmp_path) -> None:
    """Процент обязан быть виден в окне, а не только на карточке: человек смотрит на
    крупный кадр и решает «моё / не моё», а рядом должно быть то, что нашёл поиск.

    Позиция «N из M» нужна, чтобы стрелки не были прыжком в неизвестность: без неё
    человек не знает, сколько ещё кадров впереди и дошёл ли он до конца.
    """
    put = narisovat(tmp_path, box=LITSO)
    okno.pokazat(put, [RamkaLica(LITSO, 1)], 1, MAX_DIM, procent=62, poziciya=7, vsego=131, kopiruetsya=True)

    tekst = okno.slovo.text()
    assert "62" in tekst, f"процента нет на форме: {tekst!r}"
    assert "7" in tekst and "131" in tekst, f"позиции нет на форме: {tekst!r}"


def test_strelki_prosjat_perehod_a_ne_taschut_fajly_sami(okno, tmp_path) -> None:
    """Окно не знает список находок — оно просит шаг. Владелец порядка — сетка, и если
    окно начнёт ходить по своему copy списка, оно разъедется с тем, что на экране."""
    put = narisovat(tmp_path, box=LITSO)
    zaprosy: list = []
    okno.prosyat_perehod.connect(lambda shag: zaprosy.append(shag))
    okno.pokazat(put, [RamkaLica(LITSO, 1)], 1, MAX_DIM, procent=62, poziciya=2, vsego=3, kopiruetsya=True)

    QTest.mouseClick(okno.prev_button, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, okno.prev_button.rect().center())
    QTest.mouseClick(okno.next_button, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, okno.next_button.rect().center())

    assert zaprosy == [-1, 1], f"стрелки шлют не то: {zaprosy}"


def test_kraine_spiska_gasyat_strelki(okno, tmp_path) -> None:
    """На первом кадре «назад» некуда, на последнем — «вперёд». Живая стрелка, которая
    по нажатию не делает ничего, выглядит как сломанная программа."""
    put = narisovat(tmp_path, box=LITSO)
    okno.pokazat(put, [RamkaLica(LITSO, 1)], 1, MAX_DIM, procent=62, poziciya=1, vsego=1, kopiruetsya=True)
    assert okno.prev_button.isEnabled() is False
    assert okno.next_button.isEnabled() is False

    okno.pokazat(put, [RamkaLica(LITSO, 1)], 1, MAX_DIM, procent=62, poziciya=1, vsego=5, kopiruetsya=True)
    assert okno.prev_button.isEnabled() is False
    assert okno.next_button.isEnabled() is True

    okno.pokazat(put, [RamkaLica(LITSO, 1)], 1, MAX_DIM, procent=62, poziciya=5, vsego=5, kopiruetsya=True)
    assert okno.prev_button.isEnabled() is True
    assert okno.next_button.isEnabled() is False


def test_odin_okno_na_vse_peremeshcheniya(okno, tmp_path) -> None:
    """Стрелки листают в этом же окне: каждый новый кадр — не новое окно поверх старого."""
    pervyj = narisovat(tmp_path, "pervyj.jpg", box=LITSO)
    vtoroj = narisovat(tmp_path, "vtoroj.jpg", box=LITSO)
    okno.pokazat(pervyj, [RamkaLica(LITSO, 1)], 1, MAX_DIM, procent=62, poziciya=1, vsego=2, kopiruetsya=True)
    okno.pokazat(vtoroj, [RamkaLica(LITSO, 1)], 1, MAX_DIM, procent=40, poziciya=2, vsego=2, kopiruetsya=False)

    assert okno.pokazyvaet == vtoroj
    assert okno.copy_check.isChecked() is False
    assert "40" in okno.slovo.text()
    assert vtoroj.name in okno.windowTitle()


# --- T12: все лица на снимке, каждое под своим номером ----------------------------------

LITSO_A = (100.0, 100.0, 200.0, 230.0)
LITSO_B = (600.0, 120.0, 760.0, 300.0)
LITSO_C = (350.0, 420.0, 470.0, 560.0)


def narisovat_tri(tmp_path, imja="tri.jpg"):
    """Снимок с тремя тёмными лицами в разных углах кадра."""
    kadr = np.full((600, 800, 3), 210, dtype=np.uint8)
    for box in (LITSO_A, LITSO_B, LITSO_C):
        x1, y1, x2, y2 = (int(v) for v in box)
        kadr[y1:y2, x1:x2] = 20
    put = tmp_path / imja
    Image.fromarray(kadr).save(put, "JPEG", quality=95)
    return put


def test_ramki_na_vseh_licah_a_ne_tolko_na_vybrannom(tmp_path) -> None:
    """В окне выбора эталона человек решает «какое из трёх лиц — мой ребёнок», и рамка
    обязана быть на каждом: одна рамка не отвечает на вопрос, а три — отвечает."""
    put = narisovat_tri(tmp_path)
    ramki = [RamkaLica(LITSO_A, 1), RamkaLica(LITSO_B, 2), RamkaLica(LITSO_C, 3)]
    photo, _, beda = kad_dlya_prosmotra(put, ramki, 2, MAX_DIM)

    assert beda is None
    zelenyj = (photo[:, :, 1] > 200) & (photo[:, :, 0] < 80) & (photo[:, :, 2] < 80)
    naidennye = []
    for box in (LITSO_A, LITSO_B, LITSO_C):
        x1, y1, x2, y2 = (int(v) for v in box)
        naidennye.append(bool(zelenyj[max(0, y1 - 4):y2 + 5, max(0, x1 - 4):x2 + 5].any()))
    assert naidennye == [True, True, True], f"рамки легли не на все лица: {naidennye}"


def test_krop_beretsya_iz_vybrannogo_lica_a_ne_iz_pervogo(tmp_path) -> None:
    """Крупный квадрат обязан соответствовать лицу, по которому кликнули. Иначе человек
    смотрит на одного ребёнка, а отмечает совсем другого."""
    put = narisovat_tri(tmp_path)
    ramki = [RamkaLica(LITSO_A, 1), RamkaLica(LITSO_B, 2), RamkaLica(LITSO_C, 3)]

    _, krop_vtorogo, _ = kad_dlya_prosmotra(put, ramki, 2, MAX_DIM)
    _, krop_tretego, _ = kad_dlya_prosmotra(put, ramki, 3, MAX_DIM)

    # лица в разных углах кадра, поэтому их кропы не могут совпасть
    assert not np.array_equal(krop_vtorogo, krop_tretego), "кроп не зависит от выбора"
    # правое лицо лежит справа в кадре: в его кропе тёмное смещено вправо
    xs_pravo = np.where(krop_vtorogo.mean(axis=2) < 100)[1].mean()
    xs_levo = np.where(krop_tretego.mean(axis=2) < 100)[1].mean()
    assert xs_pravo > xs_levo, "квадраты перепутались между лицами"


def test_bez_vybrannogo_lica_kropa_net_a_ramki_ostajutsja(tmp_path) -> None:
    """Список лиц показать можно, а вырезать нечего: выбранного номера нет."""
    put = narisovat_tri(tmp_path)
    ramki = [RamkaLica(LITSO_A, 1), RamkaLica(LITSO_B, 2)]
    photo, krop, beda = kad_dlya_prosmotra(put, ramki, None, MAX_DIM)

    assert beda is None and krop is None and photo is not None


def test_vybrannyj_nomer_vne_spiska_ne_lomayet_kadr(tmp_path) -> None:
    """Между кликом и пересчётом список лиц мог перестроиться. Несуществующий номер —
    это «нет кропа», а не падение окна."""
    put = narisovat_tri(tmp_path)
    photo, krop, beda = kad_dlya_prosmotra(put, [RamkaLica(LITSO_A, 1)], 7, MAX_DIM)
    assert beda is None and krop is None and photo is not None


def test_okno_etalona_beze_galochki_kopirovaniya(okno, tmp_path) -> None:
    """В окне выбора эталона галочка «копировать» не имеет смысла: там решают, кого
    искать, а не что переносить. Живая галочка там была бы обещанием действия,
    которого окно не делает."""
    put = narisovat_tri(tmp_path)
    ramki = [RamkaLica(LITSO_A, 1), RamkaLica(LITSO_B, 2)]
    okno.pokazat(put, ramki, 2, MAX_DIM, procent=18, poziciya=5, vsego=16)

    assert okno.copy_check.isHidden(), "галочка копирования показана там, где её быть не должно"


def test_okno_etalona_pokazyvaet_nomer_lica_na_snimke(okno, tmp_path) -> None:
    """«Лицо 2 из 3» обязано быть на форме: без него человек не свяжет увиденное с
    карточкой, которую он кликнул."""
    put = narisovat_tri(tmp_path)
    ramki = [RamkaLica(LITSO_A, 1), RamkaLica(LITSO_B, 2), RamkaLica(LITSO_C, 3)]
    okno.pokazat(put, ramki, 2, MAX_DIM, procent=18, poziciya=5, vsego=16)

    tekst = okno.slovo.text()
    assert "лицо 2 из 3" in tekst, f"номера лица нет на форме: {tekst!r}"
    assert "18" in tekst and "5" in tekst and "16" in tekst


# --- по какому выбранному лицу нашёлся снимок (T15) ---------------------------------------
#
# Отдельное число на форме, а не на карточке: человек решает «мой / не мой», глядя сюда,
# на крупный кадр. Имена строк эталона окну ПРИНОСЯТ числом — свою арифметику оно не
# знает и не заводит второго владельца выбора лиц.


def test_podpis_govorit_kto_najden(okno, tmp_path) -> None:
    """Подпись содержит «найден: мама» и стоит между позицией в списке и процентом.

    Порядок важен: имя объясняет число, которое идёт правее, а не висит в конце строки
    отдельным фактом. Прежнее «по лицу 2» называло отметку человека, а не человека.
    """
    put = narisovat(tmp_path, box=LITSO)
    okno.pokazat(put, [RamkaLica(LITSO, 1)], 1, MAX_DIM, procent=45, poziciya=3, vsego=131,
                 kopiruetsya=True, imya_cheloveka="мама")

    tekst = okno.slovo.text()
    assert "найден: мама" in tekst, f"имени человека на форме нет: {tekst!r}"
    assert "45" in tekst and "3 из 131" in tekst, f"старых частей подписки нет: {tekst!r}"
    assert tekst.index("3 из 131") < tekst.index("найден: мама") < tekst.index("45"), tekst


def test_imya_cheloveka_peredajutsja_slovom_kak_ego_nazval_roditel(okno, tmp_path) -> None:
    """Окно не переименовывает человека: как назвали в шаге 2, так и написано.

    Номер вместо имени — это ровно тот дефект, из-за которого правка и началась:
    «лицо 9» не отвечает родителю на вопрос, кто на снимке.
    """
    put = narisovat(tmp_path, box=LITSO)
    okno.pokazat(put, [RamkaLica(LITSO, 1)], 1, MAX_DIM, procent=62, poziciya=1, vsego=9,
                 kopiruetsya=True, imya_cheloveka="Соня")

    tekst = okno.slovo.text()
    assert "найден: Соня" in tekst, f"имя переиначено: {tekst!r}"
    assert "лицо 1 ·" not in tekst.replace("лицо 1 из", ""), tekst


def test_bez_imeni_cheloveka_podpis_kak_byla(okno, tmp_path) -> None:
    """На кадре не нашлось никого из искомых — и подписи про человека нет.

    При одном искомом человеке главное окно молчит об имени (сетка считает, что это
    шум), и окно не имеет права выдумать «найден: » с пустым местом.
    """
    put = narisovat(tmp_path, box=LITSO)
    for pustoe in (None, ""):
        okno.pokazat(put, [RamkaLica(LITSO, 1)], 1, MAX_DIM, procent=62, poziciya=1,
                     vsego=9, kopiruetsya=True, imya_cheloveka=pustoe)
        tekst = okno.slovo.text()
        assert "найден" not in tekst, f"подпись появилась без имени: {tekst!r}"
        assert "62" in tekst, f"процент поехал: {tekst!r}"




def test_chuzhoj_nomer_stroki_ne_lomayet_podpis(okno, tmp_path) -> None:
    """Между пересчётом и показом выбор мог сократиться: несуществующий номер строки
    не имеет права ни на подпись, ни на падение окна — это слот Qt, и необработанное
    исключение было бы молчаливой кнопкой."""
    put = narisovat(tmp_path, box=LITSO)
    okno.pokazat(put, [RamkaLica(LITSO, 1)], 1, MAX_DIM, procent=62, poziciya=1, vsego=9,
                 kopiruetsya=True, imya_cheloveka=9)

    assert "лицу 9" in okno.slovo.text() or "найдено по" not in okno.slovo.text()
    assert not okno.foto.isHidden()


# --- окно поверх остальных: показ обязан вернуть его вперёд ------------------------------


def test_pokaz_vydvigaet_okno_esli_ono_ostalos_za_roditelyem(qapp, tmp_path) -> None:
    """Окно просмотра уехало за главное — двойной клик обязан вернуть его на перед.

    `show()` на уже видимом окне не делает ничего, и прежнее окно молча меняло
    содержимое там, где человек его не видит: он жмёт на карточку, а на экране не
    происходит ровным счётом ничего. Отличаем это по активному окну приложения.
    """
    from PySide6.QtWidgets import QWidget

    roditel = QWidget()
    roditel.show()
    dlg = PhotoViewDialog(roditel)
    put = narisovat(tmp_path, box=LITSO)
    dlg.pokazat(put, [RamkaLica(LITSO, 1)], 1, MAX_DIM, procent=62, poziciya=1,
                vsego=3, kopiruetsya=True)
    qapp.processEvents()
    assert qapp.activeWindow() is dlg, "первый показ не вывел окно"

    roditel.activateWindow()
    qapp.processEvents()
    assert qapp.activeWindow() is roditel, "родитель не забрал активость — тест слепой"

    dlg.pokazat(put, [RamkaLica(LITSO, 1)], 1, MAX_DIM, procent=62, poziciya=1,
                vsego=3, kopiruetsya=True)
    qapp.processEvents()
    assert qapp.activeWindow() is dlg, "повторный показ оставил окно за родителем"

    dlg.close()
    roditel.close()


def test_pokaz_drugogo_foto_vydvigaet_okno_esli_ono_ostalos_za_roditelyem(
        qapp, tmp_path) -> None:
    """Другой снимок в уехавшем окне: показать мало — окно должно вернуться на перед.

    Второй половины той же жалобы: человек дважды кликает по новому фото, данные
    переставились, а кадр остался спрятанным под главным окном.
    """
    from PySide6.QtWidgets import QWidget

    roditel = QWidget()
    roditel.show()
    dlg = PhotoViewDialog(roditel)
    pervoe = narisovat(tmp_path, "pervoe.jpg", box=LITSO)
    vtoroe = narisovat(tmp_path, "vtoroe.jpg", box=LITSO)
    dlg.pokazat(pervoe, [RamkaLica(LITSO, 1)], 1, MAX_DIM, procent=62, poziciya=1,
                vsego=3, kopiruetsya=True)
    roditel.activateWindow()
    qapp.processEvents()

    dlg.pokazat(vtoroe, [RamkaLica(LITSO, 1)], 1, MAX_DIM, procent=62, poziciya=2,
                vsego=3, kopiruetsya=True)
    qapp.processEvents()
    assert dlg.pokazyvaet == vtoroe
    assert qapp.activeWindow() is dlg, "новое фото показали, но окно осталось за родителем"

    dlg.close()
    roditel.close()
