"""Иконка приложения: рисуем сами, без внешних картинок и без скачивания.

Зачем скрипт в репозитории, а не только готовый файл. Иконка — вход сборки: без
`assets/icon.icns` PyInstaller не собирает `.app` вообще (см. `FindChild.spec`).
Значит у человека, склонировавшего проект, должен быть способ её пересобрать и
поправить, а не искать картинку в интернете. Отсюда же и тест на то, что файл лежит
в репозитории, — тот же приём, что с моделью YuNet.

Рисуем Pillow по той же причине, по какой поиск не доверяет «глазу»: результат
детерминирован, правится числом в константе и не зависит от сети.

Формат `.icns` собирает `iconutil` — он есть в самой macOS, ставить нечего.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

STORONA = 1024
RADUS_KORPUSA = 229            # ~22 % стороны — скругление в стиле macOS

# Верх и низ подложки: тёплый кремовый в персиковый. Иконка читается и на светлом,
# и на тёмном рабочем столе, потому что сама себе корпус.
VERH = (255, 246, 236)
NIZ = (255, 217, 186)

CVET_KOZHI = (242, 201, 164)
CVET_VOLOS = (90, 70, 54)
CVET_GLAZ = (58, 44, 34)
CVET_SHCHEK = (250, 186, 168)

# Тот же зелёный, что рисует рамку вокруг найденного лица в окне, — только спокойнее:
# чистый (0, 255, 0) на иконке в 16 px превращается в грязь.
CVET_RAMKI = (51, 204, 85)
DLINA_UGOLKA = 118
TORSHINA_RAMKI = 26
STORONA_RAMKI = 560            # квадрат «видоискателя» вокруг лица

# Центр мордочки и радиусы. Голова сдвинута вниз относительно центра рамки: волосы
# должны остаться внутри квадрата, иначе уголки «режут» причёску.
TSENTR = (STORONA // 2, 512)
R_GOLOVA = 196
R_UHO = 34


def _korpus() -> Image.Image:
    """Круглоквадрат с вертикальным градиентом и альфа-каналом по контуру."""
    korpus = Image.new("RGBA", (STORONA, STORONA), (0, 0, 0, 0))
    mask = Image.new("L", (STORONA, STORONA), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, STORONA - 1, STORONA - 1], radius=RADUS_KORPUSA, fill=255)

    stolbec = Image.new("RGB", (1, STORONA))
    for y in range(STORONA):
        dolya = y / (STORONA - 1)
        stolbec.putpixel(
            (0, y), tuple(round(a + (b - a) * dolya) for a, b in zip(VERH, NIZ)))
    fond = Image.new("RGB", (STORONA, STORONA))
    fond.paste(stolbec.resize((STORONA, STORONA)), (0, 0))
    korpus.paste(fond, (0, 0), mask)
    return korpus


def _litso(kartinka: Image.Image) -> None:
    """Мордочка: волосы полукругом, лицо, уши, глаза, щёки, улыбка."""
    x, y = TSENTR
    kist = ImageDraw.Draw(kartinka)

    kist.ellipse([x - R_GOLOVA - 12, y - R_GOLOVA - 12,
                  x + R_GOLOVA + 12, y + R_GOLOVA + 12], fill=CVET_VOLOS)
    for storona in (-1, 1):
        ux = x + storona * (R_GOLOVA - 4)
        kist.ellipse([ux - R_UHO, y - R_UHO, ux + R_UHO, y + R_UHO], fill=CVET_KOZHI)
    # Лицо ниже и шире волос: тёмным остаётся только верх — причёска. Если сунуть
    # лицо внутрь круга волос с одинаковым отступом со всех сторон, на иконке
    # получается тёмное кольцо вокруг головы, и это выглядит капюшоном.
    kist.ellipse([x - R_GOLOVA + 14, y - R_GOLOVA + 66,
                  x + R_GOLOVA - 14, y + R_GOLOVA + 12], fill=CVET_KOZHI)

    for storona in (-1, 1):
        ex = x + storona * 74
        kist.ellipse([ex - 17, y - 26, ex + 17, y + 26], fill=CVET_GLAZ)
        kist.ellipse([ex - 5, y - 19, ex + 5, y - 9], fill=(255, 255, 255))
        kx = x + storona * 96
        kist.ellipse([kx - 26, y + 44, kx + 26, y + 84], fill=CVET_SHCHEK)

    kist.arc([x - 78, y + 18, x + 78, y + 132], start=20, end=160,
             fill=CVET_GLAZ, width=16)


def _ramka(kartinka: Image.Image) -> None:
    """Четыре уголка видоискателя — тот жест, что делает приложение на найденном лице."""
    kist = ImageDraw.Draw(kartinka)
    polovina = STORONA_RAMKI // 2
    y = TSENTR[1]
    for storona_x in (-1, 1):
        for storona_y in (-1, 1):
            ugol_x = TSENTR[0] + storona_x * polovina
            ugol_y = y + storona_y * polovina
            kist.line([ugol_x, ugol_y, ugol_x - storona_x * DLINA_UGOLKA, ugol_y],
                      fill=CVET_RAMKI, width=TORSHINA_RAMKI)
            kist.line([ugol_x, ugol_y, ugol_x, ugol_y - storona_y * DLINA_UGOLKA],
                      fill=CVET_RAMKI, width=TORSHINA_RAMKI)


def narisovat(put: Path) -> Path:
    """Мастер-файл 1024×1024 в PNG."""
    kartinka = _korpus()
    _litso(kartinka)
    _ramka(kartinka)
    put.parent.mkdir(parents=True, exist_ok=True)
    kartinka.save(put, "PNG")
    return put


# Имена внутри .iconset требует `iconutil`: свой набор на каждый размер и на каждую
# плотность. Не угадаешь — иконка в Доке останется кругляшом из Pillow.
RAZMERY = {
    "icon_16x16.png": 16,
    "icon_16x16@2x.png": 32,
    "icon_32x32.png": 32,
    "icon_32x32@2x.png": 64,
    "icon_128x128.png": 128,
    "icon_128x128@2x.png": 256,
    "icon_256x256.png": 256,
    "icon_256x256@2x.png": 512,
    "icon_512x512.png": 512,
    "icon_512x512@2x.png": 1024,
}


def sdelat_icns(master: Path, ikonka: Path) -> Path:
    """Размножить мастер во все размеры и собрать `.icns` через `iconutil`."""
    if sys.platform != "darwin":
        raise SystemExit("`iconutil` есть только в macOS — .icns не собрать")
    set_ikonok = master.parent / "icon.iconset"
    if set_ikonok.exists():
        shutil.rmtree(set_ikonok)
    set_ikonok.mkdir(parents=True)
    karta = Image.open(master).convert("RGBA")
    for imja, storona in RAZMERY.items():
        karta.resize((storona, storona), Image.LANCZOS).save(set_ikonok / imja, "PNG")

    itog = subprocess.run(["iconutil", "-c", "icns", str(set_ikonok),
                           "-o", str(ikonka)], capture_output=True, text=True)
    if itog.returncode != 0:
        raise SystemExit(f"iconutil отказался собирать иконку: {itog.stderr.strip()}")
    shutil.rmtree(set_ikonok)
    return ikonka


if __name__ == "__main__":
    koren = Path(__file__).resolve().parent
    mash = narisovat(koren / "assets" / "icon.png")
    print(mash)
    print(sdelat_icns(mash, koren / "assets" / "icon.icns"))
