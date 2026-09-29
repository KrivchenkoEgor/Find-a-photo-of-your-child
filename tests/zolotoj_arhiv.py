"""Где лежит боевой архив: в `arhiv.local.json`, а не в репозитории.

Почему это вынесено из теста. Боевой прогон сверяется со списком снимков, на которых
человек глазами подтвердил ребёнка. Имена этих файлов — часть семейного архива: по ним
вместе с логином на GitHub можно восстановить, чьи именно фото разбирают, а сами снимки
при этом в репозиторий не попадают и кажутся в безопасности. Значит не в репозитории
должны лежать и имена, и путь к папке.

Файл `arhiv.local.json` закрыт в `.gitignore`, его образец — `arhiv.example.json`.
Без файла боевой тест не падает и не «проходит молча»: он пропускается с причиной, в
которой названо, что скопировать.
"""

from __future__ import annotations

import json
from pathlib import Path

KOREN = Path(__file__).resolve().parents[1]
LOKALNYJ = KOREN / "arhiv.local.json"
OBRAZETS = "arhiv.example.json"

# Те же поля, что в образце. `photos` и `reference` — пути от корня репозитория, чтобы
# локальный конфиг можно было перенести вместе с папкой.
Polya = ("photos", "reference", "anchor", "confirmed")


class Arhiv:
    """Один прочитанный боевой архив: папки, эталонный снимок и подтверждённый список."""

    def __init__(self, dannye: dict, koren: Path = KOREN) -> None:
        net_polya = [p for p in Polya if p not in dannye]
        if net_polya:
            raise ValueError(
                f"в {LOKALNYJ.name} нет полей {', '.join(net_polya)} — "
                f"сравните с {OBRAZETS}")
        self.photos = koren / dannye["photos"]
        self.ref = koren / dannye["reference"]
        self.anchor = dannye["anchor"]
        self.confirmed = frozenset(dannye["confirmed"])

    @property
    def net_na_meste(self) -> str:
        """Пустая строка, если архив можно разбирать; иначе — что не так."""
        if not self.photos.is_dir():
            return f"нет папки с фото: {self.photos}"
        if not self.ref.is_dir():
            return f"нет папки с эталонными снимками: {self.ref}"
        if not self.confirmed:
            return (f"в {LOKALNYJ.name} пуст список `confirmed` — сверять не с чем, "
                    "см. образец")
        return ""


def prochitat() -> Arhiv | None:
    """Локальный архив или None, если файла нет. Ошибку не бросаем намеренно.

    Файл может отсутствовать по-хорошему: на свежем клоне, на машине без личных фото,
    в контейнере сборки. Падение в этом случае отучило бы запускать `pytest` целиком, а
    молчаливый «зелёный» прогон без архива был бы хуже — тест ничего не проверил.
    """
    if not LOKALNYJ.is_file():
        return None
    return Arhiv(json.loads(LOKALNYJ.read_text(encoding="utf-8")))


def prichina_propuska(arkhiv: Arhiv | None) -> str:
    """Почему боевой прогон нельзя запустить здесь и сейчас. Пусто — можно."""
    if arkhiv is None:
        return (f"нет файла {LOKALNYJ.name}: скопируйте {OBRAZETS} и укажите свои "
                "папки — личные фото и их имена в репозиторий не попадают")
    return arkhiv.net_na_meste
