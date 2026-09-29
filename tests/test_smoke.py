def test_pakety_dostupny() -> None:
    """Пакеты `src/` видны импорту И это наши пакеты, а не однофамильцы.

    `__name__ == "core"` доказывает только то, что имя не занято чем-то другим: на
    пустом каталоге `core/` без `__init__.py` python собирает namespace-пакет, у
    которого нет ни одного модуля, и весь остальной набор тестов утонул бы в
    `ModuleNotFoundError` вместо одного понятного провала. Проверка `__file__` ловит
    это, а проверка корня — случай, когда в окружении лежит установленный пакет с тем
    же именем и тесты начинают проверять не тот код.
    """
    from pathlib import Path

    koren = Path(__file__).resolve().parents[1] / "src"

    import core
    import ui
    import utils

    for modul in (core, ui, utils):
        fayl = getattr(modul, "__file__", None)
        assert fayl is not None, f"пакет {modul.__name__} — namespace, модулей в нём нет"
        assert Path(fayl).resolve().is_relative_to(koren), \
            f"{modul.__name__} импортирован из {fayl}, а не из {koren}"


def test_kvadraty_eto_ne_pustoshka(qapp) -> None:
    from PySide6.QtWidgets import QLabel

    assert QLabel("ок").text() == "ок"
