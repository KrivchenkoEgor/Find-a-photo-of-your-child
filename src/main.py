"""Запуск приложения: `.venv/bin/python src/main.py`.

Импорт сам по себе не создаёт ни окна, ни цикла событий: `QApplication` и `MainWindow`
появляются только внутри `main()`. Поэтому модуль можно импортировать в тесте, не рискуя
зависнуть в `app.exec()`.

`sys.path` поправляется здесь, а не в `conftest.py`: этот файл — точка входа и у
сборки PyInstaller, и у запуска из исходников, и в обоих случаях `src/` должен быть
виден, чтобы `from ui.main_window import ...` нашёлся без установки пакета.

Окно держится МОДУЛЬНОЙ ссылкой (`_OKNO`), а не локальной переменной. Без parent Qt-объект
принадлежит Python, и возврат из `main()` убил бы окно вместе с рабочим потоком, который
в нём ещё пишет оглавление: «QThread: Destroyed while thread is still running» в консоли и
недописанная база в папке пользователя. `closeEvent` окна сам не отпускает поток по той
же причине — здесь второй конец той же страховки.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtWidgets import QApplication                  # noqa: E402

from ui.main_window import MainWindow                       # noqa: E402

# Единственный держатель окна на время работы цикла событий: см. docstring модуля.
_OKNO: MainWindow | None = None


def main() -> int:
    """Собрать приложение и вернуть код выхода цикла событий."""
    global _OKNO
    app = QApplication(sys.argv)
    app.setApplicationName("FindChild")
    _OKNO = MainWindow()
    _OKNO.show()
    itog = app.exec()
    _OKNO = None                        # окно закрывает оглавление в своём `closeEvent`
    return itog


if __name__ == "__main__":
    raise SystemExit(main())
