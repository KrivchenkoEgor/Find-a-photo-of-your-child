"""Общие фикстуры. Платформу Qt выбираем до импорта PySide6."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from collections.abc import Iterator

import pytest


@pytest.fixture(scope="session")
def qapp() -> Iterator["QApplication"]:
    """Один QApplication на весь прогон: без него виджеты не создаются."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
