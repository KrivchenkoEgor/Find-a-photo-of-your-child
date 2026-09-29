"""Рецепт сборки `.app` должен остаться целым.

PyInstaller здесь не запускается — это минута работы и полгигабайта на диске. Зато
проверяется то, что тихо ломается при правке: файл хука, на который ссылается
спецификация, и поведение самого хука. Молча удалённый «за ненадобностью» хук
обрушивает приложение не на этом Маке, а внутри архива — сообщением
`recursion is detected during loading of "cv2" binary extensions`.
"""

import subprocess
import sys
from pathlib import Path

KOREN = Path(__file__).resolve().parents[1]
SPEC = KOREN / "FindChild.spec"
HUK_CV2 = KOREN / "hooks" / "runtime_hook_cv2.py"


def test_spec_ssylaetsja_na_sushchestvushchij_huk() -> None:
    """Ссылка в спецификации обязана вести в файл, который лежит в репозитории."""
    tekst = SPEC.read_text(encoding="utf-8")

    assert "runtime_hook_cv2.py" in tekst, "спецификация больше не зовёт хук cv2"
    assert HUK_CV2.is_file(), f"нет файла хука: {HUK_CV2}"


def test_vesa_i_ikonka_ozhidajutsja_iz_arhiva_a_ne_iz_domashnej_papki() -> None:
    """Спецификация кладёт веса в `models/buffalo_l`, а приложение читает их оттуда.

    Две константы в двух местах — это будущий разъезд: сборщик положит файл по одному
    пути, `utils.config` будет искать по другому, и проверка «понесём на другой Мак»
    провалится уже в готовом архиве.
    """
    tekst = SPEC.read_text(encoding="utf-8")

    assert '"models/buffalo_l"' in tekst, "не указан каталог весов внутри архива"
    assert "icon.icns" in tekst, "иконка выпала из сборки"


def _flag_posle_huka(zamorozhen: bool) -> str:
    """Что стоит `sys.OpenCV_REPLACE_SYS_PATH_0` после выполнения хука."""
    itog = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.frozen = {zamorozhen!r}; "
         f"exec(open({str(HUK_CV2)!r}, encoding='utf-8').read()); "
         "print(getattr(sys, 'OpenCV_REPLACE_SYS_PATH_0', 'ne postavlen'))"],
        capture_output=True, text=True, check=True)
    return itog.stdout.strip()


def test_huk_stavit_flag_tolko_v_sobrannom_prilozhenii() -> None:
    """В архиве — ставит, в исходниках — не трогает.

    Флаг меняет порядок обхода `sys.path` у всего процесса. В режиме разработки, где
    `cv2` импортируется из обычного окружения и так, он не нужен и только затемняет
    причину, если OpenCV однажды перестанет грузиться.
    """
    assert _flag_posle_huka(True) == "True"
    assert _flag_posle_huka(False) == "ne postavlen"
