"""Лицо как данные и способы получить лица с картинки.

Face — валюта между всеми модулями: её кладут в кэш, её сравнивает matcher,
её рисует интерфейс. Меняя поля здесь, меняйте и to_dict/from_dict.

Сравнение и хеширование. `embedding` — массив, и генератор dataclass с ним не справляется:
numpy в ответ на `==` даёт массив булев, а не одно True, поэтому `a == b` на двух разных
объектах бросает ValueError, а `hash(a)` — TypeError (зафиксировано тестом). Лица сверяют
по полям (`box`, `detector`, `landmarks`) или по идентичности объекта; в множества и как
ключи словаря их не кладут.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

EMBEDDING_LENGTH = 512      # длина отпечатка arcface w600k_r50: она же ширина строки матрицы

# Меньшая сторона рамки лица в пикселях, меньше которой лицо считается шумом. Единственный
# владелец — этот модуль: правило одинаковое для обоих путей, а расхождение на один шаг
# дало бы два разных ответа на вопрос «кто на фото» в зависимости от выбранного способа
# поиска. Значение снято на 24-мегапиксельных снимках архива проекта.
MIN_FACE_PX = 20


def prigoden_otpechatok(embedding: Any, length: int = EMBEDDING_LENGTH) -> bool:
    """Годен ли отпечаток к сравнению. ЕДИНСТВЕННЫЙ такой вопрос в проекте.

    Три условия, и порядок проверок важен: сначала размерность и длина, потом
    `isfinite`, и только потом норма. Проверенный по всем числам вектор не успевает
    ни попасть в BLAS с NaN внутри, ни дать деление на нулевую норму, ни вызвать
    предупреждение numpy.

    Почему правило живёт здесь, а не в каждом пути по-своему. До этой правки оно было
    написано дважды — в `insight.py` и внутри `yunet.detect` — и оба раза только на
    записи: при чтении оглавления (`Face.from_dict`) и при сравнении (`matcher`) его
    не было вовсе. Вектор с NaN или чужой ширины, записанный старой версией приложения,
    доезжал до `round(nan)` / `matmul` и ронял разбор фото без всякой надежды на
    «это лицо не считается». Теперь вопрос задаётся один раз и в четырёх местах:
    `Face.from_dict`, `embeddings_of`, фильтр Insight-пути, фильтр YuNet-пути.

    Предикат ничего не бросает, даже если подать `None`, строку или список списков:
    его вызывают на данных из чужого файла оглавления, и ответ «нет» должен быть
    ответом, а не исключением.
    """
    try:
        emb = np.asarray(embedding, dtype=np.float32)
    except (TypeError, ValueError):
        return False
    if emb.ndim != 1 or emb.shape[0] != length:
        return False
    if not bool(np.isfinite(emb).all()):
        return False
    return bool(float(np.linalg.norm(emb)) > 0.0)


@dataclass(frozen=True)
class Face:
    box: tuple[float, float, float, float]      # x1, y1, x2, y2 в пикселях картинки
    landmarks: tuple[float, ...] | None          # 10 чисел: глаз, глаз, нос, рот, рот
    embedding: np.ndarray                        # 512 float32, длина 1.0
    detector: str                                # 'yunet' | 'insight'

    @property
    def x1(self) -> float:
        return self.box[0]

    @property
    def y1(self) -> float:
        return self.box[1]

    @property
    def x2(self) -> float:
        return self.box[2]

    @property
    def y2(self) -> float:
        return self.box[3]

    @property
    def size(self) -> float:
        """Меньшая сторона рамки — по ней судят, насколько лицо крупное."""
        return min(self.box[2] - self.box[0], self.box[3] - self.box[1])

    def to_dict(self) -> dict:
        """Словарь для оглавления. Отпечаток — base64 от float32, а не 512 чисел.

        Список чисел в JSON весит 8–11 КБ на лицо: на архиве в 20 000 фото по два
        лица это 330–440 МБ вместо ~113 МБ (замер на этом формате). base64 даёт
        ровно те же байты, что и массив.
        """
        return {
            "box": [float(v) for v in self.box],
            "landmarks": [float(v) for v in self.landmarks] if self.landmarks else None,
            "embedding": base64.b64encode(
                np.ascontiguousarray(self.embedding, dtype=np.float32).tobytes()
            ).decode("ascii"),
            "detector": self.detector,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Face":
        """Лицо из строки оглавления. Непригодный отпечаток или битая рамка — ValueError.

        Оглавление переживает не одну версию приложения, и старые файлы писали вектор
        без всякой проверки: в строке может лежать NaN, Inf, нули или 256 чисел вместо
        512. Такое лицо нельзя вернуть «как есть»: оно проехало бы дальше до первого
        сравнения и обрывало разбор фото.

        Рамка проверяется по тому же правилу конечности, что и отпечаток. JSON честно
        перевозит `Infinity`, а дальше Inf-координата доезжает до потока интерфейса:
        `int(inf * k)` в отрисовке зелёной рамки бросает `OverflowError`, и одна битая
        строка обрывала показ всего архива уже ПОСЛЕ того, как разбор успешно кончился.

        ValueError — не «упасть», а штатный сигнал для `FaceCache.get`: он его ловит,
        удаляет строку и отдаёт None, то есть «этого снимка в оглавлении нет, разберём
        заново». Так битая строка деградирует до пересчёта одного фото: наружу не
        уходит ни исключение из рабочего потока, ни молча обнулённый архив.
        """
        lm = data.get("landmarks")
        raw = base64.b64decode(data["embedding"])
        emb = np.frombuffer(raw, dtype=np.float32).copy()
        if not prigoden_otpechatok(emb):
            raise ValueError(
                "отпечаток в оглавлении непригоден: "
                f"{emb.shape[0]} чисел вместо {EMBEDDING_LENGTH}, либо NaN/Inf, "
                f"либо одни нули (детектор: {data.get('detector')!r})"
            )
        box = tuple(float(v) for v in data["box"])
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            raise ValueError(
                "рамка лица в оглавлении непригодна: нужно четыре конечных числа, "
                f"а пришло {len(box)}: {box!r} (детектор: {data.get('detector')!r})"
            )
        return cls(
            box=box,
            landmarks=tuple(float(v) for v in lm) if lm else None,
            embedding=emb,
            detector=data["detector"],
        )


def prigodnye_lica(faces: Sequence[Face]) -> list[Face]:
    """Те лица, чьи отпечатки годны к сравнению; остальные считаются ненайденными.

    Отдельная функция нужна не для красоты: `embeddings_of` отбрасывает непригодные
    строки, и если вызывающий код потом нумерует по ним исходный список, индексы
    разъезжаются. Кто сопоставляет строку матрицы с лицом — обязан сначала прогнать
    список здесь и работать уже с этим результатом (так делает `score_photo`).
    """
    return [f for f in faces if prigoden_otpechatok(f.embedding)]


def embeddings_of(faces: Sequence[Face]) -> np.ndarray:
    """Матрица (n, 512) из списка лиц; для пустого списка — (0, 512).

    Непригодные отпечатки в матрицу не попадают: одна строка с NaN или чужой шириной
    означала бы падение всего сравнения вместо одного потерянного лица. Поэтому
    число строк не обязано совпадать с длиной `faces` — см. `prigodnye_lica`.
    """
    useful = prigodnye_lica(faces)
    if not useful:
        return np.zeros((0, EMBEDDING_LENGTH), dtype=np.float32)
    return np.stack([np.asarray(f.embedding, dtype=np.float32) for f in useful])


class FaceEngine(Protocol):
    """Один способ получить лица с картинки. Реализации: yunet, insight, both."""

    name: str

    loaded: bool
    """Модели загружены, `load()` больше не нужен.

    Признак спрашивают снаружи перед каждым прогоном: `load()` стоит 13–21 с, а на
    живом `BothEngine` второй вызов ещё и падает. Без него отличить «движок готов» от
    «движок не готов» нечем, и каждый прогон грузил бы веса заново. Реализации дают
    это свойством, а не полем-переменной: отвечать надо за то, что реально стоит
    внутри, а не за заведённый флажок.
    """

    def load(self) -> None:
        """Загрузить модели. Долго, только вне потока интерфейса."""
        ...

    def detect(self, image: np.ndarray) -> list[Face]:
        ...


def make_engine(mode: str, yunet_model: Path,
                buffalo_l: Path | None = None) -> FaceEngine:
    """Собирает движок по строке настройки: 'yunet' | 'insight' | 'both'.

    Импоры внутри функции, а не на уровне модуля: `engine.py` читают все, включая
    тесты, которым не нужны ни OpenCV, ни путь к моделям. Здесь же они нужны ровно
    тем, кто просит конкретный режим.

    Оба пути к весам называются здесь, в одной строке вызова: YuNet и распознавание
    живут в разных местах, и решить, откуда их брать, должен тот, кто собирает движок,
    а не сам движок (см. `utils.config`).

    Фабрика только собирает объекты: тяжёлое начинается по `load()`. Поэтому
    интерфейс может строить движок заранее и не платить за это секундами.
    """
    from .both import BothEngine
    from .insight import InsightEngine
    from .yunet import YunetEngine

    if mode == "insight":
        return InsightEngine(buffalo_l=buffalo_l)
    if mode == "yunet":
        return YunetEngine(Path(yunet_model))
    if mode == "both":
        return BothEngine(YunetEngine(Path(yunet_model)),
                          InsightEngine(buffalo_l=buffalo_l))
    raise ValueError(f"неизвестный режим распознавания: {mode!r}")
