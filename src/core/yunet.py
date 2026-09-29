"""Детектор OpenCV YuNet: быстрый, лёгкий, находит больше лиц, чем RetinaFace.

Отпечаток лица для него строит модель InsightFace: YuNet отдаёт только рамки и точки.
"""

from __future__ import annotations

import dataclasses
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .engine import MIN_FACE_PX, Face, prigoden_otpechatok

MIN_SCORE = 0.6      # порог уверенности самого детектора
RECOGNITION_MODEL = "w600k_r50.onnx"    # единственная модель распознавания в buffalo_l

# Файл детектора весит 230 КБ и лежит в репозитории (`assets/models/`), потому что
# без него первое же «Начать разбор папки» с клона непобедимо: человек обязан мочь
# достать модель одной ссылкой, а не искать её по названию файла.
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")


@lru_cache(maxsize=1)
def _load_recognition() -> Any:
    """Модель arcface из пакета buffalo_l — и только она, один раз на все движки.

    Замысел простой: детектором здесь стоит YuNet, а отпечатки строит InsightFace.
    Способ `FaceAnalysis(allowed_modules=["recognition"])` этого не умеет: в
    insightface 2.0 конструктор заканчивается `assert "detection" in self.models`
    и падает с AssertionError (проверено на нашей версии). Читаем файл распознавания
    напрямую через model_zoo: 3.0 с вместо 11.0 с и без RetinaFace, которую в этом
    пути никто никогда не вызывает.

    Кэш висит на самой функции, и это не украшение: BothEngine из задачи 7 создаёт два
    движка, оба просят распознаватель, и 174 МБ должны читаться один раз. Кэш внутри тела
    функции (как в листинге плана) не кэшировал бы ничего — объект `functools.lru_cache`
    создавался бы заново на каждый вызов.

    В режиме «Оба» сюда вообще не заходят: распознаватель уже загружен внутри
    FaceAnalysis, и BothEngine передаёт его в `adopt_recognition()`. Кэш остаётся
    страховкой на случай, когда движков YuNet в процессе несколько (например,
    несколько прогонов подряд в одном открытом окне).
    """
    from insightface.model_zoo import get_model           # локальный импорт: тестам не нужны модели
    from insightface.utils import ensure_available

    path = Path(ensure_available("models", "buffalo_l")) / RECOGNITION_MODEL
    if not path.exists():
        raise FileNotFoundError(
            f"нет модели распознавания: {path}. Распакуйте buffalo_l в ~/.insightface/models/"
        )
    model = get_model(str(path))
    if getattr(model, "taskname", None) != "recognition":
        raise RuntimeError(
            f"ожидали модель распознавания, а получили {getattr(model, 'taskname', None)!r}: {path}"
        )
    model.prepare(ctx_id=-1)
    return model


def _norm_crop(img: np.ndarray, landmark: np.ndarray, image_size: int = 112,
               mode: str = "arcface") -> np.ndarray:
    """Тонкая обёртка: тесты подменяют её, не загружая insightface.

    Без неё проверка выравнивания строк и отпечатков была бы тестом только на словах:
    `detect` импортировал бы `norm_crop` внутри себя, и подставить заглушку было бы нечего.
    """
    from insightface.utils.face_align import norm_crop   # локальный импорт: тестам не нужны модели
    return norm_crop(img, landmark=landmark, image_size=image_size, mode=mode)


def face_from_row(row: np.ndarray) -> Face:
    """Одна строка detect() → Face без отпечатка. Отпечаток дописывает YunetEngine."""
    x, y, w, h = (float(v) for v in row[:4])
    return Face(
        box=(x, y, x + w, y + h),
        landmarks=tuple(float(v) for v in row[4:14]),
        embedding=np.zeros(512, dtype=np.float32),
        detector="yunet",
    )


def keep_row(row: np.ndarray, min_size: int, min_score: float) -> bool:
    """Тот же фильтр для обоих путей: уверенности выше порога и лицо не меньше min_size."""
    w, h = float(row[2]), float(row[3])
    return float(row[14]) >= min_score and min(w, h) >= min_size


class YunetEngine:
    name = "yunet"

    def __init__(self, model_path: Path, score_threshold: float = MIN_SCORE,
                 min_face: int = MIN_FACE_PX, recognition: Any | None = None) -> None:
        self.model_path = Path(model_path)
        self.score_threshold = score_threshold
        self.min_face = min_face
        self._detector = None
        self._recognition = recognition
        # Сколько лиц отброшено как непригодные — число для отчёта (задача 9→10),
        # см. то же поле у `InsightEngine`.
        self.otsortirovannye_lica = 0

    @property
    def loaded(self) -> bool:
        """Готов ли движок — по факту: обе части, которые нужны `detect()`, на месте.

        Отдельный флажок не заводим: его пришлось бы синхронизировать с тем, что
        реально стоит внутри, а расхождение вида «говорит, готов, а `detect()` падает»
        было бы лишь вопросом времени. Обе модели создаёт только `load()` — они и есть
        признак готовности.

        Распознаватель здесь не деталь, а ровно та половина, которая чаще всего и не
        поднимается: `w600k_r50` тянет ~290 МБ из сети, и обрыв на этой закачке — типичный
        ПЕРВЫЙ запуск человека в режиме «Быстрее». Прежний ответ «готов» по одному
        детектору значил, что `obespechit_zagruzku` поверил ему навсегда и больше не
        пробовал, а все снимки архива умирали в `detect()` на
        `'NoneType' object has no attribute 'get_feat'` — советом пользователю
        «перезапустите приложение», который ничего не меняет.
        """
        return self._detector is not None and self._recognition is not None

    def adopt_recognition(self, model: Any) -> None:
        """Взять уже загруженный распознаватель вместо чтения своей копии весов.

        Нужно режиму «Оба»: InsightEngine грузит ту же w600k_r50 внутри FaceAnalysis,
        и второй экземпляр тех же 174 МБ в памяти не нужен.

        Только до `load()`: после загрузки отпечатки уже считаются этой моделью, и
        молчаливая замена дала бы в архиве лица с отпечатками от двух разных сетей.
        Ровно этот же объект можно отдать и позже — ничего не меняется, а второй
        `load()` в рабочем потоке (задача 9 зовёт его на каждый прогон) иначе падал бы
        с ошибкой там, где никакого расхождения нет.
        """
        if model is self._recognition:
            return
        if self._detector is not None:
            raise RuntimeError(
                "юнет уже загружен: распознаватель подставляют до load()")
        self._recognition = model

    def load(self) -> None:
        """Поднять детектор и распознаватель; оба атрибута ставятся в самом конце.

        Порядок здесь важнее порядка строк: распознаватель — это закачка ~290 МБ, и она
        имеет право оборваться. Если записать детектор до неё, движок останется
        наполовину собранным, `loaded` (см. свойство) про это не знает, и
        `obespechit_zagruzku` больше никогда не попробует. Присваивание из локальных
        переменных после обеих удач означает, что при любом промахе ничего не
        записано и следующий запуск начинает загрузку заново.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"нет файла детектора лиц: {self.model_path}\n"
                f"Скачайте его одним файлом по ссылке и положите на это место:\n"
                f"{YUNET_URL}"
            )
        detektor = cv2.FaceDetectorYN_create(
            str(self.model_path), "", (320, 320), self.score_threshold
        )
        raspoznavanie = self._recognition
        if raspoznavanie is None:
            # Чужая модель уже подошла — перечитывать её с диска незачем.
            raspoznavanie = _load_recognition()
        self._detector = detektor
        self._recognition = raspoznavanie

    def detect(self, image: np.ndarray) -> list[Face]:
        if self._detector is None:
            raise RuntimeError("модели не загружены: сначала load()")
        h, w = image.shape[:2]
        self._detector.setInputSize((w, h))       # в OpenCV 5 метод называется так
        _, rows = self._detector.detect(image)
        out: list[Face] = []
        # Цикл идёт по исходным строкам, а фильтр применяется здесь же. Если сначала
        # отфильтровать список, а потом сопоставить его строкам по номеру, индексы
        # разъедутся на первом же отброшенном лице — и каждый отпечаток встанет не к
        # своей рамке. Это не только слова: держит тест test_otpechatok_ne_uezhaet_na_drugoe_lico.
        for row in rows if rows is not None else []:
            if not keep_row(row, self.min_face, self.score_threshold):
                continue
            face = face_from_row(row)
            crop = _norm_crop(image, np.array(face.landmarks).reshape(5, 2))
            emb = np.asarray(self._recognition.get_feat(crop), dtype=np.float32).flatten()
            if not prigoden_otpechatok(emb):
                # Вырожденный кроп (лицо на границе кадра) либо сбой модели дают нулевой,
                # NaN-вектор или вектор чужой ширины. Делить на такую норму нельзя:
                # получился бы NaN, который честно ездит через base64 в оглавлении и
                # отравляет каждое сравнение дальше. Правило — одно на все пути,
                # `engine.prigoden_otpechatok`; лицо без пригодного отпечатка считаем
                # ненайденным — но не потерянным молча: счётчик помнит его.
                self.otsortirovannye_lica += 1
                continue
            out.append(dataclasses.replace(
                face, embedding=(emb / float(np.linalg.norm(emb))).astype(np.float32)))
        return out
