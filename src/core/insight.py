"""Точный путь: RetinaFace находит лица, arcface строит отпечатки. Один вызов."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .engine import MIN_FACE_PX, Face, prigoden_otpechatok

DET_SIZE = 1280        # при 640 мелкие лица на 24-Мп снимке теряются
DET_THRESH = 0.5


def convert_insight_faces(faces: Sequence[Any]) -> list[Face]:
    """Объекты insightface → Face. Ничего не пересчитываем, только разворачиваем поля.

    Отпечаток копируем: `np.asarray(..., dtype=np.float32)` при уже float32 отдаёт вид
    на массив, которым владеет insightface, и следующий кадр перезаписал бы его — лица
    уехали бы в архив чужими отпечатками. Тот же приём, что в `Face.from_dict`.

    Вырожденный отпечаток — не «лицо с нулевым сходством», а ненайденное лицо:
    такие строки отсекаем здесь, до оглавления. Правило одно на все пути и живёт
    в `engine.prigoden_otpechatok` — здесь его только вызывают.
    """
    out: list[Face] = []
    for f in faces:
        emb = np.asarray(f.normed_embedding, dtype=np.float32).copy()
        if not prigoden_otpechatok(emb):
            continue
        x1, y1, x2, y2 = (float(v) for v in f.bbox)
        out.append(Face(
            box=(x1, y1, x2, y2),
            landmarks=tuple(float(v) for v in np.asarray(f.kps).reshape(-1)),
            embedding=emb,
            detector="insight",
        ))
    return out


class InsightEngine:
    name = "insight"

    def __init__(self, det_size: int = DET_SIZE, det_thresh: float = DET_THRESH,
                 min_face: int = MIN_FACE_PX,
                 buffalo_l: Path | None = None) -> None:
        self.det_size = det_size
        self.det_thresh = det_thresh
        self.min_face = min_face
        # Каталог весов, приехавший извне — как путь YuNet у `YunetEngine`. None значит
        # «имени пакета», и тогда insightface берёт кэш `~/.insightface`, а при его
        # отсутствии скачивает 290 МБ. Путь внутрь собранного приложения передаёт
        # `make_engine` — см. `utils.config.buffalo_l_dir`.
        self.buffalo_l = Path(buffalo_l) if buffalo_l is not None else None
        self._app = None
        # Сколько лиц отброшено как непригодные (сбойный кроп дал нули, NaN или вектор
        # чужой ширины). Число для отчёта: без него «лицо было и исчезло» неотличимо
        # от «лица не было», и пользователь теряет снимок молча.
        self.otsortirovannye_lica = 0

    @property
    def loaded(self) -> bool:
        """Поднят ли `FaceAnalysis`: он создаётся только внутри `load()`.

        Рабочий поток спрашивает именно это, прежде чем грузить модели: `load()` стоит
        13–21 с и тянет ~300 МБ весов, а второй заход ещё и пересобирает распознаватель.
        """
        return self._app is not None

    def load(self) -> None:
        from insightface.app import FaceAnalysis   # локальный импорт: тестам не нужны модели

        # Каталог вместо имени: insightface принимает путь в `name` и в этой ветке не
        # трогает сеть. Имя же («buffalo_l») означает «ищи в ~/.insightface, а нет —
        # качай 290 МБ», и на чужом Маке без весов в архиве это единственный выход.
        imja = (str(self.buffalo_l) if self.buffalo_l is not None and
                self.buffalo_l.is_dir() else "buffalo_l")
        self._app = FaceAnalysis(name=imja,
                                 allowed_modules=["detection", "recognition"])
        self._app.prepare(ctx_id=-1, det_size=(self.det_size, self.det_size),
                          det_thresh=self.det_thresh)

    @property
    def recognition_model(self) -> Any | None:
        """Распознаватель, уже загруженный внутри FaceAnalysis (arcface w600k_r50).

        Отдаём его режиму «Оба», чтобы YuNetEngine не читал те же 174 МБ весов вторично.
        Повторного `prepare()` здесь нет намеренно: у модели распознавания `prepare`
        принимает только `ctx_id`, а `-1` ей уже выставлен вызовом выше; `det_size` и
        `det_thresh` касаются лишь детектора. То есть расхождения параметров между
        двумя путями не возникает, и перенастраивать модель нельзя — второй `prepare`
        сбросил бы её состояние у того, кто уже её использует.
        """
        if self._app is None:
            return None
        return getattr(self._app, "models", {}).get("recognition")

    def detect(self, image: np.ndarray) -> list[Face]:
        if self._app is None:
            raise RuntimeError("модели не загружены: сначала load()")
        naidennye = self._app.get(image)
        lica = convert_insight_faces(naidennye)
        # Разница «нашла модель» минус «отдала» — это ровно отбраковка отпечатков:
        # фильтр по размеру лица идёт следующей строкой и в счётчик не попадает,
        # мелкая рамка — решение детектора, а не испорченные данные.
        self.otsortirovannye_lica += max(0, len(naidennye) - len(lica))
        return [f for f in lica if f.size >= self.min_face]
