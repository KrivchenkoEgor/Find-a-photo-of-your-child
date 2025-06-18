import os
import shutil
import time
import warnings
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2

try:
    import insightface
    from insightface.app import FaceAnalysis
    print(f"✅ [{time.strftime('%H:%M:%S')}] InsightFace успешно импортирован")
except ImportError as e:
    print(f"⚠️ [{time.strftime('%H:%M:%S')}] Ошибка импорта insightface: {str(e)}. Режим 'reference' будет недоступен. Установите: pip install insightface onnxruntime")

# ===== НАСТРОЙКИ =====
DESTINATION_DIR = Path("/Users/egorkrivchenko/FacesPhotos")
SEARCH_PATHS = [Path("/Users/egorkrivchenko/Downloads/strekoza/Photos")]
REFERENCE_DIR = Path("/Users/egorkrivchenko/Pictures/referface")
SEARCH_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.heic', '.webp']
WORKERS_COUNT = 6
MATCH_TOLERANCE = 0.6  # Порог совпадения лиц
MAX_IMAGE_DIMENSION = 1200

# ===== ФУНКЦИИ =====
warnings.filterwarnings("ignore", category=UserWarning)

def parse_args():
    parser = argparse.ArgumentParser(description="Find and copy photos with faces")
    parser.add_argument('--mode', choices=['all', 'reference'], default='all',
                        help='Mode: "all" to find all faces, "reference" to match faces from REFERENCE_DIR')
    return parser.parse_args()

def check_gpu():
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print(f"✅ [{time.strftime('%H:%M:%S')}] Обнаружен GPU (CUDA)")
        else:
            print(f"⚠️ [{time.strftime('%H:%M:%S')}] GPU не обнаружен, используется CPU")
    except AttributeError:
        print(f"⚠️ [{time.strftime('%H:%M:%S')}] OpenCV без CUDA, используется CPU")
    return "CPUExecutionProvider"

def load_image(image_path):
    image_path = Path(image_path)
    if image_path.suffix.lower() in ('.heic', '.webp'):
        try:
            img = Image.open(image_path).convert('RGB')
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"⚠️ [{time.strftime('%H:%M:%S')}] Ошибка загрузки {image_path}: {str(e)}")
            return None
    return cv2.imread(str(image_path))

def resize_image(image, max_dimension=MAX_IMAGE_DIMENSION):
    if image is None:
        return None
    h, w = image.shape[:2]
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        return cv2.resize(image, (int(w * scale), int(h * scale)))
    return image

def load_reference_encodings(reference_dir, provider):
    try:
        app = FaceAnalysis(providers=[provider])
        app.prepare(ctx_id=0, det_size=(640, 640))
        print(f"✅ [{time.strftime('%H:%M:%S')}] InsightFace инициализирован")
    except Exception as e:
        print(f"⚠️ [{time.strftime('%H:%M:%S')}] Ошибка инициализации InsightFace: {str(e)}")
        return None, None
    encodings = []
    ref_files = [f for f in reference_dir.glob("*") if f.suffix.lower() in SEARCH_EXTENSIONS]
    for ref_path in ref_files:
        try:
            image = load_image(ref_path)
            image = resize_image(image)
            if image is None:
                continue
            faces = app.get(image)
            if faces:
                encodings.append(faces[0].normed_embedding)
            print(f"✅ [{time.strftime('%H:%M:%S')}] Загружено эталонное изображение: {ref_path}")
        except Exception as e:
            print(f"⚠️ [{time.strftime('%H:%M:%S')}] Ошибка обработки {ref_path}: {str(e)}")
    if not encodings:
        print(f"⚠️ [{time.strftime('%H:%M:%S')}] Не найдены лица в {reference_dir}")
        return None, None
    return encodings, app

def detect_faces(image_path, reference_encodings=None, face_app=None):
    try:
        image = load_image(image_path)
        image = resize_image(image)
        if image is None:
            return False, None, None

        if reference_encodings and face_app:
            faces = face_app.get(image)
            if not faces:
                return False, None, None
            for face in faces:
                similarity = np.dot(face.normed_embedding, np.array(reference_encodings).T).max()
                distance = 1 - similarity
                if distance <= MATCH_TOLERANCE:
                    return True, distance, faces
            return False, None, None
        else:
            detector = cv2.FaceDetectorYN.create(
                model="/Users/egorkrivchenko/PycharmProjects/find_faces/.venv/lib/python3.9/site-packages/cv2/data/face_detector_yunet_2023mar.onnx",
                config="",
                input_size=(int(image.shape[1]), int(image.shape[0])),
                score_threshold=0.7,
                backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                target_id=cv2.dnn.DNN_TARGET_CPU
            )
            if detector is None:
                print(f"⚠️ [{time.strftime('%H:%M:%S')}] Не удалось загрузить модель детектора лиц для {image_path}")
                return False, None, None
            faces = detector.detect(image)[1]
            print(f"DEBUG: {image_path}, faces: {faces}")
            return len(faces) > 0 if faces is not None else False, None, faces
    except Exception as e:
        print(f"⚠️ [{time.strftime('%H:%M:%S')}] Ошибка обработки {image_path}: {str(e)}")
        return False, None, None

def find_image_files(paths, extensions):
    all_files = []
    extensions = [ext.lower() for ext in extensions]
    for path in paths:
        if not path.exists():
            print(f"⚠️ [{time.strftime('%H:%M:%S')}] Путь не существует: {path}")
            continue
        print(f"🔍 [{time.strftime('%H:%M:%S')}] Сканирование: {path}")
        for file in path.rglob("*"):
            if file.suffix.lower() in extensions:
                all_files.append(file)
    return all_files

def copy_with_unique_name(src, dst_dir):
    src = Path(src)
    dst_dir = Path(dst_dir)
    rel_path = src.relative_to(Path(os.path.commonpath([str(p) for p in SEARCH_PATHS])))
    dest_path = dst_dir / rel_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        base, ext = rel_path.stem, rel_path.suffix
        counter = 1
        while True:
            new_path = dest_path.parent / f"{base}_{counter}{ext}"
            if not new_path.exists():
                dest_path = new_path
                break
            counter += 1
    try:
        shutil.copy2(src, dest_path)
        print(f"✅ [{time.strftime('%H:%M:%S')}] Файл скопирован: {dest_path}")
        return dest_path
    except Exception as e:
        print(f"❌ [{time.strftime('%H:%M:%S')}] Ошибка копирования {src}: {str(e)}")
        return None

def process_batch(batch, dest_dir, reference_encodings=None, face_app=None):
    results = []
    distances = []
    faces_count = 0
    files_copied = 0
    for file_path in tqdm(batch, desc="Обработка файлов", leave=False):
        matched, distance, faces = detect_faces(file_path, reference_encodings, face_app)
        if matched and faces is not None:
            faces_count += len(faces)
            try:
                dest_path = copy_with_unique_name(file_path, dest_dir)
                if dest_path:
                    results.append(dest_path)
                    files_copied += 1
                    if distance is not None:
                        distances.append(distance)
                    print(f"✅ [{time.strftime('%H:%M:%S')}] Обработан: {file_path}, найдено {len(faces)} лиц")
                else:
                    print(f"⚠️ [{time.strftime('%H:%M:%S')}] Не удалось скопировать: {file_path}, найдено {len(faces)} лиц")
            except Exception as e:
                print(f"⚠️ [{time.strftime('%H:%M:%S')}] Ошибка обработки {file_path}: {str(e)}, найдено {len(faces)} лиц")
    return results, distances, faces_count, files_copied

def main():
    args = parse_args()
    DESTINATION_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 [{time.strftime('%H:%M:%S')}] Файлы с лицами будут сохранены в: {DESTINATION_DIR}")

    provider = check_gpu()

    reference_encodings = None
    face_app = None
    if args.mode == 'reference':
        if not REFERENCE_DIR.exists():
            print(f"⚠️ [{time.strftime('%H:%M:%S')}] Папка {REFERENCE_DIR} не найдена, переключаюсь на режим 'all'")
            args.mode = 'all'
        else:
            print(f"\n🔍 [{time.strftime('%H:%M:%S')}] Загрузка эталонных изображений из: {REFERENCE_DIR}")
            reference_encodings, face_app = load_reference_encodings(REFERENCE_DIR, provider)

    print(f"\n🔎 [{time.strftime('%H:%M:%S')}] Поиск файлов изображений...")
    start_time = time.time()
    image_files = find_image_files(SEARCH_PATHS, SEARCH_EXTENSIONS)
    search_time = time.time() - start_time

    if not image_files:
        print(f"\n🛑 [{time.strftime('%H:%M:%S')}] Файлы изображений не найдены!")
        return

    print(f"\n✅ [{time.strftime('%H:%M:%S')}] Найдено файлов: {len(image_files)}")
    print(f"⏱ [{time.strftime('%H:%M:%S')}] Время поиска: {search_time:.2f} секунд")

    print("\n=================================")
    print(f"⚙️ [{time.strftime('%H:%M:%S')}] Настройки обработки:")
    print(f"- Режим: {args.mode}")
    print(f"- Провайдер: {provider}")
    print(f"- Расширения: {', '.join(SEARCH_EXTENSIONS)}")
    print(f"- Потоки: {WORKERS_COUNT}")
    print(f"- Пути поиска: {', '.join([str(p) for p in SEARCH_PATHS])}")
    if args.mode == 'reference':
        print(f"- Эталонные изображения: {REFERENCE_DIR}")
        print(f"- Порог совпадения: {MATCH_TOLERANCE}")
    print("=================================")

    print(f"\n🔍 [{time.strftime('%H:%M:%S')}] Поиск лиц на изображениях...")
    start_processing = time.time()
    total_faces_found = 0
    total_files_copied = 0
    all_distances = []
    batch_size = 300

    with ProcessPoolExecutor(max_workers=WORKERS_COUNT) as executor:
        futures = []
        for i in range(0, len(image_files), batch_size):
            batch = image_files[i:i + batch_size]
            futures.append(executor.submit(process_batch, batch, DESTINATION_DIR, reference_encodings, face_app))

        for i, future in enumerate(futures):
            result, distances, faces_in_batch, files_copied = future.result()
            total_faces_found += faces_in_batch
            total_files_copied += files_copied
            all_distances.extend(distances)
            processed = min((i + 1) * batch_size, len(image_files))
            print(f"\n📊 [{time.strftime('%H:%M:%S')}] Прогресс: {processed}/{len(image_files)} ({processed / len(image_files) * 100:.1f}%)")
            print(f"- Всего лиц найдено: {total_faces_found}")
            print(f"- Скопировано файлов: {total_files_copied}")
            print(f"- Среднее лиц на файл: {total_faces_found / processed:.2f}")
            if all_distances and args.mode == 'reference':
                avg_distance = sum(all_distances) / len(all_distances)
                print(f"- Средний процент совпадения: {(1 - avg_distance) * 100:.1f}%")
            elapsed = time.time() - start_processing
            time_per_file = elapsed / processed if processed > 0 else 0
            remaining = time_per_file * (len(image_files) - processed)
            print(f"⏱ [{time.strftime('%H:%M:%S')}] Оставшееся время: {remaining / 60:.1f} минут")

    total_time = time.time() - start_processing
    print(f"\n✅ [{time.strftime('%H:%M:%S')}] Обработка завершена!")
    print("=================================")
    print("📊 ИТОГОВЫЙ ОТЧЕТ")
    print(f"- Всего файлов: {len(image_files)}")
    print(f"- Всего лиц найдено: {total_faces_found}")
    print(f"- Скопировано файлов: {total_files_copied}")
    print(f"- Среднее лиц на файл: {total_faces_found / len(image_files):.2f}")
    if all_distances and args.mode == 'reference':
        avg_distance = sum(all_distances) / len(all_distances)
        print(f"- Средний процент совпадения: {(1 - avg_distance) * 100:.1f}%")
    print(f"- Общее время обработки: {total_time / 60:.1f} минут")
    print(f"- Скорость обработки: {len(image_files) / total_time:.1f} файлов/сек")
    print(f"- Файлы сохранены в: {DESTINATION_DIR}")
    print("=================================")

    report_path = DESTINATION_DIR / "face_detection_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Отчет обнаружения лиц\n")
        f.write(f"Дата: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Режим: {args.mode}\n")
        f.write(f"Провайдер: {provider}\n")
        f.write(f"Всего обработано файлов: {len(image_files)}\n")
        f.write(f"Всего лиц найдено: {total_faces_found}\n")
        f.write(f"Скопировано файлов: {total_files_copied}\n")
        f.write(f"Среднее лиц на файл: {total_faces_found / len(image_files):.2f}\n")
        if all_distances and args.mode == 'reference':
            avg_distance = sum(all_distances) / len(all_distances)
            f.write(f"Средний процент совпадения: {(1 - avg_distance) * 100:.1f}%\n")
            f.write(f"Порог совпадения: {MATCH_TOLERANCE}\n")
        f.write(f"Время обработки: {total_time / 60:.1f} минут\n")
        f.write(f"Пути поиска: {', '.join([str(p) for p in SEARCH_PATHS])}\n")
        if args.mode == 'reference':
            f.write(f"Папка с эталонными изображениями: {REFERENCE_DIR}\n")
        f.write(f"Расширения файлов: {', '.join(SEARCH_EXTENSIONS)}\n")
    print(f"\n📄 [{time.strftime('%H:%M:%S')}] Полный отчет сохранен в: {report_path}")

if __name__ == "__main__":
    main()