"""Runtime-хук сборки: чиним загрузчик бинарников OpenCV внутри `.app`.

PyInstaller на macOS раскладывает содержимое архива в `Contents/Frameworks` и
`Contents/Resources`, и это одна и та же файловая система, связанная ссылками APFS.
Загрузчик `cv2/__init__.py` после этого не находит своё расширение: он сравнивает
`sys.path[0]` с `dirname(realpath(__file__))`, одно имя разрешается в `Frameworks`,
другое — в `Resources`, совпадения не выходит, и путь к расширению вставляется не
первым. `import cv2` вместо `.so` снова находит пакет `cv2/`, второй раз вызывает
свой `bootstrap()` и выдаёт:

    ImportError: ERROR: recursion is detected during loading of "cv2" binary extensions

Флаг `sys.OpenCV_REPLACE_SYS_PATH_0` — штатная отмычка самого OpenCV (там же,
строка 121): с ней путь к расширению вставляется нулевым, и импорт берёт `.so`.

Хук обязан отработать ДО первого импорта `cv2`, поэтому он runtime-хук, а не правка
в `src/`: в исходниках приложение бегает и без архива, и там этот флаг не нужен.
"""

import sys

if getattr(sys, "frozen", False):
    sys.OpenCV_REPLACE_SYS_PATH_0 = True
