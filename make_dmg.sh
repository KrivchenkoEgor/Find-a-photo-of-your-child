#!/bin/bash
# Сборка `.app` и `.dmg` для macOS.
#
#   ./make_dmg.sh        # → dist/FindChild.app и dist/FindChild.dmg
#
# Версия приложения живёт в `FindChild.spec` (`version=`), поэтому у носителя номера
# нет: два места с одним числом однажды расходятся, а здесь расхождение заметили бы
# только на чужом Маке.
#
# Что где лежит: `build/` — черновая работа PyInstaller и временная папка носителя
# (оба каталога в gitignore), `dist/` — готовое. Веса распознавания берутся из
# `~/.insightface`: спецификация остановит сборку, если их там нет.
set -euo pipefail
cd "$(dirname "$0")"

RAZMERTKA="build/nositel"

echo "→ собираю .app"
.venv/bin/python -m PyInstaller FindChild.spec --noconfirm

echo "→ укладываю для носителя"
rm -rf "$RAZMERTKA"
mkdir -p "$RAZMERTKA"
cp -R dist/FindChild.app "$RAZMERTKA/"
# Ссылка на «Программы» — чтобы с смонтированного тома приложение перетаскивали
# мышью, как из любого другого установщика.
ln -s /Applications "$RAZMERTKA/Applications"

echo "→ пишу .dmg"
rm -f dist/FindChild.dmg
# `hdiutil create` на macOS 27 объявлен устаревшим и предупреждает об этом каждым
# запуском; `diskutil image create from` — та же операция без предупреждения.
diskutil image create from --format UDZO --volumeName "Поиск фото" \
    "$RAZMERTKA" dist/FindChild.dmg

rm -rf "$RAZMERTKA"
# PyInstaller оставляет рядом с `.app` ещё и папку `dist/FindChild/` — те же 554 МБ
# вторым экземпляром. Архив автономен (проверено запуском прямо с смонтированного
# тома), так что этот дубль — просто место на диске.
rm -rf dist/FindChild
echo "готово: $(du -sh dist/FindChild.app dist/FindChild.dmg | awk '{print $2" ("$1")"}' | paste -sd' ' -)"
