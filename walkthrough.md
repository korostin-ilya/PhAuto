# Отчет о реализации пайплайна сборки сферических панорам БПЛА и замены неба

## 1. Реализованная архитектура и файлы проекта

В корневой директории [PhAuto](file:///C:/Users/Georg/Desktop/PhAuto) реализован полный производственный пайплайн согласно техническому заданию [README.md](file:///C:/Users/Georg/Desktop/PhAuto/README.md):

* **Конфигурация и контейнеризация:**
  * [Dockerfile](file:///C:/Users/Georg/Desktop/PhAuto/Dockerfile) — образ на базе `ubuntu:24.04` с пакетами `hugin-tools`, `libvips-dev`, `libimage-exiftool-perl`, `python3-dev` и компиляцией актуального `Multiblend 2.0` (`https://horman.net/multiblend/multiblend2.0rc5.tar.gz`) с флагами `-msse4.1 -pthread -ffast-math -Ofast`.
  * [docker-compose.yml](file:///C:/Users/Georg/Desktop/PhAuto/docker-compose.yml) и [.dockerignore](file:///C:/Users/Georg/Desktop/PhAuto/.dockerignore) — быстрый запуск контейнера с пробросом папок `pano/`, `sky/`, `output/`, `sky_catalog.json` и `config.yaml`.
  * [config.yaml](file:///C:/Users/Georg/Desktop/PhAuto/config.yaml) и [requirements.txt](file:///C:/Users/Georg/Desktop/PhAuto/requirements.txt) — централизованные настройки разрешения, фильтрации точек, весов метрик и Guided Filter.
* **Метаданные и астрономия:**
  * [metadata.py](file:///C:/Users/Georg/Desktop/PhAuto/src/metadata.py) — извлечение EXIF (`GPSInfo`, `ExifIFD`) и XMP (`drone-dji:*`), динамический расчет $\text{HFOV}$, определение часового пояса через `TimezoneFinder`, перевод местного времени съемки в `UTC` через `zoneinfo`, вычисление координат солнца $(X_{\text{sun\_pano}}, Y_{\text{sun\_pano}})$ через `astral.sun`, оценка $\Delta\text{EV}_{100}$ и запись выходных метаданных `GPSInfo` + `GPano` XMP.
* **Модуль 1 — Детерминированная сборка панорамы (`src/stitcher/`):**
  * [pto_generator.py](file:///C:/Users/Georg/Desktop/PhAuto/src/stitcher/pto_generator.py) — выбор опорного кадра горизонта (`index 0`) для фиксации Севера в $X = 7500$ (`!y0,!p0,!r0`, `!Eev0`), генерация `initial.pto`, геометрическое отсечение контрольных точек выше линии горизонта (`filter_sky_control_points`) и предохранитель углов (`enforce_angle_sanity_check` с откатом к углам RTK + медианной поправке соседей при отклонении $> 3.5^\circ$).
  * [hugin_runner.py](file:///C:/Users/Georg/Desktop/PhAuto/src/stitcher/hugin_runner.py) — оркестрация `cpfind --prealigned --fullscale` $\rightarrow$ очистка неба $\rightarrow$ `cpclean -n 1.5` $\rightarrow$ `pto_var` + `autooptimiser` $\rightarrow$ Sanity Check $\rightarrow$ `vig_optimize` $\rightarrow$ `nona` $\rightarrow$ `multiblend --wrap --wideblend` и реконструкция полного холста $15000 \times 7500 \times 4$ (`RGBA`) из усеченного по верхним пустым полосам `stitched_base.tif`.
* **Модуль 2 — Индексация, автоподбор и умный блендинг неба (`src/sky_replacer/`):**
  * [signature.py](file:///C:/Users/Georg/Desktop/PhAuto/src/sky_replacer/signature.py) — автоматическое определение высоты солнца $Y_{\text{sun\_sky}}$ и флага пасмурности `is_overcast_template`, формирование паспорта из $N = 20$ вертикальных срезов в полосе $Y \in [2850, 3650]$ ($K=6$ точек градиента LAB, 16-биновые гистограммы $L^*, b^*$, частотная энергия DoG $\sigma_1=15, \sigma_2=60$) и очистка срезов панорамы от высотных объектов.
  * [matcher.py](file:///C:/Users/Georg/Desktop/PhAuto/src/sky_replacer/matcher.py) — двухступенчатый выбор заготовки неба (Топ-5 по 20 срезам с метрикой Вассерштейна EMD $\rightarrow$ арбитраж микро-блендингом на копиях $600 \times 150$).
  * [blender.py](file:///C:/Users/Georg/Desktop/PhAuto/src/sky_replacer/blender.py) — циклический сдвиг шаблона по азимуту солнца `np.roll`, построение 2D-модели цвета родного неба и маски защиты высотных объектов $M_{\text{sky}}(x, y)$ с уточнением тонких граней через **Guided Filter** (`cv2.ximgproc.guidedFilter`), индивидуальный по столбцам `Smoothstep` градиент и затухающая к зениту цветокоррекция Рейнхарда в CIE LAB.
* **CLI-скрипты и тесты:**
  * [index_skies.py](file:///C:/Users/Georg/Desktop/PhAuto/scripts/index_skies.py) и [run_pipeline.py](file:///C:/Users/Georg/Desktop/PhAuto/scripts/run_pipeline.py)
  * [test_modules.py](file:///C:/Users/Georg/Desktop/PhAuto/tests/test_modules.py)

---

## 2. Результаты валидации и тестирования

### 2.1. Индексация базы заготовок неба (`sky_catalog.json`)
Скрипт [index_skies.py](file:///C:/Users/Georg/Desktop/PhAuto/scripts/index_skies.py) проиндексировал все 20 шаблонов ($15000 \times 3750$) в папке `sky/` и сформировал [sky_catalog.json](file:///C:/Users/Georg/Desktop/PhAuto/sky_catalog.json):
* **Закат (`01..05.jpg`):** определены высоты солнца у самого горизонта ($Y_{\text{sun}} \in [3273.6, 3671.8]\text{ px}$).
* **Облачность (`01..05.jpg`):** все 5 пасмурных шаблонов автоматически распознаны как `is_overcast_template = True, sun_y = None`.
* **Переменная облачность (`01..05.jpg`) и Солнечно (`01..05.jpg`):** определены высоты солнца во всем диапазоне от высокого ($Y_{\text{sun}} = 952.8\text{ px}$) до низкого ($Y_{\text{sun}} = 3665.1\text{ px}$).

### 2.2. Полная сборка панорам в Docker-контейнере (End-to-End)

Выполнена сборка Docker-образа `drone-pano-pipeline:latest` и проведены полные тестовые прогоны на реальных сериях снимков БПЛА:

| Параметр | Серия 1: `cedar-grove---tx` (DJI M3E, 25 кадров) | Серия 2: `taylor-morrison---phoenix` (DJI M4E, 27 кадров) |
| :--- | :--- | :--- |
| **Координаты GPS и таймзона** | `(29.90378, -95.16643)` (`America/Chicago`) | `(33.78425, -112.32162)` (`America/Phoenix`) |
| **Положение солнца** | $\phi = 231.16^\circ, \theta = 34.58^\circ \to X=2131.7, Y=2309.1$ | $\phi = 153.90^\circ, \theta = 43.17^\circ \to X=13912.6, Y=1951.2$ |
| **Контрольные точки (`cpfind`)** | Всего: `1540`, удалено на небе: `231` | Всего: `2343`, удалено на небе: `190` |
| **Sanity Check ($>3.5^\circ$)** | Возвращен к RTK: `1` кадр | Возвращено к RTK: `2` кадра |
| **Выбранный шаблон неба** | `Переменная облачность/04.jpg` ($Y_{\text{sun}}=2540.2$) | `Переменная облачность/03.jpg` ($Y_{\text{sun}}=2050.6$) |
| **Выходной файл** | `output/cedar_grove_pano.jpg` ($15000 \times 7500$, 42.66 МБ) | `output/phoenix_mystic_pano.jpg` ($15000 \times 7500$, 42.96 МБ) |
| **Метаданные `GPano` + `GPS`** | Полный набор `GPano:*` + `GPSInfo` | Полный набор `GPano:*` + `GPSInfo` |

````carousel
![Собранная панорама Taylor Morrison Phoenix (15000x7500) с защитой гор выше линии горизонта и выравниванием по солнцу](C:/Users/Georg/.gemini/antigravity-cli/brain/4c86fe01-e1d1-481d-8f92-8306eb5f4ffc/phoenix_mystic_preview.jpg)
<!-- slide -->
![Собранная панорама Cedar Grove TX (15000x7500)](C:/Users/Georg/.gemini/antigravity-cli/brain/4c86fe01-e1d1-481d-8f92-8306eb5f4ffc/cedar_grove_preview.jpg)
````

---

## 3. Команды для запуска

```powershell
# 1. Обработка одной конкретной серии через Docker Compose:
docker compose run --rm pano-pipeline --input /app/input/taylor-morrison---phoenix/mystic-b-south/2026-10-17_01 --output /app/output/phoenix_mystic_pano.jpg

# 2. Пакетная обработка всех серий внутри папки pano/:
docker compose run --rm pano-pipeline --input /app/input --output /app/output

# 3. Переиндексация базы заготовок неба при добавлении новых шаблонов в sky/:
.\.venv\Scripts\python.exe scripts/index_skies.py --sky-dir sky --output sky_catalog.json
```
