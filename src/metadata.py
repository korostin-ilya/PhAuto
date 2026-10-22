"""Модуль чтения EXIF/XMP метаданных снимков БПЛА, астрономического расчета солнца и записи GPano XMP."""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from astral import Observer
from astral.sun import azimuth as sun_azimuth, elevation as sun_elevation
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS
from timezonefinder import TimezoneFinder

Image.MAX_IMAGE_PIXELS = None

_TZ_FINDER: TimezoneFinder | None = None


def _get_tz_finder() -> TimezoneFinder:
    global _TZ_FINDER
    if _TZ_FINDER is None:
        _TZ_FINDER = TimezoneFinder()
    return _TZ_FINDER


def _to_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    if hasattr(val, "numerator") and hasattr(val, "denominator"):
        return float(val.numerator) / float(val.denominator) if val.denominator else default
    if isinstance(val, tuple) and len(val) == 2:
        return float(val[0]) / float(val[1]) if val[1] else default
    try:
        s = str(val).strip()
        if "/" in s:
            num, den = s.split("/", 1)
            return float(num) / float(den)
        return float(s)
    except Exception:
        return default


def _dms_to_decimal(dms: Any, ref: str | bytes | None) -> float | None:
    if not dms or len(dms) < 3:
        return None
    try:
        deg = _to_float(dms[0])
        minutes = _to_float(dms[1])
        seconds = _to_float(dms[2])
        dec = deg + minutes / 60.0 + seconds / 3600.0
        if isinstance(ref, bytes):
            ref = ref.decode("ascii", errors="ignore")
        if ref and str(ref).strip().upper() in ("S", "W"):
            dec = -dec
        return dec
    except Exception:
        return None


def _extract_xmp_dict(image_path: str) -> dict[str, str]:
    """Извлекает атрибуты и теги из пакета <x:xmpmeta> в заголовке JPEG."""
    xmp_data: dict[str, str] = {}
    try:
        with open(image_path, "rb") as f:
            chunk = f.read(512 * 1024)
        start = chunk.find(b"<x:xmpmeta")
        end = chunk.find(b"</x:xmpmeta>")
        if start == -1 or end == -1:
            with open(image_path, "rb") as f:
                chunk = f.read()
            start = chunk.find(b"<x:xmpmeta")
            end = chunk.find(b"</x:xmpmeta>")
        if start != -1 and end != -1:
            xmp_str = chunk[start : end + 12].decode("utf-8", errors="ignore")
            for k, v in re.findall(r'([a-zA-Z0-9_\-]+:[a-zA-Z0-9_\-]+)=["\']([^"\']*)["\']', xmp_str):
                xmp_data[k] = v
            for k, v in re.findall(r"<([a-zA-Z0-9_\-]+:[a-zA-Z0-9_\-]+)>([^<]*)</\1>", xmp_str):
                if k not in xmp_data:
                    xmp_data[k] = v.strip()
    except Exception:
        pass
    return xmp_data


def extract_frame_metadata(image_path: str, default_hfov: float = 73.7) -> dict[str, Any]:
    """Извлекает полные EXIF, GPS и XMP (drone-dji:*) метаданные одного кадра БПЛА."""
    meta: dict[str, Any] = {
        "path": os.path.abspath(image_path),
        "filename": os.path.basename(image_path),
        "width": 5280,
        "height": 3956,
        "make": "DJI",
        "model": "Unknown",
        "datetime_str": None,
        "exposure_time": 0.001,
        "f_number": 2.8,
        "iso": 100.0,
        "focal_length": 12.29,
        "focal_length_35mm": 24.0,
        "ev100": 13.0,
        "gps_lat": None,
        "gps_lon": None,
        "gps_alt": None,
        "gimbal_yaw": 0.0,
        "gimbal_pitch": 0.0,
        "gimbal_roll": 0.0,
        "flight_yaw": 0.0,
        "flight_pitch": 0.0,
        "flight_roll": 0.0,
        "relative_altitude": 35.0,
        "absolute_altitude": 0.0,
        "altitude_type": None,
        "calibrated_focal_length": None,
        "dewarp_data": None,
        "hfov": default_hfov,
        "xmp": {},
    }

    with Image.open(image_path) as img:
        meta["width"], meta["height"] = img.size
        exif = img.getexif()
        if exif:
            for tag_id, value in exif.items():
                tag_name = TAGS.get(tag_id, tag_id)
                if tag_name == "Make":
                    meta["make"] = str(value).strip("\x00 ")
                elif tag_name == "Model":
                    meta["model"] = str(value).strip("\x00 ")
                elif tag_name == "DateTime" and not meta["datetime_str"]:
                    meta["datetime_str"] = str(value).strip("\x00 ")

            # Вложенный каталог ExifIFD (0x8769)
            try:
                exif_ifd = exif.get_ifd(0x8769)
            except Exception:
                exif_ifd = {}
            for tag_id, value in exif_ifd.items():
                tag_name = TAGS.get(tag_id, tag_id)
                if tag_name == "ExposureTime":
                    meta["exposure_time"] = max(_to_float(value, 0.001), 1e-6)
                elif tag_name == "FNumber":
                    meta["f_number"] = max(_to_float(value, 2.8), 0.5)
                elif tag_name in ("ISOSpeedRatings", "PhotographicSensitivity"):
                    if isinstance(value, (tuple, list)) and value:
                        meta["iso"] = max(_to_float(value[0], 100.0), 10.0)
                    else:
                        meta["iso"] = max(_to_float(value, 100.0), 10.0)
                elif tag_name == "FocalLength":
                    meta["focal_length"] = _to_float(value, 12.29)
                elif tag_name in ("FocalLengthIn35mmFilm", "FocalLenIn35mmFilm"):
                    f35 = _to_float(value, 0.0)
                    if f35 > 0:
                        meta["focal_length_35mm"] = f35
                elif tag_name == "DateTimeOriginal":
                    meta["datetime_str"] = str(value).strip("\x00 ")

            # Вложенный каталог GPSInfo (0x8825)
            try:
                gps_ifd = exif.get_ifd(0x8825)
            except Exception:
                gps_ifd = {}
            if gps_ifd:
                gps_named = {GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}
                lat = _dms_to_decimal(gps_named.get("GPSLatitude"), gps_named.get("GPSLatitudeRef"))
                lon = _dms_to_decimal(gps_named.get("GPSLongitude"), gps_named.get("GPSLongitudeRef"))
                alt = _to_float(gps_named.get("GPSAltitude"), None)
                if lat is not None and lon is not None:
                    meta["gps_lat"] = lat
                    meta["gps_lon"] = lon
                if alt is not None:
                    alt_ref = gps_named.get("GPSAltitudeRef", 0)
                    if isinstance(alt_ref, bytes):
                        alt_ref = alt_ref[0] if alt_ref else 0
                    meta["gps_alt"] = -alt if alt_ref == 1 else alt

    xmp = _extract_xmp_dict(image_path)
    meta["xmp"] = xmp

    has_gimbal_xmp = (
        "drone-dji:GimbalYawDegree" in xmp and "drone-dji:GimbalPitchDegree" in xmp
    )
    has_gpano_xmp = any(k.startswith("GPano:") for k in xmp)
    stem_lower = os.path.splitext(meta["filename"])[0].lower()
    is_full_sky_pano = bool(stem_lower == "full" or stem_lower.endswith("_full"))
    aspect_ratio = (meta["width"] / float(meta["height"])) if meta["height"] > 0 else 0.0
    is_equirect = bool(abs(aspect_ratio - 2.0) < 0.08 or meta["width"] >= 8000)
    is_named_pano = bool(
        (stem_lower.endswith("_pano") or stem_lower == "pano")
        and (aspect_ratio >= 1.65 or not has_gimbal_xmp)
    )

    meta["has_gimbal_xmp"] = has_gimbal_xmp
    meta["has_gpano_xmp"] = has_gpano_xmp
    meta["is_full_sky_pano"] = is_full_sky_pano
    meta["is_stitched_panorama"] = bool(
        has_gpano_xmp or is_equirect or is_full_sky_pano or is_named_pano
    )

    if "drone-dji:GimbalYawDegree" in xmp:
        meta["gimbal_yaw"] = _to_float(xmp["drone-dji:GimbalYawDegree"], 0.0)
    if "drone-dji:GimbalPitchDegree" in xmp:
        meta["gimbal_pitch"] = _to_float(xmp["drone-dji:GimbalPitchDegree"], 0.0)
    if "drone-dji:GimbalRollDegree" in xmp:
        meta["gimbal_roll"] = _to_float(xmp["drone-dji:GimbalRollDegree"], 0.0)

    if "drone-dji:FlightYawDegree" in xmp:
        meta["flight_yaw"] = _to_float(xmp["drone-dji:FlightYawDegree"], 0.0)
    if "drone-dji:FlightPitchDegree" in xmp:
        meta["flight_pitch"] = _to_float(xmp["drone-dji:FlightPitchDegree"], 0.0)
    if "drone-dji:FlightRollDegree" in xmp:
        meta["flight_roll"] = _to_float(xmp["drone-dji:FlightRollDegree"], 0.0)

    if "drone-dji:RelativeAltitude" in xmp:
        meta["relative_altitude"] = _to_float(xmp["drone-dji:RelativeAltitude"], 35.0)
    if "drone-dji:AbsoluteAltitude" in xmp:
        meta["absolute_altitude"] = _to_float(xmp["drone-dji:AbsoluteAltitude"], 0.0)
    if "drone-dji:AltitudeType" in xmp:
        meta["altitude_type"] = xmp["drone-dji:AltitudeType"]
    if "drone-dji:CalibratedFocalLength" in xmp:
        meta["calibrated_focal_length"] = _to_float(xmp["drone-dji:CalibratedFocalLength"], None)
    if "drone-dji:DewarpData" in xmp:
        meta["dewarp_data"] = xmp["drone-dji:DewarpData"]

    if meta["gps_lat"] is None and "drone-dji:GpsLatitude" in xmp:
        meta["gps_lat"] = _to_float(xmp["drone-dji:GpsLatitude"], None)
    if meta["gps_lon"] is None and "drone-dji:GpsLongitude" in xmp:
        meta["gps_lon"] = _to_float(xmp["drone-dji:GpsLongitude"], None)
    if meta["gps_alt"] is None and meta["absolute_altitude"]:
        meta["gps_alt"] = meta["absolute_altitude"]

    # Расчет экспозиционного числа EV100
    f_num = max(meta["f_number"], 0.5)
    exp_t = max(meta["exposure_time"], 1e-6)
    iso_val = max(meta["iso"], 10.0)
    meta["ev100"] = math.log2((f_num**2) / exp_t) - math.log2(iso_val / 100.0)

    # Динамический расчет горизонтального угла обзора HFOV с учетом соотношения сторон сенсора (4:3 vs 3:2):
    # В стандарте EXIF FocalLengthIn35mmFilm задается относительно диагонали полного кадра (sqrt(36^2 + 24^2) = 43.2666 мм).
    f35 = meta.get("focal_length_35mm", 0.0)
    if f35 and f35 > 1.0:
        w_f = float(meta.get("width", 5280))
        h_f = float(meta.get("height", 3956))
        diag_px = math.hypot(w_f, h_f)
        sensor_w_35mm = (math.hypot(36.0, 24.0) * (w_f / diag_px)) if diag_px > 0 else 36.0
        meta["hfov"] = 2.0 * math.atan(sensor_w_35mm / (2.0 * f35)) * (180.0 / math.pi)
    elif meta["calibrated_focal_length"] and meta["calibrated_focal_length"] > 100.0:
        meta["hfov"] = (
            2.0
            * math.atan(meta["width"] / (2.0 * meta["calibrated_focal_length"]))
            * (180.0 / math.pi)
        )
    else:
        meta["hfov"] = float(default_hfov)

    return meta


def _spherical_angle_distance_deg(
    yaw1_deg: float, pitch1_deg: float, yaw2_deg: float, pitch2_deg: float
) -> float:
    """
    Вычисляет угловое расстояние на единичной сфере (в градусах) между направлениями взгляда
    камеры (yaw1, pitch1) и (yaw2, pitch2). Корректно учитывает сингулярность в надире (pitch = -90°).
    """
    p1 = math.radians(pitch1_deg)
    y1 = math.radians(yaw1_deg)
    p2 = math.radians(pitch2_deg)
    y2 = math.radians(yaw2_deg)
    dot = math.cos(p1) * math.cos(p2) * math.cos(y1 - y2) + math.sin(p1) * math.sin(p2)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def _file_sha256(path: str) -> str | None:
    try:
        if not os.path.isfile(path):
            return None
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def filter_series_frames(
    frames: list[dict[str, Any]],
    filter_cfg: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Трехступенчатая фильтрация входных снимков для сборки панорамы:
    1. Отсечение уже собранных панорам (GPano:*, соотношение 2:1, ширина >= 10000, *_full.jpg).
    2. Проверка наличия XMP-телеметрии подвеса (drone-dji:Gimbal*) и единого разрешения сенсора.
    3. Удаление побайтовых (SHA-256) и угловых дубликатов на сфере (d_theta < duplicate_angle_threshold_deg).
    Возвращает (kept_raw_frames, filter_report).
    """
    cfg = filter_cfg or {}
    exclude_panos = bool(cfg.get("exclude_stitched_panoramas", True))
    require_gimbal = bool(cfg.get("require_gimbal_xmp", True))
    dup_angle_thresh = float(cfg.get("duplicate_angle_threshold_deg", 3.0))

    stitched_panoramas: list[dict[str, Any]] = []
    rejected_no_gimbal: list[str] = []
    rejected_resolution: list[str] = []
    rejected_duplicates: list[str] = []

    # Ступень 1: Отсечение уже собранных панорам
    stage1_candidates: list[dict[str, Any]] = []
    for f in frames:
        if exclude_panos and f.get("is_stitched_panorama", False):
            stitched_panoramas.append(f)
        else:
            stage1_candidates.append(f)

    # Ступень 2: Проверка телеметрии подвеса и основного разрешения сенсора
    stage2_gimbal: list[dict[str, Any]] = []
    for f in stage1_candidates:
        if require_gimbal and not f.get("has_gimbal_xmp", True):
            rejected_no_gimbal.append(f.get("filename", ""))
        else:
            stage2_gimbal.append(f)

    stage2_candidates: list[dict[str, Any]] = []
    if stage2_gimbal:
        res_counts = Counter((int(f["width"]), int(f["height"])) for f in stage2_gimbal)
        dominant_res, _ = res_counts.most_common(1)[0]
        for f in stage2_gimbal:
            if (int(f["width"]), int(f["height"])) != dominant_res:
                rejected_resolution.append(f.get("filename", ""))
            else:
                stage2_candidates.append(f)

    # Ступень 3: Удаление побайтовых и угловых дубликатов
    # Чтобы не читать полные 10-15 МБ файлы по сети/монтированию Docker ради SHA-256,
    # вычисляем хеш ТОЛЬКО тогда, когда у двух файлов в папке абсолютно одинаковый размер в байтах.
    kept_frames: list[dict[str, Any]] = []
    size_to_paths: dict[int, list[str]] = {}
    size_to_hashes: dict[int, set[str]] = {}
    alts = sorted(float(f.get("relative_altitude", 35.0)) for f in stage2_candidates)
    median_alt = alts[len(alts) // 2] if alts else 35.0

    for f in stage2_candidates:
        fpath = f.get("path", "")
        if fpath and os.path.isfile(fpath):
            try:
                fsize = os.path.getsize(fpath)
            except OSError:
                fsize = -1
            if fsize > 0:
                if fsize in size_to_paths:
                    # Вычисляем SHA-256 первого файла такого размера лениво при первой коллизии размера
                    if fsize not in size_to_hashes:
                        first_hash = _file_sha256(size_to_paths[fsize][0])
                        size_to_hashes[fsize] = {first_hash} if first_hash else set()
                    cur_hash = _file_sha256(fpath)
                    if cur_hash and cur_hash in size_to_hashes[fsize]:
                        rejected_duplicates.append(f.get("filename", ""))
                        continue
                    if cur_hash:
                        size_to_hashes[fsize].add(cur_hash)
                    size_to_paths[fsize].append(fpath)
                else:
                    size_to_paths[fsize] = [fpath]

        if dup_angle_thresh > 0.0:
            y_cur = float(f.get("gimbal_yaw", 0.0))
            p_cur = float(f.get("gimbal_pitch", 0.0))
            alt_cur = float(f.get("relative_altitude", 35.0))

            dup_idx: int | None = None
            for idx, existing in enumerate(kept_frames):
                d_theta = _spherical_angle_distance_deg(
                    y_cur,
                    p_cur,
                    float(existing.get("gimbal_yaw", 0.0)),
                    float(existing.get("gimbal_pitch", 0.0)),
                )
                if d_theta < dup_angle_thresh:
                    dup_idx = idx
                    break

            if dup_idx is not None:
                existing = kept_frames[dup_idx]
                alt_ex = float(existing.get("relative_altitude", 35.0))
                if abs(alt_cur - median_alt) < abs(alt_ex - median_alt) - 1e-3:
                    rejected_duplicates.append(existing.get("filename", ""))
                    kept_frames[dup_idx] = f
                else:
                    rejected_duplicates.append(f.get("filename", ""))
                continue

        kept_frames.append(f)

    report = {
        "total_files": len(frames),
        "kept_raw_count": len(kept_frames),
        "stitched_panoramas": stitched_panoramas,
        "stitched_panorama_files": [p.get("filename", "") for p in stitched_panoramas],
        "rejected_no_gimbal": rejected_no_gimbal,
        "rejected_resolution": rejected_resolution,
        "rejected_duplicates": rejected_duplicates,
    }
    return kept_frames, report


def select_prestitched_base_panorama(
    stitched_panoramas: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Выбирает из найденных в папке готовых панорам заготовку БЕЗ замененного неба
    (исключая *_full.jpg / full.jpg). Приоритет отдается заготовкам 15000x7500 (Autopano),
    затем внутрикамерным панорамам DJI (14400x7200).
    """
    base_candidates = [
        p for p in stitched_panoramas if not p.get("is_full_sky_pano", False)
    ]
    if not base_candidates:
        return None

    def _rank_key(p: dict[str, Any]) -> tuple[int, int, str]:
        w = int(p.get("width", 0))
        h = int(p.get("height", 0))
        is_exact_15k = 1 if (w == 15000 and h == 7500) else 0
        return (-is_exact_15k, -(w * h), p.get("filename", ""))

    return sorted(base_candidates, key=_rank_key)[0]


def inspect_shoot_directory(
    input_dir: str,
    default_hfov: float = 73.7,
    filter_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Анализирует папку съемки:
    - Если найдено >= min_raw_frames сырых кадров БПЛА -> режим 'stitch' (Модуль 1 + Модуль 2).
    - Если сырых кадров нет, но есть готовая сшитая панорама без неба -> режим 'sky_only' (только Модуль 2).
    - Иначе -> режим 'empty'.
    """
    entries = sorted(
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg")) and not f.startswith(".")
    )
    if not entries:
        raise FileNotFoundError(f"В папке {input_dir} не найдено JPEG-снимков.")

    all_frames = [
        extract_frame_metadata(os.path.join(input_dir, fname), default_hfov=default_hfov)
        for fname in entries
    ]
    raw_frames, report = filter_series_frames(all_frames, filter_cfg=filter_cfg)
    min_raw = int((filter_cfg or {}).get("min_raw_frames", 5))
    base_pano = select_prestitched_base_panorama(report["stitched_panoramas"])

    if len(raw_frames) >= min_raw:
        mode = "stitch"
    elif base_pano is not None:
        mode = "sky_only"
    else:
        mode = "empty"

    return {
        "mode": mode,
        "raw_frames": raw_frames,
        "prestitched_pano": base_pano,
        "filter_report": report,
    }


def load_series_metadata(
    input_dir: str,
    default_hfov: float = 73.7,
    filter_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Читает и фильтрует метаданные сырых JPEG-снимков серии из входной директории
    (отбрасывая уже сшитые панорамы, чужие изображения и дубликаты).
    Если в папке нет сырых кадров, но есть сшитая заготовка панорамы без неба, возвращает [prestitched_pano].
    """
    inspected = inspect_shoot_directory(
        input_dir=input_dir,
        default_hfov=default_hfov,
        filter_cfg=filter_cfg,
    )
    if inspected["raw_frames"]:
        return inspected["raw_frames"]
    if inspected["prestitched_pano"] is not None:
        return [inspected["prestitched_pano"]]
    raise ValueError(
        f"В папке {input_dir} не найдено ни сырых кадров БПЛА с телеметрией подвеса, "
        f"ни заготовки панорамы без неба."
    )


def compute_sun_position(
    frames_meta: list[dict[str, Any]],
    pano_width: int = 15000,
    horizon_y: int = 3750,
    overcast_ev_threshold: float = 1.15,
) -> dict[str, Any]:
    """
    Вычисляет часовой пояс точки съемки, UTC-время, астрономический азимут и высоту солнца,
    пиксельные координаты (X_sun_pano, Y_sun_pano) и индикатор пасмурной погоды по разбросу EV100.
    """
    ref = next(
        (f for f in frames_meta if f.get("gps_lat") is not None and f.get("gps_lon") is not None),
        frames_meta[0],
    )
    lat = ref.get("gps_lat")
    lon = ref.get("gps_lon")
    alt = ref.get("gps_alt") or 0.0
    dt_str = ref.get("datetime_str")

    tz_name = "UTC"
    if lat is not None and lon is not None:
        found_tz = _get_tz_finder().timezone_at(lng=float(lon), lat=float(lat))
        if found_tz:
            tz_name = found_tz

    local_tz = ZoneInfo(tz_name)
    if dt_str:
        try:
            dt_naive = datetime.strptime(dt_str[:19], "%Y:%m:%d %H:%M:%S")
        except ValueError:
            dt_naive = datetime.fromisoformat(dt_str[:19])
        dt_local = dt_naive.replace(tzinfo=local_tz)
    else:
        dt_local = datetime.now(tz=local_tz)

    dt_utc = dt_local.astimezone(timezone.utc)

    if lat is not None and lon is not None:
        observer = Observer(latitude=float(lat), longitude=float(lon), elevation=max(float(alt), 0.0))
        az_deg = float(sun_azimuth(observer, dateandtime=dt_utc)) % 360.0
        el_deg = float(sun_elevation(observer, dateandtime=dt_utc))
    else:
        az_deg = 180.0
        el_deg = 45.0

    # Перевод в пиксельные координаты панорамы 15000x7500 (Север = 0° -> X = 7500)
    x_sun_pano = (pano_width / 2.0 + (az_deg / 360.0) * pano_width) % pano_width
    y_sun_pano = horizon_y * (1.0 - (el_deg / 90.0))
    y_sun_pano_clamped = float(max(0.0, min(float(horizon_y), y_sun_pano)))

    # Разброс экспозиции верхнего ряда кадров (GimbalPitchDegree >= -26.0)
    upper_evs = [
        float(f["ev100"]) for f in frames_meta if float(f.get("gimbal_pitch", -90.0)) >= -26.0
    ]
    if len(upper_evs) >= 2:
        delta_ev100 = max(upper_evs) - min(upper_evs)
    else:
        all_evs = [float(f["ev100"]) for f in frames_meta]
        delta_ev100 = (max(all_evs) - min(all_evs)) if len(all_evs) >= 2 else 0.0

    is_overcast_ev = bool(delta_ev100 < overcast_ev_threshold)

    return {
        "latitude": lat,
        "longitude": lon,
        "altitude": alt,
        "timezone": tz_name,
        "datetime_local": dt_local.isoformat(),
        "datetime_utc": dt_utc.isoformat(),
        "azimuth_deg": az_deg,
        "elevation_deg": el_deg,
        "x_sun_pano": float(x_sun_pano),
        "y_sun_pano": float(y_sun_pano),
        "y_sun_pano_clamped": y_sun_pano_clamped,
        "delta_ev100_upper": float(delta_ev100),
        "is_overcast_ev": is_overcast_ev,
    }


def build_gpano_xmp_packet(
    full_width: int = 15000,
    full_height: int = 7500,
    pose_heading_deg: float = 0.0,
) -> bytes:
    """Формирует XMP-пакет стандарта Google Photo Sphere (GPano)."""
    xmp_xml = f"""<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="DronePanoPipeline">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:GPano="http://ns.google.com/photos/1.0/panorama/">
   <GPano:UsePanoramaViewer>True</GPano:UsePanoramaViewer>
   <GPano:ProjectionType>equirectangular</GPano:ProjectionType>
   <GPano:FullPanoWidthPixels>{full_width}</GPano:FullPanoWidthPixels>
   <GPano:FullPanoHeightPixels>{full_height}</GPano:FullPanoHeightPixels>
   <GPano:CroppedAreaImageWidthPixels>{full_width}</GPano:CroppedAreaImageWidthPixels>
   <GPano:CroppedAreaImageHeightPixels>{full_height}</GPano:CroppedAreaImageHeightPixels>
   <GPano:CroppedAreaLeftPixels>0</GPano:CroppedAreaLeftPixels>
   <GPano:CroppedAreaTopPixels>0</GPano:CroppedAreaTopPixels>
   <GPano:PoseHeadingDegrees>{pose_heading_deg:.1f}</GPano:PoseHeadingDegrees>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""
    return xmp_xml.encode("utf-8")


def _inject_xmp_into_jpeg_file(jpeg_path: str, xmp_packet: bytes) -> None:
    """Встраивает APP1-сегмент с XMP-пакетом в JPEG без перекодирования изображения."""
    with open(jpeg_path, "rb") as f:
        data = f.read()

    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return

    xmp_header = b"http://ns.adobe.com/xap/1.0/\x00"
    payload = xmp_header + xmp_packet
    seg_len = len(payload) + 2
    app1_xmp = b"\xff\xe1" + seg_len.to_bytes(2, "big") + payload

    # Удаляем старый XMP-сегмент, если он был, и вставляем новый сразу после SOI / первого APP0/APP1
    pos = 2
    insert_pos = 2
    cleaned_chunks = [data[:2]]
    while pos + 4 <= len(data) and data[pos] == 0xFF:
        marker = data[pos + 1]
        if marker in (0xDA, 0xD9):  # SOS или EOI
            break
        if marker in (0xD8, 0x01) or (0xD0 <= marker <= 0xD7):
            cleaned_chunks.append(data[pos : pos + 2])
            pos += 2
            continue
        length = int.from_bytes(data[pos + 2 : pos + 4], "big")
        seg_end = pos + 2 + length
        seg_bytes = data[pos:seg_end]
        if marker == 0xE1 and data[pos + 4 : pos + 4 + len(xmp_header)] == xmp_header:
            # Пропускаем существующий XMP-пакет
            pos = seg_end
            continue
        cleaned_chunks.append(seg_bytes)
        if marker in (0xE0, 0xE1):
            insert_pos = sum(len(c) for c in cleaned_chunks)
        pos = seg_end

    cleaned_chunks.append(data[pos:])
    rebuilt = b"".join(cleaned_chunks)
    final_bytes = rebuilt[:insert_pos] + app1_xmp + rebuilt[insert_pos:]

    with open(jpeg_path, "wb") as f:
        f.write(final_bytes)


def get_reference_exif_bytes(reference_jpg_path: str, full_width: int = 15000, full_height: int = 7500) -> bytes | None:
    """Извлекает EXIF (включая GPSInfo и ExifIFD) из опорного снимка серии для сохранения в выходной JPEG."""
    try:
        with Image.open(reference_jpg_path) as ref_img:
            exif = ref_img.getexif()
            if not exif:
                return None
            # Обновляем размеры и ориентацию для сшитой панорамы
            exif[0x0112] = 1  # Orientation = Normal (1)
            exif[0x0100] = full_width
            exif[0x0101] = full_height
            try:
                exif_ifd = exif.get_ifd(0x8769)
                if exif_ifd is not None:
                    exif_ifd[0xA002] = full_width  # ExifImageWidth
                    exif_ifd[0xA003] = full_height  # ExifImageHeight
            except Exception:
                pass
            return exif.tobytes()
    except Exception:
        return None


def write_output_metadata(
    output_jpg_path: str,
    reference_jpg_path: str,
    full_width: int = 15000,
    full_height: int = 7500,
    pose_heading_deg: float = 0.0,
) -> None:
    """
    Переносит геолокацию (GPSInfo), дату съемки и модель камеры из опорного снимка
    и вшивает XMP-теги Google Photo Sphere (GPano).
    Использует ExifTool при наличии в системе, а также нативный инжектор APP1 XMP.
    """
    xmp_packet = build_gpano_xmp_packet(
        full_width=full_width,
        full_height=full_height,
        pose_heading_deg=pose_heading_deg,
    )
    _inject_xmp_into_jpeg_file(output_jpg_path, xmp_packet)

    exiftool_bin = shutil.which("exiftool")
    if exiftool_bin:
        cmd = [
            exiftool_bin,
            "-overwrite_original",
            "-TagsFromFile",
            reference_jpg_path,
            "-GPS:all",
            "-ExifIFD:DateTimeOriginal",
            "-ExifIFD:CreateDate",
            "-IFD0:Make",
            "-IFD0:Model",
            "-IFD0:DateTime",
            "-IFD0:Orientation=Horizontal (normal)",
            f"-XMP-GPano:UsePanoramaViewer=True",
            f"-XMP-GPano:ProjectionType=equirectangular",
            f"-XMP-GPano:FullPanoWidthPixels={full_width}",
            f"-XMP-GPano:FullPanoHeightPixels={full_height}",
            f"-XMP-GPano:CroppedAreaImageWidthPixels={full_width}",
            f"-XMP-GPano:CroppedAreaImageHeightPixels={full_height}",
            f"-XMP-GPano:CroppedAreaLeftPixels=0",
            f"-XMP-GPano:CroppedAreaTopPixels=0",
            f"-XMP-GPano:PoseHeadingDegrees={pose_heading_deg:.1f}",
            output_jpg_path,
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception:
            # Встроенный инжектор уже записал GPano XMP и EXIF от Pillow
            pass
