"""Модуль чтения EXIF/XMP метаданных снимков БПЛА, астрономического расчета солнца и записи GPano XMP."""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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
        "datetime_iso_aware": None,
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
        "flight_x_speed": 0.0,
        "flight_y_speed": 0.0,
        "flight_z_speed": 0.0,
        "linear_speed": 0.0,
        "rtk_flag": 0,
        "rtk_std_lon": None,
        "rtk_std_lat": None,
        "rtk_std_hgt": None,
        "relative_altitude": 35.0,
        "absolute_altitude": 0.0,
        "altitude_type": None,
        "calibrated_focal_length": None,
        "calibrated_optical_center_x": None,
        "calibrated_optical_center_y": None,
        "d0": 0.0,
        "e0": 0.0,
        "dewarp_data": None,
        "hfov": default_hfov,
        "xmp": {},
    }

    exif_offset_time: str | None = None
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
                elif tag_name in ("OffsetTimeOriginal", "OffsetTime") and not exif_offset_time:
                    exif_offset_time = str(value).strip("\x00 ")

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

    # Извлекаем ISO-время со смещением часового пояса пульта (например, xmp:CreateDate="2026-10-28T00:23:23+08:00"),
    # чтобы корректно переводить время съемки в UTC даже если пульт дрона был настроен на чужой часовой пояс (+08:00 / -07:00)
    xmp_dt_str = xmp.get("xmp:CreateDate") or xmp.get("xmp:ModifyDate") or xmp.get("photoshop:DateCreated")
    if xmp_dt_str:
        meta["datetime_iso_aware"] = str(xmp_dt_str).strip()
    elif meta["datetime_str"] and exif_offset_time:
        try:
            date_part = meta["datetime_str"][:10].replace(":", "-")
            time_part = meta["datetime_str"][11:19]
            meta["datetime_iso_aware"] = f"{date_part}T{time_part}{exif_offset_time}"
        except Exception:
            pass

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

    if "drone-dji:FlightXSpeed" in xmp:
        meta["flight_x_speed"] = _to_float(xmp["drone-dji:FlightXSpeed"], 0.0)
    if "drone-dji:FlightYSpeed" in xmp:
        meta["flight_y_speed"] = _to_float(xmp["drone-dji:FlightYSpeed"], 0.0)
    if "drone-dji:FlightZSpeed" in xmp:
        meta["flight_z_speed"] = _to_float(xmp["drone-dji:FlightZSpeed"], 0.0)
    meta["linear_speed"] = math.sqrt(
        meta["flight_x_speed"] ** 2 + meta["flight_y_speed"] ** 2 + meta["flight_z_speed"] ** 2
    )

    if "drone-dji:RtkFlag" in xmp:
        meta["rtk_flag"] = int(_to_float(xmp["drone-dji:RtkFlag"], 0))
    if "drone-dji:RtkStdLon" in xmp:
        meta["rtk_std_lon"] = _to_float(xmp["drone-dji:RtkStdLon"], None)
    if "drone-dji:RtkStdLat" in xmp:
        meta["rtk_std_lat"] = _to_float(xmp["drone-dji:RtkStdLat"], None)
    if "drone-dji:RtkStdHgt" in xmp:
        meta["rtk_std_hgt"] = _to_float(xmp["drone-dji:RtkStdHgt"], None)

    if "drone-dji:RelativeAltitude" in xmp:
        meta["relative_altitude"] = _to_float(xmp["drone-dji:RelativeAltitude"], 35.0)
    if "drone-dji:AbsoluteAltitude" in xmp:
        meta["absolute_altitude"] = _to_float(xmp["drone-dji:AbsoluteAltitude"], 0.0)
    if "drone-dji:AltitudeType" in xmp:
        meta["altitude_type"] = xmp["drone-dji:AltitudeType"]
    if "drone-dji:CalibratedFocalLength" in xmp:
        meta["calibrated_focal_length"] = _to_float(xmp["drone-dji:CalibratedFocalLength"], None)
    if "drone-dji:CalibratedOpticalCenterX" in xmp:
        meta["calibrated_optical_center_x"] = _to_float(
            xmp["drone-dji:CalibratedOpticalCenterX"], None
        )
    if "drone-dji:CalibratedOpticalCenterY" in xmp:
        meta["calibrated_optical_center_y"] = _to_float(
            xmp["drone-dji:CalibratedOpticalCenterY"], None
        )
    if "drone-dji:DewarpData" in xmp:
        meta["dewarp_data"] = xmp["drone-dji:DewarpData"]

    # Смещение оптического центра (d0, e0) из заводской калибровки DJI
    oc_x = meta["calibrated_optical_center_x"]
    oc_y = meta["calibrated_optical_center_y"]
    if oc_x is not None and oc_y is not None and oc_x > 100.0 and oc_y > 100.0:
        d0_val = float(oc_x) - (float(meta["width"]) / 2.0)
        e0_val = float(oc_y) - (float(meta["height"]) / 2.0)
        if abs(d0_val) <= 250.0 and abs(e0_val) <= 250.0:
            meta["d0"] = d0_val
            meta["e0"] = e0_val

    # XMP-координаты RTK имеют точность 9 знаков после запятой (против квантованных секунд в EXIF DMS),
    # а на старых прошивках DJI FC6310S встречается тег с опечаткой drone-dji:GpsLongtitude
    xmp_lat = _to_float(xmp.get("drone-dji:GpsLatitude"), None)
    xmp_lon = _to_float(
        xmp.get("drone-dji:GpsLongitude", xmp.get("drone-dji:GpsLongtitude")),
        None,
    )
    if xmp_lat is not None and abs(xmp_lat) > 1e-5:
        meta["gps_lat"] = xmp_lat
    if xmp_lon is not None and abs(xmp_lon) > 1e-5:
        meta["gps_lon"] = xmp_lon
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


def _parse_frame_timestamp_sec(frame: dict[str, Any]) -> float | None:
    """Извлекает монотонную временную метку кадра (в секундах) из XMP/EXIF для хронологической сортировки."""
    dt_iso = frame.get("datetime_iso_aware") or (frame.get("xmp") or {}).get("xmp:CreateDate")
    if dt_iso:
        try:
            return datetime.fromisoformat(str(dt_iso).strip()).timestamp()
        except Exception:
            pass
    dt_str = frame.get("datetime_str")
    if dt_str:
        s = str(dt_str).strip()[:19]
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                continue
    return None


def _extract_filename_series_key(filename: str) -> tuple[str, int | None, str]:
    """
    Выделяет из имени файла:
    - norm_prefix: нормализованный префикс серии (например, 'pano(1)', 'pano(2)', 'dji_<ts>_')
    - frame_num: порядковый номер кадра внутри серии (1..25)
    - display_label: читаемая метка серии (например, 'Pano(1)')
    """
    stem = os.path.splitext(os.path.basename(filename))[0].strip()
    m = re.match(r"^(.*?)(\d+)(?:[_\-][A-Za-z])?$", stem)
    if not m:
        return stem.lower(), None, stem
    raw_prefix = m.group(1).strip()
    frame_num = int(m.group(2))
    # Нормализуем 14-значные метки времени в именах вида DJI_20261029071958_0001_D.JPG,
    # чтобы посекундное изменение имени не дробило одну серию на отдельные префиксы
    norm_prefix = re.sub(r"20\d{12}", "<ts>", raw_prefix.lower()).strip("_- ")
    display_label = raw_prefix.rstrip("_- ") or stem
    return norm_prefix, frame_num, display_label


def _gps_horizontal_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Вычисляет горизонтальное расстояние (в метрах) между двумя точками GPS/RTK."""
    dy = (lat2 - lat1) * 111320.0
    mean_lat_rad = math.radians((lat1 + lat2) * 0.5)
    dx = (lon2 - lon1) * 111320.0 * math.cos(mean_lat_rad)
    return math.hypot(dx, dy)


def _azimuth_octants_covered(frames_subset: list[dict[str, Any]]) -> set[int]:
    """Возвращает множество покрытых 45-градусных азимутальных секторов (0..7) для ненадирных кадров."""
    octants: set[int] = set()
    for f in frames_subset:
        if float(f.get("gimbal_pitch", 0.0)) > -75.0:
            yaw = float(f.get("gimbal_yaw", 0.0))
            oct_idx = int(((yaw + 180.0) % 360.0) // 45.0) % 8
            octants.add(oct_idx)
    return octants


def _has_nadir_frame(frames_subset: list[dict[str, Any]]) -> bool:
    return any(float(f.get("gimbal_pitch", 0.0)) <= -82.0 for f in frames_subset)


def _is_complete_sphere_series(frames_subset: list[dict[str, Any]]) -> bool:
    """
    Проверяет, образует ли набор кадров полный сферический обход панорамы:
    - покрыто >= 7 из 8 азимутальных секторов (360° по горизонтали),
    - присутствуют кадры верхнего/горизонтального пояса (pitch > -30°),
    - присутствуют кадры нижнего наклонного пояса (-75° < pitch <= -35°),
    - присутствует кадр надира (pitch <= -82°).
    """
    if len(frames_subset) < 20:
        return False
    if len(_azimuth_octants_covered(frames_subset)) < 7:
        return False
    if not _has_nadir_frame(frames_subset):
        return False
    has_upper = any(float(f.get("gimbal_pitch", 0.0)) > -30.0 for f in frames_subset)
    has_lower_oblique = any(-75.0 < float(f.get("gimbal_pitch", 0.0)) <= -35.0 for f in frames_subset)
    return has_upper and has_lower_oblique


def split_raw_frames_into_series(
    stage2_candidates: list[dict[str, Any]],
    filter_cfg: dict[str, Any] | None = None,
) -> list[list[dict[str, Any]]]:
    """
    Разделяет список валидных сырых кадров из одной папки на отдельные серии панорамной съемки
    по 4 независимым признакам:
    1. Явный префикс серии в имени файла (например, Pano(1)1..25 и Pano(2)1..25), если в каждой группе >= min_raw_frames.
    2. Разрыв по времени съемки (xmp:CreateDate / DateTimeOriginal) > series_time_gap_sec (по умолчанию 15 с).
    3. Пространственный скачок точки зависания RTK/GPS > series_gps_distance_m (по умолчанию 10 м) или высоты > 8 м.
    4. Завершение полного 360° цикла обхода подвеса (8 азимутальных секторов + надир) с началом нового цикла.
    """
    if not stage2_candidates:
        return []

    cfg = filter_cfg or {}
    min_raw = max(2, int(cfg.get("min_raw_frames", 5)))
    time_gap_thresh = float(cfg.get("series_time_gap_sec", 15.0))
    gps_dist_thresh = float(cfg.get("series_gps_distance_m", 10.0))
    alt_diff_thresh = float(cfg.get("series_altitude_diff_m", 8.0))

    enriched: list[tuple[int, str, int, float | None, dict[str, Any]]] = []
    prefix_counts: Counter[str] = Counter()
    for orig_idx, f in enumerate(stage2_candidates):
        norm_pref, f_num, _label = _extract_filename_series_key(f.get("filename", ""))
        ts = _parse_frame_timestamp_sec(f)
        prefix_counts[norm_pref] += 1
        enriched.append((orig_idx, norm_pref, f_num if f_num is not None else 10**9 + orig_idx, ts, f))

    # 1. Группировка по явному префиксу серии (например, 'pano(1)' vs 'pano(2)'),
    # если обнаружено >= 2 крупных префиксов (по >= min_raw кадров в каждом)
    major_prefixes = [pref for pref, cnt in prefix_counts.items() if cnt >= min_raw and pref != ""]
    if len(major_prefixes) >= 2:
        groups_by_pref: dict[str, list[tuple[int, str, int, float | None, dict[str, Any]]]] = {
            pref: [] for pref in major_prefixes
        }
        minor_items: list[tuple[int, str, int, float | None, dict[str, Any]]] = []
        for item in enriched:
            if item[1] in groups_by_pref:
                groups_by_pref[item[1]].append(item)
            else:
                minor_items.append(item)

        # Одиночные кадры с нестандартным именем прикрепляем к ближайшей по времени основной группе
        for m_item in minor_items:
            m_ts = m_item[3]
            best_pref = major_prefixes[0]
            best_dist = float("inf")
            for pref, g_items in groups_by_pref.items():
                g_ts_list = [x[3] for x in g_items if x[3] is not None]
                if m_ts is not None and g_ts_list:
                    d_t = min(abs(m_ts - g_ts) for g_ts in g_ts_list)
                    if d_t < best_dist:
                        best_dist = d_t
                        best_pref = pref
            groups_by_pref[best_pref].append(m_item)

        def _group_sort_key(pref: str) -> tuple[float, str]:
            g_ts = [x[3] for x in groups_by_pref[pref] if x[3] is not None]
            return (min(g_ts) if g_ts else float("inf"), pref)

        initial_groups = [groups_by_pref[p] for p in sorted(major_prefixes, key=_group_sort_key)]
    else:
        initial_groups = [enriched]

    # 2. Внутри каждой группы выполняем хронологическую сортировку и деление по времени, GPS и 360°-циклу
    final_series: list[list[dict[str, Any]]] = []

    for group in initial_groups:
        has_all_ts = sum(1 for x in group if x[3] is not None) >= max(1, int(0.8 * len(group)))
        if has_all_ts:
            sorted_group = sorted(
                group,
                key=lambda x: (
                    x[3] if x[3] is not None else float("inf"),
                    x[2],
                    x[4].get("filename", ""),
                ),
            )
        else:
            sorted_group = sorted(group, key=lambda x: (x[1], x[2], x[4].get("filename", "")))

        cur_items: list[tuple[int, str, int, float | None, dict[str, Any]]] = []
        total_n = len(sorted_group)

        for idx, item in enumerate(sorted_group):
            if not cur_items:
                cur_items.append(item)
                continue

            rem_count = total_n - idx
            cur_frames = [x[4] for x in cur_items]
            should_split = False

            if len(cur_items) >= min_raw and rem_count >= min_raw:
                prev_ts = cur_items[-1][3]
                cur_ts = item[3]
                f_cur = item[4]

                # Критерий А: Временной разрыв между последовательными кадрами > 15 с
                if prev_ts is not None and cur_ts is not None and (cur_ts - prev_ts) > time_gap_thresh:
                    should_split = True

                # Критерий Б: Скачок GPS-позиции точки зависания > 10 м или высоты > 8 м
                if not should_split:
                    lat_c = f_cur.get("gps_lat")
                    lon_c = f_cur.get("gps_lon")
                    cur_lats = sorted(
                        float(x["gps_lat"]) for x in cur_frames if x.get("gps_lat") is not None
                    )
                    cur_lons = sorted(
                        float(x["gps_lon"]) for x in cur_frames if x.get("gps_lon") is not None
                    )
                    if lat_c is not None and lon_c is not None and cur_lats and cur_lons:
                        med_lat = cur_lats[len(cur_lats) // 2]
                        med_lon = cur_lons[len(cur_lons) // 2]
                        dist_m = _gps_horizontal_distance_m(med_lat, med_lon, float(lat_c), float(lon_c))
                        if dist_m > gps_dist_thresh:
                            should_split = True

                    cur_alts = sorted(float(x.get("relative_altitude", 35.0)) for x in cur_frames)
                    med_alt = cur_alts[len(cur_alts) // 2]
                    if abs(float(f_cur.get("relative_altitude", med_alt)) - med_alt) > alt_diff_thresh:
                        should_split = True

                # Критерий В: Завершен полный 360°-цикл сферы (>= 24 кадров, все пояса pitch + надир),
                # а оставшиеся кадры также образуют полноценный сферический обход
                if not should_split and len(cur_frames) >= 24 and rem_count >= 20:
                    rem_frames = [x[4] for x in sorted_group[idx:]]
                    if _is_complete_sphere_series(cur_frames) and _is_complete_sphere_series(rem_frames):
                        prev_num = cur_items[-1][2]
                        cur_num = item[2]
                        if cur_num < prev_num or len(cur_frames) >= 25:
                            should_split = True

            if should_split:
                final_series.append([x[4] for x in cur_items])
                cur_items = [item]
            else:
                cur_items.append(item)

        if cur_items:
            final_series.append([x[4] for x in cur_items])

    return final_series


def _deduplicate_single_series(
    series_frames: list[dict[str, Any]],
    dup_angle_thresh: float = 3.0,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Удаляет побайтовые (SHA-256) и угловые дубликаты на сфере (d_theta < dup_angle_thresh)
    внутри одной серии панорамных снимков.
    """
    kept_frames: list[dict[str, Any]] = []
    rejected_duplicates: list[str] = []
    size_to_paths: dict[int, list[str]] = {}
    size_to_hashes: dict[int, set[str]] = {}
    alts = sorted(float(f.get("relative_altitude", 35.0)) for f in series_frames)
    median_alt = alts[len(alts) // 2] if alts else 35.0

    for f in series_frames:
        fpath = f.get("path", "")
        if fpath and os.path.isfile(fpath):
            try:
                fsize = os.path.getsize(fpath)
            except OSError:
                fsize = -1
            if fsize > 0:
                if fsize in size_to_paths:
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

    return kept_frames, rejected_duplicates


def _summarize_series(series_frames: list[dict[str, Any]], series_idx: int, rejected_dups: list[str]) -> dict[str, Any]:
    """Формирует краткий паспорт выделенной серии кадров (метка, время, координаты, число кадров)."""
    if not series_frames:
        return {"series_index": series_idx, "label": f"series_{series_idx}", "frame_count": 0}
    _, _, label = _extract_filename_series_key(series_frames[0].get("filename", ""))
    lats = [float(f["gps_lat"]) for f in series_frames if f.get("gps_lat") is not None]
    lons = [float(f["gps_lon"]) for f in series_frames if f.get("gps_lon") is not None]
    return {
        "series_index": series_idx,
        "label": label,
        "frame_count": len(series_frames),
        "first_frame": series_frames[0].get("filename", ""),
        "last_frame": series_frames[-1].get("filename", ""),
        "datetime_start": series_frames[0].get("datetime_iso_aware") or series_frames[0].get("datetime_str"),
        "datetime_end": series_frames[-1].get("datetime_iso_aware") or series_frames[-1].get("datetime_str"),
        "gps_lat": round(sum(lats) / len(lats), 7) if lats else None,
        "gps_lon": round(sum(lons) / len(lons), 7) if lons else None,
        "rejected_duplicates": rejected_dups,
    }


def filter_series_frames(
    frames: list[dict[str, Any]],
    filter_cfg: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Многоступенчатая фильтрация и разделение входных снимков для сборки панорамы:
    1. Отсечение уже собранных панорам (GPano:*, соотношение 2:1, ширина >= 10000, *_full.jpg).
    2. Проверка наличия XMP-телеметрии подвеса (drone-dji:Gimbal*) и единого разрешения сенсора.
    3. Разделение сырых кадров на независимые серии съемки (по префиксу имени, времени, GPS и 360°-циклу)
       + удаление побайтовых (SHA-256) и угловых дубликатов (d_theta < duplicate_angle_threshold_deg) внутри каждой серии.
    Возвращает (primary_series_frames, filter_report), а все найденные серии доступны в filter_report["raw_series"].
    """
    cfg = filter_cfg or {}
    exclude_panos = bool(cfg.get("exclude_stitched_panoramas", True))
    require_gimbal = bool(cfg.get("require_gimbal_xmp", True))
    dup_angle_thresh = float(cfg.get("duplicate_angle_threshold_deg", 3.0))
    min_raw = int(cfg.get("min_raw_frames", 5))

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

    # Ступень 3: Разделение на независимые серии съемки и удаление дубликатов внутри каждой серии
    partitioned_series = split_raw_frames_into_series(stage2_candidates, filter_cfg=cfg)
    raw_series: list[list[dict[str, Any]]] = []
    series_summaries: list[dict[str, Any]] = []

    for s_frames in partitioned_series:
        deduped, s_dups = _deduplicate_single_series(s_frames, dup_angle_thresh=dup_angle_thresh)
        rejected_duplicates.extend(s_dups)
        if len(deduped) >= min_raw or len(partitioned_series) == 1:
            s_idx = len(raw_series) + 1
            raw_series.append(deduped)
            series_summaries.append(_summarize_series(deduped, s_idx, s_dups))
        else:
            rejected_duplicates.extend(f.get("filename", "") for f in deduped)

    kept_frames = raw_series[0] if raw_series else []

    report = {
        "total_files": len(frames),
        "kept_raw_count": len(kept_frames),
        "total_kept_across_series": sum(len(s) for s in raw_series),
        "num_series": len(raw_series),
        "raw_series": raw_series,
        "series_summaries": series_summaries,
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
      Если внутри папки обнаружено несколько независимых серий (например, Pano(1)1..25 и Pano(2)1..25),
      все они возвращаются в списке 'raw_series' для раздельной сборки каждой панорамы.
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

    if len(entries) > 4:
        with ThreadPoolExecutor(max_workers=min(8, len(entries))) as ex:
            all_frames = list(
                ex.map(
                    lambda fname: extract_frame_metadata(
                        os.path.join(input_dir, fname), default_hfov=default_hfov
                    ),
                    entries,
                )
            )
    else:
        all_frames = [
            extract_frame_metadata(os.path.join(input_dir, fname), default_hfov=default_hfov)
            for fname in entries
        ]
    raw_frames, report = filter_series_frames(all_frames, filter_cfg=filter_cfg)
    min_raw = int((filter_cfg or {}).get("min_raw_frames", 5))
    raw_series = [s for s in report.get("raw_series", []) if len(s) >= min_raw]
    base_pano = select_prestitched_base_panorama(report["stitched_panoramas"])

    if raw_series:
        mode = "stitch"
    elif base_pano is not None:
        mode = "sky_only"
    else:
        mode = "empty"

    return {
        "mode": mode,
        "raw_frames": raw_series[0] if raw_series else raw_frames,
        "raw_series": raw_series,
        "series_summaries": report.get("series_summaries", []),
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
    Корректно учитывает смещение часового пояса в XMP (xmp:CreateDate), когда пульт БПЛА
    настроен на другой часовой пояс (например, +08:00 или -07:00).
    """
    ref = next(
        (f for f in frames_meta if f.get("gps_lat") is not None and f.get("gps_lon") is not None),
        frames_meta[0],
    )
    lat = ref.get("gps_lat")
    lon = ref.get("gps_lon")
    alt = ref.get("gps_alt") or 0.0
    dt_str = ref.get("datetime_str")
    dt_iso_aware = ref.get("datetime_iso_aware") or (ref.get("xmp") or {}).get("xmp:CreateDate")

    tz_name = "UTC"
    if lat is not None and lon is not None:
        found_tz = _get_tz_finder().timezone_at(lng=float(lon), lat=float(lat))
        if found_tz:
            tz_name = found_tz

    local_tz = ZoneInfo(tz_name)
    dt_naive_local: datetime | None = None
    if dt_str:
        try:
            dt_naive = datetime.strptime(dt_str[:19], "%Y:%m:%d %H:%M:%S")
        except ValueError:
            dt_naive = datetime.fromisoformat(dt_str[:19])
        dt_naive_local = dt_naive.replace(tzinfo=local_tz)

    dt_from_iso_local: datetime | None = None
    if dt_iso_aware:
        try:
            parsed_iso = datetime.fromisoformat(str(dt_iso_aware).strip())
            if parsed_iso.tzinfo is not None:
                dt_from_iso_local = parsed_iso.astimezone(local_tz)
        except Exception:
            dt_from_iso_local = None

    if lat is not None and lon is not None:
        observer = Observer(latitude=float(lat), longitude=float(lon), elevation=max(float(alt), 0.0))
        if dt_from_iso_local is not None and dt_naive_local is not None:
            el_iso = float(sun_elevation(observer, dateandtime=dt_from_iso_local.astimezone(timezone.utc)))
            el_naive = float(sun_elevation(observer, dateandtime=dt_naive_local.astimezone(timezone.utc)))
            # Если при точном учете часового пояса из XMP солнце над горизонтом (или выше, чем при наивном времени),
            # используем точный перевод из часового пояса пульта (исправляет пульты с +08:00 / -07:00)
            if el_iso >= 0.0 or el_iso >= el_naive:
                dt_local = dt_from_iso_local
            else:
                dt_local = dt_naive_local
        elif dt_from_iso_local is not None:
            dt_local = dt_from_iso_local
        elif dt_naive_local is not None:
            dt_local = dt_naive_local
        else:
            dt_local = datetime.now(tz=local_tz)

        dt_utc = dt_local.astimezone(timezone.utc)
        az_deg = float(sun_azimuth(observer, dateandtime=dt_utc)) % 360.0
        el_deg = float(sun_elevation(observer, dateandtime=dt_utc))
    else:
        dt_local = dt_from_iso_local or dt_naive_local or datetime.now(tz=local_tz)
        dt_utc = dt_local.astimezone(timezone.utc)
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


def inject_xmp_into_jpeg_bytes(data: bytes, xmp_packet: bytes) -> bytes:
    """Встраивает APP1-сегмент с XMP-пакетом в байтовый буфер JPEG в памяти без перекодирования изображения."""
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return data

    xmp_header = b"http://ns.adobe.com/xap/1.0/\x00"
    payload = xmp_header + xmp_packet
    seg_len = len(payload) + 2
    app1_xmp = b"\xff\xe1" + seg_len.to_bytes(2, "big") + payload

    pos = 2
    insert_pos = 2
    cleaned_chunks = [data[:2]]
    while pos + 4 <= len(data) and data[pos] == 0xFF:
        marker = data[pos + 1]
        if marker in (0xDA, 0xD9):
            break
        if marker in (0xD8, 0x01) or (0xD0 <= marker <= 0xD7):
            cleaned_chunks.append(data[pos : pos + 2])
            pos += 2
            continue
        length = int.from_bytes(data[pos + 2 : pos + 4], "big")
        seg_end = pos + 2 + length
        seg_bytes = data[pos:seg_end]
        if marker == 0xE1 and data[pos + 4 : pos + 4 + len(xmp_header)] == xmp_header:
            pos = seg_end
            continue
        cleaned_chunks.append(seg_bytes)
        if marker in (0xE0, 0xE1):
            insert_pos = sum(len(c) for c in cleaned_chunks)
        pos = seg_end

    cleaned_chunks.append(data[pos:])
    rebuilt = b"".join(cleaned_chunks)
    return rebuilt[:insert_pos] + app1_xmp + rebuilt[insert_pos:]


def _inject_xmp_into_jpeg_file(jpeg_path: str, xmp_packet: bytes) -> None:
    """Встраивает APP1-сегмент с XMP-пакетом в JPEG без перекодирования изображения."""
    with open(jpeg_path, "rb") as f:
        data = f.read()

    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return

    final_bytes = inject_xmp_into_jpeg_bytes(data, xmp_packet)
    with open(jpeg_path, "wb") as f:
        f.write(final_bytes)


def get_reference_exif_bytes(reference_jpg_path: str, full_width: int = 15000, full_height: int = 7500) -> bytes | None:
    """Извлекает EXIF (включая GPSInfo и ExifIFD) из опорного снимка серии для сохранения в выходной JPEG."""
    try:
        with Image.open(reference_jpg_path) as ref_img:
            exif = ref_img.getexif()
            if not exif:
                return None
            exif[0x0112] = 1
            exif[0x0100] = full_width
            exif[0x0101] = full_height
            try:
                exif_ifd = exif.get_ifd(0x8769)
                if exif_ifd is not None:
                    exif_ifd[0xA002] = full_width
                    exif_ifd[0xA003] = full_height
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
    Если EXIF и GPano XMP уже были встроены в памяти на этапе save_panorama_jpeg,
    пропускает повторную перезапись 65 МБ файла и внешний вызов exiftool.
    """
    try:
        with open(output_jpg_path, "rb") as f:
            head = f.read(128 * 1024)
        if b"GPano:UsePanoramaViewer" in head and b"Exif\x00\x00" in head:
            return
    except Exception:
        pass

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
            pass
