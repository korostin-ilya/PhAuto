"""Формирование паспорта неба из 20 вертикальных срезов, определение высоты солнца и очистка от объектов."""

from __future__ import annotations

import os
import re
from typing import Any

import cv2
import numpy as np


def read_image_bgr(path: str, max_width: int | None = None) -> np.ndarray:
    """
    Читает изображение с поддержкой кириллических путей на Windows и Linux.
    Если указан max_width <= 2000 (например, для микро-блендинга 600x150), декодирует JPEG
    с аппаратным DCT-уменьшением IMREAD_REDUCED_COLOR_8 в 10-15 раз быстрее полного 15000x3750.
    """
    buf = np.fromfile(path, dtype=np.uint8)
    reduced_flag = getattr(cv2, "IMREAD_REDUCED_COLOR_8", cv2.IMREAD_COLOR)
    flag = reduced_flag if (max_width is not None and max_width <= 2000) else cv2.IMREAD_COLOR
    img = cv2.imdecode(buf, flag)
    if img is None and flag != cv2.IMREAD_COLOR:
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Не удалось декодировать изображение: {path}")
    return img


def bgr_to_cie_lab(bgr: np.ndarray) -> np.ndarray:
    """
    Переводит BGR uint8 изображение в массив float32 CIE LAB:
    L* in [0, 100], a* in [-128, 127], b* in [-128, 127].
    """
    lab_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab = np.empty_like(lab_u8, dtype=np.float32)
    lab[..., 0] = lab_u8[..., 0] * (100.0 / 255.0)
    lab[..., 1] = lab_u8[..., 1] - 128.0
    lab[..., 2] = lab_u8[..., 2] - 128.0
    return lab


def cie_lab_to_bgr(lab: np.ndarray) -> np.ndarray:
    """Переводит массив float32 CIE LAB (L* in [0,100], a*,b* in [-128,127]) обратно в BGR uint8."""
    lab_u8 = np.empty_like(lab, dtype=np.uint8)
    lab_u8[..., 0] = np.clip(np.round(lab[..., 0] * (255.0 / 100.0)), 0, 255).astype(np.uint8)
    lab_u8[..., 1] = np.clip(np.round(lab[..., 1] + 128.0), 0, 255).astype(np.uint8)
    lab_u8[..., 2] = np.clip(np.round(lab[..., 2] + 128.0), 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab_u8, cv2.COLOR_LAB2BGR)


def parse_manual_sun_y_from_filename(filename: str) -> float | None:
    """Ищет ручное переопределение высоты солнца в имени файла, например sky_01_sunY2450.jpg."""
    m = re.search(r"sunY(\d+(?:\.\d+)?)", os.path.basename(filename), flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def detect_template_sun_y(
    sky_bgr: np.ndarray,
    config: dict[str, Any] | None = None,
    manual_sun_y: float | None = None,
    template_path: str = "",
) -> dict[str, Any]:
    """
    Шаг 2.1.1: Автоматическое определение высоты солнца Y_sun_sky на шаблоне (15000x3750).
    - Вырезает центральный столбец X in [7300, 7700], Y in [0, 3750].
    - Вычисляет сглаженный вертикальный профиль яркости L(y) и синевы b(y) (sigma = 50 px).
    - Сравнивает с яркостью на краях шаблона (X = 0, анти-солнце).
    - Если направленная асимметрия мала (Delta L < 8.0) или нет локализованного пика яркости,
      шаблон помечается как пасмурный (is_overcast_template = True, sun_y = None).
    """
    cfg = (config or {}).get("sky_index", {})
    half_w = int(cfg.get("sun_strip_half_width", 200))
    edge_w = int(cfg.get("edge_strip_width", 200))
    sigma_px = float(cfg.get("gaussian_sigma_px", 50.0))
    delta_l_thresh = float(cfg.get("overcast_delta_l_threshold", 8.0))
    peak_l_thresh = float(cfg.get("overcast_peak_l_threshold", 94.0))
    top_pct = float(cfg.get("sun_top_percentile", 95.0))

    h, w = sky_bgr.shape[:2]
    cx = w // 2

    center_strip = sky_bgr[:, max(0, cx - half_w) : min(w, cx + half_w)]
    edge_strip = np.concatenate([sky_bgr[:, :edge_w], sky_bgr[:, w - edge_w :]], axis=1)

    lab_c = bgr_to_cie_lab(center_strip)
    lab_e = bgr_to_cie_lab(edge_strip)

    l_c_profile = lab_c[:, :, 0].mean(axis=1)
    b_c_profile = lab_c[:, :, 2].mean(axis=1)
    l_e_profile = lab_e[:, :, 0].mean(axis=1)

    l_c_smooth = cv2.GaussianBlur(
        l_c_profile.reshape(-1, 1), (0, 0), sigmaX=sigma_px, sigmaY=sigma_px
    ).ravel()
    b_c_smooth = cv2.GaussianBlur(
        b_c_profile.reshape(-1, 1), (0, 0), sigmaX=sigma_px, sigmaY=sigma_px
    ).ravel()
    l_e_smooth = cv2.GaussianBlur(
        l_e_profile.reshape(-1, 1), (0, 0), sigmaX=sigma_px, sigmaY=sigma_px
    ).ravel()

    delta_l = float(np.max(l_c_smooth - l_e_smooth))
    peak_l = float(np.max(l_c_smooth))

    # Проверяем ручное переопределение из имени файла или аргумента
    fn_sun_y = parse_manual_sun_y_from_filename(template_path) if template_path else None
    override_y = manual_sun_y if manual_sun_y is not None else fn_sun_y

    if override_y is not None:
        return {
            "is_overcast_template": False,
            "sun_y": float(override_y),
            "manual_sun_y": float(override_y),
            "delta_l_asymmetry": delta_l,
            "peak_l_center": peak_l,
            "mean_b_center": float(np.mean(b_c_smooth)),
        }

    is_overcast = (delta_l < delta_l_thresh) or (
        delta_l < (delta_l_thresh * 1.5) and peak_l < peak_l_thresh
    )
    if is_overcast:
        return {
            "is_overcast_template": True,
            "sun_y": None,
            "manual_sun_y": None,
            "delta_l_asymmetry": delta_l,
            "peak_l_center": peak_l,
            "mean_b_center": float(np.mean(b_c_smooth)),
        }

    # Центр тяжести направленного солнечного пика (учитывает превышение яркости центра над анти-солнцем,
    # чтобы яркая атмосферная дымка у горизонта не смещала оценку высоты солнца вниз)
    diff_le = np.maximum(0.0, l_c_smooth - l_e_smooth)
    prominence = diff_le * np.square(np.clip(l_c_smooth / 100.0, 0.0, 1.0))
    if float(np.max(prominence)) > 1e-3:
        p_pct = min(top_pct, 93.0)
        thresh_val = float(np.percentile(prominence, p_pct))
        top_indices = np.where(prominence >= thresh_val)[0]
        weights = (prominence[top_indices] - thresh_val) + 1e-3
        sun_y = float(np.sum(top_indices * weights) / np.sum(weights))
    else:
        thresh_val = float(np.percentile(l_c_smooth, top_pct))
        top_indices = np.where(l_c_smooth >= thresh_val)[0]
        weights = (l_c_smooth[top_indices] - thresh_val) + 1e-3
        sun_y = float(np.sum(top_indices * weights) / np.sum(weights))

    return {
        "is_overcast_template": False,
        "sun_y": round(sun_y, 2),
        "manual_sun_y": None,
        "delta_l_asymmetry": round(delta_l, 3),
        "peak_l_center": round(peak_l, 3),
        "mean_b_center": round(float(np.mean(b_c_smooth)), 3),
    }


def extract_clean_pano_sky_mask_in_band(
    band_lab: np.ndarray,
    band_alpha: np.ndarray | None = None,
    num_slices: int = 20,
    sobel_percentile_cutoff: float = 75.0,
    l_min_percentile_cutoff: float = 45.0,
) -> np.ndarray:
    """
    Шаг 2.3.1: Очистка срезов панорамы от высотных объектов (зданий, кранов, ЛЭП) в полосе Y in [2850, 3650].
    Отбрасываются:
    - пустые пиксели (alpha_stitched == 0);
    - пиксели с высоким градиентом Собеля (провода, фермы кранов, края зданий);
    - темные пиксели ниже 45-го перцентиля яркости L* внутри каждого среза.
    """
    h_band, w_band = band_lab.shape[:2]
    l_chan = band_lab[:, :, 0]
    a_chan = band_lab[:, :, 1]
    b_chan = band_lab[:, :, 2]

    if band_alpha is not None:
        valid_alpha = band_alpha > 128
    else:
        valid_alpha = np.ones((h_band, w_band), dtype=bool)

    # Исключаем явно «земные» пиксели (теплая почва, кирпич, освещенная листва, фиолетовая дымка)
    sky_chroma_ok = (a_chan <= 4.0) & ((b_chan <= 8.0) | (l_chan >= 82.0))

    # Градиент Собеля на канале яркости L*
    gx = cv2.Sobel(l_chan, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(l_chan, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)

    clean_mask = np.zeros((h_band, w_band), dtype=bool)
    slice_w = w_band // num_slices

    for s_idx in range(num_slices):
        x0 = s_idx * slice_w
        x1 = w_band if s_idx == num_slices - 1 else (s_idx + 1) * slice_w

        sub_valid = valid_alpha[:, x0:x1] & sky_chroma_ok[:, x0:x1]
        if np.count_nonzero(sub_valid) < 50:
            sub_valid = valid_alpha[:, x0:x1]
        if not np.any(sub_valid):
            continue

        sub_l = l_chan[:, x0:x1]
        sub_g = grad_mag[:, x0:x1]

        valid_l_vals = sub_l[sub_valid]
        valid_g_vals = sub_g[sub_valid]

        l_thresh = float(np.percentile(valid_l_vals, l_min_percentile_cutoff))
        g_thresh = float(np.percentile(valid_g_vals, sobel_percentile_cutoff))

        sub_clean = sub_valid & (sub_l >= l_thresh) & (sub_g <= max(g_thresh, 1.5))
        # Если после строгой фильтрации осталось слишком мало пикселей, берем верхнюю половину по яркости
        if np.count_nonzero(sub_clean) < 100:
            sub_clean = sub_valid & (sub_l >= l_thresh)

        clean_mask[:, x0:x1] = sub_clean

    return clean_mask


def compute_sky_signature(
    sky_or_pano_bgr: np.ndarray,
    config: dict[str, Any] | None = None,
    valid_mask_band: np.ndarray | None = None,
    is_already_band: bool = False,
) -> list[dict[str, Any]]:
    """
    Шаг 2.1.2: Формирование паспорта из N = 20 вертикальных срезов в рабочей полосе Y in [2850, 3650].
    В каждом из 20 столбцов сохраняются:
    - K = 6 опорных точек вертикального градиента (L*, a*, b*) снизу вверх;
    - 16-биновая гистограмма яркости L* и канала b* (для метрики Вассерштейна EMD);
    - Частотная энергия облаков (Difference of Gaussians на радиусах 15 px и 60 px).
    """
    cfg = (config or {}).get("sky_index", {})
    num_slices = int(cfg.get("num_slices", 20))
    y_min = int(cfg.get("band_y_min", 2850))
    y_max = int(cfg.get("band_y_max", 3650))
    k_points = int(cfg.get("num_grad_points", 6))
    hist_bins = int(cfg.get("hist_bins", 16))
    sigma1 = float(cfg.get("dog_sigma1", 15.0))
    sigma2 = float(cfg.get("dog_sigma2", 60.0))

    if is_already_band:
        band_bgr = sky_or_pano_bgr
    else:
        h_full = sky_or_pano_bgr.shape[0]
        y0 = max(0, min(h_full - 10, y_min))
        y1 = max(y0 + 10, min(h_full, y_max))
        band_bgr = sky_or_pano_bgr[y0:y1, :]

    band_lab = bgr_to_cie_lab(band_bgr)
    h_band, w_band = band_lab.shape[:2]
    l_chan = band_lab[:, :, 0]

    # Difference of Gaussians (DoG) на радиусах sigma1=15 px и sigma2=60 px
    # Для низкочастотных радиусов 15 и 60 px вычисление на масштабе 1/2 с циклическим паддингом по X
    # математически эквивалентно и работает в 5 раз быстрее
    scale = 0.5
    small_w = max(64, int(round(w_band * scale)))
    small_h = max(16, int(round(h_band * scale)))
    l_small = cv2.resize(l_chan, (small_w, small_h), interpolation=cv2.INTER_AREA)
    pad_x = int(round(sigma2 * scale * 2))
    l_padded = np.pad(l_small, ((0, 0), (pad_x, pad_x)), mode="wrap")
    g1_pad = cv2.GaussianBlur(l_padded, (0, 0), sigmaX=sigma1 * scale, sigmaY=sigma1 * scale)
    g2_pad = cv2.GaussianBlur(l_padded, (0, 0), sigmaX=sigma2 * scale, sigmaY=sigma2 * scale)
    dog_small = np.abs(g1_pad[:, pad_x : pad_x + small_w] - g2_pad[:, pad_x : pad_x + small_w])
    dog_map = cv2.resize(dog_small, (w_band, h_band), interpolation=cv2.INTER_LINEAR)

    slice_w = w_band // num_slices
    l_bin_edges = np.linspace(0.0, 100.0, hist_bins + 1)
    b_bin_edges = np.linspace(-60.0, 60.0, hist_bins + 1)

    slices_signature: list[dict[str, Any]] = []

    for s_idx in range(num_slices):
        x0 = s_idx * slice_w
        x1 = w_band if s_idx == num_slices - 1 else (s_idx + 1) * slice_w

        sub_lab = band_lab[:, x0:x1]
        sub_dog = dog_map[:, x0:x1]
        sub_mask = (
            valid_mask_band[:, x0:x1]
            if valid_mask_band is not None
            else np.ones((h_band, x1 - x0), dtype=bool)
        )

        valid_count = int(np.count_nonzero(sub_mask))
        coverage_ratio = float(valid_count) / float(sub_mask.size)

        # K = 6 опорных точек вертикального градиента (L*, a*, b*) снизу вверх (от горизонта к зениту)
        grad_points: list[list[float]] = []
        row_edges = np.linspace(0, h_band, k_points + 1, dtype=int)
        # Идем снизу вверх: от [row_edges[k-1]:row_edges[k]] к [row_edges[0]:row_edges[1]]
        for k_idx in range(k_points - 1, -1, -1):
            r0, r1 = row_edges[k_idx], row_edges[k_idx + 1]
            blk_lab = sub_lab[r0:r1]
            blk_mask = sub_mask[r0:r1]
            if np.any(blk_mask):
                pts = blk_lab[blk_mask]
                mean_lab = np.median(pts, axis=0)
            elif np.any(sub_mask):
                mean_lab = np.median(sub_lab[sub_mask], axis=0)
            else:
                mean_lab = np.array([75.0, 0.0, -15.0], dtype=np.float32)
            grad_points.append([round(float(v), 3) for v in mean_lab])

        if np.any(sub_mask):
            valid_l = sub_lab[:, :, 0][sub_mask]
            valid_b = np.clip(sub_lab[:, :, 2][sub_mask], -60.0, 60.0)
            valid_dog = sub_dog[sub_mask]

            hist_l, _ = np.histogram(valid_l, bins=l_bin_edges, density=False)
            hist_b, _ = np.histogram(valid_b, bins=b_bin_edges, density=False)

            hist_l_norm = (hist_l / max(1, hist_l.sum())).tolist()
            hist_b_norm = (hist_b / max(1, hist_b.sum())).tolist()
            dog_energy = float(np.sqrt(np.mean(np.square(valid_dog))))
        else:
            hist_l_norm = [1.0 / hist_bins] * hist_bins
            hist_b_norm = [1.0 / hist_bins] * hist_bins
            dog_energy = 0.0

        slices_signature.append(
            {
                "slice_index": s_idx,
                "coverage": round(coverage_ratio, 4),
                "grad_lab": grad_points,
                "hist_l": [round(float(x), 5) for x in hist_l_norm],
                "hist_b": [round(float(x), 5) for x in hist_b_norm],
                "dog_energy": round(dog_energy, 4),
            }
        )

    return slices_signature
