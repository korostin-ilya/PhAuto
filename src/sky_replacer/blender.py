"""Умный градиентный блендинг неба с защитой высотных объектов (ЛЭП, краны, здания), бесшовным циклом 360° и защитой единства солнечного диска."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

from src.sky_replacer.signature import (
    bgr_to_cie_lab,
    cie_lab_to_bgr,
    extract_clean_pano_sky_mask_in_band,
    read_image_bgr,
)


def _resize_azimuth_circular(
    grid: np.ndarray,
    target_w: int,
    target_h: int = 1,
) -> np.ndarray:
    """
    Выполняет строго периодическую (360° wrap-around) горизонтальную интерполяцию сетки секторов
    размера (H_grid, N_sectors, C) к ширине target_w (15000) и высоте target_h.
    Гарантирует отсутствие скачка яркости/цвета на границе X=0 <-> X=14999.
    """
    h_grid, n_sec, channels = grid.shape
    x_idx = np.arange(target_w, dtype=np.float32)
    u = ((x_idx + 0.5) / float(target_w)) * float(n_sec) - 0.5
    u_floor = np.floor(u)
    alpha = (u - u_floor).reshape(1, target_w, 1)
    i0 = (u_floor.astype(np.int32)) % n_sec
    i1 = (i0 + 1) % n_sec

    horiz = (1.0 - alpha) * grid[:, i0, :] + alpha * grid[:, i1, :]
    if target_h == h_grid:
        return horiz.astype(np.float32)
    return cv2.resize(horiz.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def _guided_filter_color(
    guide_bgr: np.ndarray,
    src_mask: np.ndarray,
    radius: int = 15,
    eps: float = 1e-3,
    subsample: int = 4,
) -> np.ndarray:
    """
    Уточняет грубую маску неба M_raw по цветному изображению-гиду с помощью Fast Guided Filter (He et al., 2015)
    с циклическим (wrap-around) горизонтальным дополнением на границе 360° (X=0 <-> X=W-1).
    """
    h, w = src_mask.shape[:2]
    guide_f = guide_bgr.astype(np.float32) * (1.0 / 255.0)
    src_f = src_mask.astype(np.float32)

    s = max(1, int(subsample))
    if s > 1 and h > s * 8 and w > s * 8:
        sh = max(8, h // s)
        sw = max(8, w // s)
        guide_sub = cv2.resize(guide_f, (sw, sh), interpolation=cv2.INTER_LINEAR)
        src_sub = cv2.resize(src_f, (sw, sh), interpolation=cv2.INTER_LINEAR)
        r_sub = max(1, int(round(radius / float(s))))
    else:
        guide_sub = guide_f
        src_sub = src_f
        sw = w
        sh = h
        r_sub = max(1, int(radius))

    pad_x = min(sw // 2, r_sub * 2 + 2)
    guide_pad = np.pad(guide_sub, ((0, 0), (pad_x, pad_x), (0, 0)), mode="wrap")
    src_pad = np.pad(src_sub, ((0, 0), (pad_x, pad_x)), mode="wrap")

    ksize = (2 * r_sub + 1, 2 * r_sub + 1)

    def box(img: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(
            img, ddepth=cv2.CV_32F, ksize=ksize, normalize=True, borderType=cv2.BORDER_REFLECT
        )

    I_b, I_g, I_r = guide_pad[:, :, 0], guide_pad[:, :, 1], guide_pad[:, :, 2]
    mean_b, mean_g, mean_r = box(I_b), box(I_g), box(I_r)
    mean_p = box(src_pad)

    cov_Ip_b = box(I_b * src_pad) - mean_b * mean_p
    cov_Ip_g = box(I_g * src_pad) - mean_g * mean_p
    cov_Ip_r = box(I_r * src_pad) - mean_r * mean_p

    var_bb = box(I_b * I_b) - mean_b * mean_b + eps
    var_bg = box(I_b * I_g) - mean_b * mean_g
    var_br = box(I_b * I_r) - mean_b * mean_r
    var_gg = box(I_g * I_g) - mean_g * mean_g + eps
    var_gr = box(I_g * I_r) - mean_g * mean_r
    var_rr = box(I_r * I_r) - mean_r * mean_r + eps

    c00 = var_gg * var_rr - var_gr * var_gr
    c01 = var_br * var_gr - var_bg * var_rr
    c02 = var_bg * var_gr - var_br * var_gg
    c11 = var_bb * var_rr - var_br * var_br
    c12 = var_bg * var_br - var_bb * var_gr
    c22 = var_bb * var_gg - var_bg * var_bg

    det = var_bb * c00 + var_bg * c01 + var_br * c02
    inv_det = 1.0 / np.maximum(det, 1e-9)

    a_b = (c00 * cov_Ip_b + c01 * cov_Ip_g + c02 * cov_Ip_r) * inv_det
    a_g = (c01 * cov_Ip_b + c11 * cov_Ip_g + c12 * cov_Ip_r) * inv_det
    a_r = (c02 * cov_Ip_b + c12 * cov_Ip_g + c22 * cov_Ip_r) * inv_det

    b_coeff = mean_p - a_b * mean_b - a_g * mean_g - a_r * mean_r

    mean_a_b = box(a_b)[:, pad_x : pad_x + sw]
    mean_a_g = box(a_g)[:, pad_x : pad_x + sw]
    mean_a_r = box(a_r)[:, pad_x : pad_x + sw]
    mean_b_coeff = box(b_coeff)[:, pad_x : pad_x + sw]

    if (sh, sw) != (h, w):
        mean_a_b = cv2.resize(mean_a_b, (w, h), interpolation=cv2.INTER_LINEAR)
        mean_a_g = cv2.resize(mean_a_g, (w, h), interpolation=cv2.INTER_LINEAR)
        mean_a_r = cv2.resize(mean_a_r, (w, h), interpolation=cv2.INTER_LINEAR)
        mean_b_coeff = cv2.resize(mean_b_coeff, (w, h), interpolation=cv2.INTER_LINEAR)

    q = (
        mean_a_b * guide_f[:, :, 0]
        + mean_a_g * guide_f[:, :, 1]
        + mean_a_r * guide_f[:, :, 2]
        + mean_b_coeff
    )
    return np.clip(q, 0.0, 1.0)


def compute_top_edge_profile(
    alpha_upper: np.ndarray,
    horizon_y: int = 3750,
    smooth_sigma_px: float = 80.0,
    pano_upper_bgr: np.ndarray | None = None,
) -> np.ndarray:
    """
    Определяет верхнюю границу реального покрытия кадров дрона Y_top_edge(x) в каждом столбце x.
    Исключает темные/черные краевые пиксели на границе неснятого зенита и отступает вниз на 35 пикселей.
    Возвращает сглаженный одномерный массив длины W (15000).
    """
    h, w = alpha_upper.shape[:2]
    valid_mask = alpha_upper > 220
    if pano_upper_bgr is not None and pano_upper_bgr.shape[:2] == (h, w):
        valid_mask = valid_mask & (
            (pano_upper_bgr[:, :, 0] > 20)
            | (pano_upper_bgr[:, :, 1] > 20)
            | (pano_upper_bgr[:, :, 2] > 20)
        )
    has_valid = np.any(valid_mask, axis=0)
    first_valid_y = np.argmax(valid_mask, axis=0).astype(np.float32)
    first_valid_y = np.where(has_valid, first_valid_y, float(h - 1))

    # Запас вниз на 35 пикселей от кромки черных пикселей, чтобы исключить краевое виньетирование и алиасинг
    first_valid_y = np.minimum(float(h - 1), first_valid_y + 35.0)

    if smooth_sigma_px > 1.0:
        pad_w = int(min(w // 2, round(smooth_sigma_px * 3)))
        padded_edge = np.pad(first_valid_y, (pad_w, pad_w), mode="wrap").reshape(1, -1)
        smoothed_pad = cv2.GaussianBlur(
            padded_edge,
            (0, 0),
            sigmaX=smooth_sigma_px,
            sigmaY=1.0,
        ).ravel()
        smoothed = smoothed_pad[pad_w : pad_w + w]
        top_edge = np.maximum(first_valid_y, smoothed)
    else:
        top_edge = first_valid_y

    return np.clip(top_edge, 0.0, float(horizon_y - 2))


def _estimate_2d_sky_color_model(
    roi_lab: np.ndarray,
    roi_alpha: np.ndarray,
    num_sectors: int = 20,
    num_y_bands: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Строит двумерную (по азимуту X и высоте Y) цветовую модель «родного неба» панорамы
    со строго периодической (360° wrap-around) интерполяцией по азимуту.
    """
    h_roi, w_roi = roi_lab.shape[:2]
    sub_h = max(num_y_bands * 16, h_roi // 4)
    sub_w = max(num_sectors * 16, w_roi // 4)
    lab_sub = cv2.resize(roi_lab, (sub_w, sub_h), interpolation=cv2.INTER_AREA)
    alpha_sub = cv2.resize(roi_alpha, (sub_w, sub_h), interpolation=cv2.INTER_NEAREST)

    grid_means = np.zeros((num_y_bands, num_sectors, 3), dtype=np.float32)
    grid_stds = np.zeros((num_y_bands, num_sectors, 3), dtype=np.float32)
    grid_valid = np.zeros((num_y_bands, num_sectors), dtype=bool)

    slice_w = sub_w // num_sectors
    y_edges = np.linspace(0, sub_h, num_y_bands + 1, dtype=int)

    for b_idx in range(num_y_bands):
        y0, y1 = y_edges[b_idx], y_edges[b_idx + 1]
        if y1 <= y0:
            continue
        band_clean = extract_clean_pano_sky_mask_in_band(
            band_lab=lab_sub[y0:y1, :],
            band_alpha=alpha_sub[y0:y1, :],
            num_slices=num_sectors,
            sobel_percentile_cutoff=70.0,
            l_min_percentile_cutoff=35.0 if b_idx < 3 else 45.0,
        )
        for s_idx in range(num_sectors):
            x0 = s_idx * slice_w
            x1 = sub_w if s_idx == num_sectors - 1 else (s_idx + 1) * slice_w
            m = band_clean[:, x0:x1]
            if np.count_nonzero(m) >= 10:
                pts = lab_sub[y0:y1, x0:x1][m]
                grid_means[b_idx, s_idx] = np.median(pts, axis=0)
                mad = np.median(np.abs(pts - grid_means[b_idx, s_idx]), axis=0) * 1.4826
                grid_stds[b_idx, s_idx] = np.clip(
                    mad,
                    np.array([6.0, 3.2, 5.0], dtype=np.float32),
                    np.array([22.0, 15.0, 20.0], dtype=np.float32),
                )
                grid_valid[b_idx, s_idx] = True

    if not np.any(grid_valid):
        all_valid = alpha_sub > 128
        pts = lab_sub[all_valid] if np.any(all_valid) else lab_sub.reshape(-1, 3)
        grid_means[:] = np.median(pts, axis=0)
        grid_stds[:] = np.array([8.0, 4.0, 6.0], dtype=np.float32)
    else:
        for b_idx in range(num_y_bands):
            row_valid = np.where(grid_valid[b_idx])[0]
            if len(row_valid) > 0:
                for s_idx in range(num_sectors):
                    if not grid_valid[b_idx, s_idx]:
                        dists = np.minimum(
                            (s_idx - row_valid) % num_sectors,
                            (row_valid - s_idx) % num_sectors,
                        )
                        nearest_s = row_valid[int(np.argmin(dists))]
                        grid_means[b_idx, s_idx] = grid_means[b_idx, nearest_s]
                        grid_stds[b_idx, s_idx] = grid_stds[b_idx, nearest_s]
                        grid_valid[b_idx, s_idx] = True
        valid_rows = np.where(np.any(grid_valid, axis=1))[0]
        for b_idx in range(num_y_bands):
            if not np.any(grid_valid[b_idx]):
                nearest_r = valid_rows[int(np.argmin(np.abs(valid_rows - b_idx)))]
                grid_means[b_idx] = grid_means[nearest_r]
                grid_stds[b_idx] = grid_stds[nearest_r]

    # Широкое циклическое сглаживание по горизонтали (sigmaX=2.2 сектора ~ 40°),
    # чтобы сгладить скачки автоэкспозиции между первым и последним кадром серии
    pad = 6
    m_pad = np.pad(grid_means, ((0, 0), (pad, pad), (0, 0)), mode="wrap")
    s_pad = np.pad(grid_stds, ((0, 0), (pad, pad), (0, 0)), mode="wrap")
    m_smooth = cv2.GaussianBlur(m_pad, (9, 3), sigmaX=2.2, sigmaY=0.8)[:, pad : pad + num_sectors, :]
    s_smooth = cv2.GaussianBlur(s_pad, (9, 3), sigmaX=2.2, sigmaY=0.8)[:, pad : pad + num_sectors, :]

    hor_mean_sec = np.mean(m_smooth[-2:, :, :], axis=0, keepdims=True)
    hor_std_sec = np.mean(s_smooth[-2:, :, :], axis=0, keepdims=True)
    hm_pad = np.pad(hor_mean_sec, ((0, 0), (pad, pad), (0, 0)), mode="wrap")
    hs_pad = np.pad(hor_std_sec, ((0, 0), (pad, pad), (0, 0)), mode="wrap")
    hm_smooth = cv2.GaussianBlur(hm_pad, (9, 1), sigmaX=2.2)[:, pad : pad + num_sectors, :]
    hs_smooth = cv2.GaussianBlur(hs_pad, (9, 1), sigmaX=2.2)[:, pad : pad + num_sectors, :]

    horizon_mean_x = _resize_azimuth_circular(hm_smooth, target_w=w_roi, target_h=1)
    horizon_std_x = _resize_azimuth_circular(hs_smooth, target_w=w_roi, target_h=1)

    return m_smooth, s_smooth, horizon_mean_x, horizon_std_x


def build_tall_object_protection_mask(
    pano_bgr_roi: np.ndarray,
    pano_lab_roi: np.ndarray,
    pano_alpha_roi: np.ndarray,
    top_edge_rel: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Шаг 2.4.2: Строит маску принадлежности к небу M_sky(x, y) in [0, 1] в зоне перехода [roi_y_start, 3750],
    защищающую высотные здания, башенные краны, деревья и провода ЛЭП от закрашивания новым небом.
    Возвращает (m_sky, horizon_mean_x, horizon_std_x).
    """
    blend_cfg = config.get("sky_blender", {})
    num_sectors = int(blend_cfg.get("num_azimuth_sectors", 20))
    mahal_sky_t = float(blend_cfg.get("mahalanobis_sky_threshold", 3.2))
    mahal_obj_t = float(blend_cfg.get("mahalanobis_obj_threshold", 6.5))
    edge_weight = float(blend_cfg.get("edge_gradient_weight", 0.45))
    gf_radius = int(blend_cfg.get("guided_filter_radius", 15))
    gf_eps = float(blend_cfg.get("guided_filter_eps", 0.001))
    gf_subsample = int(blend_cfg.get("guided_filter_subsample", 4))
    cutoff_feather_px = int(blend_cfg.get("cutoff_feather_px", 220))

    h_roi, w_roi = pano_lab_roi.shape[:2]
    grid_mean, grid_std, horizon_mean_x, horizon_std_x = _estimate_2d_sky_color_model(
        roi_lab=pano_lab_roi,
        roi_alpha=pano_alpha_roi,
        num_sectors=num_sectors,
        num_y_bands=6,
    )

    # Периодическая по азимуту 360° интерполяция 2D модели родного неба
    sky_mean_xy = _resize_azimuth_circular(grid_mean, target_w=w_roi, target_h=h_roi)
    sky_std_xy = _resize_azimuth_circular(grid_std, target_w=w_roi, target_h=h_roi)

    l_chan = pano_lab_roi[:, :, 0]
    a_chan = pano_lab_roi[:, :, 1]
    b_chan = pano_lab_roi[:, :, 2]

    dl = l_chan - sky_mean_xy[:, :, 0]
    std_l = np.maximum(sky_std_xy[:, :, 0], 1.0)
    da_norm = (a_chan - sky_mean_xy[:, :, 1]) / np.maximum(sky_std_xy[:, :, 1], 1.0)
    db_norm = (b_chan - sky_mean_xy[:, :, 2]) / np.maximum(sky_std_xy[:, :, 2], 1.0)
    dark_penalty = np.maximum(0.0, -dl / np.maximum(std_l, 5.0)) * 0.75
    mahal_dist = np.sqrt((dl / std_l) ** 2 + 1.2 * (da_norm**2 + db_norm**2)) + dark_penalty

    gx = cv2.Sobel(l_chan, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(l_chan, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(gx, gy)
    lap_mag = np.abs(cv2.Laplacian(l_chan, cv2.CV_32F, ksize=3))
    high_freq_response = (0.6 * sobel_mag + 0.4 * lap_mag) / 10.0

    is_smooth = high_freq_response < 1.85
    is_blue_sky = (b_chan < -3.5) & (l_chan > 18.0) & (a_chan < 8.0) & (high_freq_response < 2.8)
    is_bright_cloud = (
        (l_chan > 62.0)
        & (np.abs(a_chan) < 8.5)
        & (b_chan < 9.0)
        & (is_smooth | ((l_chan > 74.0) & (high_freq_response < 3.2)))
    )
    sky_or_cloud_whitelist = is_blue_sky | is_bright_cloud

    combined_dist = np.where(
        sky_or_cloud_whitelist,
        0.5 * edge_weight * high_freq_response,
        mahal_dist + edge_weight * high_freq_response,
    )

    span_t = max(0.5, mahal_obj_t - mahal_sky_t)
    t_raw = np.clip((combined_dist - mahal_sky_t) / span_t, 0.0, 1.0)
    m_raw = 1.0 - (3.0 * t_raw**2 - 2.0 * t_raw**3)

    m_raw = np.where(pano_alpha_roi > 220, m_raw, 1.0).astype(np.float32)

    max_obj_height_px = min(h_roi, int(blend_cfg.get("max_tall_object_height_px", 680)))
    y_obj_top = max(0, h_roi - max_obj_height_px)
    if y_obj_top > 0:
        fade_h = min(140, y_obj_top)
        m_raw[: max(0, y_obj_top - fade_h), :] = 1.0
        if fade_h > 0:
            y_f = np.linspace(0.0, 1.0, fade_h, dtype=np.float32).reshape(fade_h, 1)
            s_f = 3.0 * (y_f**2) - 2.0 * (y_f**3)
            r_start = y_obj_top - fade_h
            m_raw[r_start:y_obj_top, :] = 1.0 - s_f * (1.0 - m_raw[r_start:y_obj_top, :])

    band_h = h_roi - y_obj_top
    if band_h >= 16:
        sub_h = max(8, band_h // 2)
        sub_w = max(64, w_roi // 2)
        obj_bin = (m_raw[y_obj_top:, :] < 0.72).astype(np.uint8)
        obj_small = cv2.resize(obj_bin, (sub_w, sub_h), interpolation=cv2.INTER_NEAREST)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        obj_dil = cv2.dilate(obj_small, kernel, iterations=1)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(obj_dil, connectivity=8)
        if num_labels > 1:
            horizon_touch_y = max(1, sub_h - 14)
            bottoms = stats[:, cv2.CC_STAT_TOP] + stats[:, cv2.CC_STAT_HEIGHT]
            areas = stats[:, cv2.CC_STAT_AREA]
            valid_comp = (bottoms >= horizon_touch_y) & (areas >= 6)
            valid_comp[0] = False
            grounded_small = valid_comp[labels].astype(np.uint8)
            grounded_full = cv2.resize(
                grounded_small, (w_roi, band_h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            m_raw[y_obj_top:, :] = np.where(grounded_full, m_raw[y_obj_top:, :], 1.0)

    m_sky = _guided_filter_color(
        guide_bgr=pano_bgr_roi,
        src_mask=m_raw,
        radius=gf_radius,
        eps=gf_eps,
        subsample=gf_subsample,
    )

    if cutoff_feather_px > 0:
        y_grid = np.arange(h_roi, dtype=np.float32).reshape(h_roi, 1)
        top_rel_row = top_edge_rel.reshape(1, w_roi)
        dist_from_top = y_grid - top_rel_row
        t_cut = np.clip(dist_from_top / float(cutoff_feather_px), 0.0, 1.0)
        s_cut = 3.0 * t_cut**2 - 2.0 * t_cut**3
        m_sky = 1.0 - s_cut * (1.0 - m_sky)

    return np.clip(m_sky, 0.0, 1.0), horizon_mean_x, horizon_std_x


def _smooth_template_wrap_seam(sky_bgr: np.ndarray, half_width: int = 36) -> np.ndarray:
    """
    Плавно сшивает границу X=0 <-> X=W-1 самого шаблона неба до его циклического сдвига по азимуту солнца,
    чтобы возможная нестыковка краев исходного JPG шаблона не попадала внутрь панорамы после np.roll.
    """
    h, w = sky_bgr.shape[:2]
    hw = min(half_width, w // 8)
    if hw < 4:
        return sky_bgr
    left_strip = sky_bgr[:, :hw, :].astype(np.float32)
    right_strip = sky_bgr[:, w - hw :, :].astype(np.float32)
    edge_avg = 0.5 * (left_strip[:, 0:1, :] + right_strip[:, -1:, :])
    t = np.linspace(0.0, 1.0, hw, dtype=np.float32).reshape(1, hw, 1)
    s = 3.0 * (t**2) - 2.0 * (t**3)
    # На x=0 вес edge_avg равен 1.0, к x=hw плавно сходит к 0.0
    sky_bgr[:, :hw, :] = np.clip(edge_avg * (1.0 - s) + left_strip * s + 0.5, 0.0, 255.0).astype(
        np.uint8
    )
    # На x=w-1 вес edge_avg равен 1.0, к x=w-hw плавно сходит к 0.0
    sky_bgr[:, w - hw :, :] = np.clip(
         right_strip * (1.0 - s) + edge_avg * s + 0.5, 0.0, 255.0
    ).astype(np.uint8)
    return sky_bgr


def apply_reinhard_sky_color_transfer(
    sky_shifted_bgr: np.ndarray,
    pano_sky_mean_x: np.ndarray,
    pano_sky_std_x: np.ndarray,
    roi_y_start: int,
    horizon_y: int,
    config: dict[str, Any],
) -> np.ndarray:
    """
    Шаг 2.4.3: Выполняет мягкую азимутальную цветокоррекцию Рейнхарда в пространстве CIE LAB
    для шаблона неба со строго периодической (360° wrap-around) интерполяцией по горизонтали.
    """
    blend_cfg = config.get("sky_blender", {})
    strength_horizon = float(blend_cfg.get("reinhard_strength_horizon", 0.90))
    decay_power = float(blend_cfg.get("reinhard_decay_power", 1.4))
    zenith_min_weight = float(blend_cfg.get("reinhard_zenith_weight_ratio", 0.42))
    max_ab_shift = float(blend_cfg.get("max_ab_shift", 18.0))
    max_l_shift = float(blend_cfg.get("max_l_shift", 22.0))
    num_sectors = int(blend_cfg.get("num_azimuth_sectors", 20))

    h_sky, w_sky = sky_shifted_bgr.shape[:2]

    y0 = max(0, horizon_y - 600)
    y1 = min(h_sky, horizon_y)
    sky_roi_small = cv2.resize(
        sky_shifted_bgr[y0:y1, :],
        (max(num_sectors * 16, w_sky // 4), max(16, (y1 - y0) // 4)),
        interpolation=cv2.INTER_AREA,
    )
    sky_roi_lab = bgr_to_cie_lab(sky_roi_small)
    w_small = sky_roi_lab.shape[1]

    tmpl_means = np.zeros((num_sectors, 3), dtype=np.float32)
    tmpl_stds = np.zeros((num_sectors, 3), dtype=np.float32)
    slice_w = w_small // num_sectors
    for s_idx in range(num_sectors):
        x0 = s_idx * slice_w
        x1 = w_small if s_idx == num_sectors - 1 else (s_idx + 1) * slice_w
        pts = sky_roi_lab[:, x0:x1].reshape(-1, 3)
        tmpl_means[s_idx] = np.median(pts, axis=0)
        mad = np.median(np.abs(pts - tmpl_means[s_idx]), axis=0) * 1.4826
        tmpl_stds[s_idx] = np.clip(
            mad,
            np.array([4.5, 2.5, 3.5], dtype=np.float32),
            np.array([18.0, 12.0, 16.0], dtype=np.float32),
        )

    pad = 6
    t_means_pad = np.pad(tmpl_means, ((pad, pad), (0, 0)), mode="wrap")
    t_stds_pad = np.pad(tmpl_stds, ((pad, pad), (0, 0)), mode="wrap")
    t_means_smooth = cv2.GaussianBlur(t_means_pad.reshape(1, -1, 3), (9, 1), sigmaX=2.2)[
        :, pad : pad + num_sectors, :
    ]
    t_stds_smooth = cv2.GaussianBlur(t_stds_pad.reshape(1, -1, 3), (9, 1), sigmaX=2.2)[
        :, pad : pad + num_sectors, :
    ]

    tmpl_mean_x = _resize_azimuth_circular(t_means_smooth, target_w=w_sky, target_h=1)
    _ = _resize_azimuth_circular(t_stds_smooth, target_w=w_sky, target_h=1)

    delta_mean_x = pano_sky_mean_x - tmpl_mean_x
    delta_mean_x[:, :, 0] = np.clip(delta_mean_x[:, :, 0], -max_l_shift, max_l_shift)
    is_daylight_sky = pano_sky_mean_x[:, :, 2] < 4.0
    max_positive_da = np.where(is_daylight_sky, 0.0, max_ab_shift * 0.5)
    delta_mean_x[:, :, 1] = np.clip(delta_mean_x[:, :, 1], -max_ab_shift, max_positive_da)
    delta_mean_x[:, :, 2] = np.clip(delta_mean_x[:, :, 2], -max_ab_shift, max_ab_shift)

    eff_decay = max(decay_power, 2.0)
    eff_zenith_min = min(zenith_min_weight, 0.15)
    y_norm = np.linspace(0.0, 1.0, h_sky, dtype=np.float32).reshape(h_sky, 1, 1)
    w_vert = strength_horizon * (eff_zenith_min + (1.0 - eff_zenith_min) * (y_norm**eff_decay))

    sky_lab_u8 = cv2.cvtColor(sky_shifted_bgr, cv2.COLOR_BGR2LAB)

    delta_l_u8 = (delta_mean_x[:, :, 0] * 2.55).reshape(1, w_sky)
    delta_a_u8 = delta_mean_x[:, :, 1].reshape(1, w_sky)
    delta_b_u8 = delta_mean_x[:, :, 2].reshape(1, w_sky)

    chunk_rows = 500
    for r0 in range(0, h_sky, chunk_rows):
        r1 = min(h_sky, r0 + chunk_rows)
        w_sub = w_vert[r0:r1, :, 0]
        sub_f = sky_lab_u8[r0:r1].astype(np.float32)
        orig_l = sub_f[:, :, 0]
        orig_a = sub_f[:, :, 1]
        orig_b = sub_f[:, :, 2]

        cloud_damp = np.clip((240.0 - orig_l) / 62.0, 0.0, 1.0)
        dl_step = w_sub * delta_l_u8
        dl_step = np.where(dl_step < 0.0, dl_step * cloud_damp, dl_step * (0.4 + 0.6 * cloud_damp))
        da_step = (w_sub * delta_a_u8) * cloud_damp
        db_step = (w_sub * delta_b_u8) * cloud_damp

        new_l = np.clip(orig_l + dl_step, 0.0, 255.0)
        new_a = np.clip(orig_a + da_step, 0.0, 255.0)
        new_b = np.clip(orig_b + db_step, 0.0, 255.0)

        max_allowed_a = np.where(new_b < 126.0, np.minimum(orig_a, 128.0), 255.0)
        new_a = np.minimum(new_a, max_allowed_a)

        sub_f[:, :, 0] = new_l
        sub_f[:, :, 1] = new_a
        sub_f[:, :, 2] = new_b
        sky_lab_u8[r0:r1] = sub_f.astype(np.uint8)

    return cv2.cvtColor(sky_lab_u8, cv2.COLOR_LAB2BGR)


def _detect_template_sun_y(sky_bgr: np.ndarray) -> float | None:
    """
    Быстро определяет вертикальную координату Y центра солнечного диска в верхней полосе шаблона
    (в центре X = W/2 до циклического сдвига), если шаблон является солнечным/закатным.
    """
    h, w = sky_bgr.shape[:2]
    cx = w // 2
    strip = sky_bgr[:, max(0, cx - 200) : min(w, cx + 200), :]
    strip_small = cv2.resize(strip, (80, h // 4), interpolation=cv2.INTER_AREA)
    lab_s = bgr_to_cie_lab(strip_small)
    l_s = lab_s[:, :, 0]
    peak_l = float(np.percentile(l_s, 99.0))
    if peak_l < 91.5:
        return None
    thresh = max(90.0, float(np.percentile(l_s, 96.0)))
    ys, _ = np.where(l_s >= thresh)
    if len(ys) == 0:
        return None
    return float(np.median(ys) * 4.0)


def blend_sky_with_panorama(
    pano_bgr: np.ndarray,
    alpha_mask: np.ndarray,
    selected_sky_path: str,
    x_sun_pano: float,
    config: dict[str, Any],
    y_sun_pano: float | None = None,
    template_sun_y: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Шаг 2.4: Полный цикл азимутального выравнивания шаблона неба по солнцу,
    построения маски защиты высотных объектов (Fast Guided Filter), гармонизации экспозиции неба у горизонта,
    цветокоррекции Рейнхарда и градиентного Smoothstep-слияния с защитой единства солнечного диска.
    """
    pano_cfg = config.get("panorama", {})
    blend_cfg = config.get("sky_blender", {})

    pano_width = int(pano_cfg.get("width", 15000))
    horizon_y = int(pano_cfg.get("horizon_y", 3750))
    top_margin_px = int(blend_cfg.get("top_edge_margin_px", 140))
    void_feather_px = float(blend_cfg.get("void_feather_px", 200.0))
    smooth_sigma_px = float(blend_cfg.get("top_edge_smooth_sigma_px", 80.0))

    sky_bgr = read_image_bgr(selected_sky_path)
    if sky_bgr.shape[1] != pano_width or sky_bgr.shape[0] != horizon_y:
        sky_bgr = cv2.resize(sky_bgr, (pano_width, horizon_y), interpolation=cv2.INTER_CUBIC)

    if template_sun_y is None:
        template_sun_y = _detect_template_sun_y(sky_bgr)

    sky_bgr = _smooth_template_wrap_seam(sky_bgr, half_width=36)

    # 1. Циклический сдвиг шаблона неба по азимуту солнца (X_sun_sky = 7500 -> X_sun_pano)
    shift_px = int(round(x_sun_pano - (pano_width / 2.0)))
    sky_shifted_bgr = np.roll(sky_bgr, shift=shift_px, axis=1)

    # 2. Определение реальной верхней границы кадров дрона Y_top_edge(x)
    alpha_upper = alpha_mask[:horizon_y, :]
    top_edge_y = compute_top_edge_profile(
        alpha_upper=alpha_upper,
        horizon_y=horizon_y,
        smooth_sigma_px=smooth_sigma_px,
        pano_upper_bgr=pano_bgr[:horizon_y, :],
    )

    scallop_envelope_y = min(float(horizon_y - 250), float(np.percentile(top_edge_y, 90)) + 60.0)
    void_edge_y = np.maximum(top_edge_y, scallop_envelope_y)

    roi_y_start = max(0, int(np.floor(np.min(top_edge_y))) - 10)
    roi_h = horizon_y - roi_y_start

    pano_bgr_roi = pano_bgr[roi_y_start:horizon_y, :]
    pano_alpha_roi = alpha_upper[roi_y_start:horizon_y, :]
    pano_lab_roi = bgr_to_cie_lab(pano_bgr_roi)
    void_edge_rel = void_edge_y - float(roi_y_start)

    # 3. Построение маски защиты высотных объектов M_sky(x, y) в зоне перехода [roi_y_start, horizon_y]
    m_sky_roi, sky_mean_x, sky_std_x = build_tall_object_protection_mask(
        pano_bgr_roi=pano_bgr_roi,
        pano_lab_roi=pano_lab_roi,
        pano_alpha_roi=pano_alpha_roi,
        top_edge_rel=void_edge_rel,
        config=config,
    )

    # 4. Затухающая цветокоррекция Рейнхарда для шаблона неба
    sky_corrected_bgr = apply_reinhard_sky_color_transfer(
        sky_shifted_bgr=sky_shifted_bgr,
        pano_sky_mean_x=sky_mean_x,
        pano_sky_std_x=sky_std_x,
        roi_y_start=roi_y_start,
        horizon_y=horizon_y,
        config=config,
    )

    # 5. Построение вертикального S-градиента Smoothstep с защитой единства солнечного диска:
    y_grid_roi = np.arange(roi_y_start, horizon_y, dtype=np.float32).reshape(roi_h, 1)
    envelope_y = max(3020.0, float(np.percentile(top_edge_y, 90)) + float(top_margin_px))
    y_start_x = np.minimum(
        float(horizon_y - 50),
        np.maximum(void_edge_y + float(top_margin_px), envelope_y),
    ).reshape(1, pano_width)
    y_end_x = np.full((1, pano_width), float(horizon_y), dtype=np.float32)

    # Защита единства диска солнца от разрезания полупрозрачным градиентом у горизонта:
    x_coords = np.arange(pano_width, dtype=np.float32).reshape(1, pano_width)
    dx_sun = np.abs(x_coords - float(x_sun_pano))
    dx_sun = np.minimum(dx_sun, float(pano_width) - dx_sun)
    sun_bell_x = np.exp(-0.5 * ((dx_sun / 340.0) ** 2)).astype(np.float32)

    sun_x_idx = int(round(x_sun_pano)) % pano_width
    void_at_sun = float(void_edge_y[sun_x_idx])
    y_start_at_sun = float(y_start_x[0, sun_x_idx])

    sun_protection_mode = "none"
    sun_disk_boost_roi: np.ndarray | None = None
    pano_sun_protect_roi: np.ndarray | None = None

    if template_sun_y is not None and float(template_sun_y) >= (y_start_at_sun - 260.0):
        # Случай А: Солнечный диск выбранного шаблона находится низко в зоне градиентного перехода.
        # Сдвигаем начало затухания градиента НИЖЕ солнечного диска шаблона, чтобы весь диск имел непрозрачность 100%
        target_y_start = min(
            float(horizon_y - 30),
            max(float(template_sun_y), float(y_sun_pano or template_sun_y)) + 230.0,
        )
        y_start_x = y_start_x + sun_bell_x * np.maximum(0.0, target_y_start - y_start_x)

        dy_tmpl = y_grid_roi - float(template_sun_y)
        r_ell = np.sqrt((dx_sun / 340.0) ** 2 + (dy_tmpl / 230.0) ** 2)
        t_sun = np.clip((r_ell - 0.72) / 0.38, 0.0, 1.0)
        sun_disk_boost_roi = (1.0 - (3.0 * (t_sun**2) - 2.0 * (t_sun**3))).astype(np.float32)
        # Не позволяем светлой дымке/облакам панорамы пробивать дыры в солнечном диске шаблона,
        # но сохраняем защиту темных наземных объектов (зданий/деревьев с L* < 55)
        is_dark_grounded_obj = (pano_lab_roi[:, :, 0] < 55.0) & (m_sky_roi < 0.35)
        m_sky_roi = np.where(
            is_dark_grounded_obj,
            m_sky_roi,
            np.maximum(m_sky_roi, sun_disk_boost_roi),
        )
        sun_protection_mode = f"template_sun_solid(y={template_sun_y:.0f})"
    elif (
        y_sun_pano is not None
        and float(y_sun_pano) >= (void_at_sun + 240.0)
        and float(y_sun_pano) >= (y_start_at_sun - 180.0)
    ):
        # Случай Б: Реальное низкое солнце целиком снято дроном в самой панораме ниже границы зенита,
        # а в шаблоне солнце высоко или пасмурно. Завершаем градиент ВЫШЕ диска солнца панорамы,
        # чтобы полупрозрачное небо шаблона не заливало верхнюю половину солнечного диска панорамы.
        target_y_end = max(void_at_sun + 160.0, float(y_sun_pano) - 240.0)
        y_end_x = y_end_x - sun_bell_x * np.maximum(0.0, y_end_x - target_y_end)
        y_start_x = np.minimum(y_start_x, y_end_x - 60.0)

        dy_pano = y_grid_roi - float(y_sun_pano)
        r_ell = np.sqrt((dx_sun / 340.0) ** 2 + (dy_pano / 230.0) ** 2)
        t_sun = np.clip((r_ell - 0.70) / 0.40, 0.0, 1.0)
        bright_core = np.clip((pano_lab_roi[:, :, 0] - 80.0) / 12.0, 0.0, 1.0)
        below_void = (y_grid_roi > (void_edge_y.reshape(1, pano_width) + 60.0)).astype(np.float32)
        pano_sun_protect_roi = (
            (1.0 - (3.0 * (t_sun**2) - 2.0 * (t_sun**3))) * bright_core * below_void
        ).astype(np.float32)
        sun_protection_mode = f"pano_sun_preserved(y={y_sun_pano:.0f})"

    span_x = np.maximum(40.0, y_end_x - y_start_x)
    t_grad = np.clip((y_grid_roi - y_start_x) / span_x, 0.0, 1.0)
    alpha_grad_roi = 1.0 - (3.0 * (t_grad**2) - 2.0 * (t_grad**3))

    if sun_disk_boost_roi is not None:
        alpha_grad_roi = np.maximum(alpha_grad_roi, sun_disk_boost_roi)
    if pano_sun_protect_roi is not None:
        alpha_grad_roi = alpha_grad_roi * (1.0 - pano_sun_protect_roi)

    dist_below_edge = y_grid_roi - void_edge_y.reshape(1, pano_width)
    t_void = np.clip(dist_below_edge / max(25.0, void_feather_px), 0.0, 1.0)
    edge_void_feather = 1.0 - (3.0 * (t_void**2) - 2.0 * (t_void**3))
    m_void_roi = np.maximum(
        (pano_alpha_roi <= 220).astype(np.float32),
        edge_void_feather,
    )

    alpha_final_roi = np.clip(
        np.maximum(m_void_roi, alpha_grad_roi * m_sky_roi),
        0.0,
        1.0,
    )

    # 6. Слияние верхней полусферы панорамы блоками по 500 строк:
    if roi_y_start > 0:
        pano_bgr[:roi_y_start, :] = sky_corrected_bgr[:roi_y_start, :]

    sky_roi_u8 = sky_corrected_bgr[roi_y_start:horizon_y, :]
    chunk_h = 500
    for r0 in range(0, roi_h, chunk_h):
        r1 = min(roi_h, r0 + chunk_h)
        a_sub = alpha_final_roi[r0:r1, :, None]
        s_sub = sky_roi_u8[r0:r1].astype(np.float32)
        p_sub = pano_bgr_roi[r0:r1].astype(np.float32)
        p_sub += a_sub * (s_sub - p_sub)
        pano_bgr_roi[r0:r1] = np.clip(p_sub + 0.5, 0.0, 255.0).astype(np.uint8)

    protected_pixels_ratio = float(np.mean(m_sky_roi < 0.5))
    blend_stats = {
        "shift_px": shift_px,
        "top_edge_y_min": round(float(np.min(top_edge_y)), 1),
        "top_edge_y_mean": round(float(np.mean(top_edge_y)), 1),
        "top_edge_y_max": round(float(np.max(top_edge_y)), 1),
        "void_edge_y": round(float(np.mean(void_edge_y)), 1),
        "roi_y_start": roi_y_start,
        "protected_tall_objects_ratio": round(protected_pixels_ratio, 4),
        "sun_protection_mode": sun_protection_mode,
    }

    return pano_bgr, blend_stats


def save_panorama_jpeg(
    pano_bgr: np.ndarray,
    output_path: str,
    quality: int = 96,
    subsampling: int = 0,
    exif_bytes: bytes | None = None,
) -> None:
    """
    Сохраняет итоговую панораму 15000x7500 в JPEG максимального качества
    с субдискретизацией хромы 4:4:4 (subsampling=0) для сохранения четкости проводов ЛЭП
    и мелких деталей.
    """
    pano_rgb = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(pano_rgb)
    save_kwargs: dict[str, Any] = {
        "format": "JPEG",
        "quality": int(quality),
        "subsampling": int(subsampling),
    }
    if exif_bytes:
        save_kwargs["exif"] = exif_bytes
    pil_img.save(output_path, **save_kwargs)
