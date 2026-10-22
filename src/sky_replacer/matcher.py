"""Двухступенчатый выбор заготовки неба (Топ-5 по 20 срезам -> арбитраж микро-блендингом 600x150)."""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

from src.sky_replacer.signature import (
    bgr_to_cie_lab,
    compute_sky_signature,
    extract_clean_pano_sky_mask_in_band,
    read_image_bgr,
)


def _wasserstein_1d_hist(h1: list[float], h2: list[float], value_range: float = 100.0) -> float:
    """Точное 1D расстояние Вассерштейна (Earth Mover's Distance) для нормированных гистограмм с равными бинами."""
    p = np.asarray(h1, dtype=np.float64)
    q = np.asarray(h2, dtype=np.float64)
    p = p / max(1e-12, p.sum())
    q = q / max(1e-12, q.sum())
    cdf_diff = np.abs(np.cumsum(p) - np.cumsum(q))
    return float(np.mean(cdf_diff) * value_range)


def _detect_pano_overcast(
    pano_slices: list[dict[str, Any]],
    sun_info: dict[str, Any],
    delta_l_threshold: float = 9.0,
) -> bool:
    """
    Определяет индикатор сплошной пасмурной погоды на панораме (is_overcast_pano),
    комбинируя цветовую насыщенность синевы неба (канал b*), разброс экспозиции верхнего ряда кадров (Delta EV100)
    и направленную асимметрию яркости между сектором солнца (срезы 9, 10) и анти-солнца (срезы 0, 19).
    Если родное небо панорамы явно голубое/синее (mean_b < -10.5), погода не может быть сплошь пасмурной
    даже при высоком солнце в зените (когда dEV100 мал).
    """
    n = len(pano_slices)
    if n < 4:
        return bool(sun_info.get("is_overcast_ev", False))

    # Проверяем средний уровень синевы b* по всем очищенным срезам панорамы
    all_b = [pt[2] for s in pano_slices for pt in s["grad_lab"]]
    mean_b = float(np.mean(all_b)) if all_b else 0.0
    if mean_b < -10.5:
        return False

    center_indices = [n // 2 - 1, n // 2]
    edge_indices = [0, n - 1]

    center_l = np.mean([np.mean([pt[0] for pt in pano_slices[i]["grad_lab"]]) for i in center_indices])
    edge_l = np.mean([np.mean([pt[0] for pt in pano_slices[i]["grad_lab"]]) for i in edge_indices])
    pano_delta_l = float(center_l - edge_l)

    is_ev_flat = bool(sun_info.get("is_overcast_ev", False))
    # Если небо серое/белесое (mean_b >= -10.5), разброс экспозиции мал и нет солнечного пятна
    if is_ev_flat and pano_delta_l < delta_l_threshold:
        return True
    if pano_delta_l < (delta_l_threshold * 0.45):
        return True
    return False


def _compute_stage1_distance(
    pano_slices: list[dict[str, Any]],
    tmpl_entry: dict[str, Any],
    y_sun_pano: float,
    is_overcast_pano: bool,
    weights: dict[str, float],
) -> dict[str, float]:
    """
    Вычисляет расстояние Ступени 1 между очищенными срезами панорамы и паспортом шаблона из каталога.
    Учитывает то, что в Модуле 2.4 к шаблону будет применена мягкая цветокоррекция Рейнхарда,
    поэтому оценивает как прямое цветовое совпадение, так и совпадение градиентной структуры
    после частичной нормализации базовой экспозиции (55% сдвига в пределах допусков Рейнхарда).
    """
    tmpl_slices = tmpl_entry["slices"]
    num_slices = min(len(pano_slices), len(tmpl_slices))

    # Оцениваем глобальный сдвиг средней экспозиции/оттенка между панорамой и шаблоном
    p_all_grads = np.asarray([ps["grad_lab"] for ps in pano_slices[:num_slices]], dtype=np.float64)
    t_all_grads = np.asarray([ts["grad_lab"] for ts in tmpl_slices[:num_slices]], dtype=np.float64)
    cov_w = np.asarray([max(0.05, float(ps.get("coverage", 1.0))) for ps in pano_slices[:num_slices]], dtype=np.float64)
    cov_w = cov_w / cov_w.sum()

    p_global_mean = np.sum(p_all_grads.mean(axis=1) * cov_w[:, None], axis=0)
    t_global_mean = np.sum(t_all_grads.mean(axis=1) * cov_w[:, None], axis=0)
    raw_global_diff = p_global_mean - t_global_mean

    # Допустимая пред-компенсация экспозиции Рейнхарда (не позволяет путать синее небо с закатным или серым)
    reinhard_shift = np.array(
        [
            np.clip(raw_global_diff[0] * 0.55, -14.0, 14.0),
            np.clip(raw_global_diff[1] * 0.45, -7.0, 7.0),
            np.clip(raw_global_diff[2] * 0.50, -10.0, 10.0),
        ],
        dtype=np.float64,
    )

    grad_dists: list[float] = []
    emd_l_dists: list[float] = []
    emd_b_dists: list[float] = []
    dog_dists: list[float] = []
    slice_weights: list[float] = []

    for i in range(num_slices):
        ps = pano_slices[i]
        ts = tmpl_slices[i]
        w_cov = max(0.05, float(ps.get("coverage", 1.0)))
        slice_weights.append(w_cov)

        p_grad = np.asarray(ps["grad_lab"], dtype=np.float64)
        t_grad = np.asarray(ts["grad_lab"], dtype=np.float64) + reinhard_shift
        # Взвешенная разница (L*, a*, b*) по 6 точкам вертикального градиента
        diff = p_grad - t_grad
        channel_weights = np.array([1.0, 1.35, 1.35], dtype=np.float64)
        grad_d = float(np.mean(np.sqrt(np.sum((diff * channel_weights) ** 2, axis=1))))
        grad_dists.append(grad_d)

        emd_l = _wasserstein_1d_hist(ps["hist_l"], ts["hist_l"], value_range=100.0)
        emd_b = _wasserstein_1d_hist(ps["hist_b"], ts["hist_b"], value_range=120.0)
        # Компенсируем часть глобального сдвига яркости/синевы в EMD
        emd_l_adj = max(0.0, emd_l - abs(float(reinhard_shift[0])) * 0.65)
        emd_b_adj = max(0.0, emd_b - abs(float(reinhard_shift[2])) * 0.60)
        emd_l_dists.append(emd_l_adj)
        emd_b_dists.append(emd_b_adj)

        dog_d = abs(float(ps["dog_energy"]) - float(ts["dog_energy"]))
        dog_dists.append(dog_d)

    sw = np.asarray(slice_weights, dtype=np.float64)
    sw = sw / sw.sum()

    mean_grad = float(np.sum(np.asarray(grad_dists) * sw))
    mean_emd_l = float(np.sum(np.asarray(emd_l_dists) * sw))
    mean_emd_b = float(np.sum(np.asarray(emd_b_dists) * sw))
    mean_dog = float(np.sum(np.asarray(dog_dists) * sw))

    is_overcast_tmpl = bool(tmpl_entry.get("is_overcast_template", False))
    tmpl_sun_y = tmpl_entry.get("sun_y")

    if is_overcast_pano and is_overcast_tmpl:
        sun_elev_dist = 0.0
        weather_penalty = 0.0
    elif (not is_overcast_pano) and (not is_overcast_tmpl) and (tmpl_sun_y is not None):
        sun_elev_dist = (abs(float(y_sun_pano) - float(tmpl_sun_y)) / 3750.0) * 100.0
        weather_penalty = 0.0
    else:
        sun_elev_dist = 15.0
        weather_penalty = float(weights.get("overcast_penalty", 25.0))

    total_score = (
        float(weights.get("lab_grad", 1.0)) * mean_grad
        + float(weights.get("emd_l", 0.6)) * mean_emd_l
        + float(weights.get("emd_b", 1.0)) * mean_emd_b
        + float(weights.get("dog_energy", 1.8)) * mean_dog
        + float(weights.get("sun_elevation", 1.8)) * sun_elev_dist
        + weather_penalty
    )

    return {
        "total_stage1": round(total_score, 4),
        "grad_lab": round(mean_grad, 4),
        "emd_l": round(mean_emd_l, 4),
        "emd_b": round(mean_emd_b, 4),
        "dog_energy": round(mean_dog, 4),
        "sun_elev_dist": round(sun_elev_dist, 4),
        "weather_penalty": round(weather_penalty, 4),
    }


def _evaluate_micro_blend_cost(
    pano_upper_small_lab: np.ndarray,
    pano_alpha_small: np.ndarray,
    pano_clean_mask_small: np.ndarray,
    sky_small_lab: np.ndarray,
    seam_grad_weight: float = 1.0,
    color_shift_weight: float = 0.5,
) -> dict[str, float]:
    """
    Шаг 2.3.2: Ступень 2 (Арбитраж микро-блендингом на копиях 600x150).
    Выполняет пробное слияние кандидата с панорамой и оценивает:
    1) амплитуду необходимой цветокоррекции (color_shift_cost);
    2) скачок вертикальной производной яркости на шве (seam_grad_cost).
    """
    h_s, w_s = pano_upper_small_lab.shape[:2]
    # Переходная полоса около горизонта на уменьшенной копии (соответствует Y in [2850, 3650] из 3750)
    y0 = max(1, int(round(h_s * (2850.0 / 3750.0))))
    y1 = min(h_s - 1, int(round(h_s * (3650.0 / 3750.0))))

    roi_mask = pano_clean_mask_small[y0:y1, :]
    pano_roi = pano_upper_small_lab[y0:y1, :]
    sky_roi = sky_small_lab[y0:y1, :]

    if np.count_nonzero(roi_mask) > 50:
        p_mean = np.median(pano_roi[roi_mask], axis=0)
        p_std = np.std(pano_roi[roi_mask], axis=0) + 1e-3
        s_mean = np.median(sky_roi[roi_mask], axis=0)
        s_std = np.std(sky_roi[roi_mask], axis=0) + 1e-3
    else:
        p_mean = np.median(pano_roi.reshape(-1, 3), axis=0)
        p_std = np.std(pano_roi.reshape(-1, 3), axis=0) + 1e-3
        s_mean = np.median(sky_roi.reshape(-1, 3), axis=0)
        s_std = np.std(sky_roi.reshape(-1, 3), axis=0) + 1e-3

    # Стоимость необходимой цветокоррекции
    mean_shift = float(np.linalg.norm((p_mean - s_mean) * np.array([1.0, 1.4, 1.4])))
    std_shift = float(np.linalg.norm(p_std - s_std))
    color_shift_cost = mean_shift + 0.5 * std_shift

    # Применяем мягкую затухающую коррекцию Рейнхарда к уменьшенному шаблону
    y_coords = np.linspace(0.0, 1.0, h_s, dtype=np.float32).reshape(h_s, 1, 1)
    decay = (y_coords**1.8) * 0.85
    delta_lab = (p_mean - s_mean).reshape(1, 1, 3).astype(np.float32)
    sky_corr = sky_small_lab + decay * delta_lab

    # Строим вертикальный Smoothstep градиент от верхнего края покрытия дрона до горизонта
    alpha_grad = np.ones((h_s, w_s), dtype=np.float32)
    for x in range(w_s):
        col_valid = np.where(pano_alpha_small[:, x] > 128)[0]
        top_y = int(col_valid[0]) if len(col_valid) > 0 else y0
        start_y = min(h_s - 2, top_y + 2)
        span = max(1, h_s - start_y)
        t = np.clip((np.arange(h_s, dtype=np.float32) - start_y) / span, 0.0, 1.0)
        # Наверху alpha=1 (новое небо), у горизонта alpha=0 (панорама)
        alpha_grad[:, x] = 1.0 - (3.0 * t**2 - 2.0 * t**3)

    void_mask = (pano_alpha_small <= 128).astype(np.float32)
    alpha_final = np.maximum(void_mask, alpha_grad * pano_clean_mask_small.astype(np.float32))
    alpha_3c = alpha_final[:, :, None]

    blended_lab = alpha_3c * sky_corr + (1.0 - alpha_3c) * pano_upper_small_lab

    # Оцениваем скачок вертикальной производной яркости dL/dy в зоне перехода
    dl_blended = np.abs(np.diff(blended_lab[:, :, 0], axis=0))
    dl_sky = np.abs(np.diff(sky_corr[:, :, 0], axis=0))
    seam_jump = np.abs(dl_blended[y0:y1, :] - dl_sky[y0:y1, :])
    if np.any(roi_mask):
        seam_grad_cost = float(np.percentile(seam_jump[roi_mask], 85))
    else:
        seam_grad_cost = float(np.mean(seam_jump))

    total_micro_cost = (
        color_shift_weight * color_shift_cost + seam_grad_weight * seam_grad_cost
    )
    return {
        "micro_total": round(float(total_micro_cost), 4),
        "color_shift_cost": round(float(color_shift_cost), 4),
        "seam_grad_cost": round(float(seam_grad_cost), 4),
    }


def select_best_sky_template(
    pano_bgr: np.ndarray,
    alpha_mask: np.ndarray,
    sun_info: dict[str, Any],
    catalog: dict[str, Any],
    config: dict[str, Any],
    sky_base_dir: str,
) -> dict[str, Any]:
    """
    Выполняет двухступенчатый подбор лучшего шаблона неба из sky_catalog.json:
    - Ступень 1: Сдвиг полосы панорамы Y in [2850, 3650] по солнцу, очистка от зданий/ЛЭП/кранов,
      сравнение 20 срезов с базой и отбор Топ-5 кандидатов.
    - Ступень 2: Арбитраж микро-блендингом на копиях 600x150 и выбор шаблона с минимальной стоимостью шва.
    """
    idx_cfg = config.get("sky_index", {})
    match_cfg = config.get("sky_matcher", {})
    pano_cfg = config.get("panorama", {})

    horizon_y = int(pano_cfg.get("horizon_y", 3750))
    pano_width = int(pano_cfg.get("width", 15000))
    y_min = int(idx_cfg.get("band_y_min", 2850))
    y_max = int(idx_cfg.get("band_y_max", 3650))
    num_slices = int(idx_cfg.get("num_slices", 20))

    top_k = int(match_cfg.get("top_k", 5))
    sobel_pct = float(match_cfg.get("sobel_percentile_cutoff", 75.0))
    l_min_pct = float(match_cfg.get("l_min_percentile_cutoff", 45.0))
    delta_l_pano_thresh = float(match_cfg.get("overcast_pano_delta_l_threshold", 9.0))
    weights = match_cfg.get("weights", {})
    mb_cfg = match_cfg.get("micro_blend", {})
    mb_w = int(mb_cfg.get("width", 600))
    mb_h = int(mb_cfg.get("height", 150))

    x_sun_pano = float(sun_info["x_sun_pano"])
    y_sun_pano = float(sun_info["y_sun_pano_clamped"])

    # Сдвигаем полосу неба панорамы так, чтобы X_sun_pano встал в центр X = 7500
    shift_to_center = int(round((pano_width / 2.0) - x_sun_pano))
    band_bgr = np.roll(pano_bgr[y_min:y_max, :], shift=shift_to_center, axis=1)
    band_alpha = np.roll(alpha_mask[y_min:y_max, :], shift=shift_to_center, axis=1)

    band_lab = bgr_to_cie_lab(band_bgr)
    clean_mask_band = extract_clean_pano_sky_mask_in_band(
        band_lab=band_lab,
        band_alpha=band_alpha,
        num_slices=num_slices,
        sobel_percentile_cutoff=sobel_pct,
        l_min_percentile_cutoff=l_min_pct,
    )

    pano_slices = compute_sky_signature(
        sky_or_pano_bgr=band_bgr,
        config=config,
        valid_mask_band=clean_mask_band,
        is_already_band=True,
    )

    is_overcast_pano = _detect_pano_overcast(
        pano_slices=pano_slices,
        sun_info=sun_info,
        delta_l_threshold=delta_l_pano_thresh,
    )

    templates = catalog.get("templates", [])
    if not templates:
        raise RuntimeError("Каталог шаблонов неба пуст.")

    stage1_results: list[dict[str, Any]] = []
    for entry in templates:
        s1 = _compute_stage1_distance(
            pano_slices=pano_slices,
            tmpl_entry=entry,
            y_sun_pano=y_sun_pano,
            is_overcast_pano=is_overcast_pano,
            weights=weights,
        )
        stage1_results.append({"entry": entry, "stage1": s1})

    stage1_results.sort(key=lambda item: item["stage1"]["total_stage1"])
    top_candidates = stage1_results[: min(top_k, len(stage1_results))]

    # Ступень 2: Подготовка уменьшенной в 25 раз копии верхней полусферы панорамы (600x150)
    pano_upper_small_bgr = cv2.resize(
        pano_bgr[:horizon_y, :], (mb_w, mb_h), interpolation=cv2.INTER_AREA
    )
    pano_alpha_small = cv2.resize(
        alpha_mask[:horizon_y, :], (mb_w, mb_h), interpolation=cv2.INTER_NEAREST
    )
    pano_upper_small_lab = bgr_to_cie_lab(pano_upper_small_bgr)
    pano_clean_small = extract_clean_pano_sky_mask_in_band(
        band_lab=pano_upper_small_lab,
        band_alpha=pano_alpha_small,
        num_slices=num_slices,
        sobel_percentile_cutoff=sobel_pct,
        l_min_percentile_cutoff=l_min_pct,
    )

    shift_to_pano_full = int(round(x_sun_pano - (pano_width / 2.0)))
    shift_to_pano_small = int(round(shift_to_pano_full * (mb_w / float(pano_width))))

    candidates_report: list[dict[str, Any]] = []
    for cand in top_candidates:
        entry = cand["entry"]
        rel_path = entry["relative_path"]
        full_path = os.path.join(sky_base_dir, rel_path)
        if not os.path.isfile(full_path) and os.path.isfile(entry.get("path", "")):
            full_path = entry["path"]

        sky_bgr = read_image_bgr(full_path, max_width=mb_w)
        sky_small_bgr = cv2.resize(sky_bgr, (mb_w, mb_h), interpolation=cv2.INTER_AREA)
        sky_small_shifted = np.roll(sky_small_bgr, shift=shift_to_pano_small, axis=1)
        sky_small_lab = bgr_to_cie_lab(sky_small_shifted)

        s2 = _evaluate_micro_blend_cost(
            pano_upper_small_lab=pano_upper_small_lab,
            pano_alpha_small=pano_alpha_small,
            pano_clean_mask_small=pano_clean_small,
            sky_small_lab=sky_small_lab,
            seam_grad_weight=float(mb_cfg.get("seam_grad_weight", 1.0)),
            color_shift_weight=float(mb_cfg.get("color_shift_weight", 0.5)),
        )

        # Комбинированная оценка арбитража
        combined_score = cand["stage1"]["total_stage1"] * 0.65 + s2["micro_total"] * 0.35
        candidates_report.append(
            {
                "id": entry["id"],
                "relative_path": rel_path,
                "full_path": full_path,
                "sun_y": entry.get("sun_y"),
                "is_overcast_template": entry.get("is_overcast_template"),
                "stage1": cand["stage1"],
                "stage2": s2,
                "combined_score": round(float(combined_score), 4),
            }
        )

    candidates_report.sort(key=lambda item: item["combined_score"])
    winner = candidates_report[0]

    return {
        "selected_template": winner,
        "top_candidates": candidates_report,
        "is_overcast_pano": is_overcast_pano,
        "x_sun_pano": x_sun_pano,
        "y_sun_pano": y_sun_pano,
    }
