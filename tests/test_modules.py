"""Модульные и интеграционные тесты компонентов пайплайна (метаданные, PTO-генератор, подбор и блендинг неба)."""

from __future__ import annotations

import json
import os
import sys
import tempfile

import cv2
import numpy as np
import yaml
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.run_pipeline import default_output_path_for_shoot
from src.metadata import (
    compute_sun_position,
    filter_series_frames,
    get_reference_exif_bytes,
    inspect_shoot_directory,
    load_series_metadata,
    write_output_metadata,
)
from src.sky_replacer.blender import blend_sky_with_panorama, save_panorama_jpeg
from src.sky_replacer.matcher import select_best_sky_template
from src.stitcher.hugin_runner import load_prestitched_panorama_rgba
from src.stitcher.pto_generator import (
    enforce_angle_sanity_check,
    filter_sky_control_points,
    generate_initial_pto,
    order_frames_with_horizon_anchor,
)


def run_all_tests() -> None:
    with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(os.path.join(PROJECT_ROOT, "sky_catalog.json"), "r", encoding="utf-8") as f:
        catalog = json.load(f)

    pano_root = os.path.join(PROJECT_ROOT, "pano")
    if not os.path.isdir(pano_root) and os.path.isdir("/app/input"):
        pano_root = "/app/input"

    # 1. Тест фильтрации папки с сырыми кадрами + лишними панорамами
    series_dir = os.path.join(
        pano_root, "century-communities---georgia", "ashford-ridge", "2026-10-15_01"
    )
    inspected_gl = inspect_shoot_directory(
        series_dir, filter_cfg=config["stitcher"].get("input_filter")
    )
    assert inspected_gl["mode"] == "stitch", f"Expected stitch mode, got {inspected_gl['mode']}"
    frames = inspected_gl["raw_frames"]
    assert len(frames) == 25, f"Expected 25 filtered raw frames, got {len(frames)}"
    assert len(inspected_gl["filter_report"]["stitched_panorama_files"]) >= 1, (
        f"Expected >= 1 pre-stitched panoramas filtered out, got {inspected_gl['filter_report']['stitched_panorama_files']}"
    )

    # 2. Тест папок, где нет сырых кадров, но есть готовая сшитая панорама без неба
    prestitched_dir_15k = os.path.join(
        pano_root, "cedar-grove---tx", "generation-park-east-phase-3", "2026-10-17_01"
    )
    insp_15k = inspect_shoot_directory(prestitched_dir_15k)
    assert insp_15k["mode"] == "sky_only", f"Expected sky_only mode, got {insp_15k['mode']}"
    assert insp_15k["prestitched_pano"]["filename"] == "1.jpg"

    # 3. Тест 3-ступенчатой фильтрации дубликатов, картинок без XMP и иного разрешения
    synthetic_frames = [dict(f) for f in frames]
    # Добавляем дубликат обычного кадра (разница по углам < 1°)
    dup_frame = dict(frames[3])
    dup_frame["filename"] = "DUP_PANO0004.JPG"
    dup_frame["path"] = ""
    dup_frame["gimbal_yaw"] = float(frames[3]["gimbal_yaw"]) + 0.5
    dup_frame["gimbal_pitch"] = float(frames[3]["gimbal_pitch"]) - 0.4
    # Добавляем дубликат надира (pitch = -89.8°, но другой yaw)
    nadir_dup = dict(frames[0])
    nadir_dup["filename"] = "DUP_NADIR.JPG"
    nadir_dup["path"] = ""
    nadir_dup["gimbal_pitch"] = -89.8
    nadir_dup["gimbal_yaw"] = 123.0
    # Добавляем картинку без XMP подвеса
    no_xmp_frame = dict(frames[0])
    no_xmp_frame["filename"] = "RANDOM_SCREENSHOT.JPG"
    no_xmp_frame["path"] = ""
    no_xmp_frame["has_gimbal_xmp"] = False
    # Добавляем картинку другого разрешения
    wrong_res_frame = dict(frames[0])
    wrong_res_frame["filename"] = "THUMBNAIL.JPG"
    wrong_res_frame["path"] = ""
    wrong_res_frame["width"] = 1920
    wrong_res_frame["height"] = 1080

    kept, rep = filter_series_frames(
        synthetic_frames + [dup_frame, nadir_dup, no_xmp_frame, wrong_res_frame]
    )
    assert len(kept) == 25, f"Expected 25 kept frames after removing duplicates/invalid, got {len(kept)}"
    assert "RANDOM_SCREENSHOT.JPG" in rep["rejected_no_gimbal"]
    assert "THUMBNAIL.JPG" in rep["rejected_resolution"]
    assert len(rep["rejected_duplicates"]) == 2

    ordered = order_frames_with_horizon_anchor(frames)
    assert abs(ordered[0]["gimbal_pitch"]) < 25.0, "Anchor frame must be near horizon"

    sun_info = compute_sun_position(frames)
    assert sun_info["timezone"] == "America/New_York"
    assert 0 <= sun_info["x_sun_pano"] < 15000
    assert 0 <= sun_info["y_sun_pano_clamped"] <= 3750

    with tempfile.TemporaryDirectory() as tmpdir:
        # 4. Тест ресайза и маски для готовой сшитой панорамы (черный зенит vs синтетический зенит)
        black_zenith_img = np.full((300, 600, 3), 140, dtype=np.uint8)
        black_zenith_img[:70, :, :] = 0
        bz_path = os.path.join(tmpdir, "autopano_base.jpg")
        cv2.imwrite(bz_path, black_zenith_img, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
        pano_b, alpha_b, info_b = load_prestitched_panorama_rgba(
            bz_path, pano_width=1000, pano_height=500, default_zenith_cutoff_y=180
        )
        assert pano_b.shape == (500, 1000, 3)
        assert alpha_b.shape == (500, 1000)
        assert info_b["was_resized"] is True
        assert info_b["zenith_mask_mode"] == "black_void_detected"
        assert int(alpha_b[10, 500]) == 0 and int(alpha_b[200, 500]) == 255

        synth_zenith_img = np.full((300, 600, 3), 160, dtype=np.uint8)
        sz_path = os.path.join(tmpdir, "dji_incamera.jpg")
        cv2.imwrite(sz_path, synth_zenith_img, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
        pano_s, alpha_s, info_s = load_prestitched_panorama_rgba(
            sz_path, pano_width=1000, pano_height=500, default_zenith_cutoff_y=180
        )
        assert pano_s.shape == (500, 1000, 3)
        assert info_s["zenith_mask_mode"] == "synthetic_zenith_cutoff_180px"
        assert int(alpha_s[100, 500]) == 0 and int(alpha_s[200, 500]) == 255

        pto_path = os.path.join(tmpdir, "initial.pto")
        generate_initial_pto(ordered, pto_path)
        assert os.path.isfile(pto_path)

        # Проверяем фильтрацию контрольных точек неба
        cp_raw = os.path.join(tmpdir, "cp_raw.pto")
        with open(pto_path, "r", encoding="utf-8") as f:
            base_txt = f.read()
        fake_cps = (
            "c n0 N1 x2640 y100 X2640 Y120 t0\n"
            "c n0 N1 x2640 y3200 X2640 Y3100 t0\n"
        )
        with open(cp_raw, "w", encoding="utf-8") as f:
            f.write(base_txt + fake_cps)

        cp_no_sky = os.path.join(tmpdir, "cp_no_sky.pto")
        stats = filter_sky_control_points(cp_raw, cp_no_sky, ordered)
        assert stats["total_cps"] == 2
        assert stats["removed_sky_cps"] == 1
        assert stats["kept_cps"] == 1

        # Проверяем Sanity Check (>3.5° отклонение -> возврат к углам RTK)
        drifted_pto = os.path.join(tmpdir, "drifted.pto")
        with open(cp_no_sky, "r", encoding="utf-8") as f:
            txt = f.read()
        txt_drifted = txt.replace(
            f"y{ordered[1]['gimbal_yaw']:.4f}",
            f"y{ordered[1]['gimbal_yaw'] + 45.0:.4f}",
            1,
        )
        with open(drifted_pto, "w", encoding="utf-8") as f:
            f.write(txt_drifted)

        checked_pto = os.path.join(tmpdir, "checked.pto")
        s_stats = enforce_angle_sanity_check(drifted_pto, checked_pto, ordered, max_deviation_deg=3.5)
        assert 1 in s_stats["reset_frames"], "Drifted frame 1 must be reset by Sanity Check"

    # Проверка формирования пути <папка>/<папка>_full.jpeg прямо внутри папки съемки
    expected_in_folder_out = os.path.join(series_dir, "2026-10-15_01_full.jpeg")
    actual_in_folder_out = default_output_path_for_shoot(series_dir, config=config)
    assert os.path.normcase(actual_in_folder_out) == os.path.normcase(expected_in_folder_out), (
        f"Expected {expected_in_folder_out}, got {actual_in_folder_out}"
    )

    print("[TEST PASS] Модуль 1: метаданные, астрономия, initial.pto, фильтрация точек неба и Sanity Check прошли успешно!")


if __name__ == "__main__":
    run_all_tests()
