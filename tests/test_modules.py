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

from scripts.async_pipeline import run_async_pipeline
from scripts.run_pipeline import (
    default_output_path_for_shoot,
    derive_series_output_path,
    discover_shoot_dirs,
)
from src.metadata import (
    compute_sun_position,
    filter_series_frames,
    inspect_shoot_directory,
    split_raw_frames_into_series,
)
from src.sky_replacer.blender import blend_sky_with_panorama
from src.stitcher.hugin_runner import load_prestitched_panorama_rgba
from src.stitcher.pto_generator import (
    compute_pto_cp_rms,
    enforce_angle_sanity_check,
    filter_sky_control_points,
    generate_initial_pto,
    order_frames_with_horizon_anchor,
)
from src.storage.s3_downloader import (
    load_dotenv_file,
    parse_asana_task_string,
    write_dotenv_file,
)


def run_all_tests() -> None:
    with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pano_root = os.path.join(PROJECT_ROOT, "pano")
    if not os.path.isdir(pano_root) and os.path.isdir("/app/input"):
        pano_root = "/app/input"

    shoot_dirs = discover_shoot_dirs(pano_root)
    assert len(shoot_dirs) >= 1, f"Expected >= 1 shoot directories in {pano_root}"

    # 1. Тест фильтрации папки с сырыми кадрами и разделения нескольких серий
    series_dir = shoot_dirs[0]
    inspected_gl = inspect_shoot_directory(
        series_dir, filter_cfg=config["stitcher"].get("input_filter")
    )
    assert inspected_gl["mode"] == "stitch", f"Expected stitch mode, got {inspected_gl['mode']}"
    frames = inspected_gl["raw_frames"]
    assert len(frames) >= 20, f"Expected >= 20 filtered raw frames, got {len(frames)}"

    # Если это папка с двумя сериями (Pano(1)1..25 и Pano(2)1..25), проверяем точное разделение на 2 серии по 25 кадров
    raw_series = inspected_gl.get("raw_series", [])
    if inspected_gl["filter_report"]["total_files"] == 50:
        assert len(raw_series) == 2, f"Expected 2 separated series, got {len(raw_series)}"
        assert len(raw_series[0]) == 25 and len(raw_series[1]) == 25
        assert all(f["filename"].startswith("Pano(1)") for f in raw_series[0])
        assert all(f["filename"].startswith("Pano(2)") for f in raw_series[1])

        # Проверяем, что даже без префикса Pano(1)/Pano(2) (при единой нумерации DJI_0001..DJI_0050)
        # серии всё равно разделяются на 2 по 25 кадров за счет разрыва по времени (39 с) и скачка GPS (39.4 м)
        renamed_50: list[dict] = []
        for idx, f in enumerate(raw_series[0] + raw_series[1], start=1):
            fc = dict(f)
            fc["filename"] = f"DJI_{idx:04d}.JPG"
            renamed_50.append(fc)
        split_no_prefix = split_raw_frames_into_series(
            renamed_50, filter_cfg=config["stitcher"].get("input_filter")
        )
        assert len(split_no_prefix) == 2 and len(split_no_prefix[0]) == 25 and len(split_no_prefix[1]) == 25

        # Проверяем, что даже при отсутствии разрыва по времени и GPS (пересъемка в одной точке без паузы)
        # серии разделяются по завершению полного 360°-сферического цикла (Критерий В)
        same_spot_50: list[dict] = []
        for idx, f in enumerate(raw_series[0] + raw_series[1], start=1):
            fc = dict(f)
            fc["filename"] = f"DJI_{idx:04d}.JPG"
            fc["datetime_iso_aware"] = None
            fc["datetime_str"] = None
            fc["xmp"] = {}
            fc["gps_lat"] = raw_series[0][0]["gps_lat"]
            fc["gps_lon"] = raw_series[0][0]["gps_lon"]
            fc["relative_altitude"] = raw_series[0][0]["relative_altitude"]
            same_spot_50.append(fc)
        split_by_cycle = split_raw_frames_into_series(
            same_spot_50, filter_cfg=config["stitcher"].get("input_filter")
        )
        assert len(split_by_cycle) == 2 and len(split_by_cycle[0]) == 25 and len(split_by_cycle[1]) == 25

    # 2. Тест 3-ступенчатой фильтрации дубликатов, картинок без XMP и иного разрешения
    synthetic_frames = [dict(f) for f in frames]
    dup_frame = dict(frames[3])
    dup_frame["filename"] = "DUP_PANO0004.JPG"
    dup_frame["path"] = ""
    dup_frame["gimbal_yaw"] = float(frames[3]["gimbal_yaw"]) + 0.5
    dup_frame["gimbal_pitch"] = float(frames[3]["gimbal_pitch"]) - 0.4

    nadir_dup = dict(frames[0])
    nadir_dup["filename"] = "DUP_NADIR.JPG"
    nadir_dup["path"] = ""
    nadir_dup["gimbal_pitch"] = -89.8
    nadir_dup["gimbal_yaw"] = 123.0

    no_xmp_frame = dict(frames[0])
    no_xmp_frame["filename"] = "RANDOM_SCREENSHOT.JPG"
    no_xmp_frame["path"] = ""
    no_xmp_frame["has_gimbal_xmp"] = False

    wrong_res_frame = dict(frames[0])
    wrong_res_frame["filename"] = "THUMBNAIL.JPG"
    wrong_res_frame["path"] = ""
    wrong_res_frame["width"] = 1920
    wrong_res_frame["height"] = 1080

    kept, rep = filter_series_frames(
        synthetic_frames + [dup_frame, nadir_dup, no_xmp_frame, wrong_res_frame]
    )
    assert len(kept) == len(frames), f"Expected {len(frames)} kept frames, got {len(kept)}"
    assert "RANDOM_SCREENSHOT.JPG" in rep["rejected_no_gimbal"]
    assert "THUMBNAIL.JPG" in rep["rejected_resolution"]
    assert len(rep["rejected_duplicates"]) == 2

    ordered = order_frames_with_horizon_anchor(frames)
    assert abs(ordered[0]["gimbal_pitch"]) < 25.0, "Anchor frame must be near horizon"

    sun_info = compute_sun_position(frames)
    assert 0 <= sun_info["x_sun_pano"] < 15000
    assert 0 <= sun_info["y_sun_pano_clamped"] <= 3750

    with tempfile.TemporaryDirectory() as tmpdir:
        # 3. Тест ресайза и маски для готовой сшитой панорамы (черный зенит vs синтетический зенит)
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

        rms_info = compute_pto_cp_rms(cp_no_sky)
        assert rms_info["count"] == 1
        assert rms_info["rms_px"] >= 0.0

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

        # 4. Тест сшивания экспозиции неба на стыке кадров (360° wrap + вертикальный шов) и защиты диска солнца
        w_test, h_test, hor_test = 2400, 1200, 600
        test_cfg = json.loads(json.dumps(config))
        test_cfg["panorama"]["width"] = w_test
        test_cfg["panorama"]["height"] = h_test
        test_cfg["panorama"]["horizon_y"] = hor_test
        test_cfg["sky_blender"]["top_edge_margin_px"] = 25
        test_cfg["sky_blender"]["void_feather_px"] = 35.0
        test_cfg["sky_blender"]["cutoff_feather_px"] = 40

        pano_synth = np.zeros((h_test, w_test, 3), dtype=np.uint8)
        pano_synth[:, :] = (210, 165, 115)  # голубое небо
        # Создаем резкую ступеньку экспозиции между секторами X=1200 и X=0/2399
        pano_synth[:, 1200:] = (242, 198, 148)
        pano_synth[hor_test:, :] = (70, 95, 80)  # земля
        alpha_synth = np.full((h_test, w_test), 255, dtype=np.uint8)
        alpha_synth[:420, :] = 0

        sky_tmpl = np.full((hor_test, w_test, 3), (220, 170, 120), dtype=np.uint8)
        # Низкое солнце на шаблоне (X=w_test//2, Y=490 в зоне градиентного перехода)
        cv2.circle(sky_tmpl, (w_test // 2, 490), 35, (250, 253, 255), -1)
        sky_tmpl_path = os.path.join(tmpdir, "sky_tmpl.jpg")
        cv2.imwrite(sky_tmpl_path, sky_tmpl, [int(cv2.IMWRITE_JPEG_QUALITY), 98])

        blended_pano, b_stats = blend_sky_with_panorama(
            pano_bgr=pano_synth.copy(),
            alpha_mask=alpha_synth,
            selected_sky_path=sky_tmpl_path,
            x_sun_pano=600.0,
            config=test_cfg,
            y_sun_pano=490.0,
            template_sun_y=490.0,
        )
        wrap_diff = float(
            np.mean(
                np.abs(
                    blended_pano[250:420, 0].astype(np.float32)
                    - blended_pano[250:420, -1].astype(np.float32)
                )
            )
        )
        assert wrap_diff < 1.5, f"Expected smooth 360 wrap seam, got wrap_diff={wrap_diff:.2f}"
        assert b_stats["sun_protection_mode"].startswith("template_sun_solid"), (
            f"Expected template_sun_solid protection mode, got {b_stats['sun_protection_mode']}"
        )

    folder_name = os.path.basename(os.path.abspath(series_dir))
    expected_in_folder_out = os.path.join(series_dir, f"{folder_name}_full.jpeg")
    actual_in_folder_out = default_output_path_for_shoot(series_dir, config=config)
    assert os.path.normcase(actual_in_folder_out) == os.path.normcase(expected_in_folder_out), (
        f"Expected {expected_in_folder_out}, got {actual_in_folder_out}"
    )
    second_series_out = derive_series_output_path(actual_in_folder_out, 2, config=config)
    expected_second_out = os.path.join(series_dir, f"{folder_name}_2_full.jpeg")
    assert os.path.normcase(second_series_out) == os.path.normcase(expected_second_out)

    # 5. Тесты парсера задач Asana, работы с .env и асинхронного конвейера (keep_raw=True / False)
    t1 = parse_asana_task_string("Panorama: customer_a/site_b/2026-10-31_01")
    assert t1["customer"] == "customer_a" and t1["site"] == "site_b" and t1["date"] == "2026-10-31_01"
    assert t1["s3_prefix"] == "customer_a/site_b/panoramas/2026-10-31_01/input/"

    t2 = parse_asana_task_string("anorama: customer_a/site_b/2026-10-31_02")
    assert t2["relative_dir"] == "customer_a/site_b/2026-10-31_02"

    t3 = parse_asana_task_string("s3://internals.traceair.net/cust_x/site_y/panoramas/2026-11-01_03/input")
    assert t3["bucket"] == "internals.traceair.net"
    assert t3["customer"] == "cust_x" and t3["site"] == "site_y" and t3["date"] == "2026-11-01_03"

    with tempfile.TemporaryDirectory(prefix="phauto_async_test_") as async_tmp:
        env_test_path = os.path.join(async_tmp, ".env")
        with open(env_test_path, "w", encoding="utf-8") as f:
            f.write("ASANA_PAT=2/test_asana_pat_123\nTRACEAIR_SESS_ID=s%3Atest_session_cookie\n")

        write_dotenv_file(
            env_test_path,
            {
                "AWS_ACCESS_KEY_ID": "ASIATEST123456789",
                "AWS_SECRET_ACCESS_KEY": "secret_test_xyz",
                "AWS_SESSION_TOKEN": "IQoJb3JpZ2luX2VjE_test_token",
                "AWS_DEFAULT_REGION": "us-east-1",
            },
        )
        loaded_env = load_dotenv_file(env_test_path, override=False)
        assert loaded_env["AWS_ACCESS_KEY_ID"] == "ASIATEST123456789"
        assert loaded_env["AWS_DEFAULT_REGION"] == "us-east-1"
        assert loaded_env["ASANA_PAT"] == "2/test_asana_pat_123", "ASANA_PAT must be preserved when updating AWS keys"
        assert loaded_env["TRACEAIR_SESS_ID"] == "s%3Atest_session_cookie", "TRACEAIR_SESS_ID must be preserved"

        # Проверяем работу clean_env при наличии старых ключей в os.environ
        from src.storage.s3_downloader import _clean_aws_env
        test_dirty = {"AWS_ACCESS_KEY_ID": "OLD", "AWS_SECRET_ACCESS_KEY": "OLD_SEC", "PATH": "C:\\bin"}
        cleaned = _clean_aws_env(test_dirty)
        assert "AWS_ACCESS_KEY_ID" not in cleaned and "AWS_SECRET_ACCESS_KEY" not in cleaned
        assert cleaned["PATH"] == "C:\\bin"

        # Проверяем изолированный сервис src/magic_service (если подключён)
        from src.magic_service import find_primary_panorama_file, parse_asana_tasks_payload

        sample_asana_payload = [
            {
                "gid": "120001",
                "name": "For Panorama cust1/site1/2026-10-31_01",
                "notes": "Link: https://magic.traceair.net/cust1/site1/11111111-2222-3333-4444-555555555555",
                "completed": False,
            },
            {
                "gid": "120002",
                "name": "Panorama: cust1/site1/2026-10-31_02",
                "notes": "https://magic.traceair.net/cust1/site1/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                "completed": False,
            },
        ]
        parsed_asana = parse_asana_tasks_payload(sample_asana_payload)
        assert len(parsed_asana) == 2
        assert parsed_asana[0]["card_id"] == "11111111-2222-3333-4444-555555555555"
        assert parsed_asana[1]["relative_dir"] == "cust1/site1/2026-10-31_02"

        created_raw_dirs: list[str] = []
        published_calls: list[dict] = []

        def _fake_download_fn(task_info: dict, dest_dir: str, **kwargs):
            os.makedirs(dest_dir, exist_ok=True)
            dummy_raw = os.path.join(dest_dir, "DJI_0001.JPG")
            with open(dummy_raw, "wb") as f:
                f.write(b"dummy_raw_frame")
            created_raw_dirs.append(dest_dir)
            return {
                **task_info,
                "dest_dir": os.path.abspath(dest_dir),
                "total_files": 1,
                "downloaded_files": 1,
                "skipped_files": 0,
                "total_bytes": 15,
                "elapsed_sec": 0.01,
            }

        def _fake_stitch_fn(job_payload: dict):
            raw_input_dir = job_payload["input_dir"]
            output_jpeg_path = job_payload["output_jpg_path"]
            assert os.path.isdir(raw_input_dir), "Raw input directory must exist during stitching"
            os.makedirs(os.path.dirname(output_jpeg_path), exist_ok=True)
            # Записываем >100 КБ для прохождения проверки размера готовой панорамы
            with open(output_jpeg_path, "wb") as f:
                f.write(b"x" * 150_000)
            # Имитируем наличие второй серии _2_full.jpeg в той же папке
            second_out = output_jpeg_path.replace("_full.jpeg", "_2_full.jpeg")
            with open(second_out, "wb") as f:
                f.write(b"y" * 150_000)
            return {
                "series_count": 2,
                "output_paths": [output_jpeg_path, second_out],
                "elapsed_sec": 0.01,
                "reports": [{"output_path": output_jpeg_path}, {"output_path": second_out}],
            }

        def _fake_publish_fn(task: dict, local_file: str, **kwargs):
            published_calls.append({"task": task, "local_file": str(local_file)})
            return {"status": "published", "card_id": task["card_id"], "local_file": str(local_file)}

        import asyncio

        # Проверяем режим keep_raw=False + сквозную автопубликацию (auto_publish=True)
        out_root_temp_mode = os.path.join(async_tmp, "pano_temp_mode")
        meta_map = {item["relative_dir"]: item for item in parsed_asana}
        res_temp = asyncio.run(
            run_async_pipeline(
                task_strings=["cust1/site1/2026-10-31_01", "cust1/site1/2026-10-31_02"],
                pano_root=out_root_temp_mode,
                sky_dir=os.path.join(PROJECT_ROOT, "sky"),
                catalog={"skies": []},
                config=config,
                env_path=env_test_path,
                keep_raw=False,
                prefetch_limit=2,
                stitch_workers=1,
                download_fn=_fake_download_fn,
                stitch_fn=_fake_stitch_fn,
                asana_tasks_meta=meta_map,
                auto_publish=True,
                publish_fn=_fake_publish_fn,
            )
        )
        assert len(res_temp["completed"]) == 2 and len(res_temp["failed"]) == 0
        assert len(res_temp["published"]) == 2 and len(published_calls) == 2
        for raw_d in created_raw_dirs:
            assert not os.path.exists(raw_d), f"Temporary raw dir {raw_d} must be deleted after stitching"
        final_pano_1 = os.path.join(out_root_temp_mode, "cust1", "site1", "2026-10-31_01", "2026-10-31_01_full.jpeg")
        assert os.path.isfile(final_pano_1), "Final stitched panorama must remain in output folder"
        # Убеждаемся, что find_primary_panorama_file выбирает именно первую основную панораму (_full.jpeg, а не _2_full.jpeg)
        primary_found = find_primary_panorama_file(out_root_temp_mode, "cust1", "site1", "2026-10-31_01")
        assert primary_found is not None and os.path.normcase(str(primary_found)) == os.path.normcase(final_pano_1)

        # Проверяем парсинг и нормализацию задач для Airtable
        from src.magic_service.airtable_client import (
            ASANA_NAME_COLUMN_ID,
            MY_DISPLAY_NAME,
            MY_FOREIGN_ROW_ID,
            OWNER_COLUMN_ID,
            TABLE_ID,
            make_detail_b64,
            normalize_asana_task_key,
            parse_airtable_read_queries_payload,
        )

        assert normalize_asana_task_key("Panorama: cust1/site1/2026-10-31_01") == "cust1/site1/2026-10-31_01"
        assert normalize_asana_task_key("For Panorama cust1/site1/2026-10-31_02") == "cust1/site1/2026-10-31_02"
        detail_b64 = make_detail_b64("recp7i9iLrK1bj5mX")
        assert not detail_b64.endswith("="), "Airtable detail base64 must strip trailing '='"

        mock_airtable_payload = {
            "msg": "SUCCESS",
            "data": {
                "querySlices": [
                    {
                        "tableId": TABLE_ID,
                        "rowIds": ["recRow1", "recRow2"],
                    }
                ],
                "tableDataById": {
                    TABLE_ID: {
                        "partialRowById": {
                            "recRow1": {
                                "id": "recRow1",
                                "cellValuesByColumnId": {
                                    ASANA_NAME_COLUMN_ID: "cust1/site1/2026-10-31_01",
                                    OWNER_COLUMN_ID: [
                                        {
                                            "foreignRowId": MY_FOREIGN_ROW_ID,
                                            "foreignRowDisplayName": MY_DISPLAY_NAME,
                                        }
                                    ],
                                },
                            },
                            "recRow2": {
                                "id": "recRow2",
                                "cellValuesByColumnId": {
                                    ASANA_NAME_COLUMN_ID: "cust1/site1/2026-10-31_02",
                                },
                            },
                        }
                    }
                },
            },
        }
        airtable_map = parse_airtable_read_queries_payload(mock_airtable_payload)
        assert airtable_map["cust1/site1/2026-10-31_01"]["row_id"] == "recRow1"
        assert airtable_map["cust1/site1/2026-10-31_01"]["has_me_as_owner"] is True
        assert airtable_map["cust1/site1/2026-10-31_02"]["row_id"] == "recRow2"
        assert airtable_map["cust1/site1/2026-10-31_02"]["has_me_as_owner"] is False

    print("[TEST PASS] Все тесты метаданных, разделения серий, астрономии, initial.pto, RMS, фильтрации точек неба, Sanity Check и асинхронного S3-конвейера прошли успешно!")


if __name__ == "__main__":
    run_all_tests()
