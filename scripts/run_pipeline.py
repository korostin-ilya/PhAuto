#!/usr/bin/env python3
"""
Главный CLI-скрипт производственного пайплайна сборки сферических панорам БПЛА
(Hugin CLI + Multiblend) и адаптивной замены неба (Guided Filter + Smoothstep + Reinhard LAB).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.index_skies import build_sky_catalog, resolve_sky_dir
from src.metadata import (
    compute_sun_position,
    get_reference_exif_bytes,
    inspect_shoot_directory,
    load_series_metadata,
    write_output_metadata,
)
from src.sky_replacer.blender import blend_sky_with_panorama, save_panorama_jpeg
from src.sky_replacer.matcher import select_best_sky_template
from src.stitcher.hugin_runner import (
    load_prestitched_panorama_rgba,
    run_stitching_pipeline,
)


def discover_shoot_dirs(root_dir: str) -> list[str]:
    """
    Рекурсивно находит все конечные папки со съемками БПЛА внутри root_dir:
    поддерживает как папки с серией сырых снимков (>= 5 шт.), так и папки с уже сшитой
    панорамой без неба (для прогона только через Модуль 2).
    """
    shoot_dirs: list[str] = []
    for cur_root, _dirs, files in sorted(os.walk(root_dir)):
        jpgs = [
            f
            for f in files
            if f.lower().endswith((".jpg", ".jpeg")) and not f.startswith(".")
        ]
        if not jpgs:
            continue
        non_full_jpgs = [
            f
            for f in jpgs
            if not (
                os.path.splitext(f)[0].lower() == "full"
                or os.path.splitext(f)[0].lower().endswith("_full")
            )
        ]
        if len(non_full_jpgs) >= 1:
            shoot_dirs.append(os.path.abspath(cur_root))
    return shoot_dirs


def process_single_series(
    input_dir: str,
    output_jpg_path: str,
    sky_dir: str,
    catalog: dict[str, Any],
    config: dict[str, Any],
    work_dir: str | None = None,
    keep_temp: bool = False,
) -> dict[str, Any]:
    """Обрабатывает одну серию снимков БПЛА (или готовую панораму без неба) до панорамы 15000x7500."""
    t_total = time.time()
    pano_cfg = config.get("panorama", {})
    stitch_cfg = config.get("stitcher", {})
    filter_cfg = stitch_cfg.get("input_filter", {})
    match_cfg = config.get("sky_matcher", {})

    pano_width = int(pano_cfg.get("width", 15000))
    pano_height = int(pano_cfg.get("height", 7500))
    horizon_y = int(pano_cfg.get("horizon_y", 3750))
    jpeg_quality = int(pano_cfg.get("jpeg_quality", 96))
    subsampling = int(pano_cfg.get("subsampling", 0))
    default_hfov = float(stitch_cfg.get("default_hfov", 73.7))
    default_zenith_cutoff_y = int(filter_cfg.get("default_prestitched_zenith_cutoff_y", 2600))
    overcast_ev_thresh = float(match_cfg.get("overcast_ev_spread_threshold", 1.15))

    print(f"\n{'=' * 78}")
    print(f"[Пайплайн] Обработка серии: {input_dir}")
    print(f"[Пайплайн] Выходной файл:   {output_jpg_path}")
    print(f"{'=' * 78}")

    # 1. Чтение EXIF/XMP метаданных, фильтрация лишних панорам/дубликатов и расчет солнца
    t0 = time.time()
    inspected = inspect_shoot_directory(
        input_dir=input_dir,
        default_hfov=default_hfov,
        filter_cfg=filter_cfg,
    )
    mode = inspected["mode"]
    rep = inspected["filter_report"]

    if mode == "empty":
        raise ValueError(
            f"В папке '{input_dir}' не найдено ни сырых кадров БПЛА, ни сшитой панорамы без неба."
        )

    if rep["stitched_panorama_files"] or rep["rejected_duplicates"] or rep["rejected_no_gimbal"] or rep["rejected_resolution"]:
        print(
            f"[Фильтр входа]  Всего JPEG={rep['total_files']} -> Сырых кадров={rep['kept_raw_count']} | "
            f"Отсечено готовых панорам={len(rep['stitched_panorama_files'])} {rep['stitched_panorama_files']} | "
            f"Дубликатов={len(rep['rejected_duplicates'])} {rep['rejected_duplicates']}"
        )

    if mode == "stitch":
        frames_meta = inspected["raw_frames"]
    else:
        frames_meta = [inspected["prestitched_pano"]]

    sun_info = compute_sun_position(
        frames_meta=frames_meta,
        pano_width=pano_width,
        horizon_y=horizon_y,
        overcast_ev_threshold=overcast_ev_thresh,
    )
    lat_val = sun_info["latitude"] if sun_info["latitude"] is not None else 0.0
    lon_val = sun_info["longitude"] if sun_info["longitude"] is not None else 0.0
    print(
        f"[Шаг 1.1 & 2.2] Режим={mode.upper()} | Кадров={len(frames_meta)} | Камера: {frames_meta[0]['make']} {frames_meta[0]['model']} | "
        f"HFOV={frames_meta[0]['hfov']:.2f}° ({time.time() - t0:.2f}s)"
    )
    print(
        f"[Астрономия]    GPS=({lat_val:.5f}, {lon_val:.5f}) | "
        f"Таймзона={sun_info['timezone']} | Местное время={sun_info['datetime_local']}"
    )
    print(
        f"[Солнце]        Азимут={sun_info['azimuth_deg']:.2f}°, Высота={sun_info['elevation_deg']:.2f}° -> "
        f"X_sun={sun_info['x_sun_pano']:.1f}px, Y_sun={sun_info['y_sun_pano_clamped']:.1f}px | "
        f"dEV100={sun_info['delta_ev100_upper']:.2f}"
    )

    # 2. Модуль 1: Сборка панорамы (Hugin CLI + Multiblend) ИЛИ загрузка готовой панорамы без неба
    t_stitch = time.time()
    if mode == "stitch":
        print("[Модуль 1]      Запуск сшивки панорамы (cpfind -> cpclean -> autooptimiser -> vig_optimize -> nona -> multiblend)...")
        pano_bgr, alpha_mask, ordered_frames, stitch_stats = run_stitching_pipeline(
            frames_meta=frames_meta,
            config=config,
            work_dir=work_dir,
            keep_temp=keep_temp,
        )
        tm = stitch_stats.get("timings", {})
        tm_str = (
            f" (cpfind={tm['cpfind']}s, opt={tm['opt']}s, nona={tm['nona']}s, multiblend={tm['multiblend']}s)"
            if tm
            else ""
        )
        print(
            f"[Модуль 1]      Сшивка завершена за {time.time() - t_stitch:.1f}s{tm_str} | "
            f"Опорный кадр: {stitch_stats['anchor_frame']} (pitch={stitch_stats['anchor_pitch']}°, yaw={stitch_stats['anchor_yaw']}°) | "
            f"Точки: всего={stitch_stats['sky_cp_stats']['total_cps']}, удалено на небе={stitch_stats['sky_cp_stats']['removed_sky_cps']} | "
            f"Sanity reset кадров: {len(stitch_stats['sanity_check']['reset_frames'])}"
        )
    else:
        base_pano = inspected["prestitched_pano"]
        pano_bgr, alpha_mask, pre_info = load_prestitched_panorama_rgba(
            pano_path=base_pano["path"],
            pano_width=pano_width,
            pano_height=pano_height,
            default_zenith_cutoff_y=default_zenith_cutoff_y,
        )
        stitch_stats = {
            "mode": "sky_only_prestitched",
            "prestitched_info": pre_info,
        }
        print(
            f"[Модуль 1]      Сшивка пропущена (используется готовая панорама '{pre_info['source_filename']}', "
            f"исходный размер={pre_info['original_size'][0]}x{pre_info['original_size'][1]} -> {pano_width}x{pano_height}, "
            f"маска зенита={pre_info['zenith_mask_mode']}) за {time.time() - t_stitch:.2f}s"
        )

    # 3. Модуль 2 (Шаги 2.3): Двухступенчатый автоподбор неба
    t_match = time.time()
    match_res = select_best_sky_template(
        pano_bgr=pano_bgr,
        alpha_mask=alpha_mask,
        sun_info=sun_info,
        catalog=catalog,
        config=config,
        sky_base_dir=sky_dir,
    )
    winner = match_res["selected_template"]
    print(
        f"[Модуль 2.3]    Выбран шаблон неба: '{winner['relative_path']}' "
        f"(combined={winner['combined_score']:.2f}, stage1={winner['stage1']['total_stage1']:.2f}, "
        f"stage2_seam={winner['stage2']['micro_total']:.2f}) за {time.time() - t_match:.2f}s"
    )

    # 4. Модуль 2 (Шаг 2.4): Защита высотных объектов (Guided Filter) и градиентный блендинг
    t_blend = time.time()
    pano_bgr, blend_stats = blend_sky_with_panorama(
        pano_bgr=pano_bgr,
        alpha_mask=alpha_mask,
        selected_sky_path=winner["full_path"],
        x_sun_pano=sun_info["x_sun_pano"],
        config=config,
    )
    print(
        f"[Модуль 2.4]    Блендинг и защита объектов выполнены за {time.time() - t_blend:.2f}s | "
        f"Сдвиг солнца={blend_stats['shift_px']}px | Верхняя кромка дрона Y_top={blend_stats['top_edge_y_mean']}px | "
        f"Защищено объектов в зоне перехода={blend_stats['protected_tall_objects_ratio'] * 100:.1f}%"
    )

    # 5. Сохранение JPEG 15000x7500 (4:4:4, quality=96) и запись метаданных EXIF/GPS + GPano XMP
    t_save = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(output_jpg_path)), exist_ok=True)
    ref_frame_path = frames_meta[0]["path"]
    exif_bytes = get_reference_exif_bytes(
        reference_jpg_path=ref_frame_path,
        full_width=pano_width,
        full_height=pano_height,
    )
    save_panorama_jpeg(
        pano_bgr=pano_bgr,
        output_path=output_jpg_path,
        quality=jpeg_quality,
        subsampling=subsampling,
        exif_bytes=exif_bytes,
    )
    write_output_metadata(
        output_jpg_path=output_jpg_path,
        reference_jpg_path=ref_frame_path,
        full_width=pano_width,
        full_height=pano_height,
        pose_heading_deg=0.0,
    )
    total_elapsed = time.time() - t_total
    print(
        f"[Готово]        Панорама сохранена в '{output_jpg_path}' ({pano_width}x{pano_height}, 4:4:4, GPano XMP) "
        f"| Запись={time.time() - t_save:.2f}s | Общее время={total_elapsed:.1f}s"
    )

    return {
        "input_dir": input_dir,
        "output_path": output_jpg_path,
        "sun_info": sun_info,
        "stitch_stats": stitch_stats,
        "selected_sky": winner,
        "blend_stats": blend_stats,
        "elapsed_sec": round(total_elapsed, 2),
    }


def default_output_path_for_shoot(shoot_dir: str, config: dict[str, Any] | None = None) -> str:
    """
    Формирует путь к выходному файлу внутри самой папки со снимками:
    <папка>/<папка>_full.jpeg
    """
    pano_cfg = (config or {}).get("panorama", {})
    suffix = str(pano_cfg.get("output_suffix", "_full.jpeg"))
    abs_dir = os.path.abspath(shoot_dir)
    folder_name = os.path.basename(abs_dir)
    if folder_name.lower() == "input" and os.environ.get("HOST_FOLDER_NAME"):
        folder_name = os.environ["HOST_FOLDER_NAME"]
    return os.path.join(abs_dir, f"{folder_name}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Автоматизированный пайплайн сборки сферических панорам БПЛА и адаптивной замены неба"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Путь к папке с серией снимков БПЛА (или корневой папке с несколькими сериями)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help=(
            "Необязательный путь к выходному JPEG-файлу или папке. "
            "По умолчанию сохраняет <папка>_full.jpeg прямо в папку со снимками серии."
        ),
    )
    parser.add_argument(
        "--sky-dir",
        type=str,
        default=None,
        help="Путь к папке с шаблонами неба (по умолчанию sky_templates/ или sky/)",
    )
    parser.add_argument(
        "--catalog",
        type=str,
        default=os.path.join(PROJECT_ROOT, "sky_catalog.json"),
        help="Путь к индексу шаблонов неба sky_catalog.json",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_ROOT, "config.yaml"),
        help="Путь к конфигурационному файлу config.yaml",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Директория для промежуточных файлов сшивки (.pto, тайлы)",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Не удалять временные файлы сшивки (.pto, stitched_base.tif)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Перезаписывать уже существующие итоговые панорамы (<папка>_full.jpeg)",
    )
    args = parser.parse_args()

    # Загружаем конфигурацию
    config: dict[str, Any] = {}
    if os.path.isfile(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    # Определяем входную директорию
    input_target = args.input
    if not input_target:
        for cand in ("/app/input", os.path.join(PROJECT_ROOT, "pano")):
            if os.path.isdir(cand):
                input_target = cand
                break
    if not input_target or not os.path.isdir(input_target):
        raise FileNotFoundError(
            f"Входная директория не найдена: {input_target}. Укажите --input <путь_к_папке>."
        )

    # Индексируем или загружаем каталог неба
    sky_dir = resolve_sky_dir(args.sky_dir)
    if os.path.isfile(args.catalog):
        with open(args.catalog, "r", encoding="utf-8") as f:
            catalog = json.load(f)
    else:
        catalog = build_sky_catalog(sky_dir=sky_dir, catalog_path=args.catalog, config=config)

    shoot_dirs = discover_shoot_dirs(input_target)
    if not shoot_dirs:
        raise FileNotFoundError(f"Внутри '{input_target}' не найдено ни одной серии JPEG-снимков.")

    pano_cfg = config.get("panorama", {})
    skip_existing = bool(pano_cfg.get("skip_existing_output", True)) and not args.overwrite

    if len(shoot_dirs) == 1 and args.output and args.output.lower().endswith((".jpg", ".jpeg")):
        if skip_existing and os.path.isfile(args.output) and os.path.getsize(args.output) > 100_000:
            print(f"[Пропуск] Итоговый файл уже существует: '{args.output}' (используйте --overwrite для пересборки)")
            return
        process_single_series(
            input_dir=shoot_dirs[0],
            output_jpg_path=args.output,
            sky_dir=sky_dir,
            catalog=catalog,
            config=config,
            work_dir=args.work_dir,
            keep_temp=args.keep_temp,
        )
    else:
        suffix = str(pano_cfg.get("output_suffix", "_full.jpeg"))
        out_root = args.output
        if out_root and out_root.lower().endswith((".jpg", ".jpeg")):
            out_root = os.path.dirname(os.path.abspath(out_root))
        if out_root:
            os.makedirs(out_root, exist_ok=True)

        for idx, s_dir in enumerate(shoot_dirs, start=1):
            rel_name = os.path.relpath(s_dir, input_target).replace("\\", "_").replace("/", "_")
            if rel_name in (".", ""):
                rel_name = os.path.basename(os.path.abspath(s_dir))

            if out_root:
                out_file = os.path.join(out_root, f"{rel_name}{suffix}")
            else:
                # Сохраняем прямо в папку со снимками под именем <папка>_full.jpeg
                out_file = default_output_path_for_shoot(s_dir, config=config)

            if skip_existing and os.path.isfile(out_file) and os.path.getsize(out_file) > 100_000:
                print(
                    f"\n>>> Пропуск [{idx}/{len(shoot_dirs)}]: итоговый файл уже существует ({out_file})"
                )
                continue

            sub_work = os.path.join(args.work_dir, rel_name) if args.work_dir else None
            print(f"\n>>> Обработка [{idx}/{len(shoot_dirs)}]: {s_dir} -> {out_file}")
            process_single_series(
                input_dir=s_dir,
                output_jpg_path=out_file,
                sky_dir=sky_dir,
                catalog=catalog,
                config=config,
                work_dir=sub_work,
                keep_temp=args.keep_temp,
            )


if __name__ == "__main__":
    main()
