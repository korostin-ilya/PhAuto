"""Запуск утилит Hugin CLI (cpfind, cpclean, pto_var, autooptimiser, vig_optimize, nona) и Multiblend."""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any

import cv2
import numpy as np

from src.stitcher.pto_generator import (
    clamp_photometric_pto,
    enforce_angle_sanity_check,
    filter_sky_control_points,
    generate_initial_pto,
    order_frames_with_horizon_anchor,
)


def _find_tool(tool_name: str, extra_dirs: list[str] | None = None) -> str:
    """Ищет исполняемый файл утилиты в PATH, локальной папке bin/ и стандартных директориях."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    search_dirs: list[str] = []
    if extra_dirs:
        search_dirs.extend(extra_dirs)
    search_dirs.extend(
        [
            os.path.join(project_root, "bin"),
            os.path.join(project_root, "bin", "Hugin", "bin"),
            os.path.join(project_root, "bin", "hugin"),
            os.path.join(project_root, "bin", "multiblend"),
            r"C:\Program Files\Hugin\bin",
            os.path.expanduser(r"~\scoop\apps\hugin\current\bin"),
            os.path.expanduser(r"~\scoop\shims"),
        ]
    )

    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for ext in ("", ".exe", ".bat", ".cmd"):
            cand = os.path.join(d, tool_name + ext)
            if os.path.isfile(cand):
                return cand

    found = shutil.which(tool_name)
    if found:
        return found

    raise FileNotFoundError(
        f"Утилита '{tool_name}' не найдена ни в системном PATH, ни в {os.path.join(project_root, 'bin')}."
    )


def _run_cmd(cmd: list[str], cwd: str | None = None, step_name: str = "") -> subprocess.CompletedProcess:
    """Выполняет внешнюю команду с проверкой кода возврата и информативным выводом при ошибке."""
    res = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"Ошибка на шаге '{step_name or cmd[0]}' (код {res.returncode}):\n"
            f"Команда: {' '.join(cmd)}\n"
            f"STDOUT: {res.stdout[-1500:]}\n"
            f"STDERR: {res.stderr[-1500:]}"
        )
    return res


def load_stitched_canvas_rgba(
    stitched_tif_path: str,
    pano_width: int = 15000,
    pano_height: int = 7500,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Читает сшитый TIFF (stitched_base.tif) и восстанавливает полный холст 15000x7500:
    - Возвращает (pano_bgr, alpha_mask), где:
      pano_bgr: uint8 массив (7500, 15000, 3) в формате BGR
      alpha_mask: uint8 массив (7500, 15000), где 255 — реальное покрытие кадров дрона, 0 — пустота зенита.
    Учитывает то, что multiblend обрезает пустые верхние полосы зенита (min_ypos) и записывает
    TIFF меньшей высоты, прижатый к надиру (Y = 7500) или со смещением TIFFTAG_YPOSITION.
    """
    y_offset: int | None = None
    x_offset: int = 0
    arr: np.ndarray | None = None

    try:
        import tifffile

        with tifffile.TiffFile(stitched_tif_path) as tif:
            page = tif.pages[0]
            tags = page.tags
            if "YPosition" in tags and "YResolution" in tags:
                y_pos = tags["YPosition"].value
                y_res = tags["YResolution"].value
                if isinstance(y_pos, tuple):
                    y_pos = y_pos[0] / y_pos[1] if y_pos[1] else 0.0
                if isinstance(y_res, tuple):
                    y_res = y_res[0] / y_res[1] if y_res[1] else 1.0
                if y_res > 0 and y_pos >= 0:
                    y_offset = int(round(float(y_pos) * float(y_res)))
            if "XPosition" in tags and "XResolution" in tags:
                x_pos = tags["XPosition"].value
                x_res = tags["XResolution"].value
                if isinstance(x_pos, tuple):
                    x_pos = x_pos[0] / x_pos[1] if x_pos[1] else 0.0
                if isinstance(x_res, tuple):
                    x_res = x_res[0] / x_res[1] if x_res[1] else 1.0
                if x_res > 0 and x_pos >= 0:
                    x_offset = int(round(float(x_pos) * float(x_res)))
            raw = page.asarray()
            # tifffile возвращает RGB или RGBA
            if raw.ndim == 3 and raw.shape[2] == 4:
                arr = cv2.cvtColor(raw, cv2.COLOR_RGBA2BGRA)
            elif raw.ndim == 3 and raw.shape[2] == 3:
                arr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
    except Exception:
        arr = None

    if arr is None:
        buf = np.fromfile(stitched_tif_path, dtype=np.uint8)
        arr = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise RuntimeError(f"Не удалось прочитать сшитый файл {stitched_tif_path}")

    if arr.dtype == np.uint16:
        arr = (arr >> 8).astype(np.uint8)

    h_stitched, w_stitched = arr.shape[:2]

    # Если высота меньше 7500, а надир заполнен — вычисляем смещение от нижнего края
    if y_offset is None or y_offset + h_stitched > pano_height or (y_offset == 0 and h_stitched < pano_height):
        y_offset = max(0, pano_height - h_stitched)
    if x_offset + w_stitched > pano_width:
        x_offset = 0

    pano_bgr = np.zeros((pano_height, pano_width, 3), dtype=np.uint8)
    alpha_mask = np.zeros((pano_height, pano_width), dtype=np.uint8)

    y_end = min(pano_height, y_offset + h_stitched)
    x_end = min(pano_width, x_offset + w_stitched)
    crop_h = y_end - y_offset
    crop_w = x_end - x_offset

    if arr.ndim == 3 and arr.shape[2] == 4:
        pano_bgr[y_offset:y_end, x_offset:x_end] = arr[:crop_h, :crop_w, :3]
        alpha_mask[y_offset:y_end, x_offset:x_end] = arr[:crop_h, :crop_w, 3]
    else:
        pano_bgr[y_offset:y_end, x_offset:x_end] = arr[:crop_h, :crop_w, :3]
        # Если альфа-канал отсутствует, считаем непустыми нечерные пиксели
        non_empty = np.any(arr[:crop_h, :crop_w, :3] > 0, axis=2)
        alpha_mask[y_offset:y_end, x_offset:x_end] = np.where(non_empty, 255, 0).astype(np.uint8)

    return pano_bgr, alpha_mask


def load_prestitched_panorama_rgba(
    pano_path: str,
    pano_width: int = 15000,
    pano_height: int = 7500,
    default_zenith_cutoff_y: int = 2600,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Загружает уже сшитую панораму без неба (например, 1.jpg 15000x7500 или DJI_*_V.JPG 14400x7200),
    приводит её к целевому разрешению 15000x7500 и строит маску реального покрытия alpha_mask:
    - Если верхняя часть зенита черная (Autopano Giga), граница покрытия определяется по первому
      нечерному пикселю в каждом столбце.
    - Если верхняя часть зенита залита синтетическим градиентом (внутрикамерная сшивка DJI 14400x7200),
      зона выше default_zenith_cutoff_y помечается как alpha_mask = 0 для замены на четкий шаблон неба.
    """
    buf = np.fromfile(pano_path, dtype=np.uint8)
    arr = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"Не удалось прочитать готовую панораму: {pano_path}")

    if arr.dtype == np.uint16:
        arr = (arr >> 8).astype(np.uint8)

    orig_h, orig_w = arr.shape[:2]
    resized = False
    if (orig_w, orig_h) != (pano_width, pano_height):
        arr = cv2.resize(arr, (pano_width, pano_height), interpolation=cv2.INTER_CUBIC)
        resized = True

    horizon_y = pano_height // 2

    if arr.ndim == 3 and arr.shape[2] == 4:
        pano_bgr = arr[:, :, :3].copy()
        alpha_mask = arr[:, :, 3].copy()
        zenith_mode = "embedded_alpha"
    else:
        pano_bgr = arr[:, :, :3] if arr.ndim == 3 else cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        alpha_mask = np.full((pano_height, pano_width), 255, dtype=np.uint8)

        top_strip_mean = float(np.mean(pano_bgr[:40, :, :]))
        if top_strip_mean < 8.0:
            # Заготовка с черным неснятым зенитом (например, Autopano Giga 1.jpg)
            upper_non_black = np.max(pano_bgr[:horizon_y, :, :], axis=2) > 8
            has_valid = np.any(upper_non_black, axis=0)
            first_valid_y = np.argmax(upper_non_black, axis=0)
            first_valid_y = np.where(has_valid, first_valid_y, horizon_y)
            y_grid = np.arange(horizon_y, dtype=np.int32).reshape(horizon_y, 1)
            alpha_upper = np.where(y_grid >= first_valid_y.reshape(1, pano_width), 255, 0).astype(
                np.uint8
            )
            alpha_mask[:horizon_y, :] = alpha_upper
            zenith_mode = "black_void_detected"
        else:
            # Внутрикамерная сшивка дрона с искусственно размытым зенитом (например, DJI_*_V.JPG 14400x7200)
            min_margin = max(10, horizon_y // 35)
            cutoff_y = max(0, min(horizon_y - min_margin, int(default_zenith_cutoff_y)))
            alpha_mask[:cutoff_y, :] = 0
            zenith_mode = f"synthetic_zenith_cutoff_{cutoff_y}px"

    info = {
        "source_path": pano_path,
        "source_filename": os.path.basename(pano_path),
        "original_size": (orig_w, orig_h),
        "target_size": (pano_width, pano_height),
        "was_resized": resized,
        "zenith_mask_mode": zenith_mode,
    }
    return pano_bgr, alpha_mask, info


def run_stitching_pipeline(
    frames_meta: list[dict[str, Any]],
    config: dict[str, Any],
    work_dir: str | None = None,
    keep_temp: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """
    Выполняет полный цикл Модуля 1 (Сборка панорамы):
    1.1 Динамический расчет HFOV и генерация initial.pto с опорным кадром у горизонта.
    1.2 Поиск контрольных точек cpfind --prealigned --fullscale.
    1.3 Удаление точек неба выше линии горизонта + фильтрация параллакса cpclean -n 1.5.
    1.4 Оптимизация геометрии autooptimiser + Sanity Check (>3.5° -> откат к RTK).
    1.5 Фотометрическая коррекция vig_optimize + репроекция nona + сшивка multiblend.
    Возвращает (pano_bgr, alpha_mask, ordered_frames_meta, stitch_stats).
    """
    pano_cfg = config.get("panorama", {})
    stitch_cfg = config.get("stitcher", {})

    pano_width = int(pano_cfg.get("width", 15000))
    pano_height = int(pano_cfg.get("height", 7500))
    pitch_thresh = float(stitch_cfg.get("sky_filter_pitch_threshold_deg", -26.0))
    horizon_margin = float(stitch_cfg.get("sky_horizon_margin_px", 15.0))
    cpclean_thresh = float(stitch_cfg.get("cpclean_threshold", 1.5))
    max_angle_dev = float(stitch_cfg.get("max_angle_deviation_deg", 3.5))
    max_eev_dev = float(stitch_cfg.get("max_eev_deviation", 2.5))
    use_gpu_nona = bool(stitch_cfg.get("use_gpu_nona", False))
    use_vig_optimize = bool(stitch_cfg.get("use_vig_optimize", False))
    use_cropped_tiff = bool(stitch_cfg.get("use_cropped_tiff", True))

    ordered_frames = order_frames_with_horizon_anchor(frames_meta)

    created_temp = False
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="drone_pano_stitch_")
        created_temp = True
    else:
        os.makedirs(work_dir, exist_ok=True)

    try:
        # Находим бинарные утилиты
        cpfind_bin = _find_tool("cpfind")
        cpclean_bin = _find_tool("cpclean")
        pto_var_bin = _find_tool("pto_var")
        autooptimiser_bin = _find_tool("autooptimiser")
        vig_optimize_bin = _find_tool("vig_optimize")
        nona_bin = _find_tool("nona")
        try:
            multiblend_bin = _find_tool("multiblend")
            use_multiblend = True
        except FileNotFoundError:
            multiblend_bin = _find_tool("enblend")
            use_multiblend = False

        # На Windows Hugin может не любить кириллицу в путях к исходным кадрам,
        # поэтому создаем симлинки/копии с ASCII-именами frame_0000.jpg в рабочей директории
        local_frames: list[dict[str, Any]] = []
        for idx, fm in enumerate(ordered_frames):
            fm_copy = dict(fm)
            local_img_path = os.path.join(work_dir, f"frame_{idx:04d}.jpg")
            if not os.path.exists(local_img_path):
                try:
                    os.symlink(os.path.abspath(fm["path"]), local_img_path)
                except Exception:
                    try:
                        os.link(fm["path"], local_img_path)
                    except Exception:
                        shutil.copy2(fm["path"], local_img_path)
            fm_copy["path"] = local_img_path
            local_frames.append(fm_copy)

        # Шаг 1.1: Генерация initial.pto
        initial_pto = os.path.join(work_dir, "initial.pto")
        generate_initial_pto(
            local_frames,
            initial_pto,
            pano_width=pano_width,
            pano_height=pano_height,
            use_relative_paths=True,
            use_cropped_tiff=use_cropped_tiff,
        )

        # Шаг 1.2: Поиск контрольных точек между соседями (cpfind)
        t_cp0 = time.time()
        cp_raw_pto = os.path.join(work_dir, "cp_raw.pto")
        cpfind_args = stitch_cfg.get("cpfind_args", ["--prealigned"])
        _run_cmd(
            [cpfind_bin, *cpfind_args, "-o", "cp_raw.pto", "initial.pto"],
            cwd=work_dir,
            step_name="cpfind",
        )
        t_cpfind = time.time() - t_cp0

        # Шаг 1.3.1: Удаление контрольных точек неба в Python
        t_opt0 = time.time()
        cp_no_sky_pto = os.path.join(work_dir, "cp_no_sky.pto")
        sky_cp_stats = filter_sky_control_points(
            input_pto_path=cp_raw_pto,
            output_pto_path=cp_no_sky_pto,
            frames_meta=local_frames,
            pitch_threshold_deg=pitch_thresh,
            horizon_margin_px=horizon_margin,
        )

        # Шаг 1.3.2 & 1.4: Двухэтапная геометрическая оптимизация и отсечение выбросов:
        # Этап 1: Жесткое 3D-выравнивание ориентаций (y, p, r) и глобального угла обзора (v)
        # ДО запуска cpclean, чтобы сырая погрешность подвеса в надире (-90°) не приводила к удалению
        # валидных дорожных точек в cpclean.
        opt_pass1_vars_pto = os.path.join(work_dir, "opt_pass1_vars.pto")
        opt_pass1_pto = os.path.join(work_dir, "opt_pass1.pto")
        cp_clean_pto = os.path.join(work_dir, "cp_clean.pto")
        opt_vars_pto = os.path.join(work_dir, "opt_vars.pto")
        opt_geom_pto = os.path.join(work_dir, "opt_geom.pto")
        opt_checked_pto = os.path.join(work_dir, "opt_geom_checked.pto")

        _run_cmd(
            [pto_var_bin, "--opt=y,p,r,v,!y0,!p0,!r0", "-o", "opt_pass1_vars.pto", "cp_no_sky.pto"],
            cwd=work_dir,
            step_name="pto_var (pass 1 rigid + hfov)",
        )
        _run_cmd(
            [autooptimiser_bin, "-n", "-o", "opt_pass1.pto", "opt_pass1_vars.pto"],
            cwd=work_dir,
            step_name="autooptimiser (pass 1)",
        )

        # Отсечение параллакса и случайных выбросов (cpclean) по уже выровненной 3D-сфере
        if sky_cp_stats["kept_cps"] >= 6:
            _run_cmd(
                [cpclean_bin, "-n", str(max(cpclean_thresh, 1.8)), "-o", "cp_clean.pto", "opt_pass1.pto"],
                cwd=work_dir,
                step_name="cpclean",
            )
        else:
            shutil.copy2(opt_pass1_pto, cp_clean_pto)

        # Этап 2: Точная совместная калибровка полинома дисторсии (a, b, c), сдвига центра (d, e) и углов (y, p, r, v)
        _run_cmd(
            [pto_var_bin, "--opt=y,p,r,v,a,b,c,d,e,!y0,!p0,!r0", "-o", "opt_vars.pto", "cp_clean.pto"],
            cwd=work_dir,
            step_name="pto_var (pass 2 full lens + geometry)",
        )
        _run_cmd(
            [autooptimiser_bin, "-n", "-o", "opt_geom.pto", "opt_vars.pto"],
            cwd=work_dir,
            step_name="autooptimiser (pass 2)",
        )

        sanity_stats = enforce_angle_sanity_check(
            input_pto_path=opt_geom_pto,
            output_pto_path=opt_checked_pto,
            frames_meta=local_frames,
            max_deviation_deg=max_angle_dev,
        )
        if sanity_stats.get("lens_degenerate", False):
            # Если полином дисторсии выродился на малоконтрастной сцене, откатываемся к чистому решению Этапа 1 (a=b=c=d=e=0)
            sanity_stats = enforce_angle_sanity_check(
                input_pto_path=cp_clean_pto,
                output_pto_path=opt_checked_pto,
                frames_meta=local_frames,
                max_deviation_deg=max_angle_dev,
            )

        # Шаг 1.5.1: Фотометрическая коррекция (vig_optimize — опционально, т.к. multiblend --wideblend сам выравнивает тон)
        photo_vars_pto = os.path.join(work_dir, "photo_vars.pto")
        final_raw_pto = os.path.join(work_dir, "final_raw.pto")
        final_pto = os.path.join(work_dir, "final.pto")

        if use_vig_optimize:
            _run_cmd(
                [
                    pto_var_bin,
                    "--opt=Eev,Vb,Vc,Vd,!Eev0",
                    "-o",
                    "photo_vars.pto",
                    "opt_geom_checked.pto",
                ],
                cwd=work_dir,
                step_name="pto_var (photometric)",
            )
            try:
                _run_cmd(
                    [vig_optimize_bin, "-o", "final_raw.pto", "photo_vars.pto"],
                    cwd=work_dir,
                    step_name="vig_optimize",
                )
                clamp_photometric_pto(final_raw_pto, final_pto, max_eev_deviation=max_eev_dev)
            except Exception:
                shutil.copy2(opt_checked_pto, final_pto)
        else:
            shutil.copy2(opt_checked_pto, final_pto)

        # Гарантируем, что после перезаписи .pto утилитами pto_var/autooptimiser в p-строке сохранен r:CROP и c:NONE
        if use_cropped_tiff:
            with open(final_pto, "r", encoding="utf-8", errors="ignore") as f:
                pto_text = f.read()
            pto_text = re.sub(
                r'^p\s+([^\n]*?)n"TIFF_m[^"\n]*"',
                r'p \1n"TIFF_m c:NONE r:CROP"',
                pto_text,
                count=1,
                flags=re.MULTILINE,
            )
            with open(final_pto, "w", encoding="utf-8", newline="\n") as f:
                f.write(pto_text)
        t_opt = time.time() - t_opt0

        # Шаг 1.5.2: Репроекция (nona)
        t_nona0 = time.time()
        if use_cropped_tiff:
            nona_cmd_base = [nona_bin, "-o", "tile_", "final.pto"]
            nona_cmd_gpu = [nona_bin, "-g", "-o", "tile_", "final.pto"]
        else:
            nona_cmd_base = [nona_bin, "-m", "TIFF_m", "-z", "LZW", "-o", "tile_", "final.pto"]
            nona_cmd_gpu = [nona_bin, "-g", "-m", "TIFF_m", "-z", "LZW", "-o", "tile_", "final.pto"]

        if use_gpu_nona:
            try:
                _run_cmd(
                    nona_cmd_gpu,
                    cwd=work_dir,
                    step_name="nona (GPU)",
                )
            except Exception:
                _run_cmd(nona_cmd_base, cwd=work_dir, step_name="nona (CPU fallback)")
        else:
            _run_cmd(nona_cmd_base, cwd=work_dir, step_name="nona")
        t_nona = time.time() - t_nona0

        tile_files = sorted(glob.glob(os.path.join(work_dir, "tile_*.tif")))
        if not tile_files:
            raise RuntimeError("nona не создала ни одного тайла tile_*.tif")
        tile_names = [os.path.basename(t) for t in tile_files]

        # Шаг 1.5.3: Сшивка разрезом графов (multiblend)
        t_mb0 = time.time()
        stitched_tif = os.path.join(work_dir, "stitched_base.tif")
        if use_multiblend:
            mb_args = stitch_cfg.get(
                "multiblend_args",
                ["--wrap", "--wideblend", "--compression=LZW", "--all-threads"],
            )
            _run_cmd(
                [multiblend_bin, *mb_args, "-o", "stitched_base.tif", *tile_names],
                cwd=work_dir,
                step_name="multiblend",
            )
        else:
            _run_cmd(
                [multiblend_bin, "--wrap=horizontal", "-o", "stitched_base.tif", *tile_names],
                cwd=work_dir,
                step_name="enblend",
            )
        t_mb = time.time() - t_mb0

        t_load0 = time.time()
        pano_bgr, alpha_mask = load_stitched_canvas_rgba(
            stitched_tif_path=stitched_tif,
            pano_width=pano_width,
            pano_height=pano_height,
        )
        t_load = time.time() - t_load0

        stats = {
            "num_frames": len(ordered_frames),
            "anchor_frame": ordered_frames[0]["filename"],
            "anchor_pitch": ordered_frames[0]["gimbal_pitch"],
            "anchor_yaw": ordered_frames[0]["gimbal_yaw"],
            "hfov_initial": ordered_frames[0]["hfov"],
            "sky_cp_stats": sky_cp_stats,
            "sanity_check": sanity_stats,
            "num_tiles": len(tile_names),
            "timings": {
                "cpfind": round(t_cpfind, 1),
                "opt": round(t_opt, 1),
                "nona": round(t_nona, 1),
                "multiblend": round(t_mb, 1),
                "load_tif": round(t_load, 1),
            },
        }
        return pano_bgr, alpha_mask, ordered_frames, stats

    finally:
        if created_temp and not keep_temp and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
