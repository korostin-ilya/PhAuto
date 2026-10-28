"""Генерация initial.pto, динамический расчет HFOV, фильтрация точек неба, 3D-выравнивание горизонта и Sanity Check углов."""

from __future__ import annotations

import math
import os
import re
from typing import Any

import numpy as np


def _wrap_angle_diff(angle_deg: float, ref_deg: float) -> float:
    """Возвращает разницу углов (angle_deg - ref_deg) в диапазоне [-180, +180]."""
    return ((angle_deg - ref_deg + 180.0) % 360.0) - 180.0


def _wrap_yaw(yaw_deg: float) -> float:
    """Нормализует угол рысканья в диапазон [-180, +180]."""
    return ((yaw_deg + 180.0) % 360.0) - 180.0


def _euler_to_rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """
    Строит 3x3 матрицу 3D-поворота камеры R = Ry(yaw) * Rx(pitch) * Rz(roll).
    Использование матрицы поворота устраняет сингулярность углов Эйлера (Gimbal Lock)
    вблизи надира (pitch = -90°), где малый наклон оптической оси на 1° вызывает
    скачок yaw и roll на ±60°..180° при неизменной физической ориентации кадра.
    """
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64)
    rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return ry @ rx @ rz


def _rotation_matrix_to_euler(rot: np.ndarray) -> tuple[float, float, float]:
    """
    Извлекает углы Эйлера (yaw_deg, pitch_deg, roll_deg) из 3x3 матрицы поворота
    R = Ry(yaw) * Rx(pitch) * Rz(roll).
    """
    sp = max(-1.0, min(1.0, float(-rot[1, 2])))
    p_rad = math.asin(sp)
    cp = math.cos(p_rad)
    if abs(cp) > 1e-5:
        y_rad = math.atan2(float(rot[0, 2]), float(rot[2, 2]))
        r_rad = math.atan2(float(rot[1, 0]), float(rot[1, 1]))
    else:
        y_rad = math.atan2(float(-rot[2, 0]), float(rot[0, 0]))
        r_rad = 0.0
    return (
        _wrap_yaw(math.degrees(y_rad)),
        max(-90.0, min(90.0, math.degrees(p_rad))),
        _wrap_yaw(math.degrees(r_rad)),
    )


def _rotation_angle_diff_deg(
    y1: float, p1: float, r1: float, y2: float, p2: float, r2: float
) -> float:
    """Вычисляет истинный геодезический угол 3D-поворота (в градусах) между двумя ориентациями камеры."""
    rot1 = _euler_to_rotation_matrix(y1, p1, r1)
    rot2 = _euler_to_rotation_matrix(y2, p2, r2)
    trace_val = float(np.trace(rot1.T @ rot2))
    cos_theta = max(-1.0, min(1.0, (trace_val - 1.0) * 0.5))
    return math.degrees(math.acos(cos_theta))


def order_frames_with_horizon_anchor(frames_meta: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Переупорядочивает список кадров так, чтобы первым (index 0) шел наиболее стабильный
    опорный снимок около горизонта (предпочтительно ряд -26°..+10°) с минимальным
    рассогласованием курса подвеса/корпуса, минимальным креном дрона от ветра и минимальной линейной скоростью.
    При этом Север (Yaw = 0°) всегда остается строго в центре панорамы (X = 7500).
    """
    if not frames_meta:
        return []

    def anchor_score(idx_and_frame: tuple[int, dict[str, Any]]) -> tuple[float, float, int]:
        idx, f = idx_and_frame
        pitch = float(f.get("gimbal_pitch", 0.0))
        yaw = _wrap_yaw(float(f.get("gimbal_yaw", 0.0)))
        fl_yaw = _wrap_yaw(float(f.get("flight_yaw", yaw)))
        fl_pitch = float(f.get("flight_pitch", 0.0))
        fl_roll = float(f.get("flight_roll", 0.0))
        speed = float(f.get("linear_speed", 0.0))
        yaw_lag = abs(_wrap_angle_diff(yaw, fl_yaw))
        body_tilt = math.hypot(fl_pitch, fl_roll)
        # Предпочитаем кадры у горизонта (-28° <= pitch <= +10°) с минимальным доворотом подвеса
        pitch_penalty = abs(pitch + 15.0) if -28.0 <= pitch <= 10.0 else abs(pitch) + 40.0
        stability_penalty = pitch_penalty + 6.0 * yaw_lag + 1.5 * body_tilt + 0.5 * speed
        return (stability_penalty, abs(yaw), idx)

    best_idx, _ = min(enumerate(frames_meta), key=anchor_score)
    if best_idx == 0:
        return list(frames_meta)
    ordered = [frames_meta[best_idx]] + frames_meta[:best_idx] + frames_meta[best_idx + 1 :]
    return ordered


def generate_initial_pto(
    frames_meta: list[dict[str, Any]],
    output_pto_path: str,
    pano_width: int = 15000,
    pano_height: int = 7500,
    use_relative_paths: bool = False,
    use_cropped_tiff: bool = True,
) -> str:
    """
    Формирует файл initial.pto для Hugin:
    - Выходной холст: эквиректангулярная проекция (f2), w15000 h7500 v360.
    - Каждому снимку прописываются y (GimbalYawDegree), p (GimbalPitchDegree), r (GimbalRollDegree).
    - В центре панорамы (X = 7500) жестко фиксируется направление на Север (Yaw = 0°).
    - При use_cropped_tiff=True включает r:CROP для ускорения nona и multiblend в 3-4 раза.
    """
    pto_dir = os.path.dirname(os.path.abspath(output_pto_path))
    tiff_spec = "TIFF_m c:NONE r:CROP" if use_cropped_tiff else "TIFF_m c:LZW"
    lines = [
        "# hugin project file",
        "#hugin_ptoversion 2",
        f'p f2 w{pano_width} h{pano_height} v360  E0 R0 n"{tiff_spec}"',
        "m g1 i0 f0 m2 p0.00784314",
        "",
        "# image lines",
    ]

    for idx, frame in enumerate(frames_meta):
        w = int(frame["width"])
        h = int(frame["height"])
        hfov = float(frame["hfov"])
        yaw = _wrap_yaw(float(frame["gimbal_yaw"]))
        pitch = float(frame["gimbal_pitch"])
        roll = float(frame["gimbal_roll"])

        img_path = frame["path"]
        if use_relative_paths:
            img_path = os.path.relpath(img_path, pto_dir)
        img_path_pto = img_path.replace("\\", "/")

        if idx == 0:
            line = (
                f"i w{w} h{h} f0 v{hfov:.6f} "
                f"Ra0 Rb0 Rc0 Rd0 Re0 Eev0 Er1 Eb1 "
                f"r{roll:.4f} p{pitch:.4f} y{yaw:.4f} "
                f"TrX0 TrY0 TrZ0 Tpy0 Tpp0 j0 "
                f"a0 b0 c0 d0 e0 g0 t0 "
                f"Va1 Vb0 Vc0 Vd0 Vx0 Vy0 Vm5 "
                f'n"{img_path_pto}"'
            )
        else:
            line = (
                f"i w{w} h{h} f0 v=0 "
                f"Ra=0 Rb=0 Rc=0 Rd=0 Re=0 Eev0 Er1 Eb1 "
                f"r{roll:.4f} p{pitch:.4f} y{yaw:.4f} "
                f"TrX0 TrY0 TrZ0 Tpy0 Tpp0 j0 "
                f"a=0 b=0 c=0 d=0 e=0 g=0 t=0 "
                f"Va=0 Vb=0 Vc=0 Vd=0 Vx=0 Vy=0 Vm5 "
                f'n"{img_path_pto}"'
            )
        lines.append(line)

    lines.append("")
    lines.append("# specify variables that should be optimized")
    lines.append("v")
    lines.append("")
    lines.append("# control points")
    lines.append("")

    with open(output_pto_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))

    return output_pto_path


def compute_horizon_y_at_x(
    x_px: float,
    width: int,
    height: int,
    hfov_deg: float,
    pitch_deg: float,
    roll_deg: float = 0.0,
) -> float:
    """
    Вычисляет вертикальную координату Y линии математического горизонта в пикселях кадра.
    В системе координат изображения Y=0 — верхний край, Y=H — нижний край.
    При pitch > 0 (камера поднята вверх) горизонт смещается вниз (Y_horizon > H/2),
    а все пиксели неба выше горизонта имеют y < Y_horizon.
    """
    half_hfov_rad = math.radians(max(10.0, min(160.0, hfov_deg)) / 2.0)
    f_px = (width / 2.0) / math.tan(half_hfov_rad)
    pitch_rad = math.radians(pitch_deg)
    roll_rad = math.radians(roll_deg)

    y_center_horizon = (height / 2.0) + f_px * math.tan(pitch_rad)
    y_roll_offset = (x_px - (width / 2.0)) * math.tan(roll_rad)
    return y_center_horizon + y_roll_offset


def filter_sky_control_points(
    input_pto_path: str,
    output_pto_path: str,
    frames_meta: list[dict[str, Any]],
    pitch_threshold_deg: float = -26.0,
    horizon_margin_px: float = 15.0,
) -> dict[str, int]:
    """
    Удаляет из .pto все контрольные точки (строки 'c ...'), лежащие выше линии горизонта
    на кадрах верхнего ряда (GimbalPitchDegree >= pitch_threshold_deg), чтобы плывущие облака
    не искажали оптимизацию геометрии.
    """
    with open(input_pto_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    cp_pattern = re.compile(
        r"^c\s+n(\d+)\s+N(\d+)\s+x([0-9.\-+eE]+)\s+y([0-9.\-+eE]+)\s+X([0-9.\-+eE]+)\s+Y([0-9.\-+eE]+)"
    )

    total_cps = 0
    kept_cps = 0
    removed_sky_cps = 0
    out_lines: list[str] = []

    for line in lines:
        if line.startswith("c "):
            total_cps += 1
            m = cp_pattern.match(line.strip())
            if m:
                n1 = int(m.group(1))
                n2 = int(m.group(2))
                x1 = float(m.group(3))
                y1 = float(m.group(4))
                x2 = float(m.group(5))
                y2 = float(m.group(6))

                is_sky = False
                for img_idx, x_pt, y_pt in ((n1, x1, y1), (n2, x2, y2)):
                    if 0 <= img_idx < len(frames_meta):
                        fm = frames_meta[img_idx]
                        pitch = float(fm["gimbal_pitch"])
                        if pitch >= pitch_threshold_deg:
                            y_hor = compute_horizon_y_at_x(
                                x_px=x_pt,
                                width=int(fm["width"]),
                                height=int(fm["height"]),
                                hfov_deg=float(fm["hfov"]),
                                pitch_deg=pitch,
                                roll_deg=float(fm["gimbal_roll"]),
                            )
                            if y_pt < (y_hor + horizon_margin_px):
                                is_sky = True
                                break

                if is_sky:
                    removed_sky_cps += 1
                    continue

            kept_cps += 1
            out_lines.append(line)
        else:
            out_lines.append(line)

    with open(output_pto_path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(out_lines)

    return {
        "total_cps": total_cps,
        "kept_cps": kept_cps,
        "removed_sky_cps": removed_sky_cps,
    }


def _parse_pto_param(line: str, key: str) -> float | None:
    m = re.search(rf"(?<=\s){key}([0-9.\-+eE]+)", line)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _replace_pto_param(line: str, key: str, new_val: float, fmt: str = ".4f") -> str:
    return re.sub(
        rf"(?<=\s){key}[0-9.\-+eE]+",
        f"{key}{new_val:{fmt}}",
        line,
        count=1,
    )


def enforce_angle_sanity_check(
    input_pto_path: str,
    output_pto_path: str,
    frames_meta: list[dict[str, Any]],
    max_deviation_deg: float = 3.5,
) -> dict[str, Any]:
    """
    Шаг 1.4 (Sanity Check):
    Сверяет оптимизированные 3D-ориентации (y, p, r) каждого снимка с исходной телеметрией подвеса RTK
    через геодезический угол между 3D-матрицами поворота (без ложных срабатываний Gimbal Lock в надире).
    Кадры без контрольных точек или с катастрофическим выбросом ориентации возвращаются к телеметрии RTK.
    """
    with open(input_pto_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Подсчитываем число контрольных точек на каждый кадр
    cp_counts: dict[int, int] = {i: 0 for i in range(len(frames_meta))}
    cp_pattern = re.compile(r"^c\s+n(\d+)\s+N(\d+)")
    for line in lines:
        if line.startswith("c "):
            m = cp_pattern.match(line.strip())
            if m:
                i1, i2 = int(m.group(1)), int(m.group(2))
                cp_counts[i1] = cp_counts.get(i1, 0) + 1
                cp_counts[i2] = cp_counts.get(i2, 0) + 1

    # Извлекаем оптимизированные углы из i-строк
    i_line_indices: list[int] = []
    opt_angles: list[tuple[float, float, float]] = []
    for line_idx, line in enumerate(lines):
        if line.startswith("i "):
            img_idx = len(i_line_indices)
            i_line_indices.append(line_idx)
            fm = frames_meta[img_idx] if img_idx < len(frames_meta) else {}
            y_val = _parse_pto_param(line, "y")
            p_val = _parse_pto_param(line, "p")
            r_val = _parse_pto_param(line, "r")
            if y_val is None:
                y_val = _wrap_yaw(float(fm.get("gimbal_yaw", 0.0)))
            if p_val is None:
                p_val = float(fm.get("gimbal_pitch", 0.0))
            if r_val is None:
                r_val = float(fm.get("gimbal_roll", 0.0))
            opt_angles.append((y_val, p_val, r_val))

    # Вычисляем медианные поправки по надежным кадрам горизонта и среднего ряда
    valid_dy: list[float] = []
    valid_dp: list[float] = []
    valid_dr: list[float] = []

    for img_idx, (y_opt, p_opt, r_opt) in enumerate(opt_angles):
        fm = frames_meta[img_idx]
        y_init = _wrap_yaw(float(fm["gimbal_yaw"]))
        p_init = float(fm["gimbal_pitch"])
        r_init = float(fm["gimbal_roll"])

        dy = _wrap_angle_diff(y_opt, y_init)
        dp = p_opt - p_init
        dr = _wrap_angle_diff(r_opt, r_init)
        rot_diff = _rotation_angle_diff_deg(y_opt, p_opt, r_opt, y_init, p_init, r_init)

        if cp_counts.get(img_idx, 0) >= 3 and p_init > -70.0 and rot_diff <= 10.0:
            valid_dy.append(dy)
            valid_dp.append(dp)
            valid_dr.append(dr)

    med_dy = float(np.median(valid_dy)) if valid_dy else 0.0
    med_dp = float(np.median(valid_dp)) if valid_dp else 0.0
    med_dr = float(np.median(valid_dr)) if valid_dr else 0.0

    reset_frames: list[int] = []
    max_observed_dev = 0.0
    lens_degenerate = False

    # Для кадров с достаточным числом наземных контрольных точек (>= 3) порог отката
    # должен отсекать только реальные срывы оптимизации (> 14° по 3D-углу), чтобы не разрывать
    # выровненную сетку дорог и зданий при ветровом крене дрона 3°..7°
    well_constrained_limit = max(float(max_deviation_deg), 14.0)
    weak_constrained_limit = max(float(max_deviation_deg), 6.0)

    for img_idx, line_idx in enumerate(i_line_indices):
        fm = frames_meta[img_idx]
        y_init = _wrap_yaw(float(fm["gimbal_yaw"]))
        p_init = float(fm["gimbal_pitch"])
        r_init = float(fm["gimbal_roll"])
        y_opt, p_opt, r_opt = opt_angles[img_idx]

        exp_y = _wrap_yaw(y_init + med_dy)
        exp_p = max(-90.0, min(90.0, p_init + med_dp))
        exp_r = r_init + med_dr

        # Истинное 3D угловое отклонение матрицы поворота без сингулярности в надире (-90°)
        frame_dev = _rotation_angle_diff_deg(y_opt, p_opt, r_opt, exp_y, exp_p, exp_r)
        max_observed_dev = max(max_observed_dev, frame_dev)

        n_cps = cp_counts.get(img_idx, 0)
        if n_cps == 0 and img_idx != 0:
            needs_reset = True
        elif n_cps <= 2:
            needs_reset = frame_dev > weak_constrained_limit
        else:
            needs_reset = frame_dev > well_constrained_limit

        line = lines[line_idx]
        if needs_reset:
            reset_frames.append(img_idx)
            line = _replace_pto_param(line, "y", exp_y)
            line = _replace_pto_param(line, "p", exp_p)
            line = _replace_pto_param(line, "r", exp_r)

        # Проверяем физическую адекватность калибровки объектива на опорном кадре (без прямого зануления
        # дисторсии без пересчета углов, чтобы не разрушать согласованность модели)
        if img_idx == 0:
            v_opt = _parse_pto_param(line, "v")
            v_init = float(fm["hfov"])
            if v_opt is not None and abs(v_opt - v_init) > 10.0:
                lens_degenerate = True
            for k_dist in ("a", "b", "c"):
                val_d = _parse_pto_param(line, k_dist)
                if val_d is not None and abs(val_d) > 0.14:
                    lens_degenerate = True
            for k_shift in ("d", "e"):
                val_s = _parse_pto_param(line, k_shift)
                if val_s is not None and abs(val_s) > 350.0:
                    lens_degenerate = True

        lines[line_idx] = line

    with open(output_pto_path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)

    return {
        "reset_frames": reset_frames,
        "max_observed_dev_deg": float(max_observed_dev),
        "median_offsets_deg": {"dy": med_dy, "dp": med_dp, "dr": med_dr},
        "lens_degenerate": lens_degenerate,
    }


def clamp_photometric_pto(
    input_pto_path: str,
    output_pto_path: str,
    max_eev_deviation: float = 2.5,
) -> None:
    """
    Ограничивает экстремальные выбросы Eev и коэффициентов виньетирования Vb/Vc/Vd после vig_optimize,
    чтобы кадры верхнего ряда с малым числом точек не становились чрезмерно пересвеченными или темными.
    """
    with open(input_pto_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    eev_vals: list[float] = []
    for line in lines:
        if line.startswith("i "):
            eev = _parse_pto_param(line, "Eev")
            if eev is not None and abs(eev) < 10.0:
                eev_vals.append(eev)

    med_eev = float(np.median(eev_vals)) if eev_vals else 0.0

    for idx, line in enumerate(lines):
        if line.startswith("i "):
            eev = _parse_pto_param(line, "Eev")
            if eev is not None and abs(eev - med_eev) > max_eev_deviation:
                clamped_eev = med_eev + math.copysign(max_eev_deviation, eev - med_eev)
                line = _replace_pto_param(line, "Eev", clamped_eev, fmt=".4f")
            lines[idx] = line

    with open(output_pto_path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(lines)


def compute_pto_cp_rms(pto_path: str, pano_width: int = 15000) -> dict[str, Any]:
    """
    Вычисляет среднеквадратичную (RMS) и 95-перцентильную (p95) угловую ошибку
    контрольных точек в пикселях итоговой панорамы (15000x7500) для диагностики качества сшивки.
    """
    with open(pto_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    img_params: list[dict[str, float]] = []
    v0 = 71.5
    a0 = 0.0
    b0 = 0.0
    c0 = 0.0
    d0 = 0.0
    e0 = 0.0

    for line in lines:
        if line.startswith("i "):
            w = _parse_pto_param(line, "w") or 5280.0
            h = _parse_pto_param(line, "h") or 3956.0
            v = _parse_pto_param(line, "v")
            a = _parse_pto_param(line, "a")
            b = _parse_pto_param(line, "b")
            c = _parse_pto_param(line, "c")
            d = _parse_pto_param(line, "d")
            e = _parse_pto_param(line, "e")
            if not img_params:
                if v is not None and v > 1.0:
                    v0 = v
                if a is not None:
                    a0 = a
                if b is not None:
                    b0 = b
                if c is not None:
                    c0 = c
                if d is not None:
                    d0 = d
                if e is not None:
                    e0 = e
            img_params.append(
                {
                    "w": w,
                    "h": h,
                    "v": v if (v is not None and v > 1.0) else v0,
                    "a": a if a is not None else a0,
                    "b": b if b is not None else b0,
                    "c": c if c is not None else c0,
                    "d": d if d is not None else d0,
                    "e": e if e is not None else e0,
                    "y": _parse_pto_param(line, "y") or 0.0,
                    "p": _parse_pto_param(line, "p") or 0.0,
                    "r": _parse_pto_param(line, "r") or 0.0,
                }
            )

    def _pixel_to_ray(img_idx: int, xp: float, yp: float) -> np.ndarray:
        p = img_params[img_idx]
        f_px = (p["w"] * 0.5) / math.tan(math.radians(p["v"] * 0.5))
        dx = xp - p["w"] * 0.5 - p["d"]
        dy = yp - p["h"] * 0.5 - p["e"]
        r_scale = min(p["w"], p["h"]) * 0.5
        r_src = math.hypot(dx, dy) / r_scale
        if r_src > 1e-7:
            d_param = 1.0 - p["a"] - p["b"] - p["c"]
            rd = r_src
            for _ in range(6):
                f_val = (((p["a"] * rd + p["b"]) * rd + p["c"]) * rd + d_param) * rd - r_src
                df_val = (4.0 * p["a"] * rd + 3.0 * p["b"]) * rd * rd + 2.0 * p["c"] * rd + d_param
                if abs(df_val) < 1e-6:
                    break
                rd -= f_val / df_val
            if rd > 0.0:
                scale = rd / r_src
                dx *= scale
                dy *= scale
        vx = dx / f_px
        vy = -dy / f_px
        vec = np.array([vx, vy, 1.0], dtype=np.float64)
        vec /= np.linalg.norm(vec)
        rot = _euler_to_rotation_matrix(p["y"], -p["p"], -p["r"])
        return rot @ vec

    cp_pattern = re.compile(
        r"^c\s+n(\d+)\s+N(\d+)\s+x([0-9.\-+eE]+)\s+y([0-9.\-+eE]+)\s+X([0-9.\-+eE]+)\s+Y([0-9.\-+eE]+)"
    )
    errors_px: list[float] = []
    px_per_deg = float(pano_width) / 360.0

    for line in lines:
        if line.startswith("c "):
            m = cp_pattern.match(line.strip())
            if m:
                n1, n2 = int(m.group(1)), int(m.group(2))
                if 0 <= n1 < len(img_params) and 0 <= n2 < len(img_params):
                    r1 = _pixel_to_ray(n1, float(m.group(3)), float(m.group(4)))
                    r2 = _pixel_to_ray(n2, float(m.group(5)), float(m.group(6)))
                    dot = max(-1.0, min(1.0, float(np.dot(r1, r2))))
                    ang_deg = math.degrees(math.acos(dot))
                    errors_px.append(ang_deg * px_per_deg)

    if not errors_px:
        return {"count": 0, "rms_px": 0.0, "p95_px": 0.0, "max_px": 0.0}

    arr = np.array(errors_px, dtype=np.float64)
    return {
        "count": len(errors_px),
        "rms_px": round(float(np.sqrt(np.mean(arr**2))), 2),
        "p95_px": round(float(np.percentile(arr, 95)), 2),
        "max_px": round(float(np.max(arr)), 2),
    }
