#!/usr/bin/env python3
"""
Шаг 2.1: Автоматическая индексация базы заготовок неба (15000x3750) и генерация sky_catalog.json.
Определяет высоту солнца Y_sun_sky (или флаг пасмурного шаблона is_overcast_template)
и рассчитывает паспорт из N = 20 вертикальных срезов в полосе над горизонтом Y in [2850, 3650].
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

from src.sky_replacer.signature import (
    compute_sky_signature,
    detect_template_sun_y,
    read_image_bgr,
)


def resolve_sky_dir(requested_dir: str | None = None) -> str:
    """Находит директорию с заготовками неба (sky_templates/ или sky/)."""
    candidates = []
    if requested_dir:
        candidates.append(requested_dir)
    candidates.extend(
        [
            os.path.join(PROJECT_ROOT, "sky_templates"),
            os.path.join(PROJECT_ROOT, "sky"),
            "/app/sky_templates",
            "/app/sky",
        ]
    )
    for c in candidates:
        if c and os.path.isdir(c):
            return os.path.abspath(c)
    raise FileNotFoundError(
        f"Папка с шаблонами неба не найдена. Проверены пути: {candidates}"
    )


def build_sky_catalog(
    sky_dir: str,
    catalog_path: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Сканирует папку шаблонов неба и создает/обновляет sky_catalog.json."""
    existing_manual_overrides: dict[str, float] = {}
    if os.path.isfile(catalog_path):
        try:
            with open(catalog_path, "r", encoding="utf-8") as f:
                old_cat = json.load(f)
            for item in old_cat.get("templates", []):
                rel = item.get("relative_path")
                man_y = item.get("manual_sun_y")
                if rel and man_y is not None:
                    existing_manual_overrides[rel] = float(man_y)
        except Exception:
            pass

    image_files: list[tuple[str, str]] = []
    for root, _dirs, files in sorted(os.walk(sky_dir)):
        for fname in sorted(files):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")) and not fname.startswith("."):
                full_p = os.path.join(root, fname)
                rel_p = os.path.relpath(full_p, sky_dir).replace("\\", "/")
                image_files.append((rel_p, full_p))

    if not image_files:
        raise FileNotFoundError(f"В папке {sky_dir} не найдено изображений шаблонов неба.")

    templates_list: list[dict[str, Any]] = []
    t_start = time.time()

    print(f"[index_skies] Найдено шаблонов неба: {len(image_files)} в '{sky_dir}'")
    for idx, (rel_p, full_p) in enumerate(image_files, start=1):
        t0 = time.time()
        sky_bgr = read_image_bgr(full_p)
        h, w = sky_bgr.shape[:2]

        manual_y = existing_manual_overrides.get(rel_p)
        sun_meta = detect_template_sun_y(
            sky_bgr=sky_bgr,
            config=config,
            manual_sun_y=manual_y,
            template_path=full_p,
        )
        slices_sig = compute_sky_signature(
            sky_or_pano_bgr=sky_bgr,
            config=config,
            valid_mask_band=None,
            is_already_band=False,
        )

        category = os.path.dirname(rel_p) or "default"
        tmpl_id = rel_p.rsplit(".", 1)[0].replace("/", "_")

        entry = {
            "id": tmpl_id,
            "relative_path": rel_p,
            "category": category,
            "width": w,
            "height": h,
            "is_overcast_template": sun_meta["is_overcast_template"],
            "sun_y": sun_meta["sun_y"],
            "manual_sun_y": sun_meta["manual_sun_y"],
            "delta_l_asymmetry": sun_meta["delta_l_asymmetry"],
            "peak_l_center": sun_meta["peak_l_center"],
            "mean_b_center": sun_meta["mean_b_center"],
            "slices": slices_sig,
        }
        templates_list.append(entry)
        dt = time.time() - t0
        sun_str = "OVERCAST (None)" if entry["is_overcast_template"] else f"Y={entry['sun_y']:.1f}px"
        print(
            f"  [{idx:02d}/{len(image_files):02d}] {rel_p:35s} | {w}x{h} | "
            f"dL={entry['delta_l_asymmetry']:5.1f} | sun={sun_str:16s} ({dt:.2f}s)"
        )

    catalog_data = {
        "version": "1.0",
        "sky_dir": sky_dir,
        "num_templates": len(templates_list),
        "templates": templates_list,
    }

    os.makedirs(os.path.dirname(os.path.abspath(catalog_path)), exist_ok=True)
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog_data, f, ensure_ascii=False, indent=2)

    print(
        f"[index_skies] Каталог успешно сохранен в '{catalog_path}' "
        f"за {time.time() - t_start:.1f} сек."
    )
    return catalog_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Автоматическая индексация базы шаблонов неба (15000x3750) и создание sky_catalog.json"
    )
    parser.add_argument(
        "--sky-dir",
        type=str,
        default=None,
        help="Путь к папке с заготовками неба (по умолчанию sky_templates/ или sky/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(PROJECT_ROOT, "sky_catalog.json"),
        help="Путь к выходному файлу sky_catalog.json",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_ROOT, "config.yaml"),
        help="Путь к файлу конфигурации config.yaml",
    )
    args = parser.parse_args()

    cfg: dict[str, Any] = {}
    if os.path.isfile(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    sky_dir = resolve_sky_dir(args.sky_dir)
    build_sky_catalog(sky_dir=sky_dir, catalog_path=args.output, config=cfg)


if __name__ == "__main__":
    main()
