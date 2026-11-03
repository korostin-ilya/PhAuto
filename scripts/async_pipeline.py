#!/usr/bin/env python3
"""
Асинхронный конвейер загрузки серий БПЛА из AWS S3 (boto3) и сборки сферических панорам.
- Автоматически проверяет сессию AWS SSO (при необходимости вызывает 'aws sso login')
  и прокидывает временные ключи в корневой файл .env.
- Параллельно скачивает серии снимков (Producer) и собирает панорамы через ProcessPoolExecutor (Consumer):
  пока первая папка обрабатывается, в фоне скачивается вторая, третья и т.д.
- Поддерживает как сохранение исходных снимков (keep_raw=True), так и скачивание только
  во временную директорию (keep_raw=False / --temp-raw) для экономии места на диске.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.storage.s3_downloader import (
    DEFAULT_AWS_REGION,
    DEFAULT_S3_BUCKET,
    download_series_boto3,
    ensure_aws_env_credentials,
    load_dotenv_file,
    parse_asana_task_string,
)


def _run_stitch_worker_task(job_payload: dict[str, Any]) -> dict[str, Any]:
    """
    Точка входа для дочернего процесса ProcessPoolExecutor.
    Запускает полный цикл сшивки и замены неба (process_single_series) для одной скачанной папки.
    """
    from scripts.run_pipeline import process_single_series

    return process_single_series(
        input_dir=job_payload["input_dir"],
        output_jpg_path=job_payload["output_jpg_path"],
        sky_dir=job_payload["sky_dir"],
        catalog=job_payload["catalog"],
        config=job_payload["config"],
        work_dir=job_payload.get("work_dir"),
        keep_temp=bool(job_payload.get("keep_temp", False)),
        skip_existing=bool(job_payload.get("skip_existing", False)),
    )


async def run_async_pipeline(
    task_strings: list[str],
    pano_root: str,
    sky_dir: str,
    catalog: dict[str, Any],
    config: dict[str, Any],
    env_path: str | None = None,
    keep_raw: bool = True,
    overwrite: bool = False,
    keep_temp: bool = False,
    work_dir: str | None = None,
    temp_raw_root: str | None = None,
    prefetch_limit: int | None = None,
    stitch_workers: int = 1,
    download_fn: Callable[..., dict[str, Any]] | None = None,
    stitch_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Запускает асинхронный конвейер (Producer-Consumer):
    - Producer в фоне скачивает кадры серий из S3 (в постоянную или временную папку).
    - Consumer забирает готовые папки из очереди и собирает панорамы через ProcessPoolExecutor.
    """
    t_pipeline_start = time.time()
    pano_cfg = config.get("panorama", {})
    s3_cfg = config.get("s3_downloader", {})

    suffix = str(pano_cfg.get("output_suffix", "_full.jpeg"))
    skip_existing = bool(pano_cfg.get("skip_existing_output", True)) and not overwrite
    default_bucket = str(s3_cfg.get("default_bucket", DEFAULT_S3_BUCKET))
    max_dl_threads = int(s3_cfg.get("max_download_threads", 8))
    max_retries = int(s3_cfg.get("max_retries", 3))

    # Если исходники качаются во временную папку (режим экономии места), ограничиваем буфер предзагрузки,
    # чтобы не забить диск десятками папок до того, как они соберутся.
    if prefetch_limit is None:
        if not keep_raw:
            effective_prefetch = max(1, int(s3_cfg.get("temp_prefetch_limit", 2)))
        else:
            effective_prefetch = 0  # Без ограничений (качаем все подряд)
    else:
        effective_prefetch = max(0, int(prefetch_limit))

    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=effective_prefetch)
    effective_dl_fn = download_fn or download_series_boto3

    completed_jobs: list[dict[str, Any]] = []
    skipped_jobs: list[dict[str, Any]] = []
    failed_jobs: list[dict[str, Any]] = []

    total_tasks = len(task_strings)
    mode_label = (
        "СОХРАНЕНИЕ ИСХОДНИКОВ (в папку pano/)"
        if keep_raw
        else f"ВРЕМЕННАЯ ДИРЕКТОРИЯ (экономия места, буфер={effective_prefetch})"
    )
    print(f"\n{'=' * 80}")
    print(f"[Конвейер] Задач в очереди: {total_tasks} | Режим хранения: {mode_label}")
    print(f"{'=' * 80}")

    async def _producer_downloader() -> None:
        for idx, raw_task in enumerate(task_strings, start=1):
            try:
                task_info = parse_asana_task_string(raw_task, default_bucket=default_bucket)
            except Exception as exc:
                print(f"[Ошибка задачи {idx}/{total_tasks}] Некорректная строка '{raw_task}': {exc}")
                failed_jobs.append({"task": raw_task, "stage": "parse", "error": str(exc)})
                continue

            rel_dir = task_info["relative_dir"].replace("/", os.sep)
            final_shoot_dir = os.path.abspath(os.path.join(pano_root, rel_dir))
            output_jpg_path = os.path.join(final_shoot_dir, f"{task_info['date']}{suffix}")

            if skip_existing and os.path.isfile(output_jpg_path) and os.path.getsize(output_jpg_path) > 100_000:
                print(
                    f"[Пропуск {idx}/{total_tasks}] Итоговая панорама уже существует: '{output_jpg_path}' "
                    f"(используйте --overwrite для пересборки)"
                )
                skipped_jobs.append({"task": raw_task, "output_path": output_jpg_path})
                continue

            if keep_raw:
                raw_download_dir = final_shoot_dir
                is_temp_raw = False
            else:
                if temp_raw_root:
                    os.makedirs(temp_raw_root, exist_ok=True)
                raw_download_dir = tempfile.mkdtemp(
                    prefix=f"pano_raw_{task_info['date']}_",
                    dir=temp_raw_root,
                )
                is_temp_raw = True

            print(
                f"\n[Загрузка {idx}/{total_tasks}] Старт скачивания: {task_info['s3_uri']} -> "
                f"{'[TEMP] ' if is_temp_raw else ''}{raw_download_dir}"
            )
            try:
                dl_stats = await asyncio.to_thread(
                    effective_dl_fn,
                    task_info=task_info,
                    dest_dir=raw_download_dir,
                    max_retries=max_retries,
                    max_threads=max_dl_threads,
                    env_path=env_path,
                )
            except Exception as exc:
                print(f"[Ошибка загрузки {idx}/{total_tasks}] '{raw_task}': {exc}")
                if is_temp_raw and os.path.isdir(raw_download_dir):
                    shutil.rmtree(raw_download_dir, ignore_errors=True)
                failed_jobs.append({"task": raw_task, "stage": "download", "error": str(exc)})
                continue

            os.makedirs(final_shoot_dir, exist_ok=True)
            job_work_dir = (
                os.path.join(work_dir, task_info["relative_dir"].replace("/", "_"))
                if work_dir
                else None
            )

            staged_item = {
                "index": idx,
                "total": total_tasks,
                "task_info": task_info,
                "input_dir": raw_download_dir,
                "output_jpg_path": output_jpg_path,
                "is_temp_raw": is_temp_raw,
                "dl_stats": dl_stats,
                "sky_dir": sky_dir,
                "catalog": catalog,
                "config": config,
                "work_dir": job_work_dir,
                "keep_temp": keep_temp,
                "skip_existing": skip_existing,
            }
            await queue.put(staged_item)

        for _ in range(max(1, stitch_workers)):
            await queue.put(None)

    async def _consumer_stitcher(pool: ProcessPoolExecutor) -> None:
        loop = asyncio.get_running_loop()
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break

            idx = item["index"]
            total = item["total"]
            task_info = item["task_info"]
            raw_dir = item["input_dir"]
            is_temp_raw = item["is_temp_raw"]
            out_path = item["output_jpg_path"]

            print(
                f"\n>>> [Сборка {idx}/{total}] Запуск обработки в ProcessPoolExecutor: "
                f"{task_info['relative_dir']} -> {out_path}"
            )
            try:
                if stitch_fn is not None:
                    stitch_res = await asyncio.to_thread(stitch_fn, item)
                else:
                    stitch_res = await loop.run_in_executor(pool, _run_stitch_worker_task, item)
                completed_jobs.append(
                    {
                        "task": task_info["raw_input"],
                        "relative_dir": task_info["relative_dir"],
                        "output_path": out_path,
                        "download_stats": item["dl_stats"],
                        "stitch_result": stitch_res,
                    }
                )
            except Exception as exc:
                print(f"[Ошибка сборки {idx}/{total}] '{task_info['relative_dir']}': {exc}")
                failed_jobs.append(
                    {
                        "task": task_info["raw_input"],
                        "stage": "stitch",
                        "error": str(exc),
                    }
                )
            finally:
                if is_temp_raw and os.path.isdir(raw_dir):
                    shutil.rmtree(raw_dir, ignore_errors=True)
                    print(f"[Очистка диска] Удалена временная папка исходников: {raw_dir}")
                queue.task_done()

    with ProcessPoolExecutor(max_workers=max(1, stitch_workers)) as pool:
        producer_task = asyncio.create_task(_producer_downloader())
        consumer_tasks = [
            asyncio.create_task(_consumer_stitcher(pool))
            for _ in range(max(1, stitch_workers))
        ]
        await asyncio.gather(producer_task, *consumer_tasks)

    total_elapsed = round(time.time() - t_pipeline_start, 1)
    print(f"\n{'=' * 80}")
    print(
        f"[Итог конвейера] Завершено за {total_elapsed}с | "
        f"Успешно: {len(completed_jobs)} | Пропущено: {len(skipped_jobs)} | Ошибок: {len(failed_jobs)}"
    )
    if failed_jobs:
        for f_item in failed_jobs:
            print(f"  [!] Ошибка ({f_item['stage']}): {f_item['task']} -> {f_item['error']}")
    print(f"{'=' * 80}\n")

    return {
        "completed": completed_jobs,
        "skipped": skipped_jobs,
        "failed": failed_jobs,
        "elapsed_sec": total_elapsed,
    }


def _collect_interactive_tasks() -> list[str]:
    """Интерактивный ввод задач из Asana в цикле (аналогично старому батнику)."""
    tasks: list[str] = []
    print("\n" + "=" * 78)
    print(" Ввод задач из Asana (формат: customer_name/site_name/date_number)")
    print(" Нажмите Enter на пустой строке, чтобы завершить ввод и запустить конвейер")
    print("=" * 78)
    while True:
        try:
            line = input(
                f"[{len(tasks) + 1}] Введите задачу Asana (или Enter для старта): "
            ).strip()
        except EOFError:
            break
        if not line:
            break
        tasks.append(line)
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Асинхронный конвейер скачивания снимков из AWS S3 и сборки панорам БПЛА"
    )
    parser.add_argument(
        "tasks_pos",
        nargs="*",
        help="Список задач Asana (customer/site/date) или S3-путей",
    )
    parser.add_argument(
        "--tasks",
        "-t",
        nargs="+",
        default=None,
        help="Список задач Asana через пробел",
    )
    parser.add_argument(
        "--tasks-file",
        "-f",
        type=str,
        default=None,
        help="Путь к текстовому файлу со списком задач Asana (по одной на строке)",
    )
    parser.add_argument(
        "--pano-dir",
        "--output-dir",
        "-o",
        dest="pano_dir",
        type=str,
        default=None,
        help="Корневая папка для сохранения панорам (по умолчанию pano/ или /app/input)",
    )
    parser.add_argument(
        "--temp-raw",
        "--no-keep-raw",
        dest="temp_raw",
        action="store_true",
        help="Скачивать исходные снимки только во временную директорию и удалять их сразу после сборки",
    )
    parser.add_argument(
        "--keep-raw",
        dest="keep_raw",
        action="store_true",
        help="Сохранять скачанные исходные снимки в папке серии (по умолчанию)",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default=None,
        help="Базовая папка для временных исходников при --temp-raw (по умолчанию системный temp)",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=None,
        help="Максимум предзагруженных папок в очереди (по умолчанию 2 для --temp-raw, без лимита для --keep-raw)",
    )
    parser.add_argument(
        "--workers",
        "-j",
        type=int,
        default=1,
        help="Число параллельных процессов сборки в ProcessPoolExecutor (по умолчанию 1)",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=os.path.join(PROJECT_ROOT, ".env"),
        help="Путь к файлу .env для записи/чтения ключей AWS (по умолчанию <корень>/.env)",
    )
    parser.add_argument(
        "--aws-profile",
        type=str,
        default=None,
        help="Имя профиля AWS SSO (необязательно)",
    )
    parser.add_argument(
        "--auth-only",
        action="store_true",
        help="Только проверить/обновить сессию AWS SSO и записать ключи в .env без запуска сборки",
    )
    parser.add_argument(
        "--sky-dir",
        type=str,
        default=None,
        help="Путь к папке с шаблонами неба",
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
        help="Не удалять временные файлы сшивки Hugin (.pto, stitched_base.tif)",
    )
    parser.add_argument(
        "--overwrite",
        "--force",
        action="store_true",
        help="Пересобирать панорамы, даже если итоговый файл <date>_full.jpeg уже существует",
    )
    args = parser.parse_args()

    # 1. Загружаем конфигурацию
    config: dict[str, Any] = {}
    if os.path.isfile(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    s3_cfg = config.get("s3_downloader", {})
    default_region = str(s3_cfg.get("default_region", DEFAULT_AWS_REGION))

    # 2. Проверяем сессию AWS SSO и прокидываем ключи в корневой .env
    ensure_aws_env_credentials(
        env_path=args.env_file,
        default_region=default_region,
        profile=args.aws_profile,
    )
    if args.auth_only:
        return

    # 3. Собираем список задач Asana
    tasks: list[str] = []
    if args.tasks_pos:
        tasks.extend(args.tasks_pos)
    if args.tasks:
        tasks.extend(args.tasks)
    if args.tasks_file:
        if not os.path.isfile(args.tasks_file):
            raise FileNotFoundError(f"Файл со списком задач не найден: {args.tasks_file}")
        with open(args.tasks_file, "r", encoding="utf-8") as f:
            for line in f:
                clean = line.strip()
                if clean and not clean.startswith("#"):
                    tasks.append(clean)

    if not tasks:
        tasks = _collect_interactive_tasks()

    if not tasks:
        print("[Конвейер] Список задач пуст. Завершение работы.")
        return

    # 4. Определяем режим сохранения исходных снимков (keep_raw vs temp_raw)
    cfg_keep_raw = bool(s3_cfg.get("keep_raw_frames", True))
    if args.temp_raw:
        keep_raw = False
    elif args.keep_raw:
        keep_raw = True
    else:
        keep_raw = cfg_keep_raw

    # 5. Определяем корневую папку для сохранения панорам
    pano_root = args.pano_dir
    if not pano_root:
        for cand in ("/app/input", os.path.join(PROJECT_ROOT, "pano")):
            if os.path.isdir(cand):
                pano_root = cand
                break
        if not pano_root:
            pano_root = os.path.join(PROJECT_ROOT, "pano")
    os.makedirs(pano_root, exist_ok=True)

    # 6. Загружаем каталог шаблонов неба
    from scripts.index_skies import build_sky_catalog, resolve_sky_dir

    sky_dir = resolve_sky_dir(args.sky_dir)
    if os.path.isfile(args.catalog):
        with open(args.catalog, "r", encoding="utf-8") as f:
            catalog = json.load(f)
    else:
        catalog = build_sky_catalog(sky_dir=sky_dir, catalog_path=args.catalog, config=config)

    # 7. Запускаем асинхронный конвейер (boto3 загрузка + ProcessPoolExecutor сборка)
    asyncio.run(
        run_async_pipeline(
            task_strings=tasks,
            pano_root=pano_root,
            sky_dir=sky_dir,
            catalog=catalog,
            config=config,
            env_path=args.env_file,
            keep_raw=keep_raw,
            overwrite=args.overwrite,
            keep_temp=args.keep_temp,
            work_dir=args.work_dir,
            temp_raw_root=args.temp_dir,
            prefetch_limit=args.prefetch,
            stitch_workers=args.workers,
        )
    )


if __name__ == "__main__":
    main()
