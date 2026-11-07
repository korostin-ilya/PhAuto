#!/usr/bin/env python3
"""
Асинхронный конвейер загрузки серий БПЛА из AWS S3 (boto3) и сборки сферических панорам.
- Автоматически проверяет сессию AWS SSO (при необходимости вызывает 'aws sso login')
  и прокидывает временные ключи в корневой файл .env (сохраняя ASANA_PAT и TRACEAIR_SESS_ID).
- Параллельно скачивает серии снимков (Producer) и собирает панорамы через ProcessPoolExecutor (Consumer):
  пока первая папка обрабатывается, в фоне скачивается вторая, третья и т.д.
- Поддерживает как сохранение исходных снимков (keep_raw=True), так и скачивание только
  во временную директорию (keep_raw=False / --temp-raw) для экономии места на диске.
- Опционально подключает изолированный сервис src/magic_service (если папка присутствует в проекте)
  для чтения задач из Asana и публикации панорам в TraceAir Magic в двух режимах:
    1) --auto-publish     : сквозной конвейер (скачивание -> сшивка -> фоновая публикация в Magic)
    2) --confirm-publish  : пауза после сшивки для визуального осмотра перед публикацией в Magic
    3) --upload-only      : публикация уже готовых локальных панорам без повторной сшивки
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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
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


def _try_get_magic_service() -> Any:
    """
    Мягкое динамическое подключение изолированного сервиса src/magic_service.
    Если папка src/magic_service удалена из проекта, возвращает None,
    и весь основной конвейер продолжает работать штатно без ошибок.
    """
    try:
        import src.magic_service as magic_service

        return magic_service
    except ImportError:
        return None


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


def check_auto_publish_qc_gate(
    stitch_res: dict[str, Any] | None,
    config: dict[str, Any],
) -> tuple[bool, str]:
    """
    Проверяет метрики качества сшивки (Auto-QC Gate) перед автоматической публикацией в Magic:
    - RMS ошибка контрольных точек не превышает stitcher.auto_publish_max_rms_px (по умолчанию 12.0 px)
    - Число сброшенных по Sanity Check кадров не превышает stitcher.auto_publish_max_reset_frames (по умолчанию 4)
    Возвращает (passed, reason).
    """
    if not isinstance(stitch_res, dict):
        return True, "ok"
    stitch_cfg = config.get("stitcher", {})
    max_rms_px = float(stitch_cfg.get("auto_publish_max_rms_px", 12.0))
    max_reset_frames = int(stitch_cfg.get("auto_publish_max_reset_frames", 4))

    stitch_stats = stitch_res.get("stitch_stats", {})
    if not isinstance(stitch_stats, dict):
        return True, "ok"

    cp_rms = stitch_stats.get("cp_rms", {})
    rms_px = float(cp_rms.get("rms_px", 0.0)) if isinstance(cp_rms, dict) else 0.0

    sanity = stitch_stats.get("sanity_check", {})
    reset_frames = sanity.get("reset_frames", []) if isinstance(sanity, dict) else []
    num_reset = len(reset_frames) if isinstance(reset_frames, list) else 0

    reasons: list[str] = []
    if max_rms_px > 0 and rms_px > max_rms_px:
        reasons.append(f"RMS={rms_px:.2f}px > порога {max_rms_px:.1f}px")
    if max_reset_frames >= 0 and num_reset > max_reset_frames:
        reasons.append(f"Sanity reset кадров={num_reset} > порога {max_reset_frames}")

    if reasons:
        return False, "; ".join(reasons)
    return True, f"RMS={rms_px:.2f}px, reset={num_reset}"


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
    asana_tasks_meta: dict[str, dict[str, Any]] | None = None,
    auto_publish: bool = False,
    publish_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Запускает асинхронный конвейер (Producer -> Stitcher [-> опционально Publisher]):
    - Producer в фоне скачивает кадры серий из S3 (в постоянную или временную папку).
    - Stitcher забирает готовые папки из очереди и собирает панорамы через ProcessPoolExecutor.
    - Publisher (при auto_publish=True) параллельно выгружает первую основную панораму (<date>_full.jpeg)
      в TraceAir Magic сразу после завершения сшивки каждой серии.
    """
    t_pipeline_start = time.time()
    pano_cfg = config.get("panorama", {})
    s3_cfg = config.get("s3_downloader", {})

    suffix = str(pano_cfg.get("output_suffix", "_full.jpeg"))
    skip_existing = bool(pano_cfg.get("skip_existing_output", True)) and not overwrite
    default_bucket = str(s3_cfg.get("default_bucket", DEFAULT_S3_BUCKET))
    max_dl_threads = int(s3_cfg.get("max_download_threads", 8))
    max_retries = int(s3_cfg.get("max_retries", 3))

    magic_service = _try_get_magic_service()
    effective_publish_fn = publish_fn or (
        magic_service.publish_panorama_to_magic if magic_service is not None else None
    )
    can_auto_publish = bool(auto_publish and effective_publish_fn is not None)
    meta_by_rel_dir = dict(asana_tasks_meta or {})

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
    publish_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    effective_dl_fn = download_fn or download_series_boto3

    completed_jobs: list[dict[str, Any]] = []
    skipped_jobs: list[dict[str, Any]] = []
    failed_jobs: list[dict[str, Any]] = []
    published_jobs: list[dict[str, Any]] = []

    total_tasks = len(task_strings)
    mode_label = (
        "СОХРАНЕНИЕ ИСХОДНИКОВ (в папку pano/)"
        if keep_raw
        else f"ВРЕМЕННАЯ ДИРЕКТОРИЯ (экономия места, буфер={effective_prefetch})"
    )
    pub_label = "ВКЛ (сквозная автопубликация в Magic)" if can_auto_publish else "ВЫКЛ (локальная сборка)"
    print(f"\n{'=' * 80}")
    print(f"[Конвейер] Задач в очереди: {total_tasks} | Хранение: {mode_label} | Автопубликация: {pub_label}")
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
                    f"[Пропуск сшивки {idx}/{total_tasks}] Итоговая панорама уже существует: '{output_jpg_path}' "
                    f"(используйте --overwrite для пересборки)"
                )
                skipped_jobs.append(
                    {
                        "task": raw_task,
                        "relative_dir": task_info["relative_dir"],
                        "output_path": output_jpg_path,
                    }
                )
                if can_auto_publish and task_info["relative_dir"] in meta_by_rel_dir:
                    await publish_queue.put(
                        {
                            "index": idx,
                            "total": total_tasks,
                            "task_meta": meta_by_rel_dir[task_info["relative_dir"]],
                            "output_path": output_jpg_path,
                        }
                    )
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
                if can_auto_publish:
                    qc_passed, qc_reason = check_auto_publish_qc_gate(stitch_res, config)
                    task_meta = meta_by_rel_dir.get(task_info["relative_dir"])
                    if not qc_passed:
                        print(
                            f"  [Auto-QC Карантин] Панорама '{task_info['relative_dir']}' сохранена локально ('{out_path}'), "
                            f"но автоматическая публикация в Magic остановлена для ручной проверки ({qc_reason})."
                        )
                    elif task_meta and task_meta.get("card_id"):
                        await publish_queue.put(
                            {
                                "index": idx,
                                "total": total_tasks,
                                "task_meta": task_meta,
                                "output_path": out_path,
                            }
                        )
                    else:
                        print(
                            f"  [Magic Пропуск] Для '{task_info['relative_dir']}' не найден card_id в Asana "
                            f"(запустите с --from-asana для автопубликации)."
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

    finalize_bg_tasks: list[asyncio.Task[None]] = []

    async def _finalize_in_bg(
        idx: int,
        total: int,
        task_meta: dict[str, Any],
        up_res: dict[str, Any],
    ) -> None:
        rel_dir = task_meta.get("relative_dir", "")
        try:
            fin_res = await asyncio.to_thread(
                magic_service.finalize_panorama_publication,
                up_res,
                60.0,
            )
            fin_res["file"] = up_res.get("local_file", up_res.get("file", ""))
            published_jobs.append(fin_res)
        except Exception as exc:
            print(f"[Ошибка финализации {idx}/{total}] '{rel_dir}': {exc}")
            failed_jobs.append(
                {
                    "task": rel_dir,
                    "stage": "publish_finalize",
                    "error": str(exc),
                }
            )

    async def _consumer_publisher() -> None:
        if not can_auto_publish or effective_publish_fn is None:
            return
        use_decoupled = (
            publish_fn is None
            and magic_service is not None
            and hasattr(magic_service, "upload_panorama_to_bo_storage")
            and hasattr(magic_service, "finalize_panorama_publication")
        )
        while True:
            pub_item = await publish_queue.get()
            if pub_item is None:
                publish_queue.task_done()
                break

            idx = pub_item["index"]
            total = pub_item["total"]
            task_meta = pub_item["task_meta"]
            out_path = pub_item["output_path"]
            rel_dir = task_meta.get("relative_dir", "")

            print(
                f"\n>>> [Выгрузка в Magic {idx}/{total}] Старт выгрузки основной панорамы: "
                f"{rel_dir} (card_id: {task_meta.get('card_id')})"
            )
            try:
                if use_decoupled:
                    up_res = await asyncio.to_thread(
                        magic_service.upload_panorama_to_bo_storage,
                        task=task_meta,
                        local_file=out_path,
                        pano_root=pano_root,
                        env_path=env_path,
                    )
                    if up_res.get("status") in ("copying_started", "uploaded_pending_publication"):
                        bg_task = asyncio.create_task(
                            _finalize_in_bg(idx, total, task_meta, up_res)
                        )
                        finalize_bg_tasks.append(bg_task)
                    else:
                        published_jobs.append(up_res)
                else:
                    pub_res = await asyncio.to_thread(
                        effective_publish_fn,
                        task=task_meta,
                        local_file=out_path,
                        pano_root=pano_root,
                        env_path=env_path,
                    )
                    published_jobs.append(pub_res)
            except Exception as exc:
                print(f"[Ошибка публикации {idx}/{total}] '{rel_dir}': {exc}")
                failed_jobs.append(
                    {
                        "task": rel_dir,
                        "stage": "publish",
                        "error": str(exc),
                    }
                )
            finally:
                publish_queue.task_done()

    with ProcessPoolExecutor(max_workers=max(1, stitch_workers)) as pool:
        producer_task = asyncio.create_task(_producer_downloader())
        consumer_tasks = [
            asyncio.create_task(_consumer_stitcher(pool))
            for _ in range(max(1, stitch_workers))
        ]
        pub_workers_count = 1 if can_auto_publish else 0
        publisher_tasks = [
            asyncio.create_task(_consumer_publisher())
            for _ in range(pub_workers_count)
        ]

        await asyncio.gather(producer_task, *consumer_tasks)
        if publisher_tasks:
            for _ in range(pub_workers_count):
                await publish_queue.put(None)
            await asyncio.gather(*publisher_tasks)
        if finalize_bg_tasks:
            print(
                f"\n[Magic] Все панорамы отправлены в bo-storage. "
                f"Ожидание серверной публикации ({len(finalize_bg_tasks)} шт., проверка каждые 60 сек.)..."
            )
            await asyncio.gather(*finalize_bg_tasks)

    total_elapsed = round(time.time() - t_pipeline_start, 1)
    print(f"\n{'=' * 80}")
    pub_summary = f" | Опубликовано в Magic: {len(published_jobs)}" if can_auto_publish else ""
    print(
        f"[Итог конвейера] Завершено за {total_elapsed}с | "
        f"Собрано: {len(completed_jobs)} | Пропущено: {len(skipped_jobs)}{pub_summary} | Ошибок: {len(failed_jobs)}"
    )
    if failed_jobs:
        for f_item in failed_jobs:
            print(f"  [!] Ошибка ({f_item['stage']}): {f_item['task']} -> {f_item['error']}")
    print(f"{'=' * 80}\n")

    return {
        "completed": completed_jobs,
        "skipped": skipped_jobs,
        "published": published_jobs,
        "failed": failed_jobs,
        "elapsed_sec": total_elapsed,
    }


def _collect_interactive_tasks(
    env_path: str | None = None,
) -> tuple[list[str], dict[str, dict[str, Any]], str]:
    """
    Интерактивный ввод задач из Asana или автоматическое чтение через src/magic_service (если доступен).
    Возвращает (task_strings, asana_tasks_meta, publish_mode),
    где publish_mode in ('none', 'confirm', 'auto', 'upload_only').
    """
    magic_service = _try_get_magic_service()
    has_asana_pat = bool(os.environ.get("ASANA_PAT", "").strip())

    print("\n" + "=" * 78)
    if magic_service is not None and has_asana_pat:
        print(" Доступна автозагрузка открытых задач из Asana (ASANA_PAT найден в .env):")
        print("   - Нажмите Enter сразу на [1], чтобы автоматически подтянуть задачи из Asana")
        print("   - Или введите задачи вручную (customer_name/site_name/date_number)")
    else:
        print(" Ввод задач из Asana (формат: customer_name/site_name/date_number)")
        print(" Нажмите Enter на пустой строке, чтобы завершить ввод и запустить конвейер")
    print("=" * 78)

    tasks: list[str] = []
    while True:
        try:
            line = input(
                f"[{len(tasks) + 1}] Введите задачу Asana (или Enter для продолжения): "
            ).strip()
        except EOFError:
            break
        if not line:
            break
        tasks.append(line)

    if not tasks and magic_service is not None and has_asana_pat:
        print("\n[Asana] Запрос открытых задач панорам, назначенных на вас...")
        asana_items = magic_service.fetch_my_asana_tasks(env_path=env_path)
        if not asana_items:
            print("[Asana] Открытых задач с панорамами не найдено.")
            return [], {}, "none"

        print(f"[Asana] Найдено открытых задач: {len(asana_items)}")
        meta_map: dict[str, dict[str, Any]] = {}
        for i, item in enumerate(asana_items, start=1):
            print(f"  {i}. {item['relative_dir']} (card_id: {item['card_id']})")
            tasks.append(item["task_string"])
            meta_map[item["relative_dir"]] = item

        print("\nВыберите режим работы с TraceAir Magic:")
        print("  [1] Собрать панорамы и спросить подтверждение перед публикацией (рекомендуется для QC)")
        print("  [2] Сквозной автопилот: скачать -> собрать -> сразу опубликовать в Magic в фоне")
        print("  [3] Только опубликовать уже готовые локальные файлы (без сшивки)")
        print("  [4] Только скачать и собрать локально (без публикации в Magic)")
        try:
            choice = input("Ваш выбор [1/2/3/4, по умолчанию 1]: ").strip()
        except EOFError:
            choice = "1"

        if choice == "2":
            return tasks, meta_map, "auto"
        if choice == "3":
            return tasks, meta_map, "upload_only"
        if choice == "4":
            return tasks, meta_map, "none"
        return tasks, meta_map, "confirm"

    return tasks, {}, "none"


def _run_post_stitch_confirm_publish(
    asana_meta_map: dict[str, dict[str, Any]],
    pano_root: str,
    env_path: str,
    ask_confirmation: bool = True,
) -> list[dict[str, Any]]:
    """
    После завершения сборки всех панорам находит первые основные файлы (<pano_id>_full.jpeg),
    при необходимости запрашивает подтверждение у пользователя в консоли и публикует их в Magic.
    """
    magic_service = _try_get_magic_service()
    if magic_service is None or not asana_meta_map:
        return []

    ready_to_publish: list[tuple[dict[str, Any], Any]] = []
    for rel_dir, meta in asana_meta_map.items():
        primary_file = magic_service.find_primary_panorama_file(
            pano_root,
            meta["customer"],
            meta["site"],
            meta["pano_id"],
        )
        if primary_file is not None:
            ready_to_publish.append((meta, primary_file))

    if not ready_to_publish:
        print("[Magic] Нет готовых локальных файлов панорам для публикации.")
        return []

    print(f"\n{'=' * 80}")
    print(f"[Magic QC] Готовы к публикации в TraceAir Magic ({len(ready_to_publish)} шт.):")
    for idx, (meta, fpath) in enumerate(ready_to_publish, start=1):
        print(f"  {idx}. {meta['relative_dir']} -> {fpath} (card_id: {meta['card_id']})")
    print(f"{'=' * 80}")

    if ask_confirmation and sys.stdin and sys.stdin.isatty():
        try:
            ans = input("Опубликовать эти панорамы в TraceAir Magic сейчас? [Y/n]: ").strip().lower()
        except EOFError:
            ans = "n"
        if ans in ("n", "no", "н", "нет"):
            print("[Magic] Публикация отменена пользователем. Вы можете запустить её позже флагом --upload-only.")
            return []

    published_results: list[dict[str, Any]] = []
    pending_futures: list[tuple[dict[str, Any], dict[str, Any], Any]] = []

    with ThreadPoolExecutor(max_workers=max(1, len(ready_to_publish))) as finalize_pool:
        for idx, (meta, fpath) in enumerate(ready_to_publish, start=1):
            print(f"\n--- [Выгрузка {idx}/{len(ready_to_publish)}] {meta['relative_dir']} ---")
            try:
                up_res = magic_service.upload_panorama_to_bo_storage(
                    task=meta,
                    local_file=fpath,
                    pano_root=pano_root,
                    env_path=env_path,
                )
                if up_res.get("status") in ("copying_started", "uploaded_pending_publication"):
                    fut = finalize_pool.submit(
                        magic_service.finalize_panorama_publication,
                        up_res,
                        60.0,
                    )
                    pending_futures.append((meta, up_res, fut))
                else:
                    published_results.append(up_res)
            except Exception as exc:
                print(f"  [ОШИБКА выгрузки] {meta['relative_dir']}: {exc}")

        if pending_futures:
            print(
                f"\n[Magic] Все файлы выгружены в bo-storage ({len(pending_futures)} шт.). "
                f"Ожидание серверной публикации и закрытия карточек (проверка каждые 60 сек.)..."
            )
            for meta, up_res, fut in pending_futures:
                try:
                    fin_res = fut.result()
                    fin_res["file"] = up_res.get("local_file", up_res.get("file", ""))
                    published_results.append(fin_res)
                except Exception as exc:
                    print(f"  [ОШИБКА финализации] {meta['relative_dir']}: {exc}")

    return published_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Асинхронный конвейер скачивания снимков из AWS S3, сборки панорам БПЛА и публикации в Magic"
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
        "--from-asana",
        action="store_true",
        help="Автоматически получить список открытых задач панорам из Asana (через ASANA_PAT в .env)",
    )
    parser.add_argument(
        "--auto-publish",
        action="store_true",
        help="Режим 1 (сквозной): сразу публиковать каждую сшитую панораму в TraceAir Magic в фоновом потоке",
    )
    parser.add_argument(
        "--confirm-publish",
        action="store_true",
        help="Режим 2 (с проверкой QC): после сшивки всех панорам спросить подтверждение перед отправкой в Magic",
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Только опубликовать уже готовые локальные файлы панорам по открытым задачам из Asana (без сшивки)",
    )
    parser.add_argument(
        "--airtable-only",
        action="store_true",
        help="Только назначить Owner ('Ilia Korostin') в Airtable для открытых задач из Asana",
    )
    parser.add_argument(
        "--no-airtable",
        action="store_true",
        help="Отключить автоназначение Owner в Airtable",
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Отключить публикацию в TraceAir Magic (только скачать из S3 и собрать локально)",
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
        help="Путь к файлу .env для записи/чтения ключей AWS, ASANA_PAT и TRACEAIR_SESS_ID",
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

    # 1. Загружаем конфигурацию и .env
    if os.path.isfile(args.env_file):
        load_dotenv_file(args.env_file, override=False)

    config: dict[str, Any] = {}
    if os.path.isfile(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    s3_cfg = config.get("s3_downloader", {})
    default_region = str(s3_cfg.get("default_region", DEFAULT_AWS_REGION))

    # Определяем корневую папку для сохранения панорам
    pano_root = args.pano_dir
    if not pano_root:
        for cand in ("/app/input", os.path.join(PROJECT_ROOT, "pano")):
            if os.path.isdir(cand):
                pano_root = cand
                break
        if not pano_root:
            pano_root = os.path.join(PROJECT_ROOT, "pano")
    os.makedirs(pano_root, exist_ok=True)

    # Если выбран режим только выгрузки готовых файлов (--upload-only) или только Airtable (--airtable-only),
    # AWS S3 и Hugin не нужны
    if args.upload_only or args.airtable_only:
        from scripts.publish_magic import run_standalone_publisher

        run_standalone_publisher(
            pano_root=pano_root,
            env_path=args.env_file,
            airtable_only=args.airtable_only,
            sync_airtable=not args.no_airtable,
        )
        return

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
    asana_meta_map: dict[str, dict[str, Any]] = {}
    publish_mode = "none"

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

    magic_service = _try_get_magic_service()
    if args.from_asana or args.auto_publish or args.confirm_publish:
        if magic_service is None:
            raise RuntimeError("Пакет src/magic_service не найден в проекте.")
        asana_items = magic_service.fetch_my_asana_tasks(env_path=args.env_file)
        for item in asana_items:
            asana_meta_map[item["relative_dir"]] = item
            if args.from_asana:
                tasks.append(item["task_string"])

    if not tasks:
        tasks, interactive_meta, publish_mode = _collect_interactive_tasks(env_path=args.env_file)
        asana_meta_map.update(interactive_meta)
        if publish_mode == "upload_only":
            from scripts.publish_magic import run_standalone_publisher

            run_standalone_publisher(
                pano_root=pano_root,
                env_path=args.env_file,
                sync_airtable=not args.no_airtable,
            )
            return

    if not tasks:
        print("[Конвейер] Список задач пуст. Завершение работы.")
        return

    # 3b. Автоматически проставляем Owner ('Ilia Korostin') в Airtable для всех открытых задач из Asana
    if (
        magic_service is not None
        and hasattr(magic_service, "sync_airtable_owners_for_tasks")
        and not args.no_airtable
        and (asana_meta_map or tasks)
    ):
        airtable_target_tasks = list(asana_meta_map.values()) if asana_meta_map else tasks
        try:
            magic_service.sync_airtable_owners_for_tasks(
                tasks=airtable_target_tasks,
                env_path=args.env_file,
            )
        except Exception as exc:
            print(f"[Airtable Предупреждение] Не удалось синхронизировать Owner в Airtable: {exc}")

    if args.no_publish:
        publish_mode = "none"
    elif args.auto_publish:
        publish_mode = "auto"
    elif args.confirm_publish:
        publish_mode = "confirm"
    elif args.from_asana and publish_mode == "none":
        publish_mode = "confirm"

    # 4. Определяем режим сохранения исходных снимков (keep_raw vs temp_raw)
    cfg_keep_raw = bool(s3_cfg.get("keep_raw_frames", True))
    if args.temp_raw:
        keep_raw = False
    elif args.keep_raw:
        keep_raw = True
    else:
        keep_raw = cfg_keep_raw

    # 5. Загружаем каталог шаблонов неба
    from scripts.index_skies import build_sky_catalog, resolve_sky_dir

    sky_dir = resolve_sky_dir(args.sky_dir)
    if os.path.isfile(args.catalog):
        with open(args.catalog, "r", encoding="utf-8") as f:
            catalog = json.load(f)
    else:
        catalog = build_sky_catalog(sky_dir=sky_dir, catalog_path=args.catalog, config=config)

    # 6. Запускаем асинхронный конвейер (boto3 загрузка + ProcessPoolExecutor сборка [+ автопубликация])
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
            asana_tasks_meta=asana_meta_map,
            auto_publish=(publish_mode == "auto"),
        )
    )

    # 7. Если выбран режим с подтверждением после сборки (confirm), запускаем публикацию после QC
    if publish_mode == "confirm" and asana_meta_map:
        _run_post_stitch_confirm_publish(
            asana_meta_map=asana_meta_map,
            pano_root=pano_root,
            env_path=args.env_file,
            ask_confirmation=True,
        )


if __name__ == "__main__":
    main()
