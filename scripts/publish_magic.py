#!/usr/bin/env python3
"""
Автономный CLI-сервис публикации готовых локальных панорам в TraceAir Magic по открытым задачам из Asana.
Не зависит от Hugin/OpenCV и может запускаться отдельно от пайплайна сшивки:
  - python scripts/publish_magic.py              (опубликовать все готовые локальные панорамы из задач Asana)
  - python scripts/publish_magic.py --list-only  (показать открытые задачи из Asana и наличие локальных файлов)

Архитектура конвейера выгрузки:
  - Фаза А (выгрузка файла в bo-storage + start-copying-panorama-image): идёт непрерывно для всех панорам подряд
    на максимальной скорости интернета, НЕ ожидая завершения серверной публикации предыдущей панорамы.
  - Фаза Б (фоновый поллер каждые 60 сек): параллельно в фоновых потоках проверяет статус 'panorama-publication'
    для всех уже загруженных панорам и закрывает шаги Quality Check и SLA по мере их готовности.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import os
import sys
from typing import Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.magic_service import (
    DEFAULT_POLL_INTERVAL_SEC,
    fetch_my_asana_tasks,
    finalize_panorama_publication,
    find_primary_panorama_file,
    sync_airtable_owners_for_tasks,
    upload_panorama_to_bo_storage,
)


def run_standalone_publisher(
    pano_root: str,
    env_path: str,
    list_only: bool = False,
    airtable_only: bool = False,
    sync_airtable: bool = True,
    filter_tasks: list[str] | None = None,
    include_already_uploaded: bool = False,
    poll_interval_sec: float = DEFAULT_POLL_INTERVAL_SEC,
) -> dict[str, Any]:
    print("=== 1. Чтение открытых задач из Asana ===")
    tasks = fetch_my_asana_tasks(
        env_path=env_path,
        include_already_uploaded=include_already_uploaded,
    )
    print(f"Найдено актуальных задач с панорамами в Asana: {len(tasks)}")

    if filter_tasks:
        norm_filters = {t.strip().lower() for t in filter_tasks if t.strip()}
        tasks = [
            t
            for t in tasks
            if t["relative_dir"].lower() in norm_filters
            or t["pano_id"].lower() in norm_filters
            or any(f in t["relative_dir"].lower() for f in norm_filters)
        ]

    airtable_summary: dict[str, Any] = {}
    if tasks and (sync_airtable or airtable_only) and not list_only:
        print("\n=== 1b. Назначение Owner в Airtable (Pano List) для открытых задач из Asana ===")
        try:
            airtable_summary = sync_airtable_owners_for_tasks(
                tasks=tasks,
                env_path=env_path,
            )
        except Exception as exc:
            print(f"[Airtable Предупреждение] Не удалось синхронизировать Owner в Airtable: {exc}")

    if airtable_only:
        return {
            "total_asana_tasks": len(tasks),
            "ready_count": 0,
            "airtable": airtable_summary,
            "published": [],
            "skipped": [],
            "failed": [],
        }

    ready_tasks: list[dict[str, Any]] = []
    missing_tasks: list[dict[str, Any]] = []

    for idx, t in enumerate(tasks, start=1):
        local_file = find_primary_panorama_file(pano_root, t["customer"], t["site"], t["pano_id"])
        if local_file:
            t["local_file"] = local_file
            ready_tasks.append(t)
            print(f"  [{idx}/{len(tasks)}] [ГОТОВО ЛОКАЛЬНО] {t['relative_dir']} -> {local_file}")
        else:
            missing_tasks.append(t)
            expected = os.path.join(pano_root, t["customer"], t["site"], t["pano_id"], f"{t['pano_id']}_full.jpeg")
            print(f"  [{idx}/{len(tasks)}] [НЕТ ФАЙЛА]       {t['relative_dir']} (ожидается: {expected})")

    if list_only or not ready_tasks:
        if not ready_tasks and not list_only:
            print("\nНет задач с готовыми локальными файлами для публикации.")
        return {
            "total_asana_tasks": len(tasks),
            "ready_count": len(ready_tasks),
            "published": [],
            "skipped": [],
            "failed": [],
        }

    print(
        f"\n=== 2. Непрерывная загрузка файлов в bo-storage ({len(ready_tasks)} шт.) "
        f"+ фоновая проверка публикации каждые {int(poll_interval_sec)}с ==="
    )
    published: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    # Пул фоновых поллеров (по одному на каждую загруженную карточку, чтобы они ждали 60с параллельно)
    with ThreadPoolExecutor(max_workers=max(4, len(ready_tasks))) as poll_pool:
        pending_finalizers: dict[Future[dict[str, Any]], dict[str, Any]] = {}

        # Фаза А: загружаем все файлы подряд на полной скорости интернета без ожидания публикации
        for idx, task in enumerate(ready_tasks, start=1):
            print(f"\n--- [Выгрузка {idx}/{len(ready_tasks)}] {task['relative_dir']} (card_id: {task['card_id']}) ---")
            try:
                upload_ctx = upload_panorama_to_bo_storage(
                    task=task,
                    local_file=task["local_file"],
                    pano_root=pano_root,
                    env_path=env_path,
                    interactive_prompt=True,
                )
                if upload_ctx.get("status") == "skipped_already_done":
                    skipped.append(upload_ctx)
                else:
                    fut = poll_pool.submit(
                        finalize_panorama_publication,
                        upload_ctx,
                        poll_interval_sec,
                    )
                    pending_finalizers[fut] = task
            except Exception as exc:
                print(f"  [ОШИБКА выгрузки] {task['relative_dir']}: {exc}")
                failed.append({"task": task["relative_dir"], "card_id": task["card_id"], "error": str(exc)})

        # Фаза Б: когда все файлы уже отправлены в bo-storage, дожидаемся завершения фоновых поллеров публикации
        if pending_finalizers:
            print(
                f"\n=== Все файлы отправлены в bo-storage! Ожидание завершения серверной публикации "
                f"({len(pending_finalizers)} шт., проверка каждые {int(poll_interval_sec)}с)... ==="
            )
            for fut in as_completed(pending_finalizers):
                task = pending_finalizers[fut]
                try:
                    res = fut.result()
                    published.append(res)
                except Exception as exc:
                    print(f"  [ОШИБКА публикации] {task['relative_dir']}: {exc}")
                    failed.append({"task": task["relative_dir"], "card_id": task["card_id"], "error": str(exc)})

    print(
        f"\n=== Итог публикации в Magic: Опубликовано: {len(published)} | "
        f"Пропущено (уже DONE): {len(skipped)} | Ошибок: {len(failed)} ==="
    )
    return {
        "total_asana_tasks": len(tasks),
        "ready_count": len(ready_tasks),
        "published": published,
        "skipped": skipped,
        "failed": failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Автономный сервис публикации готовых локальных панорам в TraceAir Magic по задачам из Asana.",
    )
    parser.add_argument(
        "--pano-dir",
        type=str,
        default=None,
        help="Корневая папка с панорамами (по умолчанию ./pano или /app/input)",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=os.path.join(PROJECT_ROOT, ".env"),
        help="Путь к файлу .env с ASANA_PAT и TRACEAIR_SESS_ID",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Только вывести открытые задачи из Asana и статус локальных файлов без публикации",
    )
    parser.add_argument(
        "--airtable-only",
        action="store_true",
        help="Только проставить Owner ('Ilia Korostin') в Airtable для открытых задач из Asana",
    )
    parser.add_argument(
        "--no-airtable",
        action="store_true",
        help="Отключить автоназначение Owner в Airtable при публикации",
    )
    parser.add_argument(
        "--include-uploaded",
        action="store_true",
        help="Включать задачи, в комментариях которых уже есть отметка о загрузке на office.traceair.net",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL_SEC,
        help=f"Интервал проверки статуса публикации в секундах (по умолчанию {int(DEFAULT_POLL_INTERVAL_SEC)}с)",
    )
    parser.add_argument(
        "--filter",
        nargs="*",
        default=None,
        help="Опциональный фильтр по подстроке customer/site/pano_id",
    )
    args = parser.parse_args()

    pano_root = args.pano_dir
    if not pano_root:
        for cand in ("/app/input", os.path.join(PROJECT_ROOT, "pano")):
            if os.path.isdir(cand):
                pano_root = cand
                break
        if not pano_root:
            pano_root = os.path.join(PROJECT_ROOT, "pano")

    run_standalone_publisher(
        pano_root=pano_root,
        env_path=args.env_file,
        list_only=args.list_only,
        airtable_only=args.airtable_only,
        sync_airtable=not args.no_airtable,
        filter_tasks=args.filter,
        include_already_uploaded=args.include_uploaded,
        poll_interval_sec=args.poll_interval,
    )


if __name__ == "__main__":
    main()
