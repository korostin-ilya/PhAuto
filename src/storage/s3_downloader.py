"""
Модуль управления сессией AWS SSO (.env) и многопоточной загрузки серий снимков из AWS S3 через boto3.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.parse import urlparse

DEFAULT_S3_BUCKET = "internals.traceair.net"
DEFAULT_AWS_REGION = "us-east-1"


def load_dotenv_file(env_path: str, override: bool = True) -> dict[str, str]:
    """
    Читает файл .env (поддерживает UTF-8, UTF-8-BOM и UTF-16 от PowerShell)
    и прописывает переменные в os.environ.
    """
    loaded: dict[str, str] = {}
    if not env_path or not os.path.isfile(env_path):
        return loaded

    raw_bytes = b""
    with open(env_path, "rb") as f:
        raw_bytes = f.read()

    if raw_bytes.startswith(b"\xff\xfe") or raw_bytes.startswith(b"\xfe\xff"):
        text = raw_bytes.decode("utf-16", errors="ignore")
    else:
        text = raw_bytes.decode("utf-8-sig", errors="ignore")

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        elif line.startswith("$env:"):
            line = line[len("$env:") :].strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
            val = val[1:-1]
        if key:
            loaded[key] = val
            if override or key not in os.environ or not os.environ[key]:
                os.environ[key] = val

    return loaded


def update_dotenv_vars(env_path: str, updates: dict[str, str]) -> str:
    """
    Обновляет или добавляет переданные переменные в файл .env,
    сохраняя все остальные уже существующие в .env переменные (например, ASANA_PAT, TRACEAIR_SESS_ID, AWS_*).
    Также синхронизирует обновлённые значения с os.environ.
    """
    abs_env = os.path.abspath(env_path)
    os.makedirs(os.path.dirname(abs_env), exist_ok=True)

    existing: dict[str, str] = {}
    if os.path.isfile(abs_env):
        existing = load_dotenv_file(abs_env, override=False)

    for k, v in updates.items():
        clean_k = str(k).strip()
        if not clean_k:
            continue
        clean_v = str(v).strip()
        if clean_v:
            existing[clean_k] = clean_v
            os.environ[clean_k] = clean_v
        else:
            existing.pop(clean_k, None)
            os.environ.pop(clean_k, None)

    aws_order = (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_DEFAULT_REGION",
    )
    aws_lines: list[str] = []
    for k in aws_order:
        if k in existing and existing[k]:
            aws_lines.append(f"{k}={existing[k]}")

    other_lines: list[str] = []
    for k, v in existing.items():
        if k not in aws_order and v:
            formatted_v = f'"{v}"' if (" " in v or ";" in v) and not (v.startswith('"') and v.endswith('"')) else v
            other_lines.append(f"{k}={formatted_v}")

    output_blocks: list[str] = []
    if aws_lines:
        output_blocks.append("\n".join(aws_lines))
    if other_lines:
        output_blocks.append("\n".join(other_lines))

    content = "\n\n".join(output_blocks) + ("\n" if output_blocks else "")
    with open(abs_env, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)

    return abs_env


def write_dotenv_file(env_path: str, creds: dict[str, str]) -> str:
    """
    Записывает ключи AWS в корневой файл .env в формате:
        AWS_ACCESS_KEY_ID=ASIA...
        AWS_SECRET_ACCESS_KEY=...
        AWS_SESSION_TOKEN=IQoJb3JpZ2luX2VjE...
        AWS_DEFAULT_REGION=us-east-1
    сохраняя при этом все остальные переменные в .env (ASANA_PAT, TRACEAIR_SESS_ID и др.),
    и обновляет os.environ.
    """
    access_key = creds.get("AWS_ACCESS_KEY_ID", "").strip()
    secret_key = creds.get("AWS_SECRET_ACCESS_KEY", "").strip()
    session_token = creds.get("AWS_SESSION_TOKEN", "").strip()
    region = creds.get("AWS_DEFAULT_REGION", DEFAULT_AWS_REGION).strip() or DEFAULT_AWS_REGION

    return update_dotenv_vars(
        env_path,
        {
            "AWS_ACCESS_KEY_ID": access_key,
            "AWS_SECRET_ACCESS_KEY": secret_key,
            "AWS_SESSION_TOKEN": session_token,
            "AWS_DEFAULT_REGION": region,
        },
    )


def _clean_aws_env(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """
    Возвращает копию переменных окружения без временных/статических ключей AWS,
    чтобы вызовы AWS CLI всегда читали актуальную сессию из SSO-кэша / профиля,
    а не протухшие ключи из os.environ или старого .env.
    """
    env = dict(os.environ if extra_env is None else extra_env)
    for k in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
    ):
        env.pop(k, None)
    return env


def _export_credentials_via_aws_cli(
    profile: str | None = None,
    default_region: str = DEFAULT_AWS_REGION,
) -> dict[str, str]:
    """Извлекает временные ключи из активной сессии AWS CLI (aws configure export-credentials)."""
    aws_bin = shutil.which("aws")
    if not aws_bin:
        raise FileNotFoundError("Утилита 'aws' (AWS CLI v2) не найдена в PATH.")

    profile_args = ["--profile", profile] if profile else []
    clean_env = _clean_aws_env()

    # 1. Пробуем JSON-вывод aws configure export-credentials (--format process)
    cmd_json = [aws_bin, "configure", "export-credentials", *profile_args, "--format", "process", "--no-cli-pager"]
    res_json = subprocess.run(
        cmd_json,
        env=clean_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if res_json.returncode == 0 and res_json.stdout.strip():
        stdout = res_json.stdout.strip()
        first_brace = stdout.find("{")
        last_brace = stdout.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            try:
                data = json.loads(stdout[first_brace : last_brace + 1])
                access_key = str(data.get("AccessKeyId", "")).strip()
                secret_key = str(data.get("SecretAccessKey", "")).strip()
                session_token = str(data.get("SessionToken", "")).strip()
                if access_key and secret_key:
                    return {
                        "AWS_ACCESS_KEY_ID": access_key,
                        "AWS_SECRET_ACCESS_KEY": secret_key,
                        "AWS_SESSION_TOKEN": session_token,
                        "AWS_DEFAULT_REGION": os.environ.get("AWS_DEFAULT_REGION", default_region),
                    }
            except Exception:
                pass

    # 2. Fallback: текстовый формат --format env-no-export
    cmd_env = [aws_bin, "configure", "export-credentials", *profile_args, "--format", "env-no-export", "--no-cli-pager"]
    res_env = subprocess.run(
        cmd_env,
        env=clean_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if res_env.returncode == 0 and res_env.stdout.strip():
        creds: dict[str, str] = {"AWS_DEFAULT_REGION": default_region}
        for line in res_env.stdout.splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                creds[k.strip()] = v.strip().strip('"').strip("'")
        if creds.get("AWS_ACCESS_KEY_ID") and creds.get("AWS_SECRET_ACCESS_KEY"):
            return creds

    # 3. Fallback: прямое чтение через botocore (если установлен)
    try:
        import botocore.session
        b_session = botocore.session.Session(profile=profile)
        b_creds = b_session.get_credentials()
        if b_creds:
            frozen = b_creds.get_frozen_credentials()
            if frozen and frozen.access_key and frozen.secret_key:
                return {
                    "AWS_ACCESS_KEY_ID": frozen.access_key,
                    "AWS_SECRET_ACCESS_KEY": frozen.secret_key,
                    "AWS_SESSION_TOKEN": frozen.token or "",
                    "AWS_DEFAULT_REGION": b_session.get_config_variable("region") or default_region,
                }
    except Exception:
        pass

    raise RuntimeError(
        f"Не удалось выгрузить временные ключи через 'aws configure export-credentials': "
        f"{res_json.stderr.strip() or res_env.stderr.strip()}"
    )


def ensure_aws_env_credentials(
    env_path: str,
    default_region: str = DEFAULT_AWS_REGION,
    profile: str | None = None,
    force_refresh: bool = False,
) -> dict[str, str]:
    """
    Проверяет активность сессии AWS:
    - Если запущено на хосте с AWS CLI: проверяет 'aws sts get-caller-identity'.
      Если токены истекли — запускает 'aws sso login', после чего выгружает
      AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN, AWS_DEFAULT_REGION в .env.
    - Если запущено внутри контейнера (где нет AWS CLI, но есть .env): загружает .env
      и проверяет валидность ключей через boto3 STS.
    """
    if not force_refresh and os.path.isfile(env_path):
        load_dotenv_file(env_path, override=False)

    aws_bin = shutil.which("aws")
    profile_args = ["--profile", profile] if profile else []

    if aws_bin:
        clean_env = _clean_aws_env()
        check_cmd = [aws_bin, "sts", "get-caller-identity", *profile_args, "--no-cli-pager"]
        res = subprocess.run(
            check_cmd,
            env=clean_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if res.returncode != 0 or force_refresh:
            print("[AWS Auth] Сессия AWS SSO не найдена или токены истекли. Запуск 'aws sso login'...")
            login_cmd = [aws_bin, "sso", "login", *profile_args]
            login_res = subprocess.run(login_cmd, env=clean_env)
            if login_res.returncode != 0:
                raise RuntimeError("Ошибка авторизации 'aws sso login'.")

        creds = _export_credentials_via_aws_cli(profile=profile, default_region=default_region)
        saved_path = write_dotenv_file(env_path, creds)
        key_preview = creds["AWS_ACCESS_KEY_ID"][:8] + "..." if creds.get("AWS_ACCESS_KEY_ID") else "N/A"
        print(f"[AWS Auth] Сессия активна. Ключи ({key_preview}) записаны в '{saved_path}'")
        return creds

    # Сценарий внутри Docker-контейнера (без установленного AWS CLI, ключи переданы через .env)
    load_dotenv_file(env_path, override=False)
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "").strip()
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip()
    session_token = os.environ.get("AWS_SESSION_TOKEN", "").strip()
    region = os.environ.get("AWS_DEFAULT_REGION", default_region).strip() or default_region

    if not access_key or not secret_key:
        raise RuntimeError(
            f"Не найдены ключи AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY ни в окружении, ни в '{env_path}'. "
            f"Сначала выполните вход через AWS SSO на хосте."
        )

    return {
        "AWS_ACCESS_KEY_ID": access_key,
        "AWS_SECRET_ACCESS_KEY": secret_key,
        "AWS_SESSION_TOKEN": session_token,
        "AWS_DEFAULT_REGION": region,
    }


def parse_asana_task_string(
    raw_str: str,
    default_bucket: str = DEFAULT_S3_BUCKET,
) -> dict[str, str]:
    """
    Разбирает входную строку задачи из Asana (или полный S3 URI).
    Корректно очищает префиксы 'Panorama: ', 'anorama: ', пробелы и двоеточия
    (полная обратная совместимость с 1.2.2 pano-downloader_magic — v3.bat).

    Примеры поддерживаемых форматов:
      - 'Panorama: ashton-woods---austin/lagos/2026-10-31_01'
      - 'ashton-woods---austin/lagos/2026-10-31_01'
      - 's3://internals.traceair.net/ashton-woods---austin/lagos/panoramas/2026-10-31_01/input'
    """
    s = raw_str.strip()
    if not s:
        raise ValueError("Пустая строка задачи Asana.")

    # Если передан полный S3 URI: s3://bucket/customer/site/panoramas/date[/input]
    if s.lower().startswith("s3://"):
        parsed = urlparse(s)
        bucket = parsed.netloc or default_bucket
        path_clean = parsed.path.strip("/")
        m = re.match(r"^([^/]+)/([^/]+)/panoramas/([^/]+)(?:/input)?/?$", path_clean)
        if m:
            customer, site, date_number = m.group(1), m.group(2), m.group(3)
            s3_prefix = f"{customer}/{site}/panoramas/{date_number}/input/"
        else:
            parts = [p for p in path_clean.split("/") if p]
            if len(parts) >= 3:
                customer, site, date_number = parts[0], parts[1], parts[-1]
            else:
                customer, site, date_number = "custom", "site", (parts[-1] if parts else "pano")
            s3_prefix = path_clean.rstrip("/") + "/"
        return {
            "raw_input": raw_str,
            "bucket": bucket,
            "customer": customer,
            "site": site,
            "date": date_number,
            "s3_prefix": s3_prefix,
            "s3_uri": f"s3://{bucket}/{s3_prefix.rstrip('/')}",
            "relative_dir": f"{customer}/{site}/{date_number}",
        }

    # Очистка префиксов вида "For Panorama ", "Panorama: ", "anorama: ", ": " и пробелов
    s = re.sub(r"^(?:for\s+)?[pP]?a?n?o?r?a?m?a?:?\s*", "", s, flags=re.IGNORECASE)
    s = s.lstrip(": ").strip()
    s = s.replace("\\", "/").replace(" ", "")

    parts = [p for p in s.split("/") if p]
    if len(parts) >= 4 and parts[2].lower() == "panoramas":
        customer, site, date_number = parts[0], parts[1], parts[3]
    elif len(parts) >= 3:
        customer, site, date_number = parts[0], parts[1], parts[2]
    else:
        raise ValueError(
            f"Некорректный формат задачи '{raw_str}'. "
            f"Ожидается 'customer_name/site_name/date_number' (например, 'lennar/boise/2026-10-31_01')."
        )

    s3_prefix = f"{customer}/{site}/panoramas/{date_number}/input/"
    return {
        "raw_input": raw_str,
        "bucket": default_bucket,
        "customer": customer,
        "site": site,
        "date": date_number,
        "s3_prefix": s3_prefix,
        "s3_uri": f"s3://{default_bucket}/{s3_prefix.rstrip('/')}",
        "relative_dir": f"{customer}/{site}/{date_number}",
    }


def _create_boto3_s3_client(max_threads: int = 8) -> Any:
    """Создает клиент boto3 S3 с пулом соединений под многопоточную загрузку."""
    import boto3
    from botocore.config import Config

    region = os.environ.get("AWS_DEFAULT_REGION", DEFAULT_AWS_REGION) or DEFAULT_AWS_REGION
    cfg = Config(
        region_name=region,
        max_pool_connections=max(16, max_threads * 2),
        retries={"max_attempts": 3, "mode": "standard"},
    )
    return boto3.client("s3", region_name=region, config=cfg)


def download_series_boto3(
    task_info: dict[str, str],
    dest_dir: str,
    max_retries: int = 3,
    max_threads: int = 8,
    env_path: str | None = None,
    s3_client: Any = None,
) -> dict[str, Any]:
    """
    Скачивает все файлы серии из S3 (task_info['s3_uri']) в локальную папку dest_dir через boto3.
    - Поддерживает параллельное скачивание в max_threads потоков.
    - Сверяет точный размер каждого файла в байтах (как в оригинальном батнике).
    - При истечении временного токена (ExpiredToken) и наличии AWS CLI автоматически обновляет .env.
    """
    t0 = time.time()
    bucket = task_info["bucket"]
    prefix = task_info["s3_prefix"]
    s3_uri = task_info["s3_uri"]

    abs_dest = os.path.abspath(dest_dir)
    os.makedirs(abs_dest, exist_ok=True)

    if env_path and os.path.isfile(env_path):
        load_dotenv_file(env_path, override=False)

    client = s3_client or _create_boto3_s3_client(max_threads=max_threads)

    def _list_remote_objects(c: Any) -> list[dict[str, Any]]:
        paginator = c.get_paginator("list_objects_v2")
        objs: list[dict[str, Any]] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                key = str(item.get("Key", ""))
                size = int(item.get("Size", 0))
                if not key or key.endswith("/") or size <= 0:
                    continue
                objs.append({"Key": key, "Size": size})
        return objs

    try:
        remote_objects = _list_remote_objects(client)
    except Exception as exc:
        err_str = str(exc)
        if (
            any(tok in err_str for tok in ("ExpiredToken", "InvalidClientTokenId", "TokenRefreshRequired"))
            and env_path
            and shutil.which("aws")
        ):
            print("[S3 Загрузчик] Токен истек, обновляем сессию AWS SSO и .env...")
            ensure_aws_env_credentials(env_path=env_path, force_refresh=True)
            client = _create_boto3_s3_client(max_threads=max_threads)
            remote_objects = _list_remote_objects(client)
        else:
            raise

    if not remote_objects:
        raise FileNotFoundError(
            f"По адресу '{s3_uri}' в S3 не найдено ни одного файла."
        )

    remote_total_bytes = sum(obj["Size"] for obj in remote_objects)
    remote_mb = round(remote_total_bytes / (1024 * 1024), 1)
    print(
        f"[S3 Загрузчик]  {s3_uri} -> найдено файлов: {len(remote_objects)} "
        f"(общий объем: {remote_mb} МБ)"
    )

    def _download_single_object(obj: dict[str, Any]) -> tuple[str, int]:
        key = obj["Key"]
        expected_size = obj["Size"]
        rel_path = key[len(prefix) :].lstrip("/") if key.startswith(prefix) else os.path.basename(key)
        if not rel_path:
            rel_path = os.path.basename(key)
        local_file = os.path.join(abs_dest, rel_path)
        os.makedirs(os.path.dirname(local_file), exist_ok=True)

        if os.path.isfile(local_file) and os.path.getsize(local_file) == expected_size:
            return local_file, expected_size

        tmp_part = f"{local_file}.part.{os.getpid()}"
        try:
            client.download_file(bucket, key, tmp_part)
            actual_size = os.path.getsize(tmp_part) if os.path.isfile(tmp_part) else -1
            if actual_size != expected_size:
                raise IOError(
                    f"Размер скачанного файла '{rel_path}' ({actual_size} Б) "
                    f"не совпадает с S3 ({expected_size} Б)"
                )
            os.replace(tmp_part, local_file)
            return local_file, actual_size
        finally:
            if os.path.exists(tmp_part):
                try:
                    os.remove(tmp_part)
                except OSError:
                    pass

    local_total_bytes = 0
    downloaded_paths: list[str] = []
    last_error: Exception | None = None

    for attempt in range(1, max(1, max_retries) + 1):
        downloaded_paths = []
        local_total_bytes = 0
        errors: list[Exception] = []

        with ThreadPoolExecutor(max_workers=max(1, max_threads)) as pool:
            futures = {pool.submit(_download_single_object, obj): obj for obj in remote_objects}
            for fut in as_completed(futures):
                try:
                    l_path, f_bytes = fut.result()
                    downloaded_paths.append(l_path)
                    local_total_bytes += f_bytes
                except Exception as e:
                    errors.append(e)
                    last_error = e

        local_mb = round(local_total_bytes / (1024 * 1024), 1)
        pct = round((local_total_bytes / remote_total_bytes) * 100.0, 1) if remote_total_bytes > 0 else 100.0
        print(
            f"[S3 Загрузчик]  Попытка {attempt}/{max_retries}: { task_info['relative_dir'] } | "
            f"S3={remote_mb} МБ, Локально={local_mb} МБ ({pct}%)"
        )

        if not errors and local_total_bytes == remote_total_bytes:
            break

        if attempt < max_retries:
            print(f"[S3 Загрузчик]  Несовпадение размера или ошибка сети ({last_error}). Повтор через 3 сек...")
            time.sleep(3.0)
        else:
            raise RuntimeError(
                f"Не удалось полностью скачать серию '{s3_uri}' за {max_retries} попыток. "
                f"Скачано {local_total_bytes} из {remote_total_bytes} байт. Последняя ошибка: {last_error}"
            )

    elapsed = time.time() - t0
    speed_mbps = (remote_mb / elapsed) if elapsed > 0 else 0.0
    print(
        f"[S3 Загрузчик]  Готово: '{task_info['relative_dir']}' ({len(downloaded_paths)} файлов, "
        f"{remote_mb} МБ за {elapsed:.1f}с, {speed_mbps:.1f} МБ/с)"
    )

    return {
        "s3_uri": s3_uri,
        "dest_dir": abs_dest,
        "files_count": len(downloaded_paths),
        "remote_bytes": remote_total_bytes,
        "local_bytes": local_total_bytes,
        "remote_mb": remote_mb,
        "elapsed_sec": round(elapsed, 2),
        "attempts": attempt,
    }


if __name__ == "__main__":
    import argparse

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    parser = argparse.ArgumentParser(
        description="Проверка сессии AWS SSO и экспорт временных ключей в .env"
    )
    parser.add_argument(
        "--env-file",
        default=os.path.join(project_root, ".env"),
        help="Путь к файлу .env",
    )
    parser.add_argument("--profile", default=None, help="Имя профиля AWS SSO")
    parser.add_argument("--region", default=DEFAULT_AWS_REGION, help="Регион AWS")
    parser.add_argument("--force", action="store_true", help="Принудительный sso login")
    parsed_args = parser.parse_args()

    ensure_aws_env_credentials(
        env_path=parsed_args.env_file,
        default_region=parsed_args.region,
        profile=parsed_args.profile,
        force_refresh=parsed_args.force,
    )
