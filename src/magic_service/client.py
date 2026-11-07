"""
Клиент интеграции с Asana API и внутренними сервисами TraceAir Magic:
- Чтение назначенных открытых задач из Asana (ASANA_PAT из .env)
- Получение JWT Bearer-токена площадки через login.traceair.net/api/me (TRACEAIR_SESS_ID из .env)
- Поиск первой основной локальной панорамы (<pano_id>_full.jpeg)
- Multipart-загрузка панорамы в bo-storage.traceair.net и закрытие шагов карточки в magic-ssm / magic-action-runner
  (без закрытия самой задачи в Asana).

Модуль полностью автономен и использует стандартную библиотеку Python (urllib.request),
поэтому не требует пересборки Docker-образа или установки дополнительных пакетов.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

DEFAULT_CHUNK_SIZE = 5 * 1024 * 1024  # 5 МБ
DEFAULT_POLL_INTERVAL_SEC = 60.0       # Проверка статуса публикации каждые 60 сек

PANO_TASK_REGEX = re.compile(
    r"(?:For\s+Panorama|Panorama:?)\s+([^/\s]+)/([^/\s]+)/([^\s]+)",
    re.IGNORECASE,
)
MAGIC_LINK_REGEX = re.compile(
    r"https://magic\.traceair\.net/([^/\s]+)/([^/\s]+)/"
    r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)
JWT_TOKEN_REGEX = re.compile(r"(eyJ[A-Za-z0-9\-_]+\.eyJ[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+)")


class TraceAirSessionExpiredError(RuntimeError):
    """Исключение при истечении сессионной куки TRACEAIR_SESS_ID (HTTP 401/403)."""


class _HttpResponse:
    """Унифицированная обёртка HTTP-ответа поверх urllib.request."""

    def __init__(self, status_code: int, body_bytes: bytes, headers: dict[str, str]) -> None:
        self.status_code = status_code
        self.content = body_bytes
        self.headers = headers

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> Any:
        return json.loads(self.text)

    def get_header(self, name: str, default: str | None = None) -> str | None:
        target = name.lower()
        for k, v in self.headers.items():
            if k.lower() == target:
                return v
        return default

    def raise_for_status(self) -> None:
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status_code}: {self.text[:500]}")


def _http_request(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
    json_payload: Any = None,
    data: bytes | None = None,
    timeout: float = 60.0,
) -> _HttpResponse:
    """Выполняет HTTP-запрос через стандартный urllib.request."""
    full_url = url
    if params:
        query_str = urllib.parse.urlencode(params)
        sep = "&" if "?" in full_url else "?"
        full_url = f"{full_url}{sep}{query_str}"

    req_headers = dict(headers or {})
    body_bytes = data
    if json_payload is not None:
        body_bytes = json.dumps(json_payload).encode("utf-8")
        if not any(k.lower() == "content-type" for k in req_headers):
            req_headers["Content-Type"] = "application/json"

    req = urllib.request.Request(full_url, data=body_bytes, headers=req_headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp_bytes = resp.read()
            resp_headers = {k: v for k, v in resp.info().items()}
            return _HttpResponse(status_code=resp.getcode(), body_bytes=resp_bytes, headers=resp_headers)
    except urllib.error.HTTPError as err:
        err_bytes = err.read() if err.fp else b""
        err_headers = {k: v for k, v in err.headers.items()} if err.headers else {}
        return _HttpResponse(status_code=err.code, body_bytes=err_bytes, headers=err_headers)


def _read_env_var(key: str, env_path: str | None = None) -> str:
    """Читает переменную из os.environ или из файла .env."""
    val = os.environ.get(key, "").strip()
    if val:
        return val
    if env_path and os.path.isfile(env_path):
        with open(env_path, "rb") as f:
            raw = f.read().decode("utf-8-sig", errors="ignore")
        for line in raw.splitlines():
            clean = line.strip()
            if not clean or clean.startswith("#") or "=" not in clean:
                continue
            k, v = clean.split("=", 1)
            if k.strip() == key:
                v_clean = v.strip().strip('"').strip("'")
                os.environ[key] = v_clean
                return v_clean
    return ""


def _save_env_var(key: str, value: str, env_path: str | None = None) -> None:
    """Сохраняет обновлённую переменную (например, TRACEAIR_SESS_ID) в .env без затирания других ключей."""
    os.environ[key] = value.strip()
    if not env_path:
        return
    try:
        from src.storage.s3_downloader import update_dotenv_vars

        update_dotenv_vars(env_path, {key: value.strip()})
    except Exception:
        # Автономный fallback, если src.storage не используется
        abs_env = os.path.abspath(env_path)
        lines: list[str] = []
        found = False
        if os.path.isfile(abs_env):
            with open(abs_env, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.strip().startswith(f"{key}="):
                        lines.append(f"{key}={value.strip()}\n")
                        found = True
                    else:
                        lines.append(line)
        if not found:
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            lines.append(f"{key}={value.strip()}\n")
        with open(abs_env, "w", encoding="utf-8", newline="\n") as f:
            f.writelines(lines)


# =====================================================================
# 1. ПОЛУЧЕНИЕ ТОКЕНА TRACEAIR ЧЕРЕЗ LOGIN.TRACEAIR.NET
# =====================================================================

def fetch_traceair_token(
    customer: str,
    site: str,
    sess_id: str | None = None,
    env_path: str | None = None,
    interactive_prompt: bool = True,
) -> str:
    """
    Отправляет запрос на https://login.traceair.net/api/me с кукой traceair_sess_id
    и получает свежий JWT Bearer-токен для конкретного клиента и площадки.
    Если сессия истекла (401/403) и interactive_prompt=True — просит ввести новую куку и сохраняет в .env.
    """
    effective_sess_id = (sess_id or _read_env_var("TRACEAIR_SESS_ID", env_path)).strip()
    if effective_sess_id.startswith("traceair_sess_id="):
        effective_sess_id = effective_sess_id[len("traceair_sess_id=") :].strip()

    for attempt in range(2):
        if not effective_sess_id:
            if interactive_prompt and sys.stdin and sys.stdin.isatty():
                print("\n[Magic Auth] Переменная TRACEAIR_SESS_ID не задана в .env.")
                effective_sess_id = input("Вставьте значение куки traceair_sess_id: ").strip()
                if effective_sess_id.startswith("traceair_sess_id="):
                    effective_sess_id = effective_sess_id[len("traceair_sess_id=") :].strip()
                if effective_sess_id:
                    _save_env_var("TRACEAIR_SESS_ID", effective_sess_id, env_path)
            if not effective_sess_id:
                raise TraceAirSessionExpiredError(
                    "Не задан TRACEAIR_SESS_ID в .env. Добавьте TRACEAIR_SESS_ID=... в корневой .env."
                )

        url = "https://login.traceair.net/api/me"
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Content-Type": "application/json",
            "Cookie": f"traceair_sess_id={effective_sess_id}",
            "Origin": "https://magic.traceair.net",
            "Referer": "https://magic.traceair.net/",
        }
        payload = {"customer": customer, "site": site}

        resp = _http_request("POST", url, headers=headers, json_payload=payload, timeout=20.0)
        if resp.status_code in (401, 403):
            if attempt == 0 and interactive_prompt and sys.stdin and sys.stdin.isatty():
                print("\n[Magic Auth] Сессия TRACEAIR_SESS_ID истекла (HTTP 401/403).")
                new_cookie = input("Вставьте свежее значение куки traceair_sess_id из браузера: ").strip()
                if new_cookie.startswith("traceair_sess_id="):
                    new_cookie = new_cookie[len("traceair_sess_id=") :].strip()
                if new_cookie:
                    effective_sess_id = new_cookie
                    _save_env_var("TRACEAIR_SESS_ID", effective_sess_id, env_path)
                    print("[Magic Auth] Новый TRACEAIR_SESS_ID сохранён в .env. Повтор запроса...")
                    continue
            raise TraceAirSessionExpiredError(
                "Сессия TRACEAIR_SESS_ID истекла (401/403). Обновите TRACEAIR_SESS_ID в файле .env."
            )

        resp.raise_for_status()

        combined_response = f"{resp.text}\n{resp.headers}"
        match = JWT_TOKEN_REGEX.search(combined_response)
        if not match:
            raise RuntimeError(f"Сервер login.traceair.net не вернул JWT-токен. Ответ: {resp.text[:300]}")

        return match.group(1)

    raise TraceAirSessionExpiredError("Не удалось авторизоваться на login.traceair.net.")


# =====================================================================
# 2. ЧТЕНИЕ И ПАРСИНГ ЗАДАЧ ИЗ ASANA
# =====================================================================

def parse_asana_tasks_payload(
    raw_tasks: list[dict[str, Any]],
    include_already_uploaded: bool = False,
) -> list[dict[str, Any]]:
    """
    Извлекает из списка задач Asana открытые задачи панорам,
    содержащие идентификатор съёмки (customer/site/pano_id) и ссылку на карточку Magic (card_id)
    в заголовке, описании (notes) или комментариях (stories).
    По умолчанию пропускает задачи, в комментариях которых уже есть отметка
    'Panorama uploaded to https://office.traceair.net'.
    """
    parsed_tasks: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for task in raw_tasks:
        if task.get("completed"):
            continue

        stories_list = task.get("stories") or []
        stories_text = "\n".join(
            str(s.get("text", "")) for s in stories_list if isinstance(s, dict) and s.get("text")
        )
        full_text = f"{task.get('name', '')}\n{task.get('notes', '')}\n{stories_text}"

        already_uploaded = "panorama uploaded to https://office.traceair.net" in full_text.lower()
        if already_uploaded and not include_already_uploaded:
            continue

        pano_match = PANO_TASK_REGEX.search(full_text)
        link_match = MAGIC_LINK_REGEX.search(full_text)

        if pano_match and link_match:
            customer = pano_match.group(1).strip()
            site = pano_match.group(2).strip()
            pano_id = pano_match.group(3).strip()
            card_id = link_match.group(3).strip()
            rel_dir = f"{customer}/{site}/{pano_id}"

            dedup_key = f"{rel_dir}::{card_id}"
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            parsed_tasks.append(
                {
                    "asana_gid": str(task.get("gid", "")),
                    "name": str(task.get("name", "")).strip(),
                    "customer": customer,
                    "site": site,
                    "pano_id": pano_id,
                    "date": pano_id,
                    "card_id": card_id,
                    "relative_dir": rel_dir,
                    "task_string": f"Panorama: {customer}/{site}/{pano_id}",
                    "already_uploaded": already_uploaded,
                }
            )

    return parsed_tasks


def fetch_my_asana_tasks(
    asana_pat: str | None = None,
    env_path: str | None = None,
    include_already_uploaded: bool = False,
) -> list[dict[str, Any]]:
    """
    Запрашивает все открытые задачи панорам в Asana, назначенные лично на пользователя,
    и параллельно подгружает комментарии (stories) для извлечения ссылки на карточку Magic (card_id).
    """
    token = (asana_pat or _read_env_var("ASANA_PAT", env_path)).strip()
    if not token:
        raise RuntimeError(
            "Не найден токен ASANA_PAT ни в окружении, ни в .env. Добавьте ASANA_PAT=... в корневой .env."
        )

    asana_headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }

    me_resp = _http_request("GET", "https://app.asana.com/api/1.0/users/me", headers=asana_headers, timeout=20.0)
    me_resp.raise_for_status()
    workspaces = me_resp.json().get("data", {}).get("workspaces", [])

    candidate_tasks: list[dict[str, Any]] = []
    for ws in workspaces:
        resp = _http_request(
            "GET",
            "https://app.asana.com/api/1.0/tasks",
            headers=asana_headers,
            params={
                "assignee": "me",
                "workspace": str(ws["gid"]),
                "completed_since": "now",
                "opt_fields": "gid,name,notes,completed",
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        for t in resp.json().get("data", []):
            if t.get("completed"):
                continue
            text_head = f"{t.get('name', '')}\n{t.get('notes', '')}"
            if PANO_TASK_REGEX.search(text_head):
                candidate_tasks.append(t)

    def _enrich_task_with_stories(task_obj: dict[str, Any]) -> dict[str, Any]:
        gid = str(task_obj.get("gid", "")).strip()
        if not gid:
            return task_obj
        try:
            st_resp = _http_request(
                "GET",
                f"https://app.asana.com/api/1.0/tasks/{gid}/stories",
                headers=asana_headers,
                params={"opt_fields": "text,type,resource_subtype"},
                timeout=20.0,
            )
            if st_resp.ok:
                task_obj = {**task_obj, "stories": st_resp.json().get("data", [])}
        except Exception:
            pass
        return task_obj

    if candidate_tasks:
        with ThreadPoolExecutor(max_workers=min(8, len(candidate_tasks))) as pool:
            candidate_tasks = list(pool.map(_enrich_task_with_stories, candidate_tasks))

    return parse_asana_tasks_payload(candidate_tasks, include_already_uploaded=include_already_uploaded)


def find_primary_panorama_file(
    pano_root: str | Path,
    customer: str,
    site: str,
    pano_id: str,
) -> Path | None:
    """
    Ищет первую основную сшитую панораму по пути:
      <pano_root>/<customer>/<site>/<pano_id>/<pano_id>_full.jpeg (или .jpg)
    Игнорирует возможные вторые серии (<pano_id>_2_full.jpeg).
    """
    pano_dir = Path(pano_root) / customer / site / pano_id
    for ext in ("jpeg", "jpg"):
        candidate = pano_dir / f"{pano_id}_full.{ext}"
        if candidate.is_file() and candidate.stat().st_size > 100_000:
            return candidate
    return None


# =====================================================================
# 3. ПУБЛИКАЦИЯ ПАНОРАМЫ В MAGIC (BO-STORAGE + MAGIC-SSM + ACTION-RUNNER)
# =====================================================================

def get_card_state(customer: str, site: str, card_id: str, headers: dict[str, str]) -> dict[str, Any]:
    state_url = f"https://magic-ssm.traceair.net/api/v1/{customer}/{site}/{card_id}/get-state"
    resp = _http_request("GET", state_url, headers=headers, timeout=20.0)
    if resp.status_code == 405:
        resp = _http_request("POST", state_url, headers=headers, json_payload={}, timeout=20.0)
    resp.raise_for_status()
    return resp.json()


def run_card_action(
    customer: str,
    site: str,
    card_id: str,
    action_name: str,
    headers: dict[str, str],
    payload: dict[str, Any] | None = None,
) -> _HttpResponse:
    action_url = (
        f"https://magic-action-runner.traceair.net/api/v1/"
        f"{customer}/{site}/{card_id}/run-action/{action_name}"
    )
    resp = _http_request(
        "POST",
        action_url,
        headers={**headers, "Content-Type": "application/json"},
        json_payload=payload if payload is not None else {},
        timeout=30.0,
    )
    if not resp.ok:
        raise RuntimeError(f"Ошибка {action_name} (HTTP {resp.status_code}): {resp.text[:500]}")
    return resp


def upload_panorama_to_bo_storage(
    task: dict[str, Any],
    local_file: str | Path | None = None,
    pano_root: str | Path | None = None,
    sess_id: str | None = None,
    env_path: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_chunk_retries: int = 3,
    interactive_prompt: bool = True,
) -> dict[str, Any]:
    """
    Фаза А (только отправка файла по сети, без ожидания серверной публикации):
      1. Проверка статуса карточки (пропуск, если уже DONE).
      2. Перевод карточки на шаг 'upload-panorama'.
      3. Multipart Upload файла по частям (5 МБ) в bo-storage.traceair.net.
      4. Запуск 'start-copying-panorama-image' на сервере и немедленный возврат,
         чтобы канал интернета сразу перешёл к загрузке следующей панорамы.
    """
    t0 = time.time()
    customer = str(task["customer"])
    site = str(task["site"])
    pano_id = str(task.get("pano_id") or task.get("date") or "")
    card_id = str(task.get("card_id", "")).strip()

    if not card_id:
        raise ValueError(f"Для задачи {customer}/{site}/{pano_id} не указан card_id (UUID карточки Magic).")

    resolved_file: Path | None = None
    if local_file:
        resolved_file = Path(local_file)
    elif task.get("local_file"):
        resolved_file = Path(task["local_file"])
    elif pano_root:
        resolved_file = find_primary_panorama_file(pano_root, customer, site, pano_id)

    if not resolved_file or not resolved_file.is_file():
        raise FileNotFoundError(
            f"Не найден локальный файл панорамы для {customer}/{site}/{pano_id}: {resolved_file}"
        )

    token = fetch_traceair_token(
        customer=customer,
        site=site,
        sess_id=sess_id,
        env_path=env_path,
        interactive_prompt=interactive_prompt,
    )
    headers_auth = {
        "Authorization": f"Bearer {token}",
        "Origin": "https://magic.traceair.net",
        "Referer": "https://magic.traceair.net/",
    }

    # Проверяем, не закрыта ли уже эта карточка в Magic
    state_data = get_card_state(customer, site, card_id, headers_auth)
    steps = {s["id"]: s for s in state_data.get("steps", []) if isinstance(s, dict) and "id" in s}
    if (
        steps.get("track-sla", {}).get("status") == "DONE"
        or steps.get("panorama-publication", {}).get("status") == "DONE"
    ):
        print(f"  [Magic Пропуск] Панорама {customer}/{site}/{pano_id} (карточка {card_id}) уже опубликована.")
        return {
            "status": "skipped_already_done",
            "customer": customer,
            "site": site,
            "pano_id": pano_id,
            "card_id": card_id,
            "local_file": str(resolved_file),
            "elapsed_sec": round(time.time() - t0, 2),
        }

    # 1. Активируем шаг upload-panorama
    print(f"  [Magic 1/5] {customer}/{site}/{pano_id}: переход на шаг upload-panorama...")
    run_card_action(
        customer,
        site,
        card_id,
        "manual-set-step-active",
        headers_auth,
        {"step_id": "upload-panorama"},
    )

    # 2. Инициализируем Multipart Upload на bo-storage
    file_uuid = str(uuid.uuid4())
    base_upload_url = f"https://bo-storage.traceair.net/upload/{customer}/{site}/{file_uuid}/full_panorama.jpg"
    s3_target_path = f"s3://exchange-traceair/_upload/{customer}/{site}/{file_uuid}/full_panorama.jpg"

    print(f"  [Magic 2/5] {customer}/{site}/{pano_id}: инициализация bo-storage (UUID: {file_uuid})...")
    init_resp = _http_request("POST", f"{base_upload_url}?uploads", headers=headers_auth, timeout=30.0)
    init_resp.raise_for_status()

    match = re.search(r"<UploadId>(.+?)</UploadId>", init_resp.text)
    if not match:
        raise RuntimeError(f"Не удалось получить UploadId от bo-storage: {init_resp.text[:300]}")
    upload_id = match.group(1)

    # 3. Загружаем локальный файл частями по 5 МБ (параллельно в 4 потока) с ретраями
    file_size_mb = round(resolved_file.stat().st_size / (1024 * 1024), 1)
    print(f"  [Magic 3/5] {customer}/{site}/{pano_id}: отправка {resolved_file.name} ({file_size_mb} МБ) по частям...")
    chunks_to_upload: list[tuple[int, bytes]] = []
    part_number = 1

    with open(resolved_file, "rb") as f:
        while chunk := f.read(chunk_size):
            chunks_to_upload.append((part_number, chunk))
            part_number += 1

    def _upload_single_part(item: tuple[int, bytes]) -> tuple[int, str]:
        p_num, data_bytes = item
        part_url = f"{base_upload_url}?partNumber={p_num}&uploadId={upload_id}"
        last_err: Exception | None = None
        for attempt in range(1, max_chunk_retries + 1):
            try:
                put_resp = _http_request(
                    "PUT",
                    part_url,
                    headers={**headers_auth, "Content-Type": "application/octet-stream"},
                    data=data_bytes,
                    timeout=120.0,
                )
                put_resp.raise_for_status()
                etag = put_resp.get_header("ETag")
                if not etag:
                    raise RuntimeError("Сервер bo-storage не вернул заголовок ETag.")
                print(f"    -> [{pano_id}] Часть {p_num}/{len(chunks_to_upload)} загружена (ETag: {etag})")
                return p_num, f"<Part><ETag>{etag}</ETag><PartNumber>{p_num}</PartNumber></Part>"
            except Exception as exc:
                last_err = exc
                if attempt < max_chunk_retries:
                    print(f"    [!] Сбой загрузки части {p_num} (попытка {attempt}/{max_chunk_retries}): {exc}")
                    time.sleep(2.0 * attempt)
                else:
                    raise RuntimeError(
                        f"Не удалось загрузить часть {p_num} после {max_chunk_retries} попыток: {last_err}"
                    ) from exc
        raise RuntimeError(f"Не удалось загрузить часть {p_num}")

    max_part_workers = min(4, max(1, len(chunks_to_upload)))
    with ThreadPoolExecutor(max_workers=max_part_workers) as pool:
        uploaded_parts = list(pool.map(_upload_single_part, chunks_to_upload))
    uploaded_parts.sort(key=lambda x: x[0])
    parts_xml = [xml_str for _, xml_str in uploaded_parts]

    complete_xml = (
        '<CompleteMultipartUpload xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        + "".join(parts_xml)
        + "</CompleteMultipartUpload>"
    )
    complete_resp = _http_request(
        "POST",
        f"{base_upload_url}?uploadId={upload_id}",
        headers={**headers_auth, "Content-Type": "application/octet-stream; charset=UTF-8"},
        data=complete_xml.encode("utf-8"),
        timeout=60.0,
    )
    complete_resp.raise_for_status()

    # 4. Запускаем копирование на сервере и сразу возвращаем управление
    print(f"  [Magic 4/5] {customer}/{site}/{pano_id}: запуск start-copying-panorama-image (передано в фоновый поллер)...")
    try:
        run_card_action(
            customer,
            site,
            card_id,
            "start-copying-panorama-image",
            headers_auth,
            {"fullPanoramaS3Path": s3_target_path},
        )
    except RuntimeError as exc_camel:
        if "full_panorama_s3_path" in str(exc_camel):
            run_card_action(
                customer,
                site,
                card_id,
                "start-copying-panorama-image",
                headers_auth,
                {"full_panorama_s3_path": s3_target_path},
            )
        else:
            raise

    upload_elapsed = round(time.time() - t0, 2)
    print(
        f"  [Загрузка в bo-storage завершена за {upload_elapsed}с] {customer}/{site}/{pano_id} -> "
        f"сервер начал публикацию, переходим к следующему файлу!"
    )
    return {
        "status": "copying_started",
        "customer": customer,
        "site": site,
        "pano_id": pano_id,
        "card_id": card_id,
        "s3_target_path": s3_target_path,
        "local_file": str(resolved_file),
        "headers_auth": headers_auth,
        "t0": t0,
        "upload_elapsed_sec": upload_elapsed,
    }


def finalize_panorama_publication(
    upload_ctx: dict[str, Any],
    poll_interval_sec: float = DEFAULT_POLL_INTERVAL_SEC,
    max_poll_wait_sec: float = 1200.0,
) -> dict[str, Any]:
    """
    Фаза Б (фоновое ожидание серверной публикации и закрытие шагов карточки):
      - Опрашивает get-state каждые poll_interval_sec (по умолчанию 60 сек), не блокируя загрузку новых файлов.
      - Как только 'panorama-publication == DONE', закрывает 'confirm-quality-check-done' и 'confirm-sla-recorded'.
    """
    if upload_ctx.get("status") == "skipped_already_done":
        return upload_ctx

    customer = str(upload_ctx["customer"])
    site = str(upload_ctx["site"])
    pano_id = str(upload_ctx["pano_id"])
    card_id = str(upload_ctx["card_id"])
    headers_auth = dict(upload_ctx["headers_auth"])
    t0 = float(upload_ctx.get("t0", time.time()))

    poll_start = time.time()
    while True:
        state_data = get_card_state(customer, site, card_id, headers_auth)
        steps = {s["id"]: s for s in state_data.get("steps", []) if isinstance(s, dict) and "id" in s}

        copy_step = steps.get("panorama-image-copying", {})
        pub_step = steps.get("panorama-publication", {})

        if copy_step.get("error") or pub_step.get("error"):
            raise RuntimeError(
                f"Ошибка публикации на сервере Magic ({customer}/{site}/{pano_id}): "
                f"copy={copy_step.get('error')}, pub={pub_step.get('error')}"
            )

        pub_status = pub_step.get("status")
        print(
            f"    [Поллер 60с | {customer}/{site}/{pano_id}] "
            f"Копирование: {copy_step.get('status')} | Публикация: {pub_status}"
        )

        if pub_status == "DONE":
            break

        if time.time() - poll_start > max_poll_wait_sec:
            raise TimeoutError(
                f"Превышено время ожидания публикации ({max_poll_wait_sec}с) для карточки {card_id}."
            )

        time.sleep(poll_interval_sec)

    # 5. Закрываем шаги Quality Check и Track SLA (саму задачу в Asana не закрываем)
    print(f"  [Magic 5/5] {customer}/{site}/{pano_id}: подтверждение Quality Check и SLA...")
    run_card_action(customer, site, card_id, "confirm-quality-check-done", headers_auth, {})
    time.sleep(1.0)
    run_card_action(customer, site, card_id, "confirm-sla-recorded", headers_auth, {})

    elapsed = round(time.time() - t0, 2)
    print(f"  [Magic Готово] Панорама {customer}/{site}/{pano_id} полностью опубликована (общее время: {elapsed}с)!")
    return {
        "status": "published",
        "customer": customer,
        "site": site,
        "pano_id": pano_id,
        "card_id": card_id,
        "s3_target_path": upload_ctx.get("s3_target_path", ""),
        "local_file": upload_ctx.get("local_file", ""),
        "upload_elapsed_sec": upload_ctx.get("upload_elapsed_sec", 0.0),
        "elapsed_sec": elapsed,
    }


def publish_panorama_to_magic(
    task: dict[str, Any],
    local_file: str | Path | None = None,
    pano_root: str | Path | None = None,
    sess_id: str | None = None,
    env_path: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    poll_interval_sec: float = DEFAULT_POLL_INTERVAL_SEC,
    max_chunk_retries: int = 3,
    max_poll_wait_sec: float = 1200.0,
    interactive_prompt: bool = True,
) -> dict[str, Any]:
    """Полный цикл (Фаза А + Фаза Б) для одиночного вызова."""
    upload_ctx = upload_panorama_to_bo_storage(
        task=task,
        local_file=local_file,
        pano_root=pano_root,
        sess_id=sess_id,
        env_path=env_path,
        chunk_size=chunk_size,
        max_chunk_retries=max_chunk_retries,
        interactive_prompt=interactive_prompt,
    )
    if upload_ctx.get("status") == "skipped_already_done":
        return upload_ctx
    return finalize_panorama_publication(
        upload_ctx=upload_ctx,
        poll_interval_sec=poll_interval_sec,
        max_poll_wait_sec=max_poll_wait_sec,
    )
