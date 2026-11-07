"""
Клиент для автоматизации назначения Owner ('Ilia Korostin') в таблице Airtable 'Pano List'
через внутренние веб-запросы (/v0.3/application/.../readQueries и /v0.3/row/.../updateArrayTypeCellByAddingItem).

Входит в изолированный пакет src/magic_service: при удалении папки src/magic_service
основной конвейер сборки панорам продолжает работать автономно.
"""

from __future__ import annotations

import base64
import gzip
import io
import json
import os
import random
import re
import string
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from http.cookies import SimpleCookie
from typing import Any

from src.storage.s3_downloader import load_dotenv_file, update_dotenv_vars

# Константы схемы Airtable (Pano List)
APP_ID = "appi2FgOXf28y5dLd"
PAGE_BUNDLE_ID = "pbdTBaEcxYs9vcZgj"
PAGE_ID = "pagG9yhLr3TlsUDEl"  # Страница списка Pano List
DETAIL_PAGE_ID = "pag5ed7nZ6vAYmSAy"  # Внутренний ID боковой панели деталей
TABLE_ID = "tbl8u2bdB1KjLEVjL"
QUERY_ID = "qryrPH1mgQ3FZxmeK"
ACTIVE_PAGE_ELEMENT_ID = "pelBgELeuvCKdm8yY"
QUERY_CONTAINER_ID = "pelbFu4OKbyp9lE48"

# Колонки таблицы
DATE_COLUMN_ID = "fldgYxkvXZafp73bi"
ASANA_NAME_COLUMN_ID = "fldyJLBRqFxBN2EpZ"  # "The same name as in Asana"
OWNER_COLUMN_ID = "fldUKan6zLinHaKDW"  # "Owner"

# Данные сотрудника для назначения Owner
MY_FOREIGN_ROW_ID = "recl6F4jyjfe0S4fj"
MY_DISPLAY_NAME = "Ilia Korostin"

# Пауза между запросами записи (антибан-джиттер)
DEFAULT_MIN_WRITE_DELAY_SEC = 1.8
DEFAULT_MAX_WRITE_DELAY_SEC = 3.8

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/152.0.0.0 YaBrowser/26.10.0.0 Safari/537.36"
)
CLIENT_CODE_VERSION = "ffd39fc396c1522edc89da2fce4c2f1051c31800"
DEFAULT_PAGE_LOAD_ID = "pglQDwRedi0dTvFHa"
DEFAULT_SECRET_SOCKET_ID = "soc96BXO2GXhrpHKE"


class AirtableAuthError(RuntimeError):
    """Исключение при протухшей или невалидной сессии Airtable (HTTP 401/403)."""


def generate_airtable_id(prefix: str = "req", length: int = 14) -> str:
    """Генерирует случайный идентификатор в формате Airtable (req..., pgl..., soc...)."""
    chars = string.ascii_letters + string.digits
    return prefix + "".join(random.choices(chars, k=length))


def make_detail_b64(row_id: str) -> str:
    """Кодирует контекст открытой строки в Base64 без '=' на конце, как делает веб-клиент Airtable."""
    detail_obj = {
        "pageId": DETAIL_PAGE_ID,
        "rowId": row_id,
        "showComments": False,
        "queryOriginHint": {
            "type": "pageElement",
            "elementId": ACTIVE_PAGE_ELEMENT_ID,
            "queryContainerId": QUERY_CONTAINER_ID,
        },
    }
    raw_json = json.dumps(detail_obj, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(raw_json).decode("ascii").rstrip("=")


def normalize_asana_task_key(raw_value: str) -> str:
    """
    Приводит название задачи из Asana или из колонки Airtable 'The same name as in Asana'
    к единому ключу вида 'customer/site/pano_id' в нижнем регистре.
    Поддерживает префиксы 'Panorama:', 'For Panorama', пробелы и слэши.
    """
    text = str(raw_value or "").strip().strip("'\"")
    if not text:
        return ""
    text = re.sub(r"^(?:for\s+)?panorama\s*:?\s*", "", text, flags=re.IGNORECASE).strip()
    text = text.replace("\\", "/").strip("/")
    parts = [p.strip() for p in text.split("/") if p.strip()]
    if len(parts) >= 3:
        return f"{parts[-3]}/{parts[-2]}/{parts[-1]}"
    return "/".join(parts)


def _merge_set_cookie_headers(cookie_str: str, set_cookie_headers: list[str]) -> str:
    """
    Обновляет строку Cookie новыми значениями из заголовков Set-Cookie (например, AWSALBTG / AWSALBTGCORS),
    чтобы последовательные запросы выглядели как одна живая браузерная сессия.
    """
    if not set_cookie_headers:
        return cookie_str

    jar: dict[str, str] = {}
    order: list[str] = []
    for chunk in cookie_str.split(";"):
        part = chunk.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        if k not in jar:
            order.append(k)
        jar[k] = v.strip()

    for header_val in set_cookie_headers:
        sc = SimpleCookie()
        try:
            sc.load(header_val)
            for morsel_key, morsel in sc.items():
                if morsel_key not in jar:
                    order.append(morsel_key)
                jar[morsel_key] = morsel.value
        except Exception:
            first_pair = header_val.split(";", 1)[0].strip()
            if "=" in first_pair:
                mk, mv = first_pair.split("=", 1)
                mk = mk.strip()
                if mk not in jar:
                    order.append(mk)
                jar[mk] = mv.strip()

    return "; ".join(f"{k}={jar[k]}" for k in order if k in jar)


class _AirtableWebSession:
    """
    Легковесная HTTP-сессия для веб-запросов к Airtable, поддерживающая:
    - автоматическое обновление балансировочных кук (AWSALBTG / AWSALBTGCORS);
    - прозрачную распаковку gzip;
    - единый pageLoadId и secretSocketId на время жизни вкладки.
    Работает на стандартной библиотеке Python (не требует внешних пакетов на хосте).
    """

    def __init__(
        self,
        cookies: str,
        page_load_id: str | None = None,
        secret_socket_id: str | None = None,
    ) -> None:
        self.cookies = cookies.strip().strip("'\"")
        self.page_load_id = page_load_id or DEFAULT_PAGE_LOAD_ID
        self.secret_socket_id = secret_socket_id or DEFAULT_SECRET_SOCKET_ID

    def post_form(
        self,
        url: str,
        headers: dict[str, str],
        form_fields: dict[str, str],
        timeout: float = 45.0,
    ) -> tuple[int, dict[str, str], bytes]:
        encoded_data = urllib.parse.urlencode(form_fields).encode("utf-8")
        req_headers = dict(headers)
        req_headers["cookie"] = self.cookies
        req_headers.setdefault("accept-encoding", "gzip")

        req = urllib.request.Request(url, data=encoded_data, headers=req_headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = resp.status
                set_cookies = resp.headers.get_all("Set-Cookie") or []
                if set_cookies:
                    self.cookies = _merge_set_cookie_headers(self.cookies, set_cookies)
                raw_body = resp.read()
                content_encoding = (resp.headers.get("Content-Encoding") or "").lower()
                if content_encoding == "gzip" or raw_body[:2] == b"\x1f\x8b":
                    raw_body = gzip.decompress(raw_body)
                resp_headers = {k.lower(): v for k, v in resp.headers.items()}
                return status, resp_headers, raw_body
        except urllib.error.HTTPError as exc:
            set_cookies = exc.headers.get_all("Set-Cookie") if exc.headers else []
            if set_cookies:
                self.cookies = _merge_set_cookie_headers(self.cookies, set_cookies)
            err_body = exc.read() if exc.fp else b""
            if err_body[:2] == b"\x1f\x8b":
                try:
                    err_body = gzip.decompress(err_body)
                except Exception:
                    pass
            resp_headers = {k.lower(): v for k, v in (exc.headers.items() if exc.headers else [])}
            return exc.code, resp_headers, err_body


def _prompt_and_save_airtable_cookies(env_path: str | None = None) -> str:
    """
    Запрашивает у пользователя свежую строку Cookie из браузера при истечении сессии Airtable
    и сохраняет её в корневой .env (AIRTABLE_COOKIES).
    """
    print("\n" + "=" * 78)
    print("[Airtable Авторизация] Сессия Airtable (AIRTABLE_COOKIES) отсутствует или истекла (401/403).")
    print("Чтобы обновить её:")
    print(f"  1. Откройте в браузере https://airtable.com/{APP_ID}/{PAGE_ID}")
    print("  2. Нажмите F12 -> вкладка Network -> выберите любой запрос к airtable.com/v0.3/")
    print("  3. Скопируйте значение заголовка 'Cookie' из Request Headers")
    print("=" * 78)

    if not sys.stdin or not sys.stdin.isatty():
        raise AirtableAuthError(
            "AIRTABLE_COOKIES истёк или не задан, а интерактивный ввод недоступен. "
            "Обновите переменную AIRTABLE_COOKIES в файле .env."
        )

    new_cookies = input("Вставьте новую строку Cookie для Airtable (или нажмите Enter для пропуска): ").strip()
    if (new_cookies.startswith('"') and new_cookies.endswith('"')) or (
        new_cookies.startswith("'") and new_cookies.endswith("'")
    ):
        new_cookies = new_cookies[1:-1].strip()

    if not new_cookies:
        raise AirtableAuthError("Ввод AIRTABLE_COOKIES пропущен пользователем.")

    os.environ["AIRTABLE_COOKIES"] = new_cookies
    if env_path:
        update_dotenv_vars(env_path, {"AIRTABLE_COOKIES": new_cookies})
        print(f"[Airtable Авторизация] Новые куки сохранены в {os.path.abspath(env_path)}")
    return new_cookies


def parse_airtable_read_queries_payload(
    payload: dict[str, Any] | list[Any],
    my_foreign_row_id: str = MY_FOREIGN_ROW_ID,
    my_display_name: str = MY_DISPLAY_NAME,
) -> dict[str, dict[str, Any]]:
    """
    Разбирает ответ эндпоинта readQueries (как в формате JSON dict, так и в формате списка из msgpack)
    и возвращает словарь строк таблицы, индексированный по нормализованному имени задачи ('customer/site/pano_id'):
        {
            "customer/site/2026-11-03_01": {
                "row_id": "recp7i9iLrK1bj5mX",
                "asana_name": "customer/site/2026-11-03_01",
                "owners": [{"foreignRowId": "recl6F4jyjfe0S4fj", "foreignRowDisplayName": "Ilia Korostin"}],
                "has_me_as_owner": True,
            },
            ...
        }
    Если в таблице встречаются дубли по имени задачи, приоритет отдаётся самой свежей записи
    (согласно порядку в querySlices[0]['rowIds']).
    """
    root_dict: dict[str, Any] = {}
    if isinstance(payload, dict):
        root_dict = payload
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and ("data" in item or "tableDataById" in item):
                root_dict = item
                break

    data_node = root_dict.get("data", root_dict) if isinstance(root_dict, dict) else {}
    if not isinstance(data_node, dict):
        return {}

    table_data_by_id = data_node.get("tableDataById") or {}
    table_node = table_data_by_id.get(TABLE_ID) or {}
    if not table_node and table_data_by_id:
        first_tbl = next(iter(table_data_by_id.values()), {})
        if isinstance(first_tbl, dict):
            table_node = first_tbl

    partial_rows: dict[str, Any] = table_node.get("partialRowById") or {}
    if not isinstance(partial_rows, dict) or not partial_rows:
        return {}

    # Порядок строк из querySlices (от новых к старым)
    ordered_row_ids: list[str] = []
    query_slices = data_node.get("querySlices") or []
    if isinstance(query_slices, list) and query_slices:
        first_slice = query_slices[0]
        if isinstance(first_slice, dict) and isinstance(first_slice.get("rowIds"), list):
            ordered_row_ids = [str(r) for r in first_slice["rowIds"]]

    if not ordered_row_ids:
        ordered_row_ids = list(partial_rows.keys())
    else:
        seen_set = set(ordered_row_ids)
        for rid in partial_rows.keys():
            if rid not in seen_set:
                ordered_row_ids.append(rid)

    rows_by_task_key: dict[str, dict[str, Any]] = {}
    for row_id in ordered_row_ids:
        row_obj = partial_rows.get(row_id)
        if not isinstance(row_obj, dict):
            continue
        cells = row_obj.get("cellValuesByColumnId") or {}
        if not isinstance(cells, dict):
            continue

        raw_asana_name = cells.get(ASANA_NAME_COLUMN_ID)
        if not raw_asana_name or not isinstance(raw_asana_name, str):
            continue

        task_key = normalize_asana_task_key(raw_asana_name)
        if not task_key:
            continue

        # Если задача уже есть в индексе (более свежая строка из начала списка), не перезаписываем старой
        if task_key in rows_by_task_key:
            continue

        raw_owners = cells.get(OWNER_COLUMN_ID) or []
        owners_list: list[dict[str, str]] = []
        has_me = False
        if isinstance(raw_owners, list):
            for owner_item in raw_owners:
                if isinstance(owner_item, dict):
                    f_id = str(owner_item.get("foreignRowId", "")).strip()
                    f_name = str(owner_item.get("foreignRowDisplayName", "")).strip()
                    owners_list.append({"foreignRowId": f_id, "foreignRowDisplayName": f_name})
                    if f_id == my_foreign_row_id or (
                        my_display_name and f_name.lower() == my_display_name.lower()
                    ):
                        has_me = True

        rows_by_task_key[task_key] = {
            "row_id": str(row_obj.get("id") or row_id),
            "asana_name": raw_asana_name.strip(),
            "task_key": task_key,
            "owners": owners_list,
            "has_me_as_owner": has_me,
        }

    return rows_by_task_key


def fetch_airtable_rows_map(
    session: _AirtableWebSession,
    timeout: float = 45.0,
) -> dict[str, dict[str, Any]]:
    """
    Отправляет 1 запрос к POST /v0.3/application/{APP_ID}/readQueries
    и возвращает словарь строк таблицы Pano List по ключу 'customer/site/pano_id'.
    """
    url = f"https://airtable.com/v0.3/application/{APP_ID}/readQueries"
    headers = {
        "accept": "*/*",
        "accept-language": "ru,en;q=0.9",
        "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
        "origin": "https://airtable.com",
        "referer": f"https://airtable.com/{APP_ID}/{PAGE_ID}",
        "user-agent": USER_AGENT,
        "x-airtable-application-id": APP_ID,
        "x-airtable-client-document-visibility-state": "visible",
        "x-airtable-client-queue-time": "0.5",
        "x-airtable-integration-id": PAGE_ID,
        "x-airtable-inter-service-client": "webClient",
        "x-airtable-inter-service-client-code-version": CLIENT_CODE_VERSION,
        "x-airtable-page-load-id": session.page_load_id,
        "x-airtable-pages-context": f"view,{PAGE_BUNDLE_ID},{PAGE_ID}",
        "x-requested-with": "XMLHttpRequest",
        "x-time-zone": "Asia/Yekaterinburg",
        "x-user-locale": "en",
    }

    query_params = {
        "queries": [
            {
                "id": QUERY_ID,
                "spec": {
                    "source": {"type": "table", "tableId": TABLE_ID},
                    "columnIds": [
                        DATE_COLUMN_ID,
                        ASANA_NAME_COLUMN_ID,
                        OWNER_COLUMN_ID,
                        "fldunKvPJp8CsM5EH",
                        "fldT6jguU3Nm7dMTy",
                        "fldy1Diw9YJhgxcBH",
                        "fldp7g87BvDdPO19A",
                    ],
                    "sorts": [{"columnId": DATE_COLUMN_ID, "ascending": False}],
                    "virtualColumns": {
                        "commentCounts": {
                            "columnIds": [],
                            "includeRowLevelComments": False,
                        }
                    },
                    "filters": {
                        "conjunction": "and",
                        "filterSet": [
                            {
                                "columnId": ASANA_NAME_COLUMN_ID,
                                "operator": "isNotEmpty",
                                "value": None,
                            },
                            {
                                "columnId": ASANA_NAME_COLUMN_ID,
                                "operator": "doesNotContain",
                                "value": "smoke-test-en",
                            },
                        ],
                        "source": "permissionsFilters",
                        "filterSourcePageElementId": ACTIVE_PAGE_ELEMENT_ID,
                    },
                },
                "realm": {
                    "type": "page",
                    "pageId": PAGE_ID,
                    "pageBundleId": PAGE_BUNDLE_ID,
                },
                "mayHaveAcceptableAuthorizationError": False,
            }
        ],
        "subscribeToRealtimeUpdates": True,
        "allowMsgpackOfResult": False,
        "isBackgroundRequest": False,
    }

    form_fields = {
        "stringifiedObjectParams": json.dumps(query_params, separators=(",", ":")),
        "requestId": generate_airtable_id("req", 14),
        "secretSocketId": session.secret_socket_id,
    }

    status, resp_headers, raw_body = session.post_form(
        url=url,
        headers=headers,
        form_fields=form_fields,
        timeout=timeout,
    )

    if status in (401, 403):
        body_preview = raw_body.decode("utf-8", errors="ignore")[:300]
        raise AirtableAuthError(f"HTTP {status} при чтении Airtable readQueries: {body_preview}")

    if status < 200 or status >= 300:
        body_preview = raw_body.decode("utf-8", errors="ignore")[:500]
        raise RuntimeError(f"Ошибка Airtable readQueries (HTTP {status}): {body_preview}")

    ctype = resp_headers.get("content-type", "").lower()
    parsed_payload: Any = None
    if "json" in ctype or raw_body.lstrip().startswith(b"{"):
        parsed_payload = json.loads(raw_body.decode("utf-8"))
    else:
        import msgpack  # Фолбэк на случай бинарного потока msgpack

        unpacker = msgpack.Unpacker(io.BytesIO(raw_body), raw=False, strict_map_key=False)
        parsed_payload = list(unpacker)

    return parse_airtable_read_queries_payload(parsed_payload)


def assign_airtable_owner_to_me(
    row_id: str,
    session: _AirtableWebSession,
    my_foreign_row_id: str = MY_FOREIGN_ROW_ID,
    my_display_name: str = MY_DISPLAY_NAME,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """
    Назначает 'Ilia Korostin' (recl6F4jyjfe0S4fj) в колонку Owner (fldUKan6zLinHaKDW)
    для указанной строки row_id (rec...) через POST /v0.3/row/{row_id}/updateArrayTypeCellByAddingItem.
    При HTTP 429 делает одну паузу 15 секунд и повторяет попытку.
    """
    url = f"https://airtable.com/v0.3/row/{row_id}/updateArrayTypeCellByAddingItem"
    detail_b64 = make_detail_b64(row_id)

    headers = {
        "accept": "application/json, text/javascript, */*; q=0.01",
        "accept-language": "ru,en;q=0.9",
        "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
        "origin": "https://airtable.com",
        "referer": f"https://airtable.com/{APP_ID}/{PAGE_ID}?detail={detail_b64}",
        "user-agent": USER_AGENT,
        "x-airtable-application-id": APP_ID,
        "x-airtable-client-document-visibility-state": "visible",
        "x-airtable-client-queue-time": f"{round(random.uniform(8.0, 22.0), 1)}",
        "x-airtable-integration-id": PAGE_ID,
        "x-airtable-inter-service-client": "webClient",
        "x-airtable-inter-service-client-code-version": CLIENT_CODE_VERSION,
        "x-airtable-page-load-id": session.page_load_id,
        "x-airtable-pages-context": f"view,{PAGE_BUNDLE_ID},{PAGE_ID}",
        "x-requested-with": "XMLHttpRequest",
        "x-time-zone": "Asia/Yekaterinburg",
        "x-user-locale": "en",
    }

    params_obj = {
        "columnId": OWNER_COLUMN_ID,
        "item": {
            "foreignRowId": my_foreign_row_id,
            "foreignRowDisplayName": my_display_name,
        },
        "pageNavigationDeeplink": {
            "pageBundleId": PAGE_BUNDLE_ID,
            "pageId": PAGE_ID,
            "pageRootRowId": None,
            "pageUrlSearchParams": {
                "detail": detail_b64,
            },
            "activePageElementId": ACTIVE_PAGE_ELEMENT_ID,
        },
    }

    for attempt in range(1, 3):
        form_fields = {
            "stringifiedObjectParams": json.dumps(params_obj, separators=(",", ":")),
            "requestId": generate_airtable_id("req", 14),
            "secretSocketId": session.secret_socket_id,
        }
        status, _, raw_body = session.post_form(
            url=url,
            headers=headers,
            form_fields=form_fields,
            timeout=timeout,
        )
        if status == 429 and attempt == 1:
            print("  [Airtable Лимит 429] Пауза 15 сек. перед повторной попыткой...")
            time.sleep(15.0)
            continue
        if status in (401, 403):
            body_preview = raw_body.decode("utf-8", errors="ignore")[:300]
            raise AirtableAuthError(
                f"HTTP {status} при назначении Owner для {row_id}: {body_preview}"
            )
        if status < 200 or status >= 300:
            body_preview = raw_body.decode("utf-8", errors="ignore")[:500]
            raise RuntimeError(
                f"Ошибка назначения Owner в Airtable для {row_id} (HTTP {status}): {body_preview}"
            )
        try:
            return json.loads(raw_body.decode("utf-8"))
        except Exception:
            return {"status": "ok"}

    raise RuntimeError(f"Не удалось назначить Owner для {row_id} после 2 попыток.")


def sync_airtable_owners_for_tasks(
    tasks: list[dict[str, Any] | str],
    env_path: str | None = None,
    airtable_cookies: str | None = None,
    min_delay_sec: float = DEFAULT_MIN_WRITE_DELAY_SEC,
    max_delay_sec: float = DEFAULT_MAX_WRITE_DELAY_SEC,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Для переданного списка задач из Asana (словарей из fetch_my_asana_tasks или строк вида 'customer/site/date_01'):
      1. Выполняет 1 запрос чтения к таблице Airtable Pano List (readQueries).
      2. Находит для каждой задачи строку по полю 'The same name as in Asana' (fldyJLBRqFxBN2EpZ).
      3. Если 'Ilia Korostin' уже указан в колонке Owner — пропускает без отправки HTTP-запроса.
      4. Если ещё не указан — назначает 'Ilia Korostin' с антибан-паузой 1.8–3.8 сек. между запросами.
    """
    if not tasks:
        return {"assigned": [], "already_owner": [], "not_found": [], "failed": []}

    if env_path and os.path.isfile(env_path):
        load_dotenv_file(env_path, override=False)

    cookies = (airtable_cookies or os.environ.get("AIRTABLE_COOKIES", "")).strip().strip("'\"")
    if not cookies:
        try:
            cookies = _prompt_and_save_airtable_cookies(env_path=env_path)
        except AirtableAuthError as exc:
            print(f"[Airtable Пропуск] {exc}")
            return {"assigned": [], "already_owner": [], "not_found": [], "failed": []}

    session = _AirtableWebSession(cookies=cookies)

    # 1. Читаем таблицу Pano List (при 401/403 запрашиваем свежие куки один раз)
    print(f"\n[Airtable] Чтение списка задач Pano List ({APP_ID}/{PAGE_ID})...")
    try:
        rows_map = fetch_airtable_rows_map(session)
    except AirtableAuthError:
        cookies = _prompt_and_save_airtable_cookies(env_path=env_path)
        session = _AirtableWebSession(cookies=cookies)
        rows_map = fetch_airtable_rows_map(session)

    print(f"[Airtable] Загружено строк из таблицы: {len(rows_map)}")

    assigned: list[dict[str, Any]] = []
    already_owner: list[dict[str, Any]] = []
    not_found: list[str] = []
    failed: list[dict[str, str]] = []

    writes_done = 0
    for item in tasks:
        if isinstance(item, dict):
            raw_task_str = (
                item.get("relative_dir")
                or item.get("task_string")
                or item.get("asana_name")
                or ""
            )
        else:
            raw_task_str = str(item)

        task_key = normalize_asana_task_key(raw_task_str)
        if not task_key:
            continue

        row_info = rows_map.get(task_key)
        if row_info is None:
            print(f"  [Airtable Не найдено] '{task_key}' отсутствует в Pano List")
            not_found.append(task_key)
            continue

        row_id = row_info["row_id"]
        if row_info["has_me_as_owner"]:
            print(
                f"  [Airtable Пропуск] {task_key} ({row_id}) -> уже Owner: {MY_DISPLAY_NAME}"
            )
            already_owner.append({"task_key": task_key, "row_id": row_id})
            continue

        if dry_run:
            print(
                f"  [Airtable Dry-Run] Назначил бы Owner={MY_DISPLAY_NAME} для {task_key} ({row_id})"
            )
            assigned.append({"task_key": task_key, "row_id": row_id, "dry_run": True})
            continue

        # Антибан-пауза перед вторым и последующими запросами на запись
        if writes_done > 0 and max_delay_sec > 0:
            delay = random.uniform(min_delay_sec, max(min_delay_sec, max_delay_sec))
            time.sleep(delay)

        try:
            assign_airtable_owner_to_me(row_id=row_id, session=session)
            writes_done += 1
            row_info["has_me_as_owner"] = True
            print(
                f"  [Airtable Назначен] {task_key} ({row_id}) -> Owner: {MY_DISPLAY_NAME}"
            )
            assigned.append({"task_key": task_key, "row_id": row_id})
        except Exception as exc:
            print(f"  [Airtable ОШИБКА] {task_key} ({row_id}): {exc}")
            failed.append({"task_key": task_key, "row_id": row_id, "error": str(exc)})

    print(
        f"[Airtable Итог] Назначено новых: {len(assigned)} | "
        f"Уже стоял Owner: {len(already_owner)} | "
        f"Не найдено: {len(not_found)} | Ошибок: {len(failed)}"
    )
    return {
        "assigned": assigned,
        "already_owner": already_owner,
        "not_found": not_found,
        "failed": failed,
    }
