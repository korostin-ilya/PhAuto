"""
Изолированный сервис автоматизации Asana и публикации готовых панорам в TraceAir Magic.
Может использоваться как самостоятельно (scripts/publish_magic.py), так и в качестве
опционального плагина к асинхронному конвейеру (scripts/async_pipeline.py).
При необходимости полностью удаляется из проекта простым удалением папки src/magic_service/.
"""

from src.magic_service.airtable_client import (
    AirtableAuthError,
    assign_airtable_owner_to_me,
    fetch_airtable_rows_map,
    normalize_asana_task_key,
    parse_airtable_read_queries_payload,
    sync_airtable_owners_for_tasks,
)
from src.magic_service.client import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_POLL_INTERVAL_SEC,
    TraceAirSessionExpiredError,
    fetch_my_asana_tasks,
    fetch_traceair_token,
    finalize_panorama_publication,
    find_primary_panorama_file,
    parse_asana_tasks_payload,
    publish_panorama_to_magic,
    upload_panorama_to_bo_storage,
)

__all__ = [
    "AirtableAuthError",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_POLL_INTERVAL_SEC",
    "TraceAirSessionExpiredError",
    "assign_airtable_owner_to_me",
    "fetch_airtable_rows_map",
    "fetch_my_asana_tasks",
    "fetch_traceair_token",
    "finalize_panorama_publication",
    "find_primary_panorama_file",
    "normalize_asana_task_key",
    "parse_airtable_read_queries_payload",
    "parse_asana_tasks_payload",
    "publish_panorama_to_magic",
    "sync_airtable_owners_for_tasks",
    "upload_panorama_to_bo_storage",
]
