"""
Изолированный сервис автоматизации Asana и публикации готовых панорам в TraceAir Magic.
Может использоваться как самостоятельно (scripts/publish_magic.py), так и в качестве
опционального плагина к асинхронному конвейеру (scripts/async_pipeline.py).
При необходимости полностью удаляется из проекта простым удалением папки src/magic_service/.
"""

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
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_POLL_INTERVAL_SEC",
    "TraceAirSessionExpiredError",
    "fetch_my_asana_tasks",
    "fetch_traceair_token",
    "finalize_panorama_publication",
    "find_primary_panorama_file",
    "parse_asana_tasks_payload",
    "publish_panorama_to_magic",
    "upload_panorama_to_bo_storage",
]
