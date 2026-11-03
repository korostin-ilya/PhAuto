#!/usr/bin/env python3
"""
Скрипт проверки сессии AWS SSO и экспорта временных ключей в корневой .env на хосте.
Использует только стандартную библиотеку Python (не требует установки opencv/numpy).
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.storage.s3_downloader import DEFAULT_AWS_REGION, ensure_aws_env_credentials


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Проверка сессии AWS SSO и экспорт временных ключей в корневой .env",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=os.path.join(PROJECT_ROOT, ".env"),
        help="Путь к целевому файлу .env (по умолчанию <корень>/.env)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Имя профиля AWS SSO (если отличается от default)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=DEFAULT_AWS_REGION,
        help=f"Регион AWS по умолчанию (по умолчанию {DEFAULT_AWS_REGION})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Принудительно перезапросить sso login и обновить токены",
    )
    args = parser.parse_args()

    ensure_aws_env_credentials(
        env_path=args.env_file,
        default_region=args.region,
        profile=args.profile,
        force_refresh=args.force,
    )


if __name__ == "__main__":
    main()
