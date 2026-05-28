"""공통 유틸 — 설정 로드, 환경변수, 경로, 로깅. (독립 저장소 자체 포함본)"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# msqa/ 의 부모 = 저장소 루트
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# .env 자동 로드 (export 해도 동일하게 동작)
load_dotenv(PROJECT_ROOT / ".env")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml 을 찾을 수 없습니다: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def require_env(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise EnvironmentError(
            f"환경변수 {var} 가 설정되지 않았습니다. "
            f".env 파일에 {var}=... 를 추가하거나 set {var}=... 후 재실행하세요."
        )
    return val


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    logs_dir = ensure_dir(PROJECT_ROOT / "logs")
    fname = log_file or f"{name}_{datetime.now():%Y%m%d}.log"
    fh = logging.FileHandler(logs_dir / fname, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False
    return logger
