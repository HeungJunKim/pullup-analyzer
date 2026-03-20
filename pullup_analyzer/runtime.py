from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


APP_DATA_DIRNAME = "PullUpAnalyzer"


def is_frozen_app() -> bool:
    return bool(getattr(sys, "frozen", False))


def get_bundle_dir() -> Path:
    if is_frozen_app():
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def get_executable_dir() -> Path:
    if is_frozen_app():
        return Path(sys.executable).resolve().parent
    return get_bundle_dir()


def get_user_data_dir() -> Path:
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / APP_DATA_DIRNAME
        return Path.home() / "AppData" / "Local" / APP_DATA_DIRNAME
    return Path.home() / f".{APP_DATA_DIRNAME.lower()}"


def resolve_bundled_binary(name: str) -> Path | None:
    binary_name = name
    if sys.platform == "win32" and not binary_name.lower().endswith(".exe"):
        binary_name = f"{binary_name}.exe"

    search_roots = (
        get_bundle_dir(),
        get_executable_dir(),
    )
    candidate_dirs = (
        Path(),
        Path("ffmpeg"),
        Path("bin"),
        Path("third_party") / "ffmpeg",
    )

    seen: set[Path] = set()
    for root in search_roots:
        for candidate_dir in candidate_dirs:
            candidate = (root / candidate_dir / binary_name).resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.exists():
                return candidate
    return None


def resolve_binary(name: str) -> str | None:
    bundled_binary = resolve_bundled_binary(name)
    if bundled_binary is not None:
        return str(bundled_binary)
    return shutil.which(name)
