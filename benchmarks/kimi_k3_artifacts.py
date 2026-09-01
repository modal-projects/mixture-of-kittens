"""Deterministic artifact packaging for the Kimi K3 benchmark."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path


def reproducible_tar_bytes(source: Path) -> bytes:
    """Return a normalized tar stream for every file directly under source."""
    paths = sorted(path for path in source.iterdir() if path.is_file())
    buffer = io.BytesIO()
    with tarfile.open(
        fileobj=buffer,
        mode="w",
        format=tarfile.USTAR_FORMAT,
    ) as archive:
        for path in paths:
            payload = path.read_bytes()
            info = tarfile.TarInfo(path.name)
            info.size = len(payload)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


__all__ = ["reproducible_tar_bytes"]
