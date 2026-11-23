"""Reading JSONL files an agent's own commands can write.

Each line is cut to a bound before it is parsed, and whatever the decoder raises on a line
makes that line unreadable rather than an error: a line nested past the decoder's depth
raises RecursionError, an integer of too many digits ValueError, and neither may end the
reader. Unreadable lines are counted in one warning that names the file.
"""

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


def json_objects(
    path: Path, *, max_line_bytes: int, logger: logging.Logger | None = None
) -> Iterator[dict[str, Any]]:
    """The JSON objects of the file at *path*, one per line, in order.

    A line longer than *max_line_bytes*, one that is not a JSON object, and one the decoder
    refuses for any reason are skipped and counted, and when there were any, one warning on
    *logger* names the file and the count. Bytes that are not UTF-8 are replaced. An
    ``OSError`` opening or reading the file is raised."""
    unreadable = 0
    try:
        with open(path, "rb") as handle:
            while line := handle.readline(max_line_bytes + 1):
                if len(line) > max_line_bytes and not line.endswith(b"\n"):
                    unreadable += 1
                    while line and not line.endswith(b"\n"):
                        line = handle.readline(max_line_bytes + 1)
                    continue
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                try:
                    obj = json.loads(text)
                except Exception:  # noqa: BLE001 -- the agent can write the line
                    obj = None
                if isinstance(obj, dict):
                    yield obj
                else:
                    unreadable += 1
    finally:
        if unreadable:
            (logger or _log).warning(
                f"{path}: {unreadable} lines were not JSON objects of at most "
                f"{max_line_bytes} bytes that could be read; they are skipped"
            )
