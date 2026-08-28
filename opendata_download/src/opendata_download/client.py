"""Thin wrapper around ``ecmwf.opendata.Client``.

The upstream client retries HTTP *status* failures, but NOT broken connections
mid-stream (``ChunkedEncodingError`` / ``ProtocolError`` / timeouts), which are
common on long downloads over flaky links. We add that retry here, plus *atomic*
delivery: fields are written to a ``.part`` temp file and renamed into place
only on success, so a file at its final path is always a complete download —
which makes the downloader idempotent for unattended server use.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Mapping

import requests
from ecmwf.opendata import Client
from urllib3.exceptions import ProtocolError

logger = logging.getLogger(__name__)

# Transient network errors — safe to retry (an idempotent HTTP GET).
_RETRYABLE_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.Timeout,
    ProtocolError,
)


def _is_retryable(exc: BaseException) -> bool:
    """Transient -> True; permanent (e.g. 4xx) -> False."""
    if isinstance(exc, requests.exceptions.HTTPError):
        resp = getattr(exc, "response", None)
        # Retry only server-side 5xx / 429; a 4xx means the field is absent.
        return resp is not None and resp.status_code >= 500
    return isinstance(exc, _RETRYABLE_EXCEPTIONS)


class OpenDataClient:
    def __init__(
        self,
        source: str = "ecmwf",
        retries: int = 6,
        backoff_seconds: float = 15.0,
    ):
        self._client = Client(source=source)
        self._retries = retries
        self._backoff = backoff_seconds

    def retrieve(self, request: Mapping[str, Any], target: str) -> str:
        """Download fields to ``target`` atomically, retrying transient errors."""
        part = f"{target}.part"
        last_exc: BaseException | None = None
        try:
            for attempt in range(1, self._retries + 1):
                # Clear any partial file from a previous attempt so each retry
                # is a clean re-download.
                if os.path.exists(part):
                    os.remove(part)
                try:
                    self._client.retrieve(request=dict(request), target=part)
                    os.replace(part, target)
                    return target
                except Exception as exc:  # noqa: BLE001 - retry decision is explicit
                    last_exc = exc
                    if not _is_retryable(exc):
                        raise  # permanent error — surface it as-is
                    if attempt == self._retries:
                        break
                    wait = self._backoff * attempt
                    logger.warning(
                        "attempt %d/%d failed for %s: %s; retrying in %.0fs",
                        attempt, self._retries, target, exc, wait,
                    )
                    time.sleep(wait)
            raise RuntimeError(
                f"failed to download {target} after {self._retries} attempts"
            ) from last_exc
        finally:
            # Drop any leftover temp file on every exit path, including Ctrl-C.
            if os.path.exists(part):
                os.remove(part)

    def latest(self, request: Mapping[str, Any] | None = None):
        """Most recent fully-available cycle for ``request`` (datetime or raises)."""
        return self._client.latest(request=dict(request) if request else None)
