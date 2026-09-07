"""POST the assembled payload to the benchmark platform."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from .config import PLATFORM_URL, PUSH_ENDPOINT, PROVIDER_KEYS

logger = logging.getLogger("push_to_platform.push")


def push_payload(payload: dict, provider: str) -> dict:
    """Push one payload; return the platform's JSON response.

    Raises ``RuntimeError`` on transport error or a non-2xx HTTP status, so the
    caller (shell wrapper) can treat a non-zero exit as "not pushed".
    """
    key = PROVIDER_KEYS.get(provider)
    if not key:
        raise RuntimeError(
            f"no API key for provider {provider!r}; set BENCHMARK_PUSH_KEYS"
        )

    url = f"{PLATFORM_URL}{PUSH_ENDPOINT}"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )

    logger.info("POST %s (provider=%s, start=%s, request_id=%s)",
                url, provider, payload["start_date"], payload["request_id"])
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            status = resp.status
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(
            f"push failed: HTTP {exc.code} {exc.reason} {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"push failed: {exc.reason}") from exc

    if status >= 400:
        raise RuntimeError(f"push failed: HTTP {status} {data}")

    logger.info("push result: status=%s receipt_id=%s warnings=%s",
                data.get("status"), data.get("receipt_id"), data.get("warnings"))
    return data
