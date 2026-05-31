"""Shared HTTP retry helper for ingestion clients.

Why this lives in ``ingestion/_http.py`` and not at the call site:
* All ingestion sources share the same retry policy: 429 + 5xx are
  transient, 4xx (except 429) is permanent.
* All sources share the same backoff curve.
* Centralising means one place to add observability later.

This module is **synchronous**. The engine's ingestion path is batch /
nightly — there's no concurrency benefit from async. The Schwab client
will use async because OAuth flows naturally interleave; that's separate.
"""

from __future__ import annotations

import time
from typing import Any

import httpx


class HTTPError(RuntimeError):
    """Permanent HTTP failure surfaced to the caller."""


class TransientHTTPError(HTTPError):
    """Persisted past the retry budget; caller decides whether to give up."""


_RETRYABLE: frozenset[int] = frozenset({429, 500, 502, 503, 504})
_HTTP_BAD_REQUEST: int = 400  # lowest 4xx; everything below is success-class


def get_with_retry(
    client: httpx.Client,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    retries: int = 3,
    backoff_s: float = 0.5,
    backoff_cap_s: float = 8.0,
) -> httpx.Response:
    """GET ``url`` with exponential backoff on 429 and 5xx responses.

    :param client: a configured ``httpx.Client`` (caller owns lifecycle).
    :param url: full URL.
    :param params: query string parameters.
    :param headers: per-request headers (merged with the client's defaults).
    :param retries: maximum number of retries **after** the first attempt.
        Total attempts = retries + 1.
    :param backoff_s: base sleep, doubled per attempt up to ``backoff_cap_s``.
    :param backoff_cap_s: max single sleep between attempts.
    :raises HTTPError: 4xx (non-429) — the request is permanently broken.
    :raises TransientHTTPError: 429/5xx persisted past the retry budget,
        or the underlying transport raised on every attempt.

    Honors ``Retry-After`` header on 429 if present (seconds form only;
    HTTP-date form is parsed best-effort).
    """
    last_exc: Exception | None = None
    last_resp: httpx.Response | None = None

    for attempt in range(retries + 1):
        try:
            resp = client.get(url, params=params, headers=headers)
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last_exc = e
            _sleep_for(attempt, backoff_s, backoff_cap_s, retry_after=None)
            continue

        if resp.status_code < _HTTP_BAD_REQUEST:
            return resp

        if resp.status_code in _RETRYABLE and attempt < retries:
            last_resp = resp
            retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
            _sleep_for(attempt, backoff_s, backoff_cap_s, retry_after=retry_after)
            continue

        # Permanent failure (4xx other than 429, or exhausted budget on 5xx)
        if resp.status_code in _RETRYABLE:
            msg = f"GET {url} failed after {retries + 1} attempts: status={resp.status_code}"
            raise TransientHTTPError(msg)
        msg = f"GET {url} returned status={resp.status_code}"
        raise HTTPError(msg)

    # All attempts raised at the transport layer
    msg = f"GET {url} failed all {retries + 1} attempts at the transport layer"
    if last_resp is not None:
        msg += f" (last status={last_resp.status_code})"
    raise TransientHTTPError(msg) from last_exc


def _sleep_for(
    attempt: int,
    base_s: float,
    cap_s: float,
    *,
    retry_after: float | None,
) -> None:
    """Sleep before the next attempt. ``retry_after`` overrides the curve."""
    if retry_after is not None and retry_after >= 0:
        time.sleep(min(retry_after, cap_s))
        return
    delay = min(base_s * (2**attempt), cap_s)
    time.sleep(delay)


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header. Seconds-form only; date-form returns None."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
