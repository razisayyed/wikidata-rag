from __future__ import annotations

import time
import socket
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict
from urllib.error import HTTPError, URLError

from SPARQLWrapper import JSON, SPARQLWrapper

from ..settings import WIKIDATA_ENDPOINT, WIKIDATA_USER_AGENT

WIKIDATA_SPARQL_TIMEOUT_SECONDS = 20
WIKIDATA_SPARQL_MAX_RETRIES = 1


class WikidataServiceError(RuntimeError):
    """Raised when Wikidata SPARQL is unavailable (timeout/non-200)."""


def _is_timeout_error(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    if isinstance(exc, URLError):
        reason = getattr(exc, "reason", None)
        if isinstance(reason, (TimeoutError, socket.timeout)):
            return True
    text = str(exc).lower()
    return "timed out" in text or "timeout" in text


def _retry_after_seconds(exc: HTTPError) -> float | None:
    headers = getattr(exc, "headers", None)
    if not headers:
        return None
    raw_value = headers.get("Retry-After")
    if not raw_value:
        return None
    value = str(raw_value).strip()
    if not value:
        return None

    try:
        seconds = float(value)
        return max(0.0, seconds)
    except ValueError:
        pass

    try:
        dt = parsedate_to_datetime(value)
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return max(0.0, (dt - now).total_seconds())
    except Exception:
        return None


def get_sparql_client() -> SPARQLWrapper:
    client = SPARQLWrapper(WIKIDATA_ENDPOINT)
    client.setReturnFormat(JSON)
    client.addCustomHttpHeader("User-Agent", WIKIDATA_USER_AGENT)
    if hasattr(client, "setTimeout"):
        try:
            client.setTimeout(WIKIDATA_SPARQL_TIMEOUT_SECONDS)
        except Exception:
            pass
    return client


def run_sparql(query: str) -> Dict[str, Any]:
    client = get_sparql_client()
    client.setQuery(query)
    last_http_error: HTTPError | None = None

    for attempt in range(WIKIDATA_SPARQL_MAX_RETRIES + 1):
        try:
            result = client.query().convert()
            return result  # type: ignore[return-value]
        except HTTPError as exc:
            last_http_error = exc
            retry_after = _retry_after_seconds(exc)
            should_retry = (
                attempt < WIKIDATA_SPARQL_MAX_RETRIES and retry_after is not None
            )
            if should_retry:
                time.sleep(retry_after)
                continue
            raise WikidataServiceError(
                "WIKIDATA_UNAVAILABLE: "
                f"HTTP {exc.code} from Wikidata SPARQL endpoint"
                + (
                    f" (Retry-After={retry_after:.0f}s respected)"
                    if retry_after is not None
                    else ""
                )
            ) from exc
        except Exception as exc:
            if _is_timeout_error(exc):
                raise WikidataServiceError(
                    f"WIKIDATA_UNAVAILABLE: SPARQL request timed out after {WIKIDATA_SPARQL_TIMEOUT_SECONDS}s"
                ) from exc
            raise

    if last_http_error is not None:
        raise WikidataServiceError(
            f"WIKIDATA_UNAVAILABLE: HTTP {last_http_error.code} from Wikidata SPARQL endpoint"
        ) from last_http_error
    raise WikidataServiceError("WIKIDATA_UNAVAILABLE: Unknown SPARQL failure")
