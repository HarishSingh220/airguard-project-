"""
http_utils.py
Small retry wrapper around requests.get for the ingestion scripts that call
external APIs (Open-Meteo, CPCB/data.gov.in, NASA FIRMS). Previously a single
dropped connection, timeout, or transient 5xx from these free/public APIs
failed the whole ingestion step (or backfill chunk) immediately. This retries
a handful of times with a short backoff before giving up, and still raises
-- with the original failure as the exception's __cause__ -- if every
attempt fails, so every caller's existing try/except/log_run error handling
is unaffected; it only gets fewer spurious failures from blips that would
have succeeded on a second try.
"""
import time

import requests

# Status codes worth retrying: rate limiting and server-side/transient errors.
# 4xx other than 429 (bad request, auth, not found, ...) will not fix itself
# on a retry, so those are returned as-is for the caller's resp.raise_for_status().
# 520-524 are Cloudflare's own edge codes (web server down / origin unreachable /
# connection timed out / origin SSL handshake failed / a read timeout to the
# origin) -- data.gov.in sits behind Cloudflare and has been observed returning
# 523 ("origin is unreachable") during brief upstream outages; those are exactly
# as transient as a 502/503 from an origin that reports its own status directly,
# so they belong in this set too.
TRANSIENT_STATUS = {429, 500, 502, 503, 504, 520, 521, 522, 523, 524}


def get_with_retry(url, *, params=None, timeout=30, max_attempts=4, backoff=2.0, **kwargs):
    """requests.get with retries on connection errors, timeouts, and transient
    HTTP status codes (429/5xx).

    Returns the final `requests.Response` (which may still carry a transient
    status code if every retry also returned one -- callers already call
    resp.raise_for_status() to turn that into an exception). Raises
    requests.ConnectionError / requests.Timeout if every attempt errors out
    at the transport level before a response was even received.
    """
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout, **kwargs)
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            if attempt == max_attempts:
                raise type(exc)(
                    f"GET {url} failed after {max_attempts} attempts: {exc}"
                ) from exc
            time.sleep(backoff * attempt)
            continue
        if resp.status_code in TRANSIENT_STATUS and attempt < max_attempts:
            time.sleep(backoff * attempt)
            continue
        return resp
    raise last_exc  # pragma: no cover -- loop above always returns or raises first