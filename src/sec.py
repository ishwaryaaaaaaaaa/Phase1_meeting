"""Tiny SEC EDGAR client: polite (rate-limited), cached on disk.

Every download is saved under data/sec_cache so re-running never hits SEC again.
"""
import hashlib
import json
import time

import requests

from config import SEC_CACHE, SEC_MIN_INTERVAL, sec_headers

_last_call = [0.0]


def _cache_path(url: str, suffix: str):
    h = hashlib.sha1(url.encode()).hexdigest()[:16]
    return SEC_CACHE / f"{h}{suffix}"


def get(url: str, as_json: bool = False):
    """GET a URL from SEC (cached). Returns text, or parsed JSON if as_json."""
    path = _cache_path(url, ".json" if as_json else ".html")
    if path.exists():
        txt = path.read_text(encoding="utf-8", errors="ignore")
        return json.loads(txt) if as_json else txt

    wait = SEC_MIN_INTERVAL - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    for attempt in range(4):
        r = requests.get(url, headers=sec_headers(), timeout=60)
        _last_call[0] = time.time()
        if r.status_code == 200:
            break
        if r.status_code in (429, 503):
            time.sleep(2 ** attempt * 2)
            continue
        r.raise_for_status()
    else:
        raise RuntimeError(f"SEC request failed repeatedly: {url}")
    path.write_text(r.text, encoding="utf-8")
    return r.json() if as_json else r.text
