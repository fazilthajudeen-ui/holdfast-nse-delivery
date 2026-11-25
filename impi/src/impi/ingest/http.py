"""Polite NSE HTTP client: browser headers, cookie warm-up, >= 1 s between requests, retries with backoff."""
from __future__ import annotations

import time

import requests

from impi.common import config, log


class Fetcher:
    """get(url) -> (status, content). status: ok | missing | blocked | error.
    'missing' = 404 or an HTML page served with 200 (NSE does that on holidays)."""

    def __init__(self):
        c = config()["data"]["download"]
        self.min_interval = max(1.0, float(c["min_interval_seconds"]))
        self.retries = int(c["retries"])
        self.backoff = float(c["backoff_seconds"])
        self.timeout = float(c["timeout_seconds"])
        self.home = c["home"]
        self.user_agent = c["user_agent"]
        self._new_session()
        self._last = 0.0
        self.consecutive_blocks = 0
        self.n_requests = 0
        self.warm_up()

    def _new_session(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml,application/json;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/",
        })

    def _wait(self):
        gap = time.monotonic() - self._last
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last = time.monotonic()

    def warm_up(self):
        try:
            self._wait()
            r = self.session.get(self.home, timeout=self.timeout)
            self.n_requests += 1
            log(f"warm-up {self.home} -> HTTP {r.status_code}, cookies: {len(self.session.cookies)}")
        except requests.RequestException as e:
            log(f"warm-up failed ({e}); continuing without cookies")

    @staticmethod
    def _looks_like_html(content: bytes) -> bool:
        head = content[:300].lstrip().lower()
        return head.startswith(b"<!doctype") or head.startswith(b"<html")

    def get(self, url: str) -> tuple[str, bytes | None]:
        status, detail = "error", ""
        for attempt in range(self.retries + 1):
            self._wait()
            try:
                r = self.session.get(url, timeout=self.timeout)
                self.n_requests += 1
            except requests.RequestException as e:
                status, detail = "error", str(e)[:120]
            else:
                if r.status_code == 200:
                    self.consecutive_blocks = 0
                    if self._looks_like_html(r.content):
                        return "missing", None
                    return "ok", r.content
                if r.status_code == 404:
                    self.consecutive_blocks = 0
                    return "missing", None
                status = "blocked" if r.status_code in (401, 403) else "error"
                detail = f"HTTP {r.status_code}"
            if attempt < self.retries:
                wait = self.backoff * 2 ** attempt
                log(f"  {detail} for {url.rsplit('/', 1)[-1][:80]}; retry {attempt + 1} in {wait:.0f}s")
                time.sleep(wait)
                if status == "blocked":
                    self.warm_up()
                elif detail and "HTTP" not in detail:
                    # connection-level failure: the pooled keep-alive connection can go stale after
                    # thousands of requests -> start a fresh session (seen 10 Oct 2026, ~7,000 requests in)
                    self.session.close()
                    self._new_session()
                    self.warm_up()
        if status == "blocked":
            self.consecutive_blocks += 1
        return status, None

    def get_json(self, url: str):
        status, content = self.get(url)
        if status != "ok":
            return status, None
        import json
        try:
            return "ok", json.loads(content.decode("utf-8", "ignore"))
        except ValueError:
            # NSE APIs answer an HTML/empty body when cookies are stale: warm up and retry once
            self.warm_up()
            status, content = self.get(url)
            if status != "ok":
                return status, None
            try:
                return "ok", json.loads(content.decode("utf-8", "ignore"))
            except ValueError:
                return "error", None
