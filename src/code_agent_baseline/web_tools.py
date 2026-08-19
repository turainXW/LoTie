from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


GITHUB_REPO_RE = re.compile(r"^https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?$")


def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    max_results = max(1, min(int(max_results or 5), 10))
    if os.environ.get("BOCHA_API_KEY"):
        return _bocha_search(query, max_results)
    if os.environ.get("BRAVE_SEARCH_API_KEY"):
        return _brave_search(query, max_results)
    if os.environ.get("TAVILY_API_KEY"):
        return _tavily_search(query, max_results)
    return _github_repo_search(query, max_results)


def download_repo(repo_url: str, download_dir: str | Path, dest_name: str | None = None) -> dict[str, Any]:
    match = GITHUB_REPO_RE.match(repo_url.strip())
    if not match:
        return {
            "ok": False,
            "error": "Only public GitHub repository URLs are allowed, e.g. https://github.com/org/repo.",
        }

    owner, repo = match.groups()
    dest_root = Path(download_dir).resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    safe_name = dest_name or f"{owner}__{repo}"
    if not re.match(r"^[A-Za-z0-9_.-]+$", safe_name):
        return {"ok": False, "error": f"Unsafe destination name: {safe_name}"}

    dest = (dest_root / safe_name).resolve()
    if dest_root not in dest.parents:
        return {"ok": False, "error": "Destination escapes configured download directory."}
    if dest.exists():
        return {
            "ok": True,
            "repo_url": repo_url,
            "path": str(dest),
            "already_exists": True,
        }

    clone_url = f"https://github.com/{owner}/{repo}.git"
    completed = subprocess.run(
        ["git", "clone", "--depth", "1", clone_url, str(dest)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
    )
    return {
        "ok": completed.returncode == 0,
        "repo_url": repo_url,
        "path": str(dest),
        "returncode": completed.returncode,
        "output": completed.stdout[-4000:],
    }


def _github_repo_search(query: str, max_results: int) -> dict[str, Any]:
    params = urllib.parse.urlencode({"q": query, "per_page": str(max_results)})
    url = f"https://api.github.com/search/repositories?{params}"
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "codeagent-rl"})
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        data = _urlopen_json_with_retries(request, timeout=30)
    except Exception as exc:
        return {
            "provider": "github",
            "ok": False,
            "error": str(exc),
            "hint": "Set BOCHA_API_KEY, BRAVE_SEARCH_API_KEY, or TAVILY_API_KEY for broader web search.",
        }
    results = []
    for item in data.get("items", [])[:max_results]:
        results.append(
            {
                "name": item.get("full_name"),
                "url": item.get("html_url"),
                "description": item.get("description"),
                "stars": item.get("stargazers_count"),
                "language": item.get("language"),
                "license": (item.get("license") or {}).get("spdx_id"),
            }
        )
    return {"provider": "github", "ok": True, "query": query, "results": results}


def _bocha_search(query: str, max_results: int) -> dict[str, Any]:
    payload = json.dumps(
        {
            "query": query,
            "summary": True,
            "count": max_results,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        os.environ.get("BOCHA_SEARCH_URL", "https://api.bochaai.com/v1/web-search"),
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['BOCHA_API_KEY']}",
            "User-Agent": "codeagent-rl",
        },
    )
    try:
        data = _urlopen_json_with_retries(request, timeout=30)
    except Exception as exc:
        return {"provider": "bocha", "ok": False, "query": query, "error": str(exc), "results": []}
    results = _extract_bocha_results(data, max_results)
    return {
        "provider": "bocha",
        "ok": _bocha_response_ok(data),
        "query": query,
        "results": results,
        "raw_code": data.get("code") if isinstance(data, dict) else None,
        "raw_message": data.get("msg") or data.get("message") if isinstance(data, dict) else None,
    }


def _bocha_response_ok(data: dict[str, Any]) -> bool:
    code = data.get("code")
    if code is None:
        return bool(data.get("data") or data.get("results") or data.get("list"))
    return str(code) in {"0", "200"}


def _extract_bocha_results(data: dict[str, Any], max_results: int) -> list[dict[str, Any]]:
    nested = data.get("data") if isinstance(data.get("data"), dict) else data
    candidates = []
    web_pages = nested.get("webPages") if isinstance(nested, dict) else None
    if isinstance(web_pages, dict) and isinstance(web_pages.get("value"), list):
        candidates = web_pages["value"]
    elif isinstance(nested, dict):
        for key in ("results", "list", "items", "value"):
            if isinstance(nested.get(key), list):
                candidates = nested[key]
                break
    results = []
    for item in candidates[:max_results]:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "title": item.get("name") or item.get("title"),
                "url": item.get("url"),
                "description": item.get("summary") or item.get("snippet") or item.get("description"),
                "site_name": item.get("siteName") or item.get("site_name"),
                "date": item.get("dateLastCrawled") or item.get("date"),
            }
        )
    return results


def _brave_search(query: str, max_results: int) -> dict[str, Any]:
    params = urllib.parse.urlencode({"q": query, "count": str(max_results)})
    request = urllib.request.Request(
        f"https://api.search.brave.com/res/v1/web/search?{params}",
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": os.environ["BRAVE_SEARCH_API_KEY"],
            "User-Agent": "codeagent-rl",
        },
    )
    data = _urlopen_json_with_retries(request, timeout=30)
    results = []
    for item in data.get("web", {}).get("results", [])[:max_results]:
        results.append({"title": item.get("title"), "url": item.get("url"), "description": item.get("description")})
    return {"provider": "brave", "ok": True, "query": query, "results": results}


def _tavily_search(query: str, max_results: int) -> dict[str, Any]:
    payload = json.dumps({"query": query, "max_results": max_results}).encode("utf-8")
    request = urllib.request.Request(
        "https://api.tavily.com/search",
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}",
        },
    )
    data = _urlopen_json_with_retries(request, timeout=30)
    results = []
    for item in data.get("results", [])[:max_results]:
        results.append({"title": item.get("title"), "url": item.get("url"), "description": item.get("content")})
    return {"provider": "tavily", "ok": True, "query": query, "results": results}


def _urlopen_json_with_retries(request: urllib.request.Request, *, timeout: int, attempts: int = 3) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            last_exc = exc
            if attempt + 1 >= attempts:
                break
            time.sleep(0.5 * (2**attempt))
    assert last_exc is not None
    raise last_exc
