from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

import httpx


class ModelTransportError(RuntimeError):
    pass


def request_chat_completion(
    *,
    url: str,
    model: str,
    messages_for_attempt: Callable[[int], list[dict[str, Any]]],
    api_key: str = "",
    timeout_sec: int = 180,
    max_tokens: int = 4000,
    temperature: float = 0.0,
    top_p: float | None = None,
    thinking: str | None = None,
    reasoning_effort: str | None = None,
    retries: int = 6,
    backoff_sec: float = 2.0,
    max_backoff_sec: float = 30.0,
    retry_events: list[dict[str, Any]] | None = None,
    response_events: list[dict[str, Any]] | None = None,
    use_system_proxy: bool = True,
) -> str:
    """Call an OpenAI-compatible chat endpoint with fresh-connection retries."""

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Connection": "close",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    retryable_statuses = {400, 408, 409, 425, 429, 500, 502, 503, 504}
    events = retry_events if retry_events is not None else []
    last_error = ""

    direct_client = httpx.Client(timeout=timeout_sec, trust_env=False) if not use_system_proxy else None

    try:
        for attempt in range(retries + 1):
            messages = messages_for_attempt(attempt)
            request_payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if top_p is not None:
                request_payload["top_p"] = top_p
            if thinking is not None:
                request_payload["thinking"] = {"type": thinking}
                if is_local_openai_endpoint(url):
                    request_payload["chat_template_kwargs"] = {
                        "enable_thinking": thinking == "enabled",
                    }
            if reasoning_effort is not None:
                request_payload["reasoning_effort"] = reasoning_effort
            payload = json.dumps(request_payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            request = urllib.request.Request(url, data=payload, headers=headers)
            started = time.monotonic()
            try:
                recovered_transport_error: httpx.TransportError | None = None
                if direct_client is not None:
                    httpx_request = direct_client.build_request("POST", url, content=payload, headers=headers)
                    response = direct_client.send(httpx_request, stream=True)
                    try:
                        # HTTPStatusError keeps the streaming response. Buffer error bodies before
                        # raising so the retry handler can safely inspect response.text.
                        if response.is_error:
                            response.read()
                        response.raise_for_status()
                        raw_buffer = bytearray()
                        try:
                            for chunk in response.iter_bytes():
                                raw_buffer.extend(chunk)
                        except httpx.TransportError as exc:
                            recovered_transport_error = exc
                        raw = bytes(raw_buffer)
                    finally:
                        response.close()
                else:
                    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                        raw = response.read()
                try:
                    data = json.loads(raw.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    if recovered_transport_error is not None:
                        raise recovered_transport_error
                    raise
                choice = data["choices"][0]
                message = choice["message"]
                content = message.get("content") or ""
                if not content.strip():
                    raise ValueError("model response content is empty")
                if response_events is not None:
                    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
                    response_events.append(
                        {
                            "attempt": attempt + 1,
                            "model": data.get("model") or model,
                            "finish_reason": choice.get("finish_reason"),
                            "reasoning_chars": len(str(message.get("reasoning_content") or "")),
                            "usage": usage,
                            "response_bytes": len(raw),
                            "elapsed_sec": round(time.monotonic() - started, 4),
                        }
                    )
                if attempt:
                    events.append(
                        {
                            "attempt": attempt + 1,
                            "status": "success_after_retry",
                            "elapsed_sec": round(time.monotonic() - started, 4),
                            "response_bytes": len(raw),
                        }
                    )
                if recovered_transport_error is not None:
                    events.append(
                        {
                            "attempt": attempt + 1,
                            "status": "recovered_partial_response",
                            "category": "transport_error",
                            "error": repr(recovered_transport_error)[:2000],
                            "elapsed_sec": round(time.monotonic() - started, 4),
                            "response_bytes": len(raw),
                        }
                    )
                return content
            except httpx.HTTPStatusError as exc:
                body = exc.response.text
                last_error = f"HTTP {exc.response.status_code} from model API: {body[:2000]}"
                retryable = exc.response.status_code in retryable_statuses
                retry_after = parse_retry_after(exc.response.headers.get("Retry-After"))
                category = "http_error"
            except urllib.error.HTTPError as exc:
                body = read_http_error_body(exc)
                last_error = f"HTTP {exc.code} from model API: {body[:2000]}"
                retryable = exc.code in retryable_statuses
                retry_after = parse_retry_after(exc.headers.get("Retry-After") if exc.headers else None)
                category = "http_error"
            except (
                httpx.TransportError,
                http.client.IncompleteRead,
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                OSError,
            ) as exc:
                last_error = f"Transport error from model API: {exc!r}"
                retryable = True
                retry_after = None
                category = "transport_error"
            except (json.JSONDecodeError, UnicodeDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
                last_error = f"Invalid/incomplete model response: {exc!r}"
                retryable = True
                retry_after = None
                category = "invalid_response"

            wait_sec = retry_after if retry_after is not None else min(max_backoff_sec, backoff_sec * (2**attempt))
            events.append(
                {
                    "attempt": attempt + 1,
                    "status": "retrying" if retryable and attempt < retries else "failed",
                    "category": category,
                    "error": last_error[:2000],
                    "request_message_count": len(messages),
                    "request_chars": sum(len(str(message.get("content", ""))) for message in messages),
                    "elapsed_sec": round(time.monotonic() - started, 4),
                    "wait_sec": round(wait_sec, 3) if retryable and attempt < retries else 0.0,
                }
            )
            if not retryable or attempt >= retries:
                raise ModelTransportError(f"{last_error} after {attempt + 1} attempts")
            time.sleep(wait_sec)
    finally:
        if direct_client is not None:
            direct_client.close()

    raise ModelTransportError(last_error or "Model API failed without an error body")


def is_local_openai_endpoint(url: str) -> bool:
    return url.startswith("http://127.0.0.1:") or url.startswith("http://localhost:")


def read_http_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read()
    except http.client.IncompleteRead as incomplete:
        raw = incomplete.partial
    return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, min(120.0, float(value)))
    except ValueError:
        return None
