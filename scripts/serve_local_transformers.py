#!/usr/bin/env python3
"""Serve a local Transformers checkpoint through a small OpenAI-compatible API."""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
import uuid
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("lottie.local_model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Local Hugging Face base checkpoint directory.")
    parser.add_argument("--adapter-path", help="Optional PEFT/LoRA adapter applied to the base checkpoint.")
    parser.add_argument(
        "--merge-adapter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge a supplied adapter into the base weights for inference.",
    )
    parser.add_argument("--served-model-name", default=None, help="Model name exposed by /v1/models.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Server-side generation cap.")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking in compatible Qwen templates.")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


class TransformersBackend:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        import transformers
        from transformers import AutoConfig, AutoTokenizer

        self.torch = torch
        self.transformers = transformers
        self.model_path = str(Path(args.model_path).expanduser().resolve())
        self.adapter_path = (
            str(Path(args.adapter_path).expanduser().resolve()) if args.adapter_path else None
        )
        self.model_name = args.served_model_name or Path(self.model_path).name
        self.max_new_tokens = args.max_new_tokens
        self.enable_thinking = args.enable_thinking
        self.lock = threading.Lock()

        tokenizer_path = select_tokenizer_path(self.model_path, self.adapter_path)
        LOGGER.info("loading tokenizer from %s", tokenizer_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=args.trust_remote_code,
        )
        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=args.trust_remote_code)
        architecture = (getattr(config, "architectures", None) or [None])[0]
        model_class = getattr(transformers, architecture, None) if architecture else None
        if model_class is None:
            from transformers import AutoModelForCausalLM

            model_class = AutoModelForCausalLM

        model_kwargs: dict[str, Any] = {
            "device_map": args.device_map,
            "low_cpu_mem_usage": True,
            "trust_remote_code": args.trust_remote_code,
        }
        dtype = self._resolve_dtype(args.dtype)
        if dtype is not None:
            model_kwargs["dtype"] = dtype
        LOGGER.info("loading %s from %s with dtype=%s", model_class.__name__, self.model_path, dtype)
        self.model = model_class.from_pretrained(self.model_path, **model_kwargs)
        if self.adapter_path:
            from peft import PeftModel

            LOGGER.info("loading adapter from %s", self.adapter_path)
            self.model = PeftModel.from_pretrained(
                self.model,
                self.adapter_path,
                is_trainable=False,
            )
            if args.merge_adapter:
                LOGGER.info("merging adapter into base weights")
                self.model = self.model.merge_and_unload(safe_merge=True)
        self.model.eval()
        self.input_device = next(self.model.parameters()).device
        LOGGER.info("model ready on %s", self.input_device)

    def _resolve_dtype(self, requested: str) -> Any:
        if requested != "auto":
            return getattr(self.torch, requested)
        if self.torch.cuda.is_available():
            return self.torch.bfloat16 if self.torch.cuda.is_bf16_supported() else self.torch.float16
        return None

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = validate_messages(payload.get("messages"))
        requested_max = int(payload.get("max_tokens") or 1024)
        max_new_tokens = max(1, min(requested_max, self.max_new_tokens))
        temperature = float(payload.get("temperature") or 0.0)
        top_p = float(payload.get("top_p") or 1.0)

        template_kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "enable_thinking": self.enable_thinking,
        }
        encoded = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        if isinstance(encoded, Mapping):
            model_inputs = {
                key: value.to(self.input_device)
                for key, value in encoded.items()
                if self.torch.is_tensor(value)
            }
            input_ids = model_inputs["input_ids"]
            model_inputs.setdefault("attention_mask", self.torch.ones_like(input_ids))
        else:
            input_ids = encoded.to(self.input_device)
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": self.torch.ones_like(input_ids),
            }
        generation_kwargs: dict[str, Any] = {
            **model_inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            **generation_token_ids(self.tokenizer, self.model.config),
            "stopping_criteria": self.transformers.StoppingCriteriaList(
                [FirstJsonObjectStoppingCriteria(self.tokenizer, input_ids.shape[-1], self.torch)]
            ),
        }
        if temperature > 0:
            generation_kwargs.update(temperature=temperature, top_p=top_p)

        started = time.monotonic()
        with self.lock, self.torch.inference_mode():
            generated = self.model.generate(**generation_kwargs)
        completion_ids = generated[0, input_ids.shape[-1] :]
        content = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        elapsed = time.monotonic() - started
        LOGGER.info(
            "generated prompt_tokens=%s completion_tokens=%s elapsed=%.2fs",
            input_ids.shape[-1],
            completion_ids.shape[-1],
            elapsed,
        )
        return completion_response(
            model=str(payload.get("model") or self.model_name),
            content=content,
            prompt_tokens=int(input_ids.shape[-1]),
            completion_tokens=int(completion_ids.shape[-1]),
        )


def generation_token_ids(tokenizer: Any, model_config: Any | None = None) -> dict[str, Any]:
    eos_token_ids: list[int] = []
    sources = [tokenizer.eos_token_id]
    if model_config is not None:
        sources.append(getattr(model_config, "eos_token_id", None))
        text_config = getattr(model_config, "text_config", None)
        sources.append(getattr(text_config, "eos_token_id", None))
    for source in sources:
        values = source if isinstance(source, (list, tuple)) else [source]
        for value in values:
            if value is not None and int(value) not in eos_token_ids:
                eos_token_ids.append(int(value))
    if not eos_token_ids:
        raise ValueError("tokenizer must define eos_token_id")
    pad_token_id = tokenizer.pad_token_id
    return {
        "eos_token_id": eos_token_ids[0] if len(eos_token_ids) == 1 else eos_token_ids,
        "pad_token_id": int(pad_token_id if pad_token_id is not None else eos_token_ids[0]),
    }


def select_tokenizer_path(model_path: str, adapter_path: str | None) -> str:
    if adapter_path and (Path(adapter_path) / "tokenizer_config.json").is_file():
        return adapter_path
    return model_path


class FirstJsonObjectStoppingCriteria:
    def __init__(self, tokenizer: Any, prompt_tokens: int, torch_module: Any) -> None:
        self.tokenizer = tokenizer
        self.prompt_tokens = prompt_tokens
        self.torch = torch_module

    def __call__(self, input_ids: Any, scores: Any, **_: Any) -> Any:
        decisions = []
        for row in input_ids:
            text = self.tokenizer.decode(row[self.prompt_tokens :], skip_special_tokens=True)
            decisions.append(contains_complete_top_level_json(text))
        return self.torch.tensor(decisions, dtype=self.torch.bool, device=input_ids.device)


def contains_complete_top_level_json(text: str) -> bool:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        line_prefix = text[text.rfind("\n", 0, index) + 1 : index]
        if line_prefix.strip():
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return True
    return False


def validate_messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("messages must be a non-empty list")
    result = []
    for item in value:
        if not isinstance(item, dict) or item.get("role") not in {"system", "user", "assistant"}:
            raise ValueError("each message must contain a supported role")
        content = item.get("content")
        if not isinstance(content, str):
            raise ValueError("each message content must be a string")
        result.append({"role": str(item["role"]), "content": content})
    return result


def completion_response(
    *, model: str, content: str, prompt_tokens: int, completion_tokens: int
) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def build_handler(backend: TransformersBackend) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "LottieTransformers/1.0"

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._write_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "model": backend.model_name,
                        "base_model_path": backend.model_path,
                        "adapter_path": backend.adapter_path,
                    },
                )
                return
            if self.path == "/v1/models":
                self._write_json(
                    HTTPStatus.OK,
                    {"object": "list", "data": [{"id": backend.model_name, "object": "model"}]},
                )
                return
            self._write_json(HTTPStatus.NOT_FOUND, {"error": {"message": "route not found"}})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/chat/completions":
                self._write_json(HTTPStatus.NOT_FOUND, {"error": {"message": "route not found"}})
                return
            try:
                content_length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                self._write_json(HTTPStatus.OK, backend.complete(payload))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": {"message": str(exc)}})
            except Exception as exc:  # Keep the process alive and expose a useful API error.
                LOGGER.exception("completion failed")
                self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": {"message": repr(exc)}})

        def log_message(self, fmt: str, *args: Any) -> None:
            LOGGER.info("%s - %s", self.address_string(), fmt % args)

        def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    backend = TransformersBackend(args)
    server = ThreadingHTTPServer((args.host, args.port), build_handler(backend))
    LOGGER.info("serving model=%s at http://%s:%s/v1/chat/completions", backend.model_name, args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("stopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
