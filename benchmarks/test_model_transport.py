import http.client
import json
import unittest
from unittest import mock

import httpx

from code_agent_baseline.model_transport import ModelTransportError, parse_retry_after, request_chat_completion


class FakeResponse:
    def __init__(self, payload: bytes | Exception) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def valid_response(content: str = "ok") -> bytes:
    return json.dumps({"choices": [{"message": {"content": content}}]}).encode()


class ModelTransportTest(unittest.TestCase):
    def test_incomplete_read_retries_with_fresh_request_and_records_recovery(self) -> None:
        events = []
        first = FakeResponse(http.client.IncompleteRead(b'{"choices":'))
        second = FakeResponse(valid_response("recovered"))
        with mock.patch("urllib.request.urlopen", side_effect=[first, second]) as urlopen, mock.patch(
            "time.sleep"
        ) as sleep:
            result = request_chat_completion(
                url="https://example.test/chat/completions",
                model="demo",
                messages_for_attempt=lambda attempt: [{"role": "user", "content": f"attempt {attempt}"}],
                retries=2,
                backoff_sec=0.1,
                retry_events=events,
            )

        self.assertEqual(result, "recovered")
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(0.1)
        self.assertEqual(events[0]["category"], "transport_error")
        self.assertEqual(events[0]["status"], "retrying")
        self.assertEqual(events[1]["status"], "success_after_retry")

    def test_invalid_json_and_empty_content_are_retryable(self) -> None:
        events = []
        responses = [
            FakeResponse(b'{"choices":'),
            FakeResponse(valid_response("")),
            FakeResponse(valid_response("done")),
        ]
        with mock.patch("urllib.request.urlopen", side_effect=responses), mock.patch("time.sleep"):
            result = request_chat_completion(
                url="https://example.test/chat/completions",
                model="demo",
                messages_for_attempt=lambda attempt: [{"role": "user", "content": "task"}],
                retries=2,
                backoff_sec=0,
                retry_events=events,
            )

        self.assertEqual(result, "done")
        self.assertEqual([event["category"] for event in events[:2]], ["invalid_response", "invalid_response"])
        self.assertEqual(events[-1]["status"], "success_after_retry")

    def test_retry_exhaustion_raises_and_records_all_attempts(self) -> None:
        events = []
        failures = [FakeResponse(http.client.IncompleteRead(b"partial")) for _ in range(3)]
        with mock.patch("urllib.request.urlopen", side_effect=failures), mock.patch("time.sleep"):
            with self.assertRaises(ModelTransportError) as raised:
                request_chat_completion(
                    url="https://example.test/chat/completions",
                    model="demo",
                    messages_for_attempt=lambda attempt: [{"role": "user", "content": "task"}],
                    retries=2,
                    backoff_sec=0,
                    retry_events=events,
                )

        self.assertIn("after 3 attempts", str(raised.exception))
        self.assertEqual(len(events), 3)
        self.assertEqual(events[-1]["status"], "failed")

    def test_retry_after_is_bounded(self) -> None:
        self.assertEqual(parse_retry_after("2.5"), 2.5)
        self.assertEqual(parse_retry_after("500"), 120.0)
        self.assertIsNone(parse_retry_after("tomorrow"))

    def test_sends_v4_sampling_options_and_records_usage(self) -> None:
        response_events = []
        response = {
            "model": "deepseek-v4-flash",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": '{"tool_name":"finish","arguments":{}}', "reasoning_content": "done"},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        }
        with mock.patch("urllib.request.urlopen", return_value=FakeResponse(json.dumps(response).encode())) as urlopen:
            result = request_chat_completion(
                url="https://example.test/chat/completions",
                model="deepseek-v4-flash",
                messages_for_attempt=lambda attempt: [{"role": "user", "content": "task"}],
                temperature=1.0,
                top_p=0.95,
                thinking="enabled",
                reasoning_effort="max",
                response_events=response_events,
            )

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(result, '{"tool_name":"finish","arguments":{}}')
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertEqual(payload["reasoning_effort"], "max")
        self.assertEqual(payload["temperature"], 1.0)
        self.assertEqual(payload["top_p"], 0.95)
        self.assertEqual(response_events[0]["usage"]["prompt_tokens"], 10)
        self.assertEqual(response_events[0]["reasoning_chars"], 4)

    def test_local_vllm_receives_qwen_chat_template_thinking_control(self) -> None:
        response = {
            "model": "qwen-local",
            "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
            "usage": {},
        }
        with mock.patch("urllib.request.urlopen", return_value=FakeResponse(json.dumps(response).encode())) as urlopen:
            request_chat_completion(
                url="http://127.0.0.1:8000/v1/chat/completions",
                model="qwen-local",
                messages_for_attempt=lambda attempt: [{"role": "user", "content": "task"}],
                thinking="disabled",
            )

        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})

    def test_direct_mode_disables_system_proxy(self) -> None:
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.iter_bytes.return_value = [valid_response("direct")]
        client = mock.Mock()
        client.build_request.return_value = mock.sentinel.request
        client.send.return_value = response
        with mock.patch("httpx.Client", return_value=client) as client_class:
            result = request_chat_completion(
                url="https://example.test/chat/completions",
                model="demo",
                messages_for_attempt=lambda attempt: [{"role": "user", "content": "task"}],
                use_system_proxy=False,
            )

        self.assertEqual(result, "direct")
        client_class.assert_called_once_with(timeout=180, trust_env=False)
        client.send.assert_called_once_with(mock.sentinel.request, stream=True)
        response.close.assert_called_once()
        client.close.assert_called_once()

    def test_direct_mode_recovers_complete_json_from_incomplete_chunked_response(self) -> None:
        def chunks():
            yield valid_response("recovered partial")
            raise httpx.RemoteProtocolError("incomplete chunked read")

        events = []
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.iter_bytes.return_value = chunks()
        client = mock.Mock()
        client.build_request.return_value = mock.sentinel.request
        client.send.return_value = response
        with mock.patch("httpx.Client", return_value=client):
            result = request_chat_completion(
                url="https://example.test/chat/completions",
                model="demo",
                messages_for_attempt=lambda attempt: [{"role": "user", "content": "task"}],
                use_system_proxy=False,
                retry_events=events,
            )

        self.assertEqual(result, "recovered partial")
        self.assertEqual(events[0]["status"], "recovered_partial_response")
        response.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
