from __future__ import annotations

import json

import litellm
from jinja2 import StrictUndefined, Template
from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.utils.actions_toolcall import BASH_TOOL


WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web or public GitHub repositories for open-source projects and documentation.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


DOWNLOAD_REPO_TOOL = {
    "type": "function",
    "function": {
        "name": "download_repo",
        "description": "Download a public GitHub repository into the controlled workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "repo_url": {
                    "type": "string",
                    "description": "Public GitHub repository URL, for example https://github.com/org/repo.",
                },
                "dest_name": {
                    "type": "string",
                    "description": "Optional local directory name under the configured download directory.",
                },
            },
            "required": ["repo_url"],
        },
    },
}


class WebEnabledLitellmModel(LitellmModel):
    """LiteLLM mini model that exposes controlled web tools in addition to bash."""

    def __init__(self, *, enable_web_tools: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.enable_web_tools = enable_web_tools

    def _query(self, messages: list[dict[str, str]], **kwargs):
        tools = [BASH_TOOL]
        if self.enable_web_tools:
            tools += [WEB_SEARCH_TOOL, DOWNLOAD_REPO_TOOL]
        try:
            return litellm.completion(
                model=self.config.model_name,
                messages=messages,
                tools=tools,
                **(self.config.model_kwargs | kwargs),
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def _parse_actions(self, response) -> list[dict]:
        tool_calls = response.choices[0].message.tool_calls or []
        return parse_codeagent_toolcall_actions(
            tool_calls,
            format_error_template=self.config.format_error_template,
            template_kwargs={"finish_reason": response.choices[0].finish_reason},
            enable_web_tools=self.enable_web_tools,
        )

    def serialize(self) -> dict:
        data = super().serialize()
        data["info"]["config"]["model"]["enable_web_tools"] = self.enable_web_tools
        return data


def parse_codeagent_toolcall_actions(
    tool_calls: list,
    *,
    format_error_template: str,
    template_kwargs: dict | None = None,
    enable_web_tools: bool,
) -> list[dict]:
    template_kwargs = template_kwargs or {}
    if not tool_calls:
        raise _format_error(
            format_error_template,
            error="No tool calls found in the response. Every response MUST include at least one tool call.",
            actions=[],
            template_kwargs=template_kwargs,
        )

    actions = []
    allowed_tools = {"bash"}
    if enable_web_tools:
        allowed_tools |= {"web_search", "download_repo"}

    for tool_call in tool_calls:
        error_msg = ""
        args = {}
        try:
            args = json.loads(tool_call.function.arguments)
        except Exception as e:
            error_msg = f"Error parsing tool call arguments: {e}."

        tool_name = tool_call.function.name
        if tool_name not in allowed_tools:
            error_msg += f"Unknown tool '{tool_name}'."
        if tool_name == "bash" and (not isinstance(args, dict) or "command" not in args):
            error_msg += "Missing 'command' argument in bash tool call."
        if tool_name == "web_search" and (not isinstance(args, dict) or "query" not in args):
            error_msg += "Missing 'query' argument in web_search tool call."
        if tool_name == "download_repo" and (not isinstance(args, dict) or "repo_url" not in args):
            error_msg += "Missing 'repo_url' argument in download_repo tool call."
        if error_msg:
            raise _format_error(
                format_error_template,
                error=error_msg.strip(),
                actions=[],
                template_kwargs=template_kwargs,
            )

        action = dict(args)
        action["tool"] = tool_name
        action["tool_call_id"] = tool_call.id
        actions.append(action)
    return actions


def _format_error(format_error_template: str, *, error: str, actions: list, template_kwargs: dict) -> FormatError:
    return FormatError(
        {
            "role": "user",
            "content": Template(format_error_template, undefined=StrictUndefined).render(
                error=error,
                actions=actions,
                **template_kwargs,
            ),
            "extra": {"interrupt_type": "FormatError"},
        }
    )

