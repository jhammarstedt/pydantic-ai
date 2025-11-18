"""AG-UI adapter for handling requests."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
)

from ... import ExternalToolset, ToolDefinition
from ...messages import (
    BaseToolCallPart,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from ...output import OutputDataT
from ...tools import AgentDepsT
from ...toolsets import AbstractToolset

try:
    from ag_ui.core import (
        AssistantMessage,
        BaseEvent,
        DeveloperMessage,
        FunctionCall,
        Message,
        RunAgentInput,
        SystemMessage,
        Tool as AGUITool,
        ToolCall,
        ToolMessage,
        UserMessage,
    )

    from .. import MessagesBuilder, UIAdapter, UIEventStream
    from ._event_stream import BUILTIN_TOOL_CALL_ID_PREFIX, AGUIEventStream
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'Please install the `ag-ui-protocol` package to use AG-UI integration, '
        'you can use the `ag-ui` optional group — `pip install "pydantic-ai-slim[ag-ui]"`'
    ) from e

if TYPE_CHECKING:
    pass

__all__ = ['AGUIAdapter']


# Frontend toolset


class _AGUIFrontendToolset(ExternalToolset[AgentDepsT]):
    """Toolset for AG-UI frontend tools."""

    def __init__(self, tools: list[AGUITool]):
        """Initialize the toolset with AG-UI tools.

        Args:
            tools: List of AG-UI tool definitions.
        """
        super().__init__(
            [
                ToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters_json_schema=tool.parameters,
                )
                for tool in tools
            ]
        )

    @property
    def label(self) -> str:
        """Return the label for this toolset."""
        return 'the AG-UI frontend tools'  # pragma: no cover


class AGUIAdapter(UIAdapter[RunAgentInput, Message, BaseEvent, AgentDepsT, OutputDataT]):
    """UI adapter for the Agent-User Interaction (AG-UI) protocol."""

    @classmethod
    def build_run_input(cls, body: bytes) -> RunAgentInput:
        """Build an AG-UI run input object from the request body."""
        return RunAgentInput.model_validate_json(body)

    def build_event_stream(self) -> UIEventStream[RunAgentInput, BaseEvent, AgentDepsT, OutputDataT]:
        """Build an AG-UI event stream transformer."""
        return AGUIEventStream(self.run_input, accept=self.accept)

    @cached_property
    def messages(self) -> list[ModelMessage]:
        """Pydantic AI messages from the AG-UI run input."""
        return self.load_messages(self.run_input.messages)

    @cached_property
    def toolset(self) -> AbstractToolset[AgentDepsT] | None:
        """Toolset representing frontend tools from the AG-UI run input."""
        if self.run_input.tools:
            return _AGUIFrontendToolset[AgentDepsT](self.run_input.tools)
        return None

    @cached_property
    def state(self) -> dict[str, Any] | None:
        """Frontend state from the AG-UI run input."""
        return self.run_input.state

    @classmethod
    def load_messages(cls, messages: Sequence[Message]) -> list[ModelMessage]:
        """Transform AG-UI messages into Pydantic AI messages."""
        builder = MessagesBuilder()
        tool_calls: dict[str, str] = {}  # Tool call ID to tool name mapping.

        for msg in messages:
            if isinstance(msg, UserMessage | SystemMessage | DeveloperMessage) or (
                isinstance(msg, ToolMessage) and not msg.tool_call_id.startswith(BUILTIN_TOOL_CALL_ID_PREFIX)
            ):
                if isinstance(msg, UserMessage):
                    builder.add(UserPromptPart(content=msg.content))
                elif isinstance(msg, SystemMessage | DeveloperMessage):
                    builder.add(SystemPromptPart(content=msg.content))
                else:
                    tool_call_id = msg.tool_call_id
                    tool_name = tool_calls.get(tool_call_id)
                    if tool_name is None:  # pragma: no cover
                        raise ValueError(f'Tool call with ID {tool_call_id} not found in the history.')

                    builder.add(
                        ToolReturnPart(
                            tool_name=tool_name,
                            content=msg.content,
                            tool_call_id=tool_call_id,
                        )
                    )

            elif isinstance(msg, AssistantMessage) or (  # pragma: no branch
                isinstance(msg, ToolMessage) and msg.tool_call_id.startswith(BUILTIN_TOOL_CALL_ID_PREFIX)
            ):
                if isinstance(msg, AssistantMessage):
                    if msg.content:
                        builder.add(TextPart(content=msg.content))

                    if msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            tool_call_id = tool_call.id
                            tool_name = tool_call.function.name
                            tool_calls[tool_call_id] = tool_name

                            if tool_call_id.startswith(BUILTIN_TOOL_CALL_ID_PREFIX):
                                _, provider_name, tool_call_id = tool_call_id.split('|', 2)
                                builder.add(
                                    BuiltinToolCallPart(
                                        tool_name=tool_name,
                                        args=tool_call.function.arguments,
                                        tool_call_id=tool_call_id,
                                        provider_name=provider_name,
                                    )
                                )
                            else:
                                builder.add(
                                    ToolCallPart(
                                        tool_name=tool_name,
                                        tool_call_id=tool_call_id,
                                        args=tool_call.function.arguments,
                                    )
                                )
                else:
                    tool_call_id = msg.tool_call_id
                    tool_name = tool_calls.get(tool_call_id)
                    if tool_name is None:  # pragma: no cover
                        raise ValueError(f'Tool call with ID {tool_call_id} not found in the history.')
                    _, provider_name, tool_call_id = tool_call_id.split('|', 2)

                    builder.add(
                        BuiltinToolReturnPart(
                            tool_name=tool_name,
                            content=msg.content,
                            tool_call_id=tool_call_id,
                            provider_name=provider_name,
                        )
                    )

        return builder.messages

    @classmethod
    def dump_messages(cls, messages: Sequence[ModelMessage]) -> list[Message]:
        """Transform Pydantic AI messages into AG-UI messages.

        This is the reverse operation of [`load_messages`][pydantic_ai.ui.ag_ui.AGUIAdapter.load_messages].

        Args:
            messages: Sequence of Pydantic AI ModelMessage objects (ModelRequest or ModelResponse).

        Returns:
            List of AG-UI Message objects.

        Example:
            ```python
            from pydantic_ai.messages import ModelRequest, UserPromptPart
            from pydantic_ai.ui.ag_ui import AGUIAdapter

            messages = [ModelRequest(parts=[UserPromptPart(content='Hello!')])]
            ag_ui_messages = AGUIAdapter.dump_messages(messages)
            ```

        Notes:
            - `ModelRequest` parts (UserPromptPart, SystemPromptPart, ToolReturnPart, RetryPromptPart)
              become separate AG-UI messages.
            - `ModelResponse` parts (TextPart, ToolCallPart, BuiltinToolCallPart) are combined
              into a single AssistantMessage.
            - `BuiltinToolReturnPart` becomes a separate ToolMessage with prefixed ID.
            - `ThinkingPart` is skipped as it's not part of the conversational message history.
        """
        result: list[Message] = []

        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    converted = _convert_request_part(part)
                    if converted:
                        result.append(converted)

            elif isinstance(message, ModelResponse):
                assistant_messages, builtin_returns = _convert_response_parts(message.parts)
                result.extend(assistant_messages)

                # Create separate ToolMessages for builtin tool returns
                for builtin_return in builtin_returns:
                    prefixed_id = _get_builtin_tool_call_id(
                        builtin_return.tool_call_id, builtin_return.provider_name or ''
                    )
                    result.append(
                        ToolMessage(
                            id=str(uuid.uuid4()),
                            tool_call_id=prefixed_id,
                            content=builtin_return.model_response_str(),
                        )
                    )

        return result


def _convert_request_part(part: ModelRequestPart) -> Message | None:
    """Convert a ModelRequest part to an AG-UI message.

    Args:
        part: A part from a ModelRequest.

    Returns:
        An AG-UI Message object, or None if the part should be skipped.
    """
    match part:
        case UserPromptPart():
            return UserMessage(
                id=str(uuid.uuid4()),
                content=part.content if isinstance(part.content, str) else str(part.content),
            )
        case SystemPromptPart():
            return SystemMessage(
                id=str(uuid.uuid4()),
                content=part.content if isinstance(part.content, str) else str(part.content),
            )
        case ToolReturnPart():
            return ToolMessage(
                id=str(uuid.uuid4()),
                tool_call_id=part.tool_call_id,
                content=part.model_response_str(),
            )
        case RetryPromptPart():
            if part.tool_call_id:
                return ToolMessage(
                    id=str(uuid.uuid4()),
                    tool_call_id=part.tool_call_id,
                    content=part.model_response(),
                )
            else:
                return UserMessage(
                    id=str(uuid.uuid4()),
                    content=part.model_response(),
                )
        case _:  # pragma: no cover
            return None


def _convert_response_parts(parts: Sequence[ModelResponsePart]) -> tuple[list[Message], list[BuiltinToolReturnPart]]:
    """Convert ModelResponse parts to AG-UI messages and collect builtin returns.

    Args:
        parts: Sequence of parts from a ModelResponse.

    Returns:
        A tuple of (list of AG-UI messages, list of builtin tool return parts).
    """
    content_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    builtin_returns: list[BuiltinToolReturnPart] = []
    last_was_text = False

    for part in parts:
        if isinstance(part, TextPart):
            content_parts.append(part.content)
            last_was_text = True
        elif isinstance(part, BaseToolCallPart):
            tool_call_id = part.tool_call_id
            if isinstance(part, BuiltinToolCallPart):
                # Text parts that are interrupted by a built-in tool call should not be joined together directly
                if last_was_text:
                    content_parts.append('\n\n')
                    last_was_text = False
                tool_call_id = _get_builtin_tool_call_id(tool_call_id, part.provider_name or '')
            tool_calls.append(
                ToolCall(
                    id=tool_call_id,
                    function=FunctionCall(
                        name=part.tool_name,
                        arguments=part.args_as_json_str(),
                    ),
                )
            )
        elif isinstance(part, BuiltinToolReturnPart):
            builtin_returns.append(part)
            # Built-in tool returns also interrupt text flow
            last_was_text = False
        elif isinstance(part, ThinkingPart):
            # ThinkingPart is not currently supported in AssistantMessage format
            # It's handled separately in the streaming events
            continue

    messages: list[Message] = []
    if content_parts or tool_calls:
        messages.append(
            AssistantMessage(
                id=str(uuid.uuid4()),
                content=''.join(content_parts) if content_parts else None,
                tool_calls=tool_calls if tool_calls else None,
            )
        )

    return messages, builtin_returns


def _get_builtin_tool_call_id(tool_call_id: str, provider_name: str) -> str:
    """Generate a prefixed tool call ID for builtin tools.

    Args:
        tool_call_id: The original tool call ID.
        provider_name: The name of the provider (e.g., 'function', 'openai').

    Returns:
        The prefixed tool call ID in the format 'pyd_ai_builtin|{provider_name}|{tool_call_id}'.
    """
    return f'{BUILTIN_TOOL_CALL_ID_PREFIX}|{provider_name}|{tool_call_id}'
