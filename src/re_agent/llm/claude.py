"""Claude (Anthropic) LLM provider implementation."""
from __future__ import annotations

import uuid
from typing import Any

import anthropic

from re_agent.llm.protocol import Message


class ClaudeProvider:
    """LLM provider backed by the Anthropic Claude API.

    Implements :class:`LLMProvider` using the ``anthropic`` Python SDK.

    Args:
        api_key: Anthropic API key.  If ``None``, the SDK falls back to the
            ``ANTHROPIC_API_KEY`` environment variable.
        model: Model identifier (e.g. ``"claude-sonnet-4-5-20250929"``).
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature (``0.0`` = deterministic).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-opus-4-8",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._conversations: dict[str, list[Message]] = {}

    @staticmethod
    def _is_adaptive_thinking_model(model: str) -> bool:
        """Models that reject temperature/top_p/budget_tokens and use adaptive
        thinking + the effort parameter (Fable 5, Opus 4.6+, Sonnet 4.6).

        For these, we must NOT send `temperature` (400) and instead steer with
        `thinking={"type": "adaptive"}` + `output_config={"effort": ...}`.
        """
        m = model.lower()
        modern = ("claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6",
                  "claude-sonnet-4-6", "claude-fable-5")
        return any(m.startswith(p) for p in modern)

    # -- LLMProvider interface ------------------------------------------------

    def send(self, messages: list[Message], **kwargs: Any) -> str:
        """Send messages to Claude and return the assistant response text."""
        system_text: str | None = None
        api_messages: list[dict[str, str]] = []

        for msg in messages:
            if msg.role == "system":
                system_text = msg.content
            else:
                api_messages.append({"role": msg.role, "content": msg.content})

        model = kwargs.get("model", self._model)
        create_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
            "messages": api_messages,
        }
        if self._is_adaptive_thinking_model(model):
            # Opus 4.8 / 4.7 / 4.6, Sonnet 4.6, Fable 5: sampling params 400.
            # Steer with adaptive thinking + high effort for reversing accuracy.
            create_kwargs["thinking"] = {"type": "adaptive"}
            create_kwargs["output_config"] = {"effort": "high"}
        else:
            create_kwargs["temperature"] = kwargs.get("temperature", self._temperature)
        if system_text is not None:
            create_kwargs["system"] = system_text

        response = self._client.messages.create(**create_kwargs)

        # Extract text from content blocks.
        parts: list[str] = []
        for block in response.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts)

    @property
    def supports_conversations(self) -> bool:
        """Claude supports multi-turn conversations (client-side history)."""
        return True

    def new_conversation(self, system: str) -> str:
        """Create a new conversation with a system prompt, returning its ID."""
        cid = uuid.uuid4().hex
        self._conversations[cid] = [Message(role="system", content=system)]
        return cid

    def resume(self, conversation_id: str, message: str) -> str:
        """Append a user message to the conversation and return the response."""
        history = self._conversations.get(conversation_id)
        if history is None:
            raise KeyError(f"Unknown conversation ID: {conversation_id}")

        history.append(Message(role="user", content=message))
        response_text = self.send(list(history))
        history.append(Message(role="assistant", content=response_text))
        return response_text
