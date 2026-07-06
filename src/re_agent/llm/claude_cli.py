"""Claude Code CLI-backed LLM provider.

Drives the local ``claude`` CLI in print mode (``claude -p``), authenticating
via the user's existing Claude Code subscription instead of an Anthropic API
key. Modeled on :class:`re_agent.llm.codex_cli.CodexCLIProvider`.

The whole conversation (system prompt included) is flattened into one text
blob and piped via **stdin**, so no large or special-character content ever
touches the command line — this sidesteps Windows ``cmd.exe`` quoting issues
when the ``claude`` entry point is a ``.cmd`` shim.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from re_agent.llm.protocol import Message

# Agentic tools are disabled so print mode behaves as a pure completion engine
# (the reverser only needs text out, never file/shell access).
_DISALLOWED_TOOLS = [
    "Bash", "Edit", "Write", "Read", "Glob", "Grep",
    "WebFetch", "WebSearch", "Task", "NotebookEdit",
]


class ClaudeCLIProvider:
    """LLM provider backed by the local ``claude`` CLI (subscription auth)."""

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        timeout_s: int = 1800,
        claude_bin: str = "claude",
    ) -> None:
        self._model = model
        self._timeout_s = timeout_s
        self._claude_bin = claude_bin
        self._conversations: dict[str, list[Message]] = {}

    def send(self, messages: list[Message], **kwargs: Any) -> str:
        model = kwargs.get("model", self._model)

        # The system prompt must go through the REAL --system-prompt channel:
        # Claude Code treats inline "[SYSTEM]" role markers as untrusted text
        # and refuses to obey them. A temp file avoids command-line quoting of
        # large / special-character system prompts on Windows.
        system_text = "\n\n".join(
            m.content.strip() for m in messages if m.role == "system"
        ).strip()
        convo = [m for m in messages if m.role != "system"]
        prompt = self._render_messages(convo)

        sys_path: Path | None = None
        if system_text:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", suffix=".txt", delete=False
            ) as tf:
                tf.write(system_text)
                sys_path = Path(tf.name)

        try:
            argv = self._build_argv(str(model), sys_path)
            proc = subprocess.run(
                argv,
                input=prompt,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                timeout=self._timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"claude CLI timed out after {self._timeout_s}s") from exc
        except FileNotFoundError as exc:
            raise RuntimeError(f"claude CLI not found: {self._claude_bin}") from exc
        finally:
            if sys_path is not None:
                sys_path.unlink(missing_ok=True)

        if proc.returncode != 0:
            raise RuntimeError(
                f"claude CLI failed with exit code {proc.returncode}\n"
                f"{(proc.stderr or proc.stdout or '').strip()}"
            )
        return (proc.stdout or "").strip()

    # -- helpers --------------------------------------------------------------

    def _build_argv(self, model: str, sys_path: Path | None) -> list[str]:
        """Resolve the claude entry point and assemble the print-mode argv.

        On Windows the entry point is typically ``claude.cmd``; resolve it via
        ``shutil.which`` so subprocess can launch it, and route through
        ``cmd /c`` when needed. Only fixed, safe tokens appear here — never
        prompt content — so command-line quoting is a non-issue.
        """
        resolved = shutil.which(self._claude_bin) or self._claude_bin
        flags = [
            "-p",
            "--model", model,
            "--output-format", "text",
        ]
        if sys_path is not None:
            flags += ["--system-prompt-file", str(sys_path)]
        flags += ["--disallowedTools", *_DISALLOWED_TOOLS]
        if os.name == "nt" and resolved.lower().endswith((".cmd", ".bat")):
            return ["cmd", "/c", resolved, *flags]
        return [resolved, *flags]

    @property
    def supports_conversations(self) -> bool:
        return True

    def new_conversation(self, system: str) -> str:
        cid = uuid.uuid4().hex
        self._conversations[cid] = [Message(role="system", content=system)]
        return cid

    def resume(self, conversation_id: str, message: str) -> str:
        history = self._conversations.get(conversation_id)
        if history is None:
            raise KeyError(f"Unknown conversation ID: {conversation_id}")

        history.append(Message(role="user", content=message))
        response_text = self.send(list(history))
        history.append(Message(role="assistant", content=response_text))
        return response_text

    @staticmethod
    def _render_messages(messages: list[Message]) -> str:
        parts: list[str] = []
        for msg in messages:
            role = msg.role.upper()
            parts.append(f"[{role}]\n{msg.content.strip()}")
        return "\n\n".join(parts).strip()
