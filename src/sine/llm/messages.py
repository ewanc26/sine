"""Provider-neutral conversation messages.

Sine models the smallest message vocabulary that every researched API family can
express: a system instruction plus alternating user and assistant turns. Providers
that place the system prompt outside the message list, or that use ``model`` in
place of ``assistant`` for the model's own turns, do that translation inside their
adapter.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from sine.models.base import HistoryModel


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(HistoryModel):
    """A single conversational turn."""

    role: MessageRole
    content: str = Field(min_length=1)

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role=MessageRole.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role=MessageRole.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> Message:
        return cls(role=MessageRole.ASSISTANT, content=content)


def split_system(messages: tuple[Message, ...]) -> tuple[str, tuple[Message, ...]]:
    """Separate the system prompt from the conversation.

    System instructions are collected from anywhere in the list, joined in
    order, and removed from the conversation. This keeps the core free to compose
    prompts in a natural order while adapters that only accept a leading system
    turn, or a single top-level system field, receive a shape they accept.
    """

    system_parts = [m.content for m in messages if m.role is MessageRole.SYSTEM]
    conversation = tuple(m for m in messages if m.role is not MessageRole.SYSTEM)
    return "\n\n".join(system_parts), conversation


def collapse_consecutive(messages: tuple[Message, ...]) -> tuple[Message, ...]:
    """Merge adjacent turns that share a role.

    Several APIs reject two consecutive turns with the same role. This keeps the
    core free to build prompts freely without every adapter re-implementing the
    rule.
    """

    collapsed: list[Message] = []
    for message in messages:
        if collapsed and collapsed[-1].role is message.role:
            previous = collapsed[-1]
            collapsed[-1] = previous.model_copy(
                update={"content": f"{previous.content}\n\n{message.content}"}
            )
        else:
            collapsed.append(message)
    return tuple(collapsed)
