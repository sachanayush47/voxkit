"""Streaming a LangGraph agent's spoken reply."""

from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph


async def stream_agent_text(agent: CompiledStateGraph, text: str, thread_id: str) -> AsyncIterator[str]:
    """Run one agent turn and yield the text of its reply as it streams.

    Uses ``stream_mode="messages"``, which also emits tool results
    (``ToolMessage``) and other non-reply messages; only the assistant's own
    text (:class:`~langchain_core.messages.AIMessage` content) is yielded, so
    tool output is never read aloud. Text is read via ``.text``, which handles
    both plain-string and content-block message formats.

    Args:
        agent: A compiled LangGraph graph taking ``{"messages": [...]}`` input.
        text: The user's utterance for this turn.
        thread_id: Passed as ``configurable.thread_id`` so checkpointed
            conversation memory persists across turns.

    Yields:
        Successive chunks of the assistant's reply text.
    """
    async for message, _metadata in agent.astream(
        {"messages": [("user", text)]},
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="messages",
    ):
        if isinstance(message, AIMessage) and (chunk := message.text):
            yield chunk
