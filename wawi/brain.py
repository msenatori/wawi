"""The brain: a bounded tool-use loop over :mod:`wawi.tools`.

Wawi is a voice interface to a model: you ask out loud, it answers out loud
and in writing. A turn is one bounded tool-use loop ending in one short spoken
answer.

Three constraints shape this module:

1. **Cost.** Tool schemas and the system prompt are re-sent on every turn, so the
   SDK still runs lean: no CLAUDE.md, no settings.json, no user/project MCP
   servers, no built-in tool schemas. The ONLY thing added to context is the
   three terse schemas from :mod:`wawi.tools` plus one optional screen tool.
2. **Voice.** The final text goes straight to TTS, so it is sanitized here:
   markdown, code fences, emoji and URLs never reach the speaker.
3. **Blast radius.** Nothing the model can call changes code or runs a
   command; see :mod:`wawi.tools`. A mistranscription costs a screenshot
   at worst, which is what makes an always-listening assistant tolerable.

Backends:

* ``claude`` — claude-agent-sdk with subscription auth, full tool loop.
* ``ollama`` — local HTTP, TEXT ONLY. It degrades loudly: the spoken answer
  says outright that it is running on a local model with no tools.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from wawi import lang, tools, working

if TYPE_CHECKING:
    from wawi.connectors import Connectors

log = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 5
"""Hard cap on tool executions per turn. A confused model cannot spin forever."""

MCP_SERVER_NAME = "wawi"
"""SDK MCP server name. Tools are exposed to the model as ``mcp__wawi__<tool>``."""

SCREEN_TOOL = "see_screen"
#: What the mascot does while she looks at the screen (wawi/working.py).
SCREEN_ANIMATION = "camera"
"""Optional, expensive tool. The default path carries NO image."""


class BrainError(Exception):
    """Raised when the brain call signals is_error=True or another fatal condition."""


class BrainBudgetError(BrainError):
    """The turn died having spent its SDK turn budget without saying anything.

    Split from :class:`BrainError` for one reason: the spoken fallback. Every
    other brain failure is a connection problem and "I could not connect" is
    true of it. This one ran fine and simply never got to an answer, so saying
    the same sentence sends the user to check their network over a model that
    looped. See :data:`wawi.lang.BUDGET_TEXT`.
    """


# --------------------------------------------------------------------------- #
# Spanish phrasing (spoken output only)
# --------------------------------------------------------------------------- #

#: Kept as a module constant because it is part of this module's tested
#: surface; the text itself now lives with every other language-dependent
#: string in :mod:`wawi.lang`.
_ES_DEGRADED = lang.DEGRADED_SUFFIX["es"]

# Model-facing text (never spoken) stays English, like the rest of the code.
_TOOL_BUDGET_EXHAUSTED = (
    "Tool budget exhausted for this turn. Answer now, in Spanish, "
    "with what you already know."
)

# A second look in the same turn would only add a second image: nothing the
# model can call moves a window, so the screen it would see is the one it
# already has above. Answered with the note instead of another ~1,400 tokens.
_SCREEN_ALREADY_SEEN = (
    "Already looked at the screen this turn {note}; nothing you can do changes "
    "it, so use that image."
)


def screen_notes_preamble(notes: tuple[str, ...] | list[str]) -> str:
    """Render the captures of an earlier turn as text, or "" when there were none.

    This is what an old screenshot becomes once it stops travelling: one line
    per capture, so "eso que viste" still has something to point at, and an
    explicit hint that looking again is the way to see it. An image the user
    dropped on her (:class:`ImageAttachment`) travels the same way.
    """
    if not notes:
        return ""
    return (
        f"(Earlier you looked at these images: {', '.join(notes)}. "
        "They are no longer attached; call see_screen again if the "
        "question needs the screen.)\n\n"
    )


@dataclass(frozen=True)
class ImageAttachment:
    """An image put in front of the model IN the user turn (something the
    owner dropped on her, wawi/drops.py), not through a tool.

    Costs what a screenshot costs, so it is treated like one: its *note* joins
    :attr:`BrainResult.screen_notes`, which keeps the session holding it from
    being resumed and rotates the live connection (see :class:`Engine`).
    """

    data: str  # base64
    media_type: str = "image/jpeg"
    note: str = ""


def _with_images(text: str, images: "tuple[ImageAttachment, ...] | list[ImageAttachment]"):
    """The user turn as content blocks (images first, then the text), as the
    async iterable the SDK takes instead of a plain string prompt."""
    content: list[dict[str, Any]] = [
        {"type": "image",
         "source": {"type": "base64", "media_type": a.media_type, "data": a.data}}
        for a in images
    ]
    content.append({"type": "text", "text": text})

    async def _messages():
        yield {"type": "user", "message": {"role": "user", "content": content},
               "parent_tool_use_id": None}

    return _messages()


@dataclass(frozen=True)
class ScreenOptions:
    """Capture settings for the optional :data:`SCREEN_TOOL`."""

    monitor: int = 1
    # Same values as wawi.screen.INLINE_*; see the reasoning there.
    max_edge: int = 1280
    jpeg_quality: int = 65


@dataclass
class BrainResult:
    """Outcome of one brain turn.

    Attributes:
        text: The sanitized, TTS-ready Spanish utterance.
        cost_usd: Turn cost, or None when the backend does not bill.
        input_tokens: input + cache_creation + cache_read, or None.
        output_tokens: Output tokens, or None.
        tool_calls: Names of the tools actually executed, in order.
        session_id: SDK session this turn ran in; pass it as ``resume`` next
            turn to continue the same conversation.
        degraded: True when this turn ran without tool access.
    """

    text: str
    cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    tool_calls: tuple[str, ...] = ()
    #: The SDK session this turn ran in. Handing it back on the NEXT turn is
    #: what turns a series of one-shot questions into one conversation — and
    #: what lets the server keep (and cache) the history instead of us
    #: re-sending a rendered summary of it every time.
    session_id: str | None = None
    degraded: bool = False
    #: One line per screenshot this turn put in front of the model, e.g.
    #: ``[captura 14:50:08, 1280x800, monitor 1]``. Non-empty means the
    #: session now holds an image, and resuming it would re-send that image on
    #: every later turn — so the caller starts the next turn fresh and carries
    #: these lines instead (see :func:`screen_notes_preamble`).
    screen_notes: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Speech sanitization — the last gate before TTS
# --------------------------------------------------------------------------- #

_RE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_RE_FENCE_OPEN = re.compile(r"```.*", re.DOTALL)
_RE_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_RE_URL = re.compile(r"(?:https?://|www\.)\S+")
_RE_BULLET = re.compile(r"^\s*(?:[-*•+]|\d+[.)])\s+", re.MULTILINE)
_RE_HEADING = re.compile(r"^\s*#{1,6}\s*", re.MULTILINE)
_RE_MD_CHARS = re.compile(r"[*`~#>|]+")
_RE_UNDERSCORE = re.compile(r"_+")
_RE_EMOJI = re.compile(
    "["
    "\U0001f000-\U0001ffff"
    "\U00002190-\U000021ff"
    "\U00002300-\U000023ff"
    "\U000024c2-\U0001f251"
    "\U00002600-\U000027bf"
    "\U0000fe00-\U0000fe0f"
    "\U0001f900-\U0001f9ff"
    "]+",
    flags=re.UNICODE,
)
_RE_WS = re.compile(r"\s+")

_ES_LINK = "el enlace"


def sanitize_for_speech(text: str) -> str:
    """Return *text* reduced to something a TTS voice can read aloud.

    Removes fenced code blocks entirely (their content is unspeakable), turns
    markdown links into their label, replaces URLs with a spoken placeholder,
    drops bullets, headings, emphasis characters and emoji, and collapses all
    whitespace so the voice never pauses on a stray newline.
    """
    if not text:
        return ""

    out = _RE_FENCE.sub(" ", text)
    out = _RE_FENCE_OPEN.sub(" ", out)  # unterminated fence
    out = _RE_MD_LINK.sub(r"\1", out)
    out = _RE_URL.sub(_ES_LINK, out)
    out = _RE_BULLET.sub("", out)
    out = _RE_HEADING.sub("", out)
    out = _RE_MD_CHARS.sub("", out)
    # A space, not nothing: "auth_test" must stay two spoken words.
    out = _RE_UNDERSCORE.sub(" ", out)
    out = _RE_EMOJI.sub("", out)
    out = _RE_WS.sub(" ", out)
    return out.strip()


# --------------------------------------------------------------------------- #
# Streaming — hand finished sentences out while the answer is still growing
# --------------------------------------------------------------------------- #


class _SentenceEmitter:
    """Turn a growing answer into finished sentences, one callback each.

    This is the whole point of streaming: a 19-second answer with no tool
    calls used to be 19 seconds of silence because the text was joined only
    after the last message arrived. Emitting per sentence means the voice
    starts on the first one.

    Two rules keep what is spoken honest:

    * Only sentences the splitter calls FINISHED are emitted. Kokoro handed
      half a sentence reads it as one, pause and all.
    * Text inside an unterminated code fence is held back. ``sanitize_for_speech``
      deletes an open fence and everything after it, so emitting mid-fence
      would speak a truncation of a block that the finished answer drops
      entirely — the two would disagree about what was said.
    """

    def __init__(self, on_sentence: Callable[[str], None]) -> None:
        self._on_sentence = on_sentence
        self._buffer = ""
        self.emitted = 0

    def feed(self, text: str) -> None:
        """Append streamed text (already contiguous — no separator inserted)."""
        if not text:
            return
        self._buffer += text
        self._drain()

    def feed_block(self, text: str) -> None:
        """Append a whole text block, joined the way :func:`_collect` joins them.

        The separator matters: the recorded answer is ``" ".join(parts)``, so
        feeding blocks back to back without it would make the spoken text and
        the remembered text disagree at every block seam.
        """
        if not text:
            return
        self._buffer = f"{self._buffer} {text}" if self._buffer else text
        self._drain()

    def flush(self) -> None:
        """Emit whatever is left, finished or not.

        Called in a ``finally``, so an answer cut short by an SDK error still
        reaches the speaker instead of dying as a half-spoken sentence.
        """
        remainder, self._buffer = self._buffer, ""
        self._emit(remainder)

    def _drain(self) -> None:
        from wawi.voice import take_complete_sentences

        # Closed code blocks are dropped from the buffer here, not left for
        # per-sentence sanitizing: a block spans sentence boundaries, so by the
        # time it is split up the fence markers sit in different sentences and
        # each half reads as prose. The finished answer deletes the block
        # outright, and so must this.
        self._buffer = _RE_FENCE.sub(" ", self._buffer)

        # An odd number of fences means one is still open. Only the text from
        # that fence onwards has to wait — the sentences BEFORE it are finished
        # and holding them back would make an answer that shows code go quiet
        # exactly when it is explaining itself.
        head, held = self._buffer, ""
        if self._buffer.count("```") % 2:
            opened = self._buffer.rfind("```")
            head, held = self._buffer[:opened], self._buffer[opened:]
        finished, remainder = take_complete_sentences(head)
        self._buffer = remainder + held
        for sentence in finished:
            self._emit(sentence)

    def _emit(self, raw: str) -> None:
        spoken = sanitize_for_speech(raw)
        if not spoken:
            return
        self.emitted += 1
        try:
            self._on_sentence(spoken)
        except Exception:
            # A broken consumer must not kill a turn that is answering fine:
            # the answer still returns, it is only spoken late.
            log.warning("brain: streaming consumer raised; dropping it", exc_info=True)
            self._on_sentence = lambda _text: None


def _stream_text_delta(event: dict[str, Any]) -> str:
    """Pull the text out of one raw Anthropic stream event, or "".

    Only ``content_block_delta`` / ``text_delta`` carries answer text. Every
    other event shape (tool input deltas, block starts, pings) is ignored by
    returning "" rather than by enumerating them, so a new event type added
    upstream cannot break a turn.
    """
    if event.get("type") != "content_block_delta":
        return ""
    delta = event.get("delta") or {}
    if delta.get("type") != "text_delta":
        return ""
    return str(delta.get("text") or "")


# --------------------------------------------------------------------------- #
# Per-turn state
# --------------------------------------------------------------------------- #


@dataclass
class _TurnState:
    """Mutable bookkeeping for a single turn: the iteration cap and the gate."""

    max_iterations: int = MAX_TOOL_ITERATIONS
    calls: list[str] = field(default_factory=list)
    #: Capture notes, one per image actually returned this turn.
    shots: list[str] = field(default_factory=list)

    def note(self, name: str) -> bool:
        """Record an attempted tool call. False when the budget is spent."""
        if len(self.calls) >= self.max_iterations:
            log.warning(
                "brain: tool budget of %d exhausted, refusing %r",
                self.max_iterations,
                name,
            )
            return False
        self.calls.append(name)
        log.info("brain: tool call %d/%d — %s", len(self.calls), self.max_iterations, name)
        return True


# --------------------------------------------------------------------------- #
# Tool bridge — model tool call in, MCP content out
# --------------------------------------------------------------------------- #


def _content(text: str, *, is_error: bool = False) -> dict[str, Any]:
    """Wrap *text* in the MCP tool-result shape the SDK expects."""
    return {"content": [{"type": "text", "text": text}], "is_error": is_error}


def _run_tool(name: str, args: dict | None, state: _TurnState) -> dict[str, Any]:
    """Execute one model tool call against :func:`wawi.tools.dispatch`.

    Enforces two invariants the model cannot argue with:

    * the per-turn iteration cap, and
    """
    payload = dict(args or {})

    if not state.note(name):
        return _content(_TOOL_BUDGET_EXHAUSTED)

    # The mascot shows what she is doing while the tool runs (the tool's own
    # "animation"; wawi/working.py). Tools that animate themselves (the pet,
    # the games) declare none on purpose and show nothing extra.
    if tools.has_no_animation(name):
        result = tools.dispatch(name, payload)
    else:
        with working.doing(name, tools.animation_for(name)):
            result = tools.dispatch(name, payload)


    return _content(
        json.dumps(
            {"ok": result.ok, "speak": result.speak, "data": result.data},
            ensure_ascii=False,
        )
    )


def _run_screen_tool(
    state: _TurnState, opts: ScreenOptions, monitor: int | None = None
) -> dict[str, Any]:
    """Capture the screen and hand it back as an image block plus a note.

    Kept out of the default path on purpose: an image costs about 1,400 input
    tokens at the default 1280px (width x height / 750), and almost no spoken
    question needs one. Only ONE image per turn: a repeat call gets the note
    of the capture it already has, never a second picture.
    """
    if not state.note(SCREEN_TOOL):
        return _content(_TOOL_BUDGET_EXHAUSTED)
    if state.shots:
        return _content(_SCREEN_ALREADY_SEEN.format(note=state.shots[-1]))

    selected_monitor = opts.monitor if monitor is None else monitor
    try:
        from datetime import datetime

        from wawi import screen

        with working.doing(SCREEN_TOOL, SCREEN_ANIMATION):
            img = screen.capture(selected_monitor)
            image_b64, (width, height) = screen.encode_jpeg(
                img, max_edge=opts.max_edge, jpeg_quality=opts.jpeg_quality
            )
    except Exception as exc:  # capture can fail on permissions, headless, etc.
        log.warning("brain: screen capture failed: %s", exc)
        return _content(f"Screen capture failed: {exc}", is_error=True)

    note = (
        f"[captura {datetime.now():%H:%M:%S}, {width}x{height}, "
        f"monitor {selected_monitor}]"
    )
    state.shots.append(note)
    log.info("brain: screen capture %s", note)
    return {
        "content": [
            {"type": "image", "data": image_b64, "mimeType": "image/jpeg"},
            {"type": "text", "text": note},
        ],
        "is_error": False,
    }


@dataclass
class _Bridge:
    """The tools handed to the SDK, plus the state they write into."""

    state: _TurnState
    handlers: dict[str, Callable[[dict], Any]]
    sdk_tools: list[Any]
    tool_names: list[str]


def _build_bridge(state: _TurnState, screen_options: ScreenOptions | None) -> _Bridge:
    """Wrap every ACTIVE schema from :mod:`wawi.tools` as an SDK MCP tool.

    Only these tools are added to the model's context, and every one of their
    descriptions is re-sent — and re-paid for — on every single turn, which is
    why the set is small. The screen tool is included only when the caller
    supplied capture settings.
    """
    from claude_agent_sdk import tool as sdk_tool  # type: ignore[import]

    sdk_tools: list[Any] = []
    handlers: dict[str, Callable[[dict], Any]] = {}

    def _register(name: str, description: str, schema: dict, fn) -> None:
        handlers[f"mcp__{MCP_SERVER_NAME}__{name}"] = fn
        sdk_tools.append(sdk_tool(name, description, schema)(fn))

    from wawi import plugins

    for spec in tools.active_tools():
        name = str(spec["name"])

        if plugins.is_plugin_tool(name):
            # An agregado makes a network call (up to 10 s) and a write waits
            # for a click on screen: off the engine loop, so the SDK keeps
            # reading its transport meanwhile.
            async def _handler(args: dict, _name: str = name) -> dict[str, Any]:
                return await asyncio.to_thread(_run_tool, _name, args, state)
        else:
            async def _handler(args: dict, _name: str = name) -> dict[str, Any]:
                return _run_tool(_name, args, state)

        _register(name, str(spec["description"]), dict(spec["input_schema"]), _handler)

    if screen_options is not None:
        async def _screen(args: dict) -> dict[str, Any]:
            return _run_screen_tool(state, screen_options, args.get("monitor"))

        _register(
            SCREEN_TOOL,
            "Look at the user's screen. Expensive; only for visual questions. "
            "Choose monitor 1 for the primary display, 2 for the secondary, "
            "or 0 for all displays. Omit to use the configured default.",
            {
                "type": "object",
                "properties": {"monitor": {"type": "integer", "minimum": 0}},
            },
            _screen,
        )

    return _Bridge(
        state=state,
        handlers=handlers,
        sdk_tools=sdk_tools,
        tool_names=list(handlers.keys()),
    )


# --------------------------------------------------------------------------- #
# Claude backend
# --------------------------------------------------------------------------- #

#: How long the engine waits on each stage before giving up. Generous: the
#: fallback for a timeout is a whole turn on the slow path, so being patient
#: costs less than being wrong.
_CONNECT_TIMEOUT_S = 60.0
_TURN_TIMEOUT_S = 300.0
_DISCONNECT_TIMEOUT_S = 10.0

_LEAN_FIELDS: dict[str, object] = {
    "setting_sources": [],        # skip CLAUDE.md / settings.json
    "strict_mcp_config": True,    # ignore user/project/plugin MCP servers
    "tools": [],                  # zero BUILT-IN tool schemas (ours are MCP)
    "permission_mode": "dontAsk",  # never block waiting for a human approval
}
"""Fields probed one at a time so an SDK upgrade can only raise cost, not crash."""


def _build_env() -> dict[str, str]:
    """Return os.environ minus ANTHROPIC_API_KEY, plus CLAUDE_CODE_DISABLE_AUTO_MEMORY."""
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
    return env


def _build_options(
    bridge: _Bridge,
    *,
    model: str,
    max_iterations: int,
    system: str | None = None,
    bypass_permissions: bool = False,
    resume: str | None = None,
    partial_messages: bool = False,
    connectors: "Connectors | None" = None,
) -> Any:
    """Assemble ClaudeAgentOptions, dropping any field this SDK version rejects.

    *connectors* adds the servers the user enabled in config.yaml, their
    pre-approved tools and the hook that gates them. Without one enabled the
    options are exactly what they were before connectors existed.
    """
    from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server  # type: ignore[import]

    from wawi.prompts import SYSTEM_PROMPT

    server = create_sdk_mcp_server(MCP_SERVER_NAME, "1.0.0", bridge.sdk_tools)

    kwargs: dict[str, object] = {
        "model": model,
        "system_prompt": system or SYSTEM_PROMPT,
        # Two extra turns, not one: when the budget runs out the model spends a
        # turn on the refused call and STILL needs one to answer. With +1 that
        # last turn did not exist, and a turn that used every tool died with
        # error_max_turns instead of speaking.
        "max_turns": max_iterations + 2,
        "env": _build_env(),
        "mcp_servers": {MCP_SERVER_NAME: server},
        "allowed_tools": [f"mcp__{MCP_SERVER_NAME}", *bridge.tool_names],
    }

    if connectors:
        # strict_mcp_config stays on: these servers are named here one by
        # one, so it still keeps out every server the user did NOT enable.
        kwargs["mcp_servers"] = {**connectors.mcp_servers(), MCP_SERVER_NAME: server}
        kwargs["allowed_tools"] = [*kwargs["allowed_tools"], *connectors.allowed_tools()]
        kwargs["hooks"] = connectors.hooks()
        hidden = connectors.disallowed_tools()
        if hidden:
            kwargs["disallowed_tools"] = hidden

    if partial_messages:
        # Without this the SDK yields one AssistantMessage per assistant TURN,
        # so a tool-free answer arrives complete or not at all and streaming
        # buys exactly nothing — which is the shape of the 19-second silence
        # this exists to fix. Probed like the lean fields: an SDK without it
        # degrades to per-block streaming, not to a crash.
        try:
            ClaudeAgentOptions(**{**kwargs, "include_partial_messages": True})
            kwargs["include_partial_messages"] = True
        except TypeError:
            log.warning(
                "claude-agent-sdk: no 'include_partial_messages' field on this "
                "version; the answer streams per message, not per token"
            )

    for name, val in _LEAN_FIELDS.items():
        try:
            ClaudeAgentOptions(**{**kwargs, name: val})
            kwargs[name] = val
        except TypeError:
            log.warning(
                "claude-agent-sdk: field %r not supported on this version; "
                "dropping it (cost may rise)",
                name,
            )

    if bypass_permissions:
        # Deprecated, and deliberately a no-op here. This engine has no
        # built-in tools at all (tools=[] above), so there is nothing for a
        # bypass to unlock -- only a blanket that would also cover any tool a
        # future SDK started sending by default. `dangerous_skip_permissions`
        # now means a one-hour yolo on the EXECUTOR, gated in wawi/guards.py.
        log.debug("bypass_permissions ignored on the conversational engine")

    if resume:
        # Continue the SAME server-side conversation instead of starting a new
        # one. Two things follow, and the second is the reason it is worth
        # doing: the model sees the real history rather than our rendered
        # summary of it, and the history it already holds is cache-resident, so
        # the tokens are re-read at a fraction of the price instead of being
        # re-sent in full. A version without the field degrades to today's
        # behaviour — a fresh conversation — which is worse, not broken.
        try:
            ClaudeAgentOptions(**{**kwargs, "resume": resume})
            kwargs["resume"] = resume
        except TypeError:
            log.warning(
                "claude-agent-sdk: no 'resume' field on this version; every "
                "turn starts a new conversation"
            )

    return ClaudeAgentOptions(**kwargs)


async def _sdk_query(prompt: Any, options: Any):
    """Thin indirection over ``claude_agent_sdk.query`` (also the test seam)."""
    from claude_agent_sdk import query  # type: ignore[import]

    async for message in query(prompt=prompt, options=options):
        yield message


async def _collect(
    stream: Any, state: _TurnState, emitter: _SentenceEmitter | None = None,
    image_notes: tuple[str, ...] = (),
) -> BrainResult:
    """Read one turn off a message stream into a :class:`BrainResult`.

    Shared by the one-shot path and the persistent one on purpose: they differ
    in how the process is started, in nothing else. Two copies of this loop
    would drift, and the drift would show up as one path handling an error the
    other turns into "no pude conectarme".

    *emitter*, when given, is fed the answer as it arrives so the voice can
    start early. It changes WHEN text leaves this function, never WHAT: the
    :class:`BrainResult` is assembled from the same ``AssistantMessage`` blocks
    as before, so the conversation memory and the metrics see the whole answer
    exactly as they did.
    """
    from claude_agent_sdk import (  # type: ignore[import]
        AssistantMessage,
        ResultMessage,
    )
    from claude_agent_sdk.types import TextBlock  # type: ignore[import]

    try:
        from claude_agent_sdk.types import StreamEvent  # type: ignore[import]
    except ImportError:
        # An SDK too old to model partial messages never yields them either.
        # The empty tuple makes the isinstance below a constant False instead
        # of a second code path nobody exercises.
        StreamEvent = ()  # type: ignore[assignment,misc]

    text_parts: list[str] = []
    cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    session_id: str | None = None
    cache_read = 0
    # Token deltas and the assistant message that closes them carry the SAME
    # text. Whichever arrives first owns the emitter for the rest of the turn,
    # so nothing is ever spoken twice — and a CLI too old to send deltas still
    # streams, only per block instead of per token.
    streamed_deltas = False

    try:
        async for msg in stream:
            if emitter is not None and isinstance(msg, StreamEvent):
                chunk = _stream_text_delta(getattr(msg, "event", None) or {})
                if chunk:
                    streamed_deltas = True
                    emitter.feed(chunk)
            elif isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)
                        if emitter is not None and not streamed_deltas:
                            emitter.feed_block(block.text)
                if emitter is not None:
                    # The message closes its text blocks. Its final sentence
                    # no longer needs to wait for the SDK's result/usage event.
                    emitter.flush()
            elif isinstance(msg, ResultMessage):
                if msg.is_error:
                    # A model that spent its whole turn budget on tools still said
                    # things along the way. Raising here threw that away and turned
                    # a long-but-successful turn into "no pude conectarme", which
                    # is both a lie and the loudest failure the user ever sees.
                    if msg.subtype == "error_max_turns" and any(
                        p.strip() for p in text_parts
                    ):
                        log.warning(
                            "brain: hit the SDK turn cap after %d tool call(s); "
                            "speaking what it had said so far",
                            len(state.calls),
                        )
                    elif msg.subtype == "error_max_turns":
                        raise BrainBudgetError(
                            f"Brain call returned is_error=True  subtype={msg.subtype!r}"
                        )
                    else:
                        raise BrainError(
                            f"Brain call returned is_error=True  subtype={msg.subtype!r}"
                        )
                cost_usd = getattr(msg, "total_cost_usd", None)
                session_id = getattr(msg, "session_id", None)
                usage = getattr(msg, "usage", None)
                if usage:
                    tok_in = usage.get("input_tokens") or 0
                    tok_cache_create = usage.get("cache_creation_input_tokens") or 0
                    cache_read = usage.get("cache_read_input_tokens") or 0
                    input_tokens = tok_in + tok_cache_create + cache_read
                    output_tokens = usage.get("output_tokens")
    finally:
        # Even on the failure paths above: text the model already produced is
        # owed to the user out loud. Dying mid-sentence and going quiet is the
        # one outcome that reads as a crash from the other side of the room.
        if emitter is not None:
            emitter.flush()

    text = sanitize_for_speech(" ".join(p for p in text_parts if p))

    # cache_read is logged on purpose: it is the only way to notice that the
    # session stopped being reused. A conversation that silently starts over
    # still answers, just worse and dearer, which is the hardest kind of
    # regression to see from the outside.
    log.info(
        "brain turn done — tools=%s chars=%d cost=%.5f cached_in=%d session=%s",
        state.calls,
        len(text),
        cost_usd or 0.0,
        cache_read,
        (session_id or "-")[:8],
    )

    return BrainResult(
        text=text,
        cost_usd=cost_usd,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tool_calls=tuple(state.calls),
        session_id=session_id,
        screen_notes=tuple(image_notes) + tuple(state.shots),
    )

async def ask(
    user_text: str,
    *,
    model: str,
    screen_options: ScreenOptions | None = None,
    max_iterations: int = MAX_TOOL_ITERATIONS,
    assistant_name: str = "Wawi",
    history: str = "",
    preferences: str = "",
    issues_hint: str = "",
    recall: str = "",
    bypass_permissions: bool = False,
    language: str = "es",
    resume: str | None = None,
    on_sentence: Callable[[str], None] | None = None,
    connectors: "Connectors | None" = None,
    images: "tuple[ImageAttachment, ...]" = (),
) -> BrainResult:
    """Run one bounded tool-use turn and return a TTS-ready answer.

    Args:
        user_text: The transcribed question.
        model: Claude model ID, e.g. ``"claude-sonnet-5-5"``.
        screen_options: When given, the model may call the screen tool. Pass
            None to keep images out of the turn entirely.
        max_iterations: Tool executions allowed this turn.
        on_sentence: Called with each finished sentence as it is generated.
            It runs on the SDK's event loop, so it must not block — the stream
            it would block is the one still producing the answer.

    Returns:
        A :class:`BrainResult` whose ``text`` is safe to speak as-is.

    Raises:
        BrainError: If the SDK reports is_error=True.
    """
    from claude_agent_sdk import (  # type: ignore[import]
        AssistantMessage,
        ResultMessage,
    )
    from claude_agent_sdk.types import TextBlock  # type: ignore[import]

    from wawi.prompts import system_prompt

    state = _TurnState(max_iterations=max_iterations)
    bridge = _build_bridge(state, screen_options)
    options = _build_options(
        bridge,
        model=model,
        max_iterations=max_iterations,
        system=system_prompt(
            assistant_name,
            preferences,
            issues_hint,
            language=language,
            recall=recall,
        ),
        bypass_permissions=bypass_permissions,
        resume=resume,
        partial_messages=on_sentence is not None,
        connectors=connectors,
    )

    # With a session to resume, the server already holds the conversation, so
    # re-sending our own rendered copy of it would be paying twice to say the
    # same thing — and the two would disagree the moment the window slid.
    prompt = user_text if resume else (f"{history}{user_text}" if history else user_text)

    emitter = _SentenceEmitter(on_sentence) if on_sentence is not None else None
    sent = _with_images(prompt, images) if images else prompt
    return await _collect(_sdk_query(sent, options), state, emitter,
                          tuple(a.note for a in images))


# --------------------------------------------------------------------------- #
# Ollama backend — text only, and it says so
# --------------------------------------------------------------------------- #


def ask_ollama(
    user_text: str,
    *,
    model: str,
    url: str,
    assistant_name: str = "Wawi",
    history: str = "",
    language: str = "es",
) -> BrainResult:
    """Answer via a local Ollama instance. NO tools — degraded on purpose.

    Ollama's tool-calling support varies by model and its schema dialect does
    not match the SDK's, so rather than pretend, this path answers from the
    model's own knowledge and says so out loud. Silently dropping tool calls
    would let the answer look identical to a full one while being blind.
    """
    from wawi.prompts import system_prompt_no_tools

    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt_no_tools(assistant_name, language=language),
                },
                {"role": "user", "content": f"{history}{user_text}" if history else user_text},
            ],
            "stream": False,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise BrainError(f"Ollama HTTP {exc.code} from {url!r}: {body[:300]}") from exc
    except urllib.error.URLError as exc:
        raise BrainError(
            f"Cannot reach Ollama at {url!r}: {exc.reason}. "
            "Is Ollama running? Start it with:  ollama serve"
        ) from exc

    try:
        text = data["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise BrainError(f"Unexpected Ollama response shape: {str(data)[:200]}") from exc

    degraded = lang.get(lang.DEGRADED_SUFFIX, language)
    spoken = sanitize_for_speech(text)
    return BrainResult(
        text=f"{spoken} {degraded}".strip() if spoken else degraded,
        degraded=True,
    )


# --------------------------------------------------------------------------- #
# Persistent engine
# --------------------------------------------------------------------------- #


def _extras_signature(connectors: "Connectors | None") -> tuple:
    """What the connectors and the agregados bake into a connection: a
    change here (a plugin added from the menu or the terminal) reconnects
    once on the next turn. The system prompt is not part of it."""
    from wawi import plugins

    return (connectors.signature() if connectors else (), plugins.signature())


class Engine:
    """A connected ``ClaudeSDKClient`` kept alive across turns.

    WHY THIS EXISTS: the one-shot path spawns the 197 MB ``claude`` binary on
    every question. Measured here, that start-up is ~1.3 s — roughly half of a
    short turn — while the harness *inside* the turn costs about 12 ms. So the
    win is not in trimming the harness; it is in not launching it again.

    THREADING: an SDK client is bound to the event loop it was connected on,
    and the orchestrator calls in from a worker thread that has none. So the
    engine owns a private loop on its own daemon thread and every call is
    marshalled onto it. This is also why the client is never touched directly
    from outside: doing so from another loop fails in ways that look like a
    hung microphone.

    FAILURE: any error tears the client down and is re-raised. The caller
    (:func:`ask_sync`) then falls back to the one-shot path for that turn, so a
    broken connection costs latency, never an answer.

    SCREENSHOTS: the live CLI keeps every tool result in memory and re-sends it
    on each later turn, and the SDK has no way to edit a result once sent (a
    PostToolUse hook rewrites output only as it is produced; ``fork_session``
    copies transcripts, it does not edit them). So a turn that looked at the
    screen ROTATES the connection when it ends: the image dies with that client,
    a fresh one is connected in the background while the user is still talking,
    and its first prompt carries the conversation as text plus a one-line note
    per old capture. Cost: the next turn re-reads a few hundred tokens of text
    instead of ~1,400 per image forever.
    """

    def __init__(self) -> None:
        self._loop: Any = None
        self._thread: Any = None
        self._client: Any = None
        self._bridge: _Bridge | None = None
        self._state: _TurnState | None = None
        #: Everything that is baked in at connect time. When any of it changes
        #: — a new language, a preference the user just taught — the prompt the
        #: client holds is stale, so the connection is rebuilt rather than
        #: quietly answering under the old instructions.
        self._signature: tuple | None = None
        #: True until the current client has been asked anything. Its first
        #: prompt is the only one that needs the conversation as text: after
        #: that the client holds the real thing.
        self._fresh = False
        #: How the last client was opened, so a rotation can open the next one
        #: identically without waiting for the next question to say how.
        self._open_args: dict[str, Any] | None = None
        self._prewarm_thread: threading.Thread | None = None
        #: The live client holds a screenshot and must not answer again.
        self._stale = False
        self._closed = False
        self._connectors: "Connectors | None" = None
        self._lock = threading.Lock()

    # -- loop ownership ------------------------------------------------

    def _ensure_loop(self) -> None:
        if self._loop is not None:
            return
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=loop.run_forever, name="wawi-brain", daemon=True
        )
        thread.start()
        self._loop, self._thread = loop, thread

    def _run(self, coro: Any, timeout: float) -> Any:
        """Run *coro* on the engine loop and wait for it from this thread."""
        self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    # -- lifecycle -----------------------------------------------------

    def _connect(self, signature: tuple, options: Any, bridge: _Bridge,
                 state: _TurnState) -> None:
        from claude_agent_sdk import ClaudeSDKClient  # type: ignore[import]

        client = ClaudeSDKClient(options=options)
        self._run(client.connect(), timeout=_CONNECT_TIMEOUT_S)
        self._client = client
        self._bridge = bridge
        self._state = state
        self._signature = signature
        self._fresh = True
        log.info("brain: persistent session connected")

    def _open(self, *, model: str, system: str, max_iterations: int,
              screen_options: ScreenOptions | None, bypass_permissions: bool,
              streaming: bool, connectors: "Connectors | None" = None) -> None:
        """Build the tools and options for one client and connect it.

        THE CALLER MUST ALREADY HOLD ``_lock``.
        """
        signature = (model, system, max_iterations, bypass_permissions,
                     screen_options, streaming,
                     _extras_signature(connectors))
        state = _TurnState(max_iterations=max_iterations)
        bridge = _build_bridge(state, screen_options)
        options = _build_options(
            bridge,
            model=model,
            max_iterations=max_iterations,
            system=system,
            bypass_permissions=bypass_permissions,
            partial_messages=streaming,
            connectors=connectors,
        )
        self._open_args = dict(
            model=model, system=system, max_iterations=max_iterations,
            screen_options=screen_options, bypass_permissions=bypass_permissions,
            streaming=streaming, connectors=connectors,
        )
        self._connect(signature, options, bridge, state)
        if connectors:
            # Annotations are only visible on a live client, so this is where
            # reads are proven. A refresh that fails leaves none proven, which
            # denies rather than allows.
            self._connectors = connectors
            connectors.set_status_source(self._client.get_mcp_status)
            self._run(connectors.refresh(), timeout=_CONNECT_TIMEOUT_S)

    def _rotate(self) -> None:
        """Retire a client that holds a screenshot and reconnect in the background.

        THE CALLER MUST ALREADY HOLD ``_lock``. The client is only MARKED here;
        disconnecting it and connecting the next one run on their own thread,
        so the turn that looked is not delayed by either. The next question
        usually arrives well after the ~1.3 s that takes, and if it does not,
        it waits on the lock for a connect already under way — or, if it wins
        the lock first, sees the mark and reconnects itself. Either way the old
        client never answers again.
        """
        log.info("brain: the session holds a screenshot — rotating it so the "
                 "image is not re-sent on every later turn")
        self._stale = True
        args = self._open_args
        if args is None:
            return

        def _prewarm() -> None:
            with self._lock:
                if self._closed:
                    return
                if self._stale:
                    self._teardown()
                if self._client is not None:
                    return
                try:
                    self._open(**args)
                except Exception:
                    # The next question connects on its own; this only cost
                    # the head start.
                    log.warning("brain: background reconnect failed",
                                exc_info=True)
                    self._teardown()

        thread = threading.Thread(target=_prewarm, name="wawi-brain-prewarm",
                                  daemon=True)
        self._prewarm_thread = thread
        thread.start()

    def prewarm(self, *, model: str, system: str, max_iterations: int,
                screen_options: ScreenOptions | None, bypass_permissions: bool,
                streaming: bool, connectors: "Connectors | None" = None) -> None:
        """Connect before the first question without delaying the microphone.

        The ordinary ask owns the same lock, so a question arriving during
        startup waits for this connection instead of launching another one.
        """
        def _start() -> None:
            with self._lock:
                if self._closed or self._client is not None:
                    return
                try:
                    self._open(
                        model=model, system=system,
                        max_iterations=max_iterations,
                        screen_options=screen_options,
                        bypass_permissions=bypass_permissions,
                        streaming=streaming, connectors=connectors,
                    )
                except Exception:
                    log.warning("brain: background prewarm failed", exc_info=True)
                    self._teardown()

        if self._closed or self._client is not None or (
            self._prewarm_thread is not None and self._prewarm_thread.is_alive()
        ):
            return
        thread = threading.Thread(target=_start, name="wawi-brain-prewarm",
                                  daemon=True)
        self._prewarm_thread = thread
        thread.start()

    def close(self) -> None:
        """Tear the connection down. Safe to call twice, and never raises."""
        with self._lock:
            self._closed = True
            self._teardown()

    def _teardown(self) -> None:
        """Close the connection. THE CALLER MUST ALREADY HOLD ``_lock``.

        Split from :meth:`close` because ``ask`` tears the connection down from
        inside its own critical section — to reconnect, or after a failure. A
        plain Lock is not reentrant, so calling the locking version there
        deadlocked the worker thread permanently: Wawi simply stopped
        answering, with no error anywhere. The tests below cover both paths.
        """
        client, self._client = self._client, None
        self._signature = None
        self._fresh = False
        self._stale = False
        connectors, self._connectors = self._connectors, None
        if connectors is not None:
            # The status source is this client, bound to this loop. Left
            # behind, the one-shot path would ask a dead client from a loop it
            # was never connected on.
            connectors.set_status_source(None)
        if client is not None:
            try:
                self._run(client.disconnect(), timeout=_DISCONNECT_TIMEOUT_S)
            except Exception:
                log.debug("brain: error while disconnecting", exc_info=True)
        loop, self._loop = self._loop, None
        self._thread = None
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)

    # -- one turn ------------------------------------------------------

    def ask(
        self,
        prompt: str,
        *,
        model: str,
        system: str,
        max_iterations: int,
        screen_options: ScreenOptions | None,
        bypass_permissions: bool,
        timeout: float = _TURN_TIMEOUT_S,
        on_sentence: Callable[[str], None] | None = None,
        history: str = "",
        connectors: "Connectors | None" = None,
        images: "tuple[ImageAttachment, ...]" = (),
    ) -> BrainResult:
        """Run one turn on the live connection, connecting first if needed.

        *history* is the conversation rendered as text. It is sent only as
        the first prompt of a freshly connected client — after a rotation or
        a reconnect — because any later turn already has the real history.
        """
        # Streaming is part of the signature because it is baked into the
        # connected client's options: a turn that wants deltas from a client
        # connected without them would silently get none.
        streaming = on_sentence is not None
        signature = (model, system, max_iterations, bypass_permissions,
                     screen_options, streaming,
                     _extras_signature(connectors))
        with self._lock:
            self._closed = False
            try:
                if (self._client is None or self._stale
                        or self._signature != signature):
                    if self._client is not None:
                        log.info("brain: prompt or model changed, or the "
                                 "session holds a screenshot — reconnecting")
                        self._teardown()
                    self._open(
                        model=model,
                        system=system,
                        max_iterations=max_iterations,
                        screen_options=screen_options,
                        bypass_permissions=bypass_permissions,
                        streaming=streaming,
                        connectors=connectors,
                    )

                # The tool handlers close over ONE state object, so the budget
                # is reset in place rather than by building a new one — a new
                # object would leave the live handlers writing into the old.
                state = self._state
                assert state is not None
                state.calls.clear()
                state.shots.clear()
                state.max_iterations = max_iterations

                emitter = (
                    _SentenceEmitter(on_sentence) if on_sentence is not None else None
                )
                sent = f"{history}{prompt}" if self._fresh and history else prompt
                self._fresh = False

                async def _turn() -> BrainResult:
                    await self._client.query(_with_images(sent, images) if images else sent)
                    return await _collect(
                        self._client.receive_response(), state, emitter,
                        tuple(a.note for a in images),
                    )

                result = self._run(_turn(), timeout=timeout)
                if result.screen_notes:
                    self._rotate()
                return result
            except Exception:
                # A half-dead client answers nothing on every later turn, so it
                # is dropped here rather than left for the next question to
                # discover.
                log.warning("brain: persistent turn failed; dropping the "
                            "connection", exc_info=True)
                self._teardown()
                raise


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def ask_sync(
    user_text: str,
    *,
    model: str,
    backend: str = "claude",
    ollama_model: str = "gemma3:4b",
    ollama_url: str = "http://localhost:11434/api/chat",
    screen_options: ScreenOptions | None = None,
    max_iterations: int = MAX_TOOL_ITERATIONS,
    assistant_name: str = "Wawi",
    history: str = "",
    preferences: str = "",
    issues_hint: str = "",
    recall: str = "",
    bypass_permissions: bool = False,
    language: str = "es",
    resume: str | None = None,
    engine: "Engine | None" = None,
    on_sentence: Callable[[str], None] | None = None,
    on_restart: Callable[[], None] | None = None,
    connectors: "Connectors | None" = None,
    images: "tuple[ImageAttachment, ...]" = (),
) -> BrainResult:
    """Synchronous brain turn. Dispatches to Claude (tools) or Ollama (degraded).

    Args:
        user_text: Transcribed question.
        model: Claude model ID; ignored when *backend* is ``"ollama"``.
        backend: ``"claude"`` (default) or ``"ollama"``.
        ollama_model: Ollama model tag.
        ollama_url: Ollama chat endpoint.
        screen_options: Enables the optional screen tool when provided.
        max_iterations: Tool executions allowed this turn.
        assistant_name: What the assistant calls itself; must match the wake
            phrase or it introduces itself as something unsummonable.
        history: Rendered conversation context, prepended to the prompt. See
            :mod:`wawi.conversation`. Ignored when *resume* is given — the
            server already holds the real thing — and on the live engine
            except for the first turn of a freshly connected client.
        recall: Exchanges carried over from previous runs
            (:mod:`wawi.recall`), rendered into the SYSTEM prompt. Unlike
            *history* it reaches every path, including a resumed session and
            the live-connection one.
        resume: SDK session id from the previous turn. Continues that
            conversation instead of starting a new one. Ignored when *engine*
            is given — a live connection already IS the conversation.
        engine: A connected :class:`Engine` to run the turn on, saving the
            ~1.3 s the CLI takes to start. A failure there falls back to the
            one-shot path for that turn, so this can only cost latency.
        bypass_permissions: Deprecated and ignored here; kept so old callers do
            not break. Permission tiers live in :mod:`wawi.guards`.
        on_sentence: Speak-as-you-go hook; see :func:`ask`. The Ollama backend
            ignores it — that endpoint is called with ``stream: False`` and has
            nothing to hand over until it is finished.
        on_restart: Called when a partly-streamed turn is abandoned and started
            again from scratch on the one-shot path. Whatever was already
            spoken is now half of an answer that no longer exists, and only the
            caller can decide what to do about it.
        connectors: Borrowed MCP servers (:mod:`wawi.connectors`). Handed
            the transcript here, before any path runs, because the user's own
            words are the only place a "sí" to a pending send may come from.
        images: Pictures attached to THIS user turn (something dropped on
            her). Their notes come back in ``screen_notes``, so the session
            that holds them is treated like one that saw the screen. Ollama
            gets the text only.

    Returns:
        A :class:`BrainResult` whose ``text`` is safe to speak as-is.
    """
    if backend == "ollama":
        log.debug("Brain: Ollama backend (model=%s) — no tools", ollama_model)
        return ask_ollama(
            user_text,
            model=ollama_model,
            url=ollama_url,
            assistant_name=assistant_name,
            history=history,
            language=language,
        )

    from wawi import plugins

    # Registry changes (menu, terminal) reach the gate before this turn's
    # connection is chosen, so the signature below sees them.
    plugins.sync(connectors)
    if connectors:
        connectors.begin_turn(user_text)

    spoken = 0

    def _counted(sentence: str) -> None:
        nonlocal spoken
        spoken += 1
        on_sentence(sentence)  # type: ignore[misc]

    stream_hook = _counted if on_sentence is not None else None

    if engine is not None:
        from wawi.prompts import system_prompt

        try:
            return engine.ask(
                user_text,
                model=model,
                system=system_prompt(
                    assistant_name,
                    preferences,
                    issues_hint,
                    language=language,
                    recall=recall,
                ),
                max_iterations=max_iterations,
                screen_options=screen_options,
                bypass_permissions=bypass_permissions,
                on_sentence=stream_hook,
                history=history,
                connectors=connectors,
                images=tuple(images),
            )
        except Exception:
            log.warning(
                "brain: falling back to a one-shot turn after a persistent "
                "failure (this turn is ~1.3s slower)"
            )
            if spoken:
                # The retry writes a NEW answer, so streaming it would speak a
                # second version on top of the fragment already said. Streaming
                # is dropped for the retry and the caller is told, so the whole
                # answer is delivered once, complete, at the end. Repeating a
                # sentence is cheap; leaving the user with an abandoned half of
                # one is not.
                log.warning(
                    "brain: %d sentence(s) of the abandoned turn were already "
                    "spoken; the retry will not stream",
                    spoken,
                )
                stream_hook = None
                if on_restart is not None:
                    on_restart()

    return asyncio.run(
        ask(
            user_text,
            model=model,
            screen_options=screen_options,
            max_iterations=max_iterations,
            assistant_name=assistant_name,
            history=history,
            preferences=preferences,
            issues_hint=issues_hint,
            recall=recall,
            bypass_permissions=bypass_permissions,
            language=language,
            resume=resume,
            on_sentence=stream_hook,
            connectors=connectors,
            images=tuple(images),
        )
    )
