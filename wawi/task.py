"""Two agents, one mailbox: the orchestrator and the executor.

Wawi already had one model — the conversational :class:`wawi.brain.Engine`,
which answers in seconds and whose tools cannot touch a file. This module adds
a SECOND one that does real work, and the whole design is about keeping them
apart.

**Wawi is the orchestrator.** It stays latent and always available. It owns
the microphone and it owns :func:`wawi.voice.say`. It dispatches work, and it
keeps attending both the user and the work — so "¿cómo va?" is answered by an
ordinary turn reading task state, not by interrupting anything.

**The executor is a separate SDK client** with Bash, Read, Write and Edit. It
never speaks, never touches the listener, never touches the microphone. Its one
outward channel is the :class:`Mailbox`.

Why the separation is the point rather than an implementation detail: an agent
that can both hold the microphone and run a shell for half an hour is an agent
that can stop answering you. Everything the executor wants to say goes into the
mailbox and waits for Wawi to decide when — and whether — it reaches the
user. The executor blocking on an answer costs the executor; it costs the
conversation nothing.

The executor's options differ from the conversational ones on every axis that
matters, and merging the two builders later would quietly hand the always-on
path a shell:

* ``tools`` is an ENABLING list here and ``[]`` (disable everything) there.
* ``permission_mode`` is ``"default"`` here so ``can_use_tool`` is consulted;
  the conversational engine uses ``"dontAsk"``, which denies without asking.
* no ``max_turns`` cap, because the executor is doing the long thing.

One executor at a time. Not a resource limit — an addressing one: two
executors posting questions into one mailbox cannot be told apart by the person
answering, so the second would collect the first one's answer.
"""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import itertools
import logging
import pathlib
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from wawi import guards, lang
from wawi.inflight import Inflight

log = logging.getLogger(__name__)

#: Built-in tools the executor may use. This is the real blast radius of the
#: feature and the reason ``tasks.enabled`` defaults to false.
DEFAULT_TOOLS: tuple[str, ...] = ("Bash", "Read", "Write", "Edit", "Glob", "Grep")

MCP_SERVER_NAME = "wawi_task"
"""Deliberately NOT ``wawi``. The executor's tools and the conversational
ones are two disjoint sets, and a shared server name is one config mistake away
from the conversational model being able to call them."""

ASK_TOOL = "ask_user"
PROGRESS_TOOL = "report_progress"

#: How long the SDK gets to start the ``claude`` binary and connect.
_CONNECT_TIMEOUT_S = 60.0

#: Progress lines kept. Bounded because they exist to be read out ("¿cómo
#: va?") and shown on a HUD strip, neither of which can use a hundred of them.
_PROGRESS_MAX = 20

#: What the executor is told when nobody answered. A string, never an
#: exception: an unanswered question is information — "decide alone" — and
#: raising would end the work over a user who walked away from the desk.
NO_ANSWER = (
    "Nobody answered in time. The user is away. Decide on your own using your "
    "best judgement and note the assumption in your final summary."
)

DENIED_TIMEOUT = "Denied: nobody approved this in time. Do not retry it."
DENIED_BY_USER = "Denied: the user said no. Find another way; do not retry this."
DENIED_CANCELLED = "Denied: the task was cancelled."

#: Cap on the tool detail read out in an approval question. It is heard, not
#: read, and a 300-character shell pipeline spoken aloud is not a question.
_SPOKEN_DETAIL_CHARS = 90

#: Cap on one progress line. Progress is a HUD strip and a spoken summary.
_PROGRESS_CHARS = 160


# --------------------------------------------------------------------------- #
# Mailbox — the ONLY coupling between the two agents
# --------------------------------------------------------------------------- #


class RequestKind(enum.Enum):
    QUESTION = "QUESTION"
    APPROVAL = "APPROVAL"


#: Event names handed to the mailbox observer. The observer runs on the
#: EXECUTOR's thread, so whatever it does has to be quick and must not block.
EVENT_PROGRESS = "progress"
EVENT_REQUEST = "request"
EVENT_RESOLVED = "resolved"
EVENT_FINISHED = "finished"


@dataclass
class Request:
    """One thing the executor is waiting on the user for.

    ``event`` is what the executor blocks on and what the orchestrator sets.
    It is inside the request rather than on the mailbox so a stale answer to a
    question that already timed out cannot resolve the NEXT one.
    """

    id: str
    kind: RequestKind
    text: str
    created_at: float = field(default_factory=time.time)
    answer: str = ""
    answered: bool = False
    spoken: bool = False
    #: A finished spoken question (``{"es": ..., "en": ...}``) for a RISKY
    #: approval. None means "say the generic sentence around ``text``".
    prompt: dict[str, str] | None = None
    event: threading.Event = field(default_factory=threading.Event)


class Mailbox:
    """Thread-safe letterbox between the executor and the orchestrator.

    Executor side (blocking is fine, it is what it is there for):
    :meth:`progress`, :meth:`ask`, :meth:`approval`.

    Orchestrator side (never blocks): :meth:`pending`, :meth:`answer`,
    :meth:`recent_progress`, :meth:`latest`.
    """

    def __init__(self, observer: Callable[[str], None] | None = None) -> None:
        self._lock = threading.Lock()
        self._progress: deque[tuple[float, str]] = deque(maxlen=_PROGRESS_MAX)
        self._pending: Request | None = None
        self._ids = itertools.count(1)
        self._observer = observer

    def set_observer(self, observer: Callable[[str], None] | None) -> None:
        self._observer = observer

    def notify(self, event: str) -> None:
        """Poke the orchestrator. Its failure is never the executor's problem."""
        observer = self._observer
        if observer is None:
            return
        try:
            observer(event)
        except Exception:
            log.warning("mailbox: observer failed on %r", event, exc_info=True)

    # -- executor side --------------------------------------------------

    def progress(self, text: str) -> None:
        """Record one line of visible progress. Never blocks, never raises."""
        text = " ".join((text or "").split())[:_PROGRESS_CHARS]
        if not text:
            return
        with self._lock:
            self._progress.append((time.time(), text))
        log.info("task progress: %s", text)
        self.notify(EVENT_PROGRESS)

    def ask(self, question: str, timeout: float) -> str:
        """Post a question and block until it is answered or *timeout* passes.

        Returns :data:`NO_ANSWER` rather than raising on a timeout — see the
        note on that constant.
        """
        request = self._post(RequestKind.QUESTION, question)
        if request is None:
            return NO_ANSWER
        answered = request.event.wait(timeout)
        answer = self._collect(request)
        if not answered or not answer:
            log.info("mailbox: question %s went unanswered", request.id)
            return NO_ANSWER
        return answer

    def approval(
        self,
        description: str,
        timeout: float,
        prompt: dict[str, str] | None = None,
    ) -> tuple[bool, str]:
        """Post an approval request. Returns ``(allowed, reason)``.

        *prompt* is the short question to say instead of the generic one; see
        :mod:`wawi.guards`. *description* still goes to the log and the HUD.

        Never hangs and never defaults to yes: a timeout, a cancelled task and
        an answer that is neither yes nor no all deny. Denying costs the
        executor one tool call, which it is built to handle. Hanging costs the
        user their assistant, and guessing yes costs the repository.
        """
        request = self._post(RequestKind.APPROVAL, description, prompt)
        if request is None:
            return False, DENIED_CANCELLED
        answered = request.event.wait(timeout)
        answer = self._collect(request)
        if not answered:
            return False, DENIED_TIMEOUT
        if not answer:
            return False, DENIED_CANCELLED
        if lang.affirmative(answer) is True:
            return True, ""
        return False, DENIED_BY_USER

    def _post(
        self, kind: RequestKind, text: str, prompt: dict[str, str] | None = None
    ) -> Request | None:
        text = " ".join((text or "").split())
        if not text:
            return None
        request = Request(id=f"q{next(self._ids)}", kind=kind, text=text, prompt=prompt)
        with self._lock:
            if self._pending is not None:
                # One executor means one waiter, so this is a bug rather than
                # contention. The old request is released denied instead of
                # being orphaned with something blocked on it forever.
                log.warning("mailbox: replacing an unanswered %s", self._pending.id)
                self._pending.event.set()
            self._pending = request
        log.info("mailbox: %s %s — %s", kind.value.lower(), request.id, text[:160])
        self.notify(EVENT_REQUEST)
        return request

    def _collect(self, request: Request) -> str:
        """Take the answer and clear the slot if it is still ours."""
        with self._lock:
            if self._pending is request:
                self._pending = None
            return request.answer if request.answered else ""

    # -- orchestrator side ----------------------------------------------

    def pending(self) -> Request | None:
        """The request still waiting on the user, or None.

        An ANSWERED request is not pending, even though the slot still holds
        it: :meth:`_collect` clears it on the executor's thread, which does
        not run until the executor wakes up. Reporting it in that window is
        how the same question gets asked out loud twice -- the orchestrator
        sees EVENT_RESOLVED, re-reads this, and finds the question it just
        got an answer to still sitting here.
        """
        with self._lock:
            if self._pending is not None and self._pending.answered:
                return None
            return self._pending

    def mark_spoken(self, request_id: str) -> bool:
        """Claim the right to speak a request. False when someone already did.

        The claim and the check are one operation on purpose: the question is
        spoken from a thread, and two of them arriving together would otherwise
        both read "not spoken yet" and ask the user twice.
        """
        with self._lock:
            request = self._pending
            if request is None or request.id != request_id or request.spoken:
                return False
            request.spoken = True
            return True

    def answer(self, text: str) -> bool:
        """Resolve whatever the executor is waiting on. False if nothing is.

        Called from an ordinary conversational turn, via the ``answer_task``
        tool: the user's reply comes in through the normal microphone path and
        the conversational model decides that it was meant for the executor.
        """
        text = (text or "").strip()
        with self._lock:
            request = self._pending
            if request is None or request.answered:
                return False
            request.answer = text
            request.answered = True
        request.event.set()
        self.notify(EVENT_RESOLVED)
        log.info("mailbox: %s answered — %r", request.id, text[:120])
        return True

    def cancel(self) -> None:
        """Release anything blocked, denied. Used when the task is stopped."""
        with self._lock:
            request, self._pending = self._pending, None
        if request is not None:
            request.event.set()

    def recent_progress(self, limit: int = 5) -> list[str]:
        with self._lock:
            return [text for _, text in list(self._progress)[-limit:]]

    def latest(self) -> str:
        with self._lock:
            return self._progress[-1][1] if self._progress else ""

    def clear(self) -> None:
        with self._lock:
            self._progress.clear()
            self._pending = None


# --------------------------------------------------------------------------- #
# Task record
# --------------------------------------------------------------------------- #


class TaskStatus(enum.Enum):
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    STOPPED = "STOPPED"


@dataclass
class Task:
    """One unit of delegated work.

    ``id`` is short and speakable ("t1") on purpose: it is said out loud when
    the work starts and again when it ends, and a uuid is not something a
    person can carry between the two.
    """

    id: str
    goal: str
    created_at: float = field(default_factory=time.time)
    status: TaskStatus = TaskStatus.RUNNING
    result: str = ""
    error: str = ""
    finished_at: float | None = None
    # The executor's SDK session, once the SDK reports it. Persisted so a run
    # cut off by a restart can be resumed instead of started over.
    session_id: str = ""
    # The session this run continues, when it is a resumed one. Empty for a
    # fresh start.
    resume: str = ""

    @property
    def elapsed_s(self) -> float:
        return (self.finished_at or time.time()) - self.created_at

    @property
    def running(self) -> bool:
        return self.status is TaskStatus.RUNNING


# --------------------------------------------------------------------------- #
# Session driver — the one place the SDK is actually spoken to
# --------------------------------------------------------------------------- #


async def _drive_session(
    goal: str,
    options: Any,
    on_client: Callable[[Any], None],
    *,
    resume: str | None = None,
    on_session: Callable[[str], None] | None = None,
) -> Any:
    """Connect a fresh client, send *goal*, and read the whole run back.

    Split out from :class:`TaskRunner` so the tests can replace the SDK
    wholesale: everything above this line is policy, everything below it is a
    subprocess.

    *on_client* hands the live client back so :meth:`TaskRunner.stop` has
    something to interrupt. There is deliberately no timeout on the response —
    the executor IS the long thing, and the bound on it is the user, who can
    see it working and can stop it.

    *resume* continues an earlier session instead of opening one. It raises
    when this SDK cannot resume, rather than quietly starting fresh: the caller
    sends a "continue" prompt that means nothing to a blank session, and only
    an error tells it to fall back to the original goal.

    *on_session* gets the session id from the FIRST message that carries one,
    not from the final result: the point is to have it on disk if the process
    dies halfway through, which is exactly when there is no final result.
    """
    from claude_agent_sdk import ClaudeSDKClient  # type: ignore[import]

    from wawi import brain

    if resume:
        options = dataclasses.replace(options, resume=resume)
    client = ClaudeSDKClient(options=options)
    await asyncio.wait_for(client.connect(), timeout=_CONNECT_TIMEOUT_S)
    on_client(client)
    try:
        await client.query(goal)
        # _collect reads this only to label its log line; the real bound on the
        # executor's tool use is can_use_tool, i.e. the user.
        state = brain._TurnState(max_iterations=10**6)
        stream = client.receive_response()
        if on_session is not None:
            stream = _watch_session(stream, on_session)
        return await brain._collect(stream, state)
    finally:
        try:
            await client.disconnect()
        except Exception:
            log.debug("task: error while disconnecting", exc_info=True)


async def _watch_session(stream: Any, on_session: Callable[[str], None]) -> Any:
    """Pass *stream* through, reporting its session id the first time one shows.

    The init SystemMessage carries it in ``data``; later messages carry it as
    an attribute. Either is the same id, and only the first is reported.
    """
    reported = False
    async for msg in stream:
        if not reported:
            session_id = getattr(msg, "session_id", None)
            data = getattr(msg, "data", None)
            if not session_id and isinstance(data, dict):
                session_id = data.get("session_id")
            if isinstance(session_id, str) and session_id:
                reported = True
                try:
                    on_session(session_id)
                except Exception:
                    log.debug("task: session callback failed", exc_info=True)
        yield msg


def _resume_prompt(goal: str) -> str:
    """What a resumed executor is told. The goal rides along in case the
    session it wakes up in holds less than it should."""
    return (
        "You were cut off by a restart in the middle of this task. Continue "
        f"where you left off; do not redo finished steps. The task was: {goal}"
    )


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TaskOptions:
    """Everything the executor needs that is not the goal itself."""

    model: str
    cwd: str
    tools: tuple[str, ...] = DEFAULT_TOOLS
    ask_timeout_s: float = 300.0
    approval_timeout_s: float = 120.0
    #: How long a "sí" to ordinary work keeps covering that same task.
    grant_minutes: float = guards.DEFAULT_GRANT_MINUTES
    #: The ``dangerous_skip_permissions`` compatibility mapping: yolo for one
    #: hour from startup, instead of the old bypass that never expired.
    yolo_at_start: bool = False


class TaskRunner:
    """Runs ONE executor at a time, on its own thread, with its own client.

    Threading: the executor owns a private event loop for its whole life,
    created and torn down on its worker thread. Unlike :class:`brain.Engine`
    nothing is marshalled in from outside — the loop serves exactly one
    coroutine — so there is no cross-thread call to get wrong except
    :meth:`stop`, which is explicit about it.

    The two blocking mailbox calls are pushed onto a thread with
    ``asyncio.to_thread``: a five-minute wait for an answer must not stop the
    loop from servicing the SDK transport underneath it.
    """

    def __init__(
        self,
        options: TaskOptions,
        mailbox: Mailbox | None = None,
        gate: guards.PermissionGate | None = None,
        inflight: Inflight | None = None,
    ) -> None:
        self._options = options
        self.mailbox = mailbox if mailbox is not None else Mailbox()
        self.gate = (
            gate if gate is not None
            else guards.PermissionGate(grant_minutes=options.grant_minutes)
        )
        if options.yolo_at_start:
            log.warning(
                "dangerous_skip_permissions is deprecated: it now means yolo for "
                "%.0f minutes from startup, then normal questions again",
                guards.YOLO_MINUTES,
            )
            self.gate.enable_yolo()
        # Where a running task is written down so a restart can offer it back
        # (wawi/inflight.py). The default keeps nothing.
        self._inflight = inflight if inflight is not None else Inflight(None)
        # Set by close(): a task ended by the process shutting down was cut,
        # not finished, and keeps its record.
        self._closing = False
        self._lock = threading.Lock()
        self._current: Task | None = None
        self._thread: threading.Thread | None = None
        self._client: Any = None
        self._loop: Any = None
        self._counter = 0

    # -- state read by the conversational side --------------------------

    @property
    def current(self) -> Task | None:
        with self._lock:
            return self._current

    @property
    def is_running(self) -> bool:
        task = self.current
        return task is not None and task.running

    def status(self) -> dict[str, Any]:
        """A snapshot the ``task_status`` tool can hand to the model.

        Everything the model needs to answer "¿cómo va?" without waking the
        executor: what it is doing, how long it has been at it, and whether it
        is in fact stuck waiting on an answer nobody gave.
        """
        task = self.current
        if task is None:
            return {"running": False, "task": None}
        pending = self.mailbox.pending()
        return {
            "running": task.running,
            "id": task.id,
            "goal": task.goal,
            "status": task.status.value,
            "elapsed_s": round(task.elapsed_s),
            "progress": self.mailbox.recent_progress(),
            "waiting_for_you": (
                {"kind": pending.kind.value, "text": pending.text}
                if pending is not None
                else None
            ),
            "result": task.result,
            "error": task.error,
        }

    # -- lifecycle ------------------------------------------------------

    def start(self, goal: str, resume: str | None = None) -> Task | None:
        """Dispatch *goal*. None means one is already running, or it was empty.

        Returns as soon as the thread is spawned. The caller is a
        conversational turn holding the turn lock, and a turn that waits for
        the executor is a microphone that stopped working.

        *resume* is the SDK session of an earlier run of the same goal, cut off
        by a restart. The executor continues it, and starts *goal* over from
        scratch if that session cannot be resumed.
        """
        goal = (goal or "").strip()
        if not goal:
            return None
        with self._lock:
            if self._current is not None and self._current.running:
                return None
            self._counter += 1
            task = Task(id=f"t{self._counter}", goal=goal, resume=resume or "")
            self._current = task
            self._client = None
            self._loop = None
            thread = threading.Thread(
                target=self._worker, args=(task,), name="wawi-task", daemon=True
            )
            self._thread = thread
        self._inflight.mark_task(task.id, goal, resume or None)
        self.mailbox.clear()
        self.gate.begin_task()
        verb = "resuming" if resume else "starting"
        self.mailbox.progress(f"{verb}: {_shorten(goal, 120)}")
        thread.start()
        log.info("task %s dispatched: %s", task.id, goal[:160])
        return task

    def stop(self) -> bool:
        """Ask the executor to stop. False when there was nothing to stop.

        Two halves, and both are needed: the mailbox is cancelled so anything
        blocked on an answer is released denied, and the SDK client is
        interrupted so the model stops mid-turn. Interrupting alone leaves a
        thread parked on an Event nobody will ever set.
        """
        task = self.current
        if task is None or not task.running:
            return False
        task.status = TaskStatus.STOPPED
        self.mailbox.cancel()
        self.gate.end_task()
        client, loop = self._client, self._loop
        if client is not None and loop is not None:
            try:
                # The client belongs to the executor's loop; calling it from
                # this thread is the one cross-loop call in this module and the
                # reason it goes through run_coroutine_threadsafe.
                future = asyncio.run_coroutine_threadsafe(client.interrupt(), loop)
                future.result(timeout=10.0)
            except Exception:
                log.warning("task %s: interrupt failed", task.id, exc_info=True)
        log.info("task %s stopped", task.id)
        return True

    def panic(self) -> bool:
        """Stop everything NOW: revoke every grant and yolo, then stop.

        Revoking comes first and takes no I/O, so even if interrupting the
        client is slow, nothing else gets approved in the meantime. Meant for
        a code path that does not go through the LLM (a panic word, a key).
        """
        self.gate.revoke_all()
        self.mailbox.cancel()
        return self.stop()

    def close(self) -> None:
        """Shut down at exit. Safe to call twice; never raises.

        The running task's record is kept: it was cut by the shutdown, and the
        next start offers to pick it up.
        """
        self._closing = True
        try:
            self.stop()
        except Exception:
            log.debug("task: close failed", exc_info=True)

    # -- the worker -----------------------------------------------------

    def _worker(self, task: Task) -> None:
        """Own an event loop for the length of the task, then give it back."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            result = loop.run_until_complete(self._execute(task))
            task.result = getattr(result, "text", "") or ""
            if task.status is not TaskStatus.STOPPED:
                task.status = TaskStatus.DONE
        except Exception as exc:
            log.warning("task %s failed", task.id, exc_info=True)
            task.error = f"{type(exc).__name__}: {exc}"[:400]
            if task.status is not TaskStatus.STOPPED:
                task.status = TaskStatus.FAILED
        finally:
            task.finished_at = time.time()
            self.gate.end_task()
            self._client = None
            self._loop = None
            if not self._closing:
                self._inflight.clear_task(task.id)
            # Anything still blocked on an answer would otherwise sit on a dead
            # task's Event until its timeout.
            self.mailbox.cancel()
            try:
                loop.close()
            except Exception:
                log.debug("task %s: loop close failed", task.id, exc_info=True)
            log.info("task %s ended %s in %.0fs", task.id, task.status.value,
                     task.elapsed_s)
            self.mailbox.progress(f"{task.status.value.lower()} ({task.id})")
            # Fired last, and separately from the progress line: the outcome is
            # the one thing the orchestrator SPEAKS, and it must not be
            # triggered by a progress update that happens to be the last one.
            self.mailbox.notify(EVENT_FINISHED)

    async def _execute(self, task: Task) -> Any:
        options = self._build_options()

        def _hold(client: Any) -> None:
            self._client = client

        def _session(session_id: str) -> None:
            task.session_id = session_id
            self._inflight.set_task_session(task.id, session_id)

        if task.resume:
            try:
                return await _drive_session(
                    _resume_prompt(task.goal), options, _hold,
                    resume=task.resume, on_session=_session,
                )
            except Exception:
                if task.status is TaskStatus.STOPPED:
                    raise
                # Expired, unknown, or an SDK that cannot resume. The goal is
                # still on hand, so the work is not lost — only its progress.
                log.warning("task %s: could not resume session %s; starting over",
                            task.id, task.resume[:8], exc_info=True)
                self.mailbox.progress("could not resume; starting over")
                self._client = None
        return await _drive_session(task.goal, options, _hold, on_session=_session)

    # -- executor-only tools --------------------------------------------

    def _sdk_tools(self) -> list[Any]:
        """The executor's own MCP tools. NOT the conversational set.

        Both of them are ways of talking to the mailbox, which is the only
        thing the executor is allowed to talk to.
        """
        from claude_agent_sdk import tool as sdk_tool  # type: ignore[import]

        async def _ask(args: dict) -> dict[str, Any]:
            # to_thread, not a direct call: the answer can be minutes away and
            # the loop underneath still has an SDK transport to service.
            answer = await asyncio.to_thread(
                self.mailbox.ask,
                str(args.get("question", "")),
                self._options.ask_timeout_s,
            )
            return _text(answer)

        async def _progress(args: dict) -> dict[str, Any]:
            self.mailbox.progress(str(args.get("text", "")))
            return _text("Noted.")

        return [
            sdk_tool(
                ASK_TOOL,
                "Ask the user a question and wait for their answer. It reaches "
                "them by voice and blocks you until they reply, so use it only "
                "for decisions that are genuinely theirs.",
                {
                    "type": "object",
                    "properties": {"question": {"type": "string"}},
                    "required": ["question"],
                },
            )(_ask),
            sdk_tool(
                PROGRESS_TOOL,
                "Report what you are doing right now, in one short sentence. "
                "Call it whenever you move to a new step: it is the only thing "
                "the user can see while you work.",
                {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )(_progress),
        ]

    # -- permissions ----------------------------------------------------

    async def _permission(self, name: str, tool_input: dict) -> Any:
        """Decide one tool call: tiers in :mod:`wawi.guards`, voice for the rest.

        Only an ASK reaches the mailbox. A READ never makes a sound, and a
        BLOCKED call is refused without asking anyone, because there is no
        answer to that question the user could give by voice that we accept.
        """
        from claude_agent_sdk.types import (  # type: ignore[import]
            PermissionResultAllow,
            PermissionResultDeny,
        )

        decision = self.gate.decide(name, tool_input, self._options.cwd)
        if decision.verdict is guards.Verdict.ALLOW:
            return PermissionResultAllow()
        if decision.verdict is guards.Verdict.DENY:
            return PermissionResultDeny(message=decision.reason)
        description = _describe_call(name, tool_input, self._options.cwd)
        allowed, reason = await asyncio.to_thread(
            self.mailbox.approval,
            description,
            self._options.approval_timeout_s,
            decision.prompt,
        )
        if allowed:
            self.gate.approved(decision)
            return PermissionResultAllow()
        return PermissionResultDeny(message=reason)

    # -- SDK plumbing ---------------------------------------------------

    def _build_options(self) -> Any:
        """Assemble ClaudeAgentOptions, dropping any field this SDK rejects.

        Same probe-one-field-at-a-time shape as
        :func:`wawi.brain._build_options`, and for the same reason: an SDK
        upgrade that renames a field should cost capability, not the feature.
        """
        from claude_agent_sdk import (  # type: ignore[import]
            ClaudeAgentOptions,
            create_sdk_mcp_server,
        )

        from wawi import brain

        server = create_sdk_mcp_server(MCP_SERVER_NAME, "1.0.0", self._sdk_tools())
        own_tools = [
            f"mcp__{MCP_SERVER_NAME}__{ASK_TOOL}",
            f"mcp__{MCP_SERVER_NAME}__{PROGRESS_TOOL}",
        ]

        # MEASURED, not assumed: this is consulted for Bash, and for a Read
        # whose path falls outside cwd -- a denial there really does stop the
        # command (verified with a filesystem canary). It is NOT consulted for
        # Grep, Glob, or a Read inside the workspace: the CLI treats read-only
        # in-workspace access as needing no permission, and never asks. So this
        # gates the dangerous calls, not every call. Anyone reasoning about the
        # blast radius should reason from the `tools` list first, and from this
        # callback second.
        async def _can_use_tool(name: str, tool_input: dict, context: Any) -> Any:
            return await self._permission(name, dict(tool_input or {}))

        kwargs: dict[str, Any] = {
            "model": self._options.model,
            "system_prompt": _system_prompt(self._options.tools),
            "env": brain._build_env(),
            "mcp_servers": {MCP_SERVER_NAME: server},
            # ONLY the executor's own two tools are auto-approved. Every
            # built-in is deliberately absent: an allowed_tools entry naming a
            # whole tool shadows can_use_tool entirely
            # (claude_agent_sdk/types.py:1833-1861), so listing "Bash" here
            # would silently disable the approval that is the only thing
            # between a mistranscription and a shell.
            # The SDK emits CanUseToolShadowedWarning for these two on every
            # start. That is EXPECTED and correct: ask_user and
            # report_progress are the executor talking to us, not acting on
            # the machine, so gating them would deadlock the very channel the
            # gate asks through. Do not "fix" the warning by removing them.
            "allowed_tools": own_tools,
            # No max_turns: the executor is the long thing. The conversational
            # engine caps turns because a question must be answered in seconds.
            "can_use_tool": _can_use_tool,
        }

        probed: list[tuple[str, Any]] = [
            ("setting_sources", []),
            ("strict_mcp_config", True),
            # Unlike the conversational path — which passes [] to disable every
            # built-in — this is an ENABLING list.
            ("tools", list(self._options.tools)),
            # "default", not "dontAsk": dontAsk denies anything not
            # pre-approved without ever consulting can_use_tool, so the
            # executor would refuse every tool call in silence.
            ("permission_mode", "default"),
            ("cwd", self._options.cwd),
        ]
        for name, value in probed:
            try:
                ClaudeAgentOptions(**{**kwargs, name: value})
                kwargs[name] = value
            except TypeError:
                log.warning(
                    "claude-agent-sdk: field %r not supported on this version; "
                    "dropping it (the executor may be less capable or less safe)",
                    name,
                )

        return ClaudeAgentOptions(**kwargs)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _text(body: str) -> dict[str, Any]:
    """The MCP tool-result shape the SDK expects."""
    return {"content": [{"type": "text", "text": body}], "is_error": False}


def _system_prompt(tools: "tuple[str, ...] | list[str]" = ()) -> str:
    """What the executor is told before it starts.

    Short, and only about what its tools do not already say: that it is not
    talking to a terminal, that asking costs the user's attention, and that
    progress is the only thing anyone can see while it works.

    *tools* is named explicitly because of a real failure: asked to open an
    application with a read-only tool set, the executor searched, found the
    app, and reported DONE -- having never been able to open anything. A model
    that is not told what it lacks reports the search it COULD do as the task
    it could not.
    """
    granted = ", ".join(tools) if tools else "none"
    return (
        "You are the executor agent behind a voice assistant. You were given a "
        "task and you finish it on your own.\n"
        f"- Your ONLY tools are: {granted}. You have nothing else.\n"
        "- If the task needs something you were not given, STOP and say so in "
        "one sentence naming what is missing. Do NOT substitute the nearest "
        "thing you can do and report success: finding a file is not opening "
        "it, and reporting it as done is worse than failing, because nobody "
        "goes looking for a failure that was announced as a success.\n"
        f"- Call {PROGRESS_TOOL} whenever you start a new step. It is the ONLY "
        "thing the user can see while you work; a silent run looks broken.\n"
        f"- Call {ASK_TOOL} only for decisions that are genuinely the user's. "
        "It reaches them by voice and blocks you until they answer.\n"
        "- Reading never needs approval. Ordinary work in the project is "
        "approved once per task. Deleting, sending, paying, pushing and "
        "writing outside the project are asked about EVERY time, so do those "
        "only when the task needs them. Some things (passwords, Keychain, "
        "System Settings, sudo) are always refused: say so, do not work "
        "around it.\n"
        "- Your final message is READ OUT LOUD, at about 15 characters per "
        "second. Two sentences, under 200 characters, no markdown, no code, "
        "no file paths spelled out. Anything longer is a monologue nobody can "
        "interrupt."
    )


def _shorten(text: str, limit: int) -> str:
    """Collapse whitespace and cut *text* to *limit*, never mid-word.

    Spoken and drawn in the HUD, so a line ending in "Befor" reads as damage
    rather than as a summary. Falls back to a hard cut only when the first
    word alone is already too long.
    """
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    space = cut.rfind(" ")
    if space > limit // 2:
        cut = cut[:space]
    return cut.rstrip(" ,.;:") + "…"


def _describe_call(tool_name: str, tool_input: dict, cwd: str = "") -> str:
    """A short phrase naming what is about to happen, fit to be spoken.

    The raw tool input is JSON with a whole file in it. What a person needs in
    order to decide is the verb and the target, so that is all this keeps.

    A target OUTSIDE *cwd* is called out, because that is the case worth
    hearing: the model does invent plausible-looking absolute paths, and a
    read-only tool set is only read-only about the workspace you meant.
    """
    detail = ""
    for key in ("command", "file_path", "path", "pattern", "url"):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            detail = value.strip()
            break
    if not detail:
        return tool_name
    outside = _is_outside(detail, cwd)
    detail = _shorten(detail, _SPOKEN_DETAIL_CHARS)
    if outside:
        return f"{tool_name} OUTSIDE the project: {detail}"
    return f"{tool_name}: {detail}"


def _is_outside(detail: str, cwd: str) -> bool:
    """True when *detail* is an absolute path that does not live under *cwd*.

    Only absolute paths are judged: a relative one resolves against the
    executor's cwd by definition, and a shell command or a grep pattern is not
    a path at all, so neither can be answered here and both are left alone.
    """
    if not cwd or not detail.startswith("/"):
        return False
    try:
        root = pathlib.Path(cwd).resolve()
        return not pathlib.Path(detail).resolve().is_relative_to(root)
    except (OSError, ValueError):
        # An unresolvable path is not evidence of safety.
        return True
