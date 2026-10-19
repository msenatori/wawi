"""Config loading and validation for Wawi.

Loads config.yaml and merges it over hardcoded defaults.
Warns loudly (stderr) if ANTHROPIC_API_KEY is set in the environment.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from wawi import paths as _paths

log = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when config.yaml is missing or malformed."""


#: Autoarreglo (wawi/selffix/): Wawi fixes her own recurring issues and builds
#: the capabilities she was asked for, in a worktree, behind gates (tests,
#: diff limits, forbidden paths, a second-opinion review, a secret scan) and a
#: watchdog that reverts a bad self-merge. Parsed by
#: wawi.selffix.settings.SelffixSettings; the menu switch lives in local_file.
SELFFIX_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    # "auto": she applies what only affects her own behaviour and ASKS for the
    # rest (wawi/selffix/classify.py). "ask": she always asks first.
    "autonomy": "auto",
    "max_attempts_per_day": 2,
    # An observed issue qualifies once it happened this many times.
    "min_count": 3,
    # A capability the MODEL said she lacks (not one the owner asked for).
    "capability_min_count": 2,
    "retry_after_days": 7,
    # Author engine and its fallback: "codex" or "claude".
    "engine": "codex",
    "fallback_engine": "claude",
    # Reviewer: "" = the other engine than the author.
    "reviewer_engine": "",
    "codex_model": "",
    "claude_model": "",
    # Per-attempt budgets. Codex has no cost cap; the time cap holds for both.
    "time_budget_s": 1800,
    "max_budget_usd": 3.0,
    "review_timeout_s": 600,
    "test_timeout_s": 900,
    # Gates.
    "max_changed_lines": 400,
    "max_files": 12,
    # Only while she is idle (no turn for this long, keyboard quiet), on AC
    # power and not in a meeting; or inside the nightly window.
    "idle_after_s": 900,
    "night_window": "02:00-06:00",
    "poll_s": 120,
    # Watchdog after a self-applied restart.
    "watchdog_startup_s": 90,
    "watchdog_watch_s": 1800,
    # Last lines of wawi.log matching the issue, scrubbed, for the brief.
    "log_excerpt_lines": 40,
    # Worktrees and attempt evidence, outside the repository.
    "work_dir": "~/.wawi/selffix",
    # Show the coding agent in an Orca terminal when Orca is running.
    "use_orca": True,
    # A ``pending`` (ask) change not decided in this many days is discarded.
    "pending_days": 3,
    # Optional remote executor for heavy jobs (wawi/selffix/remote.py):
    # an ssh alias, never a hard-coded host. Off by default.
    "remote": {"enabled": False, "host": "", "base_dir": "~/work",
               "min_free_mem_mb": 4096, "max_load_ratio": 0.6},
    "local_file": "selffix_local.json",
}


_DEFAULTS: dict[str, Any] = {
    "language": "es",
    # What the assistant calls itself. Must match the wake phrase, or it
    # introduces itself by one name to someone who has been calling it another.
    "assistant_name": "Wawi",
    # Turns of conversation the model can see. Each one is re-sent (and paid
    # for) on every later turn, so this buys continuity by the sentence.
    "memory_turns": 6,
    # Readable transcript on disk. Empty string disables it entirely and
    # leaves no file behind. This records everything said TO the assistant.
    "conversation_log": "conversation.log",
    # Durable rolling memory across restarts (wawi/recall.py). A handful of
    # recent exchanges, re-sent inside the system prompt of every turn of the
    # NEXT run so a restart is not amnesia. Empty string disables it and leaves
    # no file behind; delete the file to wipe it.
    "recall_file": "recall.txt",
    # Exchanges carried across a restart. Each one is paid for on every turn of
    # the next session, so keep it small; 0 also disables the store.
    "recall_turns": 6,
    # The turn or task a crash or restart cut off (wawi/inflight.py), so the
    # next start can offer to pick it up. Empty string disables it.
    "inflight_file": "inflight.json",
    # Older than this many hours, an interrupted turn or task is dropped
    # without asking: by then nobody is waiting for it.
    "resume_max_age_h": 12.0,
    # Deprecated. ON now means one hour of "yolo" for the executor from
    # startup: risky actions run without the spoken question, then it switches
    # itself off. The blocklist in wawi/guards.py still applies.
    "dangerous_skip_permissions": False,
    # Where the assistant files problems it notices. It can ONLY write issues
    # here — it has no way to edit code. Empty string disables it.
    "issues_dir": "issues",
    # Where a screenshot lands when no project was named. With a project it
    # goes inside that worktree instead, where its agent can open it.
    "screenshots_dir": "screenshots",
    # Standing instructions the user teaches by voice ("sé más breve"). Plain
    # text, one per line — edit or delete by hand. Empty string disables it.
    "preferences_file": "preferences.txt",
    # Also open them on GitHub with `gh issue create`. Off by default: issues
    # on disk are private, issues on GitHub are not.
    "issues_to_github": False,
    "models": {"on_demand": "claude-sonnet-5-5"},
    "audio": {"sample_rate": 16000, "push_to_talk_key": "ctrl_r"},
    "stt": {
        "engine": "mlx-whisper",
        "model": "mlx-community/whisper-large-v3-turbo",
        # Seeds Whisper's decoder. Empty means "use the sample written for the
        # active language" (wawi.lang.STT_PROMPT), which is what you want;
        # set it only to add vocabulary this project keeps mistranscribing.
        # Write it the way you want the output written -- accents included.
        "initial_prompt": "",
        # Junk filter AFTER transcription (wawi/speechgate.py): drops
        # Whisper's hallucinations on noise ("Gracias.", "¡Suscríbete al
        # canal!", echoes of initial_prompt) unless the audio clearly held
        # speech. Genuine short replies ("sí", "no", "dale") are protected.
        "filter_junk": True,
        "min_avg_logprob": -1.0,
        "max_compression_ratio": 2.4,
        "max_no_speech_prob": 0.6,
        "junk_min_voiced_ms": 700,
        "junk_min_logprob": -0.45,
    },
    "screen": {"monitor": 1, "max_edge": 1280, "jpeg_quality": 65},
    "tts": {
        "voice": "ef_dora",
        "voices_dir": None,  # resolved below
        "speed": 1.05,
        # Start speaking the first finished sentence while the rest of the
        # answer is still being written. Off means the old behaviour: the user
        # waits in silence for the whole answer, which measured 19s of nothing
        # on a turn that used no tools at all.
        "streaming": True,
    },
    "ui": {
        "show_tui": True,
        "sound_cues": True,
        "show_panel": True,
        "mascot": True,
        "mascot_only": False,
    },
    # Wawi's body: needs, desktop life, reactions. Parsed and validated by
    # wawi.pet.settings.PetSettings (its DEFAULTS are the documented ones).
    "board": {"allow_network": False},
    "pet": {},
    # What Wawi learns from you (wawi/learn/): tiny numpy models, no LLM.
    # Parsed by wawi.learn.settings.LearnSettings (its defaults are documented).
    "learn": {},
    # Her instincts (wawi/instinct/): your day's rhythm, local only.
    # Parsed by wawi.instinct.settings.InstinctSettings.
    "instinct": {},
    # Games against Wawi (wawi/games/). Parsed by wawi.games.settings.
    "games": {},
    # Coding tasks and visible terminal handoffs to Orca by voice (wawi/orca.py). Parsed by
    # wawi.orca.OrcaSettings (its DEFAULTS are the documented ones).
    "orca": {},
    # Local macOS Shortcuts voice tools (wawi/shortcuts.py).
    "shortcuts": {},
    # Her own little computer (wawi/webvm/): a browser in Docker on your server that
    # she drives for you. Parsed by wawi.webvm.settings (DEFAULTS documented).
    "webvm": {},
    # Spoken reminders and owner-approved recurring brain turns.
    "reminders": {"enabled": True, "local_file": "reminders_local.json", "poll_s": 1.0},
    # "Wawi, tomá nota" (wawi/notes.py): dictation straight to Markdown files,
    # no model involved. A session with no speech for idle_stop_min ends.
    "notes": {"dir": "~/Documents/Wawi/Notas", "idle_stop_min": 30},
    # Dragging things onto her (wawi/drops.py): she catches them and opens the
    # microphone for window_s seconds (no wake word, even muted) to hear what
    # to do; she keeps the item for hold_s for follow-ups, then forgets it.
    # Text, excerpts and transcripts are inlined up to max_text_kb each; plain
    # text files up to max_file_kb are read.
    "drops": {"window_s": 15.0, "hold_s": 180.0, "max_text_kb": 8, "max_file_kb": 200},
    # The buzón "Wawi investiga" (wawi/inbox.py): a little floating mailbox
    # you drop things in and TYPE the order for. The menu switch ("Buzón")
    # and where you left it are kept in private/inbox_local.json and win.
    "inbox": {"enabled": True},
    # Curiosidades (wawi/curiosities.py): while she idles during the day, now
    # and then a tiny fact in her bubble with a "Ver más…" that opens the
    # board. Silent. Parsed by wawi.curiosities.CuriositySettings; the menu
    # switch is kept in local_file, with the cached batch and used titles.
    "curiosities": {
        "enabled": True,
        # At most one every ~every_min minutes, ± jitter (a fraction).
        "every_min": 60.0,
        "jitter": 0.35,
        # Never within this long of a conversation.
        "quiet_after_talk_s": 300.0,
        "show_s": 25.0,
        # Batches alternate Wikipedia (es) and this model, one-shot, cached.
        "model": "claude-haiku-5-5",
        "batch": 10,
        "refill_below": 3,
        "local_file": "curiosities_local.json",
    },
    # Bailar con música (wawi/music.py): she dances while music plays, on
    # this Mac (Spotify / Music.app, never launched) or in the room (heard by
    # the microphone). Parsed by wawi.music.MusicSettings; the menu switch is
    # kept in local_file.
    "music": {
        "enabled": True,
        # Seconds of music before she starts, and of none before she stops.
        "start_after_s": 5.0,
        "stop_after_s": 6.0,
        "mac": True,
        "mic": True,
        "local_file": "music_local.json",
    },
    # Iniciativa (wawi/initiative/): she notices what you need and acts on
    # her own -- goes quiet in a meeting and offers a note after it, says
    # when an Orca agent finished or waits for you, leaves a morning summary
    # on the board. Parsed by wawi.initiative.core.InitiativeSettings; the
    # menu switches and what she learned live in local_file.
    "initiative": {
        "enabled": True,
        # One cheap look every tick_s seconds.
        "tick_s": 25.0,
        # At most this many bubbles / spoken lines of her own per hour.
        "offers_per_hour": 3,
        "says_per_hour": 1,
        # Meeting signals: a Google Meet tab in a running browser (asked by
        # AppleScript; macOS asks once for Automation permission), Zoom in a
        # call, a meeting app in front.
        "meeting": {"browsers": True, "zoom": True, "front_apps": True,
                    "exit_after_s": 60, "min_minutes": 5},
        # Orca agents finishing or waiting (`orca worktree ps`, every orca_every_s).
        "orca": True,
        "orca_every_s": 30.0,
        # The morning summary: first activity of the day inside this window,
        # or back after away_hours without any.
        "morning": {"from_hour": 6, "until_hour": 12, "away_hours": 4},
        "local_file": "initiative_local.json",
    },
    "selffix": SELFFIX_DEFAULTS,
    # Always-on listening. This is the primary way to talk to Wawi;
    # push-to-talk is the fallback. Setting enabled: false reverts to
    # push-to-talk only, as does a missing openwakeword install.
    "wake": {
        "enabled": True,
        # "openwakeword" = pretrained model, six English phrases only.
        # "whisper"      = transcribe each speech segment and match `phrases`,
        #                  so the trigger can be any word in any language.
        "engine": "whisper",
        # false = always-on: any speech becomes a turn, no phrase needed.
        "require_phrase": False,
        "phrases": ["wawi"],
        # Extra spellings Whisper produces for a phrase, matched exactly:
        # {"wawi": ["guaguí"]}. Built-in ones live in wawi.wake.KNOWN_ALIASES.
        "aliases": {},
        # Decoder prompt for the trigger check. "auto" = the phrases ("Wawi,
        # Wawi."), which is what makes a made-up name transcribable; "" = none.
        "whisper_prompt": "auto",
        # Phrase mode: seconds of speech checked for the trigger before the
        # rest is captured or ignored (0 = check whole segments, the old way).
        "window_s": 1.5,
        # openwakeword engine: "" = pretrained "hey jarvis"; "wawi" = the
        # custom model in wawi/assets/wakeword/wawi.onnx (or any path).
        "model": "",
        # Phrase mode: a match only stands if the trained acoustic model
        # (wawi/assets/wakeword/<name>.onnx) also scores >= verify_threshold.
        # Cuts "Huawei"/"Wally"/"guau" wakes from ~50 % to ~5 % (measured).
        # "" = Whisper alone. Skipped silently if the file is missing.
        "verify_model": "wawi",
        "verify_threshold": 0.5,
        # openwakeword engine: consecutive 80 ms frames over wake_threshold.
        "patience": 1,
        # Where the live switch (voice / menu) keeps the chosen mode.
        "local_file": "wake_local.json",
        "whisper_model": "mlx-community/whisper-tiny-mlx",
        "pre_roll_s": 1.5,
        "silence_ms": 800,
        # Endpointing grows with the utterance: snappy for a command, patient
        # for an explanation. 0 in silence_long_ms disables the escalation.
        "silence_long_ms": 0,
        "long_after_s": 8.0,
        # Ask an end-of-turn model at a short pause instead of always waiting
        # silence_ms, which then becomes the ceiling. Off by default: it needs
        # a one-time model download (see wawi/turn.py).
        "smart_turn": False,
        "smart_turn_threshold": 0.5,
        "smart_turn_pause_ms": 300,
        "smart_turn_sure": 0.85,
        "smart_turn_recheck_ms": 900,
        "max_utterance_s": 30.0,
        "follow_up_s": 8.0,
        # Window kept open after an answer that expects NO reply. Short (or 0)
        # on purpose: holding the channel open after a plain statement means
        # any nearby speech starts a turn.
        "follow_up_idle_s": 0.0,
        # Keep the channel open after EVERY answer, not only after a question.
        # This is the "conversation" default: the wake phrase becomes the way
        # to RESTART a conversation you dropped, not a per-turn toll.
        "conversation_mode": True,
        "vad_threshold": 0.5,
        # Noise gates (wawi/listener.py). A segment opens only after
        # min_onset_ms of frames scoring >= vad_onset_threshold inside
        # onset_window_ms, and is dropped before Whisper unless it holds
        # min_voiced_ms of speech at a voiced ratio >= min_voiced_ratio.
        "vad_onset_threshold": 0.6,
        "min_onset_ms": 150,
        "onset_window_ms": 400,
        "min_voiced_ms": 250,
        "min_voiced_ratio": 0.2,
        "min_speech_level": 0.0,
        # Let the wake phrase cut into an answer already being spoken. Needs
        # engine: whisper, and is skipped for answers that say the phrase out
        # loud (no echo canceller — the mic hears the speakers).
        "barge_in": False,
        # While the assistant speaks, a frame must reach THIS amplitude to
        # count as speech. Higher than min_speech_level on purpose: the mic
        # hears the speakers, and without this the assistant transcribes its
        # own voice non-stop and falls behind realtime.
        "barge_in_level": 0.12,
        "wake_threshold": 0.5,
    },
    # Delegated work: a SECOND agent, with real tools, driven by the
    # conversational one. OFF by default and deliberately so — it is the only
    # thing in Wawi that can run a command. See wawi/task.py.
    "tasks": {
        "enabled": False,
        # Where the executor works. Empty = the directory Wawi was started in.
        "cwd": "",
        "tools": ["Bash", "Read", "Write", "Edit", "Glob", "Grep"],
        # How long a question posted to the mailbox waits for an answer before
        # the executor is told to decide alone. Generous: the user has to hear
        # it, think, and answer out loud.
        "ask_timeout_s": 300.0,
        # Shorter, because an unanswered approval DENIES. A tool call the user
        # never heard about must not sit there half-approved.
        "approval_timeout_s": 120.0,
        # A "sí" to ordinary work (an edit in the project, a build) covers the
        # rest of THAT task for this long. Risky actions ask every time anyway.
        "grant_minutes": 30.0,
    },
    "backend": "claude",
    "ollama": {
        "model": "gemma3:4b",
        "url": "http://localhost:11434/api/chat",
    },
}

_ALLOWED_BACKENDS = frozenset({"claude", "ollama"})
_ALLOWED_WAKE_ENGINES = frozenset({"openwakeword", "whisper"})


@dataclass(frozen=True)
class Config:
    """Immutable runtime configuration."""

    language: str
    assistant_name: str
    model_on_demand: str
    sample_rate: int
    push_to_talk_key: str
    stt_engine: str
    stt_model: str
    stt_initial_prompt: str
    monitor: int
    max_edge: int
    jpeg_quality: int
    tts_voice: str
    tts_voices_dir: Path
    tts_speed: float
    show_tui: bool
    sound_cues: bool
    # Speak each finished sentence as the answer is generated instead of
    # waiting for the whole thing. False restores the old wait-then-speak path.
    tts_streaming: bool = True
    # Voice character (tts.pitch_semitones / tts.brightness, or tts.preset).
    # 0 and 0 are the voice exactly as Kokoro makes it.
    tts_pitch_semitones: float = 0.0
    tts_brightness: float = 0.0
    # Floating HUD panel (macOS only; falls back cleanly when unavailable)
    show_panel: bool = True
    # Wawi, the mascot beside the panel (needs show_panel).
    show_mascot: bool = True
    # Wawi alone: the panel starts hidden at every launch and she stands on
    # her own. Implies show_mascot. The menu-bar item can still show the panel.
    mascot_only: bool = False
    # The raw ``pet:`` section (see wawi/pet/settings.py).
    board: dict = field(default_factory=dict)
    pet: dict = field(default_factory=dict)
    # The raw ``learn:`` section (see wawi/learn/settings.py).
    learn: dict = field(default_factory=dict)
    # The raw ``instinct:`` section (see wawi/instinct/settings.py).
    instinct: dict = field(default_factory=dict)
    # The raw ``games:`` section (see wawi/games/settings.py).
    games: dict = field(default_factory=dict)
    # The raw ``orca:`` section (see wawi/orca.py).
    orca: dict = field(default_factory=dict)
    shortcuts: dict = field(default_factory=dict)
    # The raw ``webvm:`` section (see wawi/webvm/settings.py).
    webvm: dict = field(default_factory=dict)
    reminders: dict = field(default_factory=dict)
    # The raw ``notes:`` section (see wawi/notes.py).
    notes: dict = field(default_factory=dict)
    # The raw ``drops:`` section (see wawi/drops.py).
    drops: dict = field(default_factory=dict)
    # The raw ``inbox:`` section (see wawi/inbox.py).
    inbox: dict = field(default_factory=dict)
    # The raw ``curiosities:`` section (see wawi/curiosities.py).
    curiosities: dict = field(default_factory=dict)
    # The raw ``music:`` section (see wawi/music.py).
    music: dict = field(default_factory=dict)
    # The raw ``initiative:`` section (see wawi/initiative/core.py).
    initiative: dict = field(default_factory=dict)
    # The raw ``selffix:`` section (see wawi/selffix/).
    selffix: dict = field(default_factory=dict)
    # Brain backend toggle: "claude" (default) or "ollama" (local)
    backend: str = "claude"
    ollama_model: str = "gemma3:4b"
    ollama_url: str = "http://localhost:11434/api/chat"
    # Always-on listening — the primary input path.
    memory_turns: int = 6
    conversation_log: Path | None = None
    recall_file: Path | None = None
    recall_turns: int = 6
    inflight_file: Path | None = None
    resume_max_age_h: float = 12.0
    dangerous_skip_permissions: bool = False
    issues_dir: Path | None = None
    screenshots_dir: str = "screenshots"
    preferences_file: Path | None = None
    issues_to_github: bool = False
    wake_enabled: bool = True
    wake_engine: str = "whisper"
    wake_require_phrase: bool = False
    wake_phrases: tuple[str, ...] = ("wawi",)
    wake_whisper_model: str = "mlx-community/whisper-tiny-mlx"
    wake_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # None = the automatic prompt from the phrases; "" = no prompt.
    wake_whisper_prompt: str | None = None
    wake_window_s: float = 1.5
    wake_model: str = ""
    wake_verify_model: str = "wawi"
    wake_verify_threshold: float = 0.5
    wake_patience: int = 1
    wake_local_file: Path | None = None
    wake_pre_roll_s: float = 1.5
    wake_silence_ms: int = 800
    wake_silence_long_ms: int = 0
    wake_long_after_s: float = 8.0
    wake_smart_turn: bool = False
    wake_smart_turn_threshold: float = 0.5
    wake_smart_turn_pause_ms: int = 300
    wake_smart_turn_sure: float = 0.85
    wake_smart_turn_recheck_ms: int = 900
    wake_max_utterance_s: float = 30.0
    wake_follow_up_s: float = 8.0
    wake_follow_up_idle_s: float = 0.0
    conversation_mode: bool = True
    wake_vad_threshold: float = 0.5
    wake_min_speech_level: float = 0.0
    wake_vad_onset_threshold: float = 0.6
    wake_min_onset_ms: int = 150
    wake_onset_window_ms: int = 400
    wake_min_voiced_ms: int = 250
    wake_min_voiced_ratio: float = 0.2
    stt_filter_junk: bool = True
    stt_min_avg_logprob: float = -1.0
    stt_max_compression_ratio: float = 2.4
    stt_max_no_speech_prob: float = 0.6
    stt_junk_min_voiced_ms: int = 700
    stt_junk_min_logprob: float = -0.45
    barge_in: bool = False
    barge_in_level: float = 0.12
    wake_threshold: float = 0.5
    # Delegated work (wawi/task.py). Disabled means the four task tools are
    # never even sent to the model.
    tasks_enabled: bool = False
    tasks_cwd: str = ""
    tasks_tools: tuple[str, ...] = ("Bash", "Read", "Write", "Edit", "Glob", "Grep")
    tasks_ask_timeout_s: float = 300.0
    tasks_approval_timeout_s: float = 120.0
    tasks_grant_minutes: float = 30.0
    # Borrowed MCP servers (wawi/connectors.py), by the name the other app
    # gives them: {name: {"enabled", "mode", "read_tools"}}. Empty = none.
    connectors: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Sensors (wawi/sensors/) and the reflex rules that turn what they notice
    # into reactions (wawi/reflexes.py). Kept as raw mappings: each sensor
    # validates its own section, so a typo in one never stops Wawi starting.
    sensors: dict[str, Any] = field(default_factory=dict)
    reflexes: dict[str, Any] = field(default_factory=dict)

    # Wawi's mind (wawi/mind/): bubbles, triage, Claude for important moments.
    # A wawi.mind.settings.MindSettings; parsed by _parse_mind.
    mind: Any = field(default_factory=lambda: _parse_mind(None))


def _parse_mind(raw: Any) -> Any:
    """The ``mind:`` section. Its schema lives with the mind, not here."""
    from wawi.mind.settings import parse

    try:
        return parse(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"config.yaml: {exc}") from exc


def _parse_connectors(raw: Any) -> dict[str, dict[str, Any]]:
    """Validate the ``connectors`` section. A wrong mode is an error, not a
    silent default: guessing "read_send" for a typo would let Wawi send."""
    from wawi.connectors import MODES, MODE_READ_ONLY

    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError("config.yaml: connectors must be a mapping of server name -> settings")
    parsed: dict[str, dict[str, Any]] = {}
    for name, section in raw.items():
        section = section or {}
        if not isinstance(section, dict):
            raise ConfigError(f"config.yaml: connectors.{name} must be a mapping")
        mode = str(section.get("mode", MODE_READ_ONLY)).strip()
        if mode not in MODES:
            raise ConfigError(
                f"config.yaml: connectors.{name}.mode={mode!r} is not valid. "
                f"Use one of: {', '.join(MODES)}"
            )
        read_tools = section.get("read_tools")
        if isinstance(read_tools, str):
            read_tools = [read_tools]
        parsed[str(name)] = {
            "enabled": bool(section.get("enabled", False)),
            "mode": mode,
            "trust_read_tools": bool(section.get("trust_read_tools", False)),
            "read_tools": (
                None if read_tools is None
                else tuple(str(t).strip() for t in read_tools if str(t).strip())
            ),
        }
    return parsed


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base* (override wins)."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load(path: Path | None = None) -> Config:
    """Load and validate config from *path* (default: ``private/config.yaml``,
    or a root ``config.yaml`` not moved there yet -- see wawi/paths.py).

    Merges file values over _DEFAULTS. Missing file raises ConfigError.
    """
    if path is None:
        path = _paths.config_path()
    if not path.exists():
        raise ConfigError(
            f"config.yaml not found at {path!r}. Create it from the template:\n"
            "    mkdir -p private && cp config.yaml.example private/config.yaml\n"
            "then adjust it and run:  python -m wawi --check"
        )

    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"config.yaml is not valid YAML: {exc}") from exc

    merged = _deep_merge(_DEFAULTS, raw)

    # Resolve the TTS voice. "auto" is the template default and follows the
    # language, because a Spanish voice reading English is the loudest way to
    # get a half-finished language switch — see wawi/lang.py.
    # Resolve tts_voices_dir
    tts_section = merged.get("tts", {})
    raw_voices_dir = tts_section.get("voices_dir")
    if raw_voices_dir:
        voices_dir = Path(raw_voices_dir).expanduser()
    else:
        voices_dir = Path("~/.local/share/kokoro").expanduser()

    # Warn about ANTHROPIC_API_KEY (detection only — stripping happens in brain.py)
    if api_key_present():
        print(
            "[wawi] WARNING: ANTHROPIC_API_KEY is set in your environment. "
            "This will divert billing to the pay-per-use API instead of your "
            "subscription credit. Unset it with:  unset ANTHROPIC_API_KEY",
            file=sys.stderr,
        )

    # Validate and resolve backend
    backend = str(merged.get("backend", "claude")).lower()
    if backend not in _ALLOWED_BACKENDS:
        raise ConfigError(
            f"config.yaml: backend={backend!r} is not valid. "
            f"Allowed values: {sorted(_ALLOWED_BACKENDS)}"
        )

    from wawi import lang as _lang

    language = str(merged.get("language", _lang.DEFAULT)).strip().lower()
    if language not in _lang.SUPPORTED:
        raise ConfigError(
            f"config.yaml: language={language!r} has no prompt, voice or spoken "
            f"strings. Supported: {list(_lang.SUPPORTED)}. Whisper understands "
            "more languages than the assistant can SPEAK, which is why this is "
            "refused instead of half-working."
        )
    user_tts = raw.get("tts") if isinstance(raw, dict) and isinstance(raw.get("tts"), dict) else {}
    tts_voice, tts_pitch, tts_brightness = _tts_character(user_tts, language)

    ollama_section = merged.get("ollama", {})
    ollama_model = str(ollama_section.get("model", "gemma3:4b"))
    ollama_url = str(ollama_section.get("url", "http://localhost:11434/api/chat"))

    raw_prefs = str(merged.get("preferences_file", "") or "").strip()
    preferences_file = _paths.resolve(raw_prefs) if raw_prefs else None

    raw_issues_dir = str(merged.get("issues_dir", "") or "").strip()
    issues_dir = _paths.resolve(raw_issues_dir) if raw_issues_dir else None

    raw_conv_log = str(merged.get("conversation_log", "") or "").strip()
    conversation_log = _paths.resolve(raw_conv_log) if raw_conv_log else None

    raw_recall = str(merged.get("recall_file", "") or "").strip()
    recall_file = _paths.resolve(raw_recall) if raw_recall else None

    raw_inflight = str(merged.get("inflight_file", "") or "").strip()
    inflight_file = _paths.resolve(raw_inflight) if raw_inflight else None

    tasks_section = merged.get("tasks", {})
    raw_task_tools = tasks_section.get("tools") or []
    if isinstance(raw_task_tools, str):
        raw_task_tools = [raw_task_tools]
    tasks_tools = tuple(str(t).strip() for t in raw_task_tools if str(t).strip())
    # An empty list is not "use the defaults": someone who wrote `tools: []`
    # asked for an executor that cannot touch anything, and silently handing it
    # a shell instead is the worst possible reading of that.
    if not raw_task_tools:
        tasks_tools = ("Bash", "Read", "Write", "Edit", "Glob", "Grep")

    wake_section = merged.get("wake", {})

    wake_engine = str(wake_section.get("engine", "whisper")).lower()
    if wake_engine not in _ALLOWED_WAKE_ENGINES:
        raise ConfigError(
            f"config.yaml: wake.engine={wake_engine!r} is not valid. "
            f"Allowed values: {sorted(_ALLOWED_WAKE_ENGINES)}"
        )

    wake_require_phrase = bool(wake_section.get("require_phrase", False))
    # The owner's live choice (voice / menu), kept outside the tracked config.
    raw_local = str(wake_section.get("local_file", "") or "").strip()
    # A bare name goes next to the config.yaml it overrides: private/ when
    # Wawi runs from the repo, and a test's own tmp dir under pytest.
    wake_local_file = _beside(path, raw_local) if raw_local else None
    from wawi import wakemode as _wakemode

    wake_require_phrase = _wakemode.read_local(wake_local_file).get(
        "require_phrase", wake_require_phrase
    )

    raw_aliases = wake_section.get("aliases") or {}
    if isinstance(raw_aliases, (list, tuple)):
        # A bare list belongs to the first phrase: the common one-name case.
        first = str((wake_section.get("phrases") or ["wawi"])[0]).strip().lower()
        raw_aliases = {first: raw_aliases}
    if not isinstance(raw_aliases, dict):
        raise ConfigError("config.yaml: wake.aliases must be a mapping {phrase: [spellings]}")
    wake_aliases = {
        str(k).strip().lower(): tuple(str(a).strip() for a in (v or []) if str(a).strip())
        for k, v in raw_aliases.items()
    }
    raw_prompt = wake_section.get("whisper_prompt", "auto")
    raw_prompt = "" if raw_prompt is None else str(raw_prompt).strip()
    wake_whisper_prompt = None if raw_prompt.lower() == "auto" else raw_prompt
    wake_window_s = float(wake_section.get("window_s", 1.5))
    if wake_window_s < 0:
        raise ConfigError("config.yaml: wake.window_s must be >= 0")

    raw_phrases = wake_section.get("phrases") or []
    if isinstance(raw_phrases, str):
        raw_phrases = [raw_phrases]
    wake_phrases = [str(p).strip() for p in raw_phrases if str(p).strip()]
    # With require_phrase: false there is no trigger stage, so an empty
    # phrase list is not a misconfiguration -- it is the point.
    if wake_require_phrase and wake_engine == "whisper" and not wake_phrases:
        raise ConfigError(
            "config.yaml: wake.engine is 'whisper' but wake.phrases is empty. "
            'Set at least one, for example:  phrases: ["wawi"]'
        )

    return Config(
        language=language,
        assistant_name=str(merged.get("assistant_name", "Wawi")).strip() or "Wawi",
        memory_turns=max(0, int(merged.get("memory_turns", 6))),
        conversation_log=conversation_log,
        recall_file=recall_file,
        recall_turns=max(0, int(merged.get("recall_turns", 6))),
        inflight_file=inflight_file,
        resume_max_age_h=max(0.0, float(merged.get("resume_max_age_h", 12.0))),
        dangerous_skip_permissions=bool(merged.get("dangerous_skip_permissions", False)),
        issues_dir=issues_dir,
        screenshots_dir=str(_paths.resolve(merged.get("screenshots_dir") or "screenshots")),
        preferences_file=preferences_file,
        issues_to_github=bool(merged.get("issues_to_github", False)),
        model_on_demand=str(merged["models"]["on_demand"]),
        sample_rate=int(merged["audio"]["sample_rate"]),
        push_to_talk_key=str(merged["audio"]["push_to_talk_key"]),
        stt_engine=str(merged["stt"]["engine"]),
        stt_model=str(merged["stt"]["model"]),
        stt_initial_prompt=str(merged["stt"].get("initial_prompt", "")),
        monitor=int(merged["screen"]["monitor"]),
        max_edge=int(merged["screen"]["max_edge"]),
        jpeg_quality=int(merged["screen"]["jpeg_quality"]),
        tts_voice=tts_voice,
        tts_voices_dir=voices_dir,
        tts_speed=float(merged["tts"].get("speed", 1.05)),
        tts_streaming=bool(merged["tts"].get("streaming", True)),
        tts_pitch_semitones=tts_pitch,
        tts_brightness=tts_brightness,
        show_tui=bool(merged["ui"]["show_tui"]),
        sound_cues=bool(merged["ui"]["sound_cues"]),
        show_panel=bool(merged["ui"].get("show_panel", True)),
        show_mascot=bool(merged["ui"].get("mascot", True))
        or bool(merged["ui"].get("mascot_only", False)),
        mascot_only=bool(merged["ui"].get("mascot_only", False)),
        board=dict(merged.get("board") or {}),
        pet=dict(merged.get("pet") or {}),
        learn=dict(merged.get("learn") or {}),
        instinct=dict(merged.get("instinct") or {}),
        games=dict(merged.get("games") or {}),
        orca=_orca_section(merged.get("orca"), path),
        shortcuts=_shortcuts_section(merged.get("shortcuts"), path),
        webvm=_mapping(merged.get("webvm"), "webvm"),
        backend=backend,
        ollama_model=ollama_model,
        ollama_url=ollama_url,
        wake_enabled=bool(wake_section.get("enabled", True)),
        wake_engine=wake_engine,
        wake_require_phrase=wake_require_phrase,
        wake_phrases=tuple(wake_phrases),
        wake_whisper_model=str(wake_section.get("whisper_model", "mlx-community/whisper-tiny-mlx")),
        wake_aliases=wake_aliases,
        wake_whisper_prompt=wake_whisper_prompt,
        wake_window_s=wake_window_s,
        wake_model=str(wake_section.get("model", "") or "").strip(),
        wake_verify_model=str(wake_section.get("verify_model", "") or "").strip(),
        wake_verify_threshold=float(wake_section.get("verify_threshold", 0.5)),
        wake_patience=max(1, int(wake_section.get("patience", 1))),
        wake_local_file=wake_local_file,
        wake_pre_roll_s=float(wake_section.get("pre_roll_s", 1.5)),
        wake_silence_ms=int(wake_section.get("silence_ms", 800)),
        wake_silence_long_ms=int(wake_section.get("silence_long_ms", 0)),
        wake_long_after_s=float(wake_section.get("long_after_s", 8.0)),
        wake_smart_turn=bool(wake_section.get("smart_turn", False)),
        wake_smart_turn_threshold=float(wake_section.get("smart_turn_threshold", 0.5)),
        wake_smart_turn_pause_ms=int(wake_section.get("smart_turn_pause_ms", 300)),
        wake_smart_turn_sure=float(wake_section.get("smart_turn_sure", 0.85)),
        wake_smart_turn_recheck_ms=int(wake_section.get("smart_turn_recheck_ms", 900)),
        wake_max_utterance_s=float(wake_section.get("max_utterance_s", 30.0)),
        wake_follow_up_s=float(wake_section.get("follow_up_s", 8.0)),
        wake_follow_up_idle_s=float(wake_section.get("follow_up_idle_s", 0.0)),
        conversation_mode=bool(wake_section.get("conversation_mode", True)),
        wake_vad_threshold=float(wake_section.get("vad_threshold", 0.5)),
        wake_min_speech_level=float(wake_section.get("min_speech_level", 0.0)),
        wake_vad_onset_threshold=float(wake_section.get("vad_onset_threshold", 0.6)),
        wake_min_onset_ms=int(wake_section.get("min_onset_ms", 150)),
        wake_onset_window_ms=int(wake_section.get("onset_window_ms", 400)),
        wake_min_voiced_ms=int(wake_section.get("min_voiced_ms", 250)),
        wake_min_voiced_ratio=float(wake_section.get("min_voiced_ratio", 0.2)),
        stt_filter_junk=bool(merged["stt"].get("filter_junk", True)),
        stt_min_avg_logprob=float(merged["stt"].get("min_avg_logprob", -1.0)),
        stt_max_compression_ratio=float(merged["stt"].get("max_compression_ratio", 2.4)),
        stt_max_no_speech_prob=float(merged["stt"].get("max_no_speech_prob", 0.6)),
        stt_junk_min_voiced_ms=int(merged["stt"].get("junk_min_voiced_ms", 700)),
        stt_junk_min_logprob=float(merged["stt"].get("junk_min_logprob", -0.45)),
        barge_in=bool(wake_section.get("barge_in", False)),
        barge_in_level=float(wake_section.get("barge_in_level", 0.12)),
        wake_threshold=float(wake_section.get("wake_threshold", 0.5)),
        tasks_enabled=bool(tasks_section.get("enabled", False)),
        tasks_cwd=str(tasks_section.get("cwd", "") or "").strip(),
        tasks_tools=tasks_tools,
        tasks_ask_timeout_s=float(tasks_section.get("ask_timeout_s", 300.0)),
        tasks_approval_timeout_s=float(tasks_section.get("approval_timeout_s", 120.0)),
        tasks_grant_minutes=float(tasks_section.get("grant_minutes", 30.0)),
        connectors=_parse_connectors(merged.get("connectors")),
        sensors=_mapping(merged.get("sensors"), "sensors"),
        reflexes=_mapping(merged.get("reflexes"), "reflexes"),
        mind=_parse_mind(merged.get("mind")),
        reminders=_reminders_section(merged.get("reminders"), path),
        notes=dict(merged.get("notes") or {}),
        drops=dict(merged.get("drops") or {}),
        inbox=dict(merged.get("inbox") or {}),
        curiosities=_curiosities_section(merged.get("curiosities"), path),
        music=_music_section(merged.get("music"), path),
        initiative=_initiative_section(merged.get("initiative"), path),
        selffix=_selffix_section(merged.get("selffix"), path),
    )


def _beside(config_path: Path, local: str) -> Path:
    """A ``local_file`` from *config_path*: absolute as written, a bare name in
    the private folder beside it (wawi/paths.py), any other relative path
    relative to the config's own folder."""
    p = Path(local).expanduser()
    if p.is_absolute():
        return p
    if len(p.parts) == 1:
        return _paths.local_dir(config_path) / p
    return config_path.parent / p


def _orca_section(raw: Any, config_path: Path) -> dict[str, Any]:
    """The ``orca:`` section, with its local_file placed next to config.yaml
    (like wake.local_file): the same place when Wawi runs from the repo, and
    a test's own tmp dir under pytest."""
    section = _mapping(raw, "orca")
    local = str(section.get("local_file", "orca_local.json") or "").strip()
    if local:
        section["local_file"] = str(_beside(config_path, local))
    return section


def _shortcuts_section(raw: Any, config_path: Path) -> dict[str, Any]:
    section = _mapping(raw, "shortcuts")
    for key in ("allow", "confirm"):
        if key in section and not isinstance(section[key], list):
            raise ConfigError(f"config.yaml: shortcuts.{key} must be a list")
    local = str(section.get("local_file", "shortcuts_local.json") or "").strip()
    if local:
        section["local_file"] = str(_beside(config_path, local))
    return section


def _reminders_section(raw: Any, config_path: Path) -> dict[str, Any]:
    """Keep the local state beside the selected config, not the process cwd."""
    section = _mapping(raw, "reminders")
    local = str(section.get("local_file", "reminders_local.json") or "").strip()
    if local:
        section["local_file"] = str(_beside(config_path, local))
    return section


def _curiosities_section(raw: Any, config_path: Path) -> dict[str, Any]:
    """Like reminders: the local state lives beside the selected config."""
    section = _mapping(raw, "curiosities")
    local = str(section.get("local_file", "curiosities_local.json") or "").strip()
    if local:
        section["local_file"] = str(_beside(config_path, local))
    return section


def _music_section(raw: Any, config_path: Path) -> dict[str, Any]:
    """Like curiosities: the switch lives beside the selected config."""
    section = _mapping(raw, "music")
    local = str(section.get("local_file", "music_local.json") or "").strip()
    if local:
        section["local_file"] = str(_beside(config_path, local))
    return section


def _initiative_section(raw: Any, config_path: Path) -> dict[str, Any]:
    """Like music: the switches and what she learned live beside the config."""
    section = _mapping(raw, "initiative")
    local = str(section.get("local_file", "initiative_local.json") or "").strip()
    if local:
        section["local_file"] = str(_beside(config_path, local))
    return section


def _selffix_section(raw: Any, config_path: Path) -> dict[str, Any]:
    """Like music: the switch and the attempt history live beside the config."""
    section = _mapping(raw, "selffix")
    local = str(section.get("local_file", "selffix_local.json") or "").strip()
    if local:
        section["local_file"] = str(_beside(config_path, local))
    return section


def _mapping(raw: Any, name: str) -> dict[str, Any]:
    """A section that must be a mapping; empty or missing reads as {}."""
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"config.yaml: {name} must be a mapping")
    return dict(raw)


def _resolve_voice(voice: str, language: str) -> str:
    """Turn ``tts.voice: auto`` into the voice that matches *language*.

    An explicit voice always wins: someone who wrote ``em_alex`` meant it, and
    silently overriding it because the language changed would be worse than
    the mismatch it prevents.
    """
    from wawi import lang as _lang

    voice = (voice or "").strip()
    if voice.lower() in ("", "auto"):
        return _lang.voice_for(language)
    return voice


def _tts_character(user_tts: dict[str, Any], language: str) -> tuple[str, float, float]:
    """Resolve (voice, pitch_semitones, brightness) from the user's tts section.

    ``tts.preset`` fills in only what the user left unset (``voice: auto``
    counts as unset); an explicit key always wins. Without a preset and
    without the new keys the result is what it always was: the resolved
    voice, 0, 0.
    """
    from wawi import voice as _voice

    preset_name = str(user_tts.get("preset") or "").strip().lower()
    if preset_name in ("", "none", "off"):
        preset: dict[str, Any] = {}
    elif preset_name in _voice.TTS_PRESETS:
        preset = dict(_voice.TTS_PRESETS[preset_name])
    else:
        raise ConfigError(
            f"config.yaml: tts.preset={preset_name!r} is not valid. "
            f"Use one of: {', '.join(sorted(_voice.TTS_PRESETS))}"
        )

    given = user_tts.get("voice")
    raw_voice = str(given).strip() if given is not None else ""
    if raw_voice.lower() in ("", "auto") and "voice" in preset:
        raw_voice = str(preset["voice"])
    elif given is None:
        raw_voice = str(_DEFAULTS["tts"]["voice"])
    voice = _resolve_voice(raw_voice, language)
    try:
        _voice.parse_voice_spec(voice)
    except ValueError as exc:
        raise ConfigError(f"config.yaml: {exc}") from exc

    def number(key: str, lo: float, hi: float) -> float:
        value = user_tts.get(key, preset.get(key, 0.0))
        try:
            num = float(value if value is not None else 0.0)
        except (TypeError, ValueError):
            raise ConfigError(f"config.yaml: tts.{key}={value!r} is not a number") from None
        if not lo <= num <= hi:
            raise ConfigError(f"config.yaml: tts.{key}={num} must be between {lo} and {hi}")
        return num

    limit = _voice.MAX_PITCH_SEMITONES
    return voice, number("pitch_semitones", -limit, limit), number("brightness", 0.0, 1.0)


def api_key_present() -> bool:
    """Return True if ANTHROPIC_API_KEY is set and non-empty in the environment."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))
