"""Entry point for wawi.

Usage:
    python -m wawi            # run the voice assistant
    python -m wawi --check    # run doctor checks (Fase 0)
"""

import os
import sys

# Silence huggingface_hub's "Fetching N files" progress bar that mlx-whisper
# triggers on every startup (the model is cached — the bar is just TUI noise).
# Must be set before huggingface_hub is imported, so keep it at module top.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="wawi",
        description="Wawi — macOS voice assistant",
        add_help=True,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run doctor checks (Fase 0) and exit.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging and save each screenshot to disk.",
    )
    parser.add_argument(
        "--backend",
        choices=["claude", "ollama"],
        default=None,
        metavar="{claude,ollama}",
        help=(
            "Override the brain backend for this run. "
            "'claude' uses the claude-agent-sdk (default). "
            "'ollama' uses a local Ollama model (no API key required). "
            "Overrides the 'backend' key in config.yaml."
        ),
    )
    parser.add_argument(
        "--lang",
        choices=["es", "en"],
        default=None,
        metavar="{es,en}",
        help=(
            "Override the assistant's language for this run. Moves FOUR things "
            "together: what Whisper is told it is hearing, the language the "
            "system prompt orders, the Kokoro voice (when tts.voice is 'auto'), "
            "and the fixed phrases the assistant says on its own. "
            "Overrides 'language' in config.yaml."
        ),
    )
    parser.add_argument(
        "--forget",
        action="store_true",
        help=(
            "Delete the durable rolling memory (recall_file) and exit. It is a "
            "transcript of the room, so wiping it must not require talking to "
            "the assistant, or running it at all."
        ),
    )
    # Convenience alias: --ollama == --backend ollama
    parser.add_argument(
        "--ollama",
        action="store_const",
        const="ollama",
        dest="backend",
        help="Shorthand for --backend ollama.",
    )

    args = parser.parse_args()

    if args.check:
        # The overrides must reach the doctor too, or `--check --lang en`
        # cheerfully reports the Spanish voice it is NOT going to use.
        sys.exit(_doctor(lang=args.lang))

    _setup_logging(args.debug)

    from dataclasses import replace

    from wawi import config
    from wawi.orchestrator import Orchestrator

    cfg = config.load()

    if args.forget:
        # Before the Orchestrator exists, on purpose: a running Wawi holds the
        # exchanges in memory and rewrites the file at the end of every turn, so
        # deleting it underneath a live process only wipes it until the next
        # answer.
        from wawi.recall import Recall

        Recall(cfg.recall_file, max_entries=cfg.recall_turns).clear()
        print(f"Rolling memory cleared ({cfg.recall_file or 'disabled'}).")
        sys.exit(0)

    # Apply CLI overrides (they take priority over config.yaml)
    if args.backend is not None:
        cfg = replace(cfg, backend=args.backend)
    cfg = _apply_overrides(cfg, args.lang)

    app = Orchestrator(cfg, debug=args.debug)
    from wawi.confirm import AppKitPresenter, install_presenter
    install_presenter(AppKitPresenter(app.panel))
    app.run()


def _apply_overrides(cfg, lang: str | None):
    """Apply the CLI overrides to a loaded Config.

    Shared by the run path and the doctor so the two can never disagree about
    what this invocation is going to do.
    """
    from dataclasses import replace

    from wawi import lang as _lang

    if lang is not None:
        # The voice travels with the language, or --lang en answers English in
        # a Spanish voice. Only a voice the user set explicitly is kept.
        voice = cfg.tts_voice
        if voice == _lang.voice_for(cfg.language):
            voice = _lang.voice_for(lang)
        cfg = replace(cfg, language=lang, tts_voice=voice)
    return cfg


def _setup_logging(debug: bool) -> None:
    """Configure logging. Always to ./wawi.log, and to the console on --debug.

    The file handler is unconditional because the Rich Live TUI would clobber
    console output when it is enabled. Tail it from another terminal:

        tail -f wawi.log

    ``--debug`` raises the level to DEBUG, dumps each turn's screenshot to
    disk, AND mirrors the log to stderr — running with a flag named debug and
    seeing nothing on screen is the wrong surprise to hand someone who is
    trying to find out what the assistant is doing.
    """
    import logging
    import sys

    level = logging.DEBUG if debug else logging.INFO
    handlers: list[logging.Handler] = [
        logging.FileHandler("wawi.log", mode="a", encoding="utf-8")
    ]
    if debug:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(logging.Formatter("%(levelname)-7s %(name)-18s %(message)s"))
        handlers.append(console)

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )
    # Third-party debug output drowns ours; keep the noisy ones at INFO.
    # The SDK logs a full opentelemetry ImportError traceback at DEBUG on
    # every single turn; it is harmless and drowns the pipeline's own lines.
    for noisy in (
        "httpx",
        "httpcore",
        "urllib3",
        "markdown_it",
        "PIL",
        "claude_agent_sdk._internal.transport.subprocess_cli",
        "mcp.server.lowlevel.server",
        "asyncio",
    ):
        logging.getLogger(noisy).setLevel(logging.INFO)

    logging.getLogger("wawi").info("=== wawi starting (debug=%s) ===", debug)


def _doctor(lang: str | None = None) -> int:
    """Run all Fase 0 doctor checks. Returns 0 (all OK/WARN) or 1 (any FAIL).

    Args:
        lang: ``--lang`` override, applied to the loaded config so the report
            describes the run the user actually asked for.
    """
    import os
    import shutil

    from rich.console import Console

    console = Console()
    any_fail = False

    def ok(label: str, detail: str = "") -> None:
        suffix = f"  {detail}" if detail else ""
        console.print(f"[green]  OK[/green]  {label}{suffix}")

    def warn(label: str, detail: str = "") -> None:
        suffix = f"\n       {detail}" if detail else ""
        console.print(f"[yellow]WARN[/yellow]  {label}{suffix}")

    def fail(label: str, detail: str = "") -> None:
        nonlocal any_fail
        any_fail = True
        suffix = f"\n       {detail}" if detail else ""
        console.print(f"[red]FAIL[/red]  {label}{suffix}")

    console.print("\n[bold]Wawi — doctor checks[/bold]\n")

    # --- Config loading ---
    cfg = None
    try:
        from pathlib import Path
        from wawi import config as _config
        cfg = _config.load(Path("config.yaml"))
        cfg = _apply_overrides(cfg, lang)
        ok("config.yaml loaded", f"language={cfg.language}  voice={cfg.tts_voice}")
    except Exception as exc:
        fail("config.yaml", f"{exc}  →  copy config.yaml from the template")

    # --- API key guard ---
    try:
        from wawi import config as _config
        if _config.api_key_present():
            warn(
                "ANTHROPIC_API_KEY is set",
                "This will divert billing to the pay-per-use API instead of your "
                "subscription credit. Run:  unset ANTHROPIC_API_KEY",
            )
        else:
            ok("ANTHROPIC_API_KEY not set (subscription auth only)")
    except Exception as exc:
        warn("ANTHROPIC_API_KEY check", str(exc))

    # --- Subscription reachability (zero tokens) ---
    try:
        _check_subscription_auth(ok, warn, fail)
    except Exception as exc:
        fail("subscription auth", str(exc))

    # --- Dependency imports ---
    console.print()
    console.print("[bold]Dependencies[/bold]")

    _dep_import(console, ok, fail, "mlx_whisper", "mlx-whisper")
    _dep_import(console, ok, fail, "sounddevice", "sounddevice")
    _dep_import(console, ok, fail, "pynput", "pynput")
    _dep_import(console, ok, fail, "mss", "mss")
    _dep_import(console, ok, fail, "PIL", "Pillow")
    _dep_import(console, ok, fail, "yaml", "pyyaml")
    _dep_import(console, ok, fail, "rich", "rich")
    _dep_import(console, ok, fail, "anthropic", "anthropic")
    _dep_import(console, ok, fail, "claude_agent_sdk", "claude-agent-sdk")
    _dep_import(console, ok, fail, "numpy", "numpy")
    _dep_import(console, ok, fail, "soundfile", "soundfile")

    # kokoro-onnx package
    try:
        import kokoro_onnx  # noqa: F401
        ok("kokoro-onnx")
    except ImportError:
        fail("kokoro-onnx", "pip install kokoro-onnx")

    # --- Kokoro model files ---
    if cfg is not None:
        try:
            model_onnx = cfg.tts_voices_dir / "kokoro-v1.0.onnx"
            voices_bin = cfg.tts_voices_dir / "voices-v1.0.bin"
            if model_onnx.exists() and voices_bin.exists():
                ok(f"Kokoro model files  ({cfg.tts_voices_dir})")
            else:
                missing = []
                if not model_onnx.exists():
                    missing.append(str(model_onnx))
                if not voices_bin.exists():
                    missing.append(str(voices_bin))
                fail(
                    "Kokoro model files",
                    f"Missing: {', '.join(missing)}"
                    "  →  download kokoro-v1.0.onnx and voices-v1.0.bin from "
                    "https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0"
                    f"  into {cfg.tts_voices_dir}",
                )
        except Exception as exc:
            warn("Kokoro model files", str(exc))
    else:
        warn("Kokoro model files", "skipped (config not loaded)")

    # --- Audio device listing ---
    console.print()
    console.print("[bold]Audio devices[/bold]")
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        inputs = [d for d in devices if d["max_input_channels"] > 0]
        if inputs:
            for d in inputs:
                console.print(f"  mic  index={d['index']}  {d['name']}")
        else:
            warn("No microphone input devices found")
    except Exception as exc:
        warn("audio device listing", str(exc))

    console.print()
    console.print("[bold]Screen monitors[/bold]")
    try:
        import mss
        with mss.MSS() as sct:
            for i, mon in enumerate(sct.monitors):
                console.print(f"  monitor index={i}  {mon['width']}x{mon['height']}")
    except Exception as exc:
        warn("monitor listing", str(exc))

    # --- Wake word (always-listening mode) ---
    console.print()
    console.print("[bold]Wake word[/bold]")
    _check_wakeword(ok, warn)
    _check_smart_turn(ok, warn)

    # --- macOS permissions (best-effort, must not crash) ---
    console.print()
    console.print("[bold]macOS permissions[/bold]")
    _check_mic_permission(ok, warn)
    _check_screen_permission(ok, warn)
    _check_accessibility_permission(ok, warn)

    console.print()
    if any_fail:
        console.print("[red bold]Result: FAIL — fix the items above before running wawi[/red bold]")
        return 1
    else:
        console.print("[green bold]Result: OK — wawi is ready[/green bold]")
        return 0


def _check_wakeword(ok, warn) -> None:
    """Report wake-word readiness without triggering a model download.

    Missing models are a WARN, not a FAIL: Wawi still runs in push-to-talk
    mode. The point is that the user learns about it here instead of hitting a
    silent hang on first launch.
    """
    try:
        from wawi import listener
    except Exception as exc:
        warn("wawi.listener import", str(exc))
        return

    engine = _configured_wake_engine()
    if engine == "whisper":
        _check_whisper_wake(ok, warn)
        return

    try:
        import openwakeword  # noqa: F401
    except Exception:
        warn("openwakeword not installed", "always-listening disabled; push-to-talk still works")
        return

    try:
        if listener.models_ready():
            ok("wake word models present", "openWakeWord 'hey_jarvis' ready (CC BY-NC-SA 4.0)")
        else:
            warn(
                "wake word models missing",
                "run: python -c 'import openwakeword.utils as u; u.download_models()'",
            )
    except Exception as exc:
        warn("wake word model probe", str(exc))

    _check_onnxruntime_version(ok, warn)


def _check_smart_turn(ok, warn) -> None:
    """Report the end-of-turn model when wake.smart_turn is on. Never downloads.

    A WARN, not a FAIL: without the model the listener simply waits the full
    wake.silence_ms, as it always did.
    """
    try:
        from wawi import config

        if not config.load().wake_smart_turn:
            return
        from wawi import turn
    except Exception as exc:
        warn("smart turn probe", str(exc))
        return
    if turn.model_ready():
        ok("smart turn model present", f"{turn.MODEL_FILE} ({turn.MODEL_LICENSE})")
    else:
        warn(
            "smart turn model missing — endpointing waits the full wake.silence_ms",
            "download it once (~32 MB):  python -c 'from wawi.turn import "
            "download_model; download_model()'",
        )


def _configured_wake_engine() -> str:
    """Read wake.engine from config.yaml, defaulting to openwakeword."""
    try:
        from wawi import config

        return config.load().wake_engine
    except Exception:
        return "openwakeword"


def _check_vad(ok, warn) -> None:
    """Report the Silero VAD weights, which every listening mode needs.

    The VAD is openwakeword's silero_vad.onnx EVEN IN WHISPER MODE and in
    always-on mode — those replace the trigger stage, not the speech detector.
    Skipping this check once produced a clean "Result: OK" followed by a silent
    fallback to push-to-talk at runtime. Only the VAD file is required here:
    the pretrained wake-word files are irrelevant without that engine.
    """
    try:
        from wawi import listener as _listener

        missing = [
            p for p in _listener.missing_model_files()
            if "silero" in os.path.basename(p).lower()
        ]
        if not missing:
            ok("VAD model present  (silero_vad.onnx)")
        else:
            warn(
                "VAD model missing — Wawi CANNOT listen hands-free",
                "Wawi will fall back to push-to-talk. Download it once (~8 MB):"
                "  python -c 'from wawi.listener import download_models; "
                "download_models()'",
            )
    except Exception as exc:
        warn("VAD model probe", str(exc))


def _check_whisper_wake(ok, warn) -> None:
    """Report readiness of the transcription-based wake engine.

    Only the VAD comes from openwakeword in this mode; the trigger itself is
    Whisper, so a missing wake-word model file is irrelevant here.
    """
    from wawi import config

    cfg = config.load()
    if not (cfg.wake_require_phrase or cfg.barge_in):
        ok(
            "always-on: any speech starts a turn",
            "no trigger model is loaded at all — the panel switch is what "
            "opens and closes the microphone",
        )
        _check_vad(ok, warn)
        return

    try:
        from wawi.wake import WhisperWakeDetector

        detector = WhisperWakeDetector(
            cfg.wake_phrases, model=cfg.wake_whisper_model, language=cfg.language
        )
    except Exception as exc:
        warn("whisper wake engine misconfigured", str(exc))
        return

    if cfg.wake_require_phrase:
        ok(f"wake engine: whisper  {detector.phrases}", f"model {detector.model}")
    elif cfg.barge_in:
        ok(
            "always-on: any speech starts a turn",
            f"{detector.phrases} are used only to recognise the assistant's own "
            "voice coming back while barge-in keeps the mic open",
        )
    else:
        ok(
            "always-on: any speech starts a turn",
            "no trigger model is loaded at all — the panel switch is what "
            "opens and closes the microphone",
        )

    try:
        import mlx_whisper  # noqa: F401

        ok("mlx-whisper installed", "trigger transcription available")
    except ImportError:
        warn(
            "mlx-whisper not installed",
            "the whisper wake engine cannot run:  pip install mlx-whisper",
        )

    _check_vad(ok, warn)


# onnxruntime releases known to break openwakeword inference. On these the
# models load and run, the features look sane, and EVERY wake model scores
# ~0.0000 — no error, no warning, detection just never fires. Verified against
# openWakeWord's own ground-truth sample, which scores 1.0000 on 1.28.0.
_BROKEN_ONNXRUNTIME = {"1.27.0"}


def _check_onnxruntime_version(ok, warn) -> None:
    """Catch onnxruntime builds that silently reduce every wake score to zero."""
    try:
        import onnxruntime
    except Exception:
        warn("onnxruntime not installed", "the wake word cannot run without it")
        return

    version = getattr(onnxruntime, "__version__", "unknown")
    if version in _BROKEN_ONNXRUNTIME:
        warn(
            f"onnxruntime {version} SILENTLY BREAKS the wake word",
            "every wake score becomes 0.0000 and detection never fires. "
            "Fix:  pip install 'onnxruntime>=1.28.0'",
        )
    else:
        ok(f"onnxruntime {version}", "not a known-broken build")


def _dep_import(console, ok, fail, module_name: str, display_name: str) -> None:
    try:
        __import__(module_name)
        ok(display_name)
    except ImportError as exc:
        fail(display_name, str(exc))


def _check_subscription_auth(ok, warn, fail) -> None:
    import os
    import shutil
    from pathlib import Path

    # 1. OAuth token in env
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        ok("subscription auth  (CLAUDE_CODE_OAUTH_TOKEN present)")
        return

    # 2. ~/.claude.json credential file
    claude_json = Path.home() / ".claude.json"
    if claude_json.exists():
        try:
            import json
            data = json.loads(claude_json.read_text())
            # presence of oauthToken key is enough; don't validate value
            if "oauthToken" in data or "oauth_token" in data:
                ok("subscription auth  (~/.claude.json has OAuth token)")
                return
            # file exists but no recognizable token key — fall through
        except Exception:
            pass
        ok("subscription auth  (~/.claude.json found)")
        return

    # 3. claude binary on PATH
    if shutil.which("claude"):
        fail(
            "subscription auth",
            "claude CLI found but no credential file.  Run:  claude login  (Pro/Max required)",
        )
    else:
        fail(
            "subscription auth",
            "claude CLI not found.  Install Claude Code CLI and run:  claude login",
        )


def _check_mic_permission(ok, warn) -> None:
    try:
        import sounddevice as sd
        stream = sd.InputStream(samplerate=16000, channels=1, dtype="float32")
        stream.start()
        stream.read(160)
        stream.stop()
        stream.close()
        ok("Microphone  (System Settings → Privacy & Security → Microphone)")
    except Exception:
        warn(
            "Microphone  — permission may be missing",
            "System Settings → Privacy & Security → Microphone → grant Terminal / your IDE",
        )


def _check_screen_permission(ok, warn) -> None:
    try:
        import mss
        with mss.MSS() as sct:
            sct.grab(sct.monitors[1])
        ok("Screen Recording  (System Settings → Privacy & Security → Screen Recording)")
    except Exception:
        warn(
            "Screen Recording  — permission may be missing",
            "System Settings → Privacy & Security → Screen Recording → grant Terminal / your IDE",
        )


def _check_accessibility_permission(ok, warn) -> None:
    """Report whether this process may monitor keyboard events.

    NOTE the obvious check is wrong: ``keyboard.Listener.running`` is True as
    soon as its thread starts, whether or not macOS trusts the process, so it
    reports OK while every key event is silently discarded. Ask the OS instead
    — ``AXIsProcessTrusted()`` is the authority, and pynput's own
    ``IS_TRUSTED`` flag is the fallback when pyobjc is not installed.
    """
    try:
        trusted: bool
        try:
            from ApplicationServices import AXIsProcessTrusted

            trusted = bool(AXIsProcessTrusted())
        except ImportError:
            from pynput import keyboard

            trusted = bool(getattr(keyboard.Listener, "IS_TRUSTED", False))

        if trusted:
            ok("Accessibility  (System Settings → Privacy & Security → Accessibility)")
        else:
            warn(
                "Accessibility  — NOT granted; push-to-talk will not respond",
                "System Settings → Privacy & Security → Accessibility → add Terminal / "
                "your IDE, then restart it. The wake word does not need this.",
            )
    except Exception:
        warn(
            "Accessibility  — could not probe",
            "System Settings → Privacy & Security → Accessibility → grant Terminal / your IDE",
        )


if __name__ == "__main__":
    main()
