# Wawi

**A voice assistant for your Mac that lives on your desktop as a dog.**

![Wawi, the mascot, at rest](docs/wawi.gif)

Talk to Wawi the way you'd talk to someone sitting next to you. She answers out
loud, can look at your screen, browse the web on her own little computer, hand
coding work to agents, take your notes and remind you of things. When there's
nothing to do she does what a real dog does: she naps, keeps half an eye on you,
greets you when you come back and sometimes brings you the ball.

Speech is recognised and spoken on your Mac. The thinking runs on Claude,
through the Claude Code subscription you already have, so you don't need an API key.

> **About the name.** Wawi was the author's dog. The mascot comes from a
> drawing by the author's niece; every animation frame is redrawn from it, in
> the same crayon style.

---

## Highlights

- **Just talk.** No button, no wake word, unless you want one. Speech is
  detected, transcribed and spoken locally, and she knows when you've finished
  a sentence rather than waiting for a long silence.
- **She sees what you see.** "Look at my screen and explain this error." She
  takes a screenshot only when the question needs one.
- **Drop it in her mailbox.** A tiny floating *Wawi investiga* mailbox takes
  anything you drag onto it, such as a screenshot, a file, a link or some text.
  Type what you want and she gets on it. Double-click the mailbox to type a
  question.
- **Her own computer.** Wawi can browse the web for you in a sandboxed Chromium
  running in Docker. A small live view slides out next to her and expands if
  you want to watch. She asks first before logging in, buying, paying, posting
  or sending anything.
- **She gets work done.** Ask her to send a feature to an
  Orca agent and she writes the task. With
  your "yes" she starts the agent in its own worktree and tells you when it's
  done. She can also open terminals and run macOS Shortcuts, and she handles
  timers, reminders, recurring routines, dictation and visual cards ("show me a
  table of…").
- **A dog, not a cartoon.** Her behaviour follows how dogs really spend their day:
  - She sleeps a lot and rests with her eyes on you.
  - She gets used to the things that happen all the time.
  - How she greets you depends on how long you were gone.
  - She plays only now and then.

  She also gets hungry and thirsty, walks along your window edges and climbs them
  with her little wooden ladder.
- **Shareable personalities.** How she lives is a YAML file, either *calm dog*,
  *attentive dog* or *puppy*, or one you write and share. Switch between them
  from the menu or by voice.
- **She learns you.** Small on-device models (pure numpy) learn which tricks
  you enjoy, which alerts matter and your daily rhythm. She never reads window
  titles or content.
- **She fixes herself.** Wawi can work on her own recurring problems in a
  separate worktree. Each fix must pass the full test suite, a secret scan and a
  second-opinion review. If something breaks after she applies it, it's
  reverted automatically.
- **Private by design.** Everything personal lives in one gitignored folder,
  `private/`. Audio never leaves your Mac.

## What she can do

### Talk

- **Voice turns.**
  - Silero VAD decides what is speech and Whisper transcribes it on your Mac.
  - Claude answers, and Kokoro speaks the answer, sentence by sentence as it
    streams.
  - She starts talking before the answer is finished.
- **Smart end of turn** (optional): Smart Turn v3 hears whether a sentence
  *sounded* finished.
- **Noise filtering.** Claps, typing and Whisper's classic hallucinations
  ("Thanks for watching!") never become turns.
- **Optional wake word "Wawi"**, switchable by voice or from the menu (see
  [Wake word](#wake-word)).
- **Spanish and English** (`language: es | en`). Speech recognition, prompt,
  voice and fixed phrases all switch together.
- **Memory.** She keeps a window of recent turns and a rolling summary that
  survives restarts. You can also teach her preferences by voice ("be
  briefer").
- **She resumes what was cut.** If she's closed in the middle of a turn or
  task, she offers to pick it up again.

### See and hand things over

- **See your screen** (`see_screen`). Only the latest screenshot travels to the
  model.
- **The mailbox.** Drag things onto the floating mailbox and type the order.
  Paste images with ⌘V. Toggle it from the menu bar.
- **Drag things onto Wawi herself** and *say* what to do with them. She reads
  text, files and PDFs, sees pictures and transcribes audio files locally.
- **Clipboard.** "Copy that" leaves her answer on your clipboard.

### Her own computer (web)

- **What it is:** a Chromium driven by Playwright on Xvfb, running in a Docker
  container on any Docker host on your network. It's built and managed with
  `scripts/wawictl webvm`.
- **What she can do there:** open pages, read them, click, type and scroll,
  then come back with the answer.
- **The live view:** noVNC runs in a small window next to her, and you can
  expand it.
- **Safety:**
  - The controls are protected by a token and a VNC password stored in the
    macOS Keychain.
  - Your LAN is blocked unless you allow it.
  - Cookies are thrown away unless you choose to keep them.
  - Any login, purchase, payment, message, post or personal data waits for
    *your* spoken yes. The model can't approve it on your behalf.

### Get work done

- **Orca tasks.** "Wawi, send it to Orca."
  - She turns the conversation into a complete task brief.
  - She asks you once, and only your spoken "yes" starts the paid agent.
  - Then she waits on it and tells you when it finishes, fails or has a
    question.
  - She can also open visible Claude, Codex or shell terminals in a project.
- **Delegated tasks** (off by default) run on an executor agent, with
  permissions enforced in code:
  - Reading never asks.
  - Ordinary work asks once.
  - Deleting, sending, paying and `git push` always ask.
  - The Keychain, password managers, System Settings and `sudo` are always
    refused.
- **macOS Shortcuts by voice.** "List my shortcuts" or "run Shazam". You can
  limit which ones she may run.
- **Timers, reminders and routines.**
  - "Remind me in 20 minutes to call Ana."
  - "Every morning at 9 check X and tell me." A routine only starts after
    your spoken yes.
- **Notes.** "Take a note" turns her into a dictation recorder:
  - No model is involved.
  - The text is written to a Markdown file line by line, so a crash loses at
    most one sentence.
  - It ends with a giant STOP button or when you say "stop".
- **The board.** Tables, recipes, lists and calculations show up as a visual
  card next to her, in Markdown or sandboxed interactive HTML.
- **Your own services.** Plug in a REST API or an MCP server from the menu
  (*Agregados…*) or with `wawictl plugins`.
  - They're read-only by default.
  - Tokens are kept in the Keychain.
  - A health check puts her in a firefighter suit when one goes down.
- **Borrowed MCP connectors** from Claude Code or Claude Desktop, read-only and
  opt-in.

### A real dog

- **A state machine based on research into dog behaviour:**
  - It has deep sleep, light doze, resting alert, active and play states.
  - She gets used to stimuli that repeat.
  - Greetings scale with how long you were away.
  - She asks you for things only rarely.
- **Personalities** are YAML files in [`personalities/`](personalities/README.md).
  Change one from the menu (*Personalidad*) or by voice. Your choice is
  remembered.
- **Needs:** food, water, sleep, fun and affection. You can feed her from the
  menu or by voice. Nothing bad happens if you forget.
- **Desktop life:**
  - She walks along window edges and climbs with her ladder.
  - She falls when you close the window she was standing on.
  - She comes to sit next to where you're typing.
- **Touch:** pet her head, boop her nose, rub her belly or grab her tail.
- **Games:** rock-paper-scissors (she learns your habits), tic-tac-toe,
  guess-the-number, fetch with your mouse, hide-and-seek behind your windows
  and dog trivia.
- **Senses** (all local plugins):
  - your websites, disk and battery, processes, ports and containers;
  - the weather, RSS feeds and your breaks.

  She reacts without AI: she wears a firefighter suit when a site is down,
  digs when the disk fills up and opens an umbrella when rain is coming.
- **Optional local brain** (MLX, about 1 GB): a small on-device model gives
  her speech bubbles some personality. Wawi works fully without it.

## Requirements

- An **Apple Silicon** Mac (mlx-whisper uses Metal) with a recent macOS.
- **Python 3.12** and [**uv**](https://docs.astral.sh/uv/).
- **Claude Code**, logged in with a Claude Pro/Max subscription (`claude`, then
  `/login`). Wawi uses the
  [Claude Agent SDK](https://docs.claude.com/en/docs/agent-sdk/overview) on
  that session, so **no API key is needed**. If `ANTHROPIC_API_KEY` is set,
  `--check` warns you, because usage would go to the paid API.
- macOS permissions (System Settings → Privacy & Security) for the terminal or
  IDE you run her from:
  - **Microphone** (required);
  - **Screen Recording**, for `see_screen` and screenshots;
  - **Accessibility**, only for push-to-talk.
- Optional:
  - [Ollama](https://ollama.com) for a 100% local backend;
  - a Docker host for her own computer;
  - Orca for agent tasks.

## Install

```bash
git clone https://github.com/msenatori/wawi.git && cd wawi
uv venv --python 3.12
uv pip install -r requirements.txt          # requirements-dev.txt adds pytest

mkdir -p private
cp config.yaml.example private/config.yaml  # then adjust it to your machine
```

Model weights are not included in the repo. The doctor tells you exactly what
is missing:

```bash
.venv/bin/python -m wawi --check
```

What it usually asks for:

```bash
# Voice (Kokoro): kokoro-v1.0.onnx and voices-v1.0.bin in ~/.local/share/kokoro
mkdir -p ~/.local/share/kokoro && cd ~/.local/share/kokoro
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
cd -

# VAD (Silero, via openWakeWord, ~8 MB)
.venv/bin/python -c 'from wawi.listener import download_models; download_models()'

# Smart end of turn (optional, ~32 MB; then set wake.smart_turn: true)
.venv/bin/python -c 'from wawi.turn import download_model; download_model()'
```

Whisper (`mlx-community/whisper-large-v3-turbo`, about 1.6 GB) downloads itself
from Hugging Face the first time it transcribes.

Optional local brain:

```sh
uv pip install -r requirements-cerebro.txt
# then in private/config.yaml:
#   mind:
#     local_model: mlx-community/Qwen3-1.7B-4bit
```

Optional, her own computer: set `webvm.enabled: true`, set the Docker
`context`, the `host` and the `ssh_host` in `private/config.yaml`, then:

```bash
scripts/wawictl webvm build    # builds the image on that host
```

## Run

```bash
.venv/bin/python -m wawi --check     # doctor: config, dependencies, models, permissions
.venv/bin/python -m wawi             # in the foreground
.venv/bin/python -m wawi --lang en   # one run in English
.venv/bin/python -m wawi --backend ollama
```

To keep her running in the background:

```bash
scripts/wawictl start | status | restart | stop
scripts/wawictl logs      # follow questions and answers live (Ctrl+C stops watching, not Wawi)
```

There is deliberately no launchd plist and no auto-restart. Wawi holds your
microphone, and something that comes back on its own with the mic open is not
something you should install by accident.

## Wake word

By default Wawi **always listens**: anything the VAD recognises as speech
becomes a turn. If you prefer her to answer only when called, there is a "Wawi"
wake-word mode. You can turn it on in three ways:

- by voice, by asking her to use the wake word or to always listen;
- from the menu;
- with `wake.require_phrase: true`.

A choice made by voice or from the menu is saved in `private/wake_local.json`.

In wake-word mode:

1. A small Whisper, primed with the name, transcribes the first 1.5 s.
2. A tiny openWakeWord model trained on "Wawi" gives a second opinion. It was
   trained with sound-alikes ("Huawei", "Wally", "wow", "Hawaii") as negatives.
3. "Wawi, what time is it?" in one breath is a single turn. "Wawi" on its own
   gets a quick "yes?".

Measured on synthetic Kokoro clips (`scripts/wakeword/measure_phrase.py`):

| Variant | Detects "Wawi" | False hits on sound-alikes | False hits on common phrases | Cost |
|---|---|---|---|---|
| Whisper with name hint | 97.5 % | 52 % | 0 % | ~17 ms |
| Whisper + acoustic model (default) | 94.2 % | 4.8 % | 0 % | ~100 ms, only after Whisper says yes |
| End to end (VAD + window + model) | 89.2 % | 6.3 % | 0 % | same |

The acoustic model was trained on synthetic voices. To teach it yours, record
yourself and retrain (about 15 minutes of CPU). See
[`scripts/wakeword/README.md`](scripts/wakeword/README.md).

## Configuration

Everything lives in `private/config.yaml`.
[`config.yaml.example`](config.yaml.example) documents every key. A few to
start with:

| Key | What it does |
|---|---|
| `language` | `es` or `en` |
| `wake.require_phrase` | `false` (default): everything you say is a turn |
| `wake.silence_ms`, `wake.smart_turn` | how long she waits before answering |
| `pet.personality` | `perra_tranquila` (calm, default), `perra_atenta` (attentive), `cachorra` (puppy), or your own |
| `inbox.enabled` | the *Wawi investiga* mailbox |
| `webvm` | her own computer (off by default) |
| `tasks.enabled` | the executor agent (off by default) |
| `orca` | tasks for Orca agents |
| `connectors` | borrowed read-only MCP servers |
| `ui.mascot`, `ui.mascot_only` | the mascot, with or without the panel |
| `backend` | `claude` or `ollama` |

## Privacy

**Stays on your Mac, all of it in `private/`, which is gitignored:**

- The audio. Whisper, the VAD, Smart Turn and the voice all run locally.
- Logs, transcripts, memory and saved screenshots.
- What she learned about you, and your routines. Routines are recorded only
  as *what kind* of app and *when*, never titles or content.
- Your config and every `*_local.json`.

Backing up means copying that one folder.

**Sent to Claude (Anthropic)** through your Claude Code session:

- each turn's transcribed text, the recent conversation, the rolling memory
  and your preferences;
- screenshots, when she decides to look;
- tool and connector results.

**Goes to Orca** (local) only what you confirmed with a "yes": the task she
wrote. The agent that runs it is paid and uses its own account.

**Her own computer** runs on your Docker host. Pages she visits see that
machine's connection, not your Mac's.

**With `backend: ollama`**, nothing leaves your machine (and there are no tools).

With `wake.require_phrase: false`, any conversation near the microphone can
start a turn. That's what the panel's mic switch is for.

## Licenses

- **Code:** [MIT](LICENSE), © 2026 Marcos Senatori.
- **Mascot art** (the drawing, derived frames and icons, and `docs/wawi.gif`):
  [CC BY-NC 4.0](wawi/assets/LICENSE-art.md), published with the artist's
  permission. For commercial use, replace it with your own art.

Third-party models and packages are downloaded separately and not
redistributed here:

| Component | License |
|---|---|
| Kokoro-82M (voice) | Apache-2.0 |
| kokoro-onnx | MIT |
| Whisper (OpenAI) / mlx-whisper | MIT |
| Silero VAD | MIT |
| Smart Turn v3 (Pipecat / Daily) | BSD-2-Clause |
| openWakeWord (code) | Apache-2.0 |
| openWakeWord `hey_jarvis` (weights) | **CC BY-NC-SA 4.0** (non-commercial) |
| Playwright / Chromium (her computer's image) | Apache-2.0 / BSD |
| `wawi/assets/wakeword/wawi.onnx` (included) | MIT, like the code. Trained with Kokoro (Apache-2.0), LibriSpeech and FLEURS (CC BY 4.0); see [its README](wawi/assets/wakeword/README.md) |

`hey_jarvis` is **not used with the default configuration**. It only loads
with `wake.engine: openwakeword` and no custom model, but the VAD download
helper fetches it along with the other openWakeWord files. Don't enable that
engine for commercial use.

## Tests

```bash
uv pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

## Credits

- [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python) (Anthropic)
- [mlx-whisper](https://github.com/ml-explore/mlx-examples) (Apple MLX) and Whisper (OpenAI)
- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) and [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx)
- [openWakeWord](https://github.com/dscripka/openWakeWord) and [Silero VAD](https://github.com/snakers4/silero-vad)
- [Smart Turn](https://github.com/pipecat-ai/smart-turn) (Pipecat)
- [Playwright](https://playwright.dev) and [noVNC](https://novnc.com)
- The Wawi drawing: the author's niece. Thank you.
