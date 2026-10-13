# Wawi

**Un asistente de escritorio para macOS que usa IA para hablar.**
*A macOS desktop assistant you talk to — and that talks back.*

![Wawi, la mascota, en reposo](docs/wawi.gif)

Wawi queda prendido mientras trabajás: le hablás, te contesta en voz alta y
por escrito en un panel flotante, puede mirar tu pantalla cuando la pregunta lo
necesita y delegar trabajo largo a un segundo agente. Al costado del panel vive
Wawi, una perrita animada que reacciona a lo que pasa.

> **Sobre el nombre.** Wawi fue la perra del autor. La mascota sale de un
> dibujo que hizo su sobrina; los cuadros animados están redibujados a partir
> de ese dibujo, en su mismo estilo.

[English below ↓](#english)

---

## Qué hace

- **Turnos por voz, sin palabra de activación.** El VAD (Silero) decide qué es
  habla; Whisper transcribe en tu Mac; el modelo contesta; Kokoro lo dice en
  voz alta. El switch `escuchando` del panel abre y cierra el micrófono.
- **Fin de turno inteligente** (opcional): Smart Turn v3 escucha si la frase
  *sonó* terminada y corta ahí, en vez de esperar siempre el silencio completo.
- **Filtro de ruido** antes y después de transcribir: aplausos, teclas y las
  alucinaciones típicas de Whisper ("Gracias.", "¡Suscríbete al canal!") no se
  convierten en turnos.
- **Mira tu pantalla** cuando hace falta (herramienta `see_screen`); solo la
  última captura viaja al modelo.
- **Tareas delegadas** a un agente ejecutor con permisos por niveles, en código
  y no en el prompt: leer nunca pregunta; el trabajo común pregunta una vez;
  borrar, mandar, pagar o `git push` preguntan siempre; Keychain, gestores de
  contraseñas, Ajustes del Sistema y `sudo` se niegan sin preguntar.
  Apagado por defecto (`tasks.enabled`).
- **Retoma lo cortado**: si se cierra o se cuelga a mitad de un turno o una
  tarea, al volver pregunta una sola vez si lo retoma.
- **Conectores MCP de solo lectura**: reusa los servidores MCP que ya tenés en
  Claude Code o Claude Desktop, apagados hasta que los nombrás en el config.
- **Copiar al portapapeles**: pedile que te deje algo copiado, o usá el botón
  del panel para copiar la última respuesta.
- **Memoria corta y durable**: ventana de los últimos turnos, un resumen
  rodante que sobrevive reinicios y preferencias que le enseñás por voz
  ("sé más breve").
- **Español e inglés** (`language: es | en`), que mueve juntos el STT, el
  prompt, la voz y las frases fijas.
- **La mascota**: respira, parpadea, escucha, habla, se rasca, cava, persigue
  su cola, se duerme… Reacciona al tacto (acariciarle la cabeza, tocarle la
  nariz o la panza, agarrarle la cola) y al volumen de tu voz. Puede vivir sola
  en una esquina sin el panel (`ui.mascot_only`).

## Requisitos

- Mac con **Apple Silicon** (mlx-whisper usa Metal) y macOS reciente.
- **Python 3.12** y [**uv**](https://docs.astral.sh/uv/).
- **Claude Code** instalado y con sesión iniciada con una suscripción
  Claude Pro/Max (`claude`, luego `/login`). Wawi usa el
  [Claude Agent SDK](https://docs.claude.com/en/docs/agent-sdk/overview) con
  esa sesión: **no hace falta API key**, y si `ANTHROPIC_API_KEY` está
  definida, `--check` te avisa (el gasto iría a la API paga).
- Permisos de macOS (Ajustes del Sistema → Privacidad y seguridad), para la
  terminal o el IDE desde donde lo corrés:
  - **Micrófono** — obligatorio.
  - **Grabación de pantalla** — para `see_screen` y las capturas.
  - **Accesibilidad** — solo para el push-to-talk (`audio.push_to_talk_key`).
- Opcional: [Ollama](https://ollama.com) para un backend 100 % local.

## Instalación

```bash
git clone <este repo> wawi && cd wawi
uv venv --python 3.12
uv pip install -r requirements.txt          # o requirements-dev.txt para correr tests

cp config.yaml.example config.yaml          # y ajustalo a tu máquina
```

Los **pesos de los modelos** no vienen en el repo ni en los paquetes. El doctor
dice exactamente qué falta:

```bash
.venv/bin/python -m wawi --check
```

Lo que normalmente pide:

```bash
# Voz (Kokoro): kokoro-v1.0.onnx y voices-v1.0.bin en ~/.local/share/kokoro
mkdir -p ~/.local/share/kokoro && cd ~/.local/share/kokoro
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
cd -

# VAD (Silero, vía openWakeWord, ~8 MB)
.venv/bin/python -c 'from wawi.listener import download_models; download_models()'

# Fin de turno inteligente (opcional, ~32 MB; después poné wake.smart_turn: true)
.venv/bin/python -c 'from wawi.turn import download_model; download_model()'
```

Whisper (`mlx-community/whisper-large-v3-turbo`, ~1.6 GB) se baja solo de
Hugging Face la primera vez que transcribe.

## Uso

```bash
.venv/bin/python -m wawi --check     # doctor: config, dependencias, modelos, permisos
.venv/bin/python -m wawi             # en primer plano
.venv/bin/python -m wawi --lang en   # un run en inglés
.venv/bin/python -m wawi --backend ollama
```

Para dejarlo corriendo en segundo plano:

```bash
scripts/wawictl start     # arranca desacoplado y te devuelve la terminal
scripts/wawictl logs      # sigue preguntas y respuestas en vivo (Ctrl+C deja de mirar, no lo apaga)
scripts/wawictl status
scripts/wawictl restart
scripts/wawictl stop
```

No hay plist de launchd ni auto-reinicio, a propósito: Wawi tiene tu
micrófono, y algo que vuelve solo con el micrófono abierto no es para
instalarlo sin querer.

Hablale y listo. Mientras habla, la tecla de push-to-talk lo interrumpe (el
micrófono está cerrado mientras habla, así no se escucha a sí mismo).

## Configuración

Todo está en `config.yaml`, documentado clave por clave en
[`config.yaml.example`](config.yaml.example). Algunas para empezar:

| Clave | Para qué |
|---|---|
| `language` | `es` o `en` |
| `wake.require_phrase` | `false` (default): todo lo que digas es un turno. `true`: hay que decir "wawi" |
| `wake.silence_ms`, `wake.smart_turn` | cuánto espera antes de contestar |
| `wake.min_speech_level` | umbral de ruido de tu cuarto (`python scripts/diag_wake.py --measure-noise`) |
| `tasks.enabled` | prende el agente ejecutor (apagado por defecto) |
| `connectors` | servidores MCP prestados, de solo lectura |
| `ui.mascot`, `ui.mascot_only` | la mascota, con o sin panel |
| `backend` | `claude` u `ollama` |

## Privacidad

- **Queda en tu Mac:** el audio (nunca sale: Whisper, el VAD, Smart Turn y la
  voz corren localmente), los logs (`wawi.log`, `answers.log`,
  `conversation.log`), la memoria (`recall.txt`, `preferences.txt`,
  `inflight.json`), las capturas guardadas y los issues que anota. Todos esos
  archivos están en `.gitignore`.
- **Va a Claude (Anthropic)**, a través de tu sesión de Claude Code: el texto
  transcripto de cada turno, la ventana de conversación reciente, la memoria
  rodante y tus preferencias, las capturas de pantalla cuando decide mirar, y
  lo que devuelvan las herramientas y los conectores que habilitaste.
- **Con `backend: ollama`** no sale nada de tu máquina (y no hay herramientas).
- **`issues_to_github`** está apagado. Si lo prendés, publica issues con
  `gh` en el repo del directorio donde arrancó Wawi — si es un repo público,
  lo que dijiste queda público.
- Con `wake.require_phrase: false`, cualquier conversación cerca del micrófono
  puede abrir un turno. Para eso está el switch del panel.

## Licencias

- **Código:** [MIT](LICENSE), © 2026 Marcos Senatori.
- **Arte de la mascota** (el dibujo, los cuadros e íconos derivados y
  `docs/wawi.gif`): [CC BY-NC 4.0](wawi/assets/LICENSE-art.md), publicado con
  permiso de su autora. Para uso comercial, reemplazalo por tu propio arte.

Modelos y paquetes de terceros (se descargan aparte; no se redistribuyen acá):

| Componente | Licencia |
|---|---|
| Kokoro-82M (voz) | Apache-2.0 |
| kokoro-onnx | MIT |
| Whisper (OpenAI) / mlx-whisper | MIT |
| Silero VAD | MIT |
| Smart Turn v3 (Pipecat / Daily) | BSD-2-Clause |
| openWakeWord (código) | Apache-2.0 |
| openWakeWord `hey_jarvis` (pesos) | **CC BY-NC-SA 4.0** (no comercial) |

Sobre `hey_jarvis`: es el modelo de palabra de activación preentrenado de
openWakeWord. **Con la configuración por defecto no se usa** (motor `whisper`,
o `require_phrase: false`): solo se carga con `wake.engine: openwakeword` y
`wake.require_phrase: true`, y entonces la frase que reconoce es "hey jarvis",
en inglés. El comando de descarga del VAD de arriba igual baja ese archivo
junto con los demás de openWakeWord. Si vas a usar Wawi con fines comerciales,
no actives ese motor.

## Créditos

- [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python) (Anthropic)
- [mlx-whisper](https://github.com/ml-explore/mlx-examples) (Apple MLX) y Whisper (OpenAI)
- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) y [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx)
- [openWakeWord](https://github.com/dscripka/openWakeWord) y [Silero VAD](https://github.com/snakers4/silero-vad)
- [Smart Turn](https://github.com/pipecat-ai/smart-turn) (Pipecat)
- El dibujo de Wawi: la sobrina del autor. Gracias.

---

<a id="english"></a>

## English

**Wawi is an always-on voice assistant for macOS.** You talk; it answers out
loud and in a floating panel, can look at your screen when the question needs
it, and can hand longer work to a second agent. An animated dog — Wawi — sits
next to the panel and reacts to what is going on. Wawi was the author's dog;
the mascot comes from a drawing by his niece.

**How it works.** Silero VAD decides what counts as speech, mlx-whisper
transcribes it locally, Smart Turn v3 (optional) detects the end of a turn, the
Claude Agent SDK produces the answer (or a local model through Ollama), and
Kokoro speaks it locally.

**Features.** Voice turns with no wake word (the panel's switch opens and
closes the mic); noise and Whisper-hallucination filtering; screen capture on
demand; delegated tasks with tiered permissions enforced in code (off by
default); resuming a turn or task cut off by a restart; borrowed, read-only
MCP connectors from Claude Code / Claude Desktop; copy to clipboard; short and
durable memory plus spoken preferences; Spanish and English; and a mascot with
idle behaviours, touch reactions (head, nose, belly, tail) and volume
reactions.

**Requirements.** Apple Silicon Mac, Python 3.12, `uv`, Claude Code logged in
with a Claude Pro/Max subscription (no API key), and macOS permissions for
Microphone, Screen Recording (screen features) and Accessibility (push-to-talk
only).

**Install and run.**

```bash
uv venv --python 3.12
uv pip install -r requirements.txt     # requirements-dev.txt adds pytest
cp config.yaml.example config.yaml
.venv/bin/python -m wawi --check       # lists every missing model and permission
.venv/bin/python -m wawi               # or: scripts/wawictl start | logs | stop
```

Model downloads (Kokoro voice files, the VAD, optional Smart Turn) are listed
in the Spanish section above; Whisper downloads itself on first use. Set
`language: en` (or pass `--lang en`) for English.

**Privacy.** Audio never leaves the Mac. Transcribed text, recent conversation,
rolling memory, preferences, screenshots the assistant decides to take, and
tool/connector results are sent to Claude through your Claude Code session.
Logs and memory files stay local and are gitignored. `issues_to_github` is off;
turning it on posts issues — quoting what you said — to the GitHub repo of the
working directory, which may be public.

**Licenses.** Code: MIT. Mascot art (drawing, derived frames and icons,
`docs/wawi.gif`): CC BY-NC 4.0, used with the artist's permission — see
`wawi/assets/LICENSE-art.md`. Third-party models: Kokoro-82M Apache-2.0,
kokoro-onnx MIT, Whisper MIT, Silero VAD MIT, Smart Turn v3 BSD-2-Clause.
openWakeWord's `hey_jarvis` weights are CC BY-NC-SA 4.0 (non-commercial); they
are **not used with the default configuration** and only load with
`wake.engine: openwakeword` plus `wake.require_phrase: true`, although the VAD
download helper fetches the file alongside the VAD.

**Tests.** `uv pip install -r requirements-dev.txt && .venv/bin/python -m pytest -q`
