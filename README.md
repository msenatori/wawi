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
  voz alta. El switch `escuchando` del panel abre y cierra el micrófono:
  silenciada solo deja de oír, sigue viviendo (pasea, reacciona, habla) con
  una marquita al lado de la cabeza. Opcional: [palabra de activación](#palabra-de-activación).
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
- **Le manda tareas a Orca**: si charlando sale una feature o un arreglo,
  "mandalo a Orca" y ella escribe la tarea, te pregunta si la manda y, con tu
  "sí", la arranca un agente de Orca en su propio worktree. Te avisa cuando
  termina, pregunta algo o falla (ver [Tareas en Orca](#tareas-en-orca)).
- **Retoma lo cortado**: si se cierra o se cuelga a mitad de un turno o una
  tarea, al volver pregunta una sola vez si lo retoma.
- **Conectores MCP de solo lectura**: reusa los servidores MCP que ya tenés en
  Claude Code o Claude Desktop, apagados hasta que los nombrás en el config.
- **Copiar al portapapeles**: pedile que te deje algo copiado, o usá el botón
  del panel para copiar la última respuesta.
- **Atajos de macOS por voz**: pedile "listá mis atajos" o "corré el atajo
  Shazam shortcut". Wawi compara el nombre con la lista real, incluso con
  acentos o una transcripción aproximada. Configurá `shortcuts.allow` para
  limitar cuáles puede correr y `shortcuts.confirm` para exigir un "sí" en tu
  próxima frase. La salida textual vuelve al modelo, recortada; la lista se
  cachea cinco minutos en `shortcuts_local.json` (ignorado por Git).
- **Memoria corta y durable**: ventana de los últimos turnos, un resumen
  rodante que sobrevive reinicios y preferencias que le enseñás por voz
  ("sé más breve").
- **Español e inglés** (`language: es | en`), que mueve juntos el STT, el
  prompt, la voz y las frases fijas.
- **La mascota**: respira, parpadea, escucha, habla, se rasca, cava, persigue
  su cola, se duerme… Reacciona al tacto (acariciarle la cabeza, tocarle la
  nariz o la panza, agarrarle la cola) y al volumen de tu voz. Puede vivir sola
  en una esquina sin el panel (`ui.mascot_only`). De vez en cuando se larga a
  hacer algo: zoomies y jadeo, saltos cada vez más altos, el paracaídas (desde
  el borde de una ventana), un hueso o una pelota en el piso, perseguir un
  pájaro, una mariposa en la nariz, una hoja que le hace estornudar, sacudirse.
  Para verlas a pedido: `WAWI_ACTIVITY=all` (o `paracaidas,pajaro`) al lanzarla.
  Mientras usa una herramienta se la ve haciéndolo: en el cohete, con la cámara,
  la libreta, la llave inglesa o la laptop (cada herramienta declara su
  `animation` en `wawi/tools.py`).
- **Vive en tu escritorio**: tiene hambre, sed, sueño y ganas de jugar (le das
  de comer desde su menú o por voz), camina por los bordes de las ventanas,
  trepa con su escalerita de madera (la apoya, sube peldaño a peldaño y la
  sube tras de sí; `WAWI_ACTIVITY=escalera` para verla), se cae si cerrás la
  ventana donde estaba y, si la tenés olvidada,
  viene a sentarse al lado de donde estás escribiendo.
- **Sentidos** (plugins, todo local): tus webs, disco y batería, procesos,
  puertos y contenedores, el clima (Open-Meteo), RSS y tus pausas. Reacciona
  sin IA: se viste de bombera si se cae una web, cava si se llena el disco,
  saca el paraguas si va a llover. Con un tope de interrupciones y silencio en
  reuniones y pantalla completa. Se configura hablándole: "vigilá example.com".
- **Aprende de vos** con numpy puro (sin GPU ni modelos grandes): qué gracias
  te gustan, qué avisos te importan y qué noticias leés. Un cerebro de unas
  miles de neuronas se reentrena solo mientras duerme.
- **Instintos**: mira cómo va tu día (qué tipo de app tenés al frente y cuándo
  tocás el teclado, nunca títulos ni contenido) y aprende tus rutinas: a qué
  hora arrancás, cuándo almorzás, cada cuánto parás, tus bloques de
  concentración. De vez en cuando hace algo por su cuenta — te saluda cuando
  volvés, se acurruca cerca mientras te concentrás, te trae la pelota si se te
  pasó la pausa, te pregunta "¿comiste?" — y aprende cuáles te gustan.
  "Wawi, ¿qué sabés de mí?" te lo cuenta; "olvidá lo que aprendiste" lo borra.
- **Juega con vos**: piedra, papel o tijera (aprende tus mañas y se agranda
  cuando te gana), ta-te-tí en un tablerito al lado de ella, adivinar el
  número (el suyo o el tuyo), buscar la pelota que le tirás con el mouse,
  escondidas detrás de tus ventanas y una trivia de perros. Por voz, con
  clics o desde el menú "Jugar". Si está aburrida o volvés de una pausa, a
  veces te propone uno (nunca en reuniones ni mientras escribís) y aprende
  cuáles y a qué hora te gustan.
- **"Andá a ver esta página"**: se sube a una nave, viaja a la web y vuelve con
  el resumen.
- **Cerebro embebido opcional** (MLX, sin servidor ni Ollama): un modelo chico
  local le da personalidad a sus globitos. Wawi funciona completa sin él.

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

Opcional, el cerebro embebido (MLX, corre dentro de Wawi, ~1 GB):

```sh
uv pip install -r requirements-cerebro.txt
# y en config.yaml:
#   mind:
#     local_model: mlx-community/Qwen3-1.7B-4bit
```

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

## Palabra de activación

Por defecto Wawi **escucha siempre**: todo lo que el VAD reconoce como habla es
un turno (`wake.require_phrase: false`). Si preferís que conteste solo cuando
la llamás, hay un modo con palabra clave, "Wawi":

- **Por voz:** "Wawi, usá palabra clave" / "escuchá siempre".
- **Desde el menú** (barra de menú o clic derecho sobre ella): "Esperar a que
  me digan «Wawi»" / "Escuchar siempre (sin palabra clave)".
- **En `config.yaml`:** `wake.require_phrase: true`.

El cambio por voz o menú es inmediato y queda guardado en `wake_local.json`
(local, en `.gitignore`), que pisa a `config.yaml` en el próximo arranque.
Borralo para volver a lo que dice `config.yaml`.

Cómo funciona el modo con palabra clave (`wake.engine: whisper`): el VAD abre
un candidato, y al juntar `wake.window_s` (1.5 s) de habla un Whisper chico
transcribe solo ese pedacito con el nombre como pista ("Wawi, Wawi."). Si
empieza con "Wawi" (y el modelo acústico entrenado para "Wawi" está de
acuerdo, ver abajo), sigue grabando y te muestra que escucha mientras hablás;
si no, descarta el resto de esa charla sin transcribirla. "Wawi, ¿qué hora es?"
de un tirón es un solo turno: el nombre se saca y se pregunta "¿qué hora es?".
"Wawi" solo contesta "¿Sí?" y te deja hablar, sin gastar un turno del modelo.

Medido con clips sintéticos de Kokoro (120 positivos, 160 negativos, voces
españolas y con acento, ruido y eco; `scripts/wakeword/measure_phrase.py`):

| Variante | Detecta "Wawi" | Falsos con palabras parecidas | Falsos con frases comunes | Costo |
|---|---|---|---|---|
| Antes (whisper-tiny sin pista) | 1.7 % | 0 % | 0 % | ~90 ms |
| Whisper con pista (decodificación propia) | 97.5 % | 52 % | 0 % | ~17 ms |
| Whisper con pista + modelo acústico (default) | 94.2 % | 4.8 % | 0 % | ~100 ms, solo si Whisper dijo que sí |
| De punta a punta (VAD + ventana + modelo) | 89.2 % | 6.3 % | 0 % | idem |

"Wawi" suena muy parecido a "Huawei", "Wally", "wow", "guay" o "Hawái", y con
la pista Whisper las escribe como "Wawi". Por eso hay una segunda opinión: un
modelo openWakeWord chiquito entrenado para "Wawi" con esas palabras como
negativos (`wake.verify_model: wawi`, `wawi/assets/wakeword/wawi.onnx`). Con
frases normales no se dispara, y cuando se equivoca con una palabra suelta lo
que pasa es un "¿Sí?", no un turno pago. Ojo: el modelo se entrenó con voces
sintéticas y nunca escuchó la tuya; si no te reconoce, poné
`wake.verify_model: ""` o reentrenalo con tus grabaciones
([`scripts/wakeword/README.md`](scripts/wakeword/README.md), unos 15 min de
CPU con los datos ya generados).

El mismo modelo puede ser el motor entero: `wake.engine: openwakeword` +
`wake.model: wawi` (sin Whisper; ~1.2 ms cada 80 ms). Medido con datos
separados (3.9 h de habla real en español e inglés como negativos): 97.4 % de
detección y 2.8 falsas por hora a 0.9; con `wake.patience: 2` y
`wake_threshold: 0.8`, 95.8 % y 1.3/h. "Huawei" todavía lo despierta. Si falta
el archivo, vuelve solo al modo whisper. Sin `model` usa el "hey jarvis"
preentrenado (inglés, licencia no comercial).

## Tareas en Orca

Cuando charlando con Wawi aparece algo para hacer en el código, se lo puede
pasar a un agente de Orca (la app de escritorio
para varios agentes a la vez), que trabaja en un worktree propio mientras vos seguís:

1. **"Wawi, mandalo a Orca"** (o "que lo haga un agente"). Ella escribe una
   tarea completa a partir de lo que hablaron: qué tocar, qué cambiar, qué no
   romper, qué puede editar el agente y cómo se comprueba que quedó. Le suma
   las reglas del repo: commits en español con la línea de co-autoría, cómo
   correr los tests, no tocar `config.yaml` ni los `*_local.json`, no hacer
   push y commitear en su rama sin mergear.
2. **Te pregunta** "Tarea «…» para jarvis. ¿La mando a Orca?". Solo un "sí"
   (o "dale") como respuesta la manda: se lee de lo que dijiste, no de lo que
   decide el modelo, porque cada tarea arranca un agente pago. "No" la descarta;
   cualquier otra cosa (por ejemplo "sí, pero cambiale el título") vuelve al
   modelo para que la rehaga.
3. **Te avisa** como mascota, con globito y voz (respetando el modo silencio):
   "¡Terminé lo de «…»! Quedó en la rama …" con festejo, o cara triste y el
   motivo si falló. Si el agente pregunta algo, Wawi te lo lee y lo que le
   contestes por voz se lo pasa.

También: **"¿cómo van las tareas?"** (o el menú "Tareas en Orca") te dice qué
hay en curso, y **"cortá la tarea de Orca"** la corta, también después de un
"sí". Por defecto la tarea va al repo de Wawi; "mandalo a Orca en bruno" la
manda a otro repo que Orca conozca.

Cómo funciona por dentro: Wawi tiene en Orca su propia terminal ("Wawi") y una
Run, guardadas en `orca_local.json` junto con las tareas en curso (si cerrás
Wawi, al volver sigue esperándolas). Cada tarea es un `orca orchestration
worker-start` con `--worktree new-top-level`, y la queda esperando con `check
--wait`. Sigue las reglas de Orca al pie de la letra: si un arranque falla te
lo dice y no lo reintenta, una espera vacía no es motivo para cortar nada, y la
terminal del agente se libera solo después de su `worker_done`. Cada tarea
mandada queda escrita en `issues/<tarea>.md`. Necesita la app Orca abierta; sin
el CLI de Orca esto se apaga solo (`orca.enabled` en `config.yaml`).

### Terminales visibles en Orca

También podés pedir «abrí una shell en Wawi» o «abrí Codex en bruno».
Wawi abre una terminal **visible en el checkout existente**, sin crear una
tarea supervisada ni un worktree nuevo. Podés pedir «listá las terminales que
abriste», «leé la terminal de Codex» o «mandale este texto a la terminal».
Antes de abrir Claude/Codex o escribir cualquier texto, Wawi espera tu «sí»
en la próxima frase; una shell vacía se abre directamente. Las terminales
abiertas quedan registradas en `orca_local.json` (gitignored). La opción
`orca.terminal_handoff_enabled` permite desactivar estas herramientas.

## Agregados (tus servicios)

Wawi puede leer servicios tuyos: una API REST o un servidor MCP. Se agregan
desde el menú (**Agregados…**) o la terminal, y valen desde la próxima
pregunta, sin reiniciar:

```bash
scripts/wawictl plugins add "Mi Tienda" --rest https://api.mitienda.com/v1 \
  --desc "pedidos y stock" --auth bearer --token      # el token se pide y va al Llavero
scripts/wawictl plugins endpoint add mi_tienda --name pedidos --path /orders \
  --desc "pedidos del día" --query estado --health
scripts/wawictl plugins add Notas --mcp-cmd "npx -y notas-mcp" --secret-env NOTAS_KEY
scripts/wawictl plugins list | show <id> | test <id> | enable <id> | disable <id> | remove <id>
```

Todo es de solo lectura salvo que actives `--allow-writes`, y aun así cada
cambio espera tu «Aceptar» en pantalla. Los tokens viven en el Llavero de
macOS (o en una variable de entorno, `--token-env VAR`); `plugins_local.json`
(gitignored) guarda solo la referencia. Con un chequeo de salud, si el
servicio se cae Wawi se pone el traje de bombero, como con una web caída.
«Wawi, ¿qué servicios tenés?» te los lista.

## Configuración

Todo está en `config.yaml`, documentado clave por clave en
[`config.yaml.example`](config.yaml.example). Algunas para empezar:

| Clave | Para qué |
|---|---|
| `language` | `es` o `en` |
| `wake.require_phrase` | `false` (default): todo lo que digas es un turno. `true`: hay que decir "wawi" (ver [Palabra de activación](#palabra-de-activación)) |
| `wake.silence_ms`, `wake.smart_turn` | cuánto espera antes de contestar |
| `wake.min_speech_level` | umbral de ruido de tu cuarto (`python scripts/diag_wake.py --measure-noise`) |
| `tasks.enabled` | prende el agente ejecutor (apagado por defecto) |
| `connectors` | servidores MCP prestados, de solo lectura |
| `orca` | tareas para agentes de Orca (ver [Tareas en Orca](#tareas-en-orca)) |
| `ui.mascot`, `ui.mascot_only` | la mascota, con o sin panel |
| `backend` | `claude` u `ollama` |

## Privacidad

- **Queda en tu Mac:** el audio (nunca sale: Whisper, el VAD, Smart Turn y la
  voz corren localmente), los logs (`wawi.log`, `answers.log`,
  `conversation.log`), la memoria (`recall.txt`, `preferences.txt`,
  `inflight.json`), las capturas guardadas y los issues que anota, lo que
  aprendió de vos (`learn_*`) y tus rutinas (`instinct_state.json`: solo qué
  tipo de app y cuándo, por hora, 8 semanas). Todos esos archivos están en
  `.gitignore`.
- **Va a Claude (Anthropic)**, a través de tu sesión de Claude Code: el texto
  transcripto de cada turno, la ventana de conversación reciente, la memoria
  rodante y tus preferencias, las capturas de pantalla cuando decide mirar, y
  lo que devuelvan las herramientas y los conectores que habilitaste.
- **Va a Orca** (local, en tu Mac) solo lo que confirmaste con un "sí": la
  tarea que Wawi escribió a partir de la charla. El agente que la hace es de
  pago y usa su propia cuenta, como cualquier agente que arrancás en Orca.
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
| `wawi/assets/wakeword/wawi.onnx` (incluido) | MIT, como el código; entrenado con Kokoro (Apache-2.0), LibriSpeech y FLEURS (CC BY 4.0). Ver [su README](wawi/assets/wakeword/README.md) |

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

## Recordatorios y rutinas

Pedile por voz «Wawi, poné un temporizador de 5 minutos para el té» o
«Wawi, recordame en 20 minutos llamar a Ana». También entiende «a las 18»
y «mañana a las 9». Podés pedirle que liste o cancele recordatorios.

Para una instrucción recurrente: «Wawi, todas las mañanas a las 9 revisá X y
contame» o «los viernes a las 18 revisá X y contame». Wawi propone la rutina;
**recién tu próxima respuesta de voz «sí» la confirma**. También podés listar
y quitar rutinas. La instrucción se ejecuta como un turno del cerebro y Wawi
te cuenta el resultado.

La agenda vive en `reminders_local.json` al lado de `config.yaml`, fuera de Git.
Si un recordatorio venció mientras Wawi estaba apagada, al arrancar avisa
«Se te pasó…». Si está escuchando o hablando, espera su turno. Los
recordatorios explícitos se anuncian aun durante el modo silencio; las
reacciones incidentales conservan su propio límite de interrupciones. Para
desactivar todo: `reminders.enabled: false` en `config.yaml`.

### La pizarra

Wawi puede abrir una tarjeta visual al lado suyo cuando una tabla, receta,
lista o cálculo se entiende mejor mirando que escuchando. También podés decir
«Wawi, mostrame una tabla de…», «mostrame la pizarra de antes» o «Wawi, cerrá la
pizarra». El contenido puede ser Markdown o HTML interactivo. Por defecto,
WebKit bloquea cargas de red y navegación externa; `board.allow_network: true`
permite recursos de red, pero no navegación fuera de la tarjeta. Los últimos
10 documentos se conservan en memoria y como HTML en `board_local/` (ignorado
por Git). Los botones permiten copiar, abrir en el navegador y guardar.
