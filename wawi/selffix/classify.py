"""Does a fix need the owner? The rule he gave, made mechanical.

    "Todo sola, salvo que sea algo que me involucre o necesite ayuda de mi
    parte; su objetivo es serme útil."

Three answers:

- ``auto``: an internal fix or a capability that only changes HER behaviour.
  She applies it and he reads about it in the day's summary.
- ``ask``: it involves him. It changes how she deals with him (talking more,
  new proactive interruptions, permissions, what data leaves the Mac or goes
  to Claude or a third party), or touches his accounts, services, agregados,
  credentials or other repositories, or adds a dependency. She builds it and
  OFFERS it (bubble + the diff on the board, Aplicar / Descartar).
- ``blocked``: it cannot be done without him (his voice recorded, a macOS
  permission granted, something installed, a token, a decision between
  options). She does not attempt it; she files a short friendly request the
  initiative layer can bring up later (:func:`owner_request`).

Conservative by construction: ``auto`` needs EVERY signal to be clean (every
path on the internal list, no sensitive pattern in the added lines, the agent
declaring no owner impact). Any doubt is ``ask``.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field

from wawi.selffix.issues import IssueRecord, plain

AUTO, ASK, BLOCKED = "auto", "ask", "blocked"

# --------------------------------------------------------------------------- #
# Before the attempt: is it blocked on the owner?
# --------------------------------------------------------------------------- #

#: need -> (words in the issue that mean it, the request she makes)
_NEEDS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("voice", (
        "solo tu voz", "tu voz", "su voz", "mi voz", "verificar hablante", "hablante",
        "reentren", "re-entren", "entrenar la palabra", "entrenar el wake", "wake word",
        "wakeword", "grabaciones", "grabar muestras",
    ), "Para dejar de ignorarte necesito que grabes 50 veces «Wawi», ¿lo hacemos?"),
    ("permission", (
        "permiso de macos", "permisos de macos", "accesibilidad", "grabacion de pantalla",
        "privacidad y seguridad", "tcc", "permiso del microfono", "permiso de la camara",
        "full disk", "acceso total al disco",
    ), "Para esto necesito que me des un permiso en Ajustes del Sistema, ¿te muestro cuál?"),
    ("install", (
        "instalar", "instalacion", "brew install", "descargar el modelo", "descargar un modelo",
        "pip install",
    ), "Para esto hay que instalar algo en tu Mac, ¿lo vemos juntos?"),
    ("token", (
        # Not a bare "token": issues talk about the model's tokens (cost,
        # context) all the time, and that blocked them as if they needed
        # a login.
        "un token", "tu token", "token de acceso", "token de api", "access token",
        "api key", "apikey", "clave de api", "credencial", "contrasena", "iniciar sesion",
        "tu login", "loguearte", "oauth",
    ), "Para esto necesito que me pases un acceso (un token o un login), ¿lo cargamos?"),
    ("decision", (
        "elegir entre", "decidir entre", "cual preferis", "que preferis", "a o b",
        "no se si", "opciones:",
    ), "Para esto necesito que elijas cómo lo querés, ¿te muestro las opciones?"),
)


@dataclass(frozen=True)
class Precheck:
    verdict: str            # "work" or BLOCKED
    need: str = ""
    request: str = ""


def precheck(issue: IssueRecord) -> Precheck:
    """Blocked on the owner, or workable? From the issue's own words."""
    low = plain(issue.text())
    for need, words, request in _NEEDS:
        # Whole words: "tcc" inside "tccutil" or "contrasena" inside
        # "gestores de contrasenas" describe the code, not something he owes.
        if any(re.search(rf"(?<!\w){re.escape(plain(w))}(?!\w)", low) for w in words):
            return Precheck(BLOCKED, need, request)
    return Precheck("work")


def owner_request(issue: IssueRecord, check: Precheck) -> dict:
    """The request she files instead of attempting a blocked issue."""
    return {"id": issue.id, "issue": issue.title, "need": check.need, "text": check.request}


# --------------------------------------------------------------------------- #
# After the attempt: may she apply it alone?
# --------------------------------------------------------------------------- #

#: Paths that are hers alone: changing them changes nothing he deals with.
INTERNAL_PATHS = (
    "tests/*", "tests/**",
    "wawi/*.py", "wawi/**/*.py",
    "wawi/assets/**",
    "docs/*", "docs/**",
)

#: Paths that change how she deals with him, or reach his accounts and
#: services, even though they are "her" code. Checked before INTERNAL_PATHS.
OWNER_PATHS: tuple[tuple[str, str], ...] = (
    ("wawi/plugins.py", "agregados"), ("wawi/plugins_ui.py", "agregados"),
    ("wawi/connectors.py", "conectores"), ("wawi/orca.py", "Orca y otros repos"),
    ("wawi/shortcuts.py", "atajos de macOS"), ("wawi/task.py", "permisos de comandos"),
    ("wawi/guards.py", "permisos"), ("wawi/confirm.py", "confirmaciones"),
    ("wawi/brain.py", "lo que se manda a Claude"), ("wawi/prompts.py", "cómo te habla"),
    ("wawi/petvoice.py", "cuánto habla"), ("wawi/voice.py", "su voz"),
    ("wawi/reflexes.py", "interrupciones"), ("wawi/curiosities.py", "interrupciones"),
    ("wawi/mind/*", "interrupciones"), ("wawi/initiative/*", "interrupciones"),
    ("wawi/sensors/*", "lo que vigila"), ("wawi/screen.py", "capturas de pantalla"),
    ("wawi/fetch*", "datos que salen"), ("wawi/images.py", "datos que salen"),
    ("wawi/reminders/*", "tus recordatorios"), ("wawi/notes*.py", "tus notas"),
    ("wawi/selfimprove.py", "lo que se publica en GitHub"),
    ("wawi/wake.py", "cuándo te escucha"), ("wawi/listener.py", "cuándo te escucha"),
    ("wawi/speechgate.py", "cuándo te escucha"),
    ("requirements*.txt", "instalar dependencias"), ("pyproject.toml", "instalar dependencias"),
    ("setup.py", "instalar dependencias"), ("scripts/*", "scripts"),
    ("config.yaml.example", "configuración"), ("README.md", "documentación pública"),
)

#: Added lines that mean data leaves, she speaks/interrupts more, or a
#: permission/credential is involved. Matched on ``+`` lines only.
OWNER_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\burllib\b|\brequests\.|\bhttpx\b|\bhttp\.client\b|\bsocket\.|https?://", "red / datos que salen"),
    (r"\banthropic\b|claude_agent_sdk|\bopenai\b", "datos a un modelo"),
    (r"\bcurl\b|\bwget\b|\bssh\b|\bscp\b", "red / datos que salen"),
    (r"speak\s*=\s*True|\.say\(|\bpetvoice\b|\bspeak_", "habla más"),
    (r"Reaction\([^)]*priority\s*=\s*[23]", "interrupciones"),
    (r"Keychain|find-generic-password|add-generic-password|\bsecurity\s", "credenciales"),
    (r"requestAccess|AVCaptureDevice|CGRequestScreenCaptureAccess|AXIsProcessTrusted", "permisos de macOS"),
    (r"\bosascript\b|\bAppleScript\b", "controla tus apps"),
    (r"gh issue|gh pr|git push", "publica afuera"),
    (r"(?<![\w.])~/(?!\.wawi)|/Users/", "otros archivos tuyos"),
)

#: What the author agent may declare at the end of its report.
_DECLARE_RE = re.compile(r"OWNER_IMPACT:\s*([a-z_ ,-]+)", re.I)


@dataclass
class Classification:
    verdict: str                 # AUTO or ASK
    reasons: list[str] = field(default_factory=list)


def _matches(path: str, pattern: str) -> bool:
    return fnmatch.fnmatch(path, pattern) or (pattern.endswith("/*") and path.startswith(pattern[:-1]))


def classify_change(paths: list[str], diff: str, *, agent_report: str = "",
                    capability: bool = False, new_deps: bool = False,
                    autonomy: str = "auto") -> Classification:
    """AUTO only when every signal is clean; otherwise ASK, with the reasons."""
    reasons: list[str] = []
    if autonomy != "auto":
        reasons.append("modo «preguntar» activado")
    if new_deps:
        reasons.append("agrega una dependencia")
    for path in paths:
        owner = next((why for pat, why in OWNER_PATHS if _matches(path, pat)), None)
        if owner is not None:
            reasons.append(f"toca {path} ({owner})")
            continue
        if not any(_matches(path, pat) for pat in INTERNAL_PATHS):
            reasons.append(f"toca {path}, que no sé si es solo mío")
    added = "\n".join(line[1:] for line in diff.splitlines()
                      if line.startswith("+") and not line.startswith("+++"))
    for pattern, why in OWNER_PATTERNS:
        if re.search(pattern, added):
            reasons.append(why)
    declared = _DECLARE_RE.search(agent_report or "")
    if declared is None:
        reasons.append("el agente no declaró el impacto")
    else:
        value = declared.group(1).strip().lower()
        if value not in ("none", "ninguno", "no"):
            reasons.append(f"el agente declaró impacto: {value}")
    if capability and not paths:
        reasons.append("capacidad sin archivos")
    unique = list(dict.fromkeys(reasons))
    return Classification(ASK if unique else AUTO, unique)
