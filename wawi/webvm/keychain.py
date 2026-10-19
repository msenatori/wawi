"""The two secrets of her little computer, kept in the macOS Keychain.

Service ``wawi-webvm`` (like the agregados' ``wawi-plugin-<id>``), accounts
``api-token`` (the control API's bearer token) and ``vnc-password`` (to watch
the screen). Generated on first use, read on every start, never written to a
file, a log or the image: they travel to the container as environment
variables of the ``docker run`` process (see wawi/webvm/lifecycle.py).
"""

from __future__ import annotations

import logging
import secrets as _random
import string
import subprocess
from typing import Callable

log = logging.getLogger(__name__)

SERVICE = "wawi-webvm"
TOKEN_ACCOUNT = "api-token"
VNC_ACCOUNT = "vnc-password"
#: VNC authentication only uses the first 8 characters.
VNC_LENGTH = 8

Security = Callable[..., subprocess.CompletedProcess]


class SecretError(RuntimeError):
    """The Keychain could not give or keep a secret. Safe to say: no value."""


def _security(args: list[str], stdin: str | None = None) -> subprocess.CompletedProcess:
    """Run ``/usr/bin/security`` (the test seam). Output is never logged."""
    return subprocess.run(["security", *args], input=stdin, capture_output=True,
                          text=True, timeout=10, check=False)


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def generate(account: str) -> str:
    if account == VNC_ACCOUNT:
        alphabet = string.ascii_letters + string.digits
        return "".join(_random.choice(alphabet) for _ in range(VNC_LENGTH))
    return _random.token_urlsafe(32)


def read(account: str, security: Security = _security) -> str | None:
    try:
        result = security(["find-generic-password", "-s", SERVICE, "-a", account, "-w"])
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.rstrip("\n") or None


def store(account: str, value: str, security: Security = _security) -> None:
    """Through ``security -i`` on stdin, so the value never shows in ``ps``."""
    command = (f"add-generic-password -U -s {_quote(SERVICE)} -a {_quote(account)} "
               f"-w {_quote(value)}\n")
    result = security(["-i"], stdin=command)
    if result.returncode != 0:
        raise SecretError("No pude guardar la clave de mi compu en el Llavero.")
    log.info("webvm: stored %s in the Keychain (%s)", account, SERVICE)


def get_or_create(account: str, security: Security = _security) -> str:
    value = read(account, security)
    if value:
        return value
    value = generate(account)
    store(account, value, security)
    if read(account, security) != value:
        raise SecretError("El Llavero no devolvió la clave de mi compu.")
    return value


def credentials(security: Security = _security) -> tuple[str, str]:
    """``(api_token, vnc_password)``, created on first use."""
    return get_or_create(TOKEN_ACCOUNT, security), get_or_create(VNC_ACCOUNT, security)
