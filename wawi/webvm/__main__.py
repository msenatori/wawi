"""``python -m wawi.webvm start|stop|status|build`` (also ``scripts/wawictl webvm``).

Reads the ``webvm:`` section of the config Wawi uses (private/config.yaml);
works even with ``webvm.enabled: false``, so it can be built before turning
it on. Never prints a secret.
"""

from __future__ import annotations

import argparse
import sys

from wawi.webvm.client import WebVMError


def _settings():
    from wawi.webvm.settings import WebVMSettings

    raw: dict = {}
    try:
        from wawi import config as _config

        raw = dict(getattr(_config.load(), "webvm", {}) or {})
    except Exception as exc:  # no config yet: the defaults
        first = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        print(f"(sin config: uso los valores por defecto — {first})", file=sys.stderr)
    return WebVMSettings.from_dict(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wawictl webvm", description="Su compu (wawi/webvm)")
    parser.add_argument("command", choices=("start", "stop", "status", "build"))
    args = parser.parse_args(argv)

    from wawi.webvm import keychain
    from wawi.webvm.lifecycle import Lifecycle

    settings = _settings()
    life = Lifecycle(settings, keychain.credentials)
    try:
        if args.command == "build":
            life.build()
            print(f"listo: imagen {settings.image} en {settings.context}")
        elif args.command == "start":
            got = life.ensure_running()
            print(f"prendida: {got.get('url', '')}")
            print(f"API  {settings.api_url}   (token en el Llavero: {keychain.SERVICE}/{keychain.TOKEN_ACCOUNT})")
            print(f"vista {settings.view_url}vnc.html   (clave en el Llavero: {keychain.SERVICE}/{keychain.VNC_ACCOUNT})")
        elif args.command == "stop":
            print("apagada" if life.stop() else "no estaba prendida")
        else:
            status = life.status()
            print(status.describe())
            return 0 if status.healthy else 1
    except WebVMError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
