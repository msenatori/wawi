"""The voice tool: "Wawi, ¿qué aprendiste de mí?".

Kept here so wawi/tools.py only needs one small block to register it. It
reads what the learner already knows; no model, no I/O.
"""

from __future__ import annotations

from typing import Any

SCHEMA: dict[str, Any] = {
    "name": "what_learned",
    "animation": "notebook",
    # Terse on purpose: every schema is re-sent (and paid for) on every turn,
    # and tests/test_mind.py holds the mind's schemas to a budget.
    "description": "¿Qué aprendiste de mí?",
    "input_schema": {"type": "object", "properties": {}},
}


def what_learned() -> tuple[bool, str]:
    from wawi import learn

    return learn.get() is not None, learn.summary()
