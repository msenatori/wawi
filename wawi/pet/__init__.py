"""Wawi's body: needs, moods, her life on the desktop, and acting out reactions.

``wawi/mascot.py`` draws her and knows every clip; this package decides what
she DOES with them: how hungry she is (:mod:`.needs`), where she can walk and
sit on screen (:mod:`.world`), what a reaction from the sensors or the mind
looks like (:mod:`.acting`), when she comes to watch you type (:mod:`.watch`),
and :mod:`.life`, which ties it all to the animator every frame. Only
:mod:`.desktop` touches macOS (read-only window frames, the caret, the time
since the last key press); everything else is pure and tested.

:mod:`.api` is the thread-safe door for the voice brain: ``feed()``,
``water()``, ``play()``, ``status()``.
"""
