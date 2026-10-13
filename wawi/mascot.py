"""Wawi, the mascot: assets and the pure animation logic behind her window.

Wawi is the dog the assistant is named after, drawn by the author's niece
(``assets/wawi_dibujo.jpg``). She is shown two ways:

* **Frames** (``assets/wawi/*.png``): a set of poses redrawn in the drawing's
  style — breathing, blinking, listening, talking, scratching, sleeping… —
  swapped like a flip-book by the clip engine below, with the old whole-sprite
  motion (:class:`Pose`) layered on top, toned down.
* **The single cutout** (``assets/wawi.png``), only moved, scaled, tilted and
  faded. This is the fallback whenever the frames are not there.

The frames are optional one by one: a missing frame falls back to its nearest
relative (``listening_b`` → ``listening`` → ``idle_a``), a behaviour whose
frames are missing is simply never played, and with no ``idle_a`` at all she
is the single cutout again. Nothing here may crash or log per frame over art.

Nothing here imports AppKit. The panel's window draws whatever the
:class:`MascotAnimator` hands it (a :class:`Pose`, a frame name, and the frame
being crossfaded away), and the offline previews use the very same animator,
so what the tests check is what is on screen.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NamedTuple

from wawi.state import State

log = logging.getLogger(__name__)

#: Shipped next to this module, so it works from a checkout and from an
#: installed package alike.
ASSET_DIR = Path(__file__).resolve().parent / "assets"

SPRITE = "wawi.png"
MENUBAR_ICON = "wawi_menubar.png"
FACE_ICON = "wawi_face.png"
APP_ICON = "Wawi.icns"

#: Folder (inside :data:`ASSET_DIR`) with the animation frames, all the same
#: size and aligned to each other (``scripts/make_wawi_assets.py --frames``).
FRAME_DIR = "wawi"
#: Every frame the engine knows how to use. Any subset may be present.
FRAME_NAMES = (
    "idle_a", "idle_b", "blink",
    "look_left", "look_right", "yawn", "scratch_a", "scratch_b", "stretch",
    "sniff", "wag_a", "wag_b", "tilt", "roll",
    "listening", "listening_b",
    "speaking_a", "speaking_b", "speaking_c",
    "thinking", "thinking_b",
    "happy", "happy_b", "jump",
    "sleeping", "sleeping_b",
    # Idle behaviours with personality: digging, a run, staring at you, and
    # the ironic set (eye-roll, smug, side-eye, shrug), plus fiddling with
    # the volume knob and chasing her tail.
    "dig_a", "dig_b", "dig_c", "dig_hole",
    "run_a", "run_b", "run_c", "run_d", "run_stop_look",
    "stare", "stare_squint",
    "eyeroll", "smug", "side_eye", "shrug",
    "volume_knob",
    # Listening extras: the ear trumpet, the cupped ear, and the mic level.
    "megaphone_a", "megaphone_b", "ear_cup", "volume_a", "volume_b",
    # Touch reactions and the eyes following the cursor.
    "pet_head", "pet_head_b", "boop", "sneeze", "tickle_a", "tickle_b",
    "tail_grab", "tail_chase_a", "tail_chase_b", "ear_flop", "paw_up",
    "grumpy", "follow_left", "follow_right", "follow_up",
)
#: The one frame everything else can fall back to. Without it, frames are off.
BASE_FRAME = "idle_a"
#: Where a missing frame falls back to, step by step, until one exists.
FRAME_FALLBACK = {
    "idle_b": "idle_a",
    "blink": "idle_a",
    "listening": "idle_a",
    "listening_b": "listening",
    "speaking_a": "idle_a",
    # Half-open reads as talking better than closed, so it borrows "open".
    "speaking_c": "speaking_b",
    "speaking_b": "speaking_a",
    "thinking": "idle_a",
    "thinking_b": "thinking",
    "sleeping": "idle_a",
    "sleeping_b": "sleeping",
    "happy": "idle_a",
    "happy_b": "happy",
    "jump": "happy",
    # Listening variants must never show something unrelated: they fall back
    # to the plain listening frames.
    "megaphone_a": "listening_b",
    "megaphone_b": "megaphone_a",
    "ear_cup": "listening",
    "volume_a": "listening_b",
    "volume_b": "volume_a",
    "pet_head_b": "pet_head",
    "pet_head": "happy",
    "sneeze": "boop",
    "tickle_b": "tickle_a",
    "tail_chase_b": "tail_chase_a",
    "stare_squint": "stare",
}

#: On-screen sprite width in points. Height follows the drawing's aspect.
SPRITE_WIDTH = 124.0
#: Room around the sprite inside its window, for the hop, the stretch, the
#: tilt (a 7-degree tilt swings the top corners ~10 pt sideways and ~8 pt up)
#: and the status dot.
PAD_X = 16.0
PAD_TOP = 30.0
PAD_BOTTOM = 2.0
#: How far she tucks in under the panel's corner, so she reads as sitting
#: against it rather than floating next to it.
OVERLAP = 22.0

_missing_logged: set[str] = set()


def asset_path(name: str, directory: Path | None = None) -> Path | None:
    """Path to a bundled asset, or None when it is missing.

    A missing asset is logged ONCE and then degrades silently: the mascot is a
    companion, and nothing about it is worth a crash or a log line per frame.
    """
    path = (directory or ASSET_DIR) / name
    if path.is_file():
        return path
    if name not in _missing_logged:
        _missing_logged.add(name)
        log.warning("Mascot asset missing: %s (Wawi will not be shown there)", path)
    return None


def frame_path(name: str, directory: Path | None = None) -> Path:
    """Where frame *name* lives (whether or not it is there)."""
    return (directory or ASSET_DIR) / FRAME_DIR / f"{name}.png"


def available_frames(directory: Path | None = None) -> frozenset[str]:
    """The frames present on disk, as names. Logs what is missing ONCE.

    Missing frames are normal while the art is being drawn, so this is one
    INFO line per process, not a warning per frame; with no base frame at all
    she is the single cutout, which is worth a warning.
    """
    present = frozenset(n for n in FRAME_NAMES if frame_path(n, directory).is_file())
    tag = f"frames:{directory or ASSET_DIR}"
    if tag not in _missing_logged:
        _missing_logged.add(tag)
        missing = [n for n in FRAME_NAMES if n not in present]
        if BASE_FRAME not in present:
            log.warning(
                "Wawi's animation frames are missing (%s); using the single drawing",
                frame_path(BASE_FRAME, directory).parent,
            )
        elif missing:
            log.info("Wawi frames not drawn yet (fallbacks used): %s", ", ".join(missing))
    return present if BASE_FRAME in present else frozenset()


def resolve_frame(name: str, available: Iterable[str]) -> str | None:
    """The frame to show for *name*: itself, or its nearest present relative.

    None only when not even :data:`BASE_FRAME` is there (frames are off).
    """
    have = available if isinstance(available, (set, frozenset)) else set(available)
    seen = set()
    while name not in have:
        if name in seen:
            return None
        seen.add(name)
        name = FRAME_FALLBACK.get(name, BASE_FRAME if name != BASE_FRAME else "")
        if not name:
            return None
    return name


@dataclass(frozen=True)
class Pose:
    """One frame of the sprite, relative to its resting position.

    Scale and rotation pivot on the bottom-centre of the sprite — where she
    touches the ground — so breathing and squashing never make her float.
    ``rot`` is in degrees, counter-clockwise (AppKit and PIL agree on this);
    a positive value leans her head, which is on the left, forward.
    """

    dx: float = 0.0
    dy: float = 0.0
    sx: float = 1.0
    sy: float = 1.0
    rot: float = 0.0
    alpha: float = 1.0

    def lerp(self, other: "Pose", w: float) -> "Pose":
        w = min(1.0, max(0.0, w))
        return Pose(*(a + (b - a) * w for a, b in zip(self.astuple(), other.astuple())))

    def astuple(self) -> tuple[float, float, float, float, float, float]:
        return (self.dx, self.dy, self.sx, self.sy, self.rot, self.alpha)


REST = Pose()

#: Muted reads as lying down: flatter, wider and dimmer — still there, not
#: listening. This one survives Reduce Motion, because it is information.
_MUTED = Pose(sx=1.05, sy=0.87, alpha=0.55)

#: Syllable rate of the fallback talking rhythm, in Hz. Close to real speech
#: (four to five syllables a second reads as chatter; three reads as talking).
_TALK_HZ = 3.1


def mood(state: State, muted: bool) -> str:
    """The animation key: the microphone gate wins over everything else."""
    if muted:
        return "muted"
    return state.value.lower() if isinstance(state, State) else str(state).lower()


def badge(key: str) -> str | None:
    """Which tiny status dot to draw by her head, if any.

    Without the panel she is the only thing on screen, and the two states a
    pose alone can leave in doubt are the ones that matter most: "is it
    recording me right now?" and "is the microphone off?". Everything else is
    obvious from the motion, so it gets no dot.
    """
    return key if key in ("listening", "muted") else None


def _bump(u: float) -> float:
    """A single smooth hump over 0..1, zero elsewhere."""
    return math.sin(math.pi * u) if 0.0 < u < 1.0 else 0.0


def talk_envelope(t: float) -> float:
    """A synthetic mouth: 0..1, syllable-paced, with phrase-length swells.

    Used only when the real voice level is not available. It has to READ as
    talking at a glance, so it is a steady syllable beat rather than noise,
    and it closes fully between syllables so the bob visibly lands.
    """
    syllable = 0.5 - 0.5 * math.cos(2 * math.pi * _TALK_HZ * t)
    phrase = 0.7 + 0.3 * math.sin(2 * math.pi * t / 1.9)
    return syllable ** 1.4 * phrase


def mascot_pose(
    key: str,
    t: float,
    level: float = 0.0,
    reduce_motion: bool = False,
    voice: float | None = None,
) -> Pose:
    """The pose for animation *key* at time *t* (seconds).

    Each state has ONE unmistakable shape, because without the panel she is
    the whole interface and has to be read from the corner of an eye on a
    124-point sprite:

    * listening — head UP and held, lifted and stretched tall: alert and still.
    * speaking  — a rhythmic bob that follows her own voice, swaying.
    * thinking  — a slow head tilt from side to side, on the ground.
    * idle      — sitting, breathing.
    * muted     — lying down, dimmed.

    Amplitudes are in points at :data:`SPRITE_WIDTH`.

    Args:
        key: ``idle``, ``listening``, ``thinking``, ``speaking`` or ``muted``.
        t: Seconds on the animation clock.
        level: Live microphone level 0..1 (only ``listening`` uses it).
        reduce_motion: The system's Reduce Motion setting: hold still.
        voice: Level 0..1 of what she is SAYING right now (only ``speaking``
            uses it). None when unknown: a synthetic talking rhythm stands in.
    """
    if reduce_motion:
        return _MUTED if key == "muted" else REST

    level = min(1.0, max(0.0, float(level)))
    breathe = math.sin(2 * math.pi * t / 3.6)

    if key == "listening":
        # Alert: head raised (negative rot lifts the head, which is on the
        # left), body lifted and stretched, and HELD — the stillness is what
        # tells it apart from talking. The louder you are, the taller she
        # stands; now and then a small curious head-cock.
        cock = _bump((t % 4.4) / 1.1)
        return Pose(
            dy=5.0 + 3.0 * level + 0.4 * math.sin(2 * math.pi * t / 1.6),
            sx=0.98 - 0.02 * level,
            sy=1.06 + 0.04 * level + 0.004 * breathe,
            rot=-5.0 - 2.0 * level - 2.0 * cock,
        )

    if key == "speaking":
        # The bob IS her mouth: up and stretched on each syllable, down and
        # squashed between them, so it follows the voice you hear. A slow
        # sway on top, which listening never has.
        env = talk_envelope(t) if voice is None else min(1.0, max(0.0, float(voice)))
        sway = math.sin(2 * math.pi * t / 1.15)
        return Pose(
            dx=1.2 * sway,
            dy=0.2 + 5.8 * env,
            sx=1.025 - 0.045 * env,
            sy=0.97 + 0.08 * env,
            rot=3.5 * sway,
        )

    if key == "thinking":
        # Head tilt from side to side, slow enough to read as pondering, and
        # on the ground: thinking is the one active state that never lifts.
        sway = math.sin(2 * math.pi * t / 2.6)
        return Pose(dx=1.5 * sway, sy=0.99 + 0.008 * breathe, rot=5.0 * sway)

    if key == "muted":
        slow = math.sin(2 * math.pi * t / 5.0)
        return Pose(
            sx=_MUTED.sx, sy=_MUTED.sy + 0.005 * slow, alpha=_MUTED.alpha
        )

    # idle: slow breathing and, now and then, a small settling bob.
    settle = _bump((t % 7.5) / 0.9)
    return Pose(dy=1.6 * settle, sx=1.0 - 0.004 * breathe, sy=1.0 + 0.012 * breathe)


def framed_pose(
    key: str,
    t: float,
    level: float = 0.0,
    reduce_motion: bool = False,
    voice: float | None = None,
) -> Pose:
    """Whole-sprite motion layered over the FRAMES: the same shapes, smaller.

    With frames the drawing itself shows the ears going up, the mouth moving
    and the eyes closing, so the motion only has to support it: a lift while
    listening, a light bob on each syllable, a slow sway while thinking.
    ``asleep`` is idle-asleep (full opacity); ``muted`` keeps its dimming,
    which is information and survives Reduce Motion.
    """
    if reduce_motion:
        return _MUTED_FRAMED if key == "muted" else REST
    level = min(1.0, max(0.0, float(level)))
    breathe = math.sin(2 * math.pi * t / 3.6)
    if key == "listening":
        cock = _bump((t % 4.4) / 1.1)
        return Pose(
            dy=2.0 + 1.5 * level + 0.3 * math.sin(2 * math.pi * t / 1.6),
            sy=1.02 + 0.02 * level + 0.003 * breathe,
            rot=-2.0 - 1.0 * level - 1.5 * cock,
        )
    if key == "speaking":
        env = talk_envelope(t) if voice is None else min(1.0, max(0.0, float(voice)))
        sway = math.sin(2 * math.pi * t / 1.15)
        return Pose(
            dx=0.6 * sway,
            dy=0.2 + 2.5 * env,
            sx=1.01 - 0.02 * env,
            sy=0.99 + 0.035 * env,
            rot=1.2 * sway,
        )
    if key == "thinking":
        sway = math.sin(2 * math.pi * t / 2.6)
        return Pose(dx=0.8 * sway, rot=2.0 * sway)
    slow = math.sin(2 * math.pi * t / 5.0)
    if key == "muted":
        return Pose(sy=1.0 + 0.004 * slow, alpha=_MUTED_FRAMED.alpha)
    if key == "asleep":
        return Pose(sy=1.0 + 0.006 * slow)
    # idle: the idle_a/idle_b frames breathe; this only keeps her soft.
    return Pose(sy=1.0 + 0.005 * breathe)


#: Muted with frames: the sleeping frame already lies her down, so only dim.
_MUTED_FRAMED = Pose(alpha=0.55)


# ---------------------------------------------------------------------------
# Clips: sequences of frames
# ---------------------------------------------------------------------------

#: Default crossfade between two consecutive frames, in seconds. The frames
#: were drawn one by one, so their outlines drift a little; a short dissolve
#: hides that without smearing the change itself.
FRAME_XFADE = 0.075


class Step(NamedTuple):
    """One frame held for *duration* seconds, with an optional overlay motion
    (see :func:`motion_overlay`)."""

    frame: str
    duration: float
    motion: str | None = None


@dataclass(frozen=True)
class Clip:
    """A named sequence of frames; looping (state) or one-shot (behaviour)."""

    name: str
    steps: tuple[Step, ...]
    loop: bool = False
    xfade: float = FRAME_XFADE

    @property
    def duration(self) -> float:
        return sum(s.duration for s in self.steps)

    def step_at(self, t: float) -> tuple[Step, float]:
        """The step showing at clip time *t*, and the time inside it."""
        total = self.duration
        if total <= 0.0:
            return self.steps[0], 0.0
        t = max(0.0, t)
        if self.loop:
            t %= total
        elif t >= total:
            return self.steps[-1], self.steps[-1].duration
        acc = 0.0
        for step in self.steps:
            if t < acc + step.duration:
                return step, t - acc
            acc += step.duration
        return self.steps[-1], self.steps[-1].duration

    def frame_at(self, t: float) -> str:
        return self.step_at(t)[0].frame

    def done(self, t: float) -> bool:
        return not self.loop and t >= self.duration

    def playable(self, available: Iterable[str]) -> "Clip | None":
        """This clip with its missing frames dropped, or None if it would not
        show anything of its own (only resting frames, or nothing at all)."""
        have = set(available)
        steps = tuple(s for s in self.steps if s.frame in have)
        if not any(s.frame not in _RESTING for s in steps):
            return None
        return Clip(self.name, steps, self.loop, self.xfade)


#: Frames that are "just sitting there": a clip made only of these shows
#: nothing, so it is not worth playing.
_RESTING = frozenset({"idle_a", "idle_b"})


def _alt(a: str, b: str, each: float, times: int, motion: str | None = None) -> tuple[Step, ...]:
    """*a*, *b*, *a*, *b*… *times* pairs, each frame held *each* seconds."""
    return tuple(Step(f, each, motion) for _ in range(times) for f in (a, b))


#: The gallop, and how long each of its frames is held.
_RUN_CYCLE = ("run_a", "run_b", "run_c", "run_d")
_RUN_STEP = 0.09
#: How far the dash takes her, in points, and so her speed. She never goes
#: further than this from her spot, and comes back to it.
RUN_DISTANCE = 36.0
_RUN_SPEED = RUN_DISTANCE / (2 * len(_RUN_CYCLE) * _RUN_STEP)

#: The state loops. Listening and speaking are not clips: their frame follows
#: a live level (see MascotAnimator._state_frame).
STATE_CLIPS: dict[str, Clip] = {
    "idle": Clip("idle", (Step("idle_a", 1.9), Step("idle_b", 1.7)), loop=True, xfade=0.45),
    "thinking": Clip(
        "thinking", (Step("thinking", 1.6), Step("thinking_b", 1.4)), loop=True, xfade=0.25
    ),
    "sleep": Clip(
        "sleep", (Step("sleeping", 2.3), Step("sleeping_b", 2.1)), loop=True, xfade=0.8
    ),
}

#: One-shot clips: idle behaviours, celebrations and reactions.
CLIPS: dict[str, Clip] = {
    "blink": Clip("blink", (Step("blink", 0.13),), xfade=0.04),
    "look_left": Clip("look_left", (Step("look_left", 1.4, "glance"),)),
    "look_right": Clip("look_right", (Step("look_right", 1.4, "glance"),)),
    "look_around": Clip(
        "look_around",
        (Step("look_left", 0.9), Step("idle_a", 0.25), Step("look_right", 0.9)),
    ),
    "yawn": Clip("yawn", (Step("yawn", 1.4, "yawn"),), xfade=0.12),
    "scratch": Clip(
        "scratch",
        _alt("scratch_a", "scratch_b", 0.11, 4, "wiggle") + (Step("scratch_a", 0.25),),
        xfade=0.05,
    ),
    "stretch": Clip("stretch", (Step("stretch", 1.6, "stretch"),), xfade=0.15),
    "sniff": Clip(
        "sniff",
        (Step("sniff", 0.45, "sniff"), Step("idle_a", 0.15), Step("sniff", 0.7, "sniff")),
    ),
    "wag": Clip("wag", _alt("wag_a", "wag_b", 0.15, 5, "wag"), xfade=0.06),
    "tilt": Clip("tilt", (Step("tilt", 1.5, "tilt"),), xfade=0.1),
    "roll": Clip(
        "roll", (Step("roll", 2.2, "wobble"), Step("idle_a", 0.2)), xfade=0.12
    ),
    # Celebrations: "cheer" after an answer, "party" when a task got done.
    "cheer": Clip(
        "cheer", _alt("happy", "happy_b", 0.18, 2, "bounce") + (Step("happy", 0.4),)
    ),
    "party": Clip(
        "party",
        _alt("happy", "happy_b", 0.16, 3, "bounce")
        + (Step("jump", 0.42, "hop"), Step("happy", 0.5)),
    ),
    # A click on her: a little hop of joy.
    "hop": Clip("hop", (Step("happy", 0.2), Step("jump", 0.42, "hop"), Step("happy", 0.4))),
    # Waking up and dozing off.
    "wake": Clip(
        "wake", (Step("stretch", 1.1, "stretch"), Step("yawn", 0.9, "yawn")), xfade=0.15
    ),
    "lie_down": Clip("lie_down", (Step("yawn", 1.3, "yawn"),), xfade=0.15),
    # -- personality ------------------------------------------------------
    # Digging: three quick paw strokes, then proud of a hole nobody asked for.
    "dig": Clip(
        "dig",
        tuple(Step(f, 0.13, "dig") for _ in range(3) for f in ("dig_a", "dig_b", "dig_c"))
        + (Step("dig_hole", 1.3, "proud"),),
        xfade=0.05,
    ),
    # A short dash: she runs off the way she faces (her head is on the left),
    # stops to look back, and runs home mirrored. The distance is not in the
    # pose (it would leave her window): see clip_excursion.
    "run": Clip(
        "run",
        tuple(Step(f, _RUN_STEP, "run_out") for _ in range(2) for f in _RUN_CYCLE)
        + (Step("run_stop_look", 0.9, "glance"),)
        + tuple(Step(f, _RUN_STEP, "run_back") for _ in range(2) for f in _RUN_CYCLE),
        xfade=0.04,
    ),
    # Looking straight at you. "squint" is the ironic one: …really?
    "stare": Clip("stare", (Step("stare", 1.8, "lean"),), xfade=0.15),
    "squint": Clip(
        "squint", (Step("stare", 0.8, "lean"), Step("stare_squint", 1.5, "lean")), xfade=0.15
    ),
    "eyeroll": Clip("eyeroll", (Step("eyeroll", 1.6, "roll_eyes"),), xfade=0.12),
    "smug": Clip("smug", (Step("smug", 1.7, "chin_up"),), xfade=0.12),
    "side_eye": Clip("side_eye", (Step("side_eye", 1.7, "glance"),), xfade=0.12),
    "shrug": Clip("shrug", (Step("shrug", 1.4, "shrug"),), xfade=0.12),
    "volume_knob": Clip(
        "volume_knob", (Step("volume_knob", 0.5), Step("volume_knob", 1.3, "fiddle")),
        xfade=0.12,
    ),
    "tail_chase": Clip(
        "tail_chase",
        _alt("tail_chase_a", "tail_chase_b", 0.15, 4, "spin") + (Step("tail_grab", 0.5),),
        xfade=0.05,
    ),
    # -- touch reactions (see touch_zone) ---------------------------------
    "pet": Clip(
        "pet",
        (Step("pet_head", 0.45), Step("pet_head_b", 0.45), Step("pet_head", 0.45),
         Step("happy", 0.5)),
        xfade=0.12,
    ),
    # Stroked with the cursor: loops for as long as the strokes go on.
    "petting": Clip(
        "petting", (Step("pet_head", 0.5), Step("pet_head_b", 0.5)), loop=True, xfade=0.15
    ),
    "petted": Clip(
        "petted", _alt("happy", "happy_b", 0.18, 2, "bounce") + (Step("happy", 0.3),)
    ),
    "boop": Clip("boop", (Step("boop", 0.9, "recoil"),), xfade=0.05),
    "sneeze": Clip(
        "sneeze", (Step("boop", 0.35, "recoil"), Step("sneeze", 0.6, "sneeze")), xfade=0.05
    ),
    "tickle": Clip(
        "tickle", _alt("tickle_a", "tickle_b", 0.14, 3, "wiggle") + (Step("tickle_a", 0.3),),
        xfade=0.05,
    ),
    "tail_grab": Clip("tail_grab", (Step("tail_grab", 1.3, "glance"),), xfade=0.1),
    "ear_flop": Clip("ear_flop", (Step("ear_flop", 1.5, "tilt"),), xfade=0.08),
    "paw_up": Clip("paw_up", (Step("paw_up", 1.7),), xfade=0.12),
    "grumpy": Clip("grumpy", (Step("grumpy", 2.2, "huff"),), xfade=0.12),
}

#: How often each idle behaviour is picked. Roll is the rare treat.
BEHAVIOUR_WEIGHTS: dict[str, float] = {
    "look_left": 3.0,
    "look_right": 3.0,
    "look_around": 1.5,
    "yawn": 1.5,
    "scratch": 2.0,
    "stretch": 1.5,
    "sniff": 2.0,
    "wag": 2.5,
    "tilt": 2.0,
    "roll": 0.5,
    "dig": 1.5,
    "run": 1.0,
    "stare": 1.2,
    "squint": 0.8,
    "eyeroll": 0.8,
    "smug": 0.8,
    "side_eye": 1.0,
    "shrug": 0.8,
    "volume_knob": 0.8,
    "tail_chase": 1.0,
}
#: The ironic set: what she does when she has been ignored for a while.
IRONIC = ("side_eye", "eyeroll", "squint")
#: Nobody has talked to her or touched her for this long: she gets ironic.
IGNORED_AFTER = 240.0
#: What she may get up for when she wakes on her own: the big ones.
WAKE_BEHAVIOURS = (
    "dig", "run", "tail_chase", "stare", "squint", "volume_knob", "roll", "scratch",
    "side_eye", "eyeroll",
)
#: Asleep, she wakes on her own every this many seconds, does one rare thing
#: and goes back to sleep. Asleep stays the resting look; this is a treat.
WAKE_GAP = (70.0, 180.0)
#: An answer at least this long (seconds of talking) is followed by a shrug or
#: a smug look instead of the usual cheer: she knows she went on a bit.
LONG_ANSWER = 12.0
#: What a click on her may do (awake and idle) outside any zone.
POKE_REACTIONS = ("hop", "tilt", "wag", "cheer")
#: What a touch on each zone of her body may do, with weights.
ZONE_REACTIONS: dict[str, tuple[tuple[str, float], ...]] = {
    "head": (("pet", 1.0),),
    "back": (("pet", 1.0),),
    "nose": (("boop", 3.0), ("sneeze", 1.0)),
    "belly": (("tickle", 1.0),),
    "tail": (("tail_grab", 3.0), ("tail_chase", 1.0)),
    "ear": (("ear_flop", 1.0),),
    "paw": (("paw_up", 1.0),),
}
#: More than this many touches inside GRUMPY_WINDOW seconds and she is fed up.
GRUMPY_TOUCHES = 6
GRUMPY_WINDOW = 8.0
#: Under Reduce Motion a touch shows ONE frame, held this long, no motion.
STILL_HOLD = 1.5

#: Seconds between idle behaviours, drawn uniformly.
BEHAVIOUR_GAP = (6.0, 20.0)
#: Seconds between blinks.
BLINK_GAP = (2.0, 6.0)
#: Idle with nobody around for this long and she lies down to sleep. Short on
#: purpose: the owner likes her asleep as the resting look, so after a turn she
#: stays awake just long enough for a stretch or a glance, then dozes off.
SLEEP_AFTER = 10.0


def _plateau(u: float, d: float, ramp: float = 0.25) -> float:
    """0 → 1 → 0 over a step of *d* seconds, easing in and out in *ramp*."""
    if d <= 0.0:
        return 0.0
    r = min(ramp, d / 2.0)
    x = min(1.0, u / r, (d - u) / r)
    x = max(0.0, x)
    return x * x * (3 - 2 * x)


def motion_overlay(motion: str | None, u: float, d: float) -> Pose:
    """The extra whole-sprite motion of a behaviour step, as a delta.

    ``dx``/``dy``/``rot`` add; ``sx``/``sy`` multiply. *u* is the time inside
    the step and *d* its length. Small on purpose: the frame does the acting.
    """
    if motion is None:
        return REST
    k = _plateau(u, d)
    if motion == "hop":
        # A real hop: up and down once over the jump frame.
        return Pose(dy=7.0 * _bump(u / d) if d > 0 else 0.0)
    if motion == "bounce":
        return Pose(dy=2.0 * abs(math.sin(2 * math.pi * 2.6 * u)))
    if motion == "wiggle":
        return Pose(dx=0.7 * math.sin(2 * math.pi * 9.0 * u), rot=1.8 * math.sin(2 * math.pi * 9.0 * u))
    if motion == "wag":
        return Pose(rot=0.8 * math.sin(2 * math.pi * 3.3 * u))
    if motion == "stretch":
        return Pose(sx=1.0 + 0.035 * k, sy=1.0 - 0.025 * k)
    if motion == "yawn":
        return Pose(sy=1.0 + 0.03 * k, rot=-1.5 * k)
    if motion == "tilt":
        return Pose(rot=3.5 * k)
    if motion == "glance":
        return Pose(rot=1.2 * k)
    if motion == "sniff":
        return Pose(dx=0.8 * math.sin(2 * math.pi * 6.0 * u), rot=1.0 * k)
    if motion == "wobble":
        return Pose(rot=3.0 * math.sin(2 * math.pi * 1.1 * u) * k)
    if motion == "dig":
        return Pose(dx=0.6 * math.sin(2 * math.pi * 7.0 * u), rot=0.8 * math.sin(2 * math.pi * 7.0 * u))
    if motion == "proud":
        # Chin up (negative rot lifts the head, which is on the left).
        return Pose(dy=1.2 * k, rot=-1.5 * k)
    if motion in ("run_out", "run_back"):
        # The gallop's bounce; the distance travelled is clip_excursion's.
        return Pose(dy=1.5 * abs(math.sin(2 * math.pi * 5.5 * u)))
    if motion == "lean":
        return Pose(dy=0.8 * k, sy=1.0 + 0.01 * k)
    if motion == "roll_eyes":
        return Pose(rot=-1.2 * k)
    if motion == "chin_up":
        return Pose(dy=0.6 * k, rot=-2.2 * k)
    if motion == "shrug":
        return Pose(dy=2.0 * k)
    if motion == "fiddle":
        return Pose(rot=1.0 * math.sin(2 * math.pi * 2.2 * u) * k)
    if motion == "spin":
        w = math.sin(2 * math.pi * 3.0 * u)
        return Pose(dx=1.5 * w, rot=2.5 * w)
    if motion == "recoil":
        # Booped: a jerk back (away from the nose, to the right) that settles.
        return Pose(dx=1.8 * _bump(min(1.0, u / 0.45)), rot=-1.0 * _bump(min(1.0, u / 0.45)))
    if motion == "sneeze":
        return Pose(dx=-2.0 * _bump(min(1.0, u / 0.3)), rot=2.0 * _bump(min(1.0, u / 0.3)))
    if motion == "huff":
        return Pose(sx=1.0 + 0.015 * k, sy=1.0 - 0.02 * k)
    return REST


def clip_excursion(clip: Clip, t: float) -> float:
    """How far (points, + is right) a clip has carried her at clip time *t*.

    Only the dash moves her for real: ``run_out`` steps travel the way she
    faces (left) at :data:`_RUN_SPEED`, ``run_back`` steps travel back at the
    same speed. The window moves, not the drawing, so she can go further
    than her own padding; out and back are the same frames, so a clip with
    missing frames still comes home.
    """
    if t <= 0.0:
        return 0.0
    x = 0.0
    acc = 0.0
    for step in clip.steps:
        if acc >= t:
            break
        spent = min(step.duration, t - acc)
        if step.motion == "run_out":
            x -= _RUN_SPEED * spent
        elif step.motion == "run_back":
            x += _RUN_SPEED * spent
        acc += step.duration
    return x


def _apply(p: Pose, o: Pose) -> Pose:
    if o is REST:
        return p
    return Pose(p.dx + o.dx, p.dy + o.dy, p.sx * o.sx, p.sy * o.sy, p.rot + o.rot, p.alpha)


def crossfade_alphas(mix: float, alpha: float = 1.0) -> tuple[float, float]:
    """Opacities ``(previous, current)`` for a frame crossfade at *mix* 0..1.

    The new frame is drawn OVER the old one. Fading both linearly would leave
    her half see-through mid-way (two 50% layers are 75% opaque); instead the
    new frame comes in over the first half while the old one stays solid, and
    the old one goes over the second half. Inside her body that is a true
    blend; at the outline it is a quick dissolve.
    """
    mix = min(1.0, max(0.0, float(mix)))
    return (alpha * min(1.0, 2.0 * (1.0 - mix)), alpha * min(1.0, 2.0 * mix))


# ---------------------------------------------------------------------------
# Touch: where on her body a click lands, petting, and being fed up
# ---------------------------------------------------------------------------

#: Body zones as boxes ``(zone, u0, v0, u1, v1)`` in the drawn frame, u from
#: her left edge to her right (0..1), v from the ground up (0..1), for the
#: frames as drawn: head on the LEFT. First match wins, so small parts (nose,
#: ears) come before the big ones they sit on. Measured on ``idle_a`` (sitting)
#: and ``sleeping`` (lying, head low) in the shared crop.
_ZONES_SITTING: tuple[tuple[str, float, float, float, float], ...] = (
    ("nose", 0.24, 0.36, 0.34, 0.45),
    ("ear", 0.015, 0.51, 0.19, 0.73),
    ("ear", 0.50, 0.47, 0.64, 0.70),
    ("head", 0.10, 0.31, 0.55, 0.80),
    ("paw", 0.04, 0.0, 0.48, 0.15),
    ("belly", 0.05, 0.15, 0.55, 0.31),
    ("tail", 0.66, 0.0, 1.0, 0.43),
    ("back", 0.50, 0.0, 0.70, 0.55),
)
_ZONES_LYING: tuple[tuple[str, float, float, float, float], ...] = (
    ("nose", 0.20, 0.14, 0.32, 0.27),
    ("ear", 0.03, 0.30, 0.17, 0.50),
    ("ear", 0.45, 0.35, 0.62, 0.52),
    ("head", 0.03, 0.12, 0.62, 0.55),
    ("paw", 0.03, 0.0, 0.45, 0.12),
    ("tail", 0.80, 0.0, 1.0, 0.42),
    ("back", 0.55, 0.0, 0.85, 0.50),
)


def touch_zone(u: float, v: float, mirrored: bool = False, lying: bool = False) -> str | None:
    """The part of her body at (*u*, *v*) of the drawn frame, or None.

    *mirrored* is True while she is drawn flipped (running back), so her head
    is on the right; *lying* uses the sleeping layout. None is "somewhere on
    her window, but not on a part with its own reaction".
    """
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        return None
    if mirrored:
        u = 1.0 - u
    for zone, u0, v0, u1, v1 in (_ZONES_LYING if lying else _ZONES_SITTING):
        if u0 <= u <= u1 and v0 <= v <= v1:
            return zone
    return None


def sprite_uv(
    point: tuple[float, float],
    view_width: float,
    sprite_size: tuple[float, float],
    pose: "Pose",
) -> tuple[float, float]:
    """A point in her view (AppKit points, y up) as (u, v) of the drawn frame.

    The exact inverse of how the view draws her: translate to the bottom
    centre plus the pose offset, rotate, scale. Mirroring is left to
    :func:`touch_zone`. Values outside 0..1 are outside the frame.
    """
    w, h = sprite_size
    x = point[0] - (view_width / 2.0 + pose.dx)
    y = point[1] - (PAD_BOTTOM + pose.dy)
    r = math.radians(-pose.rot)
    x, y = x * math.cos(r) - y * math.sin(r), x * math.sin(r) + y * math.cos(r)
    x /= pose.sx or 1.0
    y /= pose.sy or 1.0
    return ((x + w / 2.0) / max(1e-6, w), y / max(1e-6, h))


#: Where her head is in the sitting frame, as (u, v): what the eyes follow
#: the cursor from.
HEAD_UV = (0.33, 0.55)


class PetDetector:
    """Tells strokes over her head from a cursor just passing by.

    Fed the cursor while it hovers over her with no button down. Petting is
    the cursor going back and forth over her head: at least
    :attr:`REVERSALS` changes of direction, each after a real stroke
    (:attr:`STROKE` of her width), within :attr:`WINDOW` seconds. It stops
    :attr:`STOP_AFTER` seconds after the strokes do, or when the cursor
    leaves her head. Pure: time comes in with each sample.
    """

    STROKE = 0.06
    REVERSALS = 2
    WINDOW = 1.6
    STOP_AFTER = 0.6
    ZONES = frozenset({"head", "ear"})
    _JITTER = 0.004

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._last_u: float | None = None
        self._dir = 0
        self._leg_from = 0.0
        self._reversals: list[float] = []
        self._last_move = -1e9
        self._on = False

    def feed(self, t: float, u: float, zone: str | None) -> None:
        if zone not in self.ZONES:
            self.reset()
            return
        if self._last_u is None:
            self._last_u = self._leg_from = u
            return
        d = u - self._last_u
        if abs(d) < self._JITTER:
            return
        sign = 1 if d > 0 else -1
        if self._dir == 0:
            self._dir = sign
        elif sign != self._dir:
            if abs(self._last_u - self._leg_from) >= self.STROKE:
                self._reversals.append(t)
            self._leg_from = self._last_u
            self._dir = sign
        self._last_u = u
        self._last_move = t

    def stroking(self, t: float) -> bool:
        """True while she is being petted (sticky while strokes go on)."""
        if t - self._last_move > self.STOP_AFTER:
            self._on = False
            return False
        recent = [r for r in self._reversals if t - r <= self.WINDOW]
        self._reversals = recent
        if len(recent) >= self.REVERSALS:
            self._on = True
        return self._on


def too_many_touches(times: Iterable[float], now: float,
                     limit: int = GRUMPY_TOUCHES, window: float = GRUMPY_WINDOW) -> bool:
    """More than *limit* touches in the last *window* seconds?"""
    return sum(1 for t in times if now - t <= window) > limit


#: The eyes follow the cursor inside this distance from her head (points)…
FOLLOW_RADIUS = 220.0
#: …except right on top of her, where she just looks at you.
FOLLOW_DEADZONE = 18.0


def follow_frame(dx: float, dy: float) -> str | None:
    """Which way she looks for a cursor at (*dx*, *dy*) points from her head
    (y up), or None to look ahead as usual (too far, too close, or below)."""
    dist = math.hypot(dx, dy)
    if dist > FOLLOW_RADIUS or dist < FOLLOW_DEADZONE:
        return None
    if dy > 0 and dy >= 0.8 * abs(dx):
        return "follow_up"
    if dy < 0 and abs(dx) < 0.5 * -dy:
        return None  # straight below her: nothing to look at
    return "follow_left" if dx < 0 else "follow_right"


# Microphone levels (the panel's spectrum peak, 0..1, smoothed) for the
# listening frames. Normal speech sits around 0.5-0.8 on that meter, so only
# a real shout reaches the blown-back ears.
LEVEL_QUIET = 0.18
VOLUME_LOUD = 0.82
VOLUME_BLAST = 0.94
#: The ear trumpet: up after this long into listening, down at the end.
MEGAPHONE_FROM = 0.25
MEGAPHONE_UNTIL = 1.8
#: Quiet this long with the trumpet up: she leans in.
LEAN_IN_AFTER = 0.7
#: Quiet this long: a paw to her ear, "no te escucho".
EAR_CUP_AFTER = 2.2


def listening_frame(
    level: float,
    ears_up: bool,
    t: float,
    quiet_for: float,
    megaphone: bool = False,
    ear_cup: bool = False,
    perk: bool = False,
) -> str:
    """The listening frame: loud wins, then the trumpet, then the cupped ear.

    *t* is seconds since listening began, *quiet_for* how long the level has
    been under :data:`LEVEL_QUIET`. *megaphone* / *ear_cup* say whether she
    does those this time (decided once per turn, so it is "sometimes").
    ``listening`` / ``listening_b`` stay the default.
    """
    if level >= VOLUME_BLAST:
        return "volume_b"
    if level >= VOLUME_LOUD:
        return "volume_a"
    if megaphone and MEGAPHONE_FROM <= t < MEGAPHONE_UNTIL:
        return "megaphone_b" if quiet_for >= LEAN_IN_AFTER else "megaphone_a"
    if ear_cup and quiet_for >= EAR_CUP_AFTER:
        return "ear_cup"
    return "listening_b" if ears_up or perk else "listening"


class MascotAnimator:
    """Animation clock: the pose, the frame, and everything she does on her own.

    Every render tick calls :meth:`update` with the assistant's state. It
    returns the :class:`Pose` and leaves the frame to draw in :attr:`frame`
    (with :attr:`prev_frame` crossfading away at :attr:`frame_mix`).

    * A state change never snaps the pose: the old motion fades into the new
      one over :attr:`FADE`. Frames swap at once (listening must react
      instantly) with only the short frame crossfade.
    * Each state has a base: idle breathes and blinks; listening pricks her
      ears as you get louder; speaking moves her mouth with her own voice;
      thinking looks from side to side; muted sleeps.
    * Idle, she gets up to things now and then (:data:`BEHAVIOUR_WEIGHTS`),
      never the same twice running, and after :data:`SLEEP_AFTER` with nobody
      around she lies down and sleeps until something happens.
    * Any state change interrupts whatever she was doing.
    * Reduce Motion: no behaviours, no blinks, no bob; each state just shows
      its frame, swapped without a crossfade.

    Without frames (``available`` empty, the default) this is exactly the
    old single-sprite animator: :attr:`frame` stays None.
    """

    FADE = 0.4
    _LEVEL_ATTACK = 0.35
    _LEVEL_DECAY = 0.08
    _VOICE_ATTACK = 0.7
    _VOICE_DECAY = 0.45
    #: Microphone level hysteresis for the ears-up frame.
    EARS_UP = 0.30
    EARS_DOWN = 0.18
    #: Voice level thresholds for the three mouths: closed, half, open.
    MOUTH_HALF = 0.18
    MOUTH_OPEN = 0.5
    #: A mouth is held at least this long, or noise would make it flicker.
    MOUTH_HOLD = 0.06
    #: Per turn, how often she raises the ear trumpet / cups her ear when
    #: you are quiet. Most turns are plain listening.
    MEGAPHONE_CHANCE = 0.3
    EAR_CUP_CHANCE = 0.5
    #: A listening frame is held at least this long (except getting LOUDER,
    #: which shows at once), so a level hovering on a threshold cannot flicker.
    LISTEN_HOLD = 0.15
    #: Two touches closer than this are one (a double-click is not two pets).
    TOUCH_DEBOUNCE = 0.4
    #: Eyes stay on one side at least this long.
    FOLLOW_HOLD = 0.3

    def __init__(
        self,
        available: Iterable[str] | None = None,
        rng: random.Random | None = None,
        sleep_after: float = SLEEP_AFTER,
    ) -> None:
        self.t = 0.0
        self.key = "idle"
        self.sleep_after = sleep_after
        self._prev_pose_key = "idle"
        self._fade_left = 0.0
        self.level = 0.0
        #: Smoothed level of her own voice; None while it is unknown.
        self.voice: float | None = None
        self.rng = rng if rng is not None else random.Random()

        #: Frame to draw now (None = no frames: draw the single sprite).
        self.frame: str | None = None
        #: Frame being crossfaded away, and how far along (1.0 = done).
        self.prev_frame: str | None = None
        self.frame_mix = 1.0
        self._xfade = FRAME_XFADE
        self._xfade_left = 0.0

        #: Name of the one-shot clip playing now (behaviour, celebration…).
        self.behaviour: str | None = None
        self._oneshot: Clip | None = None
        self._oneshot_t = 0.0
        self._then_sleep = False
        #: The one-shot playing was started by a click.
        self._reacting = False
        self._last_behaviour: str | None = None
        self._last_poke: str | None = None
        self._pending_celebration: str | None = None
        #: Seconds idle with nothing happening; sleep comes at sleep_after.
        self.idle_for = 0.0
        self.asleep = False
        self._base_t = 0.0
        self._ears_up = False
        self._mouth = "speaking_a"
        self._mouth_hold = 0.0
        self._reduce = False

        #: Seconds since anyone talked to her or touched her.
        self.ignored_for = 0.0
        self._next_wake = self._gap(WAKE_GAP)
        self._long_answer = False
        self._touches: list[float] = []
        self._last_touch_t = -1e9
        #: Being stroked by the cursor right now.
        self.petting = False
        #: Reduce Motion: a touch shows this single frame for _still_left s.
        self.still: str | None = None
        self._still_left = 0.0
        self._cursor: tuple[float, float] | None = None
        self._follow: str | None = None
        self._follow_hold = 0.0
        self._listen_frame = "listening"
        self._listen_hold = 0.0
        self._quiet_for = 0.0
        self._megaphone = False
        self._ear_cup = False

        self.frames: frozenset[str] = frozenset()
        self._clips: dict[str, Clip] = {}
        self._next_behaviour = self._gap(BEHAVIOUR_GAP)
        self._next_blink = self._gap(BLINK_GAP)
        self.set_available(available or ())

    # -- configuration -----------------------------------------------------

    def set_available(self, available: Iterable[str]) -> None:
        """Tell the animator which frames exist. Without the base frame, none
        are used (single-sprite mode)."""
        frames = frozenset(available)
        self.frames = frames if BASE_FRAME in frames else frozenset()
        self._clips = {}
        for name, clip in CLIPS.items():
            ok = clip.playable(self.frames)
            if ok is not None:
                self._clips[name] = ok
        if not self.frames:
            self.frame = self.prev_frame = None
            self.frame_mix = 1.0
            self._stop_oneshot()

    def _gap(self, span: tuple[float, float]) -> float:
        return self.rng.uniform(*span)

    @property
    def framed(self) -> bool:
        return bool(self.frames)

    def playable(self) -> list[str]:
        """Idle behaviours she can actually do with the frames present."""
        return [n for n in BEHAVIOUR_WEIGHTS if n in self._clips]

    # -- events from outside ---------------------------------------------

    def celebrate(self, big: bool = False) -> None:
        """Queue a celebration for the next time she is idle (and awake).

        "cheer" after an answer, "party" (with a jump) when a task got done;
        a party is never downgraded to a cheer by a later request.
        """
        if not self.framed:
            return
        if big or self._pending_celebration == "party":
            self._pending_celebration = "party"
        else:
            self._pending_celebration = "cheer"

    def poke(self) -> bool:
        """She was clicked somewhere without a zone of its own."""
        return self.touch(None)

    def touch(self, zone: str | None) -> bool:
        """She was clicked on *zone* (see :func:`touch_zone`). True if she
        reacted. Touches are interaction, so they also push sleep back.

        Asleep, she wakes up (stretch, yawn) and then reacts to where she was
        touched. Touched more than :data:`GRUMPY_TOUCHES` times in
        :data:`GRUMPY_WINDOW` seconds, she is fed up. Under Reduce Motion a
        zone shows its one frame, still, and nothing else.
        """
        self.idle_for = 0.0
        self.ignored_for = 0.0
        if not self.framed or self.key != "idle":
            return False
        if self._reduce:
            frame = self._still_for_zone(zone)
            if frame is None:
                return False
            self.still, self._still_left = frame, STILL_HOLD
            return True
        now = self.t
        self._touches = [t for t in self._touches if now - t <= GRUMPY_WINDOW] + [now]
        if not self.asleep and too_many_touches(self._touches, now) and "grumpy" in self._clips:
            self._touches.clear()
            self._last_touch_t = now
            return self._react("grumpy")
        if self.asleep:
            self._last_touch_t = now
            self._wake(then=self._zone_reaction(zone) if zone is not None else None)
            self._reacting = True
            return True
        if self._oneshot is not None and self._reacting and (
            self.behaviour == "grumpy" or now - self._last_touch_t < self.TOUCH_DEBOUNCE
        ):
            return False  # already reacting; a double-click is not two reactions
        name = self._zone_reaction(zone)
        if name is None:
            return False
        self._last_touch_t = now
        return self._react(name)

    def double_click(self) -> bool:
        """A double-click: a hop of joy (from asleep: wake, then hop)."""
        self.idle_for = 0.0
        self.ignored_for = 0.0
        if not self.framed or self.key != "idle" or self._reduce:
            return False
        self._touches.append(self.t)
        if self.asleep:
            self._wake(then="hop")
            self._reacting = True
            return True
        if self.behaviour == "grumpy":
            return False
        self._last_poke = "hop"
        return self._react("hop")

    def set_petting(self, on: bool) -> None:
        """The cursor is stroking her head (True) or stopped (False).

        While it lasts she melts into it (the petting loop); afterwards she is
        happy. Asleep she enjoys it without waking, and sleeps on.
        """
        on = bool(on) and self.framed and self.key == "idle"
        if on:
            self.ignored_for = 0.0
            if not self.asleep:
                self.idle_for = 0.0
        if on == self.petting:
            return
        self.petting = on
        if self._reduce:
            frame = self._still_for_zone("head") if on else None
            self.still, self._still_left = frame, (1e9 if frame else 0.0)
            return
        if on:
            if self._start("petting"):
                self._reacting = True
        else:
            if self.behaviour == "petting":
                self._stop_oneshot()
            if not self.asleep:
                self._react("petted")

    def look_at(self, offset: tuple[float, float] | None) -> None:
        """Where the cursor is from her head, in points (y up), or None."""
        self._cursor = offset

    def interrupt(self) -> None:
        """Stop whatever one-shot she is doing (she is being dragged)."""
        if self._oneshot is not None:
            then_sleep = self._then_sleep
            self._stop_oneshot()
            if then_sleep and self.key == "idle":
                self._fall_asleep()

    @property
    def excursion(self) -> float:
        """How far the dash has carried her from her spot (points, + right).
        The panel moves her WINDOW by this; it is 0 except while running."""
        if self._oneshot is None or self._reduce:
            return 0.0
        return clip_excursion(self._oneshot, self._oneshot_t)

    @property
    def mirrored(self) -> bool:
        """Draw her flipped (head on the right): running back home."""
        if self._oneshot is None or self._reduce:
            return False
        step, _u = self._oneshot.step_at(self._oneshot_t)
        return step.motion == "run_back"

    @property
    def lying(self) -> bool:
        """The frame on screen is a lying-down one (zones move with it)."""
        return self.frame in ("sleeping", "sleeping_b")

    # -- the tick ------------------------------------------------------------

    def update(
        self,
        dt: float,
        state: State,
        muted: bool = False,
        level: float = 0.0,
        reduce_motion: bool = False,
        voice: float | None = None,
    ) -> Pose:
        """Advance by *dt* seconds and return the pose to draw now.

        *voice* is the level of the speech being played (None when not
        known); see :func:`mascot_pose`. The frame is left in :attr:`frame`.
        """
        dt = max(0.0, float(dt))
        self.t += dt
        level = min(1.0, max(0.0, float(level)))
        k = self._LEVEL_ATTACK if level > self.level else self._LEVEL_DECAY
        self.level += (level - self.level) * k
        if voice is None:
            self.voice = None
        else:
            # Syllables are fast: a voice smoothed like the microphone would
            # blur into one long hop. Quick both ways, a touch quicker up.
            voice = min(1.0, max(0.0, float(voice)))
            prev = self.voice if self.voice is not None else voice
            k = self._VOICE_ATTACK if voice > prev else self._VOICE_DECAY
            self.voice = prev + (voice - prev) * k

        if reduce_motion and not self._reduce:
            self._stop_oneshot()
        if not reduce_motion and self._reduce:
            self.still = None
        self._reduce = bool(reduce_motion)

        key = mood(state, muted)
        if key != self.key:
            self._change_key(key)
        else:
            self._base_t += dt
        if self.key == "listening":
            self._quiet_for = self._quiet_for + dt if self.level < LEVEL_QUIET else 0.0

        if self.framed:
            self._run_behaviours(dt)
            name, xfade = self._choose_frame()
            self._show(name, dt, xfade)

        return self._pose(dt)

    # -- internals ---------------------------------------------------------

    def _pose_key(self) -> str:
        return "asleep" if self.asleep and self.key == "idle" else self.key

    def _pose_for(self, key: str) -> Pose:
        if self.framed:
            return framed_pose(key, self.t, self.level, self._reduce, self.voice)
        return mascot_pose(key, self.t, self.level, self._reduce, self.voice)

    def _begin_pose_fade(self, from_key: str) -> None:
        self._prev_pose_key = from_key
        self._fade_left = self.FADE

    def _pose(self, dt: float) -> Pose:
        target = self._pose_for(self._pose_key())
        if self._fade_left > 0.0 and not self._reduce:
            self._fade_left = max(0.0, self._fade_left - dt)
            before = self._pose_for(self._prev_pose_key)
            w = 1.0 - self._fade_left / self.FADE
            w = w * w * (3 - 2 * w)  # smoothstep
            target = before.lerp(target, w)
        else:
            self._fade_left = 0.0
        if self._oneshot is not None and not self._reduce:
            step, u = self._oneshot.step_at(self._oneshot_t)
            target = _apply(target, motion_overlay(step.motion, u, step.duration))
        return target

    def _change_key(self, key: str) -> None:
        old_pose_key = self._pose_key()
        was_resting = self.asleep or self.key == "muted"
        if self.key == "speaking":
            # She went on a bit: the next cheer becomes a shrug or a smirk.
            self._long_answer = self._base_t >= LONG_ANSWER
        self.key = key
        self._base_t = 0.0
        self._stop_oneshot()
        self.asleep = False
        self.idle_for = 0.0
        self.petting = False
        self.still = None
        self._follow = None
        self._begin_pose_fade(old_pose_key)
        if key in ("listening", "thinking", "speaking"):
            self.ignored_for = 0.0
        if key == "listening":
            # Someone is talking to her again: whatever she was going to
            # celebrate is old news.
            self._pending_celebration = None
            self._long_answer = False
            # Decided once per turn, so the trumpet and the cupped ear are
            # "sometimes", not a flicker in and out.
            self._megaphone = ("megaphone_a" in self.frames
                               and self.rng.random() < self.MEGAPHONE_CHANCE)
            self._ear_cup = "ear_cup" in self.frames and self.rng.random() < self.EAR_CUP_CHANCE
            self._quiet_for = 0.0
            self._listen_frame = "listening"
            self._listen_hold = 0.0
        if key == "idle":
            self._next_behaviour = self._gap(BEHAVIOUR_GAP)
            self._next_blink = self._gap(BLINK_GAP)
            if was_resting and self.framed and not self._reduce:
                # Unmuted: she gets up the way dogs do.
                self._start("wake")

    def _stop_oneshot(self) -> None:
        self._oneshot = None
        self.behaviour = None
        self._oneshot_t = 0.0
        self._then_sleep = False
        self._reacting = False

    def _start(self, name: str) -> bool:
        clip = self._clips.get(name)
        if clip is None:
            return False
        self._oneshot = clip
        self._oneshot_t = 0.0
        self.behaviour = name
        self._then_sleep = False
        self._reacting = False
        return True

    def _react(self, name: str) -> bool:
        started = self._start(name)
        self._reacting = started
        return started

    def _zone_reaction(self, zone: str | None) -> str | None:
        """The reaction to a touch on *zone*; a generic one outside zones or
        when that zone's frames are missing."""
        options = [(n, w) for n, w in ZONE_REACTIONS.get(zone or "", ()) if n in self._clips]
        if options:
            total = sum(w for _, w in options)
            r = self.rng.random() * total
            for n, w in options:
                r -= w
                if r < 0.0:
                    return n
            return options[-1][0]
        generic = [n for n in POKE_REACTIONS if n in self._clips and n != self._last_poke]
        if not generic:
            generic = [n for n in POKE_REACTIONS if n in self._clips]
        if not generic:
            return None
        name = self.rng.choice(generic)
        self._last_poke = name
        return name

    def _still_for_zone(self, zone: str | None) -> str | None:
        """Reduce Motion: the one frame that says what a touch on *zone* did."""
        for name, _w in ZONE_REACTIONS.get(zone or "", ()):
            clip = self._clips.get(name)
            if clip is not None:
                return next(s.frame for s in clip.steps if s.frame not in _RESTING)
        return None

    def _wake(self, then: str | None = None) -> None:
        """Get up (stretch, yawn) and, with *then*, go straight into that."""
        self.asleep = False
        self.idle_for = 0.0
        self._begin_pose_fade("asleep")
        self._next_behaviour = self._gap(BEHAVIOUR_GAP)
        if not self._start("wake"):
            if then is not None:
                self._start(then)
            return
        after = self._clips.get(then) if then else None
        if after is not None and self._oneshot is not None:
            self._oneshot = Clip("wake", self._oneshot.steps + after.steps, False,
                                 self._oneshot.xfade)

    def _wake_for_a_bit(self) -> bool:
        """Asleep for a while: get up on her own for one rare thing, then lie
        down again. The resting look stays asleep; this is the surprise."""
        names = [n for n in WAKE_BEHAVIOURS if n in self._clips]
        if self.ignored_for >= IGNORED_AFTER:
            names = [n for n in IRONIC if n in self._clips] or names
        name = self._weighted(names)
        if name is None:
            return False
        clip = self._clips[name]
        lie = self._clips.get("lie_down")
        self.asleep = False
        self.idle_for = 0.0
        self._begin_pose_fade("asleep")
        self._oneshot = Clip(name, clip.steps + (lie.steps if lie else ()), False, clip.xfade)
        self._oneshot_t = 0.0
        self.behaviour = name
        self._reacting = False
        self._then_sleep = True
        self._last_behaviour = name
        return True

    def _fall_asleep(self) -> None:
        self._stop_oneshot()
        self.asleep = True
        self._base_t = 0.0
        self._begin_pose_fade("idle")

    def _run_behaviours(self, dt: float) -> None:
        if self.still is not None:
            self._still_left -= dt
            if self._still_left <= 0.0:
                self.still = None
        if self.key == "idle":
            self.ignored_for += dt
        if self._oneshot is not None:
            self._oneshot_t += dt
            if self._oneshot.done(self._oneshot_t):
                then_sleep = self._then_sleep
                self._stop_oneshot()
                if then_sleep and self.key == "idle":
                    self._fall_asleep()
        if self._reduce or self.key != "idle":
            return

        if self._pending_celebration is not None:
            name = self._pending_celebration
            self._pending_celebration = None
            if name == "cheer" and self._long_answer:
                # A long answer: not pride, a shrug (or a smirk). Ironic.
                after = [n for n in ("shrug", "smug") if n in self._clips]
                if after:
                    name = self.rng.choice(after)
            self._long_answer = False
            if self.asleep:
                self.asleep = False
                self._begin_pose_fade("asleep")
            self.idle_for = 0.0
            if self._start(name):
                return

        if self.asleep:
            if self._oneshot is None and not self.petting:
                self._next_wake -= dt
                if self._next_wake <= 0.0:
                    self._next_wake = self._gap(WAKE_GAP)
                    self._wake_for_a_bit()
            return
        self.idle_for += dt
        if self._oneshot is not None:
            return
        if self.idle_for >= self.sleep_after and "sleeping" in self.frames:
            if self._start("lie_down"):
                self._then_sleep = True
            else:
                self._fall_asleep()
            return
        self._next_behaviour -= dt
        if self._next_behaviour <= 0.0:
            self._next_behaviour = self._gap(BEHAVIOUR_GAP)
            name = self._pick_behaviour()
            if name is not None:
                self._last_behaviour = name
                self._start(name)
                return
        self._next_blink -= dt
        if self._next_blink <= 0.0:
            self._next_blink = self._gap(BLINK_GAP)
            self._start("blink")

    def _pick_behaviour(self) -> str | None:
        """A weighted random behaviour, never the one she just did (unless it
        is the only one she has frames for). Ignored for a while, she only
        does the ironic ones."""
        names = self.playable()
        if self.ignored_for >= IGNORED_AFTER:
            names = [n for n in IRONIC if n in names] or names
        return self._weighted(names)

    def _weighted(self, names: list[str]) -> str | None:
        names = list(names)
        if len(names) > 1 and self._last_behaviour in names:
            names.remove(self._last_behaviour)
        if not names:
            return None
        total = sum(BEHAVIOUR_WEIGHTS.get(n, 1.0) for n in names)
        r = self.rng.random() * total
        for n in names:
            r -= BEHAVIOUR_WEIGHTS.get(n, 1.0)
            if r < 0.0:
                return n
        return names[-1]

    def _choose_frame(self) -> tuple[str | None, float]:
        """The frame for now and the crossfade to reach it with."""
        if self.still is not None and self._reduce:
            return self._resolve(self.still), 0.0
        if self._oneshot is not None and not self._reduce:
            step, _u = self._oneshot.step_at(self._oneshot_t)
            return self._resolve(step.frame), self._oneshot.xfade
        name, xfade = self._state_frame()
        return self._resolve(name), xfade

    def _state_frame(self) -> tuple[str, float]:
        key = self.key
        if self._reduce:
            still = {
                "idle": "idle_a",
                "listening": "listening",
                "thinking": "thinking",
                "speaking": "speaking_c",
                "muted": "sleeping",
            }
            return still.get(key, "idle_a"), 0.0
        if key == "muted" or (key == "idle" and self.asleep):
            clip = STATE_CLIPS["sleep"]
            return clip.frame_at(self._base_t), clip.xfade
        if key == "listening":
            if self._ears_up:
                self._ears_up = self.level > self.EARS_DOWN
            else:
                self._ears_up = self.level > self.EARS_UP
            # Quiet, she still pricks her ears now and then: she is waiting.
            perk = (self._base_t % 3.2) > 2.7
            want = listening_frame(self.level, self._ears_up, self._base_t, self._quiet_for,
                                   self._megaphone, self._ear_cup, perk)
            louder = want in ("volume_a", "volume_b") and (
                self._listen_frame != "volume_b" or want == "volume_b")
            if want != self._listen_frame and (self._listen_hold <= 0.0 or louder):
                self._listen_frame = want
                self._listen_hold = self.LISTEN_HOLD
            return self._listen_frame, 0.08
        if key == "speaking":
            return self._mouth_frame(), 0.045
        if key == "thinking":
            clip = STATE_CLIPS["thinking"]
            return clip.frame_at(self._base_t), clip.xfade
        if self._cursor is not None:
            # Awake and idle: her eyes follow the cursor when it is near.
            want = follow_frame(*self._cursor)
            if want not in self.frames:
                want = None
        else:
            want = None
        if want != self._follow and (self._follow_hold <= 0.0 or want is None
                                     and self._cursor is None):
            self._follow = want
            self._follow_hold = self.FOLLOW_HOLD
        if self._follow is not None:
            return self._follow, 0.1
        clip = STATE_CLIPS["idle"]
        return clip.frame_at(self._base_t), clip.xfade

    def _mouth_frame(self) -> str:
        """Lip-sync: closed, half open or open, by how loud she is now."""
        env = talk_envelope(self.t) if self.voice is None else self.voice
        if env >= self.MOUTH_OPEN:
            want = "speaking_b"
        elif env >= self.MOUTH_HALF:
            want = "speaking_c"
        else:
            want = "speaking_a"
        if want != self._mouth and self._mouth_hold <= 0.0:
            self._mouth = want
            self._mouth_hold = self.MOUTH_HOLD
        return self._mouth

    def _resolve(self, name: str) -> str | None:
        return resolve_frame(name, self.frames)

    def _show(self, name: str | None, dt: float, xfade: float) -> None:
        self._mouth_hold = max(0.0, self._mouth_hold - dt)
        self._listen_hold = max(0.0, self._listen_hold - dt)
        self._follow_hold = max(0.0, self._follow_hold - dt)
        if name != self.frame:
            if self.frame is None or name is None or self._reduce or xfade <= 0.0:
                self.prev_frame = None
                self._xfade_left = 0.0
            else:
                self.prev_frame = self.frame
                self._xfade = xfade
                self._xfade_left = xfade
            self.frame = name
        elif self._xfade_left > 0.0:
            self._xfade_left = max(0.0, self._xfade_left - dt)
        if self._xfade_left <= 0.0:
            self.prev_frame = None
            self.frame_mix = 1.0
        else:
            self.frame_mix = 1.0 - self._xfade_left / self._xfade


def window_size(sprite_w: float, sprite_h: float) -> tuple[float, float]:
    """Mascot window size for a sprite drawn at *sprite_w* x *sprite_h*."""
    return (sprite_w + 2 * PAD_X, sprite_h + PAD_TOP + PAD_BOTTOM)


def mascot_origin(
    panel: tuple[float, float, float, float],
    size: tuple[float, float],
    screen: tuple[float, float, float, float] | None = None,
    offset: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float]:
    """Bottom-left origin for the mascot window, beside the panel.

    She sits just outside the panel's bottom-LEFT corner, tucked in by
    :data:`OVERLAP`; when that would put her off the screen (the panel is
    against the left edge) she moves to the bottom-right corner instead.
    *offset* is where the user dragged her relative to that anchor. Origins
    are AppKit's: bottom-left, y up. All rects are ``(x, y, w, h)``.
    """
    px, py, pw, _ph = panel
    w, _h = size
    x = px - w + OVERLAP
    if screen is not None and x < screen[0]:
        x = px + pw - OVERLAP
    return (x + offset[0], py - PAD_BOTTOM + offset[1])


#: Gap between Wawi on her own and the edges of the screen, in points.
ALONE_MARGIN = 24.0

Rect = tuple[float, float, float, float]


def alone_default_origin(size: tuple[float, float], screen: Rect) -> tuple[float, float]:
    """Where she stands with no panel and no saved spot: bottom-right.

    *screen* is the main screen's VISIBLE frame (no menu bar, no Dock), so the
    margin is from what you can actually see. AppKit origins: bottom-left, y up.
    """
    sx, sy, sw, _sh = screen
    w, _h = size
    return (sx + sw - w - ALONE_MARGIN, sy + ALONE_MARGIN)


def alone_origin(
    saved: tuple[float, float] | None,
    size: tuple[float, float],
    screens: list[Rect] | tuple[Rect, ...],
    main: Rect | None = None,
) -> tuple[float, float] | None:
    """Her on-screen origin when she stands alone, always fully visible.

    A saved spot is kept on whichever screen holds her centre, nudged just far
    enough to be entirely inside it. If no screen holds it — typically the
    second monitor she was left on is unplugged — she goes back to the default
    corner of *main* rather than staying somewhere nobody can see or reach.

    Args:
        saved: The remembered origin, or None for "never placed".
        size: Her window size.
        screens: Visible frames of every connected screen.
        main: Visible frame of the main screen; the first of *screens* when
            omitted.

    Returns:
        The origin, or None when there is no screen at all to put her on.
    """
    screens = list(screens)
    home = main if main is not None else (screens[0] if screens else None)
    if home is None:
        return None
    if saved is None:
        return alone_default_origin(size, home)

    w, h = size
    cx, cy = saved[0] + w / 2.0, saved[1] + h / 2.0
    for sx, sy, sw, sh in screens:
        if sx <= cx <= sx + sw and sy <= cy <= sy + sh:
            x = min(max(saved[0], sx), sx + sw - w)
            y = min(max(saved[1], sy), sy + sh - h)
            return (x, y)
    return alone_default_origin(size, home)
