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
from typing import Callable, Iterable, NamedTuple

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
#: Idle activities (see ACTIVITIES): zoomies and panting, jumping ever higher,
#: the parachute, playing on the floor, the bird, the butterfly, the falling
#: leaf and shaking off. All optional, like every frame.
ACTIVITY_FRAMES = (
    "pant_a", "pant_b", "skid", "jump_crouch", "jump_reach", "flop_back",
    "chute_hang_a", "chute_hang_b", "chute_land", "chute_fold",
    "chew_a", "chew_b", "bat_ball_a", "bat_ball_b",
    "stalk_a", "stalk_b", "pounce", "look_up",
    "butterfly_nose", "leaf_nose", "shake_a", "shake_b",
)
#: Climbing like a dog (wawi/pet/world.py, Ladder): carrying her little
#: ladder, leaning it up, climbing it rung by rung, stepping off at the top,
#: pulling it up after her, and pleased with herself.
LADDER_FRAMES = (
    "ladder_carry", "ladder_place", "ladder_climb_a", "ladder_climb_b", "ladder_climb_c",
    "ladder_step", "ladder_pull", "ladder_proud",
)
#: The ladder itself, drawn in its own window behind her (see
#: :func:`ladder_layout`): a middle stretch that tiles (LADDER_TILE_RUNGS
#: rungs), the rounded tops of its rails, its rubber feet, and the poof it
#: vanishes in.
LADDER_PROPS = ("prop_ladder", "prop_ladder_top", "prop_ladder_foot", "prop_poof")
#: Without these she has no ladder and climbs the old way, clinging on.
LADDER_NEEDS = ("ladder_climb_a", "ladder_place", "prop_ladder")
#: Busy with a tool in the background (WORK_CLIPS, wawi/working.py).
WORK_FRAMES = (
    "camera_a", "camera_b", "carry_paper", "wrench_a", "wrench_b",
    "laptop_a", "laptop_b", "binoculars", "present",
)
#: Taking notes (wawi/notes.py): listening with her little recorder, its red
#: light on (a), off (b), and her head tilted to listen closer (c).
NOTES_FRAMES = ("notes_a", "notes_b", "notes_c")
#: Dancing to music (wawi/music.py): head tilted left / right with paws up,
#: a bob down, half a spin, paws up high.
DANCE_FRAMES = ("dance_a", "dance_b", "dance_c", "dance_d", "dance_e")
#: Catching what the owner drags onto her (wawi/drops.py): ears up and mouth
#: open while it hovers over her, the item (a little envelope) in her mouth,
#: and her nose down sniffing it on the ground.
CATCH_FRAMES = ("catch_ready", "catch_bite", "catch_sniff")
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
    # Her body on the desktop (wawi/pet/world.py): walking, climbing a side,
    # falling and landing, sitting on a window's top edge, and watching you
    # type.
    "walk_a", "walk_b", "walk_c", "walk_d", "climb_a", "climb_b", "fall", "land",
    "sit_edge", "watch", "watch_b",
    # Voice without words, and her needs (wawi/pet/needs.py).
    "bark_a", "bark_b", "howl", "eat_a", "eat_b", "drink_a", "drink_b", "hungry", "full",
    # Cursor proximity reactions; ordinary bark and wag frames remain reusable.
    "tail_wag_c", "growl", "bark_alert", "ear_twitch",
    *NOTES_FRAMES, *DANCE_FRAMES, *CATCH_FRAMES,
    # Costumes and reactions to what the sensors noticed (wawi/reactions.py).
    "firefighter_a", "firefighter_b", "firefighter_hose", "alarm",
    "detective", "detective_b", "doctor", "umbrella", "sunglasses", "scarf",
    "newspaper", "mail", "ball_bring",
    "ship_board", "ship_fly_a", "ship_fly_b", "ship_return",
    # Instincts (wawi/instinct/) and not listening (awake, ears down).
    "welcome_a", "welcome_b", "curl_focus", "lunch_ask", "late_night", "moon_yawn",
    "coffee", "think_bubble", "pattern_notebook", "ears_down",
    # Games (wawi/games/): fetching with the ball, peeking while hiding,
    # giggling when found, the hand shapes, thinking a move, winning, counting.
    "run_ball_a", "run_ball_b", "run_ball_c", "run_ball_d",
    "peek_ears", "peek_tail_a", "peek_tail_b", "giggle_a", "giggle_b",
    "rps_rock", "rps_paper", "rps_scissors", "think_paw_chin", "smug_win", "cover_eyes",
    # Idle activities, tool work and the ladder (all anchored, see LOOSE_FRAMES).
    *ACTIVITY_FRAMES, *WORK_FRAMES, *LADDER_FRAMES,
)
#: Props: small separate sprites drawn on the ground in front of her (the food
#: bowl, the ball). Not frames: they are never swapped in for her, never part
#: of the shared crop box, and any of them may be missing.
PROP_NAMES = (
    "prop_bowl_food", "prop_bowl_water", "prop_bowl_empty", "prop_bone", "prop_ball",
    # The "not listening" badge shown near her while the mic is off.
    "muted_badge",
    # Props that move on their own (PROP_PATHS): the bird, the butterfly, the
    # falling leaf and the parachute's canopy.
    "prop_bird_sit", "prop_bird_peck", "prop_bird_fly_a", "prop_bird_fly_b",
    "prop_butterfly_a", "prop_butterfly_b", "prop_leaf", "prop_chute",
    # Her ladder, in its own window behind her (wawi/pet/world.py, Ladder).
    *LADDER_PROPS,
    # The capabilities scroll's roller, embedded in the local board HTML.
    "prop_abilities_roll",
)
#: The little "not listening" icon (a folded ear with a slashed mic) drawn by
#: her head while the microphone is off. Loaded like a prop, drawn like the
#: status dot; without it the hollow grey ring stands in.
MUTED_BADGE = "muted_badge"
#: Frames that do NOT share the baseline and centre of the others (she hangs
#: off a ledge, clings to a wall, tumbles, or is a rocket in the sky). The
#: importer crops each of these to its own box and records where that box sits
#: relative to the shared one (:data:`ANCHORS_FILE`).
LOOSE_FRAMES = (
    "climb_a", "climb_b", "fall", "sit_edge", "ship_fly_a", "ship_fly_b",
    # These stand on the baseline but reach far above her head (the rocket,
    # the umbrella, the howl): anchored too, so they do not make the shared
    # box — and so every other frame's window — taller.
    "ship_board", "ship_return", "umbrella", "howl",
    # A moon, a nightcap or a thought cloud above her head.
    "moon_yawn", "late_night", "think_bubble",
    # Every activity and work frame is anchored on its own: lying, leaping,
    # hanging from a parachute or holding a laptop, none of them may grow the
    # shared box (and so shrink her everywhere else). Up on her ladder too.
    *ACTIVITY_FRAMES, *WORK_FRAMES, *LADDER_FRAMES,
)
#: Where the paws of every shared-box frame touch the ground on the source
#: canvases (1024 px).
SOURCE_BASELINE = 922
#: Frames whose ground line is elsewhere on their canvas: sit_edge sits ON a
#: ledge drawn at y≈700 with her legs hanging below it. The importer moves
#: that line onto the baseline, so the window edge she sits on is the ledge.
SOURCE_GROUND = {"sit_edge": 700}
#: Per-frame placement written by ``scripts/make_wawi_assets.py`` next to the
#: frames. See :func:`frame_rect`.
ANCHORS_FILE = "anchors.json"
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
    # New frames borrow the closest pose that exists, so a half-drawn set
    # still reads: a walk is a slow gallop, a bark is an open mouth, eating
    # is sniffing the ground. Costume frames have NO fallback on purpose: a
    # costume she does not have is simply not worn (see costume_frames).
    "walk_a": "run_a",
    "walk_b": "run_b",
    "walk_c": "run_c",
    "walk_d": "run_d",
    "climb_b": "climb_a",
    "climb_a": "stretch",
    # Without her ladder frames she has no ladder at all (LADDER_NEEDS); a
    # missing one of the set borrows its nearest sibling.
    "ladder_climb_b": "ladder_climb_a",
    "ladder_climb_c": "ladder_climb_a",
    "ladder_climb_a": "climb_a",
    "ladder_carry": "walk_a",
    "ladder_place": "ladder_climb_a",
    "ladder_step": "walk_a",
    "ladder_pull": "sniff",
    "ladder_proud": "happy",
    "fall": "jump",
    "land": "idle_a",
    "sit_edge": "idle_a",
    "watch": "stare",
    "watch_b": "watch",
    "bark_a": "speaking_b",
    "bark_b": "speaking_a",
    "tail_wag_c": "wag_b",
    "growl": "grumpy",
    "bark_alert": "bark_a",
    "ear_twitch": "sleeping_b",
    "notes_a": "listening",
    "notes_b": "notes_a",
    "notes_c": "notes_a",
    # Without the dance art she dances with the joy frames she has.
    "dance_a": "happy",
    "dance_b": "happy_b",
    "dance_c": "wag_a",
    "dance_d": "tail_chase_a",
    "dance_e": "jump",
    # Without the catch art: all ears (listening), the joy of a fetched ball,
    # and an ordinary sniff.
    "catch_ready": "listening_b",
    "catch_bite": "ball_bring",
    "catch_sniff": "sniff",
    "howl": "listening_b",
    "eat_a": "sniff",
    "eat_b": "eat_a",
    "drink_a": "eat_a",
    "drink_b": "drink_a",
    "hungry": "stare",
    "full": "happy",
    "detective_b": "sniff",
    "detective": "sniff",
    "doctor": "tilt",
    "alarm": "bark_a",
    "newspaper": "happy",
    "mail": "newspaper",
    "ball_bring": "paw_up",
    "ship_board": "jump",
    "ship_fly_a": "jump",
    "ship_fly_b": "ship_fly_a",
    "ship_return": "happy",
    # The gallop with the ball is the plain gallop (the ball then rides in
    # its own window, see wawi/games/ui.py).
    "run_ball_a": "run_a",
    "run_ball_b": "run_b",
    "run_ball_c": "run_c",
    "run_ball_d": "run_d",
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


def available_props(directory: Path | None = None) -> frozenset[str]:
    """The prop sprites present on disk (no logging: props are optional)."""
    return frozenset(n for n in PROP_NAMES if frame_path(n, directory).is_file())


Box = tuple[float, float, float, float]


def load_anchors(directory: Path | None = None) -> dict[str, Box]:
    """Per-frame boxes from :data:`ANCHORS_FILE`, or {} when there is none.

    Each entry is ``(x, y, w, h)`` in output pixels, relative to the bottom-left
    of the shared box (y up); ``"_box"`` is the shared box itself,
    ``(0, 0, w, h)``. A broken file is ignored (logged once): frames then all
    share the box, as before anchors existed.
    """
    path = (directory or ASSET_DIR) / FRAME_DIR / ANCHORS_FILE
    if not path.is_file():
        return {}
    try:
        import json

        raw = json.loads(path.read_text())
        return {
            str(k): tuple(float(x) for x in v)  # type: ignore[misc]
            for k, v in raw.items()
            if isinstance(v, (list, tuple)) and len(v) == 4
        }
    except Exception:
        if "anchors" not in _missing_logged:
            _missing_logged.add("anchors")
            log.warning("Could not read %s; frames will share one box", path)
        return {}


def frame_rect(name: str, sprite_size: tuple[float, float],
               anchors: dict[str, Box] | None = None) -> Box:
    """Where frame (or prop) *name* is drawn, in points, relative to her
    ground point (the bottom-centre of the shared box): ``(x, y, w, h)``.

    Frames that share the box fill the sprite rect. A loose frame or a prop
    with an anchor is placed and scaled exactly as the importer measured it,
    so a ledge-sitting pose can hang below the ground line and a rocket can sit
    above her head, all at the same scale as every other frame.
    """
    w, h = sprite_size
    base = (-w / 2.0, 0.0, w, h)
    if not anchors:
        return base
    box = anchors.get("_box")
    a = anchors.get(name)
    if box is None or a is None or box[2] <= 0:
        return base
    k = w / box[2]
    return (-w / 2.0 + a[0] * k, a[1] * k, a[2] * k, a[3] * k)


def prop_rect(name: str, sprite_size: tuple[float, float],
              anchors: dict[str, Box] | None = None,
              natural: tuple[float, float] | None = None) -> Box:
    """Where prop *name* stands: on the ground, in front of her paws (her head
    is on the LEFT; the view mirrors it with her). Scaled like the frames when
    the importer measured it, else to a third of her width."""
    w, h = sprite_size
    box = (anchors or {}).get("_box")
    a = (anchors or {}).get(name)
    if box is not None and a is not None and box[2] > 0:
        k = w / box[2]
        pw, ph = a[2] * k, a[3] * k
    else:
        nw, nh = natural or (1.0, 1.0)
        pw = 0.32 * w
        ph = pw * (nh / nw if nw > 0 else 1.0)
    cx = -w / 2.0 + 0.14 * w
    return (cx - pw / 2.0, 0.0, pw, ph)


def canvas_extent(rects: Iterable[Box], sprite_size: tuple[float, float],
                  cap: float = 0.8) -> tuple[float, float, float]:
    """How much room beyond the plain sprite her window needs for *rects*:
    ``(side, below, above)`` in points, each capped at *cap* x her size so one
    oversized frame cannot turn her window into a wall of glass."""
    w, h = sprite_size
    side = below = above = 0.0
    for x, y, rw, rh in rects:
        side = max(side, -w / 2.0 - x, x + rw - w / 2.0)
        below = max(below, -y)
        above = max(above, y + rh - h)
    return (min(side, cap * w), min(below, cap * h), min(above, cap * h))


def ladder_ready(frames: Iterable[str], props: Iterable[str]) -> bool:
    """She has the art to climb with her ladder (:data:`LADDER_NEEDS`)."""
    have = set(frames) | set(props)
    return all(n in have for n in LADDER_NEEDS)


#: Rungs in one ``prop_ladder`` tile (the importer keeps it whole, top to
#: bottom of its canvas, so stacked tiles join seamlessly).
LADDER_TILE_RUNGS = 5


def _piece_size(name: str, sprite_size: tuple[float, float],
                anchors: dict[str, Box] | None) -> tuple[float, float] | None:
    box = (anchors or {}).get("_box")
    a = (anchors or {}).get(name)
    if box is None or a is None or box[2] <= 0:
        return None
    k = sprite_size[0] / box[2]
    return (a[2] * k, a[3] * k)


def ladder_rung(sprite_size: tuple[float, float], anchors: dict[str, Box] | None) -> float:
    """The drawn ladder's rung spacing in points (0 without the art)."""
    mid = _piece_size("prop_ladder", sprite_size, anchors)
    return mid[1] / LADDER_TILE_RUNGS if mid else 0.0


def ladder_layout(lo: float, hi: float, sprite_size: tuple[float, float],
                  anchors: dict[str, Box] | None,
                  have: Iterable[str] = LADDER_PROPS) -> list[tuple[str, Box, Box]]:
    """How to draw the part of the ladder from *lo* to *hi* points above its
    foot: ``(prop, rect, clip)`` triples, rects and clips relative to its
    foot on its centre line (y up).

    The middle tile repeats from the foot up on a fixed grid of rungs, so it
    always reaches however tall the climb is; what shows is cut on that
    grid (a telescoping ladder comes out and goes back in a rung at a time),
    the top ends just above a rung with the rails' rounded caps, and the
    rubber feet stand at the bottom when the bottom is on the floor. [] when
    there is nothing to draw (no art, or less than a rung showing).
    """
    names = set(have)
    mid = _piece_size("prop_ladder", sprite_size, anchors) if "prop_ladder" in names else None
    if mid is None or hi - lo <= 0.5:
        return []
    mw, mh = mid
    rung = mh / LADDER_TILE_RUNGS
    # Rungs sit at (k + 0.5) * rung; the rails end 0.45 rung above the last.
    top = (max(1, round(hi / rung + 0.05)) - 0.05) * rung
    bottom = max(0.0, math.floor(lo / rung + 0.05) * rung)
    if top - bottom < 0.9 * rung:
        return []
    out: list[tuple[str, Box, Box]] = []
    clip = (-mw, bottom, 2 * mw, top - bottom)
    i = int(bottom // mh)
    while i * mh < top:
        out.append(("prop_ladder", (-mw / 2.0, i * mh, mw, mh), clip))
        i += 1
    whole = (-mw * 2, -mh, mw * 4, top + 2 * mh)
    caps = _piece_size("prop_ladder_top", sprite_size, anchors) if "prop_ladder_top" in names \
        else None
    if caps is not None:
        cw, ch = caps
        out.append(("prop_ladder_top", (-cw / 2.0, top - ch * 0.55, cw, ch), whole))
    feet = _piece_size("prop_ladder_foot", sprite_size, anchors) \
        if "prop_ladder_foot" in names and bottom <= 0.0 else None
    if feet is not None:
        fw, fh = feet
        out.append(("prop_ladder_foot", (-fw / 2.0, 0.0, fw, fh), whole))
    return out


def poof_rect(progress: float, sprite_size: tuple[float, float],
              anchors: dict[str, Box] | None) -> tuple[Box, float] | None:
    """The poof the ladder vanishes in, *progress* 0..1 through it: its rect
    (centred on the ladder's line, around the given height) and its alpha.
    None without the art."""
    size = _piece_size("prop_poof", sprite_size, anchors)
    if size is None or progress < 0.0:
        return None
    pw, ph = size
    g = 0.7 + 0.5 * min(1.0, progress / 0.6)
    alpha = 1.0 if progress < 0.45 else max(0.0, (1.0 - progress) / 0.55)
    return ((-pw * g / 2.0, -ph * g * 0.35, pw * g, ph * g), alpha)


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

#: Syllable rate of the fallback talking rhythm, in Hz. Close to real speech
#: (four to five syllables a second reads as chatter; three reads as talking).
_TALK_HZ = 3.1


def mood(state: State, muted: bool = False) -> str:
    """The animation key: the assistant's state, and nothing else.

    *muted* is accepted and deliberately ignored. Muting used to win over
    everything (she lay down, dimmed, and stopped doing things), which made
    "stop listening" look like "turn Wawi off". The microphone gate only
    decides whether she HEARS; her life goes on, and the mute is shown by
    :func:`badge` instead.
    """
    del muted
    return state.value.lower() if isinstance(state, State) else str(state).lower()


def badge(key: str, muted: bool = False, musing: bool = False) -> str | None:
    """Which tiny status mark to draw by her head, if any.

    Without the panel she is the only thing on screen, and the two states a
    pose alone can leave in doubt are the ones that matter most: "is it
    recording me right now?" and "is the microphone off?". Everything else is
    obvious from the motion, so it gets no mark. The mute wins: it holds
    whatever she is doing, asleep or playing, and is the one thing nothing
    else on screen says.

    *musing* (wawi/working.musing: work she started on her own, like fixing
    herself) is the quietest: three little dots, only when nothing more
    important is up and she is not already showing she is thinking.
    """
    if muted:
        return "muted"
    if key == "listening":
        return key
    if musing and key not in ("thinking", "speaking"):
        return "musing"
    return None


#: The "…" of :func:`badge` "musing": one wave across the three dots, this long.
MUSING_PERIOD_S = 1.2


def musing_dots(t: float) -> tuple[float, float, float]:
    """How high each of the three "…" dots is at time *t*, 0..1: a wave that
    runs left to right, one dot up at a time, like someone typing."""
    u = (t % MUSING_PERIOD_S) / MUSING_PERIOD_S
    return tuple(_bump((u - i * 0.22) / 0.4) for i in range(3))  # type: ignore[return-value]


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

    Amplitudes are in points at :data:`SPRITE_WIDTH`.

    Args:
        key: ``idle``, ``listening``, ``thinking`` or ``speaking``.
        t: Seconds on the animation clock.
        level: Live microphone level 0..1 (only ``listening`` uses it).
        reduce_motion: The system's Reduce Motion setting: hold still.
        voice: Level 0..1 of what she is SAYING right now (only ``speaking``
            uses it). None when unknown: a synthetic talking rhythm stands in.
    """
    if reduce_motion:
        return REST

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
    ``asleep`` is idle-asleep.
    """
    if reduce_motion:
        return REST
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
    if key == "asleep":
        return Pose(sy=1.0 + 0.006 * slow)
    # idle: the idle_a/idle_b frames breathe; this only keeps her soft.
    return Pose(sy=1.0 + 0.005 * breathe)



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
    #: A prop shown on the ground in front of her during this step.
    prop: str | None = None
    #: How that prop moves on its own (a bird flying in, the parachute's
    #: canopy swaying): a :data:`PROP_PATHS` name, optionally with the part
    #: of the path this step covers, ``"bird_in@2/6"``. None: it stands still
    #: in front of her paws.
    path: str | None = None


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

    def resolved(self, available: Iterable[str]) -> "Clip | None":
        """This clip with each missing frame replaced by its nearest present
        relative (:data:`FRAME_FALLBACK`), or None if it would show nothing of
        its own. Used for acts and locomotion, which must happen even before
        their own art exists (a walk is a slow gallop until walk_a is drawn)."""
        have = available if isinstance(available, (set, frozenset)) else set(available)
        steps = []
        for s in self.steps:
            frame = resolve_frame(s.frame, have)
            if frame is not None:
                steps.append(s._replace(frame=frame))
        if not any(s.frame not in _RESTING for s in steps):
            return None
        return Clip(self.name, tuple(steps), self.loop, self.xfade)


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
    "moon_yawn": Clip("moon_yawn", (Step("moon_yawn", 2.2, "yawn"),), xfade=0.15),
    "late_night": Clip("late_night", (Step("late_night", 2.0, "breathe"),), xfade=0.15),
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
    "cursor_wag": Clip("cursor_wag", (Step("wag_a", 0.32), Step("tail_wag_c", 0.32),
                                         Step("wag_b", 0.32), Step("tail_wag_c", 0.32))),
    "cursor_growl": Clip("cursor_growl", (Step("growl", 0.75),), xfade=0.08),
    "cursor_bark": Clip("cursor_bark", (Step("bark_alert", 0.4, "hop"),
                                           Step("bark_a", 0.3)), xfade=0.05),
    "cursor_ear_twitch": Clip("cursor_ear_twitch", (Step("ear_twitch", 0.35),
                                                 Step("sleeping", 0.35)), xfade=0.08),
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
    # Microphone off: now and then she folds her ears down for a while. Only
    # ever picked while muted (MascotAnimator._pick_behaviour), never forced.
    "ears_down": Clip("ears_down", (Step("ears_down", 3.5, "breathe"),), xfade=0.3),
    "paw_up": Clip("paw_up", (Step("paw_up", 1.7),), xfade=0.12),
    "grumpy": Clip("grumpy", (Step("grumpy", 2.2, "huff"),), xfade=0.12),
    # -- games (wawi/games/) -------------------------------------------------
    # Found while hiding: a paw over her mouth, giggling.
    "giggle": Clip(
        "giggle", _alt("giggle_a", "giggle_b", 0.16, 3, "wiggle") + (Step("giggle_a", 0.5),),
        xfade=0.06,
    ),
    # Paws over her eyes while she counts (escondidas' count is 3.5 s).
    "cover_eyes": Clip("cover_eyes", (Step("cover_eyes", 3.6),), xfade=0.15),
    # Her hand in piedra, papel o tijera.
    "rps_rock": Clip("rps_rock", (Step("rps_rock", 1.5, "lean"),), xfade=0.08),
    "rps_paper": Clip("rps_paper", (Step("rps_paper", 1.5, "lean"),), xfade=0.08),
    "rps_scissors": Clip("rps_scissors", (Step("rps_scissors", 1.5, "lean"),), xfade=0.08),
    # Paw on her chin, thinking her ta-te-tí move.
    "think_paw_chin": Clip("think_paw_chin", (Step("think_paw_chin", 1.4, "tilt"),), xfade=0.12),
    # A whole match won: the trophy.
    "smug_win": Clip("smug_win", (Step("smug_win", 2.4, "chin_up"),), xfade=0.12),
}

_WALK_CYCLE = ("walk_a", "walk_b", "walk_c", "walk_d")
_WALK_STEP = 0.16
RUN_BALL_CYCLE = ("run_ball_a", "run_ball_b", "run_ball_c", "run_ball_d")
#: Each frame of the climb up the ladder (two per rung).
LADDER_STEP = 0.38
_FOOD = "prop_bowl_food"
_WATER = "prop_bowl_water"
_EMPTY = "prop_bowl_empty"
_BALL = "prop_ball"

#: Locomotion: loops shown while her BODY moves across the desktop
#: (wawi/pet/world.py). They win over everything, even a voice turn: a dog in
#: mid-air cannot stop to listen.
BODY_CLIPS: dict[str, Clip] = {
    "walk": Clip("walk", tuple(Step(f, _WALK_STEP, "walk") for f in _WALK_CYCLE),
                 loop=True, xfade=0.06),
    "trot": Clip("trot", tuple(Step(f, _RUN_STEP, "gallop") for f in _RUN_CYCLE),
                 loop=True, xfade=0.04),
    "climb": Clip("climb", (Step("climb_a", 0.3, "climb"), Step("climb_b", 0.3, "climb")),
                  loop=True, xfade=0.08),
    "fall": Clip("fall", (Step("fall", 1.0, "tumble"),), loop=True, xfade=0.05),
    "land": Clip("land", (Step("land", 0.3, "squash"), Step("idle_a", 0.15)), xfade=0.05),
    "fly": Clip("fly", _alt("ship_fly_a", "ship_fly_b", 0.12, 1, "wobble"), loop=True,
                xfade=0.05),
    # The gallop with the ball in her mouth (fetch, bringing it back).
    "trot_ball": Clip("trot_ball", tuple(Step(f, _RUN_STEP, "gallop") for f in RUN_BALL_CYCLE),
                      loop=True, xfade=0.04),
    # Floating down off a window's edge under the parachute (wawi/pet/life.py).
    "chute": Clip("chute", (Step("chute_hang_a", 0.6, "chute_l", "prop_chute", "chute_l"),
                            Step("chute_hang_b", 0.6, "chute_r", "prop_chute", "chute_r")),
                  loop=True, xfade=0.08),
    # The ladder (wawi/pet/world.py, Ladder): to the foot of the climb with
    # it in her mouth, leaning it up, up it paw over paw, off the top onto
    # the edge, and pulling it up after her (or folding it at its foot).
    "ladder_carry": Clip("ladder_carry", (Step("ladder_carry", 0.32, "walk"),), loop=True,
                         xfade=0.06),
    "ladder_set": Clip("ladder_set", (Step("ladder_place", 0.7, "lean"),
                                      Step("ladder_place", 0.7, "breathe")),
                       loop=True, xfade=0.1),
    "ladder": Clip("ladder", (Step("ladder_climb_a", LADDER_STEP, "rung"),
                              Step("ladder_climb_c", LADDER_STEP, "rung"),
                              Step("ladder_climb_b", LADDER_STEP, "rung"),
                              Step("ladder_climb_c", LADDER_STEP, "rung")),
                   loop=True, xfade=0.07),
    "ladder_step": Clip("ladder_step", (Step("ladder_step", 0.6, "proud"),), loop=True,
                        xfade=0.08),
    "ladder_pull": Clip("ladder_pull", (Step("ladder_pull", 0.5, "tug"),), loop=True,
                        xfade=0.08),
    "ladder_stow": Clip("ladder_stow", (Step("ladder_place", 0.6, "lean"),), loop=True,
                        xfade=0.08),
    "ladder_held": Clip("ladder_held", (Step("ladder_proud", 1.0, "proud"),), loop=True,
                        xfade=0.12),
}

#: Acts: what she does for a reaction (wawi/reactions.py) or an interaction
#: (food, water, the ball). Never picked at random unless a mood asks for one
#: (wawi/pet/needs.py); missing frames borrow a relative.
ACT_CLIPS: dict[str, Clip] = {
    "bark": Clip("bark", _alt("bark_a", "bark_b", 0.17, 3, "bark"), xfade=0.04),
    "howl": Clip("howl", (Step("howl", 2.4, "howl"),), xfade=0.15),
    "alarm": Clip("alarm", _alt("alarm", "bark_a", 0.25, 4, "bark"), xfade=0.05),
    "eat": Clip(
        "eat",
        tuple(Step(f, 0.32, "munch", _FOOD) for _ in range(5) for f in ("eat_a", "eat_b"))
        + (Step("full", 1.8, "proud", _EMPTY),),
        xfade=0.06,
    ),
    "drink": Clip(
        "drink",
        tuple(Step(f, 0.24, "lap", _WATER) for _ in range(5) for f in ("drink_a", "drink_b"))
        + (Step("full", 1.4, "proud", _WATER),),
        xfade=0.05,
    ),
    "play": Clip(
        "play",
        (Step("ball_bring", 1.1, "lean", _BALL), Step("jump", 0.42, "hop", _BALL))
        + tuple(Step(f, _RUN_STEP, "bounce", _BALL) for f in _RUN_CYCLE)
        + (Step("happy", 0.4, "bounce", _BALL), Step("jump", 0.42, "hop", _BALL),
           Step("happy_b", 0.6, "bounce", _BALL)),
        xfade=0.05,
    ),
    "hungry_bowl": Clip("hungry_bowl", (Step("hungry", 2.8, "lean", _EMPTY),), xfade=0.15),
    "watch": Clip("watch", (Step("watch", 2.6), Step("watch_b", 1.8)), loop=True, xfade=0.3),
    "detective": Clip(
        "detective",
        (Step("detective", 1.5, "glance"), Step("detective_b", 1.4, "sniff"),
         Step("detective", 1.2, "glance")),
        xfade=0.15,
    ),
    "doctor": Clip("doctor", (Step("doctor", 2.6, "tilt"),), xfade=0.15),
    "newspaper": Clip("newspaper", (Step("newspaper", 2.4, "proud"),), xfade=0.15),
    "mail": Clip("mail", (Step("mail", 2.4, "proud"),), xfade=0.15),
    "ball_bring": Clip("ball_bring", (Step("ball_bring", 2.6, "lean"),), xfade=0.15),
    "ship_board": Clip("ship_board", (Step("ship_board", 1.6, "wiggle"),), xfade=0.1),
    "ship_return": Clip("ship_return", (Step("ship_return", 2.4, "proud"),), xfade=0.12),
    "sad": Clip("sad", (Step("ear_flop", 0.9, "droop"), Step("grumpy", 2.0, "droop")),
                xfade=0.2),
    # Watching you type, now and then: …really?
    "watch_side_eye": Clip("watch_side_eye", (Step("side_eye", 1.6, "glance"),), xfade=0.15),
    "watch_eyeroll": Clip("watch_eyeroll", (Step("eyeroll", 1.5, "roll_eyes"),), xfade=0.15),
    # Asleep on a window's top edge she lies down like anywhere else; awake
    # she sits on it with her legs over the side.
    "sit_edge": Clip("sit_edge", (Step("sit_edge", 2.0, "breathe"),), loop=True, xfade=0.3),
    # The end of the desktop versions of two activities (wawi/pet/life.py):
    # landed under the parachute and folding it; back from the zoomies.
    "chute_end": Clip("chute_end", (Step("chute_land", 0.7, "squash"),
                                    Step("chute_fold", 1.6, "proud")), xfade=0.1),
    "zoomies_end": Clip("zoomies_end", (Step("skid", 0.4, "skid"),)
                        + _alt("pant_a", "pant_b", 0.28, 5, "pant"), xfade=0.05),
}

#: Two-frame costumes have a run cycle: ``<costume>_a`` / ``<costume>_b``.
COSTUME_RUN = ("_a", "_b")


def costume_frames(costume: str | None, available: Iterable[str]) -> tuple[str, ...]:
    """The frames of *costume* that exist (``firefighter`` → ``firefighter_a``,
    ``firefighter_b``, ``firefighter_hose``), sorted; () when she has none.

    Only frames she really has count — a costume is never faked with a
    fallback: no umbrella drawn means no umbrella worn.
    """
    if not costume:
        return ()
    c = costume.strip().lower()
    if not c or not c.replace("_", "").isalnum():
        return ()
    have = set(available)
    return tuple(sorted(n for n in FRAME_NAMES
                        if n in have and (n == c or n.startswith(c + "_"))))


def costume_clip(costume: str, available: Iterable[str], running: bool = False) -> Clip | None:
    """A loop of *costume*'s frames, worn for as long as a reaction lasts.

    *running* asks for its run cycle (``<costume>_a`` / ``_b``) when it has
    one. None when she has no frame of that costume at all.
    """
    frames = costume_frames(costume, available)
    if not frames:
        return None
    c = costume.strip().lower()
    run = [f"{c}{s}" for s in COSTUME_RUN if f"{c}{s}" in frames]
    if running and len(run) == 2:
        return Clip(f"costume:{c}:run",
                    tuple(Step(f, _RUN_STEP * 1.4, "gallop") for f in run * 2),
                    loop=True, xfade=0.04)
    still = [f for f in frames if f not in run] or frames
    return Clip(f"costume:{c}", tuple(Step(f, 1.6, "breathe") for f in still),
                loop=True, xfade=0.35)



# ---------------------------------------------------------------------------
# Idle activities: longer things she gets up to on her own, now and then
# ---------------------------------------------------------------------------

def _cycle(frames: tuple[str, ...], each: float, n: int, motion: str | None = None,
           prop: str | None = None, path: str | None = None,
           props: tuple[str, ...] | None = None) -> tuple[Step, ...]:
    """*n* steps cycling through *frames*, each held *each* seconds. With
    *path*, step i covers part i/n of it (``"bird_in@3/8"``), so a prop
    travels smoothly across many short steps; *props* cycles the prop too (a
    bird flapping its wings)."""
    out = []
    for i in range(n):
        p = props[i % len(props)] if props else prop
        out.append(Step(frames[i % len(frames)], each, motion, p,
                        f"{path}@{i}/{n}" if path else None))
    return tuple(out)


#: Zoomies: each frame of the sprint, and how far each way she goes (points,
#: by the panel: the window moves; on the desktop her body does, further).
ZOOM_STEP = 0.07
ZOOM_DISTANCE = 64.0
_ZOOM_SPEED = ZOOM_DISTANCE / (2 * len(_RUN_CYCLE) * ZOOM_STEP)
#: Jump attempts: how high each try goes (points, rising), and how long it
#: stays in the air.
LEAP_HEIGHTS = (14.0, 28.0, 44.0, 62.0)
_LEAP_TIMES = (0.42, 0.5, 0.58, 0.66)
#: The parachute without a window to jump off: how high she leaps first.
CHUTE_TOP = 150.0
#: Where the canopy hangs, from the prop's resting spot: in her widths
#: (sideways) and heights (up). Its strings meet at her raised paws.
CHUTE_DX = 0.36
CHUTE_DY = 0.99
#: Where the bird lands, from the prop's resting spot (her widths; negative
#: is ahead of her: her head is on the left).
BIRD_SPOT = -0.30
#: Stalking: each sneaky step carries her this far (points); the pounce this
#: far. She walks the whole way back home afterwards.
CREEP_STEP = 5.0
POUNCE_DISTANCE = 16.0
#: Her nose in the sitting frame, from the prop's resting spot (widths,
#: heights): where the butterfly and the leaf land.
NOSE = (0.13, 0.33)

_BIRD_FLAP = ("prop_bird_fly_a", "prop_bird_fly_b")
_BUTTERFLY = ("prop_butterfly_a", "prop_butterfly_b")


def _zoom(motion: str, laps: int) -> tuple[Step, ...]:
    return tuple(Step(f, ZOOM_STEP, motion) for _ in range(laps) for f in _RUN_CYCLE)


def _jump_tries(flop: bool) -> tuple[Step, ...]:
    """Four tries, each higher than the last; the last one lands (to be
    celebrated) or ends on her back (*flop*)."""
    out: list[Step] = []
    last = len(LEAP_HEIGHTS) - 1
    for i, air in enumerate(_LEAP_TIMES):
        out.append(Step("jump_crouch", 0.45 + 0.1 * i, "crouch"))
        out.append(Step("jump_reach", air, f"leap{i}"))
        if i == last and flop:
            out.append(Step("flop_back", 2.0, "dizzy"))
            out.append(Step("idle_a", 0.3))
            break
        out.append(Step("land", 0.22, "squash"))
        if i < last:
            out.append(Step("look_up", 0.55, "glance"))
    return tuple(out)


@dataclass(frozen=True)
class Activity:
    """An idle activity: its clip, how often she picks it, the art it cannot
    do without (frames and props: missing any, it is never picked), the
    group sharing its cooldown, and whether it needs her window lifted off
    the ground (only with her body running, see wawi/pet/life.py)."""

    clip: Clip
    weight: float
    needs: tuple[str, ...]
    group: str
    cooldown: float
    lift: bool = False


ACTIVITIES: dict[str, Activity] = {
    # Zoomies: a mad sprint one way, skid, the other way, skid, home, and
    # then panting. On the desktop her body runs it (wawi/pet/life.py).
    "zoomies": Activity(
        Clip("zoomies",
             _zoom("zoom_out", 2) + (Step("skid", 0.4, "skid"),)
             + _zoom("zoom_back", 4) + (Step("skid", 0.4, "skid_back"),)
             + _zoom("zoom_out", 2)
             + _alt("pant_a", "pant_b", 0.28, 5, "pant") + (Step("pant_a", 0.4, "pant"),),
             xfade=0.04),
        0.7, ("run_a", "pant_a"), "zoomies", 240.0),
    # Trying to jump, each try higher; the last one is a triumph…
    "jump_tries": Activity(
        Clip("jump_tries",
             _jump_tries(flop=False)
             + _alt("happy", "happy_b", 0.18, 3, "bounce") + (Step("happy", 0.5),),
             xfade=0.05),
        0.4, ("jump_crouch", "jump_reach"), "jumps", 240.0, lift=True),
    # …or ends flat on her back, seeing stars.
    "jump_flop": Activity(
        Clip("jump_flop", _jump_tries(flop=True), xfade=0.05),
        0.4, ("jump_crouch", "jump_reach", "flop_back"), "jumps", 240.0, lift=True),
    # Up she goes, the parachute pops open, she floats down swaying, lands
    # and folds it. Off a window's edge, her body falls with it instead.
    "parachute": Activity(
        Clip("parachute",
             (Step("jump_crouch", 0.5, "crouch"),
              Step("jump_reach", 0.55, "chute_up"),
              Step("chute_hang_a", 0.35, "chute_pop", "prop_chute", "chute_open"))
             + tuple(Step("chute_hang_a" if i % 2 == 0 else "chute_hang_b", 0.6,
                          "chute_l" if i % 2 == 0 else "chute_r", "prop_chute",
                          "chute_l" if i % 2 == 0 else "chute_r") for i in range(8))
             + (Step("chute_land", 0.7, "squash"), Step("chute_fold", 1.6, "proud")),
             xfade=0.08),
        0.5, ("jump_reach", "chute_hang_a", "prop_chute"), "parachute", 300.0, lift=True),
    # Playing on the floor: a bone between her paws…
    "play_chew": Activity(
        Clip("play_chew",
             (Step("chew_a", 0.5),) + _alt("chew_a", "chew_b", 0.32, 6, "munch")
             + (Step("chew_a", 0.9, "breathe"),),
             xfade=0.08),
        0.8, ("chew_a",), "floor", 150.0),
    # …or on her back, batting a ball, then a roll.
    "play_ball": Activity(
        Clip("play_ball",
             _alt("bat_ball_a", "bat_ball_b", 0.28, 6, "paw_bat") + (Step("roll", 0.9, "wobble"),),
             xfade=0.06),
        0.8, ("bat_ball_a",), "floor", 150.0),
    # A bird lands in front of her. She stalks it, pounces, it flies off
    # over her head; she watches it go, barks, and walks back to her spot.
    "bird_chase": Activity(
        Clip("bird_chase",
             _cycle(("look_left",), 0.12, 9, None, path="bird_in", props=_BIRD_FLAP)
             + (Step("look_left", 0.5, "glance", "prop_bird_sit", "bird_ground"),
                Step("stalk_a", 0.5, "creep", "prop_bird_peck", "bird_ground"),
                Step("stalk_b", 0.5, "creep", "prop_bird_sit", "bird_ground"),
                Step("stalk_a", 0.5, "creep", "prop_bird_peck", "bird_ground"),
                Step("stalk_b", 0.5, "creep", "prop_bird_sit", "bird_ground"),
                Step("stalk_a", 0.6, "wiggle", "prop_bird_sit", "bird_ground"),
                Step("pounce", 0.42, "pounce", "prop_bird_fly_a", "bird_off@0/8"),
                Step("land", 0.2, "squash", "prop_bird_fly_b", "bird_off@1/8"))
             + tuple(Step("look_up", 0.16, None, _BIRD_FLAP[i % 2], f"bird_off@{i}/8")
                     for i in range(2, 8))
             + _alt("bark_a", "bark_b", 0.17, 2, "bark")
             + tuple(Step(f, _WALK_STEP, "walk_home") for _ in range(2) for f in _WALK_CYCLE),
             xfade=0.05),
        0.7, ("stalk_a", "pounce", "prop_bird_sit", "prop_bird_fly_a"), "bird", 240.0),
    # A butterfly flutters round her head and lands on her nose.
    "butterfly": Activity(
        Clip("butterfly",
             _cycle(("look_up",), 0.1, 24, None, path="flutter", props=_BUTTERFLY)
             + (Step("butterfly_nose", 1.8, "breathe"),)
             + _cycle(("look_up",), 0.1, 12, None, path="flutter_off", props=_BUTTERFLY)
             + _alt("wag_a", "wag_b", 0.15, 2, "wag"),
             xfade=0.06),
        0.6, ("butterfly_nose", "prop_butterfly_a"), "butterfly", 180.0),
    # A leaf drifts down onto her nose… and she sneezes it off.
    "catch_leaf": Activity(
        Clip("catch_leaf",
             _cycle(("look_up",), 0.3, 8, None, prop="prop_leaf", path="leaf_fall")
             + (Step("leaf_nose", 1.3, "tilt"),
                Step("sneeze", 0.6, "sneeze", "prop_leaf", "leaf_drop@0/2"),
                Step("happy", 0.8, "bounce", "prop_leaf", "leaf_drop@1/2")),
             xfade=0.06),
        0.6, ("leaf_nose", "prop_leaf"), "leaf", 180.0),
    # Shaking herself off like a wet dog.
    "shake": Activity(
        Clip("shake", _alt("shake_a", "shake_b", 0.09, 7, "shake") + (Step("idle_a", 0.3),),
             xfade=0.03),
        0.9, ("shake_a",), "shake", 120.0),
}

#: Steps whose motion faces her the other way (head on the right).
_MIRRORED_MOTIONS = frozenset({"run_back", "zoom_back", "skid_back", "walk_home",
                                "twirl_back"})
#: Motions that carry her at a steady speed (points/s, + is right)…
_SPEEDS: dict[str, float] = {
    "run_out": -_RUN_SPEED, "run_back": _RUN_SPEED,
    "zoom_out": -_ZOOM_SPEED, "zoom_back": _ZOOM_SPEED,
}
#: …and the ones that carry her a set distance over their step (points).
_TRAVEL: dict[str, float] = {"creep": -CREEP_STEP, "pounce": -POUNCE_DISTANCE}


def _smooth(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def _fade_out(p: float, start: float) -> float:
    return 1.0 if p <= start else max(0.0, (1.0 - p) / max(1e-6, 1.0 - start))


def _bird_in(p: float) -> tuple[float, float, float]:
    e = 1.0 - (1.0 - p) ** 2
    return (BIRD_SPOT - 0.4 * (1.0 - e), 0.75 * (1.0 - e) + 0.05 * math.sin(math.pi * p),
            min(1.0, p / 0.2))


def _bird_off(p: float) -> tuple[float, float, float]:
    e = p ** 1.4
    return (BIRD_SPOT + 1.05 * e, 0.95 * e, _fade_out(p, 0.55))


def _flutter(p: float) -> tuple[float, float, float]:
    a = 2 * math.pi * 1.5 * p
    x = NOSE[0] + 0.32 * math.sin(a)
    y = 0.95 + 0.12 * math.sin(2 * a)
    k = _smooth((p - 0.7) / 0.3)
    return (x + (NOSE[0] - x) * k, y + (NOSE[1] - y) * k, min(1.0, p / 0.15))


def _flutter_off(p: float) -> tuple[float, float, float]:
    return (NOSE[0] + 0.55 * p + 0.06 * math.sin(8 * math.pi * p), NOSE[1] + 0.8 * p,
            _fade_out(p, 0.6))


def _leaf_fall(p: float) -> tuple[float, float, float]:
    return (NOSE[0] + 0.28 * math.cos(3 * math.pi * p) * (1.0 - p),
            1.05 * (1.0 - p) + NOSE[1] * p, min(1.0, p / 0.15))


def _leaf_drop(p: float) -> tuple[float, float, float]:
    return (NOSE[0] - 0.12 * p, NOSE[1] * (1.0 - p) ** 2, _fade_out(p, 0.75))


#: Props that move on their own: name → position along it, 0..1 →
#: ``(dx, dy, alpha)``, dx in her widths and dy in her heights from the
#: prop's resting spot on the ground in front of her paws.
PROP_PATHS: dict[str, Callable[[float], tuple[float, float, float]]] = {
    "bird_in": _bird_in,
    "bird_ground": lambda p: (BIRD_SPOT, 0.0, 1.0),
    "bird_off": _bird_off,
    "flutter": _flutter,
    "flutter_off": _flutter_off,
    "leaf_fall": _leaf_fall,
    "leaf_drop": _leaf_drop,
    "chute_open": lambda p: (CHUTE_DX, CHUTE_DY - 0.12 * (1.0 - _smooth(p)), min(1.0, 2.5 * p)),
    "chute_l": lambda p: (CHUTE_DX + 0.03 * math.sin(math.pi * p), CHUTE_DY, 1.0),
    "chute_r": lambda p: (CHUTE_DX - 0.03 * math.sin(math.pi * p), CHUTE_DY, 1.0),
}
#: Paths that belong to the WORLD, not to her: the bird stays where it
#: landed while she creeps towards it (see :func:`prop_draw_rect`).
WORLD_PATHS = frozenset({"bird_in", "bird_ground", "bird_off"})


def prop_place(path: str | None, u: float, d: float) -> tuple[float, float, float, bool]:
    """Where a prop on *path* is at time *u* of a *d*-second step:
    ``(dx, dy, alpha, world)`` (see :data:`PROP_PATHS`). No path: resting."""
    if not path:
        return (0.0, 0.0, 1.0, False)
    name, _, part = path.partition("@")
    fn = PROP_PATHS.get(name)
    if fn is None:
        return (0.0, 0.0, 1.0, False)
    p = u / d if d > 0 else 1.0
    if part:
        try:
            i, n = (int(x) for x in part.split("/"))
            p = (i + min(1.0, max(0.0, p))) / max(1, n)
        except ValueError:
            pass
    dx, dy, alpha = fn(min(1.0, max(0.0, p)))
    return (dx, dy, min(1.0, max(0.0, alpha)), name in WORLD_PATHS)


def prop_draw_rect(name: str, sprite_size: tuple[float, float],
                   anchors: dict[str, Box] | None, natural: tuple[float, float] | None,
                   place: tuple[float, float, float, bool] | None,
                   excursion: float = 0.0) -> tuple[float, float, float, float, float]:
    """Where to draw prop *name* now: ``(x, y, w, h, alpha)`` relative to her
    ground point, like :func:`prop_rect`, moved along its path. A world prop
    (the bird) is held still in the world while the window carries her by
    *excursion* points."""
    x, y, w, h = prop_rect(name, sprite_size, anchors, natural)
    if place is None:
        return (x, y, w, h, 1.0)
    dx, dy, alpha, world = place
    sw, sh = sprite_size
    x += dx * sw - (excursion if world else 0.0)
    return (x, y + dy * sh, w, h, alpha)


def prop_reach_rects(sprite_size: tuple[float, float], anchors: dict[str, Box] | None,
                     props: Iterable[str], samples: int = 6) -> list[Box]:
    """Every place a moving prop present in *props* goes, as rects: what her
    window must make room for (:func:`canvas_extent`)."""
    have = set(props)
    out: list[Box] = []
    clips = [a.clip for a in ACTIVITIES.values()] + list(BODY_CLIPS.values())
    for clip in clips:
        for step in clip.steps:
            if not step.path or step.prop not in have:
                continue
            for k in range(samples + 1):
                place = prop_place(step.path, step.duration * k / samples, step.duration)
                if place[2] < 0.3:
                    continue
                x, y, w, h, _a = prop_draw_rect(step.prop, sprite_size, anchors, None, place)
                out.append((x, y, w, h))
    return out


#: Motions that lift her window off the ground and let it back down.
_RISE = frozenset({"chute_up"})
_SINK = frozenset({"chute_l", "chute_r"})


def clip_lift(clip: Clip, t: float) -> float:
    """How high (points) a clip has lifted her off the ground at time *t*.

    A try in the jump series goes up and comes down within its step
    (:data:`LEAP_HEIGHTS`); the parachute leaps up to :data:`CHUTE_TOP` and
    sinks back over its floating steps, so she lands exactly where she left
    whatever frames are missing. Never below the ground.
    """
    if t <= 0.0:
        return 0.0
    rise = sum(s.duration for s in clip.steps if s.motion in _RISE)
    sink = sum(s.duration for s in clip.steps if s.motion in _SINK)
    y = 0.0
    hop = 0.0
    acc = 0.0
    for step in clip.steps:
        if acc >= t:
            break
        d = step.duration
        spent = min(d, t - acc)
        f = spent / d if d > 0 else 1.0
        m = step.motion or ""
        if m in _RISE and rise > 0:
            y += CHUTE_TOP * (d / rise) * (1.0 - (1.0 - f) ** 2)
        elif m in _SINK and sink > 0 and rise > 0:
            y -= CHUTE_TOP * spent / sink
        elif m.startswith("leap") and m[4:].isdigit() and spent < d:
            i = int(m[4:])
            if i < len(LEAP_HEIGHTS):
                hop = LEAP_HEIGHTS[i] * 4.0 * f * (1.0 - f)
        acc += d
    return max(0.0, y) + hop


# ---------------------------------------------------------------------------
# Work: what she is seen doing while a tool runs (wawi/working.py)
# ---------------------------------------------------------------------------

#: Animation names a tool may declare (``"animation"`` in its spec, see
#: wawi/tools.py) → a looping clip. Missing art falls back along
#: :data:`WORK_FALLBACK`; with none of it she just shows her thinking frames.
WORK_CLIPS: dict[str, Clip] = {
    # Off to a web page in the spaceship (fetch_page).
    "ship": Clip("work:ship", _alt("ship_fly_a", "ship_fly_b", 0.12, 1, "wobble"),
                 loop=True, xfade=0.05),
    "newspaper": Clip("work:newspaper", (Step("newspaper", 1.4, "breathe"),
                                         Step("newspaper", 0.8, "glance")), loop=True, xfade=0.2),
    "binoculars": Clip("work:binoculars", (Step("binoculars", 1.2, "glance"),
                                           Step("binoculars", 1.0, "breathe")), loop=True,
                       xfade=0.2),
    "camera": Clip("work:camera", (Step("camera_a", 0.7, "breathe"), Step("camera_b", 0.14),
                                   Step("camera_a", 0.9, "breathe")), loop=True, xfade=0.04),
    "carry_paper": Clip("work:carry_paper", (Step("carry_paper", 0.9, "walk"),),
                        loop=True, xfade=0.1),
    "detective": Clip("work:detective", (Step("detective", 1.2, "glance"),
                                         Step("detective_b", 1.1, "sniff")), loop=True, xfade=0.15),
    "present": Clip("work:present", (Step("present", 1.6, "glance"),), loop=True, xfade=0.15),
    "notebook": Clip("work:notebook", (Step("pattern_notebook", 1.6, "fiddle"),),
                     loop=True, xfade=0.15),
    "wrench": Clip("work:wrench", _alt("wrench_a", "wrench_b", 0.3, 1, "fiddle"),
                   loop=True, xfade=0.06),
    "laptop": Clip("work:laptop", _alt("laptop_a", "laptop_b", 0.16, 1), loop=True, xfade=0.04),
    # Anything else (a connector, a tool with no animation of its own).
    "busy": Clip("work:busy", _alt("laptop_a", "laptop_b", 0.22, 1), loop=True, xfade=0.05),
    "think": Clip("work:think", (Step("think_bubble", 1.6, "breathe"),), loop=True, xfade=0.2),
    # Background work while she is otherwise idle (a delegated task, an Orca
    # worker): slower, calmer typing.
    "background": Clip("work:background", _alt("laptop_a", "laptop_b", 0.45, 1, "breathe"),
                       loop=True, xfade=0.12),
}
#: Where an animation without art falls back to, in order.
WORK_FALLBACK = ("busy", "think")

#: Taking notes (wawi/notes.py): the recorder's red light blinks about once a
#: second, and now and then she tilts her head to listen closer. Looped for
#: the whole session; without the art it resolves to her listening frames.
NOTES_CLIP = Clip(
    "notes",
    _alt("notes_a", "notes_b", 0.55, 4, "breathe")
    + (Step("notes_c", 1.1, "tilt"), Step("notes_c", 0.5, "breathe"))
    + _alt("notes_a", "notes_b", 0.55, 3, "breathe"),
    loop=True, xfade=0.08,
)

#: Something dragged over her (wawi/drops.py): ready to catch it, looped
#: until it is dropped or leaves.
CATCH_READY_CLIP = Clip("catch_ready", (Step("catch_ready", 0.5, "wiggle"),
                                        Step("catch_ready", 0.7, "breathe")),
                        loop=True, xfade=0.08)
#: Dropped on her: a hop to catch it, a pleased moment with it in her mouth,
#: then a good sniff of it on the ground.
CATCH_CLIP = Clip("catch", (Step("catch_bite", 0.45, "hop"), Step("catch_bite", 0.5, "breathe"),
                            Step("catch_sniff", 0.5, "sniff"), Step("catch_sniff", 0.6, "sniff"),
                            Step("catch_bite", 0.6, "breathe")), xfade=0.07)

#: Dancing to music (wawi/music.py). Step lengths are in BEATS at
#: :data:`DANCE_BPM`: the animator runs this clip's clock at the music's
#: tempo, so every frame change lands on a beat. Two bars: sway left and
#: right with paws up, bob down twice, half a spin, paws up with a hop.
DANCE_BPM = 120.0
_BEAT = 60.0 / DANCE_BPM
DANCE_CLIP = Clip(
    "dance",
    (Step("dance_a", _BEAT, "groove"), Step("dance_b", _BEAT, "groove"),
     Step("dance_a", _BEAT, "groove"), Step("dance_b", _BEAT, "groove"),
     Step("dance_c", _BEAT, "groove"), Step("dance_c", _BEAT, "groove"),
     Step("dance_d", _BEAT / 2, "twirl"), Step("dance_d", _BEAT / 2, "twirl_back"),
     Step("dance_e", _BEAT, "hop")),
    loop=True, xfade=0.06,
)
#: Tempo she dances to when the music's is not known (Spotify says nothing
#: about it, the microphone may not have caught it yet).
DANCE_DEFAULT_BPM = 112.0

#: How often each idle behaviour is picked. Roll is the rare treat.
BEHAVIOUR_WEIGHTS: dict[str, float] = {
    "look_left": 3.0,
    "look_right": 3.0,
    "look_around": 1.5,
    "yawn": 1.5,
    "moon_yawn": 0.6,
    "late_night": 0.3,
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
    "side_eye", "eyeroll", "moon_yawn", "late_night", "yawn", "stretch",
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
#: With her body on, a dog model decides instead (wawi/pet/dog.py, and
#: MascotAnimator.driven): this is the bodiless mascot's.
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
    if motion == "walk":
        return Pose(dy=0.9 * abs(math.sin(2 * math.pi * 3.1 * u)))
    if motion == "gallop":
        return Pose(dy=1.5 * abs(math.sin(2 * math.pi * 5.5 * u)))
    if motion == "climb":
        return Pose(dx=0.6 * math.sin(2 * math.pi * 1.7 * u))
    if motion == "rung":
        # A little sway as she shifts her weight from paw to paw.
        return Pose(dx=0.5 * math.sin(math.pi * u / d) if d > 0 else 0.0,
                    rot=0.8 * math.sin(math.pi * u / d) if d > 0 else 0.0)
    if motion == "tug":
        return Pose(dy=-0.8 * abs(math.sin(2 * math.pi * 2.0 * u)),
                    rot=1.2 * abs(math.sin(2 * math.pi * 2.0 * u)))
    if motion == "tumble":
        return Pose(rot=6.0 * math.sin(2 * math.pi * 1.3 * u))
    if motion == "squash":
        b = _bump(u / d) if d > 0 else 0.0
        return Pose(sx=1.0 + 0.06 * b, sy=1.0 - 0.08 * b)
    if motion == "bark":
        # A jerk forward (her head is on the left) on each woof.
        b = _bump(min(1.0, u / 0.14))
        return Pose(dx=-1.2 * b, rot=1.5 * b)
    if motion == "howl":
        return Pose(dy=1.0 * k, rot=-3.0 * k)
    if motion == "munch":
        return Pose(dy=-0.6 * abs(math.sin(2 * math.pi * 3.0 * u)))
    if motion == "lap":
        return Pose(dy=-0.4 * abs(math.sin(2 * math.pi * 4.0 * u)))
    if motion == "droop":
        return Pose(dy=-0.8 * k, sy=1.0 - 0.02 * k)
    if motion == "breathe":
        return Pose(sy=1.0 + 0.006 * math.sin(2 * math.pi * u / 1.6))
    # -- dancing (DANCE_CLIP; u and d are in beat-scaled clip time) ---------
    if motion == "groove":
        # Down on the beat, up between beats, with a little sway.
        b = _bump(u / d) if d > 0 else 0.0
        return Pose(dy=2.5 * b, sy=1.0 - 0.02 * (1.0 - b), rot=1.5 * b)
    if motion in ("twirl", "twirl_back"):
        # Half a spin each: squeezed thin at the turn, the drawing mirrored
        # for the second half (see _MIRRORED_MOTIONS).
        x = u / d if d > 0 else 0.0
        turn = abs(x - 0.5) * 2.0 if motion == "twirl" else 1.0 - abs(x - 0.5) * 2.0
        return Pose(dy=1.5 * _bump(x), sx=0.75 + 0.25 * turn, rot=3.0 * math.sin(math.pi * x))
    # -- idle activities ---------------------------------------------------
    if motion == "crouch":
        return Pose(dy=-1.0 * k, sx=1.0 + 0.03 * k, sy=1.0 - 0.05 * k)
    if motion.startswith("leap"):
        b = _bump(u / d) if d > 0 else 0.0
        return Pose(sy=1.0 + 0.04 * b, rot=-2.0 * b)
    if motion == "chute_up":
        return Pose(sy=1.05)
    if motion == "chute_pop":
        return Pose(dy=2.0 * (_bump(u / d) if d > 0 else 0.0))
    if motion in ("chute_l", "chute_r"):
        s = (1.0 if motion == "chute_l" else -1.0) * (math.sin(math.pi * u / d) if d > 0 else 0.0)
        return Pose(dx=1.5 * s, rot=5.0 * s)
    if motion in ("skid", "skid_back"):
        return Pose(dx=0.6 * math.sin(2 * math.pi * 14.0 * u), rot=-3.0 * k)
    if motion in ("zoom_out", "zoom_back"):
        return Pose(dy=2.0 * abs(math.sin(2 * math.pi * 6.5 * u)))
    if motion == "pant":
        return Pose(sy=1.0 + 0.015 * abs(math.sin(2 * math.pi * 3.5 * u)))
    if motion == "creep":
        return Pose(dy=-0.6, rot=0.6 * math.sin(2 * math.pi * 2.0 * u))
    if motion == "pounce":
        b = _bump(u / d) if d > 0 else 0.0
        return Pose(dy=10.0 * b, rot=3.0 * b)
    if motion == "dizzy":
        return Pose(rot=2.5 * math.sin(2 * math.pi * 1.5 * u))
    if motion == "walk_home":
        return Pose(dy=0.9 * abs(math.sin(2 * math.pi * 3.1 * u)))
    if motion == "shake":
        w = math.sin(2 * math.pi * 11.0 * u)
        return Pose(dx=1.5 * w, rot=3.0 * w)
    if motion == "paw_bat":
        return Pose(dy=0.6 * abs(math.sin(2 * math.pi * 3.6 * u)),
                    rot=1.2 * math.sin(2 * math.pi * 3.6 * u))
    return REST


def clip_excursion(clip: Clip, t: float) -> float:
    """How far (points, + is right) a clip has carried her at clip time *t*.

    The dash and the zoomies move her for real: ``run_out`` / ``zoom_out``
    steps travel the way she faces (left) at a fixed speed, ``run_back`` /
    ``zoom_back`` travel back at the same speed; out and back are the same
    frames, so a clip with missing frames still comes home. Stalking the bird
    is a few sneaky steps and a pounce (``creep``, ``pounce``: a fixed
    distance each), and ``walk_home`` steps share out the way back, so she
    ends exactly where she started. The window moves, not the drawing, so
    she can go further than her own padding.
    """
    if t <= 0.0:
        return 0.0
    home_time = sum(s.duration for s in clip.steps if s.motion == "walk_home")
    home_way = -sum(_TRAVEL.get(s.motion or "", 0.0) for s in clip.steps)
    x = 0.0
    acc = 0.0
    for step in clip.steps:
        if acc >= t:
            break
        d = step.duration
        spent = min(d, t - acc)
        m = step.motion or ""
        if m in _SPEEDS:
            x += _SPEEDS[m] * spent
        elif m in _TRAVEL:
            x += _TRAVEL[m] * (spent / d if d > 0 else 1.0)
        elif m == "walk_home" and home_time > 0:
            x += home_way * spent / home_time
        acc += d
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
    ground: float = PAD_BOTTOM,
) -> tuple[float, float]:
    """A point in her view (AppKit points, y up) as (u, v) of the drawn frame.

    The exact inverse of how the view draws her: translate to the bottom
    centre plus the pose offset, rotate, scale. Mirroring is left to
    :func:`touch_zone`. Values outside 0..1 are outside the frame.
    """
    w, h = sprite_size
    x = point[0] - (view_width / 2.0 + pose.dx)
    y = point[1] - (ground + pose.dy)
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
      thinking looks from side to side. The microphone gate does NOT change
      any of this (see :func:`mood`): muted, she only shows a badge and,
      now and then, folds her ears down.
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
        #: Ignored this long, she gets ironic; then this share of her picks
        #: are the ironic ones (1.0: only those). PetLife sets both from
        #: the bodiless mascot (with a body, the dog model picks instead).
        self.ignored_after = IGNORED_AFTER
        self.ironic_share = 1.0
        #: Why she last woke up ("voice", "touch", "cursor", "self"…), for
        #: the log (wawi/pet/life.py reads and clears it).
        self.wake_reason: str | None = None
        #: Her life runs a dog model (wawi/pet/dog.py) that decides when she
        #: sleeps, wakes and does things: then she never dozes off after
        #: :attr:`sleep_after`, never gets up on her own (:data:`WAKE_GAP`) and
        #: never picks idle behaviours by herself. Off (no body): as before.
        self.driven = False
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

        #: Acts (reactions, food, the ball…) and locomotion, driven from
        #: outside by wawi/pet/life.py. See act() and set_locomotion().
        self._acts: dict[str, Clip] = {}
        self._body_clips: dict[str, Clip] = {}
        self._act: Clip | None = None
        self._act_t = 0.0
        self.act_name: str | None = None
        self._loco: Clip | None = None
        self._loco_t = 0.0
        self.locomotion: str | None = None
        #: Which way she faces while she moves or acts (frames face left).
        self.facing_right = False
        #: The clip she rests in when awake and idle: "idle", or "sit_edge"
        #: while she sits on a window's top edge.
        self.idle_clip = "idle"
        #: Mood weighting of her idle behaviours (wawi/pet/needs.py): name →
        #: multiplier. Names outside BEHAVIOUR_WEIGHTS (hungry_bowl…) join the
        #: pool only through here.
        self.bias: dict[str, float] = {}
        #: Local-time multipliers supplied by PetLife (empty = old behaviour).
        self.rhythm_weights: dict[str, float] = {}
        self.rhythm_idle_gap = 1.0
        self.rhythm_wake_gap = 1.0
        self.rhythm_night = False
        #: The microphone gate is closed (set by every update). It changes
        #: nothing about how she lives: only the badge and the odd ears_down.
        self.muted = False
        #: Called with the name of every idle behaviour she picks on her own
        #: (not reactions to a touch): what she learns from (wawi/learn).
        self.on_pick: "Callable[[str], None] | None" = None

        #: Idle activities (ACTIVITIES): the prop sprites present (some need
        #: one), whether her window can be lifted off the ground (only with
        #: her body running: wawi/pet/life.py), and a hold her life puts on
        #: them (typing near her, a game, a trip). Each group's last start
        #: (animator clock), for the cooldowns.
        self.props: frozenset[str] = frozenset()
        self.can_lift = False
        self.hold_activities = False
        self._activity_at: dict[str, float] = {}
        self._activities: dict[str, Clip] = {}
        #: An activity asked for by name (the voice, a test): done next time
        #: she is idle, whatever the dice say.
        self._requested: str | None = None
        #: Busy with a tool (wawi/working.py): the animation it declared, and
        #: the one for background work (a delegated task). Set every frame.
        self.working: str | None = None
        self.background: str | None = None
        self._work: dict[str, Clip] = {}
        self._work_key: str | None = None
        self._work_t = 0.0
        self._work_shown: Clip | None = None
        #: Taking notes (wawi/notes.py): set every frame by the panel. While
        #: on she holds her recorder and does nothing else -- no behaviours,
        #: no walks, no acts -- until it is off again.
        self.taking_notes = False
        self._notes: Clip | None = None
        self._notes_t = 0.0
        #: Dancing to music (wawi/music.py): the tempo to dance at, set every
        #: frame by the panel (0 = unknown tempo); None = not dancing. She
        #: only dances while idle and free: a turn, a walk, an act, a reaction
        #: or a tool's work all win, and the dance picks up again after.
        self.dance_bpm: float | None = None
        self._dance: Clip | None = None
        self._dance_t = 0.0
        self._dance_beat = 0.0

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
        self._acts = {}
        for name, clip in ACT_CLIPS.items():
            ok = clip.resolved(self.frames) if self.frames else None
            if ok is not None:
                self._acts[name] = ok
        self._body_clips = {}
        for name, clip in BODY_CLIPS.items():
            ok = clip.resolved(self.frames) if self.frames else None
            if ok is not None:
                self._body_clips[name] = ok
        self._activities = {}
        for name, activity in ACTIVITIES.items():
            ok = activity.clip.playable(self.frames)
            if ok is not None:
                self._activities[name] = ok
        self._work = {}
        for name, clip in WORK_CLIPS.items():
            ok = clip.playable(self.frames)
            if ok is not None:
                self._work[name] = ok
        self._notes = NOTES_CLIP.resolved(self.frames) if self.frames else None
        self._dance = DANCE_CLIP.resolved(self.frames) if self.frames else None
        if not self.frames:
            self._act = self._loco = None
            self.act_name = self.locomotion = None
            self.frame = self.prev_frame = None
            self.frame_mix = 1.0
            self._stop_oneshot()

    def set_props(self, props: Iterable[str]) -> None:
        """Tell the animator which prop sprites exist (the bird, the
        parachute…): an activity needing a missing one is never picked."""
        self.props = frozenset(props)

    def _gap(self, span: tuple[float, float]) -> float:
        return self.rng.uniform(*span)

    # -- idle activities (ACTIVITIES) ----------------------------------------

    def can_do(self, name: str) -> bool:
        """She has the art for activity *name* (and, if it lifts her off the
        ground, someone to lift her window)."""
        activity = ACTIVITIES.get(name)
        if activity is None or name not in self._activities:
            return False
        have = self.frames | self.props
        if any(n not in have for n in activity.needs):
            return False
        return self.can_lift or not activity.lift

    def activities(self) -> list[str]:
        """The activities she may pick on her own right now: art present,
        nothing holding them back, and each group's cooldown over. The
        microphone being off does not stop them: muted, she keeps living
        (the owner's rule) and only the badge says she is not listening."""
        if self.hold_activities or self._reduce:
            return []
        out = []
        for name, activity in ACTIVITIES.items():
            if not self.can_do(name):
                continue
            last = self._activity_at.get(activity.group)
            if last is not None and self.t - last < activity.cooldown:
                continue
            out.append(name)
        return out

    def perform(self, name: str) -> bool:
        """Do activity (or behaviour) *name* the next time she is idle, by
        request: no dice, no cooldown. False if she cannot do it at all."""
        if not self.framed or not (self.can_do(name) or name in self._clips):
            return False
        self._requested = name
        return True

    def hand_over(self) -> None:
        """Her body takes the activity just picked from here (the desktop
        zoomies, the parachute off a window): stop the clip without dozing
        off, as a drag would make her."""
        self._stop_oneshot()

    @property
    def lift(self) -> float:
        """How high an activity has her off the ground (points): the window
        goes up by this (wawi/pet/life.py adds it to its fly offset)."""
        if self._reduce or self._loco is not None or self._oneshot is None \
                or self._act_shown():
            return 0.0
        return clip_lift(self._oneshot, self._oneshot_t)

    @property
    def framed(self) -> bool:
        return bool(self.frames)

    def playable(self) -> list[str]:
        """Idle behaviours she can actually do with the frames present (plus
        the ones her mood adds, see :attr:`bias`)."""
        names = [n for n in BEHAVIOUR_WEIGHTS if n in self._clips]
        return names + self._mood_extras()

    def _mood_extras(self) -> list[str]:
        return [n for n, w in self.bias.items()
                if w > 0 and n not in BEHAVIOUR_WEIGHTS and self._clip(n) is not None]

    def _clip(self, name: str) -> Clip | None:
        return self._clips.get(name) or self._acts.get(name) or self._activities.get(name)

    # -- acts and locomotion (wawi/pet/life.py) ----------------------------

    def has_act(self, name: str) -> bool:
        return name in self._acts or name in self._clips

    def has_locomotion(self, name: str) -> bool:
        return name in self._body_clips

    def act(self, what: "str | Clip", loop: bool | None = None) -> bool:
        """Act out a clip for a reaction or an interaction: by name (an act,
        or any one-shot she knows) or a ready-made :class:`Clip` (a costume).

        It wins over idle behaviours and sleep, and waits out a voice turn
        (the turn's frames show; the act's clock stops). It ends on its own
        when a one-shot clip is over (:attr:`acting` goes False) or when
        :meth:`stop_act` is called. False when she has no frames for it.
        """
        if not self.framed:
            return False
        if isinstance(what, Clip):
            clip: Clip | None = what.resolved(self.frames)
        else:
            clip = self._clip(what)
        if clip is None:
            return False
        if loop is not None and loop != clip.loop:
            clip = Clip(clip.name, clip.steps, loop, clip.xfade)
        self._stop_oneshot()
        self._act = clip
        self._act_t = 0.0
        self.act_name = clip.name
        self.still = None
        if self.asleep:
            self.wake_reason = self.wake_reason or "act"
            self.asleep = False
            self._begin_pose_fade("asleep")
        self.idle_for = 0.0
        return True

    def stop_act(self) -> None:
        """End the act now; she is idle again (and dozes off as usual)."""
        if self._act is not None:
            self._act = None
            self.act_name = None
            self.idle_for = 0.0

    @property
    def acting(self) -> bool:
        return self._act is not None

    def set_locomotion(self, name: str | None) -> None:
        """Show her body moving: ``walk``, ``trot``, ``climb``, ``fall``,
        ``land``, ``fly`` or None. Wins over everything while it lasts."""
        if name == self.locomotion:
            return
        clip = self._body_clips.get(name) if name else None
        if name is not None and clip is None:
            name = None
        self._loco = clip
        self._loco_t = 0.0
        self.locomotion = name
        if clip is not None:
            self._stop_oneshot()
            self.still = None
            if self.asleep:
                self.wake_reason = self.wake_reason or "move"
                self.asleep = False
                self._begin_pose_fade("asleep")
        self.idle_for = 0.0

    def wake_up(self, reason: str = "move") -> None:
        """Get up now (stretch, yawn) — she is about to go somewhere."""
        self.idle_for = 0.0
        if self.asleep and self.key == "idle":
            self.wake_reason = reason
            self._wake()

    def rouse(self, reason: str, then: str | None = None, glance: bool = False) -> bool:
        """Wake up for real and stay up: the owner is back, the cursor came
        close, a peek at what they are doing.

        Asleep, she gets up (the wake clip); then, with *glance*, looks
        toward the cursor, and does *then* (a wag, a hop…). Unlike her own
        little wakes (:meth:`_wake_for_a_bit`) she does NOT lie straight back
        down: :attr:`idle_for` starts over, so the nap comes a whole
        :attr:`sleep_after` later. Awake and free, she just does the rest.
        False when nothing could be shown (busy, a turn, Reduce Motion)."""
        self.idle_for = 0.0
        if not self.framed or self.key != "idle" or self._reduce \
                or self._act is not None or self._loco is not None or self.petting:
            return False
        was_asleep = self.asleep
        if not was_asleep and self._oneshot is not None:
            return False
        steps: list[Step] = []
        if was_asleep:
            self.asleep = False
            self.wake_reason = reason
            self._begin_pose_fade("asleep")
            self._next_behaviour = self._gap(BEHAVIOUR_GAP) * self.rhythm_idle_gap
            wake = self._clips.get("wake")
            if wake is not None:
                steps += wake.steps
        if glance and self._cursor is not None:
            look = follow_frame(*self._cursor) or (
                "follow_left" if self._cursor[0] < 0 else "follow_right")
            if look in self.frames:
                steps.append(Step(look, 0.6))
        after = self._clips.get(then) if then else None
        if after is not None:
            steps += after.steps
        if not steps:
            return was_asleep
        self._oneshot = Clip("wake" if was_asleep else (then or "glance"), tuple(steps),
                             False, after.xfade if after is not None else 0.12)
        self._oneshot_t = 0.0
        self.behaviour = self._oneshot.name
        self._then_sleep = False
        self._reacting = False
        return True

    @property
    def oneshot_free(self) -> bool:
        """No one-shot clip (behaviour, gesture, head lift…) is playing."""
        return self._oneshot is None

    def _free(self) -> bool:
        """Idle and nothing else on: no turn, act, walk, touch, one-shot,
        work, notes or dance. What her own little things need."""
        return (self.framed and self.key == "idle" and not self._reduce
                and self._act is None and self._loco is None and not self.petting
                and self._oneshot is None and not self.working and not self.background
                and not self.taking_notes and not self.dancing)

    def _steps_for(self, names: "Iterable[str]", toward: float | None = None,
                   seconds: float = 5.0) -> list[Step]:
        """Steps strung from clips she already has. ``"glance"`` is a look
        toward *toward* (the target's x offset from her, + right; else the
        cursor's side); ``"name@4"`` plays clip *name* for about 4 s;
        behaviours, acts, activities and work clips (looped to *seconds*)
        all count. Missing art is skipped."""
        steps: list[Step] = []
        for raw in names:
            name, _, want = raw.partition("@")
            try:
                length = float(want) if want else None
            except ValueError:
                length = None
            if name == "glance":
                side = toward
                if side is None and self._cursor is not None:
                    side = self._cursor[0]
                if side is None:
                    side = self.rng.choice((-1.0, 1.0)) * 10.0
                if abs(side) > 1.0:
                    look = "look_right" if side > 0 else "look_left"
                    if look in self.frames:
                        steps.append(Step(look, length or 1.2, "glance"))
                continue
            if name == "ear_twitch":
                name = "cursor_ear_twitch"
            clip = (self._clips.get(name) or self._acts.get(name)
                    or self._activities.get(name) or self._work.get(name))
            if clip is None or clip.duration <= 0:
                continue
            if length is not None:
                times = max(1, int(round(length / clip.duration)))
            elif clip.loop:
                times = max(1, int(round(seconds / clip.duration)))
            else:
                times = 1
            steps += list(clip.steps) * times
        return steps

    def _play_steps(self, name: str, steps: "Iterable[Step]", then_sleep: bool = False) -> None:
        self._oneshot = Clip(name, tuple(steps), False, 0.12)
        self._oneshot_t = 0.0
        self.behaviour = name
        self._then_sleep = then_sleep
        self._reacting = False

    def gesture(self, names: "Iterable[str]", toward: float | None = None,
                seconds: float = 5.0) -> bool:
        """A quiet gesture in place, strung from clips she already has (see
        :meth:`_steps_for`): a look, behaviours, acts (``"watch"``) or work
        clips (``"laptop"``, looped to about *seconds*).

        Only when she is awake, idle and free: it never wakes her, never
        interrupts anything, and is not one of her own picks (the learner
        does not count it). False when it could not be shown."""
        if self.asleep or not self._free():
            return False
        steps = self._steps_for(names, toward, seconds)
        if not steps:
            return False
        self._play_steps("gesture", steps)
        self.idle_for = 0.0
        return True

    # -- driven by the dog model (wawi/pet/dog.py) -----------------------------
    # With :attr:`driven` on, she never dozes off, wakes up or picks a
    # behaviour on her own: her life (wawi/pet/life.py) says when, with these.

    def lie_down(self) -> bool:
        """Awake: a sigh (the lie-down yawn) and down to sleep. False when
        busy (she lies down later)."""
        if self.asleep:
            return True
        if not self._free():
            return False
        lie = self._clips.get("lie_down")
        if lie is None or "sleeping" not in self.frames:
            self._fall_asleep()
            return True
        self._play_steps("lie_down", lie.steps, then_sleep=True)
        return True

    def head_lift(self, seconds: float, glance: bool = True) -> bool:
        """Asleep: she lifts her head (still lying, eyes open) and looks
        about for *seconds*, then rests it again. She stays asleep for her
        life; nothing else changes."""
        if not self.asleep or not self._free():
            return False
        frame = "curl_focus" if "curl_focus" in self.frames else (
            "ear_twitch" if "ear_twitch" in self.frames else None)
        if frame is None:
            return False
        steps = [Step(frame, max(0.3, seconds * (0.6 if glance else 1.0)), "breathe")]
        if glance:
            steps.append(Step(frame, max(0.2, seconds * 0.4), "glance"))
        self._play_steps("head_lift", steps)
        return True

    def ear_flick(self) -> bool:
        """Asleep: one ear twitches. Nothing else."""
        if not self.asleep or not self._free():
            return False
        clip = self._clips.get("cursor_ear_twitch")
        if clip is None:
            return False
        self._play_steps("cursor_ear_twitch", clip.steps)
        return True

    def get_up(self, reason: str, names: "Iterable[str]" = ("wake",),
               toward: float | None = None) -> bool:
        """Asleep: get up for real (*names* after standing: ``wake`` is the
        stretch and yawn, ``shake``…, any missing art skipped). Awake and
        free, just *names*. False when busy."""
        if not self._free():
            return False
        steps = self._steps_for(names, toward)
        if self.asleep:
            self.asleep = False
            self.wake_reason = reason
            self._begin_pose_fade("asleep")
        self.idle_for = 0.0
        if steps:
            self._play_steps("wake" if "wake" in names else "gesture", steps)
        return True

    def pick(self, pool: "Iterable[str]") -> str | None:
        """Awake and free: one of *pool* she has art for (activities with
        their cooldowns and holds), weighted by mood, rhythm and what she
        learned, never the same twice running. Counts as her own pick."""
        if self.asleep or not self._free():
            return None
        acts = set(self.activities())
        names = [n for n in pool if n in self._clips or n in acts]
        name = self._weighted(names)
        if name is None or not self._start(name):
            return None
        self._last_behaviour = name
        self._picked(name)
        self.idle_for = 0.0
        return name

    def _prop_step(self) -> tuple[Step, float] | None:
        if self._loco is not None:
            return None if self._reduce else self._loco.step_at(self._loco_t)
        clip, t = (self._act, self._act_t) if self._act is not None else (
            self._oneshot, self._oneshot_t)
        if clip is None or self._reduce and clip is not self._act:
            return None
        return clip.step_at(t)

    @property
    def prop(self) -> str | None:
        """The prop to draw by her paws now (a bowl, the ball, the bird…), or
        None."""
        got = self._prop_step()
        return got[0].prop if got is not None else None

    @property
    def prop_place(self) -> tuple[float, float, float, bool] | None:
        """Where that prop is along its path now (see :func:`prop_place`),
        None when it does not move."""
        got = self._prop_step()
        if got is None or not got[0].path:
            return None
        step, u = got
        return prop_place(step.path, u, step.duration)

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
        if not self.framed or self.key != "idle" or self.acting or self._loco is not None:
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
            self.wake_reason = "touch"
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
            self.wake_reason = "touch"
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

    def react_to_cursor(self, reaction: str) -> bool:
        """Play a proximity reaction without stealing voice, work, games or touch."""
        if reaction == "wake":
            # Asleep and the cursor came over: she gets up, looks at it and
            # wags, and stays up (wawi/cursor_reactions.py rations this).
            if not self.asleep or self.dancing or self.working:
                return False
            return self.rouse("cursor", then="wag", glance=True)
        names = {"wag": "cursor_wag", "growl": "cursor_growl",
                 "bark": "cursor_bark", "ear_twitch": "cursor_ear_twitch"}
        name = names.get(reaction)
        guarded_act = reaction == "growl" and self.act_name in ("eat", "play_chew")
        if self.dancing and reaction != "ear_twitch":
            return False
        if name is None or self.key != "idle" or self._reduce or (self._act is not None and not guarded_act) \
                or self._loco is not None or self.working or self.petting \
                or (self.behaviour is not None and not
                    (reaction == "growl" and self.behaviour == "play_chew")):
            return False
        if self.asleep and reaction not in ("ear_twitch", "bark"):
            return False
        if reaction == "bark" and self.asleep:
            # Woken by it for real: up for a while, not straight back down.
            self.asleep = False
            self.wake_reason = "cursor"
            self.idle_for = 0.0
            self._begin_pose_fade("asleep")
        if guarded_act:
            self.stop_act()
        if not self._start(name):
            return False
        if reaction == "wag" and self._cursor is not None and self._oneshot is not None:
            glance = follow_frame(*self._cursor)
            if glance is None:
                glance = "follow_left" if self._cursor[0] < 0 else "follow_right"
            if glance in self.frames:
                clip = self._oneshot
                self._oneshot = Clip(clip.name, (Step(glance, 0.35),) + clip.steps,
                                     clip.loop, clip.xfade)
        return True

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
        if self._reduce or self._loco is not None:
            return 0.0
        if self._act_shown():
            act = self._act
            t = self._act_t % act.duration if act.loop and act.duration > 0 else self._act_t  # type: ignore[union-attr]
            return clip_excursion(act, t)  # type: ignore[arg-type]
        if self._oneshot is None:
            return 0.0
        return clip_excursion(self._oneshot, self._oneshot_t)

    @property
    def mirrored(self) -> bool:
        """Draw her flipped (head on the right): running back home."""
        if self._loco is not None:
            return self.facing_right
        if self._act_shown():
            step, _u = self._act.step_at(self._act_t)  # type: ignore[union-attr]
            if step.motion in _MIRRORED_MOTIONS:
                return True
            if step.motion in ("run_out", "zoom_out"):
                return False
            return self.facing_right
        if self.dancing and not self._reduce and self._dance is not None:
            return self._dance.step_at(self._dance_t)[0].motion in _MIRRORED_MOTIONS
        if self._oneshot is None or self._reduce:
            return False
        step, _u = self._oneshot.step_at(self._oneshot_t)
        return step.motion in _MIRRORED_MOTIONS

    def _act_shown(self) -> bool:
        return self._act is not None and self.key == "idle"

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

        self.muted = bool(muted)
        key = mood(state)
        if key != self.key:
            self._change_key(key)
        else:
            self._base_t += dt
        if self.key == "listening":
            self._quiet_for = self._quiet_for + dt if self.level < LEVEL_QUIET else 0.0

        if self.framed and self._notes_shown():
            return self._take_notes(dt)
        self._notes_t = 0.0
        if self.framed and self.dancing:
            return self._dance_step(dt)
        self._dance_t = self._dance_beat = 0.0

        if self.framed:
            self._run_behaviours(dt)
            self._work_shown = self._work_now(dt)
            name, xfade = self._choose_frame()
            self._show(name, dt, xfade)

        return self._pose(dt)

    def _notes_shown(self) -> bool:
        """Her recorder is out: taking notes, and not saying anything (the
        "te escucho" and the "listo" are spoken with her mouth)."""
        return self.taking_notes and self._notes is not None and self.key != "speaking"

    def _take_notes(self, dt: float) -> Pose:
        """One frame of the note pose. Whatever she was doing is dropped:
        a walk, an act, a behaviour, sleep -- she sits and listens."""
        if self._notes_t == 0.0:
            self._loco = None
            self.locomotion = None
            if self._act is not None:
                self.stop_act()
            self._stop_oneshot()
            if self.asleep:
                self.asleep = False
                self._begin_pose_fade("asleep")
            self._work_shown = None
        self._notes_t += dt
        self.idle_for = 0.0
        clip = self._notes
        step, u = clip.step_at(self._notes_t)  # type: ignore[union-attr]
        # Reduce Motion: the recorder held still, its light on.
        frame = clip.steps[0].frame if self._reduce else step.frame  # type: ignore[union-attr]
        self._show(self._resolve(frame), dt, 0.0 if self._reduce else clip.xfade)  # type: ignore[union-attr]
        target = self._pose_for("listening")
        if self._reduce:
            return target
        return _apply(target, motion_overlay(step.motion, u, step.duration))

    @property
    def dancing(self) -> bool:
        """She is dancing right now (asked to, and nothing else on)."""
        return (self.dance_bpm is not None and self._dance is not None and self.key == "idle"
                and self._loco is None and self._act is None and self._oneshot is None
                and not self.working and not self.background and not self.petting
                and not self.taking_notes)

    def _dance_step(self, dt: float) -> Pose:
        """One frame of the dance, its clock running at the music's tempo.
        Reduce Motion: the first pose held, with a gentle bob on the beat."""
        if self.asleep:
            self.asleep = False
            self._begin_pose_fade("asleep")
        self.idle_for = 0.0
        self._work_shown = None
        bpm = self.dance_bpm or DANCE_DEFAULT_BPM
        speed = bpm / DANCE_BPM
        self._dance_t += dt * speed
        self._dance_beat += dt * bpm / 60.0
        clip = self._dance
        target = self._pose(dt)
        if self._reduce:
            self._show(self._resolve(clip.steps[0].frame), dt, 0.0)  # type: ignore[union-attr]
            return _apply(target, Pose(dy=1.0 * abs(math.sin(math.pi * self._dance_beat))))
        step, u = clip.step_at(self._dance_t)  # type: ignore[union-attr]
        self._show(self._resolve(step.frame), dt, clip.xfade)  # type: ignore[union-attr]
        return _apply(target, motion_overlay(step.motion, u, step.duration))

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
        if self._reduce:
            return target
        if self._loco is not None:
            step, u = self._loco.step_at(self._loco_t)
            return _apply(target, motion_overlay(step.motion, u, step.duration))
        if self._work_shown is not None:
            step, u = self._work_shown.step_at(self._work_t)
            return _apply(target, motion_overlay(step.motion, u, step.duration))
        if self._act_shown():
            step, u = self._act.step_at(self._act_t)  # type: ignore[union-attr]
            return _apply(target, motion_overlay(step.motion, u, step.duration))
        if self._oneshot is not None:
            step, u = self._oneshot.step_at(self._oneshot_t)
            target = _apply(target, motion_overlay(step.motion, u, step.duration))
        return target

    def _change_key(self, key: str) -> None:
        old_pose_key = self._pose_key()
        was_resting = self.asleep
        if self.key == "speaking":
            # She went on a bit: the next cheer becomes a shrug or a smirk.
            self._long_answer = self._base_t >= LONG_ANSWER
        self.key = key
        self._base_t = 0.0
        self._stop_oneshot()
        if was_resting:
            self.wake_reason = "voice"
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
                # Woken by a turn: she gets up the way dogs do.
                self._start("wake")

    def _stop_oneshot(self) -> None:
        self._oneshot = None
        self.behaviour = None
        self._oneshot_t = 0.0
        self._then_sleep = False
        self._reacting = False

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
        self._next_behaviour = self._gap(BEHAVIOUR_GAP) * self.rhythm_idle_gap
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
        names = self._ironic([n for n in WAKE_BEHAVIOURS if n in self._clips]
                             + self._mood_extras() + self.activities())
        name = self._weighted(names)
        if name is None:
            return False
        clip = self._clip(name)
        if clip is None:
            return False
        lie = self._clips.get("lie_down")
        self.asleep = False
        self.wake_reason = "self"
        self.idle_for = 0.0
        self._begin_pose_fade("asleep")
        self._oneshot = Clip(name, clip.steps + (lie.steps if lie else ()), False, clip.xfade)
        self._oneshot_t = 0.0
        self.behaviour = name
        self._reacting = False
        self._then_sleep = True
        self._last_behaviour = name
        self._picked(name)
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
        if self._loco is not None:
            self._loco_t += dt
            if self._loco.done(self._loco_t):
                self._loco = None
                self.locomotion = None
        if self._act is not None:
            # Its clock runs only while it is on screen: not under a voice
            # turn, not under a fall (she is happy AFTER she lands).
            if self._act_shown() and self._loco is None:
                self._act_t += dt
                if self._act.done(self._act_t):
                    self.stop_act()
            self.idle_for = 0.0
            return
        if self._loco is not None:
            self.idle_for = 0.0
            return
        if self._oneshot is not None:
            self._oneshot_t += dt
            if self._oneshot.done(self._oneshot_t):
                then_sleep = self._then_sleep
                self._stop_oneshot()
                if then_sleep and self.key == "idle":
                    self._fall_asleep()
        if self._reduce or self.key != "idle":
            return

        if self._requested is not None and not self.petting:
            name, self._requested = self._requested, None
            if self.asleep:
                self.wake_reason = "request"
                self.asleep = False
                self._begin_pose_fade("asleep")
            self.idle_for = 0.0
            if self._start(name):
                self._last_behaviour = name
                self._picked(name)
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
                self.wake_reason = "celebrate"
                self.asleep = False
                self._begin_pose_fade("asleep")
            self.idle_for = 0.0
            if self._start(name):
                return

        if self.asleep and self._background_clip() is not None and self._oneshot is None:
            # Work started in the background (a delegated task): she gets
            # up to do it, at her little laptop.
            self.wake_reason = "work"
            self._wake()
            return
        if self.asleep:
            if self.driven:
                return
            if self._oneshot is None and not self.petting:
                self._next_wake -= dt
                if self._next_wake <= 0.0:
                    self._next_wake = self._gap(WAKE_GAP) * self.rhythm_wake_gap
                    self._wake_for_a_bit()
            return
        self.idle_for += dt
        if self._oneshot is not None:
            return
        if self.driven:
            self._blink(dt)
            return
        if self.idle_for >= self.sleep_after and "sleeping" in self.frames \
                and self._background_clip() is None:
            if self._start("lie_down"):
                self._then_sleep = True
            else:
                self._fall_asleep()
            return
        self._next_behaviour -= dt
        if self._next_behaviour <= 0.0:
            self._next_behaviour = self._gap(BEHAVIOUR_GAP) * self.rhythm_idle_gap
            name = self._pick_behaviour()
            if name is not None:
                self._last_behaviour = name
                if self._start(name):
                    self._picked(name)
                return
        self._blink(dt)

    def _blink(self, dt: float) -> None:
        self._next_blink -= dt
        if self._next_blink <= 0.0:
            self._next_blink = self._gap(BLINK_GAP)
            self._start("blink")

    def _pick_behaviour(self) -> str | None:
        """A weighted random behaviour, never the one she just did (unless it
        is the only one she has frames for). Ignored for a while, she only
        does the ironic ones."""
        names = self.playable() + self.activities()
        if self.muted and "ears_down" in self._clips:
            names = names + ["ears_down"]
        return self._weighted(self._ironic(names))

    def _ironic(self, names: list[str]) -> list[str]:
        """Ignored for a while (and not at night), she leans ironic: only the
        ironic ones with :attr:`ironic_share` 1.0, else that share of the
        time — the rest of her repertoire stays in play."""
        if self.ignored_for < self.ignored_after or self.rhythm_night:
            return names
        ironic = [n for n in IRONIC if n in names or n in self._clips]
        if not ironic:
            return names
        if self.ironic_share >= 1.0 or self.rng.random() < self.ironic_share:
            return ironic
        return names
    def _weight(self, name: str) -> float:
        activity = ACTIVITIES.get(name)
        base = activity.weight if activity is not None else BEHAVIOUR_WEIGHTS.get(name, 1.0)
        return (base * max(0.0, self.bias.get(name, 1.0))
                * max(0.0, self.rhythm_weights.get(name, 1.0)))

    def _weighted(self, names: list[str]) -> str | None:
        names = [n for n in names if self._weight(n) > 0.0]
        if len(names) > 1 and self._last_behaviour in names:
            names.remove(self._last_behaviour)
        if not names:
            return None
        total = sum(self._weight(n) for n in names)
        r = self.rng.random() * total
        for n in names:
            r -= self._weight(n)
            if r < 0.0:
                return n
        return names[-1]

    def _picked(self, name: str) -> None:
        if self.on_pick is not None:
            try:
                self.on_pick(name)
            except Exception:
                log.debug("on_pick failed", exc_info=True)

    def _start(self, name: str) -> bool:
        clip = (self._clips.get(name) or self._activities.get(name)
                or (self._acts.get(name) if name in self.bias else None))
        if clip is None:
            return False
        activity = ACTIVITIES.get(name)
        if activity is not None:
            self._activity_at[activity.group] = self.t
        self._oneshot = clip
        self._oneshot_t = 0.0
        self.behaviour = name
        self._then_sleep = False
        self._reacting = False
        return True

    def _work_for(self, name: str | None) -> Clip | None:
        """The clip for work animation *name*, or its nearest fallback."""
        if not name:
            return None
        for n in (name, *WORK_FALLBACK):
            clip = self._work.get(n)
            if clip is not None:
                return clip
        return None

    def _background_clip(self) -> Clip | None:
        if not self.background or self.key != "idle":
            return None
        return self._work.get(self.background) or self._work.get("background")

    def _work_now(self, dt: float) -> Clip | None:
        """The work clip to show now, if any, keeping its own clock.

        A tool shows while she thinks (or, lingering for its minimum time,
        when she is idle again with nothing else going on); never over
        listening or speaking, which always win.
        """
        clip = None
        if self.working and not self._reduce:
            if self.key == "thinking" or (self.key == "idle" and self._oneshot is None
                                          and not self._act_shown()):
                clip = self._work_for(self.working)
        if clip is None:
            self._work_key = None
            return None
        if clip.name != self._work_key:
            self._work_key = clip.name
            self._work_t = 0.0
        else:
            self._work_t += dt
        return clip

    def _choose_frame(self) -> tuple[str | None, float]:
        """The frame for now and the crossfade to reach it with."""
        if self._loco is not None:
            step, _u = self._loco.step_at(self._loco_t)
            return self._resolve(step.frame), 0.0 if self._reduce else self._loco.xfade
        work = self._work_shown
        if work is not None:
            return self._resolve(work.frame_at(self._work_t)), work.xfade
        if self._act_shown():
            step, _u = self._act.step_at(self._act_t)  # type: ignore[union-attr]
            return self._resolve(step.frame), 0.0 if self._reduce else self._act.xfade  # type: ignore[union-attr]
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
            }
            return still.get(key, "idle_a"), 0.0
        if key == "idle" and self.asleep:
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
        background = self._background_clip()
        if background is not None:
            return background.frame_at(self._base_t), background.xfade
        if self.idle_clip != "idle":
            clip = self._acts.get(self.idle_clip)
            if clip is not None:
                return clip.frame_at(self._base_t), clip.xfade
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
