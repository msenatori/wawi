"""Local-time rhythm and independent, unobtrusive need reminders."""

from datetime import datetime
import random

from wawi import mascot
from wawi.pet.life import Env, PetLife
from wawi.pet.needs import NeedsStore
from wawi.pet.rhythm import profile
from wawi.pet.settings import PetSettings


def wall(hour: int, minute: int = 0) -> float:
    return datetime(2026, 10, 16, hour, minute).timestamp()


def test_daily_agenda_and_soft_transitions():
    night = profile(wall(2))
    noon = profile(wall(12))
    nap = profile(wall(14))
    evening = profile(wall(20))
    assert night[0] < noon[0] and night[4] and not noon[4]
    assert night[3]["zoomies"] < 0.05 and noon[3]["zoomies"] > 1
    assert night[3]["moon_yawn"] > noon[3]["moon_yawn"]
    assert nap[0] < noon[0] and evening[3]["run"] < noon[3]["run"]
    a, b = profile(wall(20, 59))[0], profile(wall(21, 0))[0]
    assert abs(a - b) < 1.0
    assert profile(wall(22), night_start=23)[0] > profile(wall(22))[0]
    assert profile(wall(2), enabled=False)[3] == {}


def test_animator_uses_rhythm_weights():
    anim = mascot.MascotAnimator(available=mascot.FRAME_NAMES, rng=random.Random(1))
    anim.rhythm_weights = profile(wall(2))[3]
    assert anim._weight("zoomies") < 0.05
    assert anim._weight("moon_yawn") > anim._weight("run")


def test_food_and_water_requests_are_independent_persistent_and_reset(tmp_path):
    """Her needs' clocks say when she would ask; the dog model (wawi/pet/dog.py)
    makes it one polite episode at a time, held in quiet mode and games, and a
    command answers it and resets that clock only."""
    path = tmp_path / "pet.json"
    settings = PetSettings.from_dict({"state_file": str(path), "wander": False,
                                      "requests": {"interval_min": [1, 1]}})
    anim = mascot.MascotAnimator(available=mascot.FRAME_NAMES, rng=random.Random(2))
    life = PetLife(settings, anim, rng=random.Random(3), store=NeedsStore(path), wall=wall(12))

    def tick(now, hour_wall, **flags):
        return life.tick(0.1, Env(now=now, wall=hour_wall, **flags))

    tick(0, wall(12))
    life.dog.until = 1e9  # resting, awake, the whole time
    assert tick(61, wall(12) + 61, quiet=True).bubble is None
    assert tick(62, wall(12) + 62, in_game=True).bubble is None
    assert life.requests.need is None
    tick(63, wall(12) + 63)
    assert life.requests.need == "food"  # one at a time: water waits its turn
    assert tick(63.1, wall(12) + 63.1).bubble == "¿Comemos algo?"
    life.store.save(life.needs, wall(12) + 64, force=True)
    loaded = NeedsStore(path).load(wall(12) + 64)
    food_due = life.needs.request_state("food")["due"]
    water_due = life.needs.request_state("water")["due"]
    assert loaded.request_state("food")["due"] == food_due
    life.command("feed")
    assert life.requests.need is None
    assert life.needs.request_state("food")["due"] > food_due
    assert life.needs.request_state("water")["due"] == water_due


def test_night_never_requests(tmp_path):
    settings = PetSettings.from_dict({"state_file": str(tmp_path / "s.json"),
                                      "requests": {"interval_min": [1, 1]}})
    anim = mascot.MascotAnimator(available=mascot.FRAME_NAMES)
    life = PetLife(settings, anim, wall=wall(22))
    life.needs.set_request_state("food", stage=0, due=wall(22) - 100)
    out = life.tick(0.1, Env(now=61, wall=wall(22) + 61))
    assert out.bubble is None
    assert life.needs.request_state("food")["stage"] == 0


def test_late_at_night_she_yawns_and_says_it_is_late_once(tmp_path):
    settings = PetSettings.from_dict({"state_file": str(tmp_path / "pet.json"), "wander": False,
                                      "requests": {"enabled": False}})
    anim = mascot.MascotAnimator(available=mascot.FRAME_NAMES, rng=random.Random(4))
    life = PetLife(settings, anim, rng=random.Random(5),
                   store=NeedsStore(tmp_path / "pet.json"), wall=wall(23))

    def tick(now, w, **flags):
        return life.tick(0.1, Env(now=now, wall=w, **flags))

    # Typing away at 23:00: one yawn, one quiet bubble.
    bubbles = [tick(t, wall(23) + t, key_idle=5.0).bubble for t in range(0, 3)]
    said = [b for b in bubbles if b]
    assert said and ("tarde" in said[0] or "sueño" in said[0])
    anim.stop_act()
    life._bubble_until = 0
    assert all(tick(t, wall(23) + t, key_idle=5.0).bubble in (None, said[0])
               for t in range(10, 13))
    # Nobody at the keyboard, or daytime: nothing.
    life2 = PetLife(settings, mascot.MascotAnimator(available=mascot.FRAME_NAMES,
                                                    rng=random.Random(6)),
                    rng=random.Random(7), store=NeedsStore(tmp_path / "p2.json"), wall=wall(15))
    assert life2.tick(0.1, Env(now=0, wall=wall(15), key_idle=5.0)).bubble is None
    assert life2._late_said_at == 0.0
