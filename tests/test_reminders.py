"""The scheduler is clock-driven: no sleeping, microphone or real audio."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from wawi import reminders, tools

ZONE = ZoneInfo("America/Argentina/Buenos_Aires")


class Clock:
    def __init__(self, at=None):
        self.at = at or datetime(2026, 10, 16, 17, 30, tzinfo=ZONE)

    def __call__(self):
        return self.at

    def advance(self, **kwargs):
        self.at += timedelta(**kwargs)


def service(tmp_path, clock, **kwargs):
    return reminders.ReminderService(tmp_path / "reminders_local.json", clock=clock, **kwargs)


@pytest.mark.parametrize("phrase,expected", [
    ("en 20 minutos", (17, 50, 16)),
    ("a las 18", (18, 0, 16)),
    ("mañana a las 9", (9, 0, 17)),
    ("en dos horas", (19, 30, 16)),
])
def test_spanish_time(phrase, expected):
    due = reminders.parse_when(phrase, Clock()())
    assert (due.hour, due.minute, due.day) == expected
    assert due.tzinfo == ZONE


def test_past_clock_hour_rolls_to_tomorrow():
    due = reminders.parse_when("a las 9", Clock()())
    assert due.day == 17 and due.hour == 9


@pytest.mark.parametrize("bad", ["a las 25", "mañana", "en 0 minutos", "cada semana", ""])
def test_invalid_time_rejected(bad):
    with pytest.raises(ValueError):
        reminders.parse_when(bad, Clock()())


def test_weekly_and_daily_schedule():
    now = Clock()()  # Friday at 17:30
    daily = reminders.parse_schedule("todas las mañanas a las 9")
    weekly = reminders.parse_schedule("los viernes a las 18")
    assert reminders.next_occurrence(daily, now).day == 17
    assert reminders.next_occurrence(weekly, now).hour == 18
    assert reminders.next_occurrence(weekly, now.replace(hour=19)).day == 23


def test_timer_waits_while_busy_and_survives_restart(tmp_path):
    clock = Clock()
    heard = []
    busy = True
    def announce(text):
        if busy:
            return False
        heard.append(text)
        return True
    scheduler = service(tmp_path, clock, announce=announce)
    scheduler.set_timer(minutes=2, label="Té")
    clock.advance(minutes=3)
    assert scheduler.step() == "busy"
    assert len(scheduler.list_items("reminder")) == 1
    restarted = service(tmp_path, clock, announce=announce)
    busy = False
    assert restarted.step() == "delivered"
    assert heard == ["Se te pasó: Té"]
    assert restarted.list_items("reminder") == []


def test_live_due_says_reminder_and_cancel(tmp_path):
    clock = Clock()
    heard = []
    scheduler = service(tmp_path, clock, announce=lambda text: heard.append(text) or True)
    item = scheduler.remind_me("en 20 minutos", "Llamar")
    other = scheduler.set_timer(seconds=30, label="Horno")
    assert scheduler.remove(other["id"], "reminder")
    assert not scheduler.remove("missing", "reminder")
    clock.advance(minutes=20)
    assert scheduler.step() == "delivered"
    assert heard == ["Recordatorio: Llamar"]
    assert scheduler.list_items("reminder") == []
    assert item["id"]


def test_routine_needs_next_owner_yes_then_repeats(tmp_path):
    clock = Clock()
    ran = []
    scheduler = service(tmp_path, clock, run_routine=lambda instruction: ran.append(instruction) or True)
    draft = scheduler.add_routine("los viernes a las 18", "Revisá el correo y contame")
    assert draft["pending"]
    assert scheduler.list_items("routine") == []
    assert scheduler.hear("sí") == "Listo, programé la rutina."
    assert scheduler.hear("sí") is None
    assert len(scheduler.list_items("routine")) == 1
    clock.advance(minutes=30)
    assert scheduler.step() == "delivered"
    assert ran == ["Revisá el correo y contame"]
    assert datetime.fromisoformat(scheduler.list_items("routine")[0]["due"]).day == 23


def test_routine_confirmation_rejects_non_answer_and_expiry(tmp_path):
    clock = Clock()
    scheduler = service(tmp_path, clock)
    scheduler.add_routine("todas las mañanas a las 9", "Revisá X")
    assert scheduler.hear("sí, pero cambiá la hora") is None
    assert not scheduler.list_items("routine")
    scheduler.add_routine("todas las mañanas a las 9", "Revisá X")
    clock.advance(seconds=181)
    assert scheduler.hear("sí") is None
    assert not scheduler.list_items("routine")


def test_tool_specs_dispatch_and_do_not_leak_animation(tmp_path):
    clock = Clock()
    scheduler = service(tmp_path, clock)
    reminders.install(scheduler)
    try:
        names = {item["name"] for item in tools.active_tools()}
        assert {"set_timer", "remind_me", "add_routine", "list_routines"} <= names
        assert all("animation" not in item for item in tools.active_tools())
        result = tools.dispatch("set_timer", {"minutes": 1, "label": "Té"})
        assert result.ok and result.data["id"]
        assert tools.dispatch("list_reminders").data["items"][0]["text"] == "Té"
        assert tools.dispatch("cancel_reminder", {"id": result.data["id"]}).ok
        assert not tools.dispatch("add_routine", {"schedule": "ayer", "instruction": "X"}).ok
    finally:
        reminders.install(None)


def test_routine_waits_for_idle_and_keeps_due_on_disk(tmp_path):
    clock = Clock()
    busy = True
    ran = []
    def run(instruction):
        if busy:
            return False
        ran.append(instruction)
        return True
    scheduler = service(tmp_path, clock, run_routine=run)
    scheduler.add_routine("todas las mañanas a las 9", "Contame el clima")
    assert scheduler.hear("sí")
    clock.advance(days=1)
    assert scheduler.step() == "busy"
    assert len(scheduler.list_items("routine")) == 1
    restarted = service(tmp_path, clock, run_routine=run)
    busy = False
    assert restarted.step() == "delivered"
    assert ran == ["Contame el clima"]
    assert datetime.fromisoformat(restarted.list_items("routine")[0]["due"]).day == 18


def test_config_state_path_is_next_to_config(tmp_path):
    from wawi.config import load
    config = tmp_path / "config.yaml"
    config.write_text("reminders:\n  local_file: private/reminders_local.json\n")
    assert load(config).reminders["local_file"] == str(tmp_path / "private/reminders_local.json")


def test_reminder_announcement_uses_locked_speech_not_petvoice():
    from wawi.orchestrator import Orchestrator
    from wawi.reactions import BUS
    orch = Orchestrator.__new__(Orchestrator)
    calls = []
    orch._speak_reaction = lambda text: calls.append(text) or True
    assert orch._announce_reminder("Recordatorio: Té")
    assert calls == ["Recordatorio: Té"]
    reaction = BUS.pop()
    assert reaction.bubble == "Recordatorio: Té"
    assert reaction.behaviour == "bark" and reaction.sound == "bark"
    assert not reaction.speak


def test_routine_can_be_accepted_from_window(tmp_path):
    from wawi.confirm import REGISTRY
    scheduler = service(tmp_path, Clock())
    scheduler.add_routine("todas las mañanas a las 9", "Revisá X")
    assert REGISTRY.resolve(scheduler._confirmation_id, True)
    assert len(scheduler.list_items("routine")) == 1
    assert scheduler.hear("sí") is None
