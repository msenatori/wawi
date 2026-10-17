"""The machine itself: disk space, battery, memory pressure, CPU heat/load.

Every reading goes through :class:`~wawi.sensors.base.Levels` so a value that
hovers on a threshold produces one event, not one per poll. All probes are
local commands with short timeouts (``pmset``, ``memory_pressure``) or plain
syscalls; nothing here needs a permission prompt.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Any, Callable

from wawi.reactions import Event
from wawi.sensors.base import Levels, Sensor, as_list, human_bytes

GB = 1_000_000_000


def run_cmd(cmd: list[str], timeout: float = 5.0) -> str:
    """Run a probe command; empty string when it is missing or fails."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout or ""


_BATT_RE = re.compile(r"(\d{1,3})%;\s*([a-zA-Z ]+?);")


def parse_pmset_batt(text: str) -> tuple[int, bool] | None:
    """('…InternalBattery-0 (id=1) 54%; discharging; 3:10 remaining') -> (54, False).

    Returns (percent, on_power) or None when there is no battery (desktops).
    """
    m = _BATT_RE.search(text or "")
    if not m:
        return None
    pct = int(m.group(1))
    state = m.group(2).strip().lower()
    on_power = "ac power" in (text or "").lower() or state in ("charging", "charged", "finishing charge")
    return pct, on_power


def parse_memory_pressure(text: str) -> int | None:
    m = re.search(r"free percentage:\s*(\d+)%", text or "")
    return int(m.group(1)) if m else None


def parse_thermal(text: str) -> bool:
    """True when macOS reports thermal throttling."""
    m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", text or "")
    if m and int(m.group(1)) < 100:
        return True
    m = re.search(r"[Tt]hermal warning level\s*[:=]\s*(\d+)", text or "")
    return bool(m and int(m.group(1)) > 0)


class SystemSensor(Sensor):
    name = "system"
    default_interval = 300.0

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        disk_usage: Callable[[str], Any] = shutil.disk_usage,
        run: Callable[[list[str]], str] = run_cmd,
        loadavg: Callable[[], tuple[float, float, float]] = os.getloadavg,
        cpu_count: int | None = None,
    ) -> None:
        super().__init__(config)
        c = self.config
        self._disk_usage = disk_usage
        self._run = run
        self._loadavg = loadavg
        self._cpus = cpu_count or os.cpu_count() or 1

        self.volumes = [str(v) for v in as_list(c.get("volumes", ["/"]))]
        low_pct = float(c.get("disk_low_pct", 10))
        low_gb = float(c.get("disk_low_gb", 10))
        crit_pct = float(c.get("disk_critical_pct", 3))
        margin_pct = float(c.get("disk_margin_pct", 2))
        margin_gb = float(c.get("disk_margin_gb", 2))
        self._disk_bytes: dict[str, tuple[int, int]] = {}

        def disk_level(reading: tuple[float, float], relaxed: bool = False) -> int:
            pct, free_gb = reading
            sp, sg = (margin_pct, margin_gb) if relaxed else (0.0, 0.0)
            if pct < crit_pct + sp:
                return 2
            if pct < low_pct + sp or free_gb < low_gb + sg:
                return 1
            return 0

        self._disk = Levels(disk_level, lambda r: disk_level(r, True))

        self.battery = bool(c.get("battery", True))
        b_low = float(c.get("battery_low_pct", 20))
        b_crit = float(c.get("battery_critical_pct", 7))
        b_margin = float(c.get("battery_margin_pct", 5))
        self._batt = Levels(
            lambda p: 2 if p <= b_crit else 1 if p <= b_low else 0,
            lambda p: 2 if p <= b_crit + b_margin else 1 if p <= b_low + b_margin else 0,
        )
        self._on_power: bool | None = None
        self._batt_pct: int | None = None

        self.memory = bool(c.get("memory", False))
        m_low = float(c.get("memory_low_pct", 10))
        self._mem = Levels(
            lambda free: 1 if free < m_low else 0,
            lambda free: 1 if free < m_low + 5 else 0,
        )
        self._mem_free: int | None = None

        self.cpu = bool(c.get("cpu", False))
        self._load_ratio = float(c.get("cpu_load_ratio", 1.5))
        self._busy_after = max(1, int(c.get("cpu_busy_polls", 3)))
        self._busy_count = 0
        self._busy = False
        self._hot = False

    # -- probes -------------------------------------------------------------
    def poll(self) -> list[Event]:
        events: list[Event] = []
        events += self._poll_disks()
        if self.battery:
            events += self._poll_battery()
        if self.memory:
            events += self._poll_memory()
        if self.cpu:
            events += self._poll_cpu()
        return events

    def _poll_disks(self) -> list[Event]:
        events = []
        for vol in self.volumes:
            try:
                usage = self._disk_usage(vol)
            except OSError:
                continue  # unplugged external drive: not an alarm
            total, free = int(usage.total), int(usage.free)
            if total <= 0:
                continue
            self._disk_bytes[vol] = (total, free)
            pct = free * 100.0 / total
            gb = free / GB
            prev, new = self._disk.update(vol, (pct, gb))
            if prev == new or (prev is None and new == 0):
                continue
            data = {
                "volume": vol,
                "free": human_bytes(free),
                "free_gb": round(gb, 1),
                "free_pct": round(pct, 1),
                "total": human_bytes(total),
            }
            key = f"disk:{vol}"
            if new == 2:
                events.append(Event("disk.critical", "urgent", key, f"Disco {vol} casi lleno", data))
            elif new == 1 and (prev is None or prev < 1):
                events.append(Event("disk.low", "warning", key, f"Poco espacio en {vol}", data))
            elif new == 1:
                # Stepped down from critical to merely low: lift the alarm only.
                events.append(Event("disk.ok", "info", key, f"{vol} ya no está crítico", data))
            elif new == 0:
                events.append(Event("disk.ok", "info", key, f"Hay lugar en {vol}", data))
        return events

    def _poll_battery(self) -> list[Event]:
        parsed = parse_pmset_batt(self._run(["pmset", "-g", "batt"]))
        if parsed is None:
            return []
        pct, on_power = parsed
        events = []
        was_on_power = self._on_power
        self._on_power, self._batt_pct = on_power, pct
        data = {"percent": pct, "on_power": on_power}
        if was_on_power is False and on_power:
            # Plugged in: a thank-you, and it lifts any low-battery alarm.
            self._batt.update("battery", 100.0)
            events.append(Event("battery.charging", "info", "battery", "Cargando", data))
            return events
        if on_power:
            self._batt.update("battery", 100.0)
            return events
        prev, new = self._batt.update("battery", float(pct))
        if new > (prev or 0):
            if new == 2:
                events.append(Event("battery.critical", "urgent", "battery", "Batería crítica", data))
            else:
                events.append(Event("battery.low", "warning", "battery", "Batería baja", data))
        return events

    def _poll_memory(self) -> list[Event]:
        free = parse_memory_pressure(self._run(["memory_pressure", "-Q"]))
        if free is None:
            return []
        self._mem_free = free
        prev, new = self._mem.update("memory", float(free))
        data = {"free_pct": free}
        if new == 1 and prev != 1:
            return [Event("memory.high", "warning", "memory", "Poca memoria libre", data)]
        if new == 0 and prev == 1:
            return [Event("memory.ok", "info", "memory", "Memoria normal", data)]
        return []

    def _poll_cpu(self) -> list[Event]:
        events = []
        try:
            load1 = float(self._loadavg()[0])
        except OSError:
            load1 = 0.0
        ratio = load1 / self._cpus
        if ratio >= self._load_ratio:
            self._busy_count += 1
        else:
            self._busy_count = 0
            if self._busy and ratio < self._load_ratio * 0.7:
                self._busy = False
        if not self._busy and self._busy_count >= self._busy_after:
            self._busy = True
            events.append(Event("cpu.busy", "notice", "cpu", "La compu está a full",
                                {"load": round(load1, 1), "cpus": self._cpus}))
        hot = parse_thermal(self._run(["pmset", "-g", "therm"]))
        if hot and not self._hot:
            events.append(Event("cpu.hot", "warning", "cpu.hot", "La compu está recalentada", {}))
        elif self._hot and not hot:
            events.append(Event("cpu.cool", "info", "cpu.hot", "Se enfrió", {}))
        self._hot = hot
        return events

    def disk_alert(self) -> tuple[str, int, int] | None:
        """(volume, level 1|2, free bytes) for the worst disk, or None if fine."""
        worst = None
        for vol, (_total, free) in self._disk_bytes.items():
            level = self._disk.level(vol) or 0
            if level and (worst is None or level > worst[1]):
                worst = (vol, level, free)
        return worst

    # -- status -------------------------------------------------------------
    def summary(self) -> str | None:
        parts = []
        for vol, (total, free) in self._disk_bytes.items():
            name = "el disco" if vol == "/" else f"el disco {vol}"
            parts.append(f"{name} tiene {human_bytes(free)} libres")
        if self._batt_pct is not None:
            plug = " y está enchufada" if self._on_power else ""
            parts.append(f"la batería está al {self._batt_pct}%{plug}")
        if self._mem_free is not None and self._mem.level("memory") == 1:
            parts.append(f"queda poca memoria ({self._mem_free}% libre)")
        if self._hot:
            parts.append("la compu está recalentada")
        return "; ".join(parts) or None
