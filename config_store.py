import asyncio
import json
import logging
import os
import tempfile
import time

log = logging.getLogger("config_store")


def atomic_write_json(data: dict, path: str) -> None:
    """Write JSON to a temp file in the same dir, fsync, then atomically replace."""
    directory = os.path.dirname(os.path.abspath(path))
    tmp = tempfile.NamedTemporaryFile("w", dir=directory, prefix=".cfg-", suffix=".tmp", delete=False)
    try:
        try:
            json.dump(data, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
        finally:
            tmp.close()
        os.replace(tmp.name, path)
    except BaseException:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise


class ConflictError(Exception):
    def __init__(self, snapshot: dict):
        super().__init__("config revision conflict")
        self.snapshot = snapshot


class EventHub:
    def __init__(self):
        self._subs: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=64)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    def publish(self, event: dict) -> None:
        for q in list(self._subs):
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    @property
    def subscriber_count(self) -> int:
        return len(self._subs)


class ConfigStore:
    def __init__(self, path: str):
        self.path = path
        self.hub = EventHub()
        self.rev = 0
        self.updated_by = None
        self.updated_at = None
        self._guilds: dict[str, dict] = {}
        self.voice: str | None = None  # GLOBAL voice (Piper file stem / Kokoro name); None = env default
        self.engine: str = "piper"     # GLOBAL TTS engine: "piper" | "kokoro"
        self._lock = asyncio.Lock()
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path) as f:
                data = json.load(f)
            guilds = {}
            for gid, entry in data.get("guilds", {}).items():
                ch = entry.get("channel_id")
                g_dict = {
                    "channel_id": int(ch) if ch is not None else None,
                    "enabled": bool(entry.get("enabled", True)),
                }
                if "speed" in entry:
                    try:
                        g_dict["speed"] = float(entry["speed"])
                    except (ValueError, TypeError):
                        pass
                if "volume" in entry:
                    try:
                        g_dict["volume"] = float(entry["volume"])
                    except (ValueError, TypeError):
                        pass
                guilds[str(gid)] = g_dict
            self.rev = int(data.get("rev", 0))
            self.updated_by = data.get("updated_by")
            self.updated_at = data.get("updated_at")
            self._guilds = guilds
            voice = data.get("voice")
            self.voice = voice if isinstance(voice, str) and voice else None
            eng = data.get("engine")
            self.engine = eng if eng in ("piper", "kokoro") else "piper"
        except FileNotFoundError:
            pass
        except Exception:
            log.exception("could not load %s; starting with empty config", self.path)
            self._quarantine()
            self.rev, self.updated_by, self.updated_at, self._guilds = 0, None, None, {}
            self.voice = None
            self.engine = "piper"

    def _quarantine(self) -> None:
        """Best-effort: move an unreadable config aside so the next write cannot destroy it."""
        dest = self.path + ".corrupt"
        try:
            os.replace(self.path, dest)
            log.error("moved unreadable config to %s", dest)
        except OSError:
            log.exception("could not move unreadable config aside")

    def snapshot(self) -> dict:
        return {
            "rev": self.rev,
            "updated_by": self.updated_by,
            "updated_at": self.updated_at,
            "voice": self.voice,
            "engine": self.engine,
            "guilds": {g: dict(e) for g, e in self._guilds.items()},
        }

    def channel_for(self, guild_id: int) -> int | None:
        entry = self._guilds.get(str(guild_id))
        return entry["channel_id"] if entry else None

    def is_enabled(self, guild_id: int) -> bool:
        entry = self._guilds.get(str(guild_id))
        return entry["enabled"] if entry else True

    def is_listen_channel(self, guild_id: int, channel_id: int) -> bool:
        entry = self._guilds.get(str(guild_id))
        return bool(entry) and entry["channel_id"] == channel_id and entry["enabled"]

    def is_command_channel(self, guild_id: int, channel_id: int) -> bool:
        entry = self._guilds.get(str(guild_id))
        return bool(entry) and entry["channel_id"] == channel_id

    def speed_for(self, guild_id: int) -> float:
        entry = self._guilds.get(str(guild_id))
        if entry and "speed" in entry:
            return float(entry["speed"])
        return 1.0

    def volume_for(self, guild_id: int) -> float:
        entry = self._guilds.get(str(guild_id))
        if entry and "volume" in entry:
            return float(entry["volume"])
        return 1.0

    async def set_channel(self, guild_id: int, channel_id: int | None, *,
                          expected_rev: int | None, editor: str) -> dict:
        return await self._apply(guild_id, "channel_id", channel_id, expected_rev, editor)

    async def set_enabled(self, guild_id: int, enabled: bool, *,
                          expected_rev: int | None, editor: str) -> dict:
        return await self._apply(guild_id, "enabled", bool(enabled), expected_rev, editor)

    async def set_speed(self, guild_id: int, speed: float, *,
                        expected_rev: int | None = None, editor: str = "api") -> dict:
        speed_val = round(float(speed), 2)
        if not (0.5 <= speed_val <= 2.0):
            raise ValueError("speed must be between 0.5 and 2.0")
        return await self._apply(guild_id, "speed", speed_val, expected_rev, editor)

    async def set_volume(self, guild_id: int, volume: float, *,
                         expected_rev: int | None = None, editor: str = "api") -> dict:
        vol_val = round(float(volume), 2)
        if not (0.1 <= vol_val <= 2.0):
            raise ValueError("volume must be between 0.1 and 2.0")
        return await self._apply(guild_id, "volume", vol_val, expected_rev, editor)

    async def set_voice(self, name: str | None, *, expected_rev: int | None, editor: str,
                        engine: str | None = None) -> dict:
        """Global voice (file stem / Kokoro name or None), optionally with the engine, in one write.
        Same semantics as set_enabled: stale rev -> ConflictError, unchanged -> no rev bump and no event,
        persist failure -> rollback and re-raise."""
        if name is not None and not isinstance(name, str):
            raise TypeError("voice must be a string or None")
        if engine is not None and engine not in ("piper", "kokoro"):
            raise ValueError("unknown engine")
        async with self._lock:
            if expected_rev is not None and expected_rev != self.rev:
                raise ConflictError(self.snapshot())
            new_engine = self.engine if engine is None else engine
            if name == self.voice and new_engine == self.engine:
                return self.snapshot()
            old, old_engine = self.voice, self.engine
            prev = (self.rev, self.updated_by, self.updated_at)
            self.voice, self.engine = name, new_engine
            self.rev += 1
            self.updated_by = editor
            self.updated_at = time.time()
            snap = self.snapshot()
            try:
                await asyncio.to_thread(atomic_write_json, snap, self.path)
            except BaseException:
                self.rev, self.updated_by, self.updated_at = prev
                self.voice, self.engine = old, old_engine
                raise
            self.hub.publish({
                "type": "config",
                "snapshot": snap,
                "change": {"guild_id": None, "field": "voice", "old": old, "new": name},
            })
            return snap

    async def seed_channel(self, guild_id: int, channel_id: int) -> bool:
        async with self._lock:
            if str(guild_id) in self._guilds:
                return False
            await self._mutate(guild_id, "channel_id", channel_id, "env-seed")
            return True

    async def _apply(self, guild_id, field, value, expected_rev, editor) -> dict:
        async with self._lock:
            if expected_rev is not None and expected_rev != self.rev:
                raise ConflictError(self.snapshot())
            entry = self._guilds.get(str(guild_id))
            default_val = None if field == "channel_id" else (True if field == "enabled" else 1.0)
            current = entry.get(field, default_val) if entry else default_val
            if entry is not None and field in entry and current == value:
                return self.snapshot()
            return await self._mutate(guild_id, field, value, editor)

    async def _mutate(self, guild_id, field, value, editor) -> dict:
        """Apply, persist, publish. Caller holds the lock."""
        key = str(guild_id)
        had = key in self._guilds
        default_val = None if field == "channel_id" else (True if field == "enabled" else 1.0)
        entry = self._guilds.get(key) or {"channel_id": None, "enabled": True}
        old = entry.get(field, default_val)
        prev = (self.rev, self.updated_by, self.updated_at, dict(self._guilds.get(key) or {}))
        self._guilds[key] = {**entry, field: value}
        self.rev += 1
        self.updated_by = editor
        self.updated_at = time.time()
        snap = self.snapshot()
        try:
            await asyncio.to_thread(atomic_write_json, snap, self.path)
        except BaseException:
            self.rev, self.updated_by, self.updated_at = prev[:3]
            if had:
                self._guilds[key] = prev[3]
            else:
                self._guilds.pop(key, None)
            raise
        self.hub.publish({
            "type": "config",
            "snapshot": snap,
            "change": {"guild_id": guild_id, "field": field, "old": old, "new": value},
        })
        return snap
