import asyncio
import json
import os
import tempfile
import unittest
from unittest import mock

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import botimport  # noqa: F401
import discord
from discord.ext import commands

import config_store
from config_store import ConfigStore, ConflictError
import webui
from aiohttp.test_utils import TestClient, TestServer
import tts_engine
from tts_engine import TTSEngine, VoiceState

bot = botimport.bot
ED = {"X-Editor": "alice"}


def make_guild(gid, channels=()):
    g = mock.MagicMock()
    g.id = gid

    def get_channel(cid):
        for c in channels:
            if c.id == cid:
                return c
        return None

    g.get_channel = get_channel
    g.members = []
    g.get_member = lambda uid: None
    return g


def make_member(uid, name="testuser"):
    m = mock.MagicMock()
    m.id = uid
    m.display_name = name
    return m


def make_context(guild, author):
    ctx = mock.MagicMock(spec=commands.Context)
    ctx.guild = guild
    ctx.author = author
    ctx.send = mock.AsyncMock()
    ctx.defer = mock.AsyncMock()
    return ctx


class ConfigStoreAudioTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "config.json")
        self.store = ConfigStore(self.path)

    async def test_speed_defaults_to_1(self):
        self.assertEqual(self.store.speed_for(1), 1.0)
        self.assertEqual(self.store.volume_for(1), 1.0)

    async def test_set_speed_and_event(self):
        q = self.store.hub.subscribe()
        self.addCleanup(lambda: self.store.hub.unsubscribe(q))

        snap = await self.store.set_speed(1, 1.25, expected_rev=None, editor="bob")
        self.assertEqual(self.store.speed_for(1), 1.25)
        self.assertEqual(snap["guilds"]["1"]["speed"], 1.25)

        ev = await q.get()
        self.assertEqual(ev["type"], "config")
        self.assertEqual(ev["change"]["field"], "speed")
        self.assertEqual(ev["change"]["old"], 1.0)
        self.assertEqual(ev["change"]["new"], 1.25)

    async def test_set_volume_and_event(self):
        q = self.store.hub.subscribe()
        self.addCleanup(lambda: self.store.hub.unsubscribe(q))

        snap = await self.store.set_volume(1, 0.75, expected_rev=None, editor="bob")
        self.assertEqual(self.store.volume_for(1), 0.75)
        self.assertEqual(snap["guilds"]["1"]["volume"], 0.75)

        ev = await q.get()
        self.assertEqual(ev["type"], "config")
        self.assertEqual(ev["change"]["field"], "volume")
        self.assertEqual(ev["change"]["old"], 1.0)
        self.assertEqual(ev["change"]["new"], 0.75)

    async def test_speed_bounds(self):
        with self.assertRaises(ValueError):
            await self.store.set_speed(1, 0.49)
        with self.assertRaises(ValueError):
            await self.store.set_speed(1, 2.01)

    async def test_volume_bounds(self):
        with self.assertRaises(ValueError):
            await self.store.set_volume(1, 0.09)
        with self.assertRaises(ValueError):
            await self.store.set_volume(1, 2.01)

    async def test_expected_rev_conflict(self):
        await self.store.set_speed(1, 1.2, expected_rev=0, editor="bob")
        with self.assertRaises(ConflictError):
            await self.store.set_speed(1, 1.3, expected_rev=0, editor="bob")


class FakeAudioController:
    def __init__(self):
        self.say_calls = []
        self.watched = [{"id": "42", "name": "alice", "voice": "en_US-lessac-medium"}]
        self.user_voice_calls = []

    def guilds_view(self):
        return [{"id": "1", "name": "TestGuild", "channels": []}]

    def status_view(self):
        return {}

    def is_text_channel(self, gid, cid):
        return True

    def say(self, gid, text, editor="", voice=None):
        self.say_calls.append((gid, text, editor, voice))
        return {"queued": True}

    def watched_users(self, gid):
        return self.watched

    async def set_user_voice(self, gid, uid, voice):
        self.user_voice_calls.append((gid, uid, voice))


class WebUIAudioTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.static = os.path.join(self.tmp.name, "static")
        os.mkdir(self.static)
        for n, c in (("index.html", "<h1>hi</h1>"), ("app.js", "//js"), ("app.css", "/*css*/")):
            with open(os.path.join(self.static, n), "w") as f:
                f.write(c)
        self.cfgpath = os.path.join(self.tmp.name, "c.json")
        self.config = ConfigStore(self.cfgpath)
        self.ctl = FakeAudioController()
        self.app = webui.make_app(self.config, self.ctl, allowed_hosts=[], static_dir=self.static)
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)

    async def test_put_speed_success(self):
        r = await self.client.put("/api/guilds/1/speed", json={"speed": 1.25, "rev": 0}, headers=ED)
        self.assertEqual(r.status, 200)
        data = await r.json()
        self.assertEqual(data["snapshot"]["guilds"]["1"]["speed"], 1.25)

    async def test_put_speed_invalid(self):
        r = await self.client.put("/api/guilds/1/speed", json={"speed": 0.2}, headers=ED)
        self.assertEqual(r.status, 400)
        r = await self.client.put("/api/guilds/1/speed", json={"speed": "fast"}, headers=ED)
        self.assertEqual(r.status, 400)

    async def test_put_volume_success(self):
        r = await self.client.put("/api/guilds/1/volume", json={"volume": 1.5, "rev": 0}, headers=ED)
        self.assertEqual(r.status, 200)
        data = await r.json()
        self.assertEqual(data["snapshot"]["guilds"]["1"]["volume"], 1.5)

    async def test_put_volume_invalid(self):
        r = await self.client.put("/api/guilds/1/volume", json={"volume": 0.05}, headers=ED)
        self.assertEqual(r.status, 400)
        r = await self.client.put("/api/guilds/1/volume", json={"volume": True}, headers=ED)
        self.assertEqual(r.status, 400)

    async def test_get_watched_users(self):
        r = await self.client.get("/api/guilds/1/watched")
        self.assertEqual(r.status, 200)
        data = await r.json()
        self.assertEqual(len(data["users"]), 1)
        self.assertEqual(data["users"][0]["voice"], "en_US-lessac-medium")

    async def test_put_user_voice(self):
        r = await self.client.put("/api/guilds/1/watched/42/voice", json={"voice": "en_US-bryce-medium"}, headers=ED)
        self.assertEqual(r.status, 200)
        self.assertEqual(self.ctl.user_voice_calls[-1], (1, 42, "en_US-bryce-medium"))

        r = await self.client.put("/api/guilds/1/watched/42/voice", json={"voice": None}, headers=ED)
        self.assertEqual(r.status, 200)
        self.assertEqual(self.ctl.user_voice_calls[-1], (1, 42, None))

    async def test_put_user_voice_bad_name(self):
        r = await self.client.put("/api/guilds/1/watched/42/voice", json={"voice": "../bad/name"}, headers=ED)
        self.assertEqual(r.status, 400)

    async def test_say_with_voice(self):
        r = await self.client.post("/api/guilds/1/say", json={"text": "hello world", "voice": "en_US-lessac-medium"}, headers=ED)
        self.assertEqual(r.status, 200)
        self.assertEqual(self.ctl.say_calls[-1], (1, "hello world", "alice", "en_US-lessac-medium"))


class BotCommandsAudioTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfgpath = os.path.join(self.tmp.name, "c.json")
        self.config = ConfigStore(self.cfgpath)
        bot.config = self.config

    async def test_speed_command(self):
        guild = make_guild(1)
        ctx = make_context(guild, make_member(10, "Tester"))

        # View current speed
        await bot.speed_cmd(ctx, speed=None)
        ctx.send.assert_awaited()
        self.assertIn("1.00x", ctx.send.await_args[0][0])

        # Set speed
        ctx.send.reset_mock()
        await bot.speed_cmd(ctx, speed=1.5)
        self.assertEqual(self.config.speed_for(1), 1.5)
        self.assertIn("1.50x", ctx.send.await_args[0][0])

        # Set out of bounds
        ctx.send.reset_mock()
        await bot.speed_cmd(ctx, speed=3.0)
        self.assertIn("between 0.5 and 2.0", ctx.send.await_args[0][0])

    async def test_volume_command(self):
        guild = make_guild(1)
        ctx = make_context(guild, make_member(10, "Tester"))

        # View current volume
        await bot.volume_cmd(ctx, volume=None)
        self.assertIn("100%", ctx.send.await_args[0][0])

        # Set volume percentage
        ctx.send.reset_mock()
        await bot.volume_cmd(ctx, volume="75%")
        self.assertEqual(self.config.volume_for(1), 0.75)
        self.assertIn("75%", ctx.send.await_args[0][0])

        # Set volume decimal
        ctx.send.reset_mock()
        await bot.volume_cmd(ctx, volume="1.2")
        self.assertEqual(self.config.volume_for(1), 1.2)
        self.assertIn("120%", ctx.send.await_args[0][0])

        # Set invalid volume
        ctx.send.reset_mock()
        await bot.volume_cmd(ctx, volume="10%")  # 10% is 0.1, valid boundary
        self.assertEqual(self.config.volume_for(1), 0.1)

        ctx.send.reset_mock()
        await bot.volume_cmd(ctx, volume="5%")  # < 10%
        self.assertIn("between 10% (0.1) and 200%", ctx.send.await_args[0][0])

    async def test_watch_store_user_voices(self):
        watch_path = os.path.join(self.tmp.name, "watch.json")
        ws = bot.WatchStore(watch_path)

        # Set custom voice
        await ws.mutate(1, "set_voice", 101, voice="en_US-bryce-medium")
        self.assertEqual(ws.get_user_voice(1, 101), "en_US-bryce-medium")
        self.assertTrue(ws.is_watched(1, 101))

        # Snapshot contains user_voices
        snap = ws.snapshot()
        self.assertEqual(snap["user_voices"]["1"]["101"], "en_US-bryce-medium")

        # Mutate remove clears user voice
        await ws.mutate(1, "remove", 101)
        self.assertIsNone(ws.get_user_voice(1, 101))
        self.assertFalse(ws.is_watched(1, 101))

        # Add multiple, clear clears all
        await ws.mutate(1, "set_voice", 102, voice="voice-a")
        await ws.mutate(1, "set_voice", 103, voice="voice-b")
        await ws.mutate(1, "clear")
        self.assertIsNone(ws.get_user_voice(1, 102))
        self.assertIsNone(ws.get_user_voice(1, 103))


class TTSEngineVoiceCacheTests(unittest.TestCase):
    def test_secondary_voice_cache(self):
        eng = TTSEngine()
        self.addCleanup(eng.close)

        fake_voice1 = mock.MagicMock()
        fake_voice1.synthesize_stream_raw.return_value = [b"\x00\x00" * 100]
        fake_voice2 = mock.MagicMock()
        fake_voice2.synthesize_stream_raw.return_value = [b"\x00\x00" * 100]

        eng._state = VoiceState(voice=fake_voice1, config={"sample_rate": 22050}, name="primary")

        # Mock loading secondary voice
        with mock.patch("tts_engine.resolve_voice", return_value=("/tmp/v2.onnx", "/tmp/v2.json")), \
             mock.patch.object(eng, "_load_piper_voice", return_value=(fake_voice2, {"sample_rate": 22050})):
            state, from_cache = eng._get_voice_for_synth("secondary")
            self.assertEqual(state.name, "secondary")
            self.assertFalse(from_cache)

            # Second call should be from cache
            state2, from_cache2 = eng._get_voice_for_synth("secondary")
            self.assertEqual(state2.name, "secondary")
            self.assertTrue(from_cache2)


if __name__ == "__main__":
    unittest.main()
