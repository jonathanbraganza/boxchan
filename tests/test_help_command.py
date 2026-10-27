import asyncio
import os
import tempfile
import unittest
from unittest import mock

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import botimport  # noqa: F401  (hermetic env + import of bot; must come first)
import discord
from discord.ext import commands

from config_store import ConfigStore

bot = botimport.bot


def assert_no_mentions(tc, send_mock):
    """The reply MUST carry allowed_mentions == AllowedMentions.none() (unconditionally)."""
    kwargs = send_mock.await_args[1]
    tc.assertIn("allowed_mentions", kwargs)
    tc.assertIsInstance(kwargs["allowed_mentions"], discord.AllowedMentions)
    tc.assertEqual(kwargs["allowed_mentions"].to_dict(), discord.AllowedMentions.none().to_dict())


def make_guild(gid, channels=(), vc=None):
    g = mock.MagicMock()
    g.id = gid
    g.voice_client = vc

    def get_channel(cid):
        for c in channels:
            if c.id == cid:
                return c
        return None

    g.get_channel = get_channel
    g.get_member = mock.MagicMock(return_value=None)
    return g


def make_text_channel(cid, name="test-channel"):
    ch = mock.MagicMock(spec=discord.TextChannel)
    ch.id = cid
    ch.name = name
    return ch


def make_member(uid, name="testuser"):
    m = mock.MagicMock()
    m.id = uid
    m.display_name = name
    m.mention = f"<@{uid}>"
    return m


def make_context(guild, author):
    ctx = mock.MagicMock(spec=commands.Context)
    ctx.guild = guild
    ctx.author = author
    ctx.send = mock.AsyncMock()
    ctx.voice_client = None
    return ctx


class CommandTestBase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.config = ConfigStore(os.path.join(tmp.name, "c.json"))
        for p in (mock.patch.object(bot, "config", self.config),
                  mock.patch.object(bot.store, "data", {}),
                  mock.patch.dict(bot.audio, {}, clear=True)):
            p.start()
            self.addCleanup(p.stop)


class HelpCommandTest(CommandTestBase):
    async def run_help(self, guild=None, vc=None, configure=True):
        guild = guild or make_guild(1, [make_text_channel(10)])
        if configure:
            await self.config.set_channel(1, 10, expected_rev=None, editor="t")
        ctx = make_context(guild, make_member(1, "testuser"))
        ctx.voice_client = vc
        await bot.help_cmd(ctx)
        ctx.send.assert_awaited_once()
        return ctx, ctx.send.await_args[0][0]

    async def test_help_lists_all_registered_commands(self):
        ctx, text = await self.run_help()
        self.assertEqual({c.name for c in bot.bot.commands},
                         {"colourpanel", "help", "join", "leave", "setvoice", "skip", "speed", "status", "stop", "voices", "volume", "watch"})
        for cmd in bot.bot.commands:
            self.assertIn(f"`{bot.PREFIX}{cmd.name}`", text)
            self.assertIn(cmd.description, text)

    async def test_help_shows_listen_channel(self):
        ctx, text = await self.run_help()
        self.assertIn("Listen channel: <#10>", text)

    async def test_help_shows_unconfigured_channel(self):
        ctx, text = await self.run_help(configure=False)
        self.assertIn("Listen channel: not set", text)
        self.assertNotIn("<#", text)

    async def test_help_voice_connected(self):
        vc = mock.MagicMock()
        vc.is_connected.return_value = True
        vc.channel.name = "General"
        ctx, text = await self.run_help(vc=vc)
        self.assertIn("Voice: connected to **General**", text)

    async def test_help_stale_voice_client_is_not_connected(self):
        vc = mock.MagicMock()
        vc.is_connected.return_value = False
        vc.channel.name = "General"
        ctx, text = await self.run_help(vc=vc)
        self.assertIn("Voice: not connected", text)
        self.assertNotIn("General", text)

    async def test_help_engine_not_loaded(self):
        with mock.patch.object(bot.engine, "ready", False):
            ctx, text = await self.run_help()
        self.assertIn("Active voice: not loaded / default", text)

    async def test_help_engine_loaded_shows_voice_and_cpu_marker(self):
        pm = lambda v: mock.patch.object(type(bot.engine), "cuda_active", new_callable=mock.PropertyMock, return_value=v)
        name = mock.patch.object(type(bot.engine), "model_name", new_callable=mock.PropertyMock, return_value="amy")
        with mock.patch.object(bot.engine, "ready", True), name:
            with pm(True):
                _, gpu = await self.run_help()
            with pm(False):
                _, cpu = await self.run_help()
        self.assertIn("Active voice: **amy**", gpu)
        self.assertNotIn("(CPU)", gpu)
        self.assertIn("Active voice: **amy** (CPU)", cpu)

    async def test_help_status_idle_reading_paused(self):
        _, idle = await self.run_help()
        self.assertIn("Status: idle", idle)

        bot.audio[1] = mock.MagicMock()
        _, reading = await self.run_help()
        self.assertIn("Status: **reading**", reading)
        self.assertNotIn("Status: **paused**", reading)

        await self.config.set_enabled(1, False, expected_rev=None, editor="t")
        _, paused = await self.run_help()
        self.assertIn("Status: **paused**", paused)
        self.assertNotIn("Status: **reading**", paused)

    async def test_help_shows_watched_users(self):
        guild = make_guild(1, [make_text_channel(10)])
        guild.get_member = mock.MagicMock(side_effect=lambda uid: make_member(uid, f"user{uid}") if uid != 7 else None)
        bot.store.data[1] = {5, 6, 7}
        _, text = await self.run_help(guild=guild)
        self.assertIn("Watched users: user5, user6, 7", text)

    async def test_help_no_watched_users(self):
        _, text = await self.run_help()
        self.assertIn("Watched users: none", text)

    async def test_help_sends_no_mentions(self):
        ctx, _ = await self.run_help()
        assert_no_mentions(self, ctx.send)

    async def test_help_caps_watched_users_at_15_without_truncation_backstop(self):
        guild = make_guild(1, [make_text_channel(10)])
        guild.get_member = mock.MagicMock(side_effect=lambda uid: make_member(uid, f"user{uid:03d}"))
        bot.store.data[1] = set(range(100))
        ctx, text = await self.run_help(guild=guild)
        line = [l for l in text.split("\n") if l.startswith("Watched users:")][0]
        shown = line[len("Watched users: "):].split(", ")
        self.assertEqual(shown[:15], [f"user{i:03d}" for i in range(15)])
        self.assertEqual(shown[15], "\u2026 +85 more")
        self.assertEqual(len(shown), 16)
        self.assertNotIn("user015", text)
        # The 2000-char backstop would produce exactly 1997 chars ending in an ellipsis; it must not have fired.
        self.assertLess(len(text), 1996)
        self.assertTrue(text.endswith("+85 more"))
        assert_no_mentions(self, ctx.send)

    async def test_help_truncation_backstop_caps_at_2000(self):
        guild = make_guild(1, [make_text_channel(10)])
        guild.get_member = mock.MagicMock(side_effect=lambda uid: make_member(uid, "x" * 300))
        bot.store.data[1] = set(range(15))
        _, text = await self.run_help(guild=guild)
        self.assertLessEqual(len(text), 2000)
        self.assertTrue(text.endswith("\u2026"))

    async def test_help_hostile_display_names_are_harmless(self):
        guild = make_guild(1, [make_text_channel(10)])
        names = {1: "@everyone", 2: "**bold** `code` <@&9>", 3: "\u00fcn\u00ef\U0001f642"}
        guild.get_member = mock.MagicMock(side_effect=lambda uid: make_member(uid, names[uid]))
        bot.store.data[1] = {1, 2, 3}
        ctx, text = await self.run_help(guild=guild)
        for n in names.values():
            self.assertIn(n, text)
        assert_no_mentions(self, ctx.send)

    async def test_help_no_webui_address(self):
        _, text = await self.run_help()
        self.assertNotIn("127.0", text)
        self.assertNotIn("100.6", text)
        self.assertNotIn("WEBUI", text)


if __name__ == "__main__":
    unittest.main()
