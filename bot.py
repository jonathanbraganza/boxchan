import asyncio
import json
import logging
import os
import re
import signal
import sys
import time

import discord
from discord.ext import commands
from dotenv import load_dotenv

from colour_roles import ColourRolesView
from config_store import ConfigStore, ConflictError, atomic_write_json
from messages import MAX_MESSAGE_CHARS, MessageError, SendLimiter, parse_mentions
import engines
import nag
from tts_engine import TTSEngine, clean_discord_text, clean_plain_text
from voices import BadVoiceError, SayError, VoiceLoadError, discover_voices, resolve_voice
from webui import (detect_tailscale_addrs, make_app, parse_allowed_hosts, start_webui,
                   validate_bind_hosts)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("bot")



TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{23,30}\.[A-Za-z0-9_-]{6,7}\.[A-Za-z0-9_-]{27,38}$")


def load_token() -> str | None:
    """Token sources, safest first: systemd credential, token file, plain env var."""
    cred_dir = os.getenv("CREDENTIALS_DIRECTORY")
    candidates = []
    if cred_dir:
        candidates.append(os.path.join(cred_dir, "discord_token"))
    candidates.append(os.path.expanduser(os.getenv("DISCORD_TOKEN_FILE", "~/.config/tts-bot/token")))
    for path in candidates:
        try:
            with open(path) as f:
                tok = f.read().strip()
            if tok:
                return tok
        except OSError:
            continue
    return os.getenv("DISCORD_TOKEN")


TOKEN = load_token()
if TOKEN and not TOKEN_RE.match(TOKEN):
    # Never log the value itself.
    raise SystemExit(f"Token malformed (len={len(TOKEN)}): expected '<id>.<ts>.<hmac>'. "
                     "No quotes, 'Bot ' prefix, or newline. Re-run scripts/install_token.sh")
PREFIX = os.getenv("COMMAND_PREFIX", "!")
MAX_PERM_CHECKS = 2000  # Cap for permission checks in member scan
MODEL_PATH = os.getenv("PIPER_MODEL_PATH", "models/en_US-lessac-medium.onnx")
CONFIG_PATH = os.getenv("PIPER_CONFIG_PATH") or None
MODELS_DIR = os.getenv("PIPER_MODELS_DIR") or os.path.dirname(MODEL_PATH) or "."
SAY_MAX_CHARS = 200
SEED_CHANNEL_IDS = {int(x) for x in os.getenv("TTS_CHANNEL_ID", "").replace(" ", "").split(",") if x.isdigit()}
MAX_CHARS = int(os.getenv("MAX_CHARS", "300"))
MAX_QUEUE = int(os.getenv("MAX_QUEUE", "20"))
STATE_FILE = os.getenv("STATE_FILE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.json"))
CONFIG_FILE = os.getenv("CONFIG_FILE", os.path.join(os.path.dirname(STATE_FILE), "config.json"))
USERS_FILE = os.getenv("USERS_FILE") or os.path.join(os.path.dirname(CONFIG_FILE), "users.json")  # empty value = default, never "no auth"
FEATURES_FILE = os.getenv("FEATURES_FILE", os.path.join(os.path.dirname(STATE_FILE), "features.json"))


def parse_port(raw: str | None, default: int = 8765) -> int:
    try:
        port = int((raw or "").strip() or default)
    except ValueError:
        port = 0
    if not 1 <= port <= 65535:
        logging.getLogger("bot").warning("Invalid WEBUI_PORT %r; using %d", raw, default)
        return default
    return port


WEBUI_PORT = parse_port(os.getenv("WEBUI_PORT"))
RECONNECT_WAIT = 10.0   # s to wait for voice reconnect before dropping a job
PLAY_TIMEOUT = 180.0    # safety: max s a single clip may block the worker

# Conda env bin/lib: ffmpeg + libopus live there; make them findable under systemd too.
_env_prefix = os.path.dirname(os.path.dirname(os.path.abspath(sys.executable)))
os.environ["PATH"] = os.path.join(_env_prefix, "bin") + os.pathsep + os.environ.get("PATH", "")
if not discord.opus.is_loaded():
    for _lib in (os.path.join(_env_prefix, "lib", "libopus.so.0"), "libopus.so.0"):
        try:
            discord.opus.load_opus(_lib)
            break
        except Exception:
            continue
    if not discord.opus.is_loaded():
        log.warning("libopus not loaded; voice playback will fail")

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
intents.members = True  # needed to resolve mentions/display names

engine = TTSEngine(MODEL_PATH, CONFIG_PATH)


# ---------- persistent state ----------
class WatchStore:
    """guild_id -> set(user_id) + guild_id -> voice_channel_id; mutations serialized by asyncio.Lock."""

    def __init__(self, path: str):
        self.path = path
        self.data: dict[int, set[int]] = {}
        self.voice_channels: dict[int, int] = {}
        self.lock = asyncio.Lock()
        try:
            with open(path) as f:
                raw = json.load(f)
            self.data = {int(g): {int(u) for u in us} for g, us in raw.get("watched", {}).items()}
            self.voice_channels = {int(g): int(vc) for g, vc in raw.get("voice_channels", {}).items() if vc}
        except FileNotFoundError:
            pass
        except Exception:
            log.exception("state file unreadable; starting empty")

    def is_watched(self, guild_id: int, user_id: int) -> bool:
        return user_id in self.data.get(guild_id, ())

    def get_voice_channel(self, guild_id: int) -> int | None:
        return self.voice_channels.get(guild_id)

    def snapshot(self) -> dict:
        snap = {"watched": {str(g): sorted(u) for g, u in self.data.items() if u}}
        if self.voice_channels:
            snap["voice_channels"] = {str(g): vc for g, vc in self.voice_channels.items() if vc}
        return snap

    async def set_voice_channel(self, guild_id: int, channel_id: int | None) -> None:
        async with self.lock:
            if channel_id:
                self.voice_channels[guild_id] = channel_id
            else:
                self.voice_channels.pop(guild_id, None)
            snap = self.snapshot()
            try:
                await asyncio.to_thread(atomic_write_json, snap, self.path)
            except Exception:
                log.exception("state save failed")

    async def mutate(self, guild_id: int, op: str, user_id: int | None = None) -> None:
        async with self.lock:
            users = self.data.setdefault(guild_id, set())
            if op == "add":
                users.add(user_id)
            elif op == "remove":
                users.discard(user_id)
            elif op == "clear":
                users.clear()
            snap = self.snapshot()
            try:
                await asyncio.to_thread(atomic_write_json, snap, self.path)
            except Exception:
                log.exception("state save failed")


store = WatchStore(STATE_FILE)
config = ConfigStore(CONFIG_FILE)


# ---------- per-guild audio ----------
class GuildAudio:
    def __init__(self, guild_id: int):
        self.guild_id = guild_id
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=MAX_QUEUE)
        self.play_done = asyncio.Event()
        self.worker: asyncio.Task | None = None


class TTSBot(commands.Bot):
    async def setup_hook(self):
        self.add_view(ColourRolesView())
        try:
            await self.tree.sync()
        except Exception:
            log.exception("slash command sync failed")


bot = TTSBot(command_prefix=PREFIX, intents=intents, help_command=None)
audio: dict[int, GuildAudio] = {}
_shutting_down = False
_initial_ready = False
_watcher_task: asyncio.Task | None = None
_idle_task: asyncio.Task | None = None
_nag_task: asyncio.Task | None = None
_webui_runner = None


def _rm(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def drain(q: asyncio.Queue) -> None:
    while True:
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            return


def ensure_audio(guild_id: int) -> GuildAudio:
    ga = audio.get(guild_id)
    if ga is None:
        ga = audio[guild_id] = GuildAudio(guild_id)
    if ga.worker is None or ga.worker.done():
        ga.worker = asyncio.create_task(audio_worker(ga), name=f"tts-worker-{guild_id}")
    return ga


async def wait_connected(guild: discord.Guild) -> discord.VoiceClient | None:
    """Tolerate transient voice reconnects (UDP drop / region move)."""
    loop = asyncio.get_running_loop()
    end = loop.time() + RECONNECT_WAIT
    while loop.time() < end:
        vc = guild.voice_client
        if vc is None:
            return None
        if vc.is_connected():
            return vc
        await asyncio.sleep(0.5)
    return None


async def audio_worker(ga: GuildAudio) -> None:
    """One per guild. Survives errors/reconnects; only cancellation ends it."""
    loop = asyncio.get_running_loop()
    while True:
        text = await ga.queue.get()
        path = None
        try:
            guild = bot.get_guild(ga.guild_id)
            if guild is None or await wait_connected(guild) is None:
                continue
            path = await engine.synthesize(text)  # off-loop, GPU-serialized
            if not path:
                continue
            vc = await wait_connected(guild)
            if vc is None:
                continue
            ga.play_done.clear()

            def after(err, p=path):
                _rm(p)
                if err:
                    log.error("playback error: %s", err)
                loop.call_soon_threadsafe(ga.play_done.set)

            try:
                vc.play(discord.FFmpegPCMAudio(path, options="-loglevel error"), after=after)
            except Exception:
                log.exception("vc.play failed")
                continue  # path cleaned in finally
            path = None  # now owned by after()
            try:
                await asyncio.wait_for(ga.play_done.wait(), timeout=PLAY_TIMEOUT)
            except asyncio.TimeoutError:
                log.error("playback timeout; stopping")
                vc.stop()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("worker error (continuing)")
        finally:
            if path:
                _rm(path)


async def teardown(guild: discord.Guild) -> None:
    """Clear queue, cancel worker, stop playback, disconnect."""
    ga = audio.pop(guild.id, None)
    if ga:
        drain(ga.queue)
        if ga.worker:
            ga.worker.cancel()
    vc = guild.voice_client
    if vc:
        try:
            vc.stop()
            await vc.disconnect(force=True)
        except Exception:
            log.exception("disconnect failed for guild %s", guild.id)


# ---------- events ----------
def has_big_bots_role(member) -> bool:
    """Return True if member has the 'Big bots' role (case-insensitive)."""
    if not member:
        return False
    roles = getattr(member, "roles", None)
    if not isinstance(roles, (list, tuple, set)):
        return False
    return any(isinstance(getattr(r, "name", None), str) and getattr(r, "name", "").strip().casefold() == "big bots" for r in roles)


@bot.check
async def only_tts_channel(ctx: commands.Context) -> bool:
    """Commands work only in the configured text channel, unless caller has 'Big bots' role."""
    if not ctx.guild:
        raise commands.CheckFailure("wrong channel")
    if ctx.command and ctx.command.name in ("colourpanel", "colorpanel"):
        return True
    if has_big_bots_role(ctx.author):
        return True
    if not config.is_command_channel(ctx.guild.id, ctx.channel.id):
        raise commands.CheckFailure("wrong channel")
    return True


async def reconnect_voice_channels():
    for guild in bot.guilds:
        vc_id = store.get_voice_channel(guild.id)
        if not vc_id:
            continue
        if guild.voice_client and guild.voice_client.is_connected():
            continue
        vchannel = guild.get_channel(vc_id)
        if not isinstance(vchannel, discord.VoiceChannel):
            log.warning("Saved voice channel %s not found in guild %s; clearing", vc_id, guild.id)
            await store.set_voice_channel(guild.id, None)
            continue
        me = guild.me or guild.get_member(bot.user.id)
        perms = vchannel.permissions_for(me) if me else None
        if not perms or not (perms.connect and perms.speak):
            log.warning("Missing Connect/Speak permissions in voice channel %s in guild %s", vchannel.name, guild.id)
            await store.set_voice_channel(guild.id, None)
            continue
        try:
            log.info("Reconnecting to voice channel %s (%s) in guild %s", vchannel.name, vchannel.id, guild.id)
            await vchannel.connect()
            ga = ensure_audio(guild.id)
            try:
                ga.queue.put_nowait("Updated and reconnected.")
            except Exception:
                pass

            txt_cid = config.channel_for(guild.id)
            txt_ch = guild.get_channel(txt_cid) if txt_cid else None
            msg = f"Updated and reconnected to **{vchannel.name}**."
            if txt_ch and isinstance(txt_ch, discord.TextChannel) and me and txt_ch.permissions_for(me).send_messages:
                await txt_ch.send(msg)
            elif me and vchannel.permissions_for(me).send_messages:
                await vchannel.send(msg)
        except Exception:
            log.exception("Failed to reconnect to voice channel %s in guild %s", vc_id, guild.id)


@bot.event
async def on_ready():
    global _initial_ready
    for guild in bot.guilds:  # env TTS_CHANNEL_ID is a first-run seed only
        for cid in sorted(SEED_CHANNEL_IDS):
            if isinstance(guild.get_channel(cid), discord.TextChannel):
                try:
                    await config.seed_channel(guild.id, cid)
                except Exception:
                    log.exception("channel seed failed for guild %s", guild.id)
                break
    # GUI tabs that connected before the guild cache filled (restart window) must refetch.
    config.hub.publish({"type": "channels", "guild_id": None})
    log.info("Logged in as %s (id=%s) prefix=%r channels=%s tts_ready=%s cuda=%s",
             bot.user, bot.user.id, PREFIX,
             {g: c for g, e in config.snapshot()["guilds"].items() if (c := e["channel_id"])},
             engine.ready, engine.cuda_active)

    if not _initial_ready:
        _initial_ready = True
        for guild in bot.guilds:
            if not getattr(guild, "chunked", False):
                try:
                    await asyncio.wait_for(guild.chunk(), timeout=10)
                except Exception as e:
                    log.warning("member chunk failed for guild %s (%s); using cache", guild.id, type(e).__name__)
        await reconnect_voice_channels()


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot or not message.guild:
        return
    if message.content.startswith(PREFIX):  # global check limits commands to the configured channel
        await bot.process_commands(message)
        return
    if not config.is_enabled(message.guild.id):  # paused: ignore
        return
    if not store.is_watched(message.guild.id, message.author.id):
        return
    vc = message.guild.voice_client
    ga = audio.get(message.guild.id)
    if not vc or not ga:
        return
    text = clean_discord_text(message, MAX_CHARS)
    if not text:  # empty-result guard: never reaches queue/GPU
        return
    try:
        ga.queue.put_nowait(text)
    except asyncio.QueueFull:
        log.warning("queue full (guild %s); dropped message %s", message.guild.id, message.id)
        try:
            await message.add_reaction("⏳")
        except discord.HTTPException:
            pass


@bot.event
async def on_voice_state_update(member, before, after):
    guild = member.guild
    vc = guild.voice_client
    if member.id == bot.user.id:
        if _shutting_down:
            return
        if after.channel is None:  # kicked / dropped for good
            await store.set_voice_channel(guild.id, None)
            if guild.id in audio:
                await teardown(guild)
        else:
            await store.set_voice_channel(guild.id, after.channel.id)
        return
    if vc and vc.channel and before.channel == vc.channel and not any(not m.bot for m in vc.channel.members):
        log.info("auto-leave guild %s (empty)", guild.id)
        await store.mutate(guild.id, "clear")
        await store.set_voice_channel(guild.id, None)
        await teardown(guild)


# ---------- commands (work as !cmd and /cmd) ----------
def voice_connected(vc) -> bool:
    """True only for a live voice client; a stale, disconnected one counts as not connected."""
    return bool(vc and vc.is_connected())


async def join_author_voice(ctx: commands.Context) -> tuple[bool, str]:
    """Helper to join the author's voice channel. Returns (ok, message).

    Handles all permission checks, connection attempts, and error handling.
    Validation failures return without deferring; the interaction is deferred only after
    validation passes, just before the (slow) connect/move.
    """
    # Early validation (no defer yet)
    if not ctx.author.voice or not ctx.author.voice.channel:
        return False, "Join a voice channel first."
    channel = ctx.author.voice.channel
    perms = channel.permissions_for(ctx.guild.me)
    if not (perms.connect and perms.speak):
        return False, f"Missing Connect/Speak permission in {channel.name}."

    # Validation passed: ack the interaction before the slow connect
    await ctx.defer()
    try:
        if ctx.voice_client:
            await ctx.voice_client.move_to(channel)
        else:
            await channel.connect()
    except discord.Forbidden:
        return False, "Forbidden: cannot join that channel."
    except (discord.ClientException, asyncio.TimeoutError) as e:
        log.exception("join failed")
        return False, f"Join failed: {e}"

    ensure_audio(ctx.guild.id)
    await store.set_voice_channel(ctx.guild.id, channel.id)
    msg = f"Joined **{channel.name}**." + ("" if engine.ready else " TTS engine NOT loaded; check logs.")
    return True, msg


@bot.hybrid_command(name="join", description="Join your voice channel")
@commands.guild_only()
async def join(ctx: commands.Context):
    ok, msg = await join_author_voice(ctx)
    await ctx.send(msg)


@bot.hybrid_command(name="leave", description="Leave voice, clear queue and watch list")
@commands.guild_only()
async def leave(ctx: commands.Context):
    await store.mutate(ctx.guild.id, "clear")
    await store.set_voice_channel(ctx.guild.id, None)
    had_vc = ctx.voice_client is not None
    await teardown(ctx.guild)
    await ctx.send("Left. Queue cleared, watch list flushed." if had_vc else "Not in voice. Watch list flushed.")


@bot.hybrid_command(name="watch", description="Read this user's messages aloud, optionally join voice")
@commands.guild_only()
async def watch(ctx: commands.Context, member: discord.Member):
    # Always add to watch list first
    await store.mutate(ctx.guild.id, "add", member.id)
    reply = f"Watching **{member.display_name}**."

    # Try to auto-join if not connected and author in voice
    if not voice_connected(ctx.voice_client):
        if ctx.author.voice and ctx.author.voice.channel:  # author in voice
            ok, join_msg = await join_author_voice(ctx)
            reply += f" {join_msg}"
        else:
            # Author not in a voice channel
            reply += f" Run `{PREFIX}join` from a voice channel to start reading."
    # else: already connected, don't try to join

    await ctx.send(reply, allowed_mentions=discord.AllowedMentions.none())


@bot.hybrid_command(name="stop", description="Stop reading this user's messages")
@commands.guild_only()
async def stop(ctx: commands.Context, member: discord.Member):
    await store.mutate(ctx.guild.id, "remove", member.id)
    await ctx.send(f"Stopped watching **{member.display_name}**.")


@bot.hybrid_command(name="skip", description="Skip the clip currently playing")
@commands.guild_only()
async def skip(ctx: commands.Context):
    vc = ctx.voice_client
    if vc and vc.is_playing():
        vc.stop()  # fires after() -> worker event -> next clip
        await ctx.send("Skipped.")
    else:
        await ctx.send("Nothing playing.")


@bot.hybrid_command(name="status", description="Show watched users and voice state")
@commands.guild_only()
async def status(ctx: commands.Context):
    names = []
    for uid in sorted(store.data.get(ctx.guild.id, ())):
        m = ctx.guild.get_member(uid)
        names.append(m.display_name if m else str(uid))
    vc = ctx.voice_client
    voice = f"connected to **{vc.channel.name}**" if voice_connected(vc) else "not connected"
    ga = audio.get(ctx.guild.id)
    await ctx.send(
        f"Voice: {voice}\nWatched: {', '.join(names) or 'none'}\n"
        f"Queue: {ga.queue.qsize() if ga else 0}/{MAX_QUEUE} | TTS: {'ready' if engine.ready else 'unavailable'}"
        f" | CUDA: {'yes' if engine.cuda_active else 'no'}"
    )


@bot.hybrid_command(name="voices", description="List available TTS voices")
@commands.guild_only()
async def voices(ctx: commands.Context):
    """Show available Piper voices and which one is currently active."""
    await ctx.defer()  # list_voices rescans the models dir; a slash interaction must be acked first
    try:
        result = await controller.list_voices()
    except Exception:
        log.exception("list_voices failed")
        await ctx.send("Could not list voices; check logs.", allowed_mentions=discord.AllowedMentions.none())
        return

    voice_list = result.get("voices", [])
    current = result.get("current")
    configured = result.get("configured")
    loading = result.get("loading")
    cuda = result.get("cuda")

    if loading:
        await ctx.send("Voice is loading; please wait...", allowed_mentions=discord.AllowedMentions.none())
        return

    lines = []

    if not voice_list:
        if current is None:
            lines.append("TTS engine not loaded.")
        else:
            lines.append("No voices available.")
    else:
        lines.append("**Available Voices**")

        # Show up to 25 voices
        for i, v in enumerate(voice_list[:25]):
            name = v.get("name", "?")
            quality = v.get("quality", "?")
            language = v.get("language", "?")
            size_mb = v.get("size_mb", "?")

            line = f"• {name} — {quality}, {language}, {size_mb} MB"

            # Mark active voice
            if name == current:
                cuda_str = " (CPU)" if cuda is False else ""
                line += f" ◀ active{cuda_str}"
            elif name == configured and configured != current:
                line += " (saved, not loaded)"

            lines.append(line)

        # Truncation
        if len(voice_list) > 25:
            extra = len(voice_list) - 25
            lines.append(f"… +{extra} more")

        lines.append("")
        if current:
            lines.append(f"Current voice: **{current}**. Change voices in the web UI.")
        else:
            lines.append("TTS engine not loaded; no voice is active. Change voices in the web UI.")

    reply = "\n".join(lines)

    # Truncate to 2000 chars if needed
    if len(reply) > 2000:
        reply = reply[:1996] + "…"

    await ctx.send(reply, allowed_mentions=discord.AllowedMentions.none())


@bot.hybrid_command(name="help", description="Show available commands and current status")
@commands.guild_only()
async def help_cmd(ctx: commands.Context):
    """Show all registered commands and current bot state."""
    lines = []

    # Header
    lines.append("**Available Commands**")

    # List all commands (sorted by name)
    sorted_commands = sorted(bot.commands, key=lambda c: c.name)
    for cmd in sorted_commands:
        desc = cmd.description or "No description"
        lines.append(f"`{PREFIX}{cmd.name}` — {desc}")

    lines.append("")  # blank line
    lines.append("**Right Now**")

    # Listen channel
    cid = config.channel_for(ctx.guild.id)
    if cid:
        lines.append(f"Listen channel: <#{cid}>")
    else:
        lines.append("Listen channel: not set")

    # Voice connection
    vc = ctx.voice_client
    if voice_connected(vc):
        lines.append(f"Voice: connected to **{vc.channel.name}**")
    else:
        lines.append("Voice: not connected")

    # Active voice
    if engine.ready:
        lines.append(f"Active voice: **{engine.model_name}**" + ("" if engine.cuda_active else " (CPU)"))
    else:
        lines.append("Active voice: not loaded / default")

    # Paused status
    ga = audio.get(ctx.guild.id)
    enabled = config.is_enabled(ctx.guild.id)
    if ga and not enabled:
        lines.append("Status: **paused**")
    elif ga:
        lines.append("Status: **reading**")
    else:
        lines.append("Status: idle")

    # Watched users
    watched_ids = sorted(store.data.get(ctx.guild.id, ()))
    watched_names = []
    for uid in watched_ids:
        m = ctx.guild.get_member(uid)
        if m:
            watched_names.append(m.display_name)
        else:
            watched_names.append(str(uid))

    # Cap at 15 watched users, then add "+N more"
    if len(watched_names) > 15:
        extra = len(watched_names) - 15
        watched_names = watched_names[:15]
        watched_names.append(f"… +{extra} more")

    if watched_names:
        lines.append(f"Watched users: {', '.join(watched_names)}")
    else:
        lines.append("Watched users: none")

    reply = "\n".join(lines)

    # Truncate to 2000 chars if needed
    if len(reply) > 2000:
        reply = reply[:1996] + "…"

    await ctx.send(reply, allowed_mentions=discord.AllowedMentions.none())


def can_post_colourpanel(ctx: commands.Context) -> bool:
    if not ctx.guild or not isinstance(ctx.author, discord.Member):
        return False
    if has_big_bots_role(ctx.author):
        return True
    perms = ctx.author.guild_permissions
    return perms.manage_roles or perms.administrator


@bot.hybrid_command(name="colourpanel", description="Post the colour-role selection menu")
@commands.guild_only()
@commands.check(can_post_colourpanel)
async def colourpanel(ctx: commands.Context):
    """Post an interactive embed panel allowing members to select their name colour."""
    embed = discord.Embed(
        title="🎨 Choose Your Name Colour",
        description=(
            "Select a colour from the dropdown menu below to change your name colour in this server.\n\n"
            "To remove your colour role, choose **Remove Colour**."
        ),
        color=0x4F46E5,
    )
    view = ColourRolesView()
    await ctx.send(embed=embed, view=view)


@colourpanel.error
async def colourpanel_error(ctx: commands.Context, error: commands.CommandError):
    if isinstance(error, (commands.CheckFailure, commands.MissingPermissions)):
        message = (
            "You need the **Big bots** role, **Manage Roles**, or **Administrator** permission "
            "to post the colour panel."
        )
        if ctx.interaction:
            await ctx.send(message, ephemeral=True)
        else:
            await ctx.send(message)
        return
    log.error("colourpanel command error: %r", error)

@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError):
    if isinstance(error, commands.CommandNotFound):
        return
    if isinstance(error, commands.CheckFailure):
        if ctx.interaction:  # slash: must answer or Discord shows "did not respond"
            cid = config.channel_for(ctx.guild.id) if ctx.guild else None
            msg = f"Use TTS commands in <#{cid}>." if cid else "No channel configured — set one in the web UI."
            await ctx.send(msg, ephemeral=True)
        return
    if isinstance(error, (commands.MemberNotFound, commands.BadArgument, commands.MissingRequiredArgument)):
        return await ctx.send("Mention a valid member.")
    if isinstance(error, commands.NoPrivateMessage):
        return await ctx.send("Server only.")
    if isinstance(error, commands.CommandInvokeError) and isinstance(error.original, discord.Forbidden):
        return await ctx.send("Missing permissions.")
    log.error("command %s failed: %r", ctx.command, error)


# ---------- web UI surface ----------
def _visible_text_channels(guild: discord.Guild) -> list[discord.TextChannel]:
    chans = [c for c in guild.text_channels if c.permissions_for(guild.me).view_channel]
    # uncategorized first, then by category position, then channel position
    chans.sort(key=lambda c: (c.category.position if c.category else -1, c.position, c.id))
    return chans


def _guild_status(guild: discord.Guild) -> dict:
    vc = guild.voice_client
    ga = audio.get(guild.id)
    return {
        "voice": vc.channel.name if vc and vc.is_connected() else None,
        "queue": ga.queue.qsize() if ga else 0,
        "playing": bool(vc and vc.is_playing()),
        "watched": len(store.data.get(guild.id, ())),
    }


def default_voice_paths() -> tuple[str, str]:
    """The env-configured fallback voice (PIPER_MODEL_PATH / PIPER_CONFIG_PATH)."""
    return MODEL_PATH, CONFIG_PATH or MODEL_PATH + ".json"


class Controller:
    """The only surface the web UI touches. Methods are non-blocking; call from the event loop
    (the voice methods are async because they scan disk / reload the engine off-loop)."""

    def __init__(self, limiter: SendLimiter | None = None, clock=None):
        self._voice_lock = asyncio.Lock()  # one swap at a time (engine.reload also serializes, this orders the whole flow)
        self._limiter = limiter or SendLimiter()   # GUI message sends, per guild
        self._chunk_lock = asyncio.Lock()          # one member-cache fill at a time
        self._clock = clock or time.monotonic      # For chunk timeout cache
        self._chunk_failures: dict[int, float] = {}  # guild_id -> timestamp of failure
        self._chunk_timeout_window = 60.0  # Seconds to skip chunk() retry after failure

    @staticmethod
    def _active_engine() -> str:
        return getattr(engine, "engine_name", "piper") if engine.ready else config.engine

    def voice_summary(self) -> dict:
        return {"current": engine.model_name if engine.ready else None,
                "configured": config.voice, "loading": engine.loading,
                "idle": bool(getattr(engine, "idle", False)),
                "cuda": engine.cuda_active if engine.ready else None,
                "engine": self._active_engine(),
                "configured_engine": config.engine}

    def _engines_view(self) -> list[dict]:
        out = [{"name": "piper", "available": True, "reason": None}]
        for name in engines.ENGINE_NAMES:
            if name == "piper":
                continue
            try:
                ok, why = engines.get(name).available(MODELS_DIR)
            except Exception:
                ok, why = False, "engine unavailable"
            out.append({"name": name, "available": ok, "reason": why})
        return out

    async def list_voices(self, engine_name: str | None = None) -> dict:
        """Rescans the models folder on every call (this is the GUI's Rescan)."""
        name = engine_name or self._active_engine()
        if not engines.valid_engine(name):
            raise BadVoiceError("unknown engine")
        if name == "piper":
            voices = await asyncio.to_thread(discover_voices, MODELS_DIR)
        else:
            voices = await asyncio.to_thread(engines.get(name).list_voices, MODELS_DIR)
        return {"voices": voices, "list_engine": name, "engines": self._engines_view(), **self.voice_summary()}

    def _publish_voice(self, loading: bool, error: str | None = None) -> None:
        ev = {"type": "voice", "loading": loading}
        if not loading:
            ev.update(current=engine.model_name if engine.ready else None, error=error,
                     cuda=engine.cuda_active if engine.ready else None)
            if hasattr(engine, "engine_name"):
                ev["engine"] = engine.engine_name if engine.ready else None
        config.hub.publish(ev)

    async def _reload_to(self, target_engine: str, name: str | None, target) -> None:
        if target_engine == "piper":
            await engine.reload(*target)
        else:
            await engine.reload_engine(target_engine, MODELS_DIR, name)

    async def apply_voice(self, name: str | None, expected_rev: int | None, editor: str,
                          engine_name: str | None = None) -> dict:
        """Hot-swap the global engine+voice. Raises BadVoiceError, ConflictError, VoiceLoadError, OSError."""
        async with self._voice_lock:
            target_engine = engine_name or self._active_engine()
            if not engines.valid_engine(target_engine):
                raise BadVoiceError("unknown engine")
            target = None
            if target_engine == "piper":
                if name is None:
                    target = default_voice_paths()
                else:
                    target = await asyncio.to_thread(resolve_voice, MODELS_DIR, name)  # BadVoiceError
            else:
                if name is None:
                    raise BadVoiceError("voice required")
                await asyncio.to_thread(engines.get(target_engine).resolve, MODELS_DIR, name)  # BadVoiceError
            # Stale-client detection happens HERE, before the engine is touched. The save below uses
            # expected_rev=None on purpose: the engine has already swapped, so a rev bump from a
            # concurrent channel/pause edit during the (slow) load must not discard the swap.
            if expected_rev is not None and expected_rev != config.rev:
                raise ConflictError(config.snapshot())
            prev = None
            if engine.ready:
                pe = getattr(engine, "engine_name", "piper")
                prev = (pe, engine.model_name if pe != "piper" else None, (engine.model_path, engine.config_path))
            self._publish_voice(True)
            err = None
            try:
                await self._reload_to(target_engine, name, target)
            except VoiceLoadError:
                err = "load_failed"
                raise
            finally:
                self._publish_voice(False, err)
            try:
                return await config.set_voice(name, expected_rev=None, editor=editor, engine=target_engine)
            except OSError:
                if prev is not None:  # best effort: keep engine and saved config in agreement
                    self._publish_voice(True)
                    err = None
                    try:
                        await self._reload_to(prev[0], prev[1], prev[2])
                    except VoiceLoadError:
                        err = "load_failed"
                        log.error("could not restore previous voice after a failed save")
                    finally:
                        self._publish_voice(False, err)
                raise

    def say(self, guild_id: int, text: str, editor: str = "") -> dict:
        """Queue a test phrase on this guild's normal audio path. Raises SayError."""
        cleaned = clean_plain_text(text, SAY_MAX_CHARS)
        if not cleaned:
            raise SayError("bad_text")
        guild = bot.get_guild(guild_id)
        vc = guild.voice_client if guild else None
        ga = audio.get(guild_id)
        if not (vc and vc.is_connected() and ga and ga.worker and not ga.worker.done()):
            raise SayError("not_connected")
        if not engine.ready:
            raise SayError("tts_unavailable")
        try:
            ga.queue.put_nowait(cleaned)
        except asyncio.QueueFull:
            raise SayError("queue_full") from None
        log.info("gui test phrase queued guild=%s editor=%s chars=%d", guild_id, editor, len(cleaned))
        return {"queued": True}

    # --- messages (Messages tab) ---
    def _message_channel(self, guild_id: int, channel_id: int | None = None):
        """(guild, TextChannel) or MessageError('no_channel'). Uses channel_id if given, else configured channel."""
        guild = bot.get_guild(guild_id)
        if guild is None:
            raise MessageError("no_channel")
        cid = channel_id if channel_id is not None else config.channel_for(guild_id)
        ch = guild.get_channel(cid) if cid else None
        if not isinstance(ch, discord.TextChannel):
            raise MessageError("no_channel")
        return guild, ch

    async def members_for(self, guild_id: int, q: str = "", channel_id: int | None = None) -> dict:
        """Members who can view the target channel, for the tag picker (max 25).
        Caps permission checks at 2000 per request; yields to the event loop every ~250 members."""
        guild, ch = self._message_channel(guild_id, channel_id)
        if guild.me is None:
            # guild.me can be None during outages; return empty result
            return {"members": [], "truncated": False}
        if not guild.chunked:
            async with self._chunk_lock:
                if not guild.chunked:
                    # Skip retry if chunk failed recently (within timeout window)
                    now = self._clock()
                    failed_at = self._chunk_failures.get(guild_id)
                    if failed_at is None or now - failed_at >= self._chunk_timeout_window:
                        try:
                            await asyncio.wait_for(guild.chunk(), timeout=10)
                        except Exception as e:  # timeout / gateway error: serve what is cached
                            log.warning("member chunk failed for guild %s (%s); using cache", guild_id, type(e).__name__)
                            self._chunk_failures[guild_id] = now
        needle = (q or "").strip().casefold()

        def names(m):
            return (m.display_name, m.name, getattr(m, "global_name", None))

        cands = [m for m in guild.members
                 if not needle or any(n and needle in n.casefold() for n in names(m))]
        cands.sort(key=lambda m: (bool(m.bot), m.display_name.casefold(), m.id))
        out, truncated, perm_checks = [], False, 0
        for idx, m in enumerate(cands):
            if idx > 0 and idx % 250 == 0:
                await asyncio.sleep(0)
            if perm_checks >= MAX_PERM_CHECKS:
                truncated = True
                break
            if not ch.permissions_for(m).view_channel:
                perm_checks += 1
                continue
            perm_checks += 1
            if len(out) == 25:
                truncated = True
                break
            out.append({"id": str(m.id), "name": m.display_name, "display_name": m.display_name, "username": m.name, "bot": bool(m.bot)})
        return {"members": out, "truncated": truncated}

    def roles_for(self, guild_id: int, q: str = "", channel_id: int | None = None) -> dict:
        """Guild roles for the tag picker (max 25, non-default)."""
        guild, ch = self._message_channel(guild_id, channel_id)
        needle = (q or "").strip().casefold()
        roles = []
        for r in getattr(guild, "roles", []):
            is_default = getattr(r, "is_default", None)
            if callable(is_default):
                if is_default():
                    continue
            elif getattr(r, "id", None) == guild.id:
                continue
            r_name = getattr(r, "name", "")
            if needle and needle not in r_name.casefold():
                continue
            roles.append(r)
        roles.sort(key=lambda r: (-getattr(r, "position", 0), getattr(r, "name", "").casefold()))
        out = []
        for r in roles[:25]:
            color = getattr(r, "color", None)
            color_val = getattr(color, "value", 0) if color else 0
            out.append({
                "id": str(r.id),
                "name": r.name,
                "color": f"#{color_val:06x}" if color_val else None,
                "is_role": True,
            })
        return {"roles": out, "truncated": len(roles) > 25}

    async def send_message(self, guild_id: int, content: str, editor: str, channel_id: int | None = None) -> dict:
        """Post `content` as the bot in the guild's target channel. Raises MessageError."""
        content = content.strip() if isinstance(content, str) else ""
        if not 1 <= len(content) <= MAX_MESSAGE_CHARS:
            raise MessageError("bad_request")
        guild, ch = self._message_channel(guild_id, channel_id)
        user_ids, role_ids = parse_mentions(content)
        for uid in user_ids:
            member = guild.get_member(uid)
            if member is None or not ch.permissions_for(member).view_channel:
                raise MessageError("bad_mention")
        for rid in role_ids:
            role = guild.get_role(rid)
            is_default = getattr(role, "is_default", None)
            is_def = is_default() if callable(is_default) else (role and getattr(role, "id", None) == guild.id)
            if role is None or is_def:
                raise MessageError("bad_mention")
        if guild.me is None:
            raise MessageError("no_channel")  # Outage: can't determine permissions
        if not ch.permissions_for(guild.me).send_messages:
            raise MessageError("missing_permission")
        allowed = discord.AllowedMentions(
            everyone=False,
            roles=[discord.Object(id=i) for i in role_ids] if role_ids else False,
            users=[discord.Object(id=i) for i in user_ids],
            replied_user=False,
        )
        self._limiter.reserve(guild_id)  # bookkeeping BEFORE the await (cancel/duplicate safe)
        try:
            msg = await ch.send(content, allowed_mentions=allowed)
        except discord.Forbidden:
            log.warning("gui message failed guild=%s channel=%s editor=%s reason=forbidden", guild_id, ch.id, editor)
            raise MessageError("missing_permission") from None
        except discord.HTTPException as e:
            log.warning("gui message failed guild=%s channel=%s editor=%s reason=http_error status=%s",
                        guild_id, ch.id, editor, getattr(e, "status", "?"))
            raise MessageError("discord_error") from None
        log.info("gui message sent guild=%s channel=%s editor=%s chars=%d mentions=%d",
                 guild_id, ch.id, editor, len(content), len(user_ids) + len(role_ids))
        return {"ok": True, "message_id": str(msg.id)}

    def guilds_view(self) -> list[dict]:
        return [{
            "id": str(g.id),
            "name": g.name,
            "channels": [{"id": str(c.id), "name": c.name,
                          "category": c.category.name if c.category else None,
                          "position": c.position} for c in _visible_text_channels(g)],
            **_guild_status(g),
        } for g in bot.guilds]

    def status_view(self) -> dict:
        return {str(g.id): _guild_status(g) for g in bot.guilds}

    def is_text_channel(self, guild_id: int, channel_id: int) -> bool:
        guild = bot.get_guild(guild_id)
        if guild is None:
            return False
        ch = guild.get_channel(channel_id)
        return isinstance(ch, discord.TextChannel) and ch.permissions_for(guild.me).view_channel

    def skip(self, guild_id: int) -> bool:
        guild = bot.get_guild(guild_id)
        vc = guild.voice_client if guild else None
        if vc and vc.is_playing():
            vc.stop()
            return True
        return False

    def clear_queue(self, guild_id: int) -> int:
        ga = audio.get(guild_id)
        if not ga:
            return 0
        n = ga.queue.qsize()
        drain(ga.queue)
        return n


controller = Controller()


async def nag_loop() -> None:
    """Easter egg. Needs NAG_USER_IDS (two user ids) and NAG_CHANNEL_ID; otherwise does nothing.
    Posts at most once per stale week, mentioning only those users."""
    ids = nag.parse_ids(os.getenv("NAG_USER_IDS"))
    ch_raw = os.getenv("NAG_CHANNEL_ID", "")
    if not ids or not (ch_raw.isascii() and ch_raw.isdigit()):
        return
    channel_id = int(ch_raw)
    state_path = os.path.join(os.path.dirname(STATE_FILE), "nag.json")
    app_dir = os.path.dirname(os.path.abspath(__file__))
    await bot.wait_until_ready()
    while True:
        try:
            now = time.time()
            newest = await asyncio.to_thread(nag.newest_mtime, app_dir)
            last = await asyncio.to_thread(nag.load_last, state_path)
            if nag.due(newest, last, now):
                ch = bot.get_channel(channel_id)
                if isinstance(ch, discord.TextChannel) and ch.permissions_for(ch.guild.me).send_messages:
                    days = int((now - newest) // 86400)
                    await ch.send(nag.build_message(ids, days),
                                  allowed_mentions=discord.AllowedMentions(
                                      users=[discord.Object(id=i) for i in ids], roles=False, everyone=False))
                    await asyncio.to_thread(atomic_write_json, {"last_nag": now}, state_path)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("nag loop failed")
        await asyncio.sleep(3600)


async def config_watcher() -> None:
    """On a listen-channel change, drop that guild's pending clips (the playing one finishes)."""
    q = config.hub.subscribe()
    try:
        while True:
            event = await q.get()
            try:
                if event.get("type") != "config":
                    continue
                change = event["change"]
                if change["field"] == "channel_id":
                    ga = audio.get(int(change["guild_id"]))
                    if ga:
                        drain(ga.queue)
            except Exception:
                log.exception("config watcher error (continuing)")
    finally:
        config.hub.unsubscribe(q)


def _publish_channels(guild: discord.Guild) -> None:
    config.hub.publish({"type": "channels", "guild_id": str(guild.id)})


@bot.event
async def on_guild_channel_create(channel):
    _publish_channels(channel.guild)


@bot.event
async def on_guild_channel_delete(channel):  # configured channel is left as is; UI shows it missing
    _publish_channels(channel.guild)


@bot.event
async def on_guild_channel_update(before, after):  # renames, moves
    _publish_channels(after.guild)


@bot.event
async def on_guild_join(guild):
    _publish_channels(guild)


@bot.event
async def on_guild_remove(guild):
    _publish_channels(guild)


@bot.event
async def on_guild_available(guild):  # guild (re)appeared after an outage or late cache fill
    _publish_channels(guild)


# ---------- lifecycle ----------
async def maybe_start_webui(cfg, ctl, port: int):
    """Start the web UI; return its runner, or None if disabled/failed. Never raises.

    Addresses: WEBUI_HOSTS (validated: loopback / Tailscale only), else 127.0.0.1 plus any detected
    Tailscale address. Host allowlist: loopback names + those addresses + WEBUI_ALLOWED_HOSTS."""
    try:
        raw = (os.getenv("WEBUI_HOSTS") or "").strip()
        auto = not raw
        if raw:
            hosts = validate_bind_hosts([h for h in raw.split(",") if h.strip()])
        else:
            hosts = validate_bind_hosts(["127.0.0.1"] + await asyncio.to_thread(detect_tailscale_addrs))
        allowed = hosts + parse_allowed_hosts(os.getenv("WEBUI_ALLOWED_HOSTS"))
        users_file = USERS_FILE  # fail closed: login is required; no file or no users = nobody gets in
        if (os.getenv("WEBUI_AUTH") or "").lower() == "off":
            users_file = None
            log.warning("web UI authentication is DISABLED (WEBUI_AUTH=off)")
        runner = await start_webui(make_app(cfg, ctl, allowed_hosts=allowed, users_file=users_file, features_file=FEATURES_FILE), hosts, port,
                                    detect=detect_tailscale_addrs if auto else None)
    except ValueError as e:
        log.error("Web UI disabled: %s", e)
        return None
    except Exception:
        log.exception("Web UI failed to start on port %s; continuing without it", port)
        return None
    log.info("Web UI addresses: %s port %s", ", ".join(hosts), port)
    return runner


async def shutdown(b: commands.Bot) -> None:
    global _shutting_down, _initial_ready
    if _shutting_down:
        return
    _shutting_down = True
    _initial_ready = False
    log.info("Shutting down")
    if _watcher_task:
        _watcher_task.cancel()
    if _idle_task:
        _idle_task.cancel()
    if _nag_task:
        _nag_task.cancel()
    for guild in list(b.guilds):
        if guild.voice_client or guild.id in audio:
            await teardown(guild)
    if _webui_runner:
        try:
            await _webui_runner.cleanup()
        except Exception:
            log.exception("web UI cleanup failed")
    engine.close()
    await b.close()


async def load_engine() -> bool:
    """Load the saved GUI engine+voice if it still resolves (single load), else the env Piper default.
    A saved voice that is missing, unavailable or fails to load is logged once and the default is used;
    the GUI then shows configured != current."""
    saved = config.voice
    if config.engine != "piper" and saved:
        mod = engines.get(config.engine)
        try:
            ok, why = await asyncio.to_thread(mod.available, MODELS_DIR)
            if not ok:
                log.warning("saved engine %r unavailable (%s); using Piper", config.engine, why)
            else:
                await asyncio.to_thread(mod.resolve, MODELS_DIR, saved)
                await engine.reload_engine(config.engine, MODELS_DIR, saved)
                return True
        except BadVoiceError:
            log.warning("saved voice %r not found for engine %r; using Piper", saved, config.engine)
        except VoiceLoadError:
            log.error("saved %s voice %r failed to load; using Piper", config.engine, saved)
        except Exception:
            log.exception("saved engine failed; using Piper")
        saved = None
    if saved:
        try:
            onnx, cfg = await asyncio.to_thread(resolve_voice, MODELS_DIR, saved)
        except BadVoiceError:
            log.warning("saved voice %r not found in the models folder; using the default voice", saved)
        else:
            if await asyncio.to_thread(engine.load, onnx, cfg):
                return True
            log.error("saved voice %r failed to load; falling back to the default voice", saved)
    model, cfg = default_voice_paths()
    return await asyncio.to_thread(engine.load, model, cfg)


async def main() -> None:
    if not TOKEN:
        raise SystemExit("No token: set ~/.config/tts-bot/token (see README)")
    engine.purge_tmp()
    if not await load_engine():
        log.error("TTS engine failed to load; bot runs but cannot speak")
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(bot)))
    global _watcher_task
    _watcher_task = asyncio.create_task(config_watcher(), name="config-watcher")
    global _idle_task, _nag_task
    _idle_task = asyncio.create_task(engine.idle_unload_loop(), name="idle-unload")
    _nag_task = asyncio.create_task(nag_loop(), name="nag")
    global _webui_runner
    _webui_runner = await maybe_start_webui(config, controller, WEBUI_PORT)
    async with bot:
        await bot.start(TOKEN)


if __name__ == "__main__":
    asyncio.run(main())
