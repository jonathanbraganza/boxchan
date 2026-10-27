"""Piper TTS wrapper (ONNX Runtime, CUDA when available, FP32) + Discord text cleaner."""
import asyncio
import gc
import logging
import os
import re
import tempfile
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import discord

import engines
from voices import VoiceLoadError

log = logging.getLogger("tts_engine")

try:
    IDLE_UNLOAD_SECONDS = float(os.getenv("IDLE_UNLOAD_SECONDS", "300"))  # 0 disables
except ValueError:
    IDLE_UNLOAD_SECONDS = 300.0

RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
RE_CUSTOM_EMOJI = re.compile(r"<a?:([a-zA-Z0-9_]+):\d+>")
RE_USER_MENTION = re.compile(r"<@!?(\d+)>")
RE_ROLE_MENTION = re.compile(r"<@&(\d+)>")
RE_CHANNEL_MENTION = re.compile(r"<#(\d+)>")
RE_TIMESTAMP = re.compile(r"<t:-?\d+(?::[a-zA-Z])?>")
RE_CODE_BLOCK = re.compile(r"```(?:[a-zA-Z0-9_+-]*\n)?(.*?)```", re.DOTALL)
RE_INLINE_CODE = re.compile(r"`([^`]+)`")
RE_SPOILER = re.compile(r"\|\|(.*?)\|\|", re.DOTALL)
RE_MD_CHARS = re.compile(r"[*_~>#`|]")
RE_EXCESSIVE_PUNCT = re.compile(r"([!?,.:;~])\1{2,}")
RE_REPEAT_LETTER = re.compile(r"([^\W\d_])\1{3,}")
RE_SPACES = re.compile(r"\s+")


@dataclass(frozen=True)
class VoiceState:
    """Immutable snapshot of the loaded Piper voice state. Assigned atomically to prevent event-loop
    readers from seeing a half-swapped state (e.g. new voice with old model_path)."""
    voice: object  # PiperVoice
    sample_rate: int
    model_path: Path
    config_path: Path
    cuda_active: bool
    engine: str = "piper"       # active backend
    name: str | None = None     # non-Piper voice name (Piper uses model_path.stem)


def _normalize(text: str, max_chars: int, user, role, channel) -> str | None:
    """Shared speech-cleaning pipeline; `user`/`role`/`channel` resolve mention regex matches."""
    text = RE_URL.sub("link", text)
    text = RE_CODE_BLOCK.sub("code block", text)
    text = RE_INLINE_CODE.sub(r"\1", text)
    text = RE_SPOILER.sub(r"\1", text)  # preference: read spoilers, drop the pipes
    text = RE_USER_MENTION.sub(user, text)
    text = RE_ROLE_MENTION.sub(role, text)
    text = RE_CHANNEL_MENTION.sub(channel, text)
    text = RE_CUSTOM_EMOJI.sub(r"\1", text)
    text = RE_TIMESTAMP.sub("timestamp", text)
    text = RE_MD_CHARS.sub(" ", text)
    text = RE_EXCESSIVE_PUNCT.sub(r"\1", text)
    text = RE_REPEAT_LETTER.sub(r"\1\1\1", text)
    text = RE_SPACES.sub(" ", text).strip()

    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "..."
    if not any(c.isalnum() for c in text):
        return None
    return text


def clean_discord_text(message: discord.Message, max_chars: int = 300) -> str | None:
    """Normalize a Discord message for speech. None if nothing speakable remains."""
    text = message.content
    guild = message.guild
    if not text or not text.strip():
        return None

    def user(m):
        member = guild.get_member(int(m.group(1))) if guild else None
        return f"at {member.display_name}" if member else "someone"

    def role(m):
        r = guild.get_role(int(m.group(1))) if guild else None
        return f"at {r.name}" if r else "role"

    def channel(m):
        c = guild.get_channel(int(m.group(1))) if guild else None
        return f"channel {c.name}" if c else "channel"

    return _normalize(text, max_chars, user, role, channel)


def clean_plain_text(text: str, max_chars: int = 300) -> str | None:
    """Same pipeline for plain text (no Discord objects): mentions become generic words."""
    if not isinstance(text, str) or not text.strip():
        return None
    return _normalize(text, max_chars, lambda m: "someone", lambda m: "role", lambda m: "channel")


class TTSEngine:
    """Single GPU context: one worker thread + asyncio.Lock serialize all inference."""

    def __init__(self, model_path: str = "/models/default.onnx", config_path: str | None = None):
        mp = Path(model_path) if model_path else Path("/models/default.onnx")
        cp = Path(config_path) if config_path else Path(str(mp) + ".json")
        self._state = VoiceState(voice=None, sample_rate=22050, model_path=mp,
                                config_path=cp,
                                cuda_active=False)
        self.ready = False
        self._pending_reloads = 0
        self._idle = False          # voice unloaded after idle; next synthesis reloads it
        self._last_use = time.monotonic()
        self._models_dir: str | None = None
        self.tmp_dir = Path(tempfile.gettempdir()) / "tts_bot"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self._voice_cache: dict[str, VoiceState] = {}
        self.MAX_CACHED_VOICES = 3
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="piper")
        self._lock: asyncio.Lock | None = None

    @property
    def state(self) -> VoiceState:
        """One consistent snapshot; read it once instead of several properties when fields must agree."""
        return self._state

    # Backwards-compatible properties for existing code and tests
    @property
    def voice(self):
        return self._state.voice

    @property
    def sample_rate(self):
        return self._state.sample_rate

    @property
    def model_path(self):
        return self._state.model_path

    @property
    def config_path(self):
        return self._state.config_path

    @property
    def cuda_active(self):
        return self._state.cuda_active

    @property
    def loading(self) -> bool:
        """True while a reload() is queued or running."""
        return self._pending_reloads > 0

    @property
    def model_name(self) -> str:
        """Stem of the loaded (or configured) model file; the voice name for non-Piper engines."""
        st = self._state
        return st.name if st.engine != "piper" and st.name else st.model_path.stem

    @property
    def idle(self) -> bool:
        """True while the voice is unloaded to save power; the next synthesis reloads it."""
        return self._idle

    @property
    def engine_name(self) -> str:
        return self._state.engine

    def purge_tmp(self) -> None:
        for f in self.tmp_dir.glob("*.wav"):
            try:
                f.unlink()
            except OSError:
                pass

    @staticmethod
    def _backend():
        """(available onnxruntime providers, PiperVoice). Isolated so tests can fake it."""
        import onnxruntime as ort
        from piper import PiperVoice
        return ort.get_available_providers(), PiperVoice

    def _build(self, model_path: Path, config_path: Path):
        """Blocking; worker thread (or load()'s caller). Build and warm up a PiperVoice: CUDA when
        available, CPU fallback. Returns (voice, cuda_used); raises if no provider works."""
        providers, PiperVoice = self._backend()
        voice = None
        cuda = False
        if "CUDAExecutionProvider" in providers:
            try:
                voice = PiperVoice.load(str(model_path), config_path=str(config_path), use_cuda=True)
                self._synth_with(voice, "warm up")
                cuda = True
            except Exception:
                log.exception("CUDA init failed; falling back to CPU")
                voice = None
        else:
            log.error("CUDAExecutionProvider unavailable (%s); using CPU", providers)
        if voice is None:
            voice = PiperVoice.load(str(model_path), config_path=str(config_path), use_cuda=False)
            self._synth_with(voice, "warm up")
        return voice, cuda

    def load(self, model_path=None, config_path=None) -> bool:
        """Never raises. False (logged) if model cannot load. Optional paths replace the configured ones."""
        target_model = Path(model_path) if model_path is not None else self.model_path
        target_config = (Path(config_path) if config_path else Path(str(target_model) + ".json")) if model_path is not None else self.config_path
        for p in (target_model, target_config):
            if not p.is_file():
                log.error("Model file missing: %s", p)
                return False
        try:
            voice, cuda = self._build(target_model, target_config)
        except Exception:
            log.exception("model load failed")
            return False
        rate = voice.config.sample_rate
        self._state = VoiceState(voice=voice, sample_rate=rate, model_path=target_model,
                               config_path=target_config, cuda_active=cuda)
        self.ready = True
        log.info("Piper ready (model=%s, cuda=%s, rate=%d)", self.model_name, cuda, rate)
        return True

    @staticmethod
    def _synth_with(voice, text: str, length_scale: float = 1.0) -> bytes:
        if length_scale == 1.0:
            try:
                return b"".join(voice.synthesize_stream_raw(text))
            except TypeError:
                pass
        try:
            return b"".join(voice.synthesize_stream_raw(text, length_scale=length_scale))
        except TypeError:
            return b"".join(voice.synthesize_stream_raw(text))

    def _unload_blocking(self) -> None:
        """Worker thread. Drop the loaded voice (frees GPU memory); paths/name stay in the state."""
        if not self.ready or self._idle or self._state.voice is None:
            return
        old = self._state.voice
        self._state = replace(self._state, voice=None)
        self._idle = True
        del old
        gc.collect()
        log.info("Voice unloaded after idle (%s)", self.model_name)

    def _resume_blocking(self) -> None:
        """Worker thread. Rebuild the voice that _unload_blocking dropped. Raises on failure."""
        st = self._state
        if st.engine == "piper":
            voice, cuda = self._build(st.model_path, st.config_path)
        else:
            voice, _rate, cuda = engines.get(st.engine).build(self._models_dir, st.name)
        self._state = replace(st, voice=voice, cuda_active=cuda)
        self._idle = False
        log.info("Voice reloaded after idle (%s, cuda=%s)", self.model_name, cuda)

    def _get_voice_for_synth(self, voice_name: str | None) -> tuple[VoiceState, bool]:
        """Worker thread. (VoiceState, is_secondary). Falls back to self._state on failure."""
        if not voice_name or voice_name == self.model_name:
            return self._state, False
        if voice_name in self._voice_cache:
            return self._voice_cache[voice_name], True

        st = self._state
        if st.engine == "piper":
            models_dir = Path(self._models_dir) if self._models_dir else (st.model_path.parent if st.model_path else Path("models"))
            target_model = models_dir / f"{voice_name}.onnx"
            target_config = models_dir / f"{voice_name}.onnx.json"
            if not (target_model.is_file() and target_config.is_file()):
                log.warning("Secondary voice %s not found; using active voice %s", voice_name, self.model_name)
                return self._state, False
            try:
                voice, cuda = self._build(target_model, target_config)
                rate = getattr(getattr(voice, "config", None), "sample_rate", self.sample_rate)
                cached = VoiceState(voice=voice, sample_rate=rate, model_path=target_model,
                                    config_path=target_config, cuda_active=cuda, engine="piper", name=voice_name)
                if len(self._voice_cache) >= self.MAX_CACHED_VOICES:
                    evict_k, evict_v = next(iter(self._voice_cache.items()))
                    del self._voice_cache[evict_k]
                    del evict_v
                    gc.collect()
                self._voice_cache[voice_name] = cached
                return cached, False
            except Exception:
                log.exception("Failed to load secondary voice %s; using active voice", voice_name)
                return self._state, False
        elif st.engine == "kokoro":
            if st.voice is not None and hasattr(st.voice, "kokoro"):
                from kokoro_engine import Handle, lang_for
                try:
                    mod = engines.get("kokoro")
                    mod.resolve(self._models_dir or "models", voice_name)
                    k_handle = Handle(st.voice.kokoro, voice_name, lang_for(voice_name), st.voice.model_path)
                    cached = VoiceState(voice=k_handle, sample_rate=24000, model_path=st.model_path,
                                        config_path=st.config_path, cuda_active=st.cuda_active,
                                        engine="kokoro", name=voice_name)
                    if len(self._voice_cache) >= self.MAX_CACHED_VOICES:
                        evict_k, evict_v = next(iter(self._voice_cache.items()))
                        del self._voice_cache[evict_k]
                        del evict_v
                        gc.collect()
                    self._voice_cache[voice_name] = cached
                    return cached, False
                except Exception:
                    log.warning("Kokoro voice %s invalid; using active voice", voice_name)
                    return self._state, False
        return self._state, False

    async def idle_unload_loop(self, timeout: float | None = None) -> None:
        """Unload the voice after `timeout` seconds without synthesis or reload. Runs forever."""
        timeout = IDLE_UNLOAD_SECONDS if timeout is None else timeout
        if timeout <= 0:
            return
        while True:
            await asyncio.sleep(min(30.0, timeout))
            if (not self.ready or self._idle or self._pending_reloads
                    or time.monotonic() - self._last_use < timeout):
                continue
            if self._lock is None:
                self._lock = asyncio.Lock()
            if self._lock.locked():
                continue
            async with self._lock:
                if time.monotonic() - self._last_use >= timeout:
                    try:
                        await asyncio.get_running_loop().run_in_executor(self._pool, self._unload_blocking)
                    except Exception:
                        log.exception("idle unload failed")

    def _synth(self, text: str, length_scale: float = 1.0, voice_state: VoiceState | None = None) -> bytes:
        if self._idle:
            self._resume_blocking()
        st = voice_state or self._state
        if st.engine != "piper":
            mod = engines.get(st.engine)
            speed = 1.0 / length_scale if length_scale > 0 else 1.0
            try:
                return mod.synth(st.voice, text, speed=speed)
            except TypeError:
                return mod.synth(st.voice, text)
        if length_scale == 1.0:
            try:
                return self._synth_with(st.voice, text)
            except TypeError:
                pass
        try:
            return self._synth_with(st.voice, text, length_scale=length_scale)
        except TypeError:
            return self._synth_with(st.voice, text)

    def _reload_engine_blocking(self, engine_name: str, models_dir: str, name: str) -> None:
        """Worker thread. Build + warm up a non-Piper candidate; swap only on success."""
        mod = engines.get(engine_name)
        try:
            handle, rate, cuda = mod.build(models_dir, name)
        except VoiceLoadError:
            raise
        except Exception as e:
            log.exception("%s load failed; keeping the current voice", engine_name)
            raise VoiceLoadError() from e
        if self.ready and self.cuda_active and not cuda:
            log.warning("refusing swap: currently using CUDA but candidate would run on CPU")
            raise VoiceLoadError("GPU unavailable for this voice")
        old = self._state.voice
        self._state = VoiceState(voice=handle, sample_rate=rate, model_path=Path(handle.model_path),
                                 config_path=Path(handle.model_path), cuda_active=cuda,
                                 engine=engine_name, name=name)
        self.ready = True
        self._idle = False
        self._models_dir = models_dir
        self._last_use = time.monotonic()
        del old
        self._voice_cache.clear()
        gc.collect()
        log.info("%s voice swapped (voice=%s, cuda=%s, rate=%d)", engine_name, name, cuda, rate)

    async def reload_engine(self, engine_name: str, models_dir: str, name: str) -> None:
        """Hot-swap to a non-Piper engine voice. Same serialization and failure contract as reload()."""
        await self._run_reload(self._reload_engine_blocking, engine_name, str(models_dir), name)

    def _reload_blocking(self, model_path: Path, config_path: Path) -> None:
        """Worker thread. Build + warm up the candidate; swap only on success. Refuses GPU->CPU
        downgrade if the current voice is already on GPU."""
        for p in (model_path, config_path):
            if not p.is_file():
                log.error("Model file missing: %s", p)
                raise VoiceLoadError("voice files not found")
        try:
            voice, cuda = self._build(model_path, config_path)
            rate = voice.config.sample_rate
        except VoiceLoadError:
            raise
        except Exception as e:
            log.exception("voice reload failed; keeping the current voice")
            raise VoiceLoadError() from e
        # Check for silent GPU->CPU downgrade: if current voice is on CUDA and candidate would fall
        # back to CPU, refuse the swap to avoid degraded performance without user awareness.
        if self.cuda_active and not cuda:
            log.warning("refusing voice swap: currently using CUDA but candidate would run on CPU")
            raise VoiceLoadError("GPU unavailable for this voice")
        old = self._state.voice
        # Single step on the only inference thread: no synthesis can observe a half-swapped engine.
        self._state = VoiceState(voice=voice, sample_rate=rate, model_path=model_path,
                                config_path=config_path, cuda_active=cuda)
        self.ready = True
        self._idle = False
        self._last_use = time.monotonic()
        del old
        self._voice_cache.clear()
        gc.collect()  # drop the old ORT session now so its (GPU) memory is released
        log.info("Piper voice swapped (model=%s, cuda=%s, rate=%d)", self.model_name, cuda, rate)

    async def reload(self, model_path, config_path) -> None:
        """Hot-swap to a Piper voice. Serialized with synthesis (same lock, same single worker thread).
        On any failure raises VoiceLoadError and the previous voice keeps serving."""
        await self._run_reload(self._reload_blocking, Path(model_path), Path(config_path))

    async def _run_reload(self, blocking, *args) -> None:
        if self._lock is None:
            self._lock = asyncio.Lock()
        self._pending_reloads += 1
        fut = None
        try:
            async with self._lock:
                fut = asyncio.get_running_loop().run_in_executor(
                    self._pool, blocking, *args)
                fut.add_done_callback(self._reload_done)
                # shield: if our caller is cancelled the job still finishes (and is counted until then)
                await asyncio.shield(fut)
        except (VoiceLoadError, asyncio.CancelledError):
            raise
        except Exception as e:  # e.g. pool already shut down
            log.exception("voice reload could not run")
            raise VoiceLoadError() from e
        finally:
            if fut is None:
                self._pending_reloads -= 1

    def _reload_done(self, fut) -> None:
        self._pending_reloads -= 1
        if not fut.cancelled():
            fut.exception()  # mark retrieved when nobody awaits it any more

    def _synth_to_wav(self, text: str, speed: float = 1.0, voice_name: str | None = None) -> str | None:
        try:
            target_st, _ = self._get_voice_for_synth(voice_name)
            clamped_speed = max(0.5, min(2.0, float(speed))) if speed else 1.0
            length_scale = 1.0 / clamped_speed
            pcm = self._synth(text, length_scale=length_scale, voice_state=target_st)
            if not pcm:
                return None
            with tempfile.NamedTemporaryFile(dir=self.tmp_dir, suffix=".wav", delete=False) as tf:
                path = tf.name
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(target_st.sample_rate)
                w.writeframes(pcm)
            return path
        except Exception:
            log.exception("synthesis failed")
            return None

    async def synthesize(self, text: str, speed: float = 1.0, voice_name: str | None = None) -> str | None:
        """Off-loop, strictly serialized. Returns temp .wav path (caller deletes) or None."""
        if not self.ready or not text.strip():
            return None
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            try:
                return await asyncio.get_running_loop().run_in_executor(
                    self._pool, self._synth_to_wav, text, speed, voice_name)
            finally:
                self._last_use = time.monotonic()

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._voice_cache.clear()
        self.purge_tmp()
