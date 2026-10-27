"""Kokoro-82M (kokoro-onnx) backend. Optional: a missing package or model only makes it unavailable.

Files live in `<models_dir>/kokoro/`: `kokoro.onnx` and `voices.bin`. Called only from the TTSEngine
worker thread (build/synth) or from `asyncio.to_thread` (available/list/resolve)."""
import logging
import os
import re
from dataclasses import dataclass

from voices import BadVoiceError, VoiceLoadError

log = logging.getLogger("kokoro")

SUBDIR = "kokoro"
MODEL_FILE = "kokoro.onnx"
VOICES_FILE = "voices.bin"
SAMPLE_RATE = 24000
NAME_RE = re.compile(r"[a-z]{2}_[a-z0-9]{1,30}")


@dataclass(frozen=True)
class Handle:
    kokoro: object
    voice: str
    lang: str
    model_path: str


def _paths(models_dir):
    base = os.path.join(os.fspath(models_dir), SUBDIR)
    return os.path.join(base, MODEL_FILE), os.path.join(base, VOICES_FILE)


def available(models_dir) -> tuple[bool, str | None]:
    try:
        import kokoro_onnx  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        return False, "kokoro-onnx not installed"
    model, voices = _paths(models_dir)
    if not (os.path.isfile(model) and os.path.isfile(voices)):
        return False, "model files missing"
    return True, None


def _voice_names(voices_path) -> list[str]:
    import numpy as np
    with np.load(voices_path, allow_pickle=False) as z:
        return sorted(k for k in z.files if NAME_RE.fullmatch(k))


def lang_for(voice: str) -> str:
    return "en-gb" if voice[:1] == "b" else "en-us"


def list_voices(models_dir) -> list[dict]:
    ok, _ = available(models_dir)
    if not ok:
        return []
    try:
        names = _voice_names(_paths(models_dir)[1])
    except Exception:
        log.exception("could not read kokoro voices")
        return []
    return [{"name": n, "sample_rate": SAMPLE_RATE, "quality": None, "language": lang_for(n)[:2] + "_" + lang_for(n)[3:].upper(),
             "size_mb": None} for n in names]


def resolve(models_dir, name) -> str:
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        raise BadVoiceError("unknown voice")
    if name not in {v["name"] for v in list_voices(models_dir)}:
        raise BadVoiceError("unknown voice")
    return name


def build(models_dir, name, providers_hint=None):
    """Blocking. Returns (Handle, sample_rate, cuda_used); warmed up. Raises VoiceLoadError."""
    ok, why = available(models_dir)
    if not ok:
        raise VoiceLoadError(why or "kokoro unavailable")
    import onnxruntime as ort
    from kokoro_onnx import Kokoro
    model, voices = _paths(models_dir)
    avail = ort.get_available_providers()
    tries = []
    if "CUDAExecutionProvider" in avail:
        tries.append((["CUDAExecutionProvider", "CPUExecutionProvider"], True))
    tries.append((["CPUExecutionProvider"], False))
    last = None
    for providers, want_cuda in tries:
        try:
            sess = ort.InferenceSession(model, providers=providers)
            cuda = "CUDAExecutionProvider" in sess.get_providers()
            k = Kokoro.from_session(sess, voices)
            handle = Handle(k, name, lang_for(name), model)
            synth(handle, "warm up")
            return handle, SAMPLE_RATE, cuda
        except Exception as e:
            last = e
            log.exception("kokoro init failed (providers=%s)", providers)
    raise VoiceLoadError() from last


def synth(handle: Handle, text: str, speed: float = 1.0) -> bytes:
    import numpy as np
    samples, _sr = handle.kokoro.create(text, voice=handle.voice, speed=float(speed), lang=handle.lang)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    return pcm.tobytes()
