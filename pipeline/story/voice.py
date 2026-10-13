"""Character voices. Provider chain (first one configured wins, or force with GG_TTS):

  elevenlabs  ELEVENLABS_API_KEY   best acting; returns per-character timestamps
  openai      OPENAI_API_KEY       gpt-4o-mini-tts with per-line acting instructions
  piper       local, free          assets/voices/*.onnx (no key, always available)

Every clip is cached by (provider, voice, text, tone), so re-renders and retries
never pay twice. Voice roles map to provider voices below; override any of them
with env vars, e.g. GG_VOICE_ROBOT_CALM=<elevenlabs voice id>.
"""
import base64
import hashlib
import json
import os
import re
import subprocess
import urllib.request
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..draw import FPS

SR = 44100
ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = ROOT / "cache" / "vo"
VOICE_DIR = ROOT / "assets" / "voices"

# role -> (elevenlabs premade voice id (current default voices), openai voice, openai acting brief, piper model, piper length_scale)
ROLES = {
    "human_male":     ("TX3LPaxmHKxFdv7VOQHJ", "ash", "Expressive everyday guy", "en_US-ryan-high", 0.95),
    "human_female":   ("cgSgspJ2msm6clMCkdW9", "coral", "Expressive, warm woman", "en_US-lessac-medium", 0.95),
    "human_gruff":    ("SOYHLrjzK2X1ezoPC6cr", "onyx", "Gruff, weathered medieval settler", "en_US-ryan-high", 1.05),
    "human_young":    ("IKne3meq5aSn9XLyUdCD", "verse", "Young, excitable", "en_US-lessac-medium", 0.9),
    "robot_calm":     ("onwK4e9ZLuTAKqWW03F9", "sage", "Calm, slow, unbothered AI. Flat affect.", "en_GB-alan-medium", 1.15),
    "robot_cold":     ("pNInz6obpgDQGcFmaJgB", "onyx", "Cold, precise military AI commander", "en_GB-alan-medium", 1.05),
    "robot_cheerful": ("FGY2WhTYpPnrIDTdsKH5", "nova", "Overly cheerful helpful AI assistant", "en_US-lessac-medium", 1.0),
    "narrator":       ("JBFqnCBsd6RMkjVDRZzb", "fable", "Epic movie-trailer narrator", "en_GB-alan-medium", 1.0),
}
ROBOT_FX = {"robot_calm": 0.35, "robot_cold": 0.45, "robot_cheerful": 0.25}


@dataclass
class Clip:
    audio: np.ndarray            # mono float32 @ SR
    words: list = field(default_factory=list)   # [(word, t0, t1)] seconds from clip start
    env: np.ndarray = None       # per-video-frame loudness 0..1 (mouth flap)

    @property
    def dur(self):
        return len(self.audio) / SR


def provider():
    forced = os.environ.get("GG_TTS")
    if forced:
        return forced
    if os.environ.get("ELEVENLABS_API_KEY"):
        return "elevenlabs"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "piper"


def _role_value(role, idx):
    env = os.environ.get(f"GG_VOICE_{role.upper()}")
    if env and idx in (0, 1):
        return env
    return ROLES[role][idx]


def decode(path):
    """Any audio file -> mono float32 @ SR (via ffmpeg)."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR),
                        "-f", "f32le", "-"], capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.float32).copy()


def _post(url, headers, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


# ---------------- providers ----------------
def _elevenlabs(text, role, tone, out_base):
    vid = _role_value(role, 0)
    model = os.environ.get("GG_ELEVEN_MODEL", "eleven_multilingual_v2")
    if tone and "v3" in model:
        text = f"[{tone}] {text}"
    body = {"text": text, "model_id": model,
            "voice_settings": {"stability": 0.35 if tone else 0.5, "similarity_boost": 0.8,
                               "style": 0.45 if tone else 0.2}}
    raw = _post(f"https://api.elevenlabs.io/v1/text-to-speech/{vid}/with-timestamps",
                {"xi-api-key": os.environ["ELEVENLABS_API_KEY"], "Content-Type": "application/json"}, body)
    data = json.loads(raw)
    mp3 = out_base.with_suffix(".mp3")
    mp3.write_bytes(base64.b64decode(data["audio_base64"]))
    al = data.get("alignment") or {}
    words = _words_from_chars(al.get("characters", []), al.get("character_start_times_seconds", []),
                              al.get("character_end_times_seconds", []))
    return decode(mp3), words


def _openai(text, role, tone, out_base):
    brief = _role_value(role, 2) + (f". Tone for this line: {tone}." if tone else ".")
    raw = _post("https://api.openai.com/v1/audio/speech",
                {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "Content-Type": "application/json"},
                {"model": os.environ.get("GG_OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
                 "voice": _role_value(role, 1), "input": text, "instructions": brief,
                 "response_format": "wav"})
    p = out_base.with_suffix(".src.wav")
    p.write_bytes(raw)
    return decode(p), None


_PIPER = {}


def _piper(text, role, tone, out_base):
    from piper import PiperVoice, SynthesisConfig
    model, length = ROLES[role][3], ROLES[role][4]
    if model not in _PIPER:
        _PIPER[model] = PiperVoice.load(str(VOICE_DIR / f"{model}.onnx"))
    p = out_base.with_suffix(".src.wav")
    with wave.open(str(p), "wb") as w:
        _PIPER[model].synthesize_wav(text, w, syn_config=SynthesisConfig(length_scale=length))
    return decode(p), None


PROVIDERS = {"elevenlabs": _elevenlabs, "openai": _openai, "piper": _piper}


# ---------------- timing helpers ----------------
def _words_from_chars(chars, starts, ends):
    words, cur, t0, t1 = [], "", None, None
    for ch, s, e in zip(chars, starts, ends):
        if ch.isspace():
            if cur:
                words.append((cur, t0, t1))
            cur, t0 = "", None
            continue
        if t0 is None:
            t0 = s
        cur += ch
        t1 = e
    if cur:
        words.append((cur, t0, t1))
    return words


def _speech_span(audio, thr=0.02):
    idx = np.nonzero(np.abs(audio) > thr * (np.abs(audio).max() + 1e-9))[0]
    if not len(idx):
        return 0.0, len(audio) / SR
    return idx[0] / SR, idx[-1] / SR


def estimate_words(text, audio):
    """Distribute words over the voiced span, weighted by length + punctuation pauses."""
    toks = text.split()
    if not toks:
        return []
    s0, s1 = _speech_span(audio)
    weights = []
    for w in toks:
        letters = len(re.sub(r"[^\w]", "", w)) + 1.5
        pause = 3.0 if re.search(r"[.!?]$", w) else 1.5 if re.search(r"[,;:]$", w) else 0.0
        weights.append((letters, pause))
    total = sum(a + b for a, b in weights)
    t, out = s0, []
    for w, (a, b) in zip(toks, weights):
        d = (s1 - s0) * a / total
        out.append((w, t, t + d))
        t += d + (s1 - s0) * b / total
    return out


def envelope(audio):
    hop = SR // FPS
    n = len(audio) // hop + 1
    pad = np.zeros(n * hop, np.float32)
    pad[:len(audio)] = audio
    rms = np.sqrt((pad.reshape(n, hop) ** 2).mean(axis=1))
    return np.clip(rms / (np.percentile(rms, 95) + 1e-6), 0, 1)


def robotize(audio, amount):
    t = np.arange(len(audio)) / SR
    ring = audio * np.sin(2 * np.pi * 62 * t)
    d = int(0.006 * SR)
    comb = audio.copy()
    comb[d:] += 0.45 * audio[:-d]
    out = (1 - amount) * comb + amount * ring * 1.6
    return out / (np.abs(out).max() + 1e-9) * np.abs(audio).max()


# ---------------- public ----------------
def speak(text, role, tone=None):
    prov = provider()
    key = hashlib.sha1(json.dumps([prov, role, _role_value(role, 0), text, tone]).encode()).hexdigest()[:16]
    CACHE.mkdir(parents=True, exist_ok=True)
    base = CACHE / key
    wav, meta = base.with_suffix(".wav"), base.with_suffix(".json")
    if wav.exists() and meta.exists():
        audio = decode(wav)
        words = [tuple(w) for w in json.loads(meta.read_text())["words"]]
    else:
        audio, words = PROVIDERS[prov](text, role, tone, base)
        if role in ROBOT_FX:
            audio = robotize(audio, ROBOT_FX[role])
        audio, off = _trim(audio)
        words = [(w, max(0.0, a - off), max(0.0, b - off)) for w, a, b in (words or [])]
        if not words:
            words = estimate_words(text, audio)
        _write_wav(wav, audio)
        meta.write_text(json.dumps({"provider": prov, "role": role, "text": text, "words": words}))
    return Clip(audio=audio, words=words, env=envelope(audio))


def _trim(audio, thr=0.01, keep=0.04):
    s0, s1 = _speech_span(audio, thr)
    a, b = max(0, int((s0 - keep) * SR)), min(len(audio), int((s1 + keep * 2) * SR))
    return audio[a:b].astype(np.float32), a / SR


def _write_wav(path, audio):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def synthesize_all(spec):
    """Return {(shot_idx, line_idx): Clip} for every dialogue line in the spec."""
    out = {}
    for i, s in enumerate(spec["shots"]):
        for j, ln in enumerate(s.get("lines", [])):
            role = spec["cast"][ln["who"]]["voice"]
            out[(i, j)] = speak(ln["say"], role, ln.get("tone"))
    return out
