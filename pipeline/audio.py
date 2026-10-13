"""Procedural soundtrack + SFX, generated from the same timeline as the picture.

No stock music = no licensing risk and no audio/visual drift. A production
version can swap the bed for a licensed track library or an AI music model;
the SFX cues stay timeline-driven.
"""
import wave

import numpy as np

SR = 44100


def _env(n, attack=0.002, decay=0.2):
    t = np.arange(n) / SR
    a = np.clip(t / attack, 0, 1)
    return a * np.exp(-t / decay)


def _add(buf, sig, t0, gain=1.0):
    i = int(t0 * SR)
    if i >= len(buf) or i < 0:
        return
    j = min(len(buf), i + len(sig))
    buf[i:j] += sig[: j - i] * gain


def _sweep(f0, f1, dur, decay=None):
    n = int(dur * SR)
    f = np.geomspace(f0, f1, n)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * _env(n, 0.003, decay or dur / 2.5)


def _noise(dur, decay, rng, lp=0.0):
    n = int(dur * SR)
    x = rng.standard_normal(n)
    if lp > 0:  # one-pole low-pass
        y = np.empty_like(x)
        acc = 0.0
        for k in range(n):
            acc += lp * (x[k] - acc)
            y[k] = acc
        x = y / (np.abs(y).max() + 1e-9)
    return x * _env(n, 0.001, decay)


def _kick():
    return _sweep(150, 42, 0.45, 0.16)


def _tick(rng):
    return _noise(0.012, 0.003, rng) * 0.6 + _sweep(2400, 1800, 0.012, 0.004) * 0.4


def _pop():
    return _sweep(380, 980, 0.09, 0.05)


def _boom(rng):
    s = _sweep(90, 32, 1.6, 0.55)
    nz = _noise(0.6, 0.18, rng, lp=0.05) * 0.5
    s[: len(nz)] += nz
    return s


def _chime(f):
    n = int(1.2 * SR)
    t = np.arange(n) / SR
    s = sum(np.sin(2 * np.pi * f * m * t) / (m * m) for m in (1, 2, 3))
    return s * _env(n, 0.004, 0.35)


def render(spec, tl, path):
    rng = np.random.default_rng(11)
    total = tl["total"] + 0.3
    buf = np.zeros(int(total * SR), np.float64)

    # --- music bed: 100 bpm pulse + low drone, drops out before the twist ---
    beat = 0.6
    t = 0.0
    while t < total:
        in_drop = tl["twist"] - 0.9 <= t < tl["twist"]
        if not in_drop:
            g = 0.55 if t < tl["end"] else 0.75
            _add(buf, _kick(), t, g)
            _add(buf, _noise(0.03, 0.01, rng) * 0.08, t + beat / 2)
        t += beat
    n = len(buf)
    tt = np.arange(n) / SR
    drone = (np.sin(2 * np.pi * 55 * tt) + 0.5 * np.sin(2 * np.pi * 82.4 * tt + 0.3)) * 0.07
    drone *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.25 * tt)
    mask = np.ones(n)
    a, b = int((tl["twist"] - 0.9) * SR), int(tl["twist"] * SR)
    mask[a:b] = np.linspace(1, 0.1, max(1, b - a))
    buf += drone * mask

    # --- chat typing + pops ---
    for key, start in [("you_line", "you"), ("ai_said", "said")]:
        _add(buf, _pop(), tl[start], 0.35)
        for k in range(len(spec[key])):
            if spec[key][k] != " ":
                _add(buf, _tick(rng), tl[start] + k / 38, 0.22 * rng.uniform(0.6, 1.0))

    # --- inner-monologue decode: glitchy granular noise + rising tone ---
    t0, t1 = tl["thought"], tl["thought_end"]
    _add(buf, _sweep(1200, 300, 0.25, 0.1), t0, 0.3)
    g = t0 + 0.35
    while g < t1:
        _add(buf, _noise(0.02, 0.008, rng) * rng.uniform(0.3, 1), g, 0.14)
        g += rng.uniform(0.025, 0.06)
    _add(buf, _sweep(180, 720, t1 - t0, (t1 - t0)) * 0.5, t0, 0.18)

    # --- game event ---
    ev = tl["event"]
    game = spec["game"]
    if game == "settler_rush":
        _add(buf, _sweep(900, 200, 0.4, 0.2), ev, 0.25)
        _add(buf, _boom(rng), ev + 0.42, 0.55)
    elif game == "poker":
        for k in range(8):  # chips on the all-in
            _add(buf, _tick(rng), tl["said"] + 0.05 + k * 0.05, 0.5)
        _add(buf, _sweep(500, 140, 0.25, 0.1), ev, 0.4)           # fold
        _add(buf, _noise(0.15, 0.05, rng), ev + 0.55, 0.25)        # card flip
        _add(buf, _noise(0.15, 0.05, rng), ev + 0.65, 0.25)
    else:
        for k in range(len(spec["scene"]["seats"]) - 1):
            _add(buf, _sweep(600 + 120 * k, 900 + 120 * k, 0.12, 0.06), ev + k * 0.2 + 0.45, 0.35)
        _add(buf, _boom(rng), ev + (len(spec["scene"]["seats"]) - 1) * 0.2 + 0.35, 0.45)

    # --- twist impact + end card ---
    _add(buf, _boom(rng), tl["twist"], 0.9)
    _add(buf, _noise(0.5, 0.2, rng), tl["twist"], 0.2)
    _add(buf, _sweep(200, 2000, 0.4, 0.2), tl["end"] - 0.1, 0.15)
    for k, tc in enumerate(tl["cta"]):
        _add(buf, _chime([440, 554, 659][k]), tc, 0.25)
    _add(buf, _chime(880), tl["url"], 0.3)

    # fade out + normalise (loudness is set properly by ffmpeg loudnorm)
    fo = int(0.6 * SR)
    buf[-fo:] *= np.linspace(1, 0, fo)
    buf = np.tanh(buf * 1.2)
    buf /= np.abs(buf).max() + 1e-9
    pcm = (buf * 0.89 * 32767).astype(np.int16)
    stereo = np.repeat(pcm[:, None], 2, axis=1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(stereo.tobytes())
    return path


def master_filter(wav_path, lufs=-14.0, tp=-1.5):
    """ffmpeg -af chain that hits the Instagram loudness target without clipping.

    Single-pass loudnorm lets peaks through (its limiter is look-ahead only, and
    AAC encoding overshoots a little more), which shows up as crackle on screams
    and impacts. So: measure first, apply one linear gain, then a hard limiter
    with ~1 dB extra headroom for the encoder."""
    import json
    import re
    import subprocess
    r = subprocess.run(["ffmpeg", "-nostats", "-i", str(wav_path), "-af",
                        f"loudnorm=I={lufs}:TP={tp}:LRA=11:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True)
    m = json.loads(re.findall(r"\{[^{}]*\}", r.stderr)[-1])
    limit = 10 ** ((tp - 1.0) / 20)
    return (f"loudnorm=I={lufs}:TP={tp}:LRA=11:linear=true:measured_I={m['input_i']}:"
            f"measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:"
            f"measured_thresh={m['input_thresh']}:offset={m['target_offset']},"
            f"alimiter=limit={limit:.3f}:attack=2:release=60:level=false")
