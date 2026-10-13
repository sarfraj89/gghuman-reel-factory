"""Storyboard timeline: shot lengths come from the synthesised voice lines, and
captions come from the per-word timings, so picture, captions and sound share
one clock."""
import re
from dataclasses import dataclass, field

LEAD = 0.25       # silence before the first line of a shot
GAP = 0.22        # silence between lines in a shot
HOLD = 0.35       # silence after the last line before cutting
MIN_SHOT = 1.0
ENDCARD = 3.6     # minimum end card length (cinematic)


@dataclass
class Line:
    shot: int
    idx: int
    who: str
    start: float
    clip: object
    show: str

    @property
    def end(self):
        return self.start + self.clip.dur


@dataclass
class Shot:
    idx: int
    start: float
    dur: float
    data: dict
    lines: list = field(default_factory=list)

    @property
    def end(self):
        return self.start + self.dur


@dataclass
class Caption:
    start: float
    end: float
    text: str
    who: str
    shot: int


@dataclass
class Timeline:
    shots: list
    lines: list
    captions: list
    total: float

    def shot_at(self, t):
        for s in self.shots:
            if t < s.end:
                return s
        return self.shots[-1]

    def speaking(self, t, fps):
        """{who: mouth_open_amount} at time t."""
        out = {}
        for ln in self.lines:
            if ln.start <= t < ln.end:
                k = int((t - ln.start) * fps)
                env = ln.clip.env
                out[ln.who] = float(env[min(k, len(env) - 1)]) if env is not None else 1.0
        return out


def build(spec, clips, max_words=1, max_chars=14):
    t = 0.0
    shots, lines = [], []
    for i, s in enumerate(spec["shots"]):
        cur = s.get("lead", LEAD) if s.get("lines") else 0.0
        sl = []
        for j, ln in enumerate(s.get("lines", [])):
            start = cur + ln.get("delay", 0.0)
            line = Line(i, j, ln["who"], t + start, clips[(i, j)], ln.get("show", ln["say"]))
            sl.append(line)
            cur = start + line.clip.dur + GAP
        talk = cur - GAP + s.get("hold", HOLD) if sl else 0.0
        floor = ENDCARD if s.get("type") == "endcard" else MIN_SHOT
        dur = max(talk, s.get("dur", 0.0), s.get("min_dur", floor))
        shots.append(Shot(i, t, dur, s, sl))
        lines += sl
        t += dur
    caps = []
    for ln in lines:
        if not ln.show or spec["shots"][ln.shot].get("type") == "endcard":
            continue
        caps += _chunks(ln, max_words, max_chars)
    # each caption stays up until the next one starts (same shot), so text never flickers
    for a, b in zip(caps, caps[1:]):
        if b.shot == a.shot and b.start - a.end < 0.6:
            a.end = b.start
    for c in caps:
        c.end = min(c.end, shots[c.shot].end)
    return Timeline(shots, lines, caps, t)


def _chunks(line, max_words, max_chars):
    """Group words into caption chunks, breaking after punctuation."""
    shown = line.show.split()
    words = line.clip.words
    if len(words) != len(shown):  # `show` differs from what was spoken: spread evenly
        d = line.clip.dur / max(1, len(shown))
        words = [(w, k * d, (k + 1) * d) for k, w in enumerate(shown)]
    else:
        words = [(s, a, b) for s, (_, a, b) in zip(shown, words)]
    out, cur = [], []
    for k, (w, a, b) in enumerate(words):
        cur.append((w, a, b))
        text = " ".join(x[0] for x in cur)
        nxt = words[k + 1][0] if k + 1 < len(words) else ""
        brk = (len(cur) >= max_words or re.search(r"[.!?,;:]$", w)
               or len(text + " " + nxt) > max_chars or not nxt)
        if brk:
            out.append(Caption(line.start + cur[0][1], line.start + cur[-1][2] + 0.25,
                               text.upper(), line.who, line.shot))
            cur = []
    return out
