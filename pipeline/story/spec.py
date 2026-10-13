"""Storyboard spec: the contract between the creative step (LLM or human) and
the two storyboard renderers.

{
  "id": "...", "format": "doodle" | "cinematic", "game": "poker" | "settler_rush" | "last_vote",
  "cast": {"wren": {"name": "WREN", "kind": "human", "voice": "human_male"},
           "brock": {"name": "BROCK", "kind": "ai", "brand": "Claude", "voice": "robot_calm"}},
  "shots": [{"visual": {...}, "lines": [{"who": "wren", "say": "...", "tone": "..."}],
             "sfx": [{"at": 0.2, "cue": "card_flip"}], "overlays": [...], ...}],
  "music": {"mood": "quirky"}, "caption": "...", "hashtags": [...]
}

Everything visual is chosen from a closed vocabulary (DOODLE_VISUALS, OVERLAYS,
...) so the renderer never has to interpret free text, except cinematic image
prompts, which go to the image model.
"""
import json
import re
from pathlib import Path

from ..spec import BANNED, CARD_RE, ROSTER

SERIES_DIR = Path(__file__).resolve().parent.parent.parent / "series"

FORMATS = ("doodle", "cinematic")
GAMES = ("poker", "settler_rush", "last_vote")
VOICES = ("human_male", "human_female", "human_gruff", "human_young", "robot_calm",
          "robot_cold", "robot_cheerful", "narrator")
EXPRESSIONS = ("neutral", "worried", "annoyed", "shocked", "scream", "happy", "smug",
               "thinking", "sad", "angry")
FIGURE_POS = ("bl", "br", "bm")
CAMERAS = ("static", "push_in", "pull_out", "pan_left", "pan_right", "pan_up", "pan_down")
TRANSITIONS = ("cut", "fade", "flash", "glitch")

# Doodle visual vocabulary -> required params
DOODLE_VISUALS = {
    "table": [],             # two-shot at the game table (game-specific props)
    "closeup": ["who"],      # one character, expression, optional hand of cards
    "clock_calendar": [],    # clock spinning above a TODAY calendar
    "calendar_grid": [],     # years popping in one by one
    "sun_earth": [],         # sun burning out over a tiny earth
    "card_reveal": ["board", "hidden"],
    "dice": ["values"],
    "hex_board": [],         # mini catan board, robber drops on `target`
    "sign": ["text"],        # hand-written sign / whiteboard
    "vote": ["accused"],     # last-vote circle, hands point at someone
}
SFX_CUES = ("pop", "whoosh", "tick", "card_flip", "chips", "boing", "sting", "impact",
            "crickets", "dice_roll", "thud", "ding", "glitch", "riser", "boom", "waves",
            "wind", "hammer", "engine", "hoofbeats", "crowd", "horn", "coins", "sword")
OVERLAYS = ("title", "faction_tag", "equation", "counter", "stolen", "alert", "banner",
            "scoreboard", "approach")
MOODS = ("quirky", "tense", "epic", "dark", "triumph")

LIMITS = {"say": 90, "hook": 42, "hook_title": 12, "sign": 26, "word": 14, "prompt": 900,
          "caption": 300, "overlay_text": 34}
DURATION = {"doodle": (10.0, 35.0), "cinematic": (18.0, 60.0)}
WORDS_PER_SEC = 2.6  # rough speech-rate estimate used before TTS runs


class StorySpecError(Exception):
    def __init__(self, errors):
        super().__init__("; ".join(errors))
        self.errors = errors


def load(path):
    """Load a storyboard spec and merge its series bible (series/<ref>.json)."""
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    return resolve(spec)


def resolve(spec):
    ser = spec.get("series")
    if ser and ser.get("ref"):
        base = json.loads((SERIES_DIR / f"{ser['ref']}.json").read_text(encoding="utf-8"))
        base.pop("episodes", None)
        merged = {**base, **{k: v for k, v in ser.items() if k != "factions"}}
        merged["factions"] = {**base.get("factions", {}), **ser.get("factions", {})}
        spec = {**spec, "series": merged}
    return spec


def is_story(spec):
    return spec.get("format") in FORMATS


def estimate_duration(spec):
    t = 0.0
    for s in spec.get("shots", []):
        words = sum(len(ln.get("say", "").split()) for ln in s.get("lines", []))
        t += max(s.get("dur", 0), words / WORDS_PER_SEC + 0.6, s.get("min_dur", 1.2))
    return t


def validate(spec):
    e = []
    for k in ("id", "format", "game", "cast", "shots", "caption"):
        if k not in spec:
            e.append(f"missing field '{k}'")
    if e:
        return e
    fmt = spec["format"]
    if fmt not in FORMATS:
        return [f"format must be one of {FORMATS}"]
    if spec["game"] not in GAMES:
        e.append(f"game must be one of {GAMES}")
    if not re.fullmatch(r"[a-z0-9_]+", spec["id"]):
        e.append("id must be lowercase snake_case")

    cast = spec["cast"]
    if not cast:
        e.append("cast is empty")
    for key, c in cast.items():
        if c.get("kind") not in ("human", "ai", "narrator"):
            e.append(f"cast.{key}.kind must be human/ai/narrator")
        if c.get("voice") not in VOICES:
            e.append(f"cast.{key}.voice must be one of {VOICES}")
        if fmt == "doodle" and c.get("kind") == "ai" and c.get("brand") not in ROSTER:
            e.append(f"cast.{key}.brand must be one of {list(ROSTER)}")
        if fmt == "cinematic" and c.get("kind") != "narrator" \
                and c.get("faction") not in spec.get("series", {}).get("factions", {}):
            e.append(f"cast.{key}.faction must be a key of series.factions")

    shots = spec["shots"]
    if not 2 <= len(shots) <= 22:
        e.append("need 2..22 shots")
    for i, s in enumerate(shots):
        p = f"shots[{i}]"
        for j, ln in enumerate(s.get("lines", [])):
            if ln.get("who") not in cast:
                e.append(f"{p}.lines[{j}].who '{ln.get('who')}' not in cast")
            say = ln.get("say", "")
            if not say.strip():
                e.append(f"{p}.lines[{j}].say is empty")
            if len(say) > LIMITS["say"]:
                e.append(f"{p}.lines[{j}].say is {len(say)} chars (max {LIMITS['say']})")
            shown = ln.get("show", say)
            for w in shown.split():
                if len(w) > LIMITS["word"]:
                    e.append(f"{p}.lines[{j}] word '{w}' too long for a caption (max {LIMITS['word']})")
        for k, fx in enumerate(s.get("sfx", [])):
            if fx.get("cue") not in SFX_CUES:
                e.append(f"{p}.sfx[{k}].cue must be one of {SFX_CUES}")
        if s.get("ambience") and s["ambience"] not in SFX_CUES:
            e.append(f"{p}.ambience must be one of {SFX_CUES}")
        if s.get("transition", "cut") not in TRANSITIONS:
            e.append(f"{p}.transition must be one of {TRANSITIONS}")
        if not s.get("lines") and not s.get("dur") and s.get("type") != "endcard":
            e.append(f"{p} has no lines, so it needs an explicit 'dur'")
        if fmt == "doodle":
            e += _validate_doodle_shot(p, s, cast)
        else:
            e += _validate_cinematic_shot(p, s, spec, i)

    if fmt == "doodle":
        hook = spec.get("hook", {})
        if len(hook.get("text", "")) > LIMITS["hook"]:
            e.append(f"hook.text over {LIMITS['hook']} chars")
        if len(hook.get("title", "")) > LIMITS["hook_title"]:
            e.append(f"hook.title over {LIMITS['hook_title']} chars")
    else:
        ser = spec.get("series", {})
        for k in ("name", "tagline", "cta", "factions", "style"):
            if k not in ser:
                e.append(f"series.{k} missing")
        if shots and shots[-1].get("type") != "endcard":
            e.append("last cinematic shot must be {'type': 'endcard'}")

    lo, hi = DURATION[fmt]
    est = estimate_duration(spec)
    if not lo <= est <= hi:
        e.append(f"estimated duration {est:.1f}s outside {lo}-{hi}s")
    if spec.get("music", {}).get("mood", "quirky") not in MOODS:
        e.append(f"music.mood must be one of {MOODS}")
    if len(spec["caption"]) > LIMITS["caption"]:
        e.append("caption too long")
    if "gghuman.ai" not in spec["caption"].lower():
        e.append("caption must mention gghuman.ai")

    blob = " ".join([spec["caption"], spec.get("hook", {}).get("text", "")] +
                    [ln.get("say", "") for s in shots for ln in s.get("lines", [])]).lower()
    for pat in BANNED:
        if re.search(pat, blob):
            e.append(f"banned phrase /{pat}/")
    return e


def _cards_ok(cards, allow_hidden=False):
    return all(CARD_RE.match(c) or (allow_hidden and c == "?") for c in cards)


def _validate_doodle_shot(p, s, cast):
    e = []
    v = s.get("visual", {})
    vt = v.get("type")
    if vt not in DOODLE_VISUALS:
        return [f"{p}.visual.type must be one of {list(DOODLE_VISUALS)}"]
    for req in DOODLE_VISUALS[vt]:
        if req not in v:
            e.append(f"{p}.visual.{req} required for {vt}")
    if v.get("who") and v["who"] not in cast:
        e.append(f"{p}.visual.who not in cast")
    if v.get("expression", "neutral") not in EXPRESSIONS:
        e.append(f"{p}.visual.expression must be one of {EXPRESSIONS}")
    fig = v.get("figure")
    if fig:
        if fig.get("who") not in cast:
            e.append(f"{p}.visual.figure.who not in cast")
        if fig.get("pos", "bl") not in FIGURE_POS:
            e.append(f"{p}.visual.figure.pos must be one of {FIGURE_POS}")
        if fig.get("expression", "neutral") not in EXPRESSIONS:
            e.append(f"{p}.visual.figure.expression invalid")
    if vt == "card_reveal":
        if not _cards_ok(v.get("board", [])) or not _cards_ok(v.get("hidden", [])):
            e.append(f"{p}: card codes must look like 'A♥', '10♣'")
        allc = v.get("board", []) + v.get("hidden", [])
        if len(set(allc)) != len(allc):
            e.append(f"{p}: duplicate cards")
    if vt == "table" and "board" in v and not _cards_ok(v["board"] + v.get("you_hand", []), True):
        e.append(f"{p}: table card codes invalid")
    if vt == "closeup" and not _cards_ok(v.get("hand", [])):
        e.append(f"{p}: closeup hand card codes invalid")
    if vt == "calendar_grid" and not 1 <= int(v.get("count", 9)) <= 12:
        e.append(f"{p}: calendar_grid.count must be 1..12")
    if vt == "dice" and not all(1 <= int(x) <= 6 for x in v.get("values", [])):
        e.append(f"{p}: dice values must be 1..6")
    if vt == "hex_board" and int(v.get("target", 8)) not in (2, 3, 4, 5, 6, 8, 9, 10, 11, 12):
        e.append(f"{p}: hex_board.target must be a legal number token (not 7)")
    if vt == "sign":
        for line in v["text"].split("\n"):
            if len(line) > LIMITS["sign"]:
                e.append(f"{p}: sign line over {LIMITS['sign']} chars")
    if vt == "vote" and v.get("accused") not in cast:
        e.append(f"{p}: vote.accused not in cast")
    return e


def _validate_cinematic_shot(p, s, spec, i):
    e = []
    if s.get("type") == "endcard":
        return e
    v = s.get("visual", {})
    if not (v.get("prompt") or v.get("clip") or v.get("image")):
        e.append(f"{p}.visual needs prompt (or clip/image path)")
    if len(v.get("prompt", "")) > LIMITS["prompt"]:
        e.append(f"{p}.visual.prompt too long")
    if v.get("camera", "push_in") not in CAMERAS:
        e.append(f"{p}.visual.camera must be one of {CAMERAS}")
    factions = spec.get("series", {}).get("factions", {})
    for k, o in enumerate(s.get("overlays", [])):
        q = f"{p}.overlays[{k}]"
        t = o.get("type")
        if t not in OVERLAYS:
            e.append(f"{q}.type must be one of {OVERLAYS}")
            continue
        if t in ("faction_tag", "approach") and o.get("faction") not in factions:
            e.append(f"{q}.faction must be a key of series.factions")
        if t == "scoreboard":
            for tm in o.get("teams", []):
                if tm.get("faction") not in factions:
                    e.append(f"{q}.teams faction '{tm.get('faction')}' unknown")
        for key in ("text", "title", "label", "main", "top"):
            if len(str(o.get(key, ""))) > LIMITS["overlay_text"]:
                e.append(f"{q}.{key} over {LIMITS['overlay_text']} chars")
    return e


def check(spec):
    errs = validate(spec)
    if errs:
        raise StorySpecError(errs)
    return spec
