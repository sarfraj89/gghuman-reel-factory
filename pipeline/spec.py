"""Video spec: the contract between the creative (LLM) step and the renderer.

Every video is one JSON object. The LLM only ever fills these fields; it never
touches layout, timing or animation. Anything that fails `validate()` is sent
back to the LLM with the error list (see ideate.py) instead of being rendered.
"""
import json
import re
from pathlib import Path

# Opponents the site actually seats (gghuman.ai research roster) + brand colour.
ROSTER = {
    "Claude": "#D97757",
    "GPT": "#19C37D",
    "Gemini": "#4C8DF6",
    "Grok": "#C9CDD3",
    "DeepSeek": "#5B73FF",
    "Kimi": "#A78BFA",
    "Qwen": "#8B5CF6",
    "Mistral": "#FA6A1F",
    "MiniMax": "#F43F5E",
}
YOU_COLOR = "#7CC243"
GAMES = {"settler_rush": "SETTLER RUSH", "poker": "NO-LIMIT HOLD'EM", "last_vote": "LAST VOTE"}

# Hard character limits. They are what keeps text inside the safe zones, so the
# renderer never has to shrink fonts or guess. (The renderer still double-checks
# real pixel widths and raises LayoutError.)
LIMITS = {
    "hook": 46,
    "you_line": 52,
    "ai_said": 60,
    "ai_thought": 84,
    "twist": 30,
    "twist_label": 22,
    "cta_item": 14,
    "caption": 220,
}
BANNED = [
    r"\bfree\b", r"\bguarantee", r"\bwin money\b", r"\bgambl", r"\bcasino\b",
    r"\bkill\b", r"\bhate\b", r"\bstupid\b", r"\bidiot\b",
]
CARD_RE = re.compile(r"^(10|[2-9JQKA])[♠♥♦♣]$")


class SpecError(Exception):
    def __init__(self, errors):
        super().__init__("; ".join(errors))
        self.errors = errors


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate(spec):
    """Return a list of human-readable errors (empty list = valid)."""
    e = []
    req = ["id", "game", "opponent", "hook", "hook_highlight", "you_line",
           "ai_said", "ai_thought", "twist", "cta", "caption", "scene"]
    for k in req:
        if k not in spec or spec[k] in ("", None, []):
            e.append(f"missing field '{k}'")
    if e:
        return e

    if spec["game"] not in GAMES:
        e.append(f"game must be one of {list(GAMES)}")
    if spec["opponent"] not in ROSTER:
        e.append(f"opponent must be one of {list(ROSTER)}")
    if not re.fullmatch(r"[0-9]{3}_[a-z0-9_]+", spec["id"]):
        e.append("id must look like 012_short_slug")

    for k in ["hook", "you_line", "ai_said", "ai_thought", "twist", "caption"]:
        if len(spec[k]) > LIMITS[k]:
            e.append(f"{k} is {len(spec[k])} chars, max {LIMITS[k]}")
    if len(spec.get("twist_label", "")) > LIMITS["twist_label"]:
        e.append(f"twist_label max {LIMITS['twist_label']} chars")
    if spec["hook_highlight"] not in spec["hook"]:
        e.append("hook_highlight must be a substring of hook")
    if not (isinstance(spec["cta"], list) and len(spec["cta"]) == 3):
        e.append("cta must be a list of exactly 3 short lines")
    else:
        for c in spec["cta"]:
            if len(c) > LIMITS["cta_item"]:
                e.append(f"cta line '{c}' longer than {LIMITS['cta_item']}")
    if "gghuman.ai" not in spec["caption"]:
        e.append("caption must contain gghuman.ai")
    if spec["opponent"].lower() not in (spec["hook"] + spec["caption"]).lower():
        e.append("opponent must be named in hook or caption")

    text = " ".join([spec["hook"], spec["you_line"], spec["ai_said"],
                     spec["ai_thought"], spec["twist"], spec["caption"]]).lower()
    for pat in BANNED:
        if re.search(pat, text):
            e.append(f"banned phrase matched: {pat}")
    if any(ord(ch) > 0x2FFF and ch not in "♠♥♦♣" for ch in text):
        e.append("no emoji / non-Latin symbols in on-screen text")

    e += _validate_scene(spec)
    return e


def _validate_scene(spec):
    e, s, g = [], spec["scene"], spec["game"]
    if g == "settler_rush":
        if s.get("target_number") not in (2, 3, 4, 5, 6, 8, 9, 10, 11, 12):
            e.append("scene.target_number must be a valid dice number (not 7)")
        if len(s.get("loss", "")) > 12:
            e.append("scene.loss max 12 chars, e.g. '-2 WHEAT'")
    elif g == "poker":
        cards = s.get("board", []) + s.get("you_hand", []) + s.get("ai_hand", [])
        if len(s.get("board", [])) != 5 or len(s.get("you_hand", [])) != 2 or len(s.get("ai_hand", [])) != 2:
            e.append("poker scene needs board[5], you_hand[2], ai_hand[2]")
        bad = [c for c in cards if not CARD_RE.match(c)]
        if bad:
            e.append(f"invalid card codes {bad}; use e.g. 'K♥', '10♠'")
        if len(set(cards)) != len(cards):
            e.append("duplicate cards in poker scene")
    elif g == "last_vote":
        seats = s.get("seats", [])
        if not (4 <= len(seats) <= 6) or seats[:1] != ["YOU"]:
            e.append("scene.seats must have 4-6 entries starting with 'YOU'")
        if spec["opponent"] not in seats:
            e.append("opponent must be one of the seats")
        if s.get("accused") not in seats or s.get("accused") in ("YOU", spec["opponent"]):
            e.append("scene.accused must be another seat (not YOU / opponent)")
        for name in seats[1:]:
            if name not in ROSTER:
                e.append(f"seat '{name}' not in roster")
    return e


def check(spec):
    errs = validate(spec)
    if errs:
        raise SpecError(errs)
    return spec


def json_schema():
    """JSON schema handed to the LLM as a forced tool, so output is structured."""
    s = {"type": "string"}
    return {
        "type": "object",
        "properties": {
            "id": s, "game": {"type": "string", "enum": list(GAMES)},
            "opponent": {"type": "string", "enum": list(ROSTER)},
            "hook": s, "hook_highlight": s, "you_line": s, "ai_said": s,
            "ai_thought": s, "twist_label": s, "twist": s,
            "cta": {"type": "array", "items": s, "minItems": 3, "maxItems": 3},
            "caption": s, "hashtags": {"type": "array", "items": s},
            "scene": {"type": "object"},
        },
        "required": ["id", "game", "opponent", "hook", "hook_highlight", "you_line",
                     "ai_said", "ai_thought", "twist", "cta", "caption", "scene"],
    }
