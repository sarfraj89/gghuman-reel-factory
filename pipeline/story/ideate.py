"""LLM writes new storyboard specs (new skits / new episodes) in the two
reference formats. Same fence as the original ideate.py:

  topic matrix -> LLM (forced JSON tool, few-shot = the reference recreations)
               -> story validator -> retry with exact errors -> dedupe -> judge

Cinematic episodes continue the season: the series bible's episode log and
scoreboard are passed in, and each accepted episode is appended to it.
"""
import itertools
import json
import random
from pathlib import Path

from .. import ideate as base
from . import spec as SS

ROOT = Path(__file__).resolve().parent.parent.parent
REF_DIR = ROOT / "specs_story"

# ---- doodle: AI behaviour x gag structure x game ----
DOODLE_BEHAVIOURS = {
    "poker": ["slow-rolls the nuts forever", "explains pot odds for way too long before calling",
              "bluffs with total air and admits it calmly", "folds the winning hand 'to be polite'",
              "goes all-in every single hand and keeps winning", "remembers every card you've ever shown"],
    "settler_rush": ["refuses every trade with extreme politeness", "builds the longest road around your town",
                     "puts the robber on your 8 every turn", "hoards 19 sheep for a reason it won't explain",
                     "offers a 'fair' 4-for-1 trade that is not fair", "calculates the dice odds out loud"],
    "last_vote": ["defends itself with a 40-point logical proof", "votes you out 'with love'",
                  "fake-claims detective so convincingly you vote yourself", "stays silent all day, then wins"],
}
GAGS = ["escalation x3: the human's complaint gets more absurd each beat, the AI only says one word, then a reveal",
        "expectation vs reality: the human predicts the AI's move, the AI does something calm and devastating",
        "misdirect: the AI looks confused/helpless, the final shot reveals it planned everything",
        "time-skip: the same situation repeats over absurd timescales (turns, years, eons)"]

# ---- cinematic: Catan beats for an episode ----
CINEMATIC_EVENTS = [
    "a 7 is rolled and everyone with too many cards must discard", "a monopoly card drains every human's brick",
    "a port trade at 2:1 turns the tide", "the race for longest road", "two AIs ally against the humans, then betray each other",
    "the humans upgrade to their first city", "a road block traps an AI, which pivots strategy",
    "the robber sits on the humans' best tile for five turns", "a year-of-plenty card arrives at the perfect moment",
    "the final turn: someone reaches 10 victory points",
]

SYSTEM = """You write storyboard specs for short vertical Instagram Reels for gghuman.ai,
where people play board games (Settler Rush = Catan-like, poker, Last Vote = mafia-like)
against frontier AI models. You must output ONE spec via the tool `submit_spec`.

The spec is rendered by code from a closed vocabulary. Never invent visual types,
overlay types, sfx cues, voices or expressions that are not in the lists given.
Dialogue is VOICED by TTS and shown as 1-3 word captions, so write short punchy
spoken lines (2-8 words each), the way people really talk. Every shot should
last 1.5-4.5s; cut often. Playful, never mean; no gambling-for-money language,
no 'free', no emoji. Game moves must be legal and plausible."""

FORMAT_BRIEF = {
    "doodle": """FORMAT "doodle": flat hand-drawn cartoon on cream paper, 15-25 seconds total.
A 'POV:'-style hook (hook.title + hook.text) shows over the first shot. A green blob
human vs a TV-headed robot in an AI brand color. Structure: setup at the game table,
a repeating comic beat that escalates, then a visual payoff, then a reaction
closeup (often 'scream'). Use visual types: {visuals}. Expressions: {expr}.
Voices: {voices}. SFX cues: {sfx}. Music moods: {moods}. Shots with no
dialogue need "dur". "music": 0.0 on a shot drops the music for a beat of silence.""",
    "cinematic": """FORMAT "cinematic": photoreal AI-generated episodic series, 25-45 seconds,
EP {ep} of "{series}". Each shot has visual.prompt (a concrete film-still description
for an image model: subject, action, setting, light; no text in the image),
visual.motion (what moves, for image-to-video), visual.camera, visual.factions
(faction keys whose look should be applied). Captions are styled per faction.
Overlays (HUD drawn by code): {overlays}. The first shot carries a
{{"type":"title","title":...}} overlay. The last shot is {{"type":"endcard"}} with a
narrator line ending in "Play leading AIs at G G human dot A I." and "show": "".
Use the scoreboard overlay near the end and keep scores consistent with the season.
Factions: {factions}. Voices: {voices}. SFX cues: {sfx}. Moods: {moods}.
Transitions: {transitions}. Cameras: {cameras}.
Season so far: {season}""",
}


def _few_shot(fmt):
    ex = []
    for p in sorted(REF_DIR.glob("ref_*.json")):
        d = json.loads(p.read_text())
        if d.get("format") == fmt:
            ex.append(d)
    return json.dumps(ex, ensure_ascii=False, indent=1)


def _schema():
    return {"type": "object", "required": ["id", "format", "game", "cast", "shots", "caption", "hashtags"],
            "properties": {"id": {"type": "string"}, "format": {"type": "string", "enum": list(SS.FORMATS)},
                           "game": {"type": "string", "enum": list(SS.GAMES)},
                           "hook": {"type": "object"}, "series": {"type": "object"},
                           "cast": {"type": "object"}, "music": {"type": "object"},
                           "shots": {"type": "array", "items": {"type": "object"}},
                           "caption": {"type": "string"}, "hashtags": {"type": "array", "items": {"type": "string"}},
                           "premise": {"type": "string", "description": "one-sentence premise, for dedupe"}}}


def _ask(client, messages):
    """Current models reject forced tool_choice, so use `auto` + an explicit
    instruction, and nudge once if the model answers without calling the tool."""
    tools = [{"name": "submit_spec", "description": "Submit one storyboard spec. Always call this tool.",
              "input_schema": _schema()}]
    for _ in range(2):
        resp = client.messages.create(model=base.MODEL, max_tokens=16000, system=SYSTEM, messages=messages,
                                      tools=tools, tool_choice={"type": "auto"})
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"model refused: {resp.stop_details}")
        block = next((b for b in resp.content if b.type == "tool_use"), None)
        if block:
            return block.input, resp
        messages = messages + [{"role": "assistant", "content": resp.content},
                               {"role": "user", "content": "Submit the spec now by calling the submit_spec tool."}]
    raise RuntimeError("model did not call submit_spec")


def topics(fmt, n, seed=None):
    rng = random.Random(seed)
    if fmt == "doodle":
        combos = [(g, b, gag, o) for g, bs in DOODLE_BEHAVIOURS.items() for b in bs for gag in GAGS
                  for o in ("Claude", "GPT", "Gemini", "Grok", "DeepSeek")]
    else:
        combos = [(e,) for e in CINEMATIC_EVENTS]
    rng.shuffle(combos)
    return list(itertools.islice(itertools.cycle(combos), n))


def brief(fmt, topic, idx, series_ref=None):
    if fmt == "doodle":
        game, behaviour, gag, opp = topic
        head = (f"Write doodle spec id 'doodle_{idx:03d}_<slug>'. game={game}. The AI (brand {opp}) "
                f"{behaviour}. Gag structure: {gag}.")
        body = FORMAT_BRIEF["doodle"].format(
            visuals=json.dumps({k: v for k, v in SS.DOODLE_VISUALS.items()}), expr=list(SS.EXPRESSIONS),
            voices=list(SS.VOICES), sfx=list(SS.SFX_CUES), moods=list(SS.MOODS))
    else:
        ser = json.loads((SS.SERIES_DIR / f"{series_ref}.json").read_text())
        ep = max([e["ep"] for e in ser.get("episodes", [])] + [0]) + 1
        head = (f"Write cinematic spec id '{series_ref}_ep{ep}_<slug>' for EP {ep}. "
                f"Set \"series\": {{\"ref\": \"{series_ref}\", \"ep\": {ep}}}. Central event: {topic[0]}.")
        body = FORMAT_BRIEF["cinematic"].format(
            ep=ep, series=ser["name"], overlays=list(SS.OVERLAYS),
            factions=json.dumps({k: v.get("label") for k, v in ser["factions"].items()}),
            voices=list(SS.VOICES), sfx=list(SS.SFX_CUES), moods=list(SS.MOODS),
            transitions=list(SS.TRANSITIONS), cameras=list(SS.CAMERAS),
            season=json.dumps(ser.get("episodes", [])))
    return f"{head}\n\n{body}\n\nLimits: {json.dumps(SS.LIMITS)}\n\nReference specs (match this quality and pacing):\n{_few_shot(fmt)}"


def generate_one(client, fmt, topic, idx, premises, series_ref=None, feedback=None):
    content = brief(fmt, topic, idx, series_ref)
    if feedback:
        content += ("\n\nREVISION. A reviewer scored your previous draft (below) and asked for at least 4/5 on "
                    "every axis. Keep what worked, fix the weak axes (especially hook strength and game "
                    f"accuracy), and resubmit a full spec.\nReview: {json.dumps(feedback[1])}\n"
                    f"Previous draft: {json.dumps(feedback[0], ensure_ascii=False)}")
    messages = [{"role": "user", "content": content}]
    errors = []
    for attempt in range(1, base.MAX_ATTEMPTS + 1):
        data, resp = _ask(client, messages)
        errors = SS.validate(SS.resolve(data))
        prem = data.get("premise", "")
        if not errors and any(base.too_similar(prem, p, 0.45) for p in premises):
            errors = ["premise too similar to an existing video; pick a different angle"]
        if not errors:
            return data, attempt
        tid = next(b.id for b in resp.content if b.type == "tool_use")
        messages += [{"role": "assistant", "content": resp.content},
                     {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "is_error": True,
                                                   "content": "Rejected by validator:\n- " + "\n- ".join(errors)
                                                              + "\nFix and resubmit."}]}]
    raise SS.StorySpecError([f"gave up after {base.MAX_ATTEMPTS} attempts"] + errors)


JUDGE = """Score this short-video storyboard 1-5 on: hook (stops the scroll in 2s?),
comedy_or_drama (does the payoff land?), game_accuracy (legal, plausible moves),
clarity (understandable from the captions alone), brand_fit (playful; makes you want
to play the AI at gghuman.ai). Reply ONLY with JSON:
{"hook":n,"comedy_or_drama":n,"game_accuracy":n,"clarity":n,"brand_fit":n,"note":"..."}"""


def judge(client, data, min_score=4):
    resp = client.messages.create(model=base.MODEL, max_tokens=4000, messages=[
        {"role": "user", "content": JUDGE + "\n\n" + json.dumps(data, ensure_ascii=False)}])
    txt = next(b.text for b in resp.content if b.type == "text")  # skip thinking blocks
    scores = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
    return all(v >= min_score for k, v in scores.items() if k != "note"), scores


def _log_episode(series_ref, data):
    path = SS.SERIES_DIR / f"{series_ref}.json"
    ser = json.loads(path.read_text())
    score = {}
    for s in data["shots"]:
        for o in s.get("overlays", []):
            if o.get("type") == "scoreboard":
                score = {t["faction"]: t["score"] for t in o["teams"]}
            if o.get("type") == "title":
                title = o["title"]
    ser.setdefault("episodes", []).append({"ep": data["series"]["ep"], "title": locals().get("title", ""),
                                           "summary": data.get("premise", ""), "score": score})
    path.write_text(json.dumps(ser, ensure_ascii=False, indent=2))


def _save_rejected(out_dir, data, scores):
    rej = Path(out_dir) / "_rejected"
    rej.mkdir(parents=True, exist_ok=True)
    k = len(list(rej.glob("*.json")))
    (rej / f"{k:03d}_{data.get('id', 'draft')}.json").write_text(
        json.dumps({**data, "_judge": scores}, ensure_ascii=False, indent=2))


def generate(fmt, n, out_dir, series_ref="settler_rush", seed=None, use_judge=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [json.loads(p.read_text()) for p in out_dir.glob("*.json")]
    premises = [e.get("premise", "") for e in existing]
    client = base._client()
    results, idx = [], len(existing) + 1
    # fresh topic draw each run (pass seed for reproducibility); up to 4 candidates per video
    for topic in topics(fmt, n * 4, seed):
        if len(results) >= n:
            break
        try:
            data, attempts = generate_one(client, fmt, topic, idx, premises, series_ref)
            if use_judge:
                ok, scores = judge(client, data)
                if not ok:  # one revision round with the judge's notes
                    _save_rejected(out_dir, data, scores)
                    print(f"  judge wants a revision of {data['id']}: "
                          f"{ {k: v for k, v in scores.items() if k != 'note'} }")
                    data, attempts = generate_one(client, fmt, topic, idx, premises, series_ref,
                                                  feedback=(data, scores))
                    ok, scores = judge(client, data)
                data["_judge"] = scores
                if not ok:
                    _save_rejected(out_dir, data, scores)
                    print(f"  judge rejected {data['id']}: { {k: v for k, v in scores.items() if k != 'note'} }")
                    continue
            data["_topic"], data["_attempts"] = list(topic), attempts
            (out_dir / f"{data['id']}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))
            if fmt == "cinematic":
                _log_episode(series_ref, data)
            premises.append(data.get("premise", ""))
            results.append(data)
            idx += 1
            print(f"  ok {data['id']} (attempts={attempts})")
        except Exception as ex:  # one bad idea never stops the batch
            print(f"  skip {topic}: {ex}")
    return results
