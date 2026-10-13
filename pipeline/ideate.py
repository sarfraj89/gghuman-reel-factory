"""Creative step: an LLM writes video specs. This is the only non-deterministic
stage, so it is fenced in on both sides:

  topic matrix  ->  LLM (forced JSON tool)  ->  validate  ->  retry with errors
                                               -> dedupe  ->  LLM judge (rubric)

Requires ANTHROPIC_API_KEY. Model is configurable with GG_MODEL.
"""
import itertools
import json
import os
import random
import re
from pathlib import Path

from . import spec as S

MODEL = os.environ.get("GG_MODEL", "claude-sonnet-5-5")
MAX_ATTEMPTS = 3

# What makes each video different. 3 games x 9 opponents x 4 deceptions x 5
# hook formulas = 540 combinations, so 100 videos never need to repeat a combo.
DECEPTIONS = {
    "settler_rush": ["fake friendly trade", "promised not to block your road",
                     "claimed it had no ore then built a city", "fake alliance against the leader"],
    "poker": ["bluff with total air", "slow-played the nuts", "fake tell / table talk",
              "claimed it 'never bluffs'"],
    "last_vote": ["swore loyalty then flipped the vote", "framed an innocent player",
                  "fake detective claim", "played dumb all game"],
}
HOOKS = ["'I asked X for ...'", "'X swore it ...'", "'POV: X is being nice to you'",
         "'X said it would never ...'", "'Day N of trusting X'"]

SYSTEM = """You write scripts for 15-second vertical Instagram Reels promoting gghuman.ai,
where people play board games (Settler Rush = Catan-like, no-limit hold'em poker,
Last Vote = mafia / Blood on the Clocktower-like) against frontier AI models that
can bluff, lie and cooperate.

Format "Said vs Thought": a hook line; the human asks the AI something in chat;
the AI replies with something friendly/confident (SAID); then its hidden inner
monologue is revealed (THOUGHT) and it is calculating and game-accurate; then the
betrayal happens on the board (TWIST, punchy, past tense, under 30 chars).

Rules:
- Every move must be legal and plausible in that game. Thoughts should sound
  like sharp strategic reasoning, not cartoon evil.
- Playful, never mean. No insults, no gambling-for-money language, no 'free'.
- Respect every character limit exactly. No emoji.
- Call the tool `submit_spec` exactly once."""


def _few_shot():
    ex = [json.loads(p.read_text()) for p in sorted(Path(__file__).parent.parent.glob("specs/00[1-3]_*.json"))]
    return json.dumps(ex, ensure_ascii=False, indent=1)


def topic_matrix(n, seed=0, exclude=()):
    rng = random.Random(seed)
    combos = [(g, o, d, h) for g in S.GAMES for o in S.ROSTER
              for d in DECEPTIONS[g] for h in HOOKS]
    rng.shuffle(combos)
    done = set(exclude)
    out = [c for c in combos if c[:3] not in done]
    # round-robin games so the feed stays varied
    by_game = {g: [c for c in out if c[0] == g] for g in S.GAMES}
    mixed = [c for trio in itertools.zip_longest(*by_game.values()) for c in trio if c]
    return mixed[:n]


def _client():
    import anthropic  # pip install anthropic
    return anthropic.Anthropic()


def _ask(client, messages):
    resp = client.messages.create(
        model=MODEL, max_tokens=16000, system=SYSTEM, messages=messages,
        tools=[{"name": "submit_spec", "description": "Submit one reel spec.",
                "input_schema": S.json_schema()}],
        tool_choice={"type": "auto"},  # forced tool_choice is rejected by current models
    )
    block = next((b for b in resp.content if b.type == "tool_use"), None)
    if block is None:
        raise RuntimeError("model did not call submit_spec")
    return block.input, resp


def _ngrams(s, n=3):
    w = re.findall(r"[a-z0-9]+", s.lower())
    return {tuple(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


def too_similar(a, b, thr=0.5):
    x, y = _ngrams(a), _ngrams(b)
    return bool(x and y) and len(x & y) / len(x | y) >= thr


def generate_one(client, idx, combo, existing_hooks):
    game, opp, deception, hook_formula = combo
    brief = (f"Write spec id '{idx:03d}_<slug>'. game={game}, opponent={opp}, "
             f"deception='{deception}', hook formula={hook_formula}.\n"
             f"Limits: {json.dumps(S.LIMITS)}.\n"
             f"Scene schema by game: settler_rush -> {{target_number, loss, seed}}; "
             f"poker -> {{board[5], you_hand[2], ai_hand[2], pot}} with cards like 'K♥'; "
             f"last_vote -> {{seats (start with 'YOU', roster names), accused, role_reveal, day}}.\n"
             f"Examples:\n{_few_shot()}")
    messages = [{"role": "user", "content": brief}]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        data, resp = _ask(client, messages)
        errors = S.validate(data)
        if not errors and any(too_similar(data["hook"], h) for h in existing_hooks):
            errors = ["hook too similar to an existing video; write a different angle"]
        if not errors:
            return data, attempt
        # feed the exact errors back - this fixes ~all failures by attempt 2
        messages += [{"role": "assistant", "content": resp.content},
                     {"role": "user", "content": [{
                         "type": "tool_result", "tool_use_id": next(b.id for b in resp.content if b.type == "tool_use"),
                         "is_error": True,
                         "content": "Rejected by validator:\n- " + "\n- ".join(errors) + "\nFix and resubmit."}]}]
    raise S.SpecError([f"gave up after {MAX_ATTEMPTS} attempts"] + errors)


JUDGE = """Score this reel script 1-5 on: hook (would it stop a scroll?),
game_accuracy (are the moves legal and the thought strategically sound?),
clarity (understandable with sound off in 15s?), brand_fit (playful, makes you
want to try beating the AI at gghuman.ai?). Reply ONLY with JSON:
{"hook":n,"game_accuracy":n,"clarity":n,"brand_fit":n,"note":"..."}"""


def judge(client, data, min_score=4):
    resp = client.messages.create(model=MODEL, max_tokens=4000, messages=[
        {"role": "user", "content": JUDGE + "\n\n" + json.dumps(data, ensure_ascii=False)}])
    txt = next(b.text for b in resp.content if b.type == "text")  # skip thinking blocks
    scores = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
    ok = all(scores[k] >= min_score for k in ("hook", "game_accuracy", "clarity", "brand_fit"))
    return ok, scores


def generate(n, out_dir, start_index=4, seed=0, use_judge=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [json.loads(p.read_text()) for p in out_dir.glob("*.json")]
    hooks = [e["hook"] for e in existing]
    done = {(e["game"], e["opponent"], e.get("_deception")) for e in existing}
    client = _client()
    results = []
    idx = start_index + len(existing)
    for combo in topic_matrix(n * 2, seed, done):  # 2x candidates: some get judged out
        if len(results) >= n:
            break
        try:
            data, attempts = generate_one(client, idx, combo, hooks)
            if use_judge:
                ok, scores = judge(client, data)
                data["_judge"] = scores
                if not ok:
                    print(f"  judge rejected {data['id']}: {scores}")
                    continue
            data["_deception"] = combo[2]
            data["_attempts"] = attempts
            (out_dir / f"{data['id']}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))
            hooks.append(data["hook"])
            results.append(data)
            idx += 1
            print(f"  ok {data['id']} (attempts={attempts})")
        except Exception as ex:  # one bad idea never stops the batch
            print(f"  skip {combo}: {ex}")
    return results
