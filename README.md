# GGhuman Reel Factory — "Said vs. Thought"

Automated pipeline that produces vertical (1080×1920, 30 fps) Instagram Reels for
[gghuman.ai](https://gghuman.ai) in a new format, and can scale to 100+ videos
unattended.

## 1. The format, and why it should beat what the page posts today

Current @gghumanai posts are either (a) static doodle comics (blob human vs robot
at a Catan table, meme caption) or (b) one-off photoreal AI clips (robots walking
through a Catan village, dice falling from the sky). Both look nice, but neither
shows the site's actual selling point: **the AI opponents bluff, lie and betray**.

"Said vs. Thought" makes that the whole story, in ~17 s:

| Beat | What's on screen |
|---|---|
| 0–2 s Hook | Big kinetic text: "I asked **Claude** for a fair trade." Board/table animates in. |
| Chat | You ask the AI something; it types back something friendly (**SAID**). |
| Reveal | "WHAT IT WAS ACTUALLY THINKING" — its inner monologue decodes from glitch characters. |
| Event | The betrayal happens on the board: robber drops on your 8 / you fold and it flips 7-2 / the vote piles onto you. |
| Twist | One punchy line slams in: "It robbed my 8." |
| End card | "It trades. It smiles. **Can you win?**" → GGhuman wordmark → PLAY NOW → gghuman.ai |

Why this is better for the page:
- **Product-true**: shows the exact thing you can only get on gghuman.ai (AIs with hidden intentions).
- **Works with sound off** (most Reels are watched muted) and has a looped-curiosity gap (what is it thinking?).
- **Infinite but consistent**: game × opponent × deception × hook = 540 combos, all in one recognizable brand style.
- **Direct CTA** every video + URL visible in the top bar the whole time.

Samples in `out/`: Settler Rush vs Claude, Poker vs GPT, Last Vote vs Gemini.

## 2. Architecture

```
topic matrix ─► LLM writes spec (forced JSON tool) ─► validator ─┐
      ▲                        ▲                                  │ errors fed back (≤3 tries)
      │                        └──────────────────────────────────┘
      │                     LLM judge (rubric ≥4/5) ─► dedupe ─► specs/*.json
      │
      └── manifest.csv ◄── QC (ffprobe, loudness, black/freeze, contact sheet) ◄── deterministic renderer
```

The key design decision: **AI does the creative writing; code does the visuals.**
The LLM never generates pixels, so the render step cannot produce a melted hand,
wrong logo or unreadable text. All creative variance lives in a ~15-field JSON
spec that is validated before anything renders. That's what makes it repeatable.

| File | Role |
|---|---|
| `pipeline/spec.py` | Spec contract: fields, character limits, banned phrases, game-legality checks (valid cards, no duplicates, legal dice numbers, seat rules). Also the JSON schema given to the LLM. |
| `pipeline/ideate.py` | Topic matrix + Claude API call with a forced `submit_spec` tool, validate→retry loop with the exact errors, n-gram dedupe of hooks, LLM-as-judge rubric. |
| `pipeline/timeline.py` | All beat timings computed from text length (typing speed, reading speed). |
| `pipeline/scenes.py` | Game scenes: hex board + robber, poker table + card flip, mafia vote circle + vote tokens + role reveal. |
| `pipeline/render.py` | Frame compositor (Pillow, supersampled shapes), safe-zone layout, pipes frames into ffmpeg (H.264 + AAC, loudnorm to −14 LUFS). |
| `pipeline/audio.py` | Procedural music bed + SFX generated from the same timeline (typing ticks, decode glitch, impact, chimes). No licensing risk, no A/V drift. |
| `pipeline/qc.py` | Post-render checks + contact sheet per video. |
| `run.py` | CLI: `render`, `batch` (parallel, resumable), `ideate`, `factory` (end to end). |

## 3. Run it

```bash
pip install pillow numpy anthropic        # ffmpeg must be on PATH
python run.py render specs/001_settler_trade.json      # one video (~50–100 s on 1 core)
python run.py batch --specs specs --out out --workers 4

export ANTHROPIC_API_KEY=...              # only needed for the LLM steps
python run.py ideate --n 10 --out specs_gen
python run.py factory --n 100 --workers 8 # write 100 specs, render, QC, auto-repair
```

Each video produces `ID.mp4`, `ID.caption.txt` (caption + hashtags), `ID.qc.json`
and `ID.sheet.png` (6 key frames for a 5-second human review).

## 4. Checking output quality

Three gates, cheapest first:

1. **Spec gate (before render)** — schema, character limits (they *are* the layout
   safe zones), banned phrases (no "free", gambling-for-money, insults), no emoji,
   game legality (real card codes, no duplicate cards, no "7" as robber target,
   accused ≠ you/opponent), opponent named in hook/caption, URL in caption.
2. **Creative gate** — LLM judge scores hook, game accuracy, clarity, brand fit
   1–5; anything under 4 is dropped. Hooks with ≥50 % trigram overlap with an
   existing video are rejected to keep the feed fresh.
3. **Render gate (after render)** — ffprobe (1080×1920, 30 fps, H.264/yuv420p,
   audio present, 11–24 s), EBU R128 loudness (−16.5…−11.5 LUFS, true peak),
   ffmpeg `blackdetect`/`freezedetect` on the content section, and the renderer
   itself raises `LayoutError` if any text overflows its box (hook > 2 lines,
   bubble > 3 lines, chat past the Instagram caption zone at y=1540, right-hand
   button column kept clear).

Plus a **human-in-the-loop sample**: review the contact sheets (one PNG per video)
for the first batch of 100 and then ~10 % at random. After launch, feed back real
metrics (3-second hold, completion, profile taps, link clicks) per hook formula and
deception type, and reweight the topic matrix toward winners.

## 5. Handling failures

| Failure | Handling |
|---|---|
| LLM returns invalid/over-long JSON | Exact validator errors sent back as a `tool_result` with `is_error`; up to 3 attempts, then the idea is skipped (the batch keeps going). |
| Hook too similar to an existing one | Treated as a validation error → rewrite. |
| Judge score < 4 | Dropped; the topic matrix over-generates 2× candidates to compensate. |
| Text doesn't fit (LayoutError) | Never auto-shrunk. Spec goes back to the LLM once with the error ("thought wraps to 4 lines, max 3"), then re-rendered. |
| Render/ffmpeg crash | Caught per job, logged in `manifest.csv` with the traceback; other jobs continue. |
| QC fail (loudness, black frames, duration) | Marked `qc_failed`, never published; shows up in the review queue. |
| Crash mid-batch | `batch` is resumable: everything with status `ok` in the manifest is skipped. |
| API outage / rate limit | Specs are written to disk as they are generated; rerun continues where it stopped. |

## 6. Producing 100 videos

`python run.py factory --n 100 --workers 8`

- **Ideation**: ~200 LLM calls (100 specs + judge, plus retries) — a few minutes,
  cents to a few dollars depending on model.
- **Render**: ~25 ms/frame + encode + QC ≈ 1–2 min per video per core (measured:
  3 videos in 2m40s on 2 cores). 100 videos ≈ 15–25 min on an 8-core box, or fan out on
  any job queue (each job is a pure function `spec.json → mp4`).
- **Publishing** (next step, not included): Instagram Graph API content publishing
  (create a REELS media container from a hosted MP4 + caption, poll status, publish),
  scheduled 1–3/day from the manifest; only `status == ok` and human-approved rows.

## 7. Upgrades I'd make with access to gghuman.ai

- **Use real game logs.** The site already records frontier models' actual moves
  and reasoning. Feeding real "said" chat lines and real reasoning traces into the
  spec instead of invented ones would make every video a true story ("this actually
  happened on GGhuman") — much stronger and more defensible.
- **AI voiceover** (e.g. an ElevenLabs/TTS voice per opponent reading the SAID line
  and a whispered THOUGHT) — slots into `audio.py`; QC already measures loudness.
- **AI-generated backgrounds** per game (image model, cached, brand-checked) behind
  the scenes for variety, keeping text/board rendered by code.
- **A/B variants**: render 2 hooks per spec and let the first 24 h of reach pick.

## 8. Storyboard formats: "doodle" skits and "cinematic" series

Two more formats, reverse-engineered from the page's best-performing reels
(slow-roll doodle; Settler Rush EP 1 / EP 3). Recreations of all three live in
`specs_story/ref_*.json` and double as the LLM's few-shot examples.

| Format | Look | Footage | Typical length |
|---|---|---|---|
| `doodle` | cream paper, hand-drawn green human vs TV-head robot in the AI's brand colour, "POV:" hook, 1-word outlined captions, GGhuman logo bottom-left | 100 % procedural (wobbly pen + line boil, expressions, lip-flap) - no API cost | 15-25 s |
| `cinematic` | photoreal episodic "Settler Rush": EP title card, faction-styled captions (serif for humans, Orbitron in the AI's colour), HUD (faction tag, resource equation, counters, stolen, alert, LARGEST ARMY banner, scoreboard), series end card | generated (fal.ai Flux -> Kling image-to-video), or your own clips | 25-45 s |

A spec is a list of **shots**; each has a `visual`, voiced `lines`
(`who`/`say`/`tone`/`expression`), `sfx`, `ambience`, `overlays`, `transition`.
Voice is synthesised first and **shot lengths come from the real voice
durations**; captions come from per-word timings. See `pipeline/story/spec.py`
for the full vocabulary and validator.

```bash
# render (any spec; run.py routes by "format")
python run.py render specs_story/ref_doodle_slowroll.json --out out_story
python run.py batch --specs specs_story --out out_story --workers 3

# new skits / next episodes written by Claude (ANTHROPIC_API_KEY)
python run.py story-ideate --format doodle --n 5
python run.py story-factory --format cinematic --n 2 --series settler_rush
```

**Providers** (all optional; the pipeline always runs without keys):

| What | Env | Fallback without key |
|---|---|---|
| Voices | `ELEVENLABS_API_KEY` (best, with word timestamps) or `OPENAI_API_KEY` (gpt-4o-mini-tts) | local Piper voices in `assets/voices/` |
| SFX | `ELEVENLABS_API_KEY` (sound-generation from the cue prompt) | procedural cue library |
| Music | drop tracks in `assets/music/<mood>/` (quirky, tense, epic, dark, triumph) | procedural bed per mood |
| Cinematic footage | `FAL_KEY` (+ `GG_IMAGE_MODEL`, `GG_VIDEO_MODEL`; `GG_VIDEO=0` = stills + Ken Burns, ~10x cheaper) | labelled placeholders, status `draft` |

The local Piper voices are not in git (63-120 MB each). Download them once:

```bash
mkdir -p assets/voices && cd assets/voices
for v in en/en_US/ryan/high/en_US-ryan-high en/en_US/lessac/medium/en_US-lessac-medium en/en_GB/alan/medium/en_GB-alan-medium; do
  curl -sfLO "https://huggingface.co/rhasspy/piper-voices/resolve/main/$v.onnx"
  curl -sfLO "https://huggingface.co/rhasspy/piper-voices/resolve/main/$v.onnx.json"
done
```

Override voices per role with `GG_VOICE_<ROLE>=<voice id>` (e.g. a cloned voice
for `ROBOT_CALM`). You can also bypass generation per shot with
`"visual": {"clip": "assets/clips/x.mp4"}` or `{"image": "..."}`.
All generated media, voices and SFX are cached in `cache/` by content hash, so
re-renders after a caption fix cost nothing.

Cinematic series continuity lives in `series/<name>.json` (style bible, faction
looks/colours/fonts, episode log with scores). `story-ideate --format cinematic`
writes the next EP number and appends it to the log.

QC adds a `draft` status (placeholder footage) and checks the end card is
excluded from black-frame detection. Note: snap-packaged ffmpeg cannot read
`/tmp` or hidden directories; keep `out*/` and `cache/` inside the project.
