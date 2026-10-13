"""Automated quality checks on a rendered MP4, plus a contact sheet for the
human spot-check. Gate 1 (spec validation) and gate 2 (layout errors) happen
before rendering; this is gate 3."""
import json
import re
import subprocess
from pathlib import Path

from PIL import Image

from . import timeline
from .draw import FFPROBE

EXPECT = dict(width=1080, height=1920, fps=30.0, min_dur=11.0, max_dur=24.0,
              lufs=(-16.5, -11.5), max_true_peak=-0.5)


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def probe(path):
    r = _run([FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)])
    return json.loads(r.stdout)


def loudness(path):
    r = _run(["ffmpeg", "-nostats", "-i", str(path), "-af", "ebur128=peak=true", "-f", "null", "-"])
    txt = r.stderr
    i = re.findall(r"I:\s+(-?[\d.]+) LUFS", txt)
    p = re.findall(r"Peak:\s+(-?[\d.]+) dBFS", txt)
    return (float(i[-1]) if i else None), (float(p[-1]) if p else None)


def black_and_freeze(path, until):
    r = _run(["ffmpeg", "-nostats", "-t", f"{until:.2f}", "-i", str(path), "-vf",
              "blackdetect=d=0.4:pic_th=0.97,freezedetect=n=0.002:d=3.5", "-an", "-f", "null", "-"])
    blacks = re.findall(r"black_start:([\d.]+)", r.stderr)
    freezes = re.findall(r"freeze_start: ([\d.]+)", r.stderr)
    return [float(b) for b in blacks], [float(f) for f in freezes]


def contact_sheet(path, times, out_png, cols=6):
    frames = []
    tmp = Path(out_png).with_suffix(".tmp.png")
    for t in times:
        _run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(path),
              "-frames:v", "1", "-vf", "scale=270:480", str(tmp)])
        frames.append(Image.open(tmp).copy())
    tmp.unlink(missing_ok=True)
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 270, rows * 480), (0, 0, 0))
    for i, f in enumerate(frames):
        sheet.paste(f, ((i % cols) * 270, (i // cols) * 480))
    sheet.save(out_png)
    return out_png


def run(path, spec):
    """Return (passed: bool, report: dict). Hard failures block publishing;
    warnings go to the human review queue."""
    tl = timeline.build(spec)
    fails, warns = [], []
    info = probe(path)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    dur = float(info["format"]["duration"])
    if not v:
        fails.append("no video stream")
    else:
        if (v["width"], v["height"]) != (EXPECT["width"], EXPECT["height"]):
            fails.append(f"resolution {v['width']}x{v['height']}")
        num, den = map(int, v["r_frame_rate"].split("/"))
        if abs(num / den - EXPECT["fps"]) > 0.01:
            fails.append(f"fps {num / den:.2f}")
        if v.get("pix_fmt") != "yuv420p" or v.get("codec_name") != "h264":
            fails.append("must be h264/yuv420p for Instagram")
    if not a:
        fails.append("no audio stream")
    if not (EXPECT["min_dur"] <= dur <= EXPECT["max_dur"]):
        fails.append(f"duration {dur:.1f}s outside {EXPECT['min_dur']}-{EXPECT['max_dur']}s")
    if a:
        lufs, peak = loudness(path)
        if lufs is None or not (EXPECT["lufs"][0] <= lufs <= EXPECT["lufs"][1]):
            fails.append(f"loudness {lufs} LUFS outside {EXPECT['lufs']}")
        if peak is not None and peak > EXPECT["max_true_peak"]:
            warns.append(f"true peak {peak} dBFS")
    else:
        lufs = peak = None
    blacks, freezes = black_and_freeze(path, tl["end"])
    if blacks:
        fails.append(f"black frames at {blacks}")
    if freezes:
        warns.append(f"frozen picture at {freezes}s (pacing too slow?)")
    if tl["you"] > 3.0:
        warns.append("first chat line after 3s: weak hook pacing")

    sheet_times = [0.8, tl["said_end"] + 0.2, tl["thought_end"] + 0.2,
                   tl["event"] + 1.0, tl["twist"] + 0.5, tl["total"] - 0.6]
    sheet = contact_sheet(path, sheet_times, Path(path).with_suffix(".sheet.png"))
    report = dict(id=spec["id"], file=str(path), duration=round(dur, 2), lufs=lufs,
                  true_peak=peak, fails=fails, warnings=warns, contact_sheet=str(sheet),
                  passed=not fails)
    Path(path).with_suffix(".qc.json").write_text(json.dumps(report, indent=2))
    return not fails, report


STORY_DUR = {"doodle": (10.0, 35.0), "cinematic": (18.0, 62.0)}


def run_story(path, spec):
    """QC for storyboard formats (doodle / cinematic). Uses the timeline.json
    written by the renderer for sheet times and the content/end-card boundary."""
    tlj = json.loads(Path(path).with_suffix(".timeline.json").read_text())
    fails, warns = [], []
    info = probe(path)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    dur = float(info["format"]["duration"])
    if not v:
        fails.append("no video stream")
    else:
        if (v["width"], v["height"]) != (EXPECT["width"], EXPECT["height"]):
            fails.append(f"resolution {v['width']}x{v['height']}")
        if v.get("pix_fmt") != "yuv420p" or v.get("codec_name") != "h264":
            fails.append("must be h264/yuv420p for Instagram")
    lo, hi = STORY_DUR[spec["format"]]
    if not lo <= dur <= hi:
        fails.append(f"duration {dur:.1f}s outside {lo}-{hi}s")
    lufs = peak = None
    if not a:
        fails.append("no audio stream")
    else:
        lufs, peak = loudness(path)
        if lufs is None or not (EXPECT["lufs"][0] <= lufs <= EXPECT["lufs"][1]):
            fails.append(f"loudness {lufs} LUFS outside {EXPECT['lufs']}")
        if peak is not None and peak > EXPECT["max_true_peak"]:
            warns.append(f"true peak {peak} dBFS")
    blacks, freezes = black_and_freeze(path, tlj["content_end"])
    if blacks:
        fails.append(f"black frames at {blacks}")
    if freezes:
        warns.append(f"frozen picture at {freezes}s")
    first = tlj["captions"][0][0] if tlj["captions"] else 99
    if first > 1.5:
        warns.append(f"first spoken caption at {first:.1f}s: weak hook")
    if tlj.get("placeholders"):
        fails.append(f"draft: {tlj['placeholders']} placeholder shots (set FAL_KEY or add clips)")
    sheet = contact_sheet(path, tlj["sheet_times"][:18], Path(path).with_suffix(".sheet.png"))
    report = dict(id=spec["id"], file=str(path), duration=round(dur, 2), lufs=lufs, true_peak=peak,
                  fails=fails, warnings=warns, contact_sheet=str(sheet), passed=not fails)
    Path(path).with_suffix(".qc.json").write_text(json.dumps(report, indent=2))
    return not fails, report
