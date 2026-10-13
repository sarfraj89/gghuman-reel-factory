"""Storyboard spec -> MP4. voice -> timeline -> (picture, captions, sound) -> ffmpeg."""
import json
import subprocess
from pathlib import Path

from .. import draw as D
from ..audio import master_filter
from . import sound, timeline, voice
from .spec import check, resolve

CAPTION_STYLE = {"doodle": dict(max_words=1, max_chars=12), "cinematic": dict(max_words=3, max_chars=18)}


def build(spec):
    spec = check(resolve(spec))
    clips = voice.synthesize_all(spec)
    tl = timeline.build(spec, clips, **CAPTION_STYLE[spec["format"]])
    if spec["format"] == "doodle":
        from .doodle import DoodleRenderer as R
    else:
        from .cinematic import CinematicRenderer as R
    return R(spec, tl), tl


def render(spec, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rend, tl = build(spec)
    spec = rend.spec
    wav = out_path.with_suffix(".wav")
    sound.mix(spec, tl, wav)
    n_frames = int(round(tl.total * D.FPS))
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{D.W}x{D.H}", "-r", str(D.FPS), "-i", "-",
           "-i", str(wav),
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
           "-af", master_filter(wav), "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
           "-shortest", "-movflags", "+faststart", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for i in range(n_frames):
            proc.stdin.write(rend.frame(i).convert("RGB").tobytes())
    finally:
        proc.stdin.close()
        if hasattr(rend, "close"):
            rend.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg encode failed")
    wav.unlink(missing_ok=True)
    preview_copy(out_path)
    out_path.with_suffix(".caption.txt").write_text(
        spec["caption"] + "\n\n" + " ".join(spec.get("hashtags", [])) + "\n")
    out_path.with_suffix(".timeline.json").write_text(json.dumps({
        "total": tl.total,
        "content_end": getattr(rend, "content_end", tl.total),
        "sheet_times": rend.sheet_times(),
        "placeholders": getattr(rend, "placeholders", 0),
        "shots": [{"start": s.start, "dur": s.dur} for s in tl.shots],
        "captions": [[c.start, c.end, c.text] for c in tl.captions],
    }, indent=1))
    return out_path


def preview_copy(path):
    """VS Code's built-in player can't decode AAC (the codec Instagram needs), so
    it plays the upload file silently. Write <id>.preview.mp4 with Opus audio for
    local review; upload the plain <id>.mp4. Disable with GG_PREVIEW=0."""
    import os
    if os.environ.get("GG_PREVIEW", "1") == "0":
        return None
    out = Path(path).with_suffix(".preview.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-c:v", "copy",
                    "-c:a", "libopus", "-b:a", "128k", "-ar", "48000", "-movflags", "+faststart", str(out)],
                   check=True)
    return out
