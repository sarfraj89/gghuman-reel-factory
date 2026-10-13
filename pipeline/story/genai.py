"""Generated footage for cinematic shots.

Per shot, the first available source wins:
  1. visual.clip   your own video file (e.g. made in Kling/Veo/Higgsfield by hand)
  2. visual.image  your own still (animated with a Ken Burns move)
  3. provider      GG_GENAI=fal (default when FAL_KEY is set):
                   text-to-image (GG_IMAGE_MODEL) -> image-to-video (GG_VIDEO_MODEL)
                   GG_VIDEO=0 keeps stills only (about 10x cheaper; Ken Burns motion)
     GG_GENAI=cloudflare: Workers AI text-to-image (free daily allowance),
                   stills only -> Ken Burns motion. CLOUDFLARE_ACCOUNT_ID +
                   CLOUDFLARE_API_TOKEN (token needs Workers AI permission).
                   GG_CF_IMAGE_MODEL overrides the model list below.
     GG_GENAI=pollinations: free, no key (low-res ~580x1015, small watermark
                   that is cropped off). Also the automatic fallback when
                   Cloudflare refuses (e.g. token without Workers AI access).
     Motion for free providers: shots with visual.motion are animated by a free
                   Wan 2.2 image-to-video Hugging Face Space (GG_HF_VIDEO_SPACE),
                   ~50 s per 3.5 s clip on shared ZeroGPU. Anonymous quota is
                   small; set HF_TOKEN (free account) for more. When the quota
                   runs out the shot falls back to the still + camera move.
  4. stub          labelled placeholder frame, so timing, captions, HUD and sound
                   can be reviewed before paying for generation. QC marks these
                   videos as drafts.

Everything is cached under cache/genai/ by a hash of (model, prompt, seed), so
re-rendering an episode (e.g. after fixing a caption) costs nothing.
"""
import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import text as T
from .voice import ROOT

CACHE = ROOT / "cache" / "genai"
IMAGE_MODEL = os.environ.get("GG_IMAGE_MODEL", "fal-ai/flux-pro/v1.1-ultra")
VIDEO_MODEL = os.environ.get("GG_VIDEO_MODEL", "fal-ai/kling-video/v2.1/master/image-to-video")
NEGATIVE = "text, captions, watermark, logo text, subtitles, blurry, deformed hands, extra limbs"


def provider():
    if os.environ.get("GG_GENAI"):
        return os.environ["GG_GENAI"]
    if os.environ.get("FAL_KEY"):
        return "fal"
    if os.environ.get("CLOUDFLARE_API_TOKEN"):
        return "cloudflare"
    return "stub"


# Portrait-capable models first; flux-1-schnell is square-only (centre-cropped).
CF_MODELS = [m for m in [os.environ.get("GG_CF_IMAGE_MODEL"), "@cf/leonardo/lucid-origin",
                         "@cf/leonardo/phoenix-1.0", "@cf/black-forest-labs/flux-1-schnell",
                         "@cf/bytedance/stable-diffusion-xl-lightning"] if m]
_CF_DEAD = set()


def cloudflare_image(prompt, seed):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"cf_{_key(CF_MODELS, prompt, seed)}.jpg"
    if path.exists():
        return path
    acct, tok = os.environ["CLOUDFLARE_ACCOUNT_ID"], os.environ["CLOUDFLARE_API_TOKEN"]
    errors = []
    for model in CF_MODELS:
        if model in _CF_DEAD:
            continue
        for body in ({"prompt": prompt[:2000], "width": 768, "height": 1344, "seed": seed, "steps": 8},
                     {"prompt": prompt[:2000], "seed": seed, "steps": 8}):
            req = urllib.request.Request(
                f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/{model}",
                data=json.dumps(body).encode(), method="POST",
                headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    raw, ctype = r.read(), r.headers.get("Content-Type", "")
            except urllib.error.HTTPError as e:
                msg = e.read()[:300].decode("utf-8", "replace")
                errors.append(f"{model}: HTTP {e.code} {msg}")
                if e.code in (401, 403):
                    raise RuntimeError("Cloudflare token lacks Workers AI permission: " + msg)
                if e.code == 404 or "No such model" in msg:
                    _CF_DEAD.add(model)
                    break
                continue
            if ctype.startswith("application/json"):
                res = json.loads(raw).get("result") or {}
                if not res.get("image"):
                    errors.append(f"{model}: no image in response")
                    continue
                raw = base64.b64decode(res["image"])
            from io import BytesIO
            Image.open(BytesIO(raw)).convert("RGB").save(path, quality=92)
            return path
    raise RuntimeError("cloudflare image failed: " + " | ".join(errors[-4:]))


def _key(*parts):
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def _http(url, body=None, method=None):
    hdr = {"Authorization": f"Key {os.environ['FAL_KEY']}", "Content-Type": "application/json"}
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=hdr, method=method or ("POST" if data else "GET"))
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def fal_run(model, payload, timeout=1200):
    """Submit to the fal.ai queue and poll until done."""
    sub = _http(f"https://queue.fal.run/{model}", payload)
    status_url, resp_url = sub["status_url"], sub["response_url"]
    t0 = time.time()
    while True:
        st = _http(status_url)
        if st.get("status") == "COMPLETED":
            return _http(resp_url)
        if st.get("status") not in ("IN_QUEUE", "IN_PROGRESS"):
            raise RuntimeError(f"fal {model} failed: {st}")
        if time.time() - t0 > timeout:
            raise TimeoutError(f"fal {model} timed out")
        time.sleep(4)


def _download(url, path):
    with urllib.request.urlopen(url, timeout=300) as r:
        path.write_bytes(r.read())
    return path


def _data_uri(path):
    im = Image.open(path).convert("RGB")
    im.thumbnail((1080, 1920))
    from io import BytesIO
    buf = BytesIO()
    im.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def compose_prompt(v, series):
    parts = [v["prompt"]]
    for f in v.get("factions", []):
        look = series.get("factions", {}).get(f, {}).get("look")
        if look:
            parts.append(look)
    parts.append(series.get("style", ""))
    parts.append("vertical 9:16 composition, cinematic, no text")
    return ". ".join(p.strip(" .") for p in parts if p)


def image(prompt, seed):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"img_{_key(IMAGE_MODEL, prompt, seed)}.jpg"
    if path.exists():
        return path
    res = fal_run(IMAGE_MODEL, {"prompt": prompt, "aspect_ratio": "9:16", "seed": seed,
                                "output_format": "jpeg", "safety_tolerance": "2", "num_images": 1})
    return _download(res["images"][0]["url"], path)


def video(img_path, motion, seed, seconds):
    path = CACHE / f"vid_{_key(VIDEO_MODEL, str(img_path.name), motion, seconds)}.mp4"
    if path.exists():
        return path
    payload = {"prompt": motion, "image_url": _data_uri(img_path)}
    if "kling" in VIDEO_MODEL:
        payload.update(duration="10" if seconds > 5.5 else "5", aspect_ratio="9:16", negative_prompt=NEGATIVE)
    res = fal_run(VIDEO_MODEL, payload)
    return _download(res["video"]["url"], path)


def placeholder(prompt, label):
    """Labelled stand-in still: tinted gradient + the prompt, for draft review."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"stub_{_key(prompt, label)}.jpg"
    if path.exists():
        return path
    h = int(hashlib.sha1(prompt.encode()).hexdigest()[:6], 16)
    rng = np.random.default_rng(h)
    c0 = rng.integers(20, 90, 3)
    c1 = rng.integers(90, 200, 3)
    y = np.linspace(0, 1, 1920)[:, None, None]
    img = (c0 * (1 - y) + c1 * y) * np.ones((1, 1080, 1))
    img += rng.normal(0, 4, img.shape)
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2))
    d = ImageDraw.Draw(im)
    for x in range(0, 1080, 120):
        d.line([(x, 0), (x, 1920)], fill=(255, 255, 255, 40), width=1)
    for yy in range(0, 1920, 120):
        d.line([(0, yy), (1080, yy)], fill=(255, 255, 255, 40), width=1)
    im = im.convert("RGBA")
    lab = T.sprite(f"PLACEHOLDER  {label}", "tech", 34, (255, 255, 255), wght=700, tracking=0.1)
    im.alpha_composite(lab, ((1080 - lab.width) // 2, 700))
    try:
        lines = T.fit_lines(prompt[:260], "sans", 30, 900, wght=500, max_lines=8)
    except Exception:
        lines = [prompt[:60]]
    for k, ln in enumerate(lines):
        s = T.sprite(ln, "sans", 30, (235, 235, 235), wght=500)
        im.alpha_composite(s, ((1080 - s.width) // 2, 790 + k * 44))
    im.convert("RGB").save(path, quality=88)
    return path


def pollinations_image(prompt, seed):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"poll_{_key(prompt, seed)}.jpg"
    if path.exists():
        return path
    import urllib.parse
    url = ("https://image.pollinations.ai/prompt/" + urllib.parse.quote(prompt[:1500]) +
           f"?width=768&height=1344&seed={seed}&nologo=true&model=flux")
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                raw = r.read()
            from io import BytesIO
            im = Image.open(BytesIO(raw)).convert("RGB")
            im = im.crop((0, 0, im.width, int(im.height * 0.95)))  # drop the corner watermark
            im.save(path, quality=93)
            return path
        except Exception as e:  # free tier is rate limited; back off and retry
            last = e
            time.sleep(8 * (attempt + 1))
    raise RuntimeError(f"pollinations failed: {last}")


HF_VIDEO_SPACE = os.environ.get("GG_HF_VIDEO_SPACE", "zerogpu-aoti/wan2-2-fp8da-aoti-faster")
_HF = {"client": None, "dead": False}


def hf_video(img_path, motion, seed, seconds):
    """Free image-to-video via a Hugging Face ZeroGPU Space. Returns None when
    unavailable (quota, queue, outage) so the caller can fall back to a still."""
    secs = round(min(5.0, max(2.0, seconds + 0.2)), 1)
    path = CACHE / f"wan_{_key(HF_VIDEO_SPACE, str(img_path.name), motion, secs)}.mp4"
    if path.exists():
        return path
    if _HF["dead"] or os.environ.get("GG_VIDEO", "1") == "0":
        return None
    try:
        from gradio_client import Client, handle_file
        if _HF["client"] is None:
            _HF["client"] = Client(HF_VIDEO_SPACE, token=os.environ.get("HF_TOKEN") or None, verbose=False)
        res = _HF["client"].predict(
            input_image=handle_file(str(img_path)),
            prompt=f"{motion}, cinematic motion, smooth natural movement",
            steps=6, duration_seconds=secs, randomize_seed=False, seed=int(seed),
            api_name="/generate_video")
        out = res[0]["video"] if isinstance(res[0], dict) else res[0]
        import shutil
        shutil.copy(out, path)
        print(f"  animated {img_path.name} ({secs}s)")
        return path
    except Exception as e:
        msg = str(e)
        if any(k in msg.lower() for k in ("quota", "gpu", "exceeded", "rate", "429")):
            _HF["dead"] = True  # don't hammer the Space for the rest of this render
            print(f"  free video quota reached ({msg[:120]}); remaining shots use stills + camera move."
                  " Set HF_TOKEN for more quota.")
        else:
            print(f"  video generation failed for {img_path.name}: {msg[:160]}; using still")
        return None


def free_image(prompt, seed):
    """Cloudflare first (better quality); Pollinations if it is unavailable."""
    if provider() == "cloudflare":
        try:
            return cloudflare_image(prompt, seed)
        except RuntimeError as e:
            if "permission" not in str(e):
                raise
            global _CF_WARNED
            if not _CF_WARNED:
                print("  cloudflare unavailable (token lacks Workers AI) -> using pollinations")
                _CF_WARNED = True
    return pollinations_image(prompt, seed)


_CF_WARNED = False


def shot_media(v, series, seed, label, want_seconds):
    """Return dict(kind='video'|'image', path=Path, placeholder=bool)."""
    if v.get("clip"):
        return dict(kind="video", path=(ROOT / v["clip"]).resolve(), placeholder=False)
    if v.get("image"):
        return dict(kind="image", path=(ROOT / v["image"]).resolve(), placeholder=False)
    prompt = compose_prompt(v, series)
    if provider() == "stub":
        return dict(kind="image", path=placeholder(v["prompt"], label), placeholder=True)
    if provider() in ("cloudflare", "pollinations"):
        img = free_image(prompt, seed)
        if v.get("motion"):
            clip = hf_video(img, v["motion"], seed, want_seconds)
            if clip:
                return dict(kind="video", path=clip, placeholder=False)
        return dict(kind="image", path=img, placeholder=False)
    img = image(prompt, seed)
    if os.environ.get("GG_VIDEO", "1") != "0" and v.get("motion"):
        return dict(kind="video", path=video(img, v["motion"], seed, want_seconds), placeholder=False)
    return dict(kind="image", path=img, placeholder=False)
