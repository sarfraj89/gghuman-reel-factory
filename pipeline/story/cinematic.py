"""Cinematic series renderer ("Settler Rush" episodes).

Footage comes from genai.shot_media (generated, user-supplied or placeholder).
Everything with text (captions, EP title, HUD, end card) is drawn by code on
top, so it is always legible, on-brand and correctly spelled.
"""
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .. import draw as D
from ..draw import LayoutError
from . import genai
from . import text as T

W, H, FPS = D.W, D.H, D.FPS
CAPTION_Y = 1440
CREAM = (243, 231, 211)
GOLD = (201, 168, 106)
TEAL = (45, 212, 191)
PANEL = (8, 10, 12, 150)
HUMAN_CAPTION = dict(font="serif", size=58, wght=700, tracking=0.02)
AI_CAPTION = dict(font="tech", size=48, wght=600, tracking=0.12)


def rgb(c):
    return D.hex2rgb(c) if isinstance(c, str) else tuple(c)


# ======================================================================
# footage sources
# ======================================================================
def _probe_dur(path):
    r = subprocess.run([D.FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                        str(path)], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 5.0


class VideoSource:
    """Streams a clip conformed to 1080x1920@30 and exactly `dur` seconds.
    Short clips are slowed down (max 1.5x) and then hold their last frame."""

    def __init__(self, path, dur, trim=0.0):
        self.path, self.dur, self.trim = path, dur, trim
        self.proc, self.pos, self.last = None, 0, None

    def _open(self):
        clip = max(0.1, _probe_dur(self.path) - self.trim)
        slow = min(1.5, max(1.0, self.dur / clip))
        vf = (f"setpts={slow:.4f}*PTS,fps={FPS},scale={W}:{H}:force_original_aspect_ratio=increase,"
              f"crop={W}:{H},tpad=stop_mode=clone:stop_duration={self.dur + 1:.2f}")
        self.proc = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{self.trim:.2f}", "-i", str(self.path),
                                      "-vf", vf, "-t", f"{self.dur + 0.2:.2f}", "-an", "-f", "rawvideo",
                                      "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        self.pos = 0

    def frame(self, k):
        if self.proc is None or k < self.pos - 1:
            self.close()
            self._open()
        while self.pos <= k:
            buf = self.proc.stdout.read(W * H * 3)
            if len(buf) < W * H * 3:
                break
            self.last = Image.frombuffer("RGB", (W, H), buf, "raw", "RGB", 0, 1)
            self.pos += 1
        return self.last.copy() if self.last is not None else Image.new("RGB", (W, H))

    def close(self):
        if self.proc:
            self.proc.stdout.close()  # ffmpeg exits on the broken pipe by itself
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    self.proc.kill()
                except PermissionError:  # snap-confined ffmpeg rejects signals; it is exiting anyway
                    pass
            self.proc = None


class ImageSource:
    """Ken Burns move over a still."""
    MOVES = {"static": ((1.04, 0, 0), (1.04, 0, 0)), "push_in": ((1.0, 0, 0), (1.14, 0, 0)),
             "pull_out": ((1.14, 0, 0), (1.0, 0, 0)), "pan_left": ((1.12, 0.05, 0), (1.12, -0.05, 0)),
             "pan_right": ((1.12, -0.05, 0), (1.12, 0.05, 0)), "pan_up": ((1.12, 0, 0.05), (1.12, 0, -0.05)),
             "pan_down": ((1.12, 0, -0.05), (1.12, 0, 0.05))}

    def __init__(self, path, dur, camera="push_in"):
        im = Image.open(path).convert("RGB")
        s = max(W / im.width, H / im.height)
        self.im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
        self.dur, self.move = dur, self.MOVES.get(camera, self.MOVES["push_in"])

    def frame(self, k):
        u = D.ease_in_out(D.clamp(k / FPS / max(self.dur, 0.01)))
        (z0, x0, y0), (z1, x1, y1) = self.move
        z, ox, oy = z0 + (z1 - z0) * u, x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
        iw, ih = self.im.size
        cw, ch = iw / z, ih / z
        cx, cy = iw / 2 + ox * iw, ih / 2 + oy * ih
        box = (D.clamp(cx - cw / 2, 0, iw - cw), D.clamp(cy - ch / 2, 0, ih - ch))
        return self.im.resize((W, H), Image.BILINEAR, box=(box[0], box[1], box[0] + cw, box[1] + ch))

    def close(self):
        pass


# ======================================================================
# HUD widgets (all return RGBA sprites; positions in `place`)
# ======================================================================
def panel(w, h, alpha=150, radius=0):
    return D.rrect(int(w), int(h), radius, (8, 10, 12, alpha)) if radius else \
        Image.new("RGBA", (int(w), int(h)), (8, 10, 12, alpha))


def divider(w, col=GOLD, diamond=True):
    im = Image.new("RGBA", (w, 24), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.line([(0, 12), (w, 12)], fill=col + (200,), width=2)
    if diamond:
        c = w // 2
        d.polygon([(c, 4), (c + 8, 12), (c, 20), (c - 8, 12)], fill=col + (255,))
    return im


def brackets(w, h, col, arm=46, lw=5):
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = col + (255,)
    for x, y, dx, dy in ((0, 0, 1, 1), (w - 1, 0, -1, 1), (0, h - 1, 1, -1), (w - 1, h - 1, -1, -1)):
        d.line([(x, y + dy * arm), (x, y), (x + dx * arm, y)], fill=c, width=lw)
    return im


class Factions:
    def __init__(self, series):
        self.f = series.get("factions", {})

    def color(self, key):
        return rgb(self.f.get(key, {}).get("color", "#FFFFFF"))

    def font(self, key):
        return self.f.get(key, {}).get("font", "tech")

    def label(self, key):
        return self.f.get(key, {}).get("label", key.upper())

    def text(self, key, s, size, wght=None, tracking=None, **kw):
        if self.font(key) == "serif":
            return T.sprite(s, "serif", size, self.color(key), wght=wght or 700,
                            tracking=0.02 if tracking is None else tracking, shadow=4, **kw)
        return T.sprite(s, "tech", size, self.color(key), wght=wght or 600,
                        tracking=0.1 if tracking is None else tracking, shadow=3, **kw)


def build_overlay(o, fx, ep=None):
    """Return (sprite or list of staggered sprites, x, y, anchor)."""
    t = o["type"]
    if t == "title":
        ep_s = T.sprite(f"EP {o.get('ep', ep or 1)}", "serif", 34, CREAM, wght=600, tracking=0.2, shadow=3)
        title = o["title"].upper()
        size = 88  # shrink to one line down to 62px, then wrap
        while size > 62 and T.measure(title, T.font("serif", size, 700)) > 960:
            size -= 2
        lines = T.fit_lines(title, "serif", size, 960, wght=700, max_lines=2)
        rows = [T.sprite(ln, "serif", size, CREAM, wght=700, shadow=6) for ln in lines]
        top = T.hstack([divider(110, GOLD, False), ep_s, divider(110, GOLD, False)], gap=18)
        return T.stack([top] + rows, gap=0), W / 2, o.get("y", 380), "mt"
    if t == "faction_tag":
        name = fx.text(o["faction"], o.get("name", fx.label(o["faction"])).upper(), 46, wght=700)
        parts = [name]
        if o.get("badge"):
            b = T.sprite(o["badge"], "tech", 30, fx.color(o["faction"]), wght=700)
            box = Image.new("RGBA", (b.width + 22, b.height + 8), (0, 0, 0, 0))
            ImageDraw.Draw(box).rounded_rectangle([1, 1, box.width - 2, box.height - 2], 8,
                                                  outline=fx.color(o["faction"]) + (255,), width=3)
            box.alpha_composite(b, (11, 4))
            parts.append(box)
        top = T.hstack(parts, gap=16)
        lab = T.sprite(o.get("label", "").upper(), "tech", 20, (225, 225, 225), wght=500, tracking=0.25, shadow=3)
        return T.stack([top, lab], gap=-6, align="l"), 70, o.get("y", 250), "lt"
    if t == "equation":
        col = fx.color(o["faction"]) if o.get("faction") else (240, 200, 160)
        rows = [T.sprite(s.upper(), "tech", 36, col, wght=600, tracking=0.1) for s in o["items"]]
        res = T.sprite(o["result"].upper(), "tech", 36, col, wght=700, tracking=0.1) if o.get("result") else None
        return ("equation", rows, res, col), 0, o.get("y", 250), "lt"
    if t == "counter":
        v = T.sprite(o["value"], "tech", 64, (255, 255, 255), wght=800, shadow=3)
        lab = T.sprite(o.get("label", "").upper(), "tech", 16, (210, 210, 210), wght=500, tracking=0.2)
        body = T.stack([v, lab], gap=-10, align="r")
        bg = panel(body.width + 50, body.height + 20, 150)
        bg.alpha_composite(body, (bg.width - body.width - 22, 10))
        return bg, W, o.get("y", 300), "rt"
    if t == "stolen":
        rows = [T.sprite(o.get("label", "STOLEN"), "tech", 16, (210, 210, 210), wght=500, tracking=0.2)]
        rows += [T.sprite(s.upper(), "tech", 34, (255, 255, 255), wght=700, tracking=0.08) for s in o["items"]]
        body = T.stack(rows, gap=-6, align="r")
        bg = panel(body.width + 70, body.height + 26, 150)
        bg.alpha_composite(body, (bg.width - body.width - 24, 13))
        return bg, W, o.get("y", 1520), "rt"
    if t == "alert":
        txt = T.sprite(o["text"].upper(), "tech", 46, (255, 255, 255), wght=700, tracking=0.08, shadow=4)
        br = brackets(txt.width + 70, txt.height + 40, fx.color(o["faction"]) if o.get("faction") else (255, 255, 255))
        br.alpha_composite(txt, (35, 20))
        return br, W / 2, o.get("y", 880), "mm"
    if t == "banner":
        rows = [T.sprite(ln, "serif", 104, CREAM, wght=700, shadow=8)
                for ln in T.fit_lines(o["title"].upper(), "serif", 104, 980, wght=700, max_lines=2)]
        if o.get("sub"):
            rows.append(T.sprite(o["sub"], "serif", 62, CREAM, wght=700, shadow=6))
        return T.stack(rows, gap=-14), W / 2, o.get("y", 640), "mt"
    if t == "scoreboard":
        head = T.sprite(o.get("title", "VICTORY POINTS"), "tech", 20, (225, 225, 225), wght=500, tracking=0.25,
                        shadow=3)
        cols = []
        for tm in o["teams"]:
            lab = T.sprite(tm.get("name", fx.label(tm["faction"])).upper(), "tech", 26, fx.color(tm["faction"]),
                           wght=600, tracking=0.08, shadow=3)
            val = T.sprite(str(tm["score"]), "tech", 84, fx.color(tm["faction"]), wght=700, shadow=4)
            cols.append(T.stack([lab, val], gap=-6))
        row = T.hstack(cols, gap=40)
        foot = T.sprite(o.get("foot", "FIRST TO 10 WINS"), "tech", 20, (225, 225, 225), wght=500, tracking=0.2,
                        shadow=3)
        return T.stack([head, row, foot], gap=0), W / 2, o.get("y", 370), "mt"
    if t == "approach":
        col = fx.color(o["faction"])
        top = T.sprite(o.get("top", "THE AIs ARE").upper().replace("AIS", "AIs"), "tech", 42, (235, 240, 240),
                       wght=600, tracking=0.12, shadow=3)
        main = T.sprite(o["main"].upper(), "tech", 84, col, wght=800, tracking=0.06, glow=10, shadow=3)
        body = T.stack([top, main], gap=-12)
        br = brackets(int(W * 0.8), body.height + 80, col)
        br.alpha_composite(body, ((br.width - body.width) // 2, 40))
        return br, W / 2, o.get("y", 300), "mt"
    raise KeyError(t)


# ======================================================================
# renderer
# ======================================================================
class CinematicRenderer:
    def __init__(self, spec, tl):
        self.spec, self.tl = spec, tl
        self.series = spec["series"]
        self.fx = Factions(self.series)
        self.cast = spec["cast"]
        self.vignette = self._vignette()
        self.placeholders = 0
        self.media = []
        for sh in tl.shots:
            if sh.data.get("type") == "endcard":
                self.media.append(None)
                continue
            m = genai.shot_media(sh.data["visual"], self.series, seed=1000 + sh.idx,
                                 label=f"shot {sh.idx + 1}", want_seconds=sh.dur)
            self.placeholders += m["placeholder"]
            self.media.append(m)
        self.sources = {}
        self.overlays = [[(o, build_overlay(o, self.fx, self.series.get("ep"))) for o in self._stack(sh)]
                         for sh in tl.shots]
        self.captions = [(c, self._caption(c)) for c in tl.captions]
        end = [s for s in tl.shots if s.data.get("type") == "endcard"]
        self.content_end = end[0].start if end else tl.total
        self.endcard_parts = self._endcard_parts()
        self._end_bg = None

    # ----- static -----
    def _vignette(self):
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt(((x - W / 2) / (W * 0.75)) ** 2 + ((y - H / 2) / (H * 0.7)) ** 2)
        a = (np.clip(d - 0.55, 0, 1) * 200).astype(np.uint8)
        im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        im.putalpha(Image.fromarray(a))
        return im

    @staticmethod
    def _stack(sh):
        """A faction tag and an equation panel both default to the top-left
        corner; when a shot has both, push the panel below the tag."""
        ovs = [dict(o) for o in sh.data.get("overlays", [])]
        tags = [o for o in ovs if o["type"] == "faction_tag"]
        for o in ovs:
            if tags and o["type"] == "equation" and "y" not in o:
                o["y"] = tags[0].get("y", 250) + 130
        return ovs

    def _caption(self, c):
        c_who = self.cast[c.who]
        fac = c_who.get("faction")
        if c_who.get("kind") == "ai" and fac:
            st = AI_CAPTION
            spr = T.sprite(c.text, st["font"], st["size"], self.fx.color(fac), wght=st["wght"],
                           tracking=st["tracking"], shadow=5, glow=6)
        else:
            st = HUMAN_CAPTION
            col = rgb(self.series["factions"].get(fac, {}).get("caption_color", CREAM))
            spr = T.sprite(c.text, st["font"], st["size"], col, wght=st["wght"], tracking=st["tracking"], shadow=7)
        if spr.width > W - 100:
            raise LayoutError(f"caption '{c.text}' is {spr.width}px wide")
        return spr

    def _endcard_parts(self):
        s = self.series
        name = T.sprite(s["name"].upper(), "serif", 118, (241, 226, 196), wght=700, shadow=6)
        tag = T.sprite(s["tagline"], "serif", 36, (225, 214, 192), wght=500, shadow=3)
        cta = T.sprite(s["cta"].upper(), "tech", 24, (190, 198, 201), wght=500, tracking=0.25)
        mark = T.wordmark(98, gg=(255, 255, 255), human=(255, 255, 255), suffix=".ai", suffix_fill=TEAL)
        y, out = 540, []
        for spr, gap in ((divider(420), 6), (name, -4), (tag, 14), (divider(300), 22), (cta, 6), (mark, 0)):
            out.append((spr, y))
            y += spr.height + gap
        return out

    # ----- per frame -----
    def _source(self, idx):
        if idx not in self.sources:
            for k in list(self.sources):  # free finished streams
                self.sources.pop(k).close()
            sh = self.tl.shots[idx]
            m = self.media[idx]
            v = sh.data["visual"]
            if m["kind"] == "video":
                self.sources[idx] = VideoSource(m["path"], sh.dur, v.get("trim", 0.0))
            else:
                self.sources[idx] = ImageSource(m["path"], sh.dur, v.get("camera", "push_in"))
        return self.sources[idx]

    def frame(self, i):
        t = i / FPS
        sh = self.tl.shot_at(t)
        lt = t - sh.start
        if sh.data.get("type") == "endcard":
            return self._endcard(lt, sh)
        img = self._source(sh.idx).frame(int(round(lt * FPS)))
        img = self._grade(img, sh, lt)
        img.paste(self.vignette, (0, 0), self.vignette)
        img = self._transition(img, sh, lt, i)
        for o, built in self.overlays[sh.idx]:
            self._overlay(img, o, built, lt, sh.dur)
        for c, spr in self.captions:
            if c.start <= t < c.end:
                D.paste(img, spr, W / 2, CAPTION_Y, alpha=D.prog(t, c.start, 0.1), anchor="mm")
        return img

    def _grade(self, img, sh, lt):
        tint = sh.data.get("tint")
        if tint:
            k = sh.data.get("tint_amount", 0.45) * (1 - D.prog(lt, sh.data.get("tint_until", 1.2), 0.5))
            if k > 0:
                img = Image.blend(img, Image.new("RGB", (W, H), rgb(tint)), k)
        return img

    def _transition(self, img, sh, lt, i):
        tr = sh.data.get("transition", "cut")
        if tr == "fade" and lt < 0.4:
            img = Image.blend(Image.new("RGB", (W, H)), img, D.ease_in_out(lt / 0.4))
        elif tr == "flash" and lt < 0.2:
            img = Image.blend(img, Image.new("RGB", (W, H), (255, 255, 255)), 0.8 * (1 - lt / 0.2))
        elif tr == "glitch" and lt < 0.55:
            rng = np.random.default_rng(i)
            a = np.asarray(img).copy()
            k = 1 - lt / 0.55
            sh_px = int(28 * k)
            a[..., 0] = np.roll(a[..., 0], sh_px, axis=1)
            a[..., 2] = np.roll(a[..., 2], -sh_px, axis=1)
            for _ in range(int(10 * k)):
                y0 = int(rng.integers(0, H - 80))
                h = int(rng.integers(10, 80))
                a[y0:y0 + h] = np.roll(a[y0:y0 + h], int(rng.integers(-60, 60)), axis=1)
            img = Image.fromarray(a)
            img = Image.blend(img, Image.new("RGB", (W, H), (200, 30, 40)), 0.35 * k)
        return img

    def _overlay(self, img, o, built, lt, dur):
        at = o.get("at", 0.15)
        end = at + o["dur"] if o.get("dur") else dur + 1
        if lt < at or lt > end:
            return
        a = D.prog(lt, at, 0.3) * (1 - D.prog(lt, end - 0.25, 0.25))
        spr, x, y, anchor = built
        slide = (1 - D.ease_out_cubic(D.prog(lt, at, 0.35))) * 24
        if isinstance(spr, tuple) and spr[0] == "equation":
            _, rows, res, col = spr
            every = o.get("every", 0.55)
            shown = [r for k, r in enumerate(rows) if lt >= at + k * every]
            body = T.stack(shown, gap=6, align="l") if shown else None
            res_on = res is not None and lt >= at + len(rows) * every
            bw = max([r.width for r in rows] + ([res.width] if res else [])) + 100
            bh = sum(r.height + 6 for r in rows) + ((res.height + 30) if res else 0) + 40
            bg = panel(bw, bh, 150)
            if body:
                bg.alpha_composite(body, (50, 20))
            if res_on:
                yy = 20 + sum(r.height + 6 for r in rows)
                ImageDraw.Draw(bg).line([(50, yy + 6), (50 + int((bw - 160) * D.prog(lt, at + len(rows) * every, 0.3)),
                                                         yy + 6)], fill=col + (200,), width=2)
                bg.alpha_composite(D.with_alpha(res, D.prog(lt, at + len(rows) * every + 0.15, 0.25)), (50, yy + 16))
            D.paste(img, bg, x, y, alpha=a)
            return
        dx = -slide if anchor.startswith("l") else slide if anchor.startswith("r") else 0
        dy = slide if anchor[0] == "m" else 0
        D.paste(img, spr, x + dx, y + dy, alpha=a, anchor=anchor)

    def _endcard(self, lt, sh):
        if self._end_bg is None:
            v = sh.data.get("visual", {})
            first = next(m for m in self.media if m)
            src = ImageSource(first["path"], 1, "static") if first["kind"] == "image" else None
            if v.get("prompt") or v.get("image"):
                m = genai.shot_media(v, self.series, 999, "end card", sh.dur)
                src = ImageSource(m["path"], 1, "static") if m["kind"] == "image" else VideoSource(m["path"], 1)
                self.placeholders += m["placeholder"]
            if src is None:
                src = VideoSource(first["path"], 1)
            bg = src.frame(0).filter(ImageFilter.GaussianBlur(3))
            src.close()
            bg = Image.blend(bg, Image.new("RGB", (W, H), (6, 10, 12)), 0.62)
            bg.paste(self.vignette, (0, 0), self.vignette)
            self._end_bg = bg
        img = Image.blend(Image.new("RGB", (W, H)), self._end_bg, D.prog(lt, 0, 0.5))
        for k, (spr, y) in enumerate(self.endcard_parts):
            a = D.prog(lt, 0.35 + k * 0.18, 0.4)
            D.paste(img, spr, W / 2, y + (1 - D.ease_out_cubic(a)) * 14, alpha=a, anchor="mt")
        return img

    def close(self):
        for s in self.sources.values():
            s.close()
        self.sources = {}

    def sheet_times(self):
        return [s.start + min(s.dur * 0.6, s.dur - 0.1) for s in self.tl.shots]
