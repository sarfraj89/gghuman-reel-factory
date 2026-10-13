"""Doodle format renderer: flat hand-drawn comic on cream paper.

Everything is drawn with a wobbly "pen" (resampled outlines + smooth noise),
re-seeded every 4 frames so lines "boil" like hand-animated cartoons. Shots are
built from a closed vocabulary (spec.DOODLE_VISUALS); characters get
expressions per shot or per line and lip-flap from the voice envelope.
"""
import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .. import draw as D
from ..draw import LayoutError
from ..spec import ROSTER
from . import text as T

W, H, FPS = D.W, D.H, D.FPS
SS = 2
BOILS = 3
BOIL_FRAMES = 4

PAPER = (236, 231, 220)
INK = (30, 28, 26, 255)
SKIN = (98, 178, 48, 255)
SHIRT = (44, 141, 214, 255)
FELT = (118, 136, 106, 255)
GROUND = (122, 140, 110, 255)
CARD = (250, 246, 236, 255)
CARD_BACK = (180, 162, 214, 255)
RED_SUIT = (200, 60, 50, 255)
SCREEN = (24, 24, 26, 255)
FRAME = (233, 222, 200, 255)
GLOW = (240, 182, 142, 255)
CAL_RED = (196, 102, 80, 255)
CELL = (214, 207, 188, 255)
WHITE = (252, 250, 244, 255)
MOUTH = (96, 22, 22, 255)
LW = 6

CAPTION_Y, HOOK_TOP = 1385, 250
LOGO_X, LOGO_Y = 64, 1548


# ======================================================================
# pen
# ======================================================================
class Pen:
    """Wobbly vector pen on a supersampled RGBA canvas. Coordinates are
    absolute frame coordinates; (ox, oy) is the canvas origin in the frame."""

    def __init__(self, w=W, h=H, seed=0, ox=0, oy=0, wobble=1.0):
        self.w, self.h, self.ox, self.oy = int(w), int(h), ox, oy
        self.im = Image.new("RGBA", (self.w * SS, self.h * SS), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.im)
        self.rng = np.random.default_rng(seed)
        self.wob = wobble

    def _xy(self, pts):
        return [((x - self.ox) * SS, (y - self.oy) * SS) for x, y in pts]

    def _jitter(self, pts, closed):
        P = np.asarray(pts, float)
        if closed:
            P = np.vstack([P, P[:1]])
        L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        if L[-1] < 2 or self.wob <= 0:
            return [tuple(p) for p in P]
        n = max(6, int(L[-1] / 9))
        s = np.linspace(0, L[-1], n)
        x, y = np.interp(s, L, P[:, 0]), np.interp(s, L, P[:, 1])
        k = max(2, n // 10)
        nz = self.rng.standard_normal((n + 2 * k, 2))
        ker = np.hanning(2 * k + 1)
        ker /= ker.sum()
        nz = np.stack([np.convolve(nz[:, i], ker, "same") for i in range(2)], 1)[k:k + n]
        nz *= self.wob * 1.4 / (nz.std() + 1e-9)
        if closed:
            ramp = np.linspace(0, 1, n)[:, None]
            nz = nz - ramp * (nz[-1] - nz[0])
        return list(zip(x + nz[:, 0], y + nz[:, 1]))

    def shape(self, pts, fill=None, lw=LW, closed=True, ink=INK):
        wp = self._xy(self._jitter(pts, closed))
        if fill is not None and closed:
            self.d.polygon(wp, fill=fill)
        if lw:
            seq = wp + ([wp[0], wp[1]] if closed else [])
            self.d.line(seq, fill=ink, width=int(lw * SS), joint="curve")
            if not closed:
                r = lw * SS / 2
                for px, py in (wp[0], wp[-1]):
                    self.d.ellipse([px - r, py - r, px + r, py + r], fill=ink)

    def ellipse(self, cx, cy, rx, ry, fill=None, lw=LW, ink=INK, a0=0, a1=360, closed=True):
        n = max(28, int((rx + ry) / 3))
        pts = [(cx + rx * math.cos(math.radians(a)), cy + ry * math.sin(math.radians(a)))
               for a in np.linspace(a0, a1, n, endpoint=(a1 - a0) < 360)]
        self.shape(pts, fill, lw, closed, ink)

    def rrect(self, x, y, w, h, r, fill=None, lw=LW, ink=INK):
        r = min(r, w / 2, h / 2)
        pts = []
        for cx, cy, a in ((x + w - r, y + r, -90), (x + w - r, y + h - r, 0),
                          (x + r, y + h - r, 90), (x + r, y + r, 180)):
            pts += [(cx + r * math.cos(math.radians(a + d)), cy + r * math.sin(math.radians(a + d)))
                    for d in range(0, 91, 15)]
        self.shape(pts, fill, lw, True, ink)

    def line(self, pts, lw=LW, ink=INK):
        self.shape(pts, None, lw, False, ink)

    def dot(self, cx, cy, r, fill=INK):
        self.ellipse(cx, cy, r, r, fill, 0)

    def text(self, x, y, s, fname="comic", size=40, fill=INK, anchor="mm", wght=None):
        f = T.font(fname, int(size * SS), wght)
        self.d.text(((x - self.ox) * SS, (y - self.oy) * SS), s, font=f, fill=fill, anchor=anchor)

    def sprite(self):
        return self.im.resize((self.w, self.h), Image.LANCZOS)


# ======================================================================
# characters
# ======================================================================
def human_face(p, cx, cy, R, expr, mouth):
    ex, ey, er = 0.3 * R, cy - 0.05 * R, 0.075 * R
    big = expr in ("shocked", "scream")
    if big:
        for sx in (-1, 1):
            p.ellipse(cx + sx * ex, ey - 0.05 * R, 0.19 * R, 0.19 * R, WHITE, LW * 0.8)
            p.dot(cx + sx * ex, ey - 0.03 * R, 0.08 * R)
    elif expr == "happy":
        for sx in (-1, 1):
            p.line([(cx + sx * ex - 0.11 * R, ey + 0.03 * R), (cx + sx * ex, ey - 0.07 * R),
                    (cx + sx * ex + 0.11 * R, ey + 0.03 * R)], LW * 0.8)
    elif expr == "thinking":
        for sx in (-1, 1):
            p.dot(cx + sx * ex + 0.05 * R, ey - 0.07 * R, er)
    else:
        for sx in (-1, 1):
            p.dot(cx + sx * ex, ey, er)
        if expr in ("annoyed", "smug"):
            for sx in ((-1, 1) if expr == "annoyed" else (1,)):
                p.line([(cx + sx * ex - 0.12 * R, ey - 0.08 * R), (cx + sx * ex + 0.12 * R, ey - 0.08 * R)],
                       LW * 0.7)
    # brows
    by = ey - 0.27 * R
    brows = {
        "worried": (0.06, -0.06), "sad": (0.08, -0.08), "neutral": (0.03, -0.03),
        "annoyed": (-0.04, 0.04), "angry": (-0.1, 0.08), "shocked": (0.0, 0.0), "scream": (0.06, -0.06),
        "smug": (0.0, -0.08), "thinking": (0.05, -0.02), "happy": (0.0, 0.0),
    }[expr]
    lift = -0.12 * R if big else 0
    for sx in (-1, 1):
        inner, outer = brows
        x_in, x_out = cx + sx * 0.14 * R, cx + sx * 0.46 * R
        p.line([(x_in, by + lift - inner * R), (x_out, by + lift - outer * R)], LW * 0.75)
    # mouth
    my = cy + 0.4 * R
    if expr == "scream":
        p.ellipse(cx, my + 0.05 * R, 0.3 * R, 0.36 * R, MOUTH, LW)
    elif mouth > 0:
        p.ellipse(cx, my, (0.1 + 0.04 * mouth) * R, (0.07 + 0.08 * mouth) * R, MOUTH, LW * 0.8)
    elif expr == "shocked":
        p.ellipse(cx, my, 0.07 * R, 0.09 * R, MOUTH, LW * 0.8)
    elif expr == "happy":
        p.ellipse(cx, my - 0.08 * R, 0.2 * R, 0.14 * R, None, LW * 0.8, a0=20, a1=160, closed=False)
    elif expr in ("sad", "angry"):
        p.ellipse(cx, my + 0.1 * R, 0.17 * R, 0.1 * R, None, LW * 0.8, a0=200, a1=340, closed=False)
    elif expr == "smug":
        p.line([(cx - 0.15 * R, my + 0.02 * R), (cx + 0.08 * R, my), (cx + 0.18 * R, my - 0.07 * R)], LW * 0.8)
    elif expr == "worried":
        p.line([(cx - 0.14 * R, my), (cx - 0.05 * R, my - 0.03 * R), (cx + 0.05 * R, my + 0.02 * R),
                (cx + 0.14 * R, my - 0.01 * R)], LW * 0.8)
    else:
        p.line([(cx - 0.13 * R, my), (cx + 0.13 * R, my)], LW * 0.8)


def human_bust(p, cx, cy, R, expr, mouth):
    p.ellipse(cx, cy + 1.95 * R, 1.55 * R, 1.15 * R, SHIRT)
    for dx, tx in ((-0.25, -0.38), (0.0, 0.02), (0.25, 0.4)):
        p.line([(cx + dx * R, cy - 0.95 * R), (cx + (dx + tx) / 2 * R, cy - 1.15 * R),
                (cx + tx * R, cy - 1.32 * R)], LW * 0.8)
    p.ellipse(cx, cy, R, R, SKIN)
    human_face(p, cx, cy, R, expr, mouth)


def human_full(p, cx, cy, R, expr, mouth):
    """Small standing figure; (cx, cy) is the head centre."""
    by = cy + 1.75 * R
    for sx in (-1, 1):
        p.line([(cx + sx * 0.35 * R, by + 1.1 * R), (cx + sx * 0.4 * R, by + 2.05 * R)], LW)
        p.line([(cx + sx * 0.7 * R, by - 0.45 * R), (cx + sx * 1.15 * R, by + 0.15 * R),
                (cx + sx * 1.3 * R, by + 0.6 * R)], LW * 1.6, ink=SHIRT)
        p.ellipse(cx + sx * 1.3 * R, by + 0.62 * R, 0.17 * R, 0.17 * R, SKIN, LW * 0.7)
    p.ellipse(cx, by, 0.85 * R, 1.15 * R, SHIRT)
    for dx, tx in ((-0.25, -0.38), (0.0, 0.02), (0.25, 0.4)):
        p.line([(cx + dx * R, cy - 0.95 * R), (cx + tx * R, cy - 1.3 * R)], LW * 0.6)
    p.ellipse(cx, cy, R, R, SKIN)
    human_face(p, cx, cy, R, expr, mouth)


def logo_glyph(p, brand, cx, cy, r):
    white = WHITE
    if brand == "Claude":
        for k in range(12):
            a = math.radians(k * 30 + 8)
            r0 = r * (0.12 if k % 2 else 0.2)
            pts = [(cx + r0 * math.cos(a), cy + r0 * math.sin(a)), (cx + r * math.cos(a), cy + r * math.sin(a))]
            p.line(pts, LW * 1.9)
            p.line(pts, LW * 1.05, ink=white)
    elif brand == "GPT":
        for k in range(6):
            a = math.radians(k * 60)
            p.ellipse(cx + 0.38 * r * math.cos(a), cy + 0.38 * r * math.sin(a), 0.5 * r, 0.5 * r, None,
                      LW * 0.7, ink=white)
    elif brand == "Gemini":
        pts = []
        for k in range(8):
            rr = r if k % 2 == 0 else 0.22 * r
            a = math.radians(k * 45 - 90)
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        p.shape(pts, white, LW * 0.7)
    elif brand == "Grok":
        p.line([(cx - 0.6 * r, cy + 0.7 * r), (cx + 0.6 * r, cy - 0.7 * r)], LW * 1.4, ink=white)
        p.line([(cx - 0.5 * r, cy - 0.5 * r), (cx - 0.05 * r, cy)], LW * 1.4, ink=white)
    else:
        p.text(cx, cy, {"DeepSeek": "DS", "MiniMax": "MM"}.get(brand, brand[0]), "comic", r * 1.2, white)


ROBOT_EYES = {  # (w, h, dy) of eye shapes, or a special token
    "neutral": (40, 24, 0), "worried": (40, 18, 4), "annoyed": (42, 12, 4), "shocked": "round",
    "scream": "round", "happy": "arc", "smug": (44, 12, 6), "thinking": (36, 22, -12),
    "sad": (34, 16, 8), "angry": "slant",
}


def robot_bust(p, cx, cy, s, expr, mouth, brand, full=False):
    col = D.hex2rgb(ROSTER.get(brand, "#D97757")) + (255,)
    if full:
        by = cy + 190 * s
        for sx in (-1, 1):
            p.line([(cx + sx * 45 * s, by + 80 * s), (cx + sx * 50 * s, by + 175 * s)], LW)
        p.ellipse(cx, by, 105 * s, 100 * s, col)
        logo_glyph(p, brand, cx, by, 48 * s)
    else:
        p.ellipse(cx, cy + 215 * s, 175 * s, 150 * s, col)
        logo_glyph(p, brand, cx, cy + 170 * s, 66 * s)
    # antenna
    ax, ay = cx + 62 * s, cy - 104 * s
    p.line([(ax, ay), (ax + 12 * s, ay - 26 * s), (ax - 6 * s, ay - 44 * s), (ax + 10 * s, ay - 66 * s)], LW * 0.8)
    p.ellipse(ax + 12 * s, ay - 82 * s, 17 * s, 17 * s, col)
    p.rrect(cx - 118 * s, cy - 106 * s, 236 * s, 212 * s, 26 * s, FRAME)
    p.rrect(cx - 92 * s, cy - 82 * s, 184 * s, 164 * s, 18 * s, SCREEN, LW * 0.6)
    eyes = ROBOT_EYES[expr]
    for sx in (-1, 1):
        ex, ey = cx + sx * 46 * s, cy - 14 * s
        if eyes == "round":
            p.ellipse(ex, ey, 22 * s, 22 * s, GLOW, 0)
        elif eyes == "arc":
            p.line([(ex - 20 * s, ey + 6 * s), (ex, ey - 10 * s), (ex + 20 * s, ey + 6 * s)], LW * 1.2, GLOW)
        elif eyes == "slant":
            p.shape([(ex - 22 * s, ey - 10 * s * -sx), (ex + 22 * s, ey + 10 * s * -sx),
                     (ex + 22 * s, ey + 18 * s), (ex - 22 * s, ey + 18 * s)], GLOW, 0)
        else:
            w, h, dy = eyes
            p.rrect(ex - w / 2 * s, ey - h / 2 * s + dy * s, w * s, h * s, h / 2 * s, GLOW, 0)
    my = cy + 40 * s
    if expr == "scream" or mouth > 0:
        mh = 34 if expr == "scream" else 12 + 10 * mouth
        p.rrect(cx - 22 * s, my - mh / 2 * s, 44 * s, mh * s, 8 * s, GLOW, 0)
    elif expr == "happy" or expr == "smug":
        p.ellipse(cx + (8 * s if expr == "smug" else 0), my - 10 * s, 24 * s, 16 * s, None, LW * 0.9,
                  ink=GLOW, a0=20, a1=160, closed=False)
    elif expr in ("sad", "angry"):
        p.ellipse(cx, my + 8 * s, 22 * s, 12 * s, None, LW * 0.9, ink=GLOW, a0=200, a1=340, closed=False)
    else:
        p.rrect(cx - 20 * s, my - 6 * s, 40 * s, 12 * s, 6 * s, GLOW, 0)


@lru_cache(maxsize=512)
def character(kind, brand, pose, x, y, scale, expr, mouth, boil):
    """Cached sprite of one character; returns (sprite, ox, oy)."""
    if kind == "ai":
        bw, bh = 460 * scale, (720 if pose == "full" else 640) * scale
        ox, oy = x - bw / 2, y - 220 * scale
        p = Pen(bw, bh, seed=boil * 7 + 1, ox=ox, oy=oy)
        robot_bust(p, x, y, scale, expr, mouth, brand, full=(pose == "full"))
    else:
        R = 100 * scale
        bw, bh = 4.0 * R, 4.4 * R
        ox, oy = x - bw / 2, y - 1.45 * R
        p = Pen(bw, bh, seed=boil * 7 + 2, ox=ox, oy=oy)
        (human_full if pose == "full" else human_bust)(p, x, y, R, expr, mouth)
    return p.sprite(), int(round(ox)), int(round(oy))


# ======================================================================
# props
# ======================================================================
@lru_cache(maxsize=512)
def card(code, w, h, boil, hidden=False):
    p = Pen(w + 12, h + 12, seed=hash((code, boil)) % 9999, ox=-6, oy=-6, wobble=0.6)
    if hidden or code == "?":
        p.rrect(0, 0, w, h, 9, CARD_BACK, LW * 0.75)
        p.text(w / 2, h / 2, "?", "comic", h * 0.42, (90, 80, 120, 255))
    else:
        rank, suit = code[:-1], code[-1]
        col = RED_SUIT if suit in "♥♦" else INK
        p.rrect(0, 0, w, h, 9, CARD, LW * 0.75)
        p.text(w / 2, h * 0.28, rank, "comic", h * 0.34, col)
        p.text(w / 2, h * 0.66, suit, "symbol", h * 0.3, col)
    return p.sprite()


def paste_card(img, code, cx, cy, w, h, boil, hidden=False, sx=1.0, alpha=1.0, scale=1.0):
    D.paste(img, card(code, w, h, boil, hidden), cx, cy, alpha=alpha, scale=scale, sx=sx * scale, anchor="mm")


def calendar(p, x, y, w, h, label, rows=3, cols=4, hsize=None):
    hh = h * 0.22
    p.rrect(x, y, w, h, 10, (246, 241, 230, 255))
    p.shape([(x, y), (x + w, y), (x + w, y + hh), (x, y + hh)], CAL_RED)
    p.text(x + w / 2, y + hh / 2 + 2, label, "comic", hsize or hh * 0.72, (250, 240, 228, 255))
    cw, ch = w / (cols + 1), (h - hh) / (rows + 1.2)
    for r in range(rows):
        for c in range(cols):
            p.rrect(x + cw * (c + 0.5) - cw * 0.3 + cw * 0.05, y + hh + ch * (r + 0.6) - ch * 0.3,
                    cw * 0.6 + cw * 0.15, ch * 0.6, 8, CELL, LW * 0.5)


def hexagon(cx, cy, r):
    return [(cx + r * math.cos(math.radians(60 * i + 30)), cy + r * math.sin(math.radians(60 * i + 30)))
            for i in range(6)]


HEX_COLORS = [(214, 178, 70, 255), (92, 140, 70, 255), (190, 110, 70, 255), (150, 150, 150, 255),
              (150, 196, 100, 255), (226, 208, 150, 255), (214, 178, 70, 255)]
HEX_NUMBERS = [6, 3, 8, 10, 5, 0, 9]


def hex_centres(cx, cy, r):
    ring = [(cx + r * 1.74 * math.cos(math.radians(60 * k)), cy + r * 1.74 * math.sin(math.radians(60 * k)))
            for k in range(6)]
    return ring[:3] + [(cx, cy)] + ring[3:]


def hex_cluster(p, cx, cy, r, numbers=HEX_NUMBERS, token=True):
    centres = hex_centres(cx, cy, r)
    for (hx, hy), col, num in zip(centres, HEX_COLORS, numbers):
        p.shape(hexagon(hx, hy, r), col, LW * 0.8)
        if num and token:
            p.ellipse(hx, hy, r * 0.32, r * 0.32, CARD, LW * 0.6)
            p.text(hx, hy + 2, str(num), "comic", r * 0.42, RED_SUIT if num in (6, 8) else INK)
    return dict(zip(numbers, centres))


def robber(p, x, y, s):
    p.ellipse(x, y + 30 * s, 28 * s, 34 * s, (70, 66, 64, 255), LW * 0.8)
    p.ellipse(x, y - 14 * s, 18 * s, 18 * s, (70, 66, 64, 255), LW * 0.8)


def die(p, cx, cy, size, value, angle=0.0):
    pts = [(cx + size / 2 * math.sqrt(2) * math.cos(math.radians(45 + 90 * k) + angle),
            cy + size / 2 * math.sqrt(2) * math.sin(math.radians(45 + 90 * k) + angle)) for k in range(4)]
    p.shape(pts, CARD, LW)
    pips = {1: [(0, 0)], 2: [(-1, -1), (1, 1)], 3: [(-1, -1), (0, 0), (1, 1)],
            4: [(-1, -1), (1, -1), (-1, 1), (1, 1)], 5: [(-1, -1), (1, -1), (0, 0), (-1, 1), (1, 1)],
            6: [(-1, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (1, 1)]}[value]
    for px, py in pips:
        rx = px * size * 0.26
        ry = py * size * 0.26
        x = cx + rx * math.cos(angle) - ry * math.sin(angle)
        y = cy + rx * math.sin(angle) + ry * math.cos(angle)
        p.dot(x, y, size * 0.08)


# ======================================================================
# shots
# ======================================================================
FIG_POS = {"bl": (215, 1235), "br": (865, 1235), "bm": (540, 1265)}


class Shot:
    """One doodle shot. Subclasses fill `back`, `front` (static, per boil),
    `chars` and `animate`."""
    focus = (540, 960)
    zoom = (1.0, 1.04)

    def __init__(self, data, cast, dur):
        self.data, self.v, self.cast, self.dur = data, data["visual"], cast, dur
        self.chars = []  # (who, pose, x, y, scale, default_expr)
        self.setup()
        z = self.v.get("zoom")
        if z:
            self.zoom = tuple(z) if isinstance(z, (list, tuple)) else (z, z)
        if self.v.get("focus"):
            self.focus = tuple(self.v["focus"])
        self.back = [self._layer(self.draw_back, b) for b in range(BOILS)]
        self.front = [self._layer(self.draw_front, b) for b in range(BOILS)]

    def _layer(self, fn, boil):
        p = Pen(seed=boil * 101 + 5)
        if fn(p, boil) is False:
            return None
        return p.sprite()

    def setup(self):
        fig = self.v.get("figure")
        if fig:
            x, y = FIG_POS[fig.get("pos", "bl")]
            self.chars.append((fig["who"], "full", x, y, fig.get("scale", 0.75), fig.get("expression", "worried")))

    def draw_back(self, p, boil):
        return False

    def draw_front(self, p, boil):
        return False

    def animate(self, img, t, boil):
        pass


class TableShot(Shot):
    def setup(self):
        humans = [k for k, c in self.cast.items() if c["kind"] == "human"]
        ais = [k for k, c in self.cast.items() if c["kind"] == "ai"]
        self.left = self.v.get("left", humans[0] if humans else None)
        self.right = self.v.get("right", ais[0] if ais else None)
        ex = self.v.get("expressions", {})
        if self.left:
            self.chars.append((self.left, "bust", 285, 855, 1.0, ex.get(self.left, "worried")))
        if self.right:
            self.chars.append((self.right, "bust", 790, 790, 0.95, ex.get(self.right, "neutral")))

    def draw_front(self, p, boil):
        game = self.v.get("game", "poker")
        p.ellipse(540, 1185, 445, 200, FELT)
        if game == "settler_rush":
            hex_cluster(p, 540, 1175, 48, token=False)
        elif game == "last_vote":
            for k, x in enumerate((420, 540, 660)):
                p.rrect(x - 34, 1140, 68, 92, 8, CARD_BACK, LW * 0.7)
        else:
            for k in range(4):
                p.ellipse(545, 1305 - k * 7, 62, 13, (214, 120, 96, 255), LW * 0.6)
        for who, x in ((self.left, 280), (self.right, 795)):
            if who:
                p.text(x, 1400, self.cast[who].get("name", who.upper()), "comic", 46)
        return True

    def animate(self, img, t, boil):
        v = self.v
        board = v.get("board", [])
        for k, c in enumerate(board):
            paste_card(img, c, 540 + (k - (len(board) - 1) / 2) * 77, 1160, 70, 96, boil)
        for k, c in enumerate(v.get("you_hand", [])):
            paste_card(img, c, 228 + k * 70, 1255, 62, 86, boil)
        for k, c in enumerate(v.get("ai_hand", ["?", "?"] if board else [])):
            paste_card(img, c, 772 + k * 78, 1235, 62, 86, boil, hidden=(c == "?"))


class CloseupShot(Shot):
    zoom = (1.0, 1.07)

    def setup(self):
        who = self.v["who"]
        kind = self.cast[who]["kind"]
        self.kind = kind
        expr = self.v.get("expression", "neutral")
        if kind == "ai":
            self.chars.append((who, "bust", 540, 1020, 2.0, expr))
            self.focus = (540, 1050)
        else:
            self.chars.append((who, "bust", 540, 1150, 2.2, expr))
            self.focus = (540, 1100)
        if expr == "scream":
            self.zoom = (1.15, 1.3)

    def draw_front(self, p, boil):
        p.ellipse(540, 1990, 1000, 470, GROUND)
        return True

    def animate(self, img, t, boil):
        for k, c in enumerate(self.v.get("hand", [])):
            paste_card(img, c, 905 + k * 75, 1555, 66, 92, boil)


class ClockCalendarShot(Shot):
    def draw_back(self, p, boil):
        calendar(p, 372, 712, 495, 395, self.v.get("label", "TODAY"))
        p.ellipse(498, 570, 136, 136, (246, 241, 230, 255))
        return True

    def animate(self, img, t, boil):
        spin = self.v.get("spin", 1.2)
        p = Pen(300, 300, seed=boil, ox=348, oy=420)
        for length, speed, lw in ((95, spin, LW), (70, spin / 12, LW)):
            a = math.radians(-90 + 360 * speed * t)
            p.line([(498, 570), (498 + length * math.cos(a), 570 + length * math.sin(a))], lw)
        D.paste(img, p.sprite(), 348, 420)


class CalendarGridShot(Shot):
    def setup(self):
        super().setup()
        n = int(self.v.get("count", 9))
        start = int(self.v.get("start_year", 2026))
        self.labels = self.v.get("labels") or [str(start + k) for k in range(n)]
        self.cells = []
        cols = 3
        for k, lab in enumerate(self.labels):
            r, c = divmod(k, cols)
            p = Pen(262, 200, seed=k, ox=-8, oy=-8)
            calendar(p, 0, 0, 244, 182, lab, rows=3, cols=4, hsize=34)
            self.cells.append((p.sprite(), 92 + c * 300, 380 + r * 225))

    def animate(self, img, t, boil):
        every = self.v.get("every", min(0.35, max(0.12, (self.dur * 0.6) / max(1, len(self.cells)))))
        for k, (spr, x, y) in enumerate(self.cells):
            k_t = D.prog(t, 0.15 + k * every, 0.22)
            if k_t > 0:
                D.paste(img, spr, x + 131, y + 100, alpha=min(1, k_t * 2), scale=0.6 + 0.4 * D.ease_out_back(k_t),
                        anchor="mm")


class SunEarthShot(Shot):
    def setup(self):
        super().setup()
        self.sun = []
        for b in range(BOILS):
            p = Pen(620, 620, seed=b, ox=380, oy=250)
            p.ellipse(690, 560, 262, 262, (222, 142, 64, 255), LW * 1.2)
            p.ellipse(690, 560, 205, 205, (158, 104, 84, 255), LW * 0.8)
            self.sun.append(p.sprite())
        self.earth = []
        for b in range(BOILS):
            p = Pen(500, 420, seed=b + 9, ox=130, oy=760)
            p.ellipse(380, 1010, 115, 115, (70, 142, 204, 255))
            p.shape([(318, 952), (370, 938), (392, 980), (352, 1012), (310, 995)], (96, 176, 72, 255), LW * 0.7)
            p.shape([(402, 1032), (452, 1010), (470, 1060), (420, 1092), (396, 1068)], (96, 176, 72, 255),
                    LW * 0.7)
            p.shape([(300, 1040), (340, 1050), (330, 1085), (296, 1070)], (96, 176, 72, 255), LW * 0.7)
            for sx, sy in ((175, 990), (560, 960), (520, 1060), (240, 1110), (600, 1020)):
                p.line([(sx - 12, sy), (sx + 12, sy)], LW * 0.6)
                p.line([(sx, sy - 12), (sx, sy + 12)], LW * 0.6)
            self.earth.append(p.sprite())
        who = self.v.get("on_earth")
        if who:
            self.chars.append((who, "full", 380, 862, 0.22, self.v.get("expression", "worried")))

    def animate(self, img, t, boil):
        dim = 1 - 0.25 * D.prog(t, 0, self.dur) if self.v.get("burning_out", True) else 1
        flick = 1 + 0.02 * math.sin(t * 23)
        D.paste(img, self.sun[boil], 690, 560, scale=(0.92 + 0.08 * dim) * flick, alpha=dim, anchor="mm")
        D.paste(img, self.earth[boil], 130, 760)


class CardRevealShot(Shot):
    def setup(self):
        super().setup()
        n = len(self.v["hidden"])
        self.flip_at = self.v.get("flip_at") or [0.5 + 0.7 * k for k in range(n)]
        self.hi_at = self.v.get("highlight_at", self.flip_at[-1] + 0.6)

    def animate(self, img, t, boil):
        board, hidden = self.v["board"], self.v["hidden"]
        hi = set(self.v.get("highlight", []))
        k_h = D.prog(t, self.hi_at, 0.35) if hi else 0
        for k, c in enumerate(board):
            lift = -24 * D.ease_out_back(k_h) if c in hi else 0
            a = 1 - 0.55 * k_h if hi and c not in hi else 1
            paste_card(img, c, 540 + (k - (len(board) - 1) / 2) * 140, 760 + lift, 120, 166, boil, alpha=a)
        for k, c in enumerate(hidden):
            x = 540 + (k - (len(hidden) - 1) / 2) * 140
            f = D.prog(t, self.flip_at[k], 0.28)
            sx = abs(math.cos(math.pi * f))
            face = f >= 0.5
            lift = -24 * D.ease_out_back(k_h) if c in hi else 0
            paste_card(img, c, x, 1100 + lift, 120, 166, boil, hidden=not face, sx=max(0.02, sx))


class DiceShot(Shot):
    def animate(self, img, t, boil):
        land = self.v.get("land_at", 0.8)
        vals = self.v["values"]
        rng = np.random.default_rng(int(t * 12))
        p = Pen(900, 700, seed=boil, ox=90, oy=520)
        for k, val in enumerate(vals):
            cx = 540 + (k - (len(vals) - 1) / 2) * 260
            if t < land:
                shown = int(rng.integers(1, 7))
                ang = t * 9 + k
                cy = 860 - 120 * abs(math.sin(t * 8 + k))
            else:
                shown, ang = int(val), 0.12 * (k - 0.5)
                cy = 860 - 40 * (1 - D.ease_out_bounce(D.prog(t, land, 0.35)))
            die(p, cx, cy, 190, shown, ang)
        D.paste(img, p.sprite(), 90, 520)


class HexBoardShot(Shot):
    def setup(self):
        super().setup()
        self.centres = dict(zip(HEX_NUMBERS, hex_centres(540, 880, 150)))

    def draw_back(self, p, boil):
        hex_cluster(p, 540, 880, 150)
        return True

    def animate(self, img, t, boil):
        tgt = int(self.v.get("target", 8))
        if tgt not in self.centres:
            return
        x, y = self.centres[tgt]
        k = D.ease_out_bounce(D.prog(t, self.v.get("drop_at", 0.6), 0.5))
        if t < self.v.get("drop_at", 0.6):
            return
        p = Pen(140, 200, seed=boil, ox=x - 70, oy=y - 150)
        robber(p, x, y - 40, 1.6)
        D.paste(img, p.sprite(), x - 70, y - 150 - (1 - k) * 600)


class SignShot(Shot):
    MAX_W = 660  # text width inside the 760px board

    def setup(self):
        super().setup()
        # largest font (74 -> 46px) at which the text fits in <= 4 wrapped lines
        for size in range(74, 44, -4):
            try:
                lines = [w for ln in self.v["text"].split("\n")
                         for w in T.fit_lines(ln, "comic", size, self.MAX_W, max_lines=4)]
            except LayoutError:
                continue
            if len(lines) <= 4:
                break
        else:
            raise LayoutError(f"sign text '{self.v['text']}' does not fit")
        self.lines, self.size = lines, size

    def draw_back(self, p, boil):
        lh = int(self.size * 1.25)
        h = 80 + lh * len(self.lines)
        p.line([(540, 560 + h), (540, 1180)], LW * 2.2, ink=(150, 110, 80, 255))
        p.rrect(160, 560, 760, h, 16, (250, 246, 236, 255))
        for k, ln in enumerate(self.lines):
            p.text(540, 560 + 40 + lh * (k + 0.5), ln, "comic", self.size)
        return True


class VoteShot(Shot):
    def setup(self):
        keys = list(self.cast)
        acc = self.v["accused"]
        others = [k for k in keys if k != acc] + [f"_npc{i}" for i in range(max(0, 3 - (len(keys) - 1)))]
        self.slots = {acc: (540, 1100)}
        for k, who in enumerate(others[:4]):
            a = math.radians(200 + k * (140 / max(1, len(others[:4]) - 1)))
            self.slots[who] = (540 + 360 * math.cos(a), 760 + 260 * math.sin(a) * -1 + 120)
        for who, (x, y) in self.slots.items():
            if who.startswith("_npc"):
                continue
            self.chars.append((who, "full", x, y, 0.5,
                               "shocked" if who == acc else self.v.get("expression", "angry")))
        self.npcs = [w for w in self.slots if w.startswith("_npc")]

    def draw_back(self, p, boil):
        for i, who in enumerate(self.npcs):
            x, y = self.slots[who]
            human_full(p, x, y, 50, "annoyed", 0)
        return True

    def animate(self, img, t, boil):
        acc = self.v["accused"]
        ax, ay = self.slots[acc]
        voters = [w for w in self.slots if w != acc]
        p = Pen(seed=boil)
        drew = False
        for k, w in enumerate(voters):
            if t < 0.4 + k * 0.35:
                continue
            x, y = self.slots[w]
            ex = x + (ax - x) * 0.55
            ey = y + 60 + (ay - y - 60) * 0.55
            p.line([(x, y + 60), (ex, ey)], LW * 1.1, ink=(200, 60, 50, 255))
            drew = True
        if drew:
            D.paste(img, p.sprite(), 0, 0)


SHOTS = {"table": TableShot, "closeup": CloseupShot, "clock_calendar": ClockCalendarShot,
         "calendar_grid": CalendarGridShot, "sun_earth": SunEarthShot, "card_reveal": CardRevealShot,
         "dice": DiceShot, "hex_board": HexBoardShot, "sign": SignShot, "vote": VoteShot}


# ======================================================================
# renderer
# ======================================================================
class DoodleRenderer:
    def __init__(self, spec, tl):
        self.spec, self.tl, self.cast = spec, tl, spec["cast"]
        self.paper = self._paper()
        self.shots = [SHOTS[s.data["visual"]["type"]](s.data, self.cast, s.dur) for s in tl.shots]
        self.hook = self._hook()
        self.hook_shots = spec.get("hook", {}).get("shots", 1)
        self.logo = self._logo()
        self.captions = [(c, self._caption(c.text)) for c in tl.captions]

    # ----- static layers -----
    def _paper(self):
        rng = np.random.default_rng(3)
        base = np.ones((H, W, 3), np.float32) * np.array(PAPER, np.float32)
        grain = rng.normal(0, 3.2, (H // 2, W // 2)).astype(np.float32)
        grain = np.kron(grain, np.ones((2, 2), np.float32))
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        vig = 1 - 0.06 * (((x - W / 2) / W) ** 2 + ((y - H / 2) / H) ** 2) * 4
        img = (base + grain[..., None]) * vig[..., None]
        return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")

    def _hook(self):
        hk = self.spec.get("hook")
        if not hk:
            return None
        parts = []
        if hk.get("title"):
            parts.append(T.sprite(hk["title"], "hand", 96, INK))
        for ln in T.balanced_lines(hk.get("text", ""), "hand", 72, 640, max_lines=3):
            parts.append(T.sprite(ln, "hand", 72, INK))
        return T.stack(parts, gap=4) if parts else None

    def _logo(self):
        mark = T.wordmark(96)
        sub1 = T.sprite("Play vs AI", "sans", 40, (17, 17, 17), wght=700, pad=2)
        sub2 = T.sprite("gghuman.ai", "sans", 40, (17, 17, 17), wght=700, pad=2)
        return T.stack([mark, sub1, sub2], gap=0, align="l")

    def _caption(self, text):
        spr = T.sprite(text, "comic", 86, (255, 255, 255), stroke=7, stroke_fill=(20, 18, 16))
        if spr.width > W - 120:
            raise LayoutError(f"caption '{text}' is {spr.width}px wide")
        return spr

    # ----- per frame -----
    def _expr(self, who, default, t):
        for ln in self.tl.lines:
            if ln.who == who and ln.start <= t < ln.end:
                e = self.spec["shots"][ln.shot]["lines"][ln.idx].get("expression")
                if e:
                    return e
        return default

    def frame(self, i):
        t = i / FPS
        sh = self.tl.shot_at(t)
        shot = self.shots[sh.idx]
        lt = t - sh.start
        boil = (i // BOIL_FRAMES) % BOILS
        img = self.paper.copy()
        if shot.back[boil] is not None:
            img.paste(shot.back[boil], (0, 0), shot.back[boil])
        talk = self.tl.speaking(t, FPS)
        for who, pose, x, y, sc, expr in shot.chars:
            c = self.cast[who]
            amt = talk.get(who, 0.0)
            mouth = 0 if amt < 0.18 else (1 if amt < 0.6 else 2)
            if (i // 2) % 2 and mouth:  # flap
                mouth = max(0, mouth - 1)
            spr, ox, oy = character(c["kind"], c.get("brand", ""), pose, x, y, sc,
                                    self._expr(who, expr, t), mouth, boil)
            img.paste(spr, (ox, oy), spr)
        if shot.front[boil] is not None:
            img.paste(shot.front[boil], (0, 0), shot.front[boil])
        shot.animate(img, lt, boil)
        img = self._camera(img, shot, lt, sh.dur, i)
        # overlays (not affected by camera)
        if self.hook is not None and sh.idx < self.hook_shots:
            D.paste(img, self.hook, W / 2, HOOK_TOP, anchor="mt")
        for c, spr in self.captions:
            if c.start <= t < c.end:
                k = D.prog(t, c.start, 0.08)
                D.paste(img, spr, W / 2, CAPTION_Y, scale=1.12 - 0.12 * k, anchor="mm")
        D.paste(img, self.logo, LOGO_X, LOGO_Y)
        return img

    def _camera(self, img, shot, lt, dur, i):
        z0, z1 = shot.zoom
        z = z0 + (z1 - z0) * D.ease_in_out(D.clamp(lt / max(dur, 0.01)))
        shake = shot.v.get("shake", shot.v.get("expression") == "scream")
        if z <= 1.001 and not shake:
            return img
        fx, fy = shot.focus
        cw, ch = W / z, H / z
        x0 = D.clamp(fx - cw / 2, 0, W - cw)
        y0 = D.clamp(fy - ch / 2, 0, H - ch)
        if shake:
            rng = np.random.default_rng(i)
            amp = 14 / z
            x0 = D.clamp(x0 + rng.uniform(-amp, amp), 0, W - cw)
            y0 = D.clamp(y0 + rng.uniform(-amp, amp), 0, H - ch)
        return img.resize((W, H), Image.BILINEAR, box=(x0, y0, x0 + cw, y0 + ch))

    def sheet_times(self):
        return [s.start + min(s.dur * 0.6, s.dur - 0.1) for s in self.tl.shots]
