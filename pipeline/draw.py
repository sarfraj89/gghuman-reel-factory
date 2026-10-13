"""Drawing primitives: fonts, anti-aliased sprites, easing, text layout."""
import math
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1080, 1920, 30
SS = 3  # supersampling factor for shapes (Pillow shapes are not anti-aliased)

FONTS = {
    "black": "/usr/share/fonts/opentype/inter/Inter-Black.otf",
    "xbold": "/usr/share/fonts/opentype/inter/Inter-ExtraBold.otf",
    "bold": "/usr/share/fonts/opentype/inter/Inter-Bold.otf",
    "bolditalic": "/usr/share/fonts/opentype/inter/Inter-BlackItalic.otf",
    "semibold": "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf",
    "medium": "/usr/share/fonts/opentype/inter/Inter-Medium.otf",
    "mono": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "symbol": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
}

BG = (10, 10, 14)
INK = (245, 244, 240)
MUTED = (150, 150, 162)
RED = (255, 70, 70)


def _ffprobe():
    import shutil
    venv = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "ffprobe"
    return shutil.which("ffprobe") or shutil.which("ffmpeg.ffprobe") or (str(venv) if venv.exists() else "ffprobe")


FFPROBE = _ffprobe()


class LayoutError(Exception):
    """Text does not fit its box. Treated as a spec failure, never auto-shrunk."""


LOCAL_FONTS = Path(__file__).resolve().parent.parent / "fonts"


@lru_cache(None)
def font(name, size):
    path = Path(FONTS[name])
    if not path.exists() and (LOCAL_FONTS / path.name).exists():
        path = LOCAL_FONTS / path.name  # fallback when fonts-inter isn't installed system-wide
    return ImageFont.truetype(str(path), size)


def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def luminance(rgb):
    return (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255


def on_color(rgb):
    return (14, 14, 18) if luminance(rgb) > 0.55 else (255, 255, 255)


def mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


# ---------- easing ----------
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def prog(t, start, dur):
    return clamp((t - start) / dur) if dur > 0 else float(t >= start)


def ease_out_cubic(x):
    return 1 - (1 - x) ** 3


def ease_in_out(x):
    return 3 * x * x - 2 * x * x * x


def ease_out_back(x, s=1.7):
    x -= 1
    return x * x * ((s + 1) * x + s) + 1


def ease_out_bounce(x):
    n, d = 7.5625, 2.75
    if x < 1 / d:
        return n * x * x
    if x < 2 / d:
        x -= 1.5 / d
        return n * x * x + 0.75
    if x < 2.5 / d:
        x -= 2.25 / d
        return n * x * x + 0.9375
    x -= 2.625 / d
    return n * x * x + 0.984375


# ---------- sprites ----------
def canvas(w, h):
    return Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))


def down(im, w, h):
    return im.resize((w, h), Image.LANCZOS)


@lru_cache(maxsize=1024)
def rrect(w, h, r, fill, outline=None, width=0):
    im = canvas(w, h)
    ImageDraw.Draw(im).rounded_rectangle(
        [0, 0, w * SS - 1, h * SS - 1], r * SS, fill=fill, outline=outline, width=width * SS)
    return down(im, w, h)


@lru_cache(maxsize=256)
def circle(d, fill, outline=None, width=0):
    im = canvas(d, d)
    ImageDraw.Draw(im).ellipse([0, 0, d * SS - 1, d * SS - 1], fill=fill,
                               outline=outline, width=width * SS)
    return down(im, d, d)


@lru_cache(maxsize=512)
def text_sprite(text, fname, size, fill, pad=4):
    f = font(fname, size)
    l, t, r, b = f.getbbox(text)
    asc, desc = f.getmetrics()
    im = Image.new("RGBA", (r - l + pad * 2, asc + desc + pad * 2), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((pad - l, pad), text, font=f, fill=fill)
    return im


_LUTS = {}


def with_alpha(sprite, alpha):
    if alpha >= 0.999:
        return sprite
    k = int(alpha * 100)
    lut = _LUTS.get(k)
    if lut is None:
        lut = _LUTS[k] = [int(i * k / 100) for i in range(256)]
    s = sprite.copy()
    s.putalpha(sprite.getchannel("A").point(lut))
    return s


def paste(base, sprite, x, y, alpha=1.0, scale=1.0, anchor="lt", sx=None):
    """Composite an RGBA sprite onto an RGB frame. anchor: l/m/r + t/m/b."""
    if alpha <= 0.003:
        return
    sw = scale if sx is None else sx
    if sw != 1.0 or scale != 1.0:
        nw = max(1, int(round(sprite.width * sw)))
        nh = max(1, int(round(sprite.height * scale)))
        sprite = sprite.resize((nw, nh), Image.BILINEAR)
    sprite = with_alpha(sprite, alpha)
    ax, ay = anchor[0], anchor[1]
    if ax == "m":
        x -= sprite.width / 2
    elif ax == "r":
        x -= sprite.width
    if ay == "m":
        y -= sprite.height / 2
    elif ay == "b":
        y -= sprite.height
    base.paste(sprite, (int(round(x)), int(round(y))), sprite)


def wrap(text, f, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if f.getlength(trial) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            if f.getlength(w) > max_w:
                raise LayoutError(f"word '{w}' wider than {max_w}px")
            cur = w
    if cur:
        lines.append(cur)
    return lines


def hexagon(cx, cy, r, rot=-30):
    return [(cx + r * math.cos(math.radians(60 * i + rot)),
             cy + r * math.sin(math.radians(60 * i + rot))) for i in range(6)]
