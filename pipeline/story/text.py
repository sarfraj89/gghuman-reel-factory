"""Fonts and text sprites for the storyboard formats (variable fonts, tracking,
outline, shadow, glow)."""
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..draw import LayoutError, hex2rgb

FONTS_DIR = Path(__file__).resolve().parent.parent.parent / "fonts"
FONT_FILES = {
    "serif": "Cinzel-VF.ttf",           # human / medieval captions, titles
    "tech": "Orbitron-VF.ttf",          # AI captions, HUD
    "sans": "Montserrat-VF.ttf",        # GGhuman wordmark
    "hand": "PatrickHand-Regular.ttf",  # doodle hook text
    "comic": "Gaegu-Bold.ttf",          # doodle captions, labels
    "symbol": "DejaVuSans-Bold.ttf",    # card suits
}


@lru_cache(None)
def font(name, size, wght=None):
    f = ImageFont.truetype(str(FONTS_DIR / FONT_FILES[name]), size)
    if wght is not None:
        f.set_variation_by_axes([wght])
    return f


def color(c, alpha=255):
    if isinstance(c, str):
        c = hex2rgb(c)
    return tuple(c[:3]) + (alpha,)


def measure(text, f, tracking=0.0):
    if not tracking:
        return f.getlength(text)
    return sum(f.getlength(ch) for ch in text) + tracking * f.size * (len(text) - 1)


@lru_cache(maxsize=4096)
def sprite(text, name, size, fill, wght=None, tracking=0.0, stroke=0, stroke_fill=(0, 0, 0),
           shadow=0, glow=0, glow_fill=None, pad=None):
    """Render one line of text to an RGBA sprite. `fill`/`stroke_fill` are RGB(A) tuples.
    shadow: blur radius of a soft dark drop shadow. glow: blur radius of a coloured halo."""
    f = font(name, size, wght)
    fill, stroke_fill = color(fill), color(stroke_fill)
    asc, desc = f.getmetrics()
    pad = pad if pad is not None else stroke + max(shadow, glow) * 2 + 4
    w = int(measure(text, f, tracking)) + pad * 2 + 2
    h = asc + desc + pad * 2
    base = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    def _draw(img, fl, sw, sf):
        d = ImageDraw.Draw(img)
        if not tracking:
            d.text((pad, pad), text, font=f, fill=fl, stroke_width=sw, stroke_fill=sf)
            return
        x = pad
        for ch in text:
            d.text((x, pad), ch, font=f, fill=fl, stroke_width=sw, stroke_fill=sf)
            x += f.getlength(ch) + tracking * f.size

    if shadow:
        sh = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        _draw(sh, (0, 0, 0, 200), stroke, (0, 0, 0, 200))
        sh = sh.filter(ImageFilter.GaussianBlur(shadow))
        base.alpha_composite(sh, (0, max(1, shadow // 2)))
    if glow:
        gl = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        _draw(gl, color(glow_fill or fill, 170), stroke + 2, color(glow_fill or fill, 170))
        base.alpha_composite(gl.filter(ImageFilter.GaussianBlur(glow)))
    _draw(base, fill, stroke, stroke_fill)
    return base


def fit_lines(text, name, size, max_w, wght=None, tracking=0.0, max_lines=2):
    """Greedy word wrap that raises LayoutError instead of shrinking."""
    f = font(name, size, wght)
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if measure(trial, f, tracking) <= max_w:
            cur = trial
            continue
        if not cur or measure(word, f, tracking) > max_w:
            raise LayoutError(f"'{word}' wider than {max_w}px at {size}px {name}")
        lines.append(cur)
        cur = word
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        raise LayoutError(f"'{text}' wraps to {len(lines)} lines (max {max_lines})")
    return lines


def stack(sprites, gap=0, align="m"):
    """Stack sprites vertically into one sprite."""
    w = max(s.width for s in sprites)
    h = sum(s.height for s in sprites) + gap * (len(sprites) - 1)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    y = 0
    for s in sprites:
        x = {"l": 0, "m": (w - s.width) // 2, "r": w - s.width}[align]
        out.alpha_composite(s, (x, y))
        y += s.height + gap
    return out


def hstack(sprites, gap=0):
    w = sum(s.width for s in sprites) + gap * (len(sprites) - 1)
    h = max(s.height for s in sprites)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    x = 0
    for s in sprites:
        out.alpha_composite(s, (x, (h - s.height) // 2))
        x += s.width + gap
    return out


def wordmark(size, gg=(17, 17, 17), human=(88, 88, 88), suffix=None, suffix_fill=None):
    """GGhuman wordmark: bold 'GG' + lighter 'human' (+ optional '.ai')."""
    parts = [sprite("GG", "sans", size, gg, wght=800, pad=2),
             sprite("human", "sans", size, human, wght=700, pad=2)]
    if suffix:
        parts.append(sprite(suffix, "sans", size, suffix_fill or human, wght=700, pad=2))
    return hstack(parts, gap=-4)


def balanced_lines(text, name, size, max_w, wght=None, max_lines=3):
    """Like fit_lines, but splits into the fewest lines with the most even widths
    (hook text reads better as 'The AI that / slow-rolls you')."""
    lines = fit_lines(text, name, size, max_w, wght, max_lines=max_lines)
    if len(lines) != 2:
        return lines
    f, words = font(name, size, wght), text.split()
    best = min(range(1, len(words)),
               key=lambda k: max(measure(" ".join(words[:k]), f), measure(" ".join(words[k:]), f)))
    return [" ".join(words[:best]), " ".join(words[best:])]
