"""Deterministic renderer: spec JSON -> 1080x1920 30fps MP4 with audio.

Same spec in => same video out. No model calls happen here, which is why the
render step cannot 'hallucinate' a broken frame: all creative variance lives in
the spec, and the spec is validated before we get here.
"""
import math
import random
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import audio
from .audio import master_filter
from . import draw as D
from . import timeline
from .scenes import SCENES, YOU
from .spec import GAMES, ROSTER, check

HOOK_Y, HOOK_MAX_W, HOOK_SIZE, HOOK_LH = 222, 936, 76, 86
CHAT_Y, CHAT_MAX_Y = 935, 1540           # IG caption/UI covers the bottom ~380px
SAFE_RIGHT = 150                         # IG action buttons on the right edge
BUBBLE_FONT, BUBBLE_LH, PAD_X, PAD_Y = 40, 50, 30, 24
THOUGHT_FONT, THOUGHT_LH = 33, 45
GLYPHS = "▓▒░#%&@$*/<>"


class Renderer:
    def __init__(self, spec):
        self.spec = check(spec)
        self.accent = D.hex2rgb(ROSTER[spec["opponent"]])
        self.tl = timeline.build(spec)
        self.n_frames = int(math.ceil(self.tl["total"] * D.FPS))
        self.scene = SCENES[spec["game"]](spec, self.tl, self.accent)
        self.bg = self._background()
        self.vignette = self._vignette()
        self.grain = self._grain()
        self.topbar = self._topbar()
        self.hook_words = self._layout_hook()
        self._layout_chat()
        self._layout_twist()
        self._build_endcard()

    # ---------------- static layers ----------------
    def _background(self):
        y, x = np.mgrid[0:D.H, 0:D.W].astype(np.float32)
        d = np.sqrt(((x - 540) / 700) ** 2 + ((y - 660) / 620) ** 2)
        glow = np.clip(1 - d, 0, 1) ** 2 * 0.22
        base = np.array(D.BG, np.float32)
        acc = np.array(self.accent, np.float32)
        img = base[None, None, :] * (1 - glow[..., None]) + acc[None, None, :] * glow[..., None]
        return Image.fromarray(img.astype(np.uint8), "RGB")

    def _vignette(self):
        y, x = np.mgrid[0:D.H, 0:D.W].astype(np.float32)
        d = np.sqrt(((x - 540) / 760) ** 2 + ((y - 960) / 1150) ** 2)
        a = (np.clip(d - 0.55, 0, 1) * 1.2 * 255).clip(0, 200).astype(np.uint8)
        im = Image.new("RGBA", (D.W, D.H), (0, 0, 0, 0))
        im.putalpha(Image.fromarray(a, "L"))
        return im

    def _grain(self):
        rng = np.random.default_rng(3)
        out = []
        for _ in range(4):
            n = rng.normal(128, 40, (D.H // 2, D.W // 2)).clip(0, 255).astype(np.uint8)
            g = Image.fromarray(n, "L").resize((D.W, D.H), Image.NEAREST).convert("RGB")
            out.append(g)
        return out

    def _topbar(self):
        im = Image.new("RGBA", (D.W, 80), (0, 0, 0, 0))
        dot = D.circle(16, (255, 70, 70, 255))
        D.paste(im, dot, 72, 40, anchor="lm")
        game = D.text_sprite(GAMES[self.spec["game"]], "xbold", 28, D.INK)
        D.paste(im, game, 98, 40, anchor="lm")
        vs = D.text_sprite(f"vs {self.spec['opponent'].upper()}", "xbold", 28, self.accent)
        D.paste(im, vs, 98 + game.width + 10, 40, anchor="lm")
        url = D.text_sprite("gghuman.ai", "semibold", 28, D.MUTED)
        D.paste(im, url, D.W - SAFE_RIGHT + 60, 40, anchor="rm")
        return im

    def _layout_hook(self):
        f = D.font("black", HOOK_SIZE)
        lines = D.wrap(self.spec["hook"], f, HOOK_MAX_W)
        if len(lines) > 2:
            raise D.LayoutError(f"hook wraps to {len(lines)} lines (max 2)")
        hl = self.spec["hook_highlight"].split()
        words, idx = [], 0
        for li, line in enumerate(lines):
            x = 72
            for w in line.split():
                is_hl = any(w.strip(".,!?'\"") == h.strip(".,!?'\"") for h in hl)
                spr = D.text_sprite(w, "black", HOOK_SIZE, self.accent if is_hl else D.INK)
                words.append((spr, x - 4, HOOK_Y + li * HOOK_LH, idx))
                x += f.getlength(w + " ")
                idx += 1
        return words

    def _bubble(self, text, kind, shown=None, t=0.0):
        """Render a chat bubble sprite. kind: you | said | thought."""
        if kind == "thought":
            f, lh, max_w = D.font("mono", THOUGHT_FONT), THOUGHT_LH, D.W - 72 - SAFE_RIGHT - PAD_X * 2
        else:
            f, lh, max_w = D.font("semibold", BUBBLE_FONT), BUBBLE_LH, 760 - PAD_X * 2
        lines = D.wrap(text, f, max_w)
        if len(lines) > 3:
            raise D.LayoutError(f"{kind} text wraps to {len(lines)} lines (max 3)")
        tw = max(f.getlength(l) for l in lines)
        bw = int(tw + PAD_X * 2) if kind != "thought" else D.W - 72 - SAFE_RIGHT
        bh = int(len(lines) * lh + PAD_Y * 2 - (lh - f.size) * 0.6)
        label_h = 40
        im = Image.new("RGBA", (bw, bh + label_h), (0, 0, 0, 0))
        if kind == "you":
            fill, tcol, label, lcol = YOU + (255,), (14, 18, 10), "YOU", YOU
            box = D.rrect(bw, bh, 30, fill)
        elif kind == "said":
            fill, tcol = self.accent + (255,), D.on_color(self.accent)
            label, lcol = f"{self.spec['opponent'].upper()} SAID", self.accent
            box = D.rrect(bw, bh, 30, fill)
        else:
            tcol, label, lcol = (255, 120, 120), "WHAT IT WAS ACTUALLY THINKING", D.RED
            box = D.rrect(bw, bh, 18, (34, 10, 14, 245), (255, 70, 70, 255), 3)
        lab = D.text_sprite(label, "xbold", 24, lcol)
        lx = 0 if kind != "said" else bw - lab.width
        D.paste(im, lab, lx, 0)
        if kind == "thought" and int(t * 3) % 2 == 0:
            D.paste(im, D.circle(14, (255, 70, 70, 255)), lab.width + 6, lab.height / 2, anchor="lm")
        D.paste(im, box, 0, label_h)
        d = ImageDraw.Draw(im)
        n = len(text) if shown is None else shown
        rng = random.Random(int(t * D.FPS))
        consumed = 0
        for li, line in enumerate(lines):
            y = label_h + PAD_Y + li * lh
            if kind == "thought":
                out = []
                for ch in line:
                    if consumed < n or ch == " ":
                        out.append(ch)
                    elif consumed < n + 14:
                        out.append(rng.choice(GLYPHS))
                    else:
                        out.append(" ")
                    consumed += 1
                d.text((PAD_X, y), "".join(out), font=f, fill=tcol)
            else:
                part = line[:max(0, n - consumed)]
                consumed += len(line)
                if part:
                    d.text((PAD_X, y), part, font=f, fill=tcol)
            consumed += 1  # the space eaten by wrapping
        return im

    def _layout_chat(self):
        s = self.spec
        self.chat = []
        y = CHAT_Y
        for kind, key, start, end in [("you", "you_line", "you", "you_end"),
                                      ("said", "ai_said", "said", "said_end"),
                                      ("thought", "ai_thought", "thought", "thought_end")]:
            full = self._bubble(s[key], kind)
            x = 72 if kind != "said" else D.W - SAFE_RIGHT - full.width
            self.chat.append(dict(kind=kind, text=s[key], x=x, y=y, h=full.height,
                                  start=self.tl[start], end=self.tl[end], full=full))
            y += full.height + 22
        if y - 22 > CHAT_MAX_Y:
            raise D.LayoutError(f"chat stack ends at y={y - 22}, past safe zone {CHAT_MAX_Y}")

    def _layout_twist(self):
        f = D.font("black", 92)
        lines = D.wrap(self.spec["twist"], f, D.W - 144 - 60)
        if len(lines) > 2:
            raise D.LayoutError("twist wraps to more than 2 lines")
        self.twist_lines = [D.text_sprite(l, "black", 92, (255, 255, 255)) for l in lines]
        lab = self.spec.get("twist_label", "")
        self.twist_label = D.text_sprite(lab, "xbold", 30, D.RED) if lab else None

    def _build_endcard(self):
        s = self.spec
        self.cta = []
        for i, line in enumerate(s["cta"]):
            col = self.accent if i == 2 else D.INK
            self.cta.append(D.text_sprite(line, "black", 112, col))
        gg = D.text_sprite("GG", "black", 92, D.INK)
        hu = D.text_sprite("human", "bolditalic", 92, D.INK)
        wm = Image.new("RGBA", (gg.width + hu.width - 8, max(gg.height, hu.height)), (0, 0, 0, 0))
        D.paste(wm, gg, 0, 0)
        D.paste(wm, hu, gg.width - 8, 0)
        self.wordmark = wm
        pill = D.rrect(700, 104, 52, (255, 255, 255, 255)).copy()
        D.paste(pill, D.text_sprite("PLAY NOW  →  gghuman.ai", "black", 40, (12, 12, 16)), 350, 52, anchor="mm")
        self.pill = pill
        self.games = D.text_sprite("Settler Rush  ·  Poker  ·  Last Vote", "semibold", 32, D.MUTED)
        self.vs = D.text_sprite(f"{len(ROSTER)} frontier AIs are waiting at the table.", "medium", 32, D.MUTED)

    # ---------------- per-frame ----------------
    def frame(self, i):
        t = i / D.FPS
        tl = self.tl
        img = self.bg.copy()
        D.paste(img, self.topbar, 0, 120, alpha=D.prog(t, 0, 0.3))
        for spr, x, y, idx in self.hook_words:
            k = D.prog(t, tl["hook"] + idx * 0.07, 0.32)
            if k > 0:
                D.paste(img, spr, x, y + (1 - D.ease_out_back(k)) * 36, alpha=min(1, k * 1.5))
        self.scene.render(img, t)
        # chat
        fade = 1 - D.ease_out_cubic(D.prog(t, tl["twist"] - 0.1, 0.2))
        for b in self.chat:
            if t < b["start"]:
                continue
            k = D.ease_out_cubic(D.prog(t, b["start"], 0.25))
            if t >= b["end"] + 0.1:
                spr = b["full"]
            elif b["kind"] == "thought":
                shown = int(max(0, t - b["start"] - 0.35) * timeline.DECODE_CPS)
                spr = self._bubble(b["text"], "thought", shown, t)
            else:
                shown = int((t - b["start"]) * timeline.TYPE_CPS)
                spr = self._bubble(b["text"], b["kind"], shown, t)
            if b["kind"] == "thought" and t >= b["end"] + 0.1:
                spr = self._bubble(b["text"], "thought", None, t)  # keep the blinking dot alive
            D.paste(img, spr, b["x"], b["y"] + (1 - k) * 40, alpha=k * fade)
        # twist
        kt = D.prog(t, tl["twist"], 0.32)
        if kt > 0:
            cy = (CHAT_Y + CHAT_MAX_Y) / 2
            n = len(self.twist_lines)
            sc = 1.35 - 0.35 * D.ease_out_cubic(kt)
            if self.twist_label:
                D.paste(img, self.twist_label, 72, cy - n * 52 - 52, alpha=kt, anchor="lm")
            for j, ls in enumerate(self.twist_lines):
                yy = cy + (j - (n - 1) / 2) * 104
                bar_w = int((ls.width + 30) * D.ease_out_cubic(D.prog(t, tl["twist"] + 0.15 + j * 0.08, 0.3)))
                if bar_w > 2:
                    D.paste(img, D.rrect(bar_w, 96, 8, self.accent + (235,)), 58, yy, anchor="lm")
                D.paste(img, ls, 72, yy, alpha=kt, scale=sc, anchor="lm")
        flash = 1 - D.prog(t, tl["twist"], 0.18) if t >= tl["twist"] else 0
        # end card crossfade
        ke = D.ease_in_out(D.prog(t, tl["end"], 0.4))
        if ke > 0:
            end = self._endcard(t)
            img = Image.blend(img, end, ke) if ke < 1 else end
        # grain + vignette
        img.paste(self.vignette, (0, 0), self.vignette)
        img = Image.blend(img, self.grain[i % 4], 0.035)
        if flash > 0:
            img = Image.blend(img, Image.new("RGB", img.size, (255, 255, 255)), 0.55 * flash)
        return img

    def _endcard(self, t):
        tl = self.tl
        im = Image.new("RGB", (D.W, D.H), (7, 7, 10))
        for j, spr in enumerate(self.cta):
            k = D.prog(t, tl["cta"][j], 0.3)
            if k > 0:
                D.paste(im, spr, 72, 560 + j * 136 + (1 - D.ease_out_back(k)) * 50, alpha=min(1, k * 1.6))
        k = D.prog(t, tl["url"], 0.4)
        if k > 0:
            e = D.ease_out_back(k)
            D.paste(im, self.wordmark, 72, 1010, alpha=min(1, k * 2))
            D.paste(im, self.pill, 72, 1170, alpha=min(1, k * 2), scale=0.85 + 0.15 * e)
            D.paste(im, self.games, 72, 1310, alpha=D.prog(t, tl["url"] + 0.25, 0.3))
            D.paste(im, self.vs, 72, 1360, alpha=D.prog(t, tl["url"] + 0.4, 0.3))
        return im

    # ---------------- output ----------------
    def render(self, out_path, preview_times=None):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        wav = out_path.with_suffix(".wav")
        audio.render(self.spec, self.tl, wav)
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{D.W}x{D.H}", "-r", str(D.FPS), "-i", "-",
               "-i", str(wav),
               "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
               "-af", master_filter(wav), "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
               "-shortest", "-movflags", "+faststart", str(out_path)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for i in range(self.n_frames):
            proc.stdin.write(self.frame(i).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg encode failed")
        wav.unlink(missing_ok=True)
        cap = out_path.with_suffix(".caption.txt")
        cap.write_text(self.spec["caption"] + "\n\n" + " ".join(self.spec.get("hashtags", [])) + "\n")
        return out_path


def render(spec, out_path):
    return Renderer(spec).render(out_path)
