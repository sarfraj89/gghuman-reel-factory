"""Game scenes. Each one draws the middle band of the reel (y = SCENE_Y .. +SCENE_H)
and owns its own 'event' animation (robber drop / fold+reveal / vote pile-on)."""
import math
import random

from PIL import Image, ImageDraw

from . import draw as D
from .spec import ROSTER, YOU_COLOR

SCENE_Y, SCENE_H = 420, 480
YOU = D.hex2rgb(YOU_COLOR)


def color_of(name):
    return YOU if name == "YOU" else D.hex2rgb(ROSTER[name])


class Scene:
    def __init__(self, spec, tl, accent):
        self.spec, self.tl, self.accent = spec, tl, accent
        self.p = spec["scene"]

    def intro(self, t):
        return D.ease_out_cubic(D.prog(t, self.tl["scene_in"], 0.55))

    def shake(self, t):
        return 0, 0

    def render(self, frame, t):
        raise NotImplementedError


def _stroked(text, fname, size, fill, stroke=(10, 10, 14), sw=6):
    f = D.font(fname, size)
    l, t_, r, b = f.getbbox(text, stroke_width=sw)
    im = Image.new("RGBA", (r - l + 8, b - t_ + 8), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((4 - l, 4 - t_), text, font=f, fill=fill,
                            stroke_width=sw, stroke_fill=stroke)
    return im


# ======================================================================
# Settler Rush (Catan-like)
# ======================================================================
RES = {
    "forest": (46, 125, 72), "pasture": (139, 195, 74), "fields": (242, 193, 78),
    "hills": (200, 85, 61), "mountains": (141, 153, 174), "desert": (226, 208, 160),
}
RES_LABEL = {"forest": "WOOD", "pasture": "WOOL", "fields": "WHEAT",
             "hills": "BRICK", "mountains": "ORE", "desert": ""}


class SettlerScene(Scene):
    R = 54

    def __init__(self, spec, tl, accent):
        super().__init__(spec, tl, accent)
        rng = random.Random(self.p.get("seed", 1))
        coords = [(q, r) for q in range(-2, 3) for r in range(max(-2, -q - 2), min(2, -q + 2) + 1)]
        tiles = ["forest"] * 4 + ["pasture"] * 4 + ["fields"] * 4 + ["hills"] * 3 + ["mountains"] * 3
        nums = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]
        rng.shuffle(tiles)
        rng.shuffle(nums)
        self.cx, self.cy = 540, SCENE_H // 2
        self.hexes = []
        ti = 0
        for q, r in coords:
            x = self.cx + self.R * math.sqrt(3) * (q + r / 2)
            y = self.cy + self.R * 1.5 * r
            if (q, r) == (0, 0):
                self.hexes.append(dict(x=x, y=y, res="desert", num=None, ring=0))
            else:
                self.hexes.append(dict(x=x, y=y, res=tiles[ti], num=nums[ti],
                                       ring=max(abs(q), abs(r), abs(-q - r))))
                ti += 1
        tgt = self.p["target_number"]
        cands = [h for h in self.hexes if h["num"] == tgt]
        if not cands:  # force the target number onto a ring-1 hex
            h = next(h for h in self.hexes if h["ring"] == 1)
            h["num"] = tgt
            cands = [h]
        cands.sort(key=lambda h: (h["ring"] != 1, h["y"]))
        self.target = cands[0]
        self.robber_home = next(h for h in self.hexes if h["res"] == "desert")
        far = max(self.hexes, key=lambda h: (h["x"] - self.target["x"]) ** 2 + (h["y"] - self.target["y"]) ** 2)
        self.ai_hex = far
        self.static = self._build_static()
        self.robber = self._robber_sprite()
        self.glow = self._hex_outline(self.accent, 7)
        self.dark = self._hex_fill((0, 0, 0, 255))
        self.loss = _stroked(self.p.get("loss", "-2"), "black", 58, D.RED)

    def _poly(self, d, cx, cy, r, **kw):
        d.polygon([(x * D.SS, y * D.SS) for x, y in D.hexagon(cx, cy, r)], **kw)

    def _build_static(self):
        S = D.SS
        im = Image.new("RGBA", (D.W * S, SCENE_H * S), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        sea = [(x * S, y * S) for x, y in D.hexagon(self.cx, self.cy, self.R * 4.75, rot=0)]
        d.polygon(sea, fill=(23, 70, 110, 255))
        sea_in = [(x * S, y * S) for x, y in D.hexagon(self.cx, self.cy, self.R * 4.5, rot=0)]
        d.polygon(sea_in, fill=(31, 92, 140, 255))
        for h in self.hexes:
            base = RES[h["res"]]
            self._poly(d, h["x"], h["y"], self.R - 1, fill=D.mix(base, (0, 0, 0), 0.35) + (255,))
            self._poly(d, h["x"], h["y"], self.R - 6, fill=base + (255,))
            if h["num"]:
                rr = 19 * S
                cx, cy = h["x"] * S, h["y"] * S
                d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(246, 238, 218, 255),
                          outline=(60, 50, 40, 255), width=2 * S)
                f = D.font("black", 22 * S)
                col = (200, 40, 40) if h["num"] in (6, 8) else (30, 30, 30)
                d.text((cx, cy), str(h["num"]), font=f, fill=col, anchor="mm")
        # pieces: your settlement + road on target, AI settlement + road far side
        self._piece(d, self.target, 5, YOU)
        self._road(d, self.target, 5, 4, YOU)
        self._piece(d, self.ai_hex, 2, self.accent)
        self._road(d, self.ai_hex, 2, 1, self.accent)
        return im.resize((D.W, SCENE_H), Image.LANCZOS)

    def _corner(self, h, i):
        return D.hexagon(h["x"], h["y"], self.R)[i]

    def _road(self, d, h, i, j, col):
        S = D.SS
        a, b = self._corner(h, i), self._corner(h, j)
        d.line([a[0] * S, a[1] * S, b[0] * S, b[1] * S], fill=(15, 15, 15, 255), width=13 * S)
        d.line([a[0] * S, a[1] * S, b[0] * S, b[1] * S], fill=col + (255,), width=8 * S)

    def _piece(self, d, h, i, col):
        S = D.SS
        x, y = self._corner(h, i)
        pts = [(x - 13, y + 11), (x - 13, y - 3), (x, y - 15), (x + 13, y - 3), (x + 13, y + 11)]
        d.polygon([(px * S, py * S) for px, py in pts], fill=col + (255,), outline=(15, 15, 15, 255), width=3 * S)

    def _hex_outline(self, col, wdt):
        r = self.R + 6
        size = int(r * 2 + 20)
        im = D.canvas(size, size)
        pts = [(x * D.SS, y * D.SS) for x, y in D.hexagon(size / 2, size / 2, r)]
        ImageDraw.Draw(im).line(pts + [pts[0]], fill=col + (255,), width=wdt * D.SS, joint="curve")
        return D.down(im, size, size)

    def _hex_fill(self, rgba):
        size = int(self.R * 2 + 4)
        im = D.canvas(size, size)
        pts = [(x * D.SS, y * D.SS) for x, y in D.hexagon(size / 2, size / 2, self.R - 6)]
        ImageDraw.Draw(im).polygon(pts, fill=rgba)
        return D.down(im, size, size)

    def _robber_sprite(self):
        w, h = 56, 92
        im = D.canvas(w, h)
        d = ImageDraw.Draw(im)
        S = D.SS
        d.polygon([(8 * S, 88 * S), (48 * S, 88 * S), (38 * S, 40 * S), (18 * S, 40 * S)],
                  fill=(28, 28, 32, 255), outline=(240, 240, 240, 255), width=2 * S)
        d.ellipse([10 * S, 4 * S, 46 * S, 40 * S], fill=(28, 28, 32, 255),
                  outline=(240, 240, 240, 255), width=2 * S)
        d.ellipse([4 * S, 80 * S, 52 * S, 91 * S], fill=(28, 28, 32, 255))
        return D.down(im, w, h)

    def shake(self, t):
        land = self.tl["event"] + 0.45
        k = D.prog(t, land, 0.35)
        if 0 < k < 1:
            amp = 14 * (1 - k)
            return int(amp * math.sin(t * 90)), int(amp * math.cos(t * 70))
        return 0, 0

    def render(self, frame, t):
        a = self.intro(t)
        sx, sy = self.shake(t)
        y0 = SCENE_Y + sy
        D.paste(frame, self.static, 540 + sx, y0 + SCENE_H / 2, alpha=a,
                scale=0.9 + 0.1 * a, anchor="mm")
        if a < 1:
            return
        tx, ty = self.target["x"] + sx, y0 + self.target["y"]
        ev = self.tl["event"]
        # foreshadow: target hex pulses while the AI is "thinking"
        if self.tl["thought"] <= t < ev + 0.5:
            pulse = 0.55 + 0.45 * math.sin((t - self.tl["thought"]) * 7)
            D.paste(frame, self.glow, tx, ty, alpha=pulse, anchor="mm")
        # robber: sits on desert, then drops onto your hex
        home = self.robber_home
        if t < ev:
            D.paste(frame, self.robber, home["x"] + sx, y0 + home["y"] + 12, anchor="mb", scale=0.8)
        else:
            k = D.prog(t, ev, 0.45)
            drop = (1 - D.ease_out_bounce(k)) * 330
            D.paste(frame, self.dark, tx, ty, alpha=0.55 * D.prog(t, ev + 0.35, 0.3), anchor="mm")
            D.paste(frame, self.robber, tx, ty + 14 - drop, anchor="mb")
            kl = D.prog(t, ev + 0.5, 1.6)
            if 0 < kl < 1:
                al = min(1, kl * 5) * (1 - D.prog(kl, 0.7, 0.3))
                D.paste(frame, self.loss, tx + 70, ty - 40 - 70 * D.ease_out_cubic(kl), alpha=al, anchor="lm")


# ======================================================================
# Poker
# ======================================================================
RED_SUITS = "♥♦"


def _card(code, face_up, back_col, w=104, h=146):
    im = D.canvas(w, h)
    d = ImageDraw.Draw(im)
    S = D.SS
    if face_up:
        d.rounded_rectangle([0, 0, w * S - 1, h * S - 1], 12 * S, fill=(250, 248, 242, 255),
                            outline=(30, 30, 30, 255), width=2 * S)
        rank, suit = code[:-1], code[-1]
        col = (205, 35, 45) if suit in RED_SUITS else (22, 22, 26)
        f1 = D.font("symbol", int(h * 0.27) * S)
        f2 = D.font("symbol", int(h * 0.42) * S)
        d.text((int(w * 0.12) * S, int(h * 0.05) * S), rank, font=f1, fill=col)
        d.text((w * S / 2, h * 0.64 * S), suit, font=f2, fill=col, anchor="mm")
    else:
        d.rounded_rectangle([0, 0, w * S - 1, h * S - 1], 12 * S, fill=(245, 245, 245, 255))
        d.rounded_rectangle([5 * S, 5 * S, (w - 5) * S, (h - 5) * S], 9 * S, fill=back_col + (255,))
        for i in range(-h, w + h, 14):
            d.line([(i * S, 5 * S), ((i + h) * S, (h - 5) * S)], fill=D.mix(back_col, (0, 0, 0), 0.25) + (255,), width=3 * S)
        d.rounded_rectangle([5 * S, 5 * S, (w - 5) * S, (h - 5) * S], 9 * S, outline=(245, 245, 245, 255), width=3 * S)
    return D.down(im, w, h)


class PokerScene(Scene):
    def __init__(self, spec, tl, accent):
        super().__init__(spec, tl, accent)
        self.table = self._table()
        self.board = [_card(c, True, accent) for c in self.p["board"]]
        self.back = _card("A♠", False, accent, 88, 124)
        self.ai_face = [_card(c, True, accent, 88, 124) for c in self.p["ai_hand"]]
        self.you = [_card(c, True, accent, 88, 124) for c in self.p["you_hand"]]
        self.chip = D.circle(46, accent + (255,), (255, 255, 255, 255), 4)
        self.chip_edge = D.circle(46, D.mix(accent, (0, 0, 0), 0.4) + (255,), (255, 255, 255, 255), 4)
        self.allin = D.rrect(170, 52, 26, (255, 70, 70, 255))
        self.allin_txt = D.text_sprite("ALL IN", "black", 30, (255, 255, 255))
        self.pot = D.text_sprite("POT " + self.p.get("pot", ""), "xbold", 30, (255, 255, 255))
        self.fold = _stroked("FOLD", "black", 54, (255, 255, 255), stroke=(200, 30, 40), sw=5)
        self.ai_tag = D.text_sprite(spec["opponent"].upper(), "xbold", 28, accent)
        self.you_tag = D.text_sprite("YOU", "xbold", 28, YOU)
        self.ring = D.rrect(196, 136, 20, None, accent + (255,), 6)

    def _table(self):
        w, h = 960, 420
        im = D.canvas(w, h)
        d = ImageDraw.Draw(im)
        S = D.SS
        d.rounded_rectangle([0, 0, w * S - 1, h * S - 1], 210 * S, fill=(52, 34, 22, 255))
        d.rounded_rectangle([20 * S, 20 * S, (w - 20) * S, (h - 20) * S], 190 * S, fill=(14, 92, 62, 255))
        d.rounded_rectangle([44 * S, 44 * S, (w - 44) * S, (h - 44) * S], 168 * S,
                            outline=(40, 130, 92, 255), width=3 * S)
        return D.down(im, w, h)

    def render(self, frame, t):
        a = self.intro(t)
        y0 = SCENE_Y
        cx = 540
        D.paste(frame, self.table, cx, y0 + 240, alpha=a, scale=0.92 + 0.08 * a, anchor="mm")
        if a <= 0:
            return
        tl = self.tl
        # board cards deal in
        for i, c in enumerate(self.board):
            k = D.ease_out_back(D.prog(t, tl["scene_in"] + 0.25 + i * 0.09, 0.35))
            if k <= 0:
                continue
            x = cx + (i - 2) * 118
            D.paste(frame, c, cx + (x - cx) * k, y0 + 240 - (1 - k) * 160, alpha=min(1, k), anchor="mm")
        # AI hole cards (top), flip at reveal
        rev = tl["event"] + 0.55
        for i in range(2):
            x = cx - 48 + i * 96
            y = y0 + 34
            kf = D.prog(t, rev + i * 0.1, 0.36)
            if kf < 0.5:
                D.paste(frame, self.back, x, y, sx=1 - kf * 2, anchor="mt", alpha=a)
            else:
                D.paste(frame, self.ai_face[i], x, y, sx=(kf - 0.5) * 2, anchor="mt")
        if t > rev + 0.5:
            D.paste(frame, self.ring, cx, y0 + 34 + 62, anchor="mm",
                    alpha=0.6 + 0.4 * math.sin((t - rev) * 8))
        D.paste(frame, self.ai_tag, cx - 120, y0 + 96, anchor="rm", alpha=a)
        # your cards (bottom), FOLD stamp at event
        dim = 1 - 0.55 * D.prog(t, tl["event"], 0.25)
        for i in range(2):
            D.paste(frame, self.you[i], cx - 48 + i * 96, y0 + 470, anchor="mb", alpha=a * dim)
        D.paste(frame, self.you_tag, cx - 120, y0 + 408, anchor="rm", alpha=a)
        kf = D.prog(t, tl["event"], 0.3)
        if kf > 0:
            D.paste(frame, self.fold, cx, y0 + 408, anchor="mm", scale=1.6 - 0.6 * D.ease_out_cubic(kf), alpha=kf)
        # all-in: chips slide from AI to pot as it speaks
        ka = D.ease_out_cubic(D.prog(t, tl["said"], 0.6))
        if ka > 0:
            px, py = cx + 205, y0 + 128
            sx_, sy_ = cx + 110, y0 + 60
            x = sx_ + (px - sx_) * ka
            y = sy_ + (py - sy_) * ka
            for j in range(6):
                D.paste(frame, self.chip_edge, x, y - j * 8 + 3, anchor="mm")
                D.paste(frame, self.chip, x, y - j * 8, anchor="mm")
            D.paste(frame, self.allin, cx + 345, y0 + 70, anchor="mm", alpha=ka,
                    scale=1 + 0.06 * math.sin(t * 9))
            D.paste(frame, self.allin_txt, cx + 345, y0 + 70, anchor="mm", alpha=ka)
            D.paste(frame, self.pot, cx + 345, y0 + 104, anchor="mt", alpha=ka)


# ======================================================================
# Last Vote (mafia)
# ======================================================================
class VoteScene(Scene):
    D_SEAT = 104

    def __init__(self, spec, tl, accent):
        super().__init__(spec, tl, accent)
        seats = self.p["seats"]
        others = seats[1:]
        n = len(others)
        self.pos = {"YOU": (540, 392)}
        for k, name in enumerate(others):
            ang = math.radians(180 - (k + 0.5) * 180 / n)
            self.pos[name] = (540 + 390 * math.cos(ang), 240 - 175 * math.sin(ang))
        self.seat = {}
        for name in seats:
            col = color_of(name)
            s = D.circle(self.D_SEAT, col + (255,), (12, 12, 16, 255), 5).copy()
            ini = "YOU" if name == "YOU" else name[:2].upper()
            ts = D.text_sprite(ini, "black", 34 if name == "YOU" else 40, D.on_color(col))
            D.paste(s, ts, self.D_SEAT / 2, self.D_SEAT / 2 + 1, anchor="mm")
            self.seat[name] = s
        self.labels = {nm: D.text_sprite(nm, "semibold", 26, D.INK) for nm in seats if nm != "YOU"}
        self.grey = D.circle(self.D_SEAT, (40, 40, 46, 215))
        self.token = {nm: D.circle(30, color_of(nm) + (255,), (255, 255, 255, 255), 3) for nm in seats}
        self.badge = D.circle(54, (255, 70, 70, 255), (255, 255, 255, 255), 3)
        self.nums = [D.text_sprite(str(i), "black", 30, (255, 255, 255)) for i in range(len(seats) + 1)]
        self.sus = D.rrect(78, 38, 19, (255, 200, 60, 255))
        self.sus_t = D.text_sprite("SUS?", "black", 20, (20, 20, 20))
        self.elim = _stroked("ELIMINATED", "black", 50, (255, 70, 70), sw=6).rotate(8, expand=True, resample=Image.BICUBIC)
        self.header = D.text_sprite(f"DAY {self.p.get('day', 1)}  ·  TOWN VOTE", "xbold", 26, D.MUTED)
        role = self.p.get("role_reveal", "MAFIA")
        self.role = D.rrect(170, 60, 12, (190, 20, 35, 255), (255, 255, 255, 255), 3).copy()
        D.paste(self.role, D.text_sprite(role, "black", 32, (255, 255, 255)), 85, 31, anchor="mm")
        self.ring = D.circle(self.D_SEAT + 26, None, accent + (255,), 6)

    def render(self, frame, t):
        a = self.intro(t)
        tl = self.tl
        y0 = SCENE_Y
        voters = [n for n in self.p["seats"] if n != "YOU"]
        D.paste(frame, self.header, 72, y0 + SCENE_H - 20, anchor="lb", alpha=a)
        for i, (name, (x, y)) in enumerate(self.pos.items()):
            k = D.ease_out_back(D.prog(t, tl["scene_in"] + 0.06 * i, 0.4))
            if k <= 0:
                continue
            D.paste(frame, self.seat[name], x, y0 + y, anchor="mm", scale=max(0.01, k))
            if name != "YOU":
                D.paste(frame, self.labels[name], x, y0 + y + 58, anchor="mt", alpha=min(1, k))
        # your accusation
        acc = self.p["accused"]
        ks = D.prog(t, tl["you"], 0.3)
        if ks > 0:
            x, y = self.pos[acc]
            D.paste(frame, self.sus, x, y0 + y - 62, anchor="mm", alpha=ks * (1 - D.prog(t, tl["event"], 0.3)))
            D.paste(frame, self.sus_t, x, y0 + y - 62, anchor="mm", alpha=ks * (1 - D.prog(t, tl["event"], 0.3)))
        # vote tokens fly to YOU
        yx, yy = self.pos["YOU"]
        arrived = 0
        for i, v in enumerate(voters):
            st = tl["event"] + i * 0.2
            k = D.prog(t, st, 0.45)
            if k <= 0:
                continue
            vx, vy = self.pos[v]
            if k >= 1:
                arrived += 1
                continue
            e = D.ease_in_out(k)
            mx, my = (vx + yx) / 2, min(vy, yy) - 120
            bx = (1 - e) ** 2 * vx + 2 * (1 - e) * e * mx + e * e * yx
            by = (1 - e) ** 2 * vy + 2 * (1 - e) * e * my + e * e * yy
            D.paste(frame, self.token[v], bx, y0 + by, anchor="mm")
        if arrived:
            D.paste(frame, self.badge, yx + 46, y0 + yy - 44, anchor="mm")
            D.paste(frame, self.nums[arrived], yx + 46, y0 + yy - 44, anchor="mm")
        done = tl["event"] + len(voters) * 0.2 + 0.35
        ke = D.prog(t, done, 0.3)
        if ke > 0:
            D.paste(frame, self.grey, yx, y0 + yy, anchor="mm", alpha=ke)
            D.paste(frame, self.elim, yx, y0 + yy + 30, anchor="mm", alpha=ke,
                    scale=1.5 - 0.5 * D.ease_out_cubic(ke))
        # role reveal on the opponent
        kr = D.prog(t, tl["twist"] - 0.2, 0.4)
        if kr > 0:
            ox, oy = self.pos[self.spec["opponent"]]
            D.paste(frame, self.ring, ox, y0 + oy, anchor="mm", alpha=0.6 + 0.4 * math.sin(t * 8))
            sxk = abs(math.cos(math.pi * (1 - kr) / 2)) if kr < 1 else 1
            D.paste(frame, self.role, ox, y0 + oy - 92, anchor="mm", sx=max(0.02, sxk))


SCENES = {"settler_rush": SettlerScene, "poker": PokerScene, "last_vote": VoteScene}
