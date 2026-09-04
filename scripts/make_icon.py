"""
Draw the desktop icon: a page with a magnifier over it.

Generated rather than committed as an opaque binary, so the shape and colors
can be changed by editing this file. Drawn at 4x and downscaled, because
ImageDraw has no anti-aliasing of its own and circles look ragged otherwise.

    python scripts/make_icon.py     ->  docs/icon.ico
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "icon.ico"

S = 1024          # working size; the .ico is downscaled from this
SCALE = S / 256   # so measurements below can be read as 256px units

INDIGO = (79, 70, 229, 255)        # matches the UI's accent
INDIGO_DEEP = (55, 48, 163, 255)
PAPER = (255, 255, 255, 255)
RULE = (165, 180, 252, 255)
LENS = (224, 231, 255, 255)


def px(v: float) -> float:
    return v * SCALE


def draw() -> Image.Image:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Flat ground. An earlier version shaded the lower half for depth; at
    # 32px that band just read as a stray shape, so it went.
    d.rounded_rectangle([px(8), px(8), px(248), px(248)],
                        radius=px(56), fill=INDIGO)

    # -- the page, sitting up and to the left to leave room for the lens --
    d.rounded_rectangle([px(52), px(40), px(172), px(196)],
                        radius=px(12), fill=PAPER)

    # ruled lines, the first short like a heading
    y = px(70)
    for width in (0.45, 0.8, 0.8, 0.64, 0.8, 0.55):
        x0 = px(70)
        x1 = x0 + (px(172) - px(70) - px(18)) * width
        d.rounded_rectangle([x0, y, x1, y + px(9)], radius=px(4), fill=RULE)
        y += px(22)

    # -- the magnifier ----------------------------------------------------
    # A white ring on white paper would disappear, so an indigo halo is laid
    # down first: the gap is what makes the lens read as sitting on top.
    cx, cy, r = px(166), px(164), px(50)
    ring = px(16)
    halo = px(11)

    d.line([cx + px(30), cy + px(30), cx + px(74), cy + px(74)],
           fill=INDIGO, width=int(px(44)))
    d.ellipse([cx - r - ring - halo, cy - r - ring - halo,
               cx + r + ring + halo, cy + r + ring + halo], fill=INDIGO)

    d.line([cx + px(32), cy + px(32), cx + px(68), cy + px(68)],
           fill=PAPER, width=int(px(24)))
    d.ellipse([cx + px(56), cy + px(56), cx + px(80), cy + px(80)], fill=PAPER)

    d.ellipse([cx - r - ring, cy - r - ring, cx + r + ring, cy + r + ring],
              fill=PAPER)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=LENS)

    # The handle and its halo are drawn past the bottom-right corner, so clip
    # the whole thing back to the rounded square instead of hand-fitting them.
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [px(8), px(8), px(248), px(248)], radius=px(56), fill=255)
    out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    out.paste(img, (0, 0), mask)
    return out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img = draw()
    # Windows picks the size it needs from the file; without the small ones it
    # downscales 256px itself and the result is mushy in the taskbar.
    sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
    img.resize((256, 256), Image.LANCZOS).save(OUT, format="ICO", sizes=sizes)
    # Only the .ico is used, by the shortcut and the favicon route. A PNG
    # alongside it was written for a while and referenced by nothing.
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
