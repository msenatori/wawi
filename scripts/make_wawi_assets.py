#!/usr/bin/env python3
"""Regenerate Wawi's derived images from the original drawing.

The drawing (``wawi/assets/wawi_dibujo.jpg``) was made by the author's niece (art license: wawi/assets/LICENSE-art.md).
It is used AS IS: this script only cuts the dog out of its background and
scales/crops that cutout. Nothing is redrawn, recoloured or retouched, and the
drawing itself is only ever read, never written.

Outputs, all under ``wawi/assets/``:

* ``wawi.png``           RGBA cutout at full resolution, tightly cropped.
* ``wawi_menubar.png``   head and face, 36 px tall (18 pt @2x) for the status item.
* ``wawi_face.png``      round face for the panel header (64 px, shown at 16 pt).
* ``Wawi.icns``          app icon, every size, via ``iconutil``.

How the cutout works: the background is sky blue, cloud cream, sun yellow and
grass/shadow greens, while the dog is white, grey and navy. Pixels are
classified as "background" when they are green (G clearly above B) or light
sky (G above R and bright). The dog is then the connected component that
contains its body; the clouds are cream like her fur but never touch her, so
connectivity drops them. Enclosed holes (freckles, eyes) are filled, and the
edge is softened inward by one pixel so it does not look jagged, without
pulling in any background colour.

Needs numpy, Pillow and scipy (scipy only for this script, not at runtime)::

    python scripts/make_wawi_assets.py

Animation frames
----------------

``--frames SRC`` instead imports the animation frames (``idle_a.png``,
``blink.png``… see ``wawi.mascot.FRAME_NAMES``) from *SRC* into
``wawi/assets/wawi/``. The sources are big square canvases (1024 x 1024)
drawn so that every pose shares the same scale and baseline; this keeps that
alignment and only:

* drops the faint alpha haze image generators leave around the drawing,
* crops EVERY frame to the same box — the union of all their silhouettes —
  so a frame swap never shifts her by a pixel,
* scales them all by the same factor to :data:`FRAME_WIDTH` px wide (about
  2.5x the on-screen width, crisp on Retina and small in the repo),
* stores them as 256-colour palette PNGs (``--no-quantize`` keeps full
  RGBA). The drawing is a handful of flat colours, so this is visually
  lossless and makes the set ~7x smaller — sixty-odd frames stay under 1 MB.

The sources share one baseline and centre, so the union box is the right
crop: it keeps every pose where it was drawn. A frame drawn far bigger than the
rest would still blow that box up and shrink her on screen, so an outlier
(one that alone grows the box by more than :data:`_OUTLIER` of its size) is
reported and clipped to the box of the others instead of moving everyone.

The output folder is emptied of old frames first: frames from an earlier run
were cropped to a different box and would no longer line up. Only numpy and
Pillow are needed for this mode::

    python scripts/make_wawi_assets.py --frames path/to/sprites
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "wawi" / "assets"
DRAWING = ASSETS / "wawi_dibujo.jpg"
FRAMES_OUT = ASSETS / "wawi"
#: Width of the shipped frames, in pixels (the sprite is ~124 pt on screen).
FRAME_WIDTH = 320
#: Alpha below this is generator haze, not drawing; it is cleared, and it
#: does not count towards the bounding box.
_HAZE = 24
#: Transparent margin kept around the shared box, in source pixels.
_MARGIN = 6
#: A frame that alone grows the shared box by more than this fraction (of
#: the box's width or height) is an outlier: clipped, not followed.
_OUTLIER = 0.15

#: A point that is certainly the dog's white body, in drawing coordinates.
_BODY_SEED = (700, 700)  # (row, col)
#: The drawing's own sky colour, used behind the icon and the round face.
SKY = (191, 213, 200)


def cutout(src: Image.Image) -> Image.Image:
    """Return the dog as RGBA, background transparent, tightly cropped."""
    from scipy import ndimage as ndi

    rgb = np.asarray(src.convert("RGB"))
    a = rgb.astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    lum = (r + g + b) / 3.0

    green = (g - b) > 30                   # grass, ground shadow, sun
    sky = ((g - r) > 11) & (lum > 150)     # sky and its lighter rays
    fg = ~green & ~sky

    labels, _ = ndi.label(fg)
    seed = labels[_BODY_SEED]
    if seed == 0:
        raise RuntimeError("the body seed is not on the dog; check the drawing")
    mask = ndi.binary_fill_holes(labels == seed)

    # Soften the edge INWARD only: outside the mask alpha stays 0, so no sky or
    # grass colour can leak in; the outermost ring of the dog goes partly
    # transparent instead of stair-stepping.
    soft = ndi.gaussian_filter(mask.astype(float), 0.7)
    alpha = np.where(mask, np.clip(soft * 1.6 - 0.3, 0.0, 1.0), 0.0)
    alpha = (alpha * 255).round().astype(np.uint8)

    rgba = np.dstack([rgb, alpha])
    ys, xs = np.where(mask)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return Image.fromarray(rgba, "RGBA").crop(box)


def head(cut: Image.Image) -> Image.Image:
    """The head with both ears and the smile: what reads at menu-bar size.

    The neck strokes start right beside the right ear, so a plain rectangle
    drags them in. Below the ears the crop follows an ellipse around the face
    instead (alpha only; no pixel of the drawing is changed).
    """
    from scipy import ndimage as ndi

    # Fractions of the cutout, measured on the drawing: the ears span the top
    # 55% of the width, the smile ends at ~56% of the height.
    w, h = cut.size
    box_w, box_h = int(w * 0.555), int(h * 0.565)
    face = cut.crop((0, 0, box_w, box_h))
    ears_bottom = h * 0.29
    cx, cy, rx, ry = w * 0.28, ears_bottom, w * 0.255, h * 0.275
    yy, xx = np.mgrid[0:box_h, 0:box_w]
    inside = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    keep = (yy < ears_bottom) | inside
    keep = ndi.gaussian_filter(keep.astype(float), 1.5)
    alpha = np.asarray(face.getchannel("A")).astype(float) * keep
    face.putalpha(Image.fromarray(alpha.round().astype(np.uint8), "L"))
    return face


def menubar_icon(cut: Image.Image, height_px: int = 36) -> Image.Image:
    face = head(cut)
    bbox = face.getbbox()
    face = face.crop(bbox)
    width = round(face.width * height_px / face.height)
    return face.resize((width, height_px), Image.LANCZOS)


def round_face(cut: Image.Image, size: int = 64) -> Image.Image:
    """Circle centred on the face, over the drawing's sky colour."""
    w, h = cut.size
    # Face centre and radius, as fractions of the cutout (eyes/nose area).
    cx, cy, rad = 0.27 * w, 0.33 * h, 0.21 * w
    box = (int(cx - rad), int(cy - rad), int(cx + rad), int(cy + rad))
    face = cut.crop(box)
    disc = Image.new("RGBA", face.size, SKY + (255,))
    disc.alpha_composite(face)
    big = size * 4
    disc = disc.resize((big, big), Image.LANCZOS)
    m = Image.new("L", (big, big), 0)
    ImageDraw.Draw(m).ellipse((0, 0, big - 1, big - 1), fill=255)
    disc.putalpha(m)
    return disc.resize((size, size), Image.LANCZOS)


def app_icon(cut: Image.Image, size: int = 1024) -> Image.Image:
    """Wawi on a soft rounded square in the drawing's sky colour."""
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    # macOS icon grid: the body of the icon is ~80% of the canvas.
    inset = round(size * 0.098)
    side = size - 2 * inset
    radius = round(side * 0.225)

    top = tuple(min(255, c + 22) for c in SKY)
    grad = Image.new("RGBA", (side, side))
    px = grad.load()
    for y in range(side):
        t = y / max(1, side - 1)
        col = tuple(round(top[i] * (1 - t) + SKY[i] * t) for i in range(3))
        for x in range(side):
            px[x, y] = col + (255,)
    m = Image.new("L", (side, side), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, side - 1, side - 1), radius, fill=255)
    grad.putalpha(m)

    # Faint shadow under the plate, so it sits on light and dark docks alike.
    shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    sh = Image.new("RGBA", (side, side), (0, 0, 0, 70))
    sh.putalpha(m.point(lambda v: v * 70 // 255))
    shadow.paste(sh, (inset, inset + round(size * 0.012)), sh)
    shadow = shadow.filter(ImageFilter.GaussianBlur(size * 0.012))
    canvas.alpha_composite(shadow)
    canvas.alpha_composite(grad, (inset, inset))

    dog_w = round(side * 0.88)
    dog = cut.resize((dog_w, round(cut.height * dog_w / cut.width)), Image.LANCZOS)
    x = inset + (side - dog.width) // 2
    y = inset + side - dog.height - round(side * 0.13)
    canvas.alpha_composite(dog, (x, y))
    return canvas


def write_icns(icon: Image.Image, dest: Path) -> None:
    if shutil.which("iconutil") is None:
        print("iconutil not found (macOS only); skipping", dest.name, file=sys.stderr)
        return
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "Wawi.iconset"
        iconset.mkdir()
        for pt in (16, 32, 128, 256, 512):
            for scale in (1, 2):
                px = pt * scale
                name = f"icon_{pt}x{pt}{'@2x' if scale == 2 else ''}.png"
                icon.resize((px, px), Image.LANCZOS).save(iconset / name)
        subprocess.run(
            ["iconutil", "-c", "icns", str(iconset), "-o", str(dest)], check=True
        )


def _frame_names() -> tuple[str, ...]:
    sys.path.insert(0, str(ROOT))
    from wawi.mascot import FRAME_NAMES

    return FRAME_NAMES


def shared_box(
    boxes: dict[str, tuple[int, int, int, int]], size: tuple[int, int]
) -> tuple[int, int, int, int]:
    """The crop shared by every frame: the union of their silhouettes, plus a
    margin, ignoring outliers (see the module docstring). Pure, for tests."""
    w, h = size

    def union(items):
        xs0, ys0, xs1, ys1 = zip(*items) if items else ((0,), (0,), (w,), (h,))
        return min(xs0), min(ys0), max(xs1), max(ys1)

    full = union(list(boxes.values()))
    keep = dict(boxes)
    for n, b in boxes.items():
        rest = union([v for k, v in keep.items() if k != n])
        rw, rh = max(1, rest[2] - rest[0]), max(1, rest[3] - rest[1])
        grow_x = (max(0, rest[0] - b[0]) + max(0, b[2] - rest[2])) / rw
        grow_y = (max(0, rest[1] - b[1]) + max(0, b[3] - rest[3])) / rh
        if len(boxes) > 2 and max(grow_x, grow_y) > _OUTLIER:
            print(f"outlier frame {n} {b}: clipped to the others' box {rest}",
                  file=sys.stderr)
            keep.pop(n)
    if keep != boxes:
        full = union(list(keep.values()))
    x0, y0, x1, y1 = full
    return (
        max(0, x0 - _MARGIN), max(0, y0 - _MARGIN),
        min(w, x1 + _MARGIN), min(h, y1 + _MARGIN),
    )


def import_frames(
    src: Path, dest: Path = FRAMES_OUT, width: int = FRAME_WIDTH, quantize: bool = True
) -> list[str]:
    """Crop every frame in *src* to one shared box and write them to *dest*.

    Returns the names written. Raises if *src* has none of the known frames,
    or if they are not all the same size (then they cannot be aligned).
    """
    names = [n for n in _frame_names() if (src / f"{n}.png").is_file()]
    if not names:
        raise SystemExit(f"no known frames in {src}")
    images = {}
    for n in names:
        img = Image.open(src / f"{n}.png").convert("RGBA")
        a = np.asarray(img).copy()
        a[..., 3][a[..., 3] < _HAZE] = 0
        images[n] = a
    sizes = {a.shape[:2] for a in images.values()}
    if len(sizes) != 1:
        raise SystemExit(f"frames differ in size, cannot align them: {sizes}")
    h, w = sizes.pop()

    boxes = {}
    for n, a in images.items():
        ys, xs = np.nonzero(a[..., 3])
        if len(xs):
            boxes[n] = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    box = shared_box(boxes, (w, h))
    bw, bh = box[2] - box[0], box[3] - box[1]
    out_size = (width, max(1, round(bh * width / bw)))

    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("*.png"):
        old.unlink()
    total = 0
    for n, a in images.items():
        frame = Image.fromarray(a, "RGBA").crop(box)
        # Resize in premultiplied alpha, or the cleared haze (black, alpha 0)
        # bleeds a dark fringe into her outline.
        frame = frame.convert("RGBa").resize(out_size, Image.LANCZOS).convert("RGBA")
        if quantize:
            # No dithering: flat crayon colours band less than they speckle.
            frame = frame.quantize(256, method=Image.FASTOCTREE, dither=Image.Dither.NONE)
        frame.save(dest / f"{n}.png", optimize=True)
        total += (dest / f"{n}.png").stat().st_size
    print(f"box {box} -> {out_size[0]}x{out_size[1]}; wrote {len(names)} frames "
          f"({total / 1024:.0f} KB) to {dest}")
    missing = [n for n in _frame_names() if n not in images]
    if missing:
        print("not in the source (fallbacks will be used):", ", ".join(missing))
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--frames", type=Path, metavar="SRC",
        help="import animation frames from SRC into wawi/assets/wawi/",
    )
    parser.add_argument("--width", type=int, default=FRAME_WIDTH,
                        help="width of the imported frames in px")
    parser.add_argument("--no-quantize", action="store_true",
                        help="keep full RGBA frames instead of 256-colour PNGs")
    args = parser.parse_args()
    if args.frames is not None:
        import_frames(args.frames, width=args.width, quantize=not args.no_quantize)
        return
    cut = cutout(Image.open(DRAWING))
    cut.save(ASSETS / "wawi.png", optimize=True)
    menubar_icon(cut).save(ASSETS / "wawi_menubar.png", optimize=True)
    round_face(cut).save(ASSETS / "wawi_face.png", optimize=True)
    write_icns(app_icon(cut), ASSETS / "Wawi.icns")
    print("wrote", ", ".join(p.name for p in sorted(ASSETS.iterdir())))


if __name__ == "__main__":
    main()
