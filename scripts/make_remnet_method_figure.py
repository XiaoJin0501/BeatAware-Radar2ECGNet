"""Create the REM-Net method overview figure for the RQ1 manuscript.

The script intentionally uses only Pillow and ReportLab, which are already
available in the project environment.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "paper" / "figures"
W, H = 3000, 1200


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT = font(34)
FONT_B = font(36, bold=True)
FONT_S = font(25)
FONT_T = font(42, bold=True)


def multiline_center(draw: ImageDraw.ImageDraw, text: str, box: tuple[int, int, int, int], fill: str, fnt) -> None:
    x0, y0, x1, y1 = box
    lines = text.split("\n")
    line_heights = [draw.textbbox((0, 0), line, font=fnt)[3] for line in lines]
    total_h = sum(line_heights) + (len(lines) - 1) * 8
    y = y0 + (y1 - y0 - total_h) / 2
    for line, lh in zip(lines, line_heights):
        bb = draw.textbbox((0, 0), line, font=fnt)
        tw = bb[2] - bb[0]
        draw.text((x0 + (x1 - x0 - tw) / 2, y), line, fill=fill, font=fnt)
        y += lh + 8


def rounded_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    text: str,
    fill: str,
    outline: str = "#2f3b52",
    width: int = 4,
    fnt=FONT,
) -> None:
    draw.rounded_rectangle(xy, radius=32, fill=fill, outline=outline, width=width)
    multiline_center(draw, text, xy, "#172033", fnt)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], fill: str = "#34495e", width: int = 7) -> None:
    draw.line([start, end], fill=fill, width=width)
    x0, y0 = start
    x1, y1 = end
    if abs(x1 - x0) >= abs(y1 - y0):
        sign = 1 if x1 > x0 else -1
        pts = [(x1, y1), (x1 - sign * 28, y1 - 18), (x1 - sign * 28, y1 + 18)]
    else:
        sign = 1 if y1 > y0 else -1
        pts = [(x1, y1), (x1 - 18, y1 - sign * 28), (x1 + 18, y1 - sign * 28)]
    draw.polygon(pts, fill=fill)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    draw.text((80, 42), "REM-Net: Representation-Event-Morphology Network", fill="#172033", font=FONT_T)
    draw.text(
        (80, 96),
        "Final morphology-first model: alignment branch was ablated and removed.",
        fill="#5a6473",
        font=FONT_S,
    )

    y = 260
    h = 220
    boxes = [
        ((80, y, 430, y + h), "Radar input\nMMECG: FMCW bins\nCR_RVS: CW I/Q", "#eef5ff"),
        ((520, y, 920, y + h), "Cardiac radar\nrepresentation\nWSST / STFT", "#eaf7f0"),
        ((1010, y, 1410, y + h), "Multi-scale\n2-D encoder\n3x3 + 5x5 conv", "#fff7e6"),
        ((1500, y, 1880, y + h), "Temporal latent\n128 x 400", "#fff7e6"),
        ((2020, y, 2390, y + h), "Dilated TCN\n1, 2, 4, 8, 16", "#fff7e6"),
        ((2510, y, 2920, y + h), "Full-window\nECG head\n1 x 1600", "#f3eefc"),
    ]
    for xy, text, fill in boxes:
        rounded_box(draw, xy, text, fill)

    for a, b in [((430, 370), (520, 370)), ((920, 370), (1010, 370)), ((1410, 370), (1500, 370)), ((1880, 370), (2020, 370)), ((2390, 370), (2510, 370))]:
        arrow(draw, a, b)

    rounded_box(draw, (1440, 675, 1940, 855), "Cardiac event head\nR-peak + QRS masks", "#e8f0fe")
    arrow(draw, (1690, 480), (1690, 675), "#4c6f99")
    arrow(draw, (1940, 765), (2070, 480), "#4c6f99")
    draw.text((1990, 620), "event projection\nresidual fusion", fill="#4c6f99", font=FONT_S)

    rounded_box(draw, (2030, 890, 2600, 1080), "Primary morphology head\nShape-200, 1 x 200", "#fceeee", "#8f3d38")
    arrow(draw, (2205, 480), (2290, 890), "#8f3d38")
    draw.text((2330, 770), "primary reported output", fill="#8f3d38", font=FONT_S)

    rounded_box(draw, (120, 720, 660, 940), "Supervised targets\nECG waveform\nR/QRS events\nShape-200 cycle", "#f7f7f7", "#b8c0cc", 3, FONT_S)
    rounded_box(draw, (800, 720, 1380, 940), "Training objective\nL_full + beta L_event\n+ lambda L_shape", "#f7f7f7", "#b8c0cc", 3, FONT_S)
    arrow(draw, (660, 830), (800, 830), "#777777", 5)

    img.save(OUT / "F0_remnet_overview.png")
    pdf_path = OUT / "F0_remnet_overview.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=(W, H))
    c.drawImage(ImageReader(img), 0, 0, width=W, height=H)
    c.showPage()
    c.save()


if __name__ == "__main__":
    main()
