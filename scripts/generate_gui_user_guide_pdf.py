from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "AutoAlias_Desktop_Editor_User_Guide.md"
OUTPUT = ROOT / "docs" / "AutoAlias_Desktop_Editor_User_Guide.pdf"
SIMSUN = Path(r"C:\Windows\Fonts\simsun.ttc")
SIMSUN_BOLD = Path(r"C:\Windows\Fonts\simsunb.ttf")


PAGE_W, PAGE_H = 1240, 1754
MARGIN_X = 95
MARGIN_TOP = 90
MARGIN_BOTTOM = 90
LINE_GAP = 12


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    font_path = SIMSUN_BOLD if bold and SIMSUN_BOLD.exists() else SIMSUN
    return ImageFont.truetype(str(font_path), size=size)


TITLE_FONT = _font(38, bold=True)
H1_FONT = _font(30, bold=True)
H2_FONT = _font(24, bold=True)
BODY_FONT = _font(21)
SMALL_FONT = _font(17)


def _measure(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> int:
    if not text:
        return 0
    bbox = draw.textbbox((0, 0), text, font=font)
    return int(bbox[2] - bbox[0])


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    text = text.strip()
    if not text:
        return [""]
    lines: list[str] = []
    current = ""
    tokens = re.findall(r"[A-Za-z0-9_./:\\-]+|\s+|.", text)
    for token in tokens:
        if token.isspace():
            candidate = current + " "
        else:
            candidate = current + token
        if current and _measure(draw, candidate, font) > max_width:
            lines.append(current.rstrip())
            current = token.lstrip()
        else:
            current = candidate
    if current.strip():
        lines.append(current.rstrip())
    return lines


class PdfWriter:
    def __init__(self) -> None:
        self.pages: list[Image.Image] = []
        self.page = self._new_page()
        self.draw = ImageDraw.Draw(self.page)
        self.y = MARGIN_TOP
        self.page_no = 1

    def _new_page(self) -> Image.Image:
        return Image.new("RGB", (PAGE_W, PAGE_H), "#ffffff")

    def _footer(self) -> None:
        self.draw.line((MARGIN_X, PAGE_H - 62, PAGE_W - MARGIN_X, PAGE_H - 62), fill="#d9e1e7", width=2)
        footer = f"AutoAlias Desktop Editor 使用教程  |  第 {self.page_no} 页"
        self.draw.text((MARGIN_X, PAGE_H - 48), footer, font=SMALL_FONT, fill="#53606b")

    def new_page(self) -> None:
        self._footer()
        self.pages.append(self.page)
        self.page_no += 1
        self.page = self._new_page()
        self.draw = ImageDraw.Draw(self.page)
        self.y = MARGIN_TOP

    def ensure(self, height: int) -> None:
        if self.y + height > PAGE_H - MARGIN_BOTTOM:
            self.new_page()

    def text(self, text: str, font: ImageFont.FreeTypeFont, fill: str = "#17202a", indent: int = 0, gap: int = LINE_GAP) -> None:
        max_width = PAGE_W - MARGIN_X * 2 - indent
        lines = _wrap(self.draw, text, font, max_width)
        line_height = int(font.size * 1.45)
        self.ensure(max(1, len(lines)) * line_height + gap)
        for line in lines:
            self.draw.text((MARGIN_X + indent, self.y), line, font=font, fill=fill)
            self.y += line_height
        self.y += gap

    def heading(self, text: str, level: int) -> None:
        if level == 1:
            self.ensure(72)
            self.draw.rounded_rectangle(
                (MARGIN_X - 22, self.y - 10, PAGE_W - MARGIN_X + 22, self.y + 52),
                radius=14,
                fill="#eef4fb",
                outline="#cfdbe8",
                width=2,
            )
            self.draw.text((MARGIN_X, self.y), text, font=H1_FONT, fill="#143d6b")
            self.y += 76
        else:
            self.text(text, H2_FONT, fill="#17202a", gap=10)

    def finish(self) -> None:
        self._footer()
        self.pages.append(self.page)


def render_pdf() -> None:
    raw = SOURCE.read_text(encoding="utf-8").splitlines()
    writer = PdfWriter()
    writer.draw.text((MARGIN_X, writer.y), "AutoAlias Desktop Editor", font=TITLE_FONT, fill="#0f4c81")
    writer.y += 58
    writer.draw.text((MARGIN_X, writer.y), "使用教程 / 功能说明", font=H1_FONT, fill="#17202a")
    writer.y += 76

    for line in raw[1:]:
        stripped = line.strip()
        if not stripped:
            writer.y += 8
            continue
        if stripped.startswith("# "):
            writer.heading(stripped[2:].strip(), 1)
        elif stripped.startswith("## "):
            writer.heading(stripped[3:].strip(), 2)
        elif stripped.startswith("- "):
            writer.text("• " + stripped[2:].strip(), BODY_FONT, indent=24)
        elif re.match(r"^\d+\. ", stripped):
            writer.text(stripped, BODY_FONT, indent=18)
        else:
            writer.text(stripped, BODY_FONT)

    writer.finish()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    writer.pages[0].save(OUTPUT, "PDF", resolution=150, save_all=True, append_images=writer.pages[1:])
    print(OUTPUT)


if __name__ == "__main__":
    render_pdf()
