from __future__ import annotations

import math
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "navimath-research-poster.png"

W, H = 2160, 2880
BG = "#F3F4F2"
PAPER = "#FAFAF8"
INK = "#20292F"
MUTED = "#68747A"
FAINT = "#D9DEDC"
GRID = "#E7EAE7"
TEAL = "#4E7D78"
TEAL_LIGHT = "#DCE9E5"
BLUE = "#5C718B"
BLUE_LIGHT = "#E0E6ED"
SAGE = "#7C8973"
SAGE_LIGHT = "#E5E9E1"
CLAY = "#9A7168"
CLAY_LIGHT = "#EEE3E0"
GOLD = "#9A8758"
GOLD_LIGHT = "#ECE8DA"

SYSTEM_FONT_DIR = Path(os.environ.get("WINDIR", "")) / "Fonts"
FONT_REGULAR = SYSTEM_FONT_DIR / "Noto Sans SC (TrueType).otf"
FONT_MEDIUM = SYSTEM_FONT_DIR / "Noto Sans SC Medium (TrueType).otf"
FONT_BOLD = SYSTEM_FONT_DIR / "Noto Sans SC Bold (TrueType).otf"
FONT_LATIN = SYSTEM_FONT_DIR / "arial.ttf"
FONT_LATIN_BOLD = SYSTEM_FONT_DIR / "arialbd.ttf"


def font(size: int, *, bold: bool = False, latin: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_LATIN_BOLD if latin and bold else FONT_LATIN if latin else FONT_BOLD if bold else FONT_REGULAR
    return ImageFont.truetype(str(path), size)


def text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], value: str, size: int, fill: str = INK,
         *, bold: bool = False, anchor: str = "la", latin: bool = False) -> None:
    draw.text(xy, value, font=font(size, bold=bold, latin=latin), fill=fill, anchor=anchor)


def line(draw: ImageDraw.ImageDraw, points, fill=FAINT, width=2) -> None:
    draw.line(points, fill=fill, width=width, joint="curve")


def arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float],
          fill: str = INK, width: int = 4, head: int = 16) -> None:
    x1, y1 = start
    x2, y2 = end
    line(draw, (start, end), fill=fill, width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    wing = math.pi * 0.82
    p1 = (x2 + head * math.cos(angle + wing), y2 + head * math.sin(angle + wing))
    p2 = (x2 + head * math.cos(angle - wing), y2 + head * math.sin(angle - wing))
    draw.polygon(((x2, y2), p1, p2), fill=fill)


def curved_arrow(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], start: int, end: int,
                 fill: str, width: int = 5) -> None:
    draw.arc(box, start=start, end=end, fill=fill, width=width)
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    rx = (box[2] - box[0]) / 2
    ry = (box[3] - box[1]) / 2
    a = math.radians(end)
    tip = (cx + rx * math.cos(a), cy + ry * math.sin(a))
    tangent = a + math.pi / 2
    head = 18
    wing = 0.72
    p1 = (tip[0] - head * math.cos(tangent - wing), tip[1] - head * math.sin(tangent - wing))
    p2 = (tip[0] - head * math.cos(tangent + wing), tip[1] - head * math.sin(tangent + wing))
    draw.polygon((tip, p1, p2), fill=fill)


def pill(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], label: str,
         fill: str, ink: str, *, size: int = 27) -> None:
    draw.rounded_rectangle(box, radius=8, fill=fill)
    text(draw, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2 - 1), label, size, ink,
         bold=True, anchor="mm", latin=True)


def module(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], number: str, title_cn: str,
           title_en: str, accent: str, soft: str, rows: list[tuple[str, str]]) -> None:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=8, fill=PAPER, outline=FAINT, width=3)
    draw.rectangle((x1, y1, x1 + 12, y2), fill=accent)
    text(draw, (x1 + 44, y1 + 45), number, 24, accent, bold=True, latin=True)
    text(draw, (x1 + 44, y1 + 92), title_cn, 40, INK, bold=True)
    text(draw, (x1 + 44, y1 + 137), title_en.upper(), 19, MUTED, bold=True, latin=True)
    yy = y1 + 192
    for label, detail in rows:
        draw.ellipse((x1 + 46, yy + 5, x1 + 60, yy + 19), fill=accent)
        text(draw, (x1 + 78, yy), label, 27, INK, bold=True)
        text(draw, (x1 + 78, yy + 38), detail, 21, MUTED)
        yy += 93
    draw.rectangle((x2 - 64, y1 + 24, x2 - 24, y1 + 30), fill=soft)


def metric(draw: ImageDraw.ImageDraw, x: int, y: int, value: str, label: str, note: str, color: str) -> None:
    text(draw, (x, y), value, 50, color, bold=True, latin=True)
    text(draw, (x, y + 69), label, 25, INK, bold=True)
    text(draw, (x, y + 111), note, 19, MUTED)


def draw_compass(draw: ImageDraw.ImageDraw, cx: int, cy: int) -> None:
    draw.ellipse((cx - 338, cy - 338, cx + 338, cy + 338), fill=PAPER, outline=FAINT, width=3)
    draw.ellipse((cx - 256, cy - 256, cx + 256, cy + 256), outline="#CFD6D3", width=2)
    draw.ellipse((cx - 175, cy - 175, cx + 175, cy + 175), fill="#EEF2EF", outline=TEAL, width=4)
    curved_arrow(draw, (cx - 282, cy - 282, cx + 282, cy + 282), 208, 502, TEAL, 7)

    for angle in (0, 72, 144, 216, 288):
        a = math.radians(angle - 90)
        r1, r2 = 260, 298
        p1 = (cx + r1 * math.cos(a), cy + r1 * math.sin(a))
        p2 = (cx + r2 * math.cos(a), cy + r2 * math.sin(a))
        line(draw, (p1, p2), fill="#BCC7C3", width=3)

    text(draw, (cx, cy - 48), "ADAPTIVE", 22, TEAL, bold=True, anchor="mm", latin=True)
    text(draw, (cx, cy + 8), "推理导航", 49, INK, bold=True, anchor="mm")
    text(draw, (cx, cy + 68), "RLoT NAVIGATOR", 22, MUTED, bold=True, anchor="mm", latin=True)

    positions = [
        (cx, cy - 300, "REASON", BLUE_LIGHT, BLUE),
        (cx + 292, cy - 95, "DECOMPOSE", SAGE_LIGHT, SAGE),
        (cx + 185, cy + 252, "DEBATE", GOLD_LIGHT, GOLD),
        (cx - 185, cy + 252, "REFINE", CLAY_LIGHT, CLAY),
        (cx - 292, cy - 95, "TERMINATE", TEAL_LIGHT, TEAL),
    ]
    for px, py, label, soft, accent in positions:
        width = 196 if label != "DECOMPOSE" else 232
        pill(draw, (int(px - width / 2), int(py - 27), int(px + width / 2), int(py + 27)), label, soft, accent, size=20)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)

    # Subtle scientific paper grid.
    for x in range(120, W - 120, 80):
        line(draw, ((x, 120), (x, H - 120)), fill=GRID, width=1)
    for y in range(120, H - 120, 80):
        line(draw, ((120, y), (W - 120, y)), fill=GRID, width=1)
    draw.rectangle((92, 92, W - 92, H - 92), outline="#CAD1CE", width=2)

    # Header.
    text(draw, (145, 155), "RESEARCH SYSTEM OVERVIEW  /  2026", 22, TEAL, bold=True, latin=True)
    text(draw, (W - 145, 155), "FIG. 01", 22, MUTED, bold=True, anchor="ra", latin=True)
    text(draw, (145, 242), "NaviMath", 112, INK, bold=True, latin=True)
    text(draw, (151, 375), "可验证 · 多候选 · 自适应的竞赛数学推理智能体", 40, INK, bold=True)
    text(draw, (151, 435), "A verifiable, multi-candidate reasoning agent for competition mathematics", 25, MUTED, latin=True)
    line(draw, ((145, 510), (2015, 510)), fill=INK, width=3)
    text(draw, (145, 548), "SYSTEM ARCHITECTURE", 19, MUTED, bold=True, latin=True)
    text(draw, (2015, 548), "ROUTE  →  REASON  →  VERIFY  →  LEARN", 19, TEAL, bold=True, anchor="ra", latin=True)

    # Main system modules.
    module(
        draw,
        (145, 660, 605, 1180),
        "01",
        "问题输入",
        "Problem intake",
        BLUE,
        BLUE_LIGHT,
        [
            ("题面与元数据", "contest · answer schema"),
            ("难度识别", "easy / medium / hard / proof"),
            ("任务约束", "预算、格式与终止条件"),
        ],
    )
    module(
        draw,
        (1555, 660, 2015, 1180),
        "03",
        "校验与归一",
        "Verify & normalize",
        CLAY,
        CLAY_LIGHT,
        [
            ("对抗校验", "假设 · 边界 · 计算攻击"),
            ("ToolVerify", "SymPy + 数值一致性"),
            ("答案归一化", "集合 · 区间 · AIME 格式"),
        ],
    )

    # Router bridge above the navigator.
    draw.rounded_rectangle((720, 640, 1440, 835), radius=8, fill=PAPER, outline=FAINT, width=3)
    draw.rectangle((720, 640, 732, 835), fill=SAGE)
    text(draw, (762, 681), "02", 22, SAGE, bold=True, latin=True)
    text(draw, (815, 680), "技能路由", 36, INK, bold=True)
    text(draw, (815, 726), "HARP-DISTILLED SKILL ROUTER", 18, MUTED, bold=True, latin=True)
    domain_x = 765
    for label, color in (("ALG", BLUE), ("GEO", TEAL), ("NT", SAGE), ("COMB", GOLD), ("CALC", CLAY)):
        pill(draw, (domain_x, 770, domain_x + 110, 811), label, "#EEF0ED", color, size=17)
        domain_x += 124

    arrow(draw, (605, 920), (700, 920), BLUE, width=6, head=22)
    arrow(draw, (1080, 835), (1080, 890), SAGE, width=6, head=22)
    arrow(draw, (1460, 920), (1555, 920), CLAY, width=6, head=22)

    draw_compass(draw, 1080, 1240)
    text(draw, (1080, 1618), "七维自评状态 + 运行时上下文 + 动态调用预算", 24, MUTED, anchor="mm")
    text(draw, (1080, 1663), "7-aspect state · context-aware action mask · calibrated rule fallback", 19, MUTED, anchor="mm", latin=True)

    # Independent reasoning tracks.
    text(draw, (145, 1305), "MULTI-CANDIDATE", 19, BLUE, bold=True, latin=True)
    text(draw, (145, 1344), "独立候选推理", 32, INK, bold=True)
    for i, (label, sub, color) in enumerate((
        ("A", "Direct solver", BLUE),
        ("B", "Domain solver", TEAL),
        ("C", "Checker", SAGE),
    )):
        yy = 1415 + i * 112
        draw.ellipse((145, yy, 203, yy + 58), fill=color)
        text(draw, (174, yy + 28), label, 20, PAPER, bold=True, anchor="mm", latin=True)
        text(draw, (226, yy + 10), sub, 24, INK, bold=True, latin=True)
        text(draw, (226, yy + 47), "independent trace", 18, MUTED, latin=True)
        line(draw, ((435, yy + 28), (590, yy + 28)), fill="#C5CDCA", width=3)

    # Stable output and learning loop.
    arrow(draw, (1080, 1696), (1080, 1770), TEAL, width=7, head=24)
    draw.rounded_rectangle((635, 1780, 1525, 1988), radius=8, fill=INK)
    text(draw, (1080, 1830), "04  CONSENSUS & OUTPUT", 20, "#AFC8C3", bold=True, anchor="ma", latin=True)
    text(draw, (1080, 1887), "稳定答案 + 可追踪证据链", 38, PAPER, bold=True, anchor="ma")
    text(draw, (1080, 1940), "equivalence clustering · consensus lock · final judge", 20, "#C7CFCC", anchor="ma", latin=True)

    # Feedback band.
    draw.rounded_rectangle((145, 2070, 2015, 2305), radius=8, fill=PAPER, outline=FAINT, width=3)
    text(draw, (190, 2120), "05", 22, GOLD, bold=True, latin=True)
    text(draw, (245, 2117), "评测诊断与持续迭代", 35, INK, bold=True)
    text(draw, (245, 2164), "EVALUATION, ERROR ANALYSIS & FEEDBACK", 18, MUTED, bold=True, latin=True)
    stages = [
        ("JSONL trace", BLUE),
        ("Accuracy", TEAL),
        ("Root cause", CLAY),
        ("Markdown report", GOLD),
        ("Router feedback", SAGE),
    ]
    sx = 245
    for idx, (label, color) in enumerate(stages):
        pill(draw, (sx, 2215, sx + 250, 2263), label, "#ECEFEC", color, size=18)
        if idx < len(stages) - 1:
            arrow(draw, (sx + 258, 2239), (sx + 294, 2239), MUTED, width=3, head=10)
        sx += 330
    # Thin feedback arc returning to routing.
    line(draw, ((1900, 2070), (1900, 2025), (1690, 2025), (1690, 620), (1440, 620)), fill=SAGE, width=3)
    arrow(draw, (1440, 620), (1378, 620), SAGE, width=3, head=14)
    text(draw, (1720, 1994), "feedback", 17, SAGE, bold=True, latin=True)

    # Evidence metrics.
    text(draw, (145, 2402), "IMPLEMENTED EVIDENCE", 20, MUTED, bold=True, latin=True)
    line(draw, ((145, 2440), (2015, 2440)), fill=FAINT, width=2)
    metric(draw, 145, 2490, "5,090", "HARP 路由评测记录", "full distilled artifact", BLUE)
    metric(draw, 520, 2490, "28", "蒸馏推理模板", "skills + proof methods", SAGE)
    metric(draw, 840, 2490, "80.94%", "领域路由准确率", "full artifact", TEAL)
    metric(draw, 1235, 2490, "93.23%", "证明模板覆盖率", "full artifact", GOLD)
    metric(draw, 1635, 2490, "95.24%", "混合动作一致率", "RLoT holdout", CLAY)

    line(draw, ((145, 2688), (2015, 2688)), fill=INK, width=2)
    text(draw, (145, 2730), "RLoT POLICY", 18, TEAL, bold=True, latin=True)
    text(draw, (315, 2730), "2,502 parameters  ·  530 transitions  ·  3,000 updates", 18, INK, latin=True)
    text(draw, (2015, 2730), "ROUTING / HOLDOUT METRICS ≠ END-TO-END SOLVING ACCURACY", 16, MUTED, anchor="ra", latin=True)
    text(draw, (145, 2770), "NaviMath · current implemented system snapshot", 16, MUTED, latin=True)
    text(draw, (2015, 2770), "github.com/Severus-C/NaviMath", 16, MUTED, anchor="ra", latin=True)

    image.save(OUT, format="PNG", optimize=True, dpi=(300, 300))
    print(OUT)


if __name__ == "__main__":
    main()
