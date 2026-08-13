"""Build paired A2-NLP poster concepts from the UW templates.

Outputs are editable PowerPoint files plus print PDFs and PNG previews. The layout functions use
recursive/axis-aligned splits: each design starts with one content rectangle and partitions it
into a small number of visual regions before placing text or figures.

FROM PR #77, WITH TWO THINGS CHANGED AND ONE STILL OPEN. Leon wrote this and it is the only tool
in the project that turns a board into something printable, which is why it is here. Three notes
for whoever picks it up.

The docstring used to say "3x4-foot" and the eight committed outputs were 36 x 48 inches. The UW
vertical template is 24 x 36 -- 2.25x smaller in area. That was my error before it was anybody
else's: the build sheet asserted 3ft x 4ft for a fortnight and this file inherited it from there.
`assert_template_size()` below now fails loudly rather than rendering a poster nobody can print,
because the defect is invisible until the printer refuses it.

THE LAYOUTS ARE STILL SIZED FOR THE LARGER BOARD. Seven geometry blocks place content out to
x = 34.75 in, which is ten inches past the right edge of a 24 in board. Re-fitting them is design
work rather than a bug fix, so it is left rather than guessed at -- see the note in
`build_classic()`.

CONTENT COMES FROM THE BOARD FILE, for the bottom poster. This file arrived with both posters'
words in two hand-written dicts, which is the one rule this project keeps and which had already
drifted -- the sources line still cited two filenames that had been renamed. `board_content.py`
reads the nine cells and three strip blocks out of `12-the-bottom-board.md`, where they are
already written to the 55-word measure. The YORUBA dict below is still hand-written, because
report 13 is prose rather than panel text and there is nothing to lift; that is the remaining
piece.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fitz
import win32com.client
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


from board_content import bottom_board, check as check_board

ROOT = Path(__file__).resolve().parent
TEMPLATES = ROOT / "templates"
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT.parent / "reports" / "figures"

PORTRAIT_TEMPLATE = TEMPLATES / "ResearchPoster_Template_Vertical_2023.pptx"
LANDSCAPE_TEMPLATE = TEMPLATES / "ResearchPoster_Template_Horizontal_2023.pptx"

UW_PURPLE = "32006E"
UW_GOLD = "E8D3A2"
INK = "1F1F24"
MUTED = "5E5A66"
PAPER = "FFFFFF"
PANEL = "F5F3F7"
LAVENDER = "EEEAF4"
GOLD_TINT = "F8F2E4"
GREEN = "1FAE7A"
BLUE = "2F78D0"
ORANGE = "F0652F"
RED = "B63A3A"
LINE = "D9D2E3"

HEADLINE_FONT = "Encode Sans Normal"
SUBHEAD_FONT = "Uni Sans Book"
BODY_FONT = "Open Sans"


YORUBA = {
    "title": "When Is It Worth Training Your Own Model?",
    "subtitle": "Yoruba from scratch against multilingual transfer",
    "eyebrow": "A2-NLP · LOW-RESOURCE LANGUAGE MODELING",
    "takeaway": (
        "A 33.8M-parameter Yoruba model is ahead of mmBERT on topic classification; "
        "the strongest evidence points to vocabulary fit—not text scarcity—as the leverage."
    ),
    "question": (
        "Low-resource NLP usually defaults to transfer: start from a multilingual encoder and "
        "fine-tune it. We tested the alternative—pretrain a small encoder only on Yoruba, with a "
        "vocabulary fitted to Yoruba—against XLM-R and mmBERT."
    ),
    "headline": (
        "On SIB-200, the 33.8M from-scratch model reached 0.688 macro-F1 versus 0.582 for mmBERT "
        "(five seeds; learning rates chosen on the dev split). The 0.106 margin is much larger "
        "than seed noise, but bootstrap intervals overlap by 0.004: say “ahead,” not “beats.”"
    ),
    "ner": (
        "MasakhaNER tells a different story: from scratch scored 0.837, behind mmBERT at 0.863. "
        "An untrained encoder still reached 0.626, showing how much entity recognition can obtain "
        "from capitalization and name shape. NER learning rates were selected on the scored items, "
        "so these values are descriptive, not a held-out model-selection comparison."
    ),
    "data": (
        "FineWeb-2 contains 69.1M Yoruba tokens. At fixed compute, an English control gained "
        "nothing measurable from 64M to 1,024M tokens, and Yoruba’s 16M→64M change also sat inside "
        "seed variation. At these budgets, lack of additional text does not explain the result."
    ),
    "tokenizer": (
        "XLM-R needs 1.76 tokens for each token used by a Yoruba-specific 16k BPE—second-highest "
        "among 17 languages. A 128-token window holds about 44 Yoruba words with XLM-R versus 77 "
        "with the fitted vocabulary. Across languages, this penalty tracks XLM-R coverage."
    ),
    "causal": (
        "Holding architecture, Yoruba text and compute fixed, swapping only the vocabulary changed "
        "downstream scores. The 16k vocabulary led the 250k XLM-R vocabulary by +0.144 on SIB-200 "
        "and +0.061 on MasakhaNER; every seed won in both tasks (four seeds per arm, exact p=0.029)."
    ),
    "variance": (
        "The pretraining mean did not separate after six registered seeds (0.930 vs 0.989 bits/char; "
        "p=0.374). Variability did: sd 0.037 versus 0.145 (F=15.1, p=0.0098). A poorly fitted "
        "vocabulary is not a guaranteed loss penalty; it makes an individual run less predictable."
    ),
    "limits": (
        "Causal downstream evidence is still Yoruba-only. SIB-200 has just 204 test items, and NER "
        "did not receive the same dev-selection protocol. The next decisive test is the matched-"
        "compute vocabulary swap across four or five languages spanning the coverage gradient."
    ),
    "methods": (
        "RoBERTa-style masked LMs trained from random initialization; shared 16k BPEs; fixed "
        "update-token budgets; SIB-200 topic classification and MasakhaNER evaluation. Results "
        "come from committed run records, recomputed 11 Aug 2026."
    ),
    "stats": [
        ("0.688", "SIB-200 macro-F1 · from scratch"),
        ("1.76×", "XLM-R tokens per fitted token"),
        ("69.1M", "available Yoruba training tokens"),
        ("33.8M", "parameters in the small encoder"),
    ],
    "sources": (
        "Data: FineWeb-2 Yoruba, SIB-200, MasakhaNER · Models: XLM-R base, mmBERT base · "
        "Source: reports/13-the-top-board.md and committed runs/ records · 11 Aug 2026"
    ),
}


FACTORY = {
    "title": "Building a Model Factory",
    "subtitle": "From one-off notebooks to repeatable, challengeable experiments",
    "eyebrow": "A2-NLP · EXPERIMENT INFRASTRUCTURE",
    "takeaway": (
        "The factory made hundreds of model comparisons affordable and auditable—then gave "
        "collaborators enough control to disprove results that initially looked convincing."
    ),
    "question": (
        "A scaling study varies data, compute, model size, language, seed and learning rate. "
        "Managing that grid by notebook state and filenames is not reproducibility. The factory "
        "turns each cell into a self-describing job, checkpoint and result record."
    ),
    "workflow": (
        "Prepare once → inspect and fingerprint → estimate on the actual GPU → pretrain one cell "
        "or queue a fleet → read curves and results through the same API. Interactive exploration "
        "stays in notebooks; expensive work moves to restartable processes."
    ),
    "speed": (
        "The same four cells fell from 25.2 to 12.2 minutes: 2.07× end-to-end. Only 1.32× was true "
        "efficiency from a better batch; the rest was two cards working in parallel. Reporting the "
        "decomposition keeps hardware scale from masquerading as algorithmic efficiency."
    ),
    "pipeline": (
        "On the Yoruba corpus, reading, fitting the BPE, encoding 260M characters and moving the "
        "flat token store to GPU took 53 seconds. One 98M-parameter training run took 85 minutes—a "
        "96× ratio that defines the boundary between interactive work and unattended queues."
    ),
    "records": (
        "Every setting that can move a number belongs in the record. Corpora and vocabularies carry "
        "fingerprints; run reuse refuses mismatched settings. The current evidence base contains "
        "197 pretraining runs, 278 fine-tuning records and 892 individual fine-tuning runs."
    ),
    "api": (
        "Collaborators touch nine functions in mlm_api: prepare, inspect, stream, estimate, "
        "pretrain, build controls and read results. The surface stays small while schedulers, "
        "masking, checkpoints, dashboards and validation checks evolve behind it."
    ),
    "statistics": (
        "At three seeds per arm, a mean difference must be about 2.27× the pooled sample spread to "
        "clear a two-sided 0.05 t-threshold. An exact permutation test cannot return below p=0.10 "
        "with 3 vs 3. A pre-registered sample size limits what a run grid is allowed to claim."
    ),
    "failure": (
        "Early stopping could not rescue collapsed runs: healthy and doomed runs overlapped at all "
        "11 checkpoints tested. Learning-rate settings also failed to transfer safely—7e-4 was "
        "best for three languages but collapsed Igbo. The usable band must be measured per language."
    ),
    "honesty": (
        "The claims gate currently reports 6 supported, 2 unsupported and 1 underpowered claim. "
        "That is a feature: numbers regenerate from records, comparative claims are tested against "
        "their nulls, and failed claims remain visible instead of being silently rewritten."
    ),
    "stats": [
        ("9", "public MLM API functions"),
        ("2.07×", "faster end-to-end · 1.32× efficiency"),
        ("197", "pretraining run records"),
        ("1.024B", "update tokens in a full budget"),
    ],
    "sources": (
        "Source: reports/09-the-poster.md and reports/12-poster-build-sheet-v2.md · "
        "Counts and figures regenerate from committed runs/ records · 12 Aug 2026"
    ),
}


@dataclass(frozen=True)
class FigureSpec:
    path: Path
    x: float
    y: float
    w: float
    h: float
    name: str


def assert_template_size(prs, expect=(24.0, 36.0), name="template") -> None:
    """Refuse to build a poster at a size nobody can print.

    PR #77 shipped eight posters at 36 x 48 because the build sheet said 3ft x 4ft and nobody
    measured the template. The failure is silent all the way to the print shop, so it gets an
    assertion rather than a comment: a wrong-size poster looks exactly like a right-size one on
    screen, which is the whole problem.
    """
    w = prs.slide_width / 914400
    h = prs.slide_height / 914400
    if (round(w, 2), round(h, 2)) != expect:
        raise SystemExit(
            f"{name} is {w:.2f} x {h:.2f} in; expected {expect[0]} x {expect[1]}. "
            f"The UW vertical template is 24 x 36. If this is deliberate, change `expect` "
            f"and re-fit the layouts -- they place content out to x = 34.75 in, which does "
            f"not fit a 24 in board."
        )


def rgb(value: str) -> RGBColor:
    return RGBColor.from_string(value)


def delete_slide(prs: Presentation, index: int) -> None:
    slide_id = prs.slides._sldIdLst[index]
    rel_id = slide_id.rId
    prs.part.drop_rel(rel_id)
    del prs.slides._sldIdLst[index]


def clear_slide(slide) -> None:
    for shape in list(slide.shapes):
        slide.shapes._spTree.remove(shape._element)


def open_template(path: Path, slide_index: int = 0) -> tuple[Presentation, object]:
    prs = Presentation(path)
    for index in reversed(range(len(prs.slides))):
        if index != slide_index:
            delete_slide(prs, index)
    slide = prs.slides[0]
    clear_slide(slide)
    return prs, slide


def set_cell_margins(text_frame, margin: float = 0.08) -> None:
    text_frame.margin_left = Inches(margin)
    text_frame.margin_right = Inches(margin)
    text_frame.margin_top = Inches(margin)
    text_frame.margin_bottom = Inches(margin)


def add_text(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    size: float,
    *,
    color: str = INK,
    font: str = BODY_FONT,
    bold: bool = False,
    italic: bool = False,
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.TOP,
    margin: float = 0.03,
    name: str | None = None,
):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        shape.name = name
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.vertical_anchor = valign
    set_cell_margins(tf, margin)
    paragraph = tf.paragraphs[0]
    paragraph.alignment = align
    paragraph.space_after = Pt(0)
    paragraph.line_spacing = 1.05
    run = paragraph.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = rgb(color)
    return shape


def add_rect(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str = PAPER,
    line: str | None = None,
    radius: bool = False,
    name: str | None = None,
):
    shape_type = (
        MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE
        if radius
        else MSO_AUTO_SHAPE_TYPE.RECTANGLE
    )
    shape = slide.shapes.add_shape(
        shape_type, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    if line:
        shape.line.color.rgb = rgb(line)
        shape.line.width = Pt(0.8)
    else:
        shape.line.fill.background()
    if name:
        shape.name = name
    return shape


def add_line(slide, x1: float, y1: float, x2: float, y2: float, color=UW_GOLD, width=1.2):
    line = slide.shapes.add_connector(
        1, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    line.line.color.rgb = rgb(color)
    line.line.width = Pt(width)
    return line


def add_header(slide, content: dict, *, landscape: bool = False) -> None:
    if landscape:
        add_text(
            slide,
            1.25,
            0.82,
            4.7,
            0.35,
            content["eyebrow"],
            10,
            color=UW_GOLD,
            font=SUBHEAD_FONT,
            bold=True,
            name="Header_Eyebrow",
        )
        add_text(
            slide,
            1.25,
            1.12,
            28.4,
            1.62,
            content["title"],
            50,
            color=PAPER,
            font=HEADLINE_FONT,
            bold=True,
            valign=MSO_ANCHOR.MIDDLE,
            name="Header_Title",
        )
        add_text(
            slide,
            1.28,
            2.78,
            26.5,
            0.55,
            content["subtitle"],
            20,
            color=PAPER,
            font=SUBHEAD_FONT,
            name="Header_Subtitle",
        )
    else:
        # The vertical template master carries the UW wordmark and a tiny
        # "research poster template" label above it. Cover only that label.
        add_rect(slide, 1.42, 1.10, 4.35, 0.30, fill=UW_PURPLE, name="MasterLabelCover")
        add_text(
            slide,
            1.50,
            2.10,
            19.3,
            2.05,
            content["title"],
            48,
            color=PAPER,
            font=HEADLINE_FONT,
            bold=True,
            valign=MSO_ANCHOR.MIDDLE,
            name="Header_Title",
        )
        add_text(
            slide,
            1.53,
            4.35,
            19.0,
            0.62,
            content["subtitle"],
            18,
            color=PAPER,
            font=SUBHEAD_FONT,
            name="Header_Subtitle",
        )
        add_text(
            slide,
            1.53,
            5.22,
            18.0,
            0.35,
            "A2-NLP Project Team · CSED 504 · University of Washington · Summer 2026",
            9.5,
            color=UW_GOLD,
            font=BODY_FONT,
            name="Header_Author",
        )


def add_takeaway(slide, content: dict, x: float, y: float, w: float, h: float) -> None:
    add_rect(slide, x, y, w, h, fill=GOLD_TINT, line=UW_GOLD, name="Takeaway_Background")
    add_text(
        slide,
        x + 0.28,
        y + 0.16,
        1.55,
        h - 0.28,
        "TAKEAWAY",
        10.5,
        color=UW_PURPLE,
        font=SUBHEAD_FONT,
        bold=True,
        valign=MSO_ANCHOR.MIDDLE,
        name="Takeaway_Label",
    )
    add_text(
        slide,
        x + 1.72,
        y + 0.15,
        w - 2.0,
        h - 0.25,
        content["takeaway"],
        15.5,
        color=INK,
        font=BODY_FONT,
        bold=True,
        valign=MSO_ANCHOR.MIDDLE,
        name="Takeaway_Text",
    )


def add_panel(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    heading: str,
    body: str,
    *,
    fill: str = PANEL,
    accent: str = UW_PURPLE,
    body_size: float = 13.3,
    heading_size: float = 19.0,
    label: str | None = None,
    name: str = "Panel",
):
    add_rect(slide, x, y, w, h, fill=fill, line=LINE, radius=True, name=f"{name}_Box")
    add_rect(slide, x, y, 0.11, h, fill=accent, name=f"{name}_Accent")
    if label:
        add_text(
            slide,
            x + 0.30,
            y + 0.23,
            w - 0.58,
            0.28,
            label.upper(),
            8.7,
            color=accent,
            font=SUBHEAD_FONT,
            bold=True,
            name=f"{name}_Label",
        )
        heading_y = y + 0.60
    else:
        heading_y = y + 0.27
    add_text(
        slide,
        x + 0.30,
        heading_y,
        w - 0.58,
        0.68,
        heading,
        heading_size,
        color=accent,
        font=HEADLINE_FONT,
        bold=True,
        name=f"{name}_Heading",
    )
    add_text(
        slide,
        x + 0.30,
        heading_y + 0.83,
        w - 0.58,
        h - (heading_y - y) - 1.03,
        body,
        body_size,
        color=INK,
        font=BODY_FONT,
        name=f"{name}_Body",
    )


def add_stat(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    value: str,
    label: str,
    *,
    fill: str = UW_PURPLE,
    value_color: str = PAPER,
    label_color: str = UW_GOLD,
    value_size: float = 29,
    name: str = "Stat",
):
    add_rect(slide, x, y, w, h, fill=fill, radius=True, name=f"{name}_Box")
    add_text(
        slide,
        x + 0.14,
        y + 0.13,
        w - 0.28,
        h * 0.55,
        value,
        value_size,
        color=value_color,
        font=HEADLINE_FONT,
        bold=True,
        align=PP_ALIGN.CENTER,
        valign=MSO_ANCHOR.MIDDLE,
        name=f"{name}_Value",
    )
    add_text(
        slide,
        x + 0.18,
        y + h * 0.60,
        w - 0.36,
        h * 0.27,
        label,
        9.4,
        color=label_color,
        font=BODY_FONT,
        bold=True,
        align=PP_ALIGN.CENTER,
        valign=MSO_ANCHOR.MIDDLE,
        name=f"{name}_Label",
    )


def add_figure_frame(
    slide,
    figure_specs: list[FigureSpec],
    filename: str,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    label: str | None = None,
    caption: str | None = None,
    fill: str = PAPER,
    pad: float = 0.15,
    name: str = "Figure",
) -> None:
    add_rect(slide, x, y, w, h, fill=fill, line=LINE, radius=True, name=f"{name}_Frame")
    label_h = 0.42 if label else 0
    caption_h = 0.46 if caption else 0
    if label:
        add_text(
            slide,
            x + 0.22,
            y + 0.12,
            w - 0.44,
            0.27,
            label.upper(),
            8.6,
            color=UW_PURPLE,
            font=SUBHEAD_FONT,
            bold=True,
            name=f"{name}_Label",
        )
    if caption:
        add_text(
            slide,
            x + 0.24,
            y + h - 0.42,
            w - 0.48,
            0.28,
            caption,
            8.7,
            color=MUTED,
            font=BODY_FONT,
            italic=True,
            align=PP_ALIGN.CENTER,
            name=f"{name}_Caption",
        )
    ext = Path(filename).suffix
    source = FIGURES / filename
    if not source.exists() and ext.lower() == ".svg":
        source = FIGURES / f"{Path(filename).stem}.png"
    figure_specs.append(
        FigureSpec(
            source,
            x + pad,
            y + pad + label_h,
            w - 2 * pad,
            h - 2 * pad - label_h - caption_h,
            name,
        )
    )


def add_footer(slide, content: dict, *, landscape: bool = False) -> None:
    if landscape:
        y = 23.36
        add_line(slide, 1.25, y, 34.75, y, UW_GOLD, 1)
        add_text(
            slide,
            1.25,
            y + 0.10,
            31.0,
            0.34,
            content["sources"],
            7.4,
            color=MUTED,
            font=BODY_FONT,
            name="Footer_Sources",
        )
        add_text(
            slide,
            32.2,
            y + 0.10,
            2.55,
            0.34,
            "A2-NLP · 2026",
            7.8,
            color=UW_PURPLE,
            font=SUBHEAD_FONT,
            bold=True,
            align=PP_ALIGN.RIGHT,
            name="Footer_Tag",
        )
    else:
        y = 35.16
        add_line(slide, 1.50, y, 22.50, y, UW_GOLD, 1)
        add_text(
            slide,
            1.50,
            y + 0.10,
            18.7,
            0.42,
            content["sources"],
            7.2,
            color=MUTED,
            font=BODY_FONT,
            name="Footer_Sources",
        )
        add_text(
            slide,
            20.0,
            y + 0.10,
            2.5,
            0.42,
            "A2-NLP · 2026",
            7.8,
            color=UW_PURPLE,
            font=SUBHEAD_FONT,
            bold=True,
            align=PP_ALIGN.RIGHT,
            name="Footer_Tag",
        )


def build_classic(content: dict, kind: str) -> tuple[Presentation, list[FigureSpec]]:
    prs, slide = open_template(PORTRAIT_TEMPLATE, 0)
    figures: list[FigureSpec] = []
    add_header(slide, content)
    add_takeaway(slide, content, 1.5, 7.65, 21.0, 1.16)

    if kind == "yoruba":
        add_panel(
            slide,
            1.5,
            9.20,
            6.5,
            8.05,
            "The choice",
            content["question"] + "\n\n" + content["headline"],
            label="Research question",
            body_size=15.5,
            name="Choice",
        )
        add_figure_frame(
            slide,
            figures,
            "01-headline.svg",
            8.4,
            9.20,
            14.1,
            8.05,
            label="Headline result",
            caption="SIB-200 uses dev-selected rates; NER does not.",
            name="HeadlineFigure",
        )
        add_figure_frame(
            slide,
            figures,
            "02-tokenizer-gradient.svg",
            1.5,
            17.72,
            10.28,
            7.65,
            label="Coverage gradient",
            caption="Tokenizer fertility across 17 languages.",
            name="GradientFigure",
        )
        add_figure_frame(
            slide,
            figures,
            "05-data-saturation.svg",
            12.22,
            17.72,
            10.28,
            7.65,
            label="Data control",
            caption="More text stops moving loss at these budgets.",
            name="DataFigure",
        )
        add_figure_frame(
            slide,
            figures,
            "17-tokenizer-lottery.svg",
            1.5,
            25.84,
            10.28,
            8.82,
            label="Run-to-run variability",
            caption="Six registered seeds separate variance, not mean.",
            name="LotteryFigure",
        )
        add_panel(
            slide,
            12.22,
            25.84,
            10.28,
            8.82,
            "What survives",
            content["tokenizer"]
            + "\n\n"
            + content["causal"]
            + "\n\n"
            + "Boundary: "
            + content["limits"],
            fill=LAVENDER,
            body_size=14.0,
            heading_size=20,
            name="Conclusions",
        )
    else:
        add_panel(
            slide,
            1.5,
            9.20,
            6.5,
            8.05,
            "Why a factory?",
            content["question"] + "\n\n" + content["workflow"],
            label="System goal",
            body_size=15.5,
            name="FactoryQuestion",
        )
        add_figure_frame(
            slide,
            figures,
            "14-where-the-speedup-came-from.svg",
            8.4,
            9.20,
            14.1,
            8.05,
            label="Measured speedup",
            caption="Efficiency and parallelism are reported separately.",
            name="SpeedFigure",
        )
        add_figure_frame(
            slide,
            figures,
            "15-what-a-run-is-made-of.svg",
            1.5,
            17.72,
            10.28,
            7.65,
            label="Notebook / queue boundary",
            caption="Cheap stages stay interactive; training becomes a job.",
            name="PipelineFigure",
        )
        add_figure_frame(
            slide,
            figures,
            "19-the-interface.svg",
            12.22,
            17.72,
            10.28,
            7.65,
            label="Public surface",
            caption="Nine stable calls hide the machinery.",
            name="InterfaceFigure",
        )
        add_figure_frame(
            slide,
            figures,
            "07-dashboard.png",
            1.5,
            25.84,
            10.28,
            8.82,
            label="Operations",
            caption="Queue, GPU state and comparable run records.",
            name="DashboardFigure",
        )
        add_panel(
            slide,
            12.22,
            25.84,
            10.28,
            8.82,
            "What made it trustworthy",
            content["records"]
            + "\n\n"
            + content["statistics"]
            + "\n\n"
            + content["honesty"],
            fill=LAVENDER,
            body_size=14.0,
            heading_size=20,
            name="Trust",
        )
    add_footer(slide, content)
    return prs, figures


def build_spine(content: dict, kind: str) -> tuple[Presentation, list[FigureSpec]]:
    prs, slide = open_template(PORTRAIT_TEMPLATE, 1)
    figures: list[FigureSpec] = []
    add_header(slide, content)
    add_takeaway(slide, content, 1.5, 7.45, 21.0, 1.16)

    left_x, mid_x, right_x = 1.5, 8.75, 15.85
    left_w, mid_w, right_w = 6.8, 6.65, 6.65
    body_y, body_h = 9.02, 25.62
    add_rect(slide, mid_x, body_y, mid_w, body_h, fill=UW_PURPLE, name="NarrativeSpine")

    if kind == "yoruba":
        add_figure_frame(
            slide,
            figures,
            "02-tokenizer-gradient.svg",
            left_x,
            body_y,
            left_w,
            8.45,
            label="Why vocabulary fit matters",
            name="SpineGradient",
        )
        add_panel(
            slide,
            left_x,
            17.85,
            left_w,
            6.25,
            "Enough text",
            content["data"],
            body_size=13.5,
            heading_size=18,
            name="SpineData",
        )
        add_figure_frame(
            slide,
            figures,
            "05-data-saturation.svg",
            left_x,
            24.48,
            left_w,
            10.16,
            label="Fixed-compute control",
            name="SpineDataFigure",
        )
        add_text(
            slide,
            mid_x + 0.40,
            body_y + 0.50,
            mid_w - 0.8,
            0.46,
            "THE EVIDENCE CHAIN",
            10,
            color=UW_GOLD,
            font=SUBHEAD_FONT,
            bold=True,
            align=PP_ALIGN.CENTER,
            name="SpineLabel",
        )
        for i, (value, label) in enumerate(content["stats"]):
            add_text(
                slide,
                mid_x + 0.45,
                body_y + 1.35 + i * 3.55,
                mid_w - 0.9,
                1.25,
                value,
                31,
                color=PAPER,
                font=HEADLINE_FONT,
                bold=True,
                align=PP_ALIGN.CENTER,
                valign=MSO_ANCHOR.MIDDLE,
                name=f"SpineStat{i}_Value",
            )
            add_text(
                slide,
                mid_x + 0.6,
                body_y + 2.58 + i * 3.55,
                mid_w - 1.2,
                0.67,
                label,
                10.2,
                color=UW_GOLD,
                font=BODY_FONT,
                bold=True,
                align=PP_ALIGN.CENTER,
                name=f"SpineStat{i}_Label",
            )
            if i < 3:
                add_line(
                    slide,
                    mid_x + 1.5,
                    body_y + 4.47 + i * 3.55,
                    mid_x + mid_w - 1.5,
                    body_y + 4.47 + i * 3.55,
                    UW_GOLD,
                    0.7,
                )
        add_text(
            slide,
            mid_x + 0.48,
            body_y + 16.25,
            mid_w - 0.96,
            7.8,
            "MEAN\nNo reliable matched-compute difference in pretraining bits per character "
            "(Welch p=0.374).\n\n"
            "VARIANCE\nThe 250k-vocabulary arm was nearly four times as variable "
            "(sd 0.145 vs 0.037; p=0.0098).\n\n"
            "DOWNSTREAM\nHolding compute fixed, the fitted vocabulary led by +0.144 on SIB-200 "
            "and +0.061 on NER; every seed won.\n\n"
            "NEXT TEST\nRepeat the matched-compute vocabulary swap across the coverage gradient. "
            "Causal downstream evidence is still Yoruba-only.",
            13.8,
            color=PAPER,
            font=BODY_FONT,
            name="SpineInterpretation",
        )
        add_figure_frame(
            slide,
            figures,
            "01-headline.svg",
            right_x,
            body_y,
            right_w,
            9.28,
            label="Downstream result",
            name="SpineHeadline",
        )
        add_panel(
            slide,
            right_x,
            18.70,
            right_w,
            6.22,
            "Tasks disagree",
            content["ner"],
            body_size=13.2,
            heading_size=18,
            name="SpineTasks",
        )
        add_figure_frame(
            slide,
            figures,
            "17-tokenizer-lottery.svg",
            right_x,
            25.30,
            right_w,
            9.34,
            label="Variance, not mean",
            name="SpineLottery",
        )
    else:
        add_figure_frame(
            slide,
            figures,
            "14-where-the-speedup-came-from.svg",
            left_x,
            body_y,
            left_w,
            8.45,
            label="Measure the speedup",
            name="SpineSpeed",
        )
        add_panel(
            slide,
            left_x,
            17.85,
            left_w,
            6.25,
            "Split by cost",
            content["pipeline"],
            body_size=13.5,
            heading_size=18,
            name="SpinePipeline",
        )
        add_figure_frame(
            slide,
            figures,
            "15-what-a-run-is-made-of.svg",
            left_x,
            24.48,
            left_w,
            10.16,
            label="Interactive → unattended",
            name="SpinePipelineFigure",
        )
        add_text(
            slide,
            mid_x + 0.4,
            body_y + 0.50,
            mid_w - 0.8,
            0.46,
            "THE FACTORY LOOP",
            10,
            color=UW_GOLD,
            font=SUBHEAD_FONT,
            bold=True,
            align=PP_ALIGN.CENTER,
            name="SpineLabel",
        )
        steps = [
            ("01", "PREPARE", "tokenize once; fingerprint"),
            ("02", "ESTIMATE", "measure this GPU"),
            ("03", "PRETRAIN", "one cell or a fleet"),
            ("04", "READ", "curves and result records"),
        ]
        for i, (num, title, desc) in enumerate(steps):
            sy = body_y + 1.45 + i * 3.55
            add_text(
                slide,
                mid_x + 0.55,
                sy,
                1.15,
                0.72,
                num,
                16,
                color=UW_GOLD,
                font=HEADLINE_FONT,
                bold=True,
                name=f"Step{i}_Num",
            )
            add_text(
                slide,
                mid_x + 1.65,
                sy,
                mid_w - 2.1,
                0.58,
                title,
                16,
                color=PAPER,
                font=HEADLINE_FONT,
                bold=True,
                name=f"Step{i}_Title",
            )
            add_text(
                slide,
                mid_x + 1.65,
                sy + 0.72,
                mid_w - 2.1,
                0.62,
                desc,
                9.7,
                color=UW_GOLD,
                font=BODY_FONT,
                name=f"Step{i}_Desc",
            )
            if i < 3:
                add_line(
                    slide,
                    mid_x + mid_w / 2,
                    sy + 1.73,
                    mid_x + mid_w / 2,
                    sy + 3.1,
                    UW_GOLD,
                    1,
                )
        add_text(
            slide,
            mid_x + 0.5,
            body_y + 16.25,
            mid_w - 1.0,
            7.8,
            "THE RESULT\nThe claims gate reports 6 supported, 2 unsupported and 1 underpowered "
            "claim. Failures stay visible.\n\n"
            "THE STANDARD\nA result must be reproducible enough for another person to challenge "
            "and overturn it. That happened twice—and is evidence the tooling worked.",
            15.0,
            color=PAPER,
            font=BODY_FONT,
            name="SpineFactoryResult",
        )
        add_figure_frame(
            slide,
            figures,
            "19-the-interface.svg",
            right_x,
            body_y,
            right_w,
            9.28,
            label="Nine-function surface",
            name="SpineInterface",
        )
        add_panel(
            slide,
            right_x,
            18.70,
            right_w,
            6.22,
            "Records survive",
            content["records"],
            body_size=13.2,
            heading_size=18,
            name="SpineRecords",
        )
        add_figure_frame(
            slide,
            figures,
            "07-dashboard.png",
            right_x,
            25.30,
            right_w,
            9.34,
            label="One operational view",
            name="SpineDashboard",
        )
    add_footer(slide, content)
    return prs, figures


def build_evidence_grid(content: dict, kind: str) -> tuple[Presentation, list[FigureSpec]]:
    prs, slide = open_template(LANDSCAPE_TEMPLATE, 0)
    figures: list[FigureSpec] = []
    add_header(slide, content, landscape=True)
    add_takeaway(slide, content, 1.25, 4.25, 33.5, 1.05)

    if kind == "yoruba":
        add_figure_frame(
            slide,
            figures,
            "01-headline.svg",
            1.25,
            5.72,
            19.9,
            9.12,
            label="Result",
            name="GridHeadline",
        )
        add_panel(
            slide,
            21.58,
            5.72,
            13.17,
            5.20,
            "A small model, a precise claim",
            content["headline"],
            fill=LAVENDER,
            body_size=16.0,
            heading_size=20,
            name="GridClaim",
        )
        add_panel(
            slide,
            21.58,
            11.34,
            13.17,
            3.50,
            "Causal vocabulary swap",
            content["causal"],
            body_size=13.5,
            heading_size=17,
            name="GridTask",
        )
        add_figure_frame(
            slide,
            figures,
            "02-tokenizer-gradient.svg",
            1.25,
            15.28,
            10.78,
            7.55,
            label="1 · Coverage",
            name="GridGradient",
        )
        add_figure_frame(
            slide,
            figures,
            "05-data-saturation.svg",
            12.45,
            15.28,
            10.78,
            7.55,
            label="2 · Data",
            name="GridData",
        )
        add_figure_frame(
            slide,
            figures,
            "17-tokenizer-lottery.svg",
            23.65,
            15.28,
            11.10,
            7.55,
            label="3 · Variability",
            name="GridLottery",
        )
    else:
        add_figure_frame(
            slide,
            figures,
            "14-where-the-speedup-came-from.svg",
            1.25,
            5.72,
            17.0,
            9.12,
            label="Measure",
            name="GridSpeed",
        )
        add_panel(
            slide,
            18.68,
            5.72,
            16.07,
            4.48,
            "From notebook state to evidence",
            content["question"] + "\n\n" + content["speed"],
            fill=LAVENDER,
            body_size=14.5,
            heading_size=20,
            name="GridFactoryIntro",
        )
        stat_w = 3.68
        for i, (value, label) in enumerate(content["stats"]):
            add_stat(
                slide,
                18.68 + i * (stat_w + 0.43),
                10.65,
                stat_w,
                4.19,
                value,
                label,
                value_size=24,
                name=f"GridStat{i}",
            )
        add_figure_frame(
            slide,
            figures,
            "15-what-a-run-is-made-of.svg",
            1.25,
            15.28,
            10.78,
            7.55,
            label="1 · Split by cost",
            name="GridPipeline",
        )
        add_figure_frame(
            slide,
            figures,
            "19-the-interface.svg",
            12.45,
            15.28,
            10.78,
            7.55,
            label="2 · Hide complexity",
            name="GridInterface",
        )
        add_figure_frame(
            slide,
            figures,
            "13-how-many-seeds.svg",
            23.65,
            15.28,
            11.10,
            7.55,
            label="3 · Limit the claim",
            name="GridSeeds",
        )
    add_footer(slide, content, landscape=True)
    return prs, figures


def build_editorial(content: dict, kind: str) -> tuple[Presentation, list[FigureSpec]]:
    prs, slide = open_template(PORTRAIT_TEMPLATE, 0)
    figures: list[FigureSpec] = []
    add_header(slide, content)
    add_takeaway(slide, content, 1.5, 7.65, 21.0, 1.16)

    if kind == "yoruba":
        add_figure_frame(
            slide,
            figures,
            "01-headline.svg",
            1.5,
            9.20,
            14.4,
            10.35,
            label="The result",
            name="EditorialHeadline",
        )
        add_rect(slide, 16.32, 9.20, 6.18, 10.35, fill=UW_PURPLE, name="EditorialStatsBox")
        add_text(
            slide,
            16.78,
            9.72,
            5.25,
            0.45,
            "READ THE RESULT AS A PAIR",
            9.2,
            color=UW_GOLD,
            font=SUBHEAD_FONT,
            bold=True,
            align=PP_ALIGN.CENTER,
            name="EditorialStatsLabel",
        )
        add_text(
            slide,
            16.82,
            10.55,
            5.15,
            1.40,
            "0.688",
            34,
            color=PAPER,
            font=HEADLINE_FONT,
            bold=True,
            align=PP_ALIGN.CENTER,
            name="EditorialStat1",
        )
        add_text(
            slide,
            16.82,
            11.88,
            5.15,
            1.05,
            "topic macro-F1\nfrom scratch",
            10.3,
            color=UW_GOLD,
            font=BODY_FONT,
            bold=True,
            align=PP_ALIGN.CENTER,
            name="EditorialStat1Label",
        )
        add_line(slide, 17.5, 13.25, 21.3, 13.25, UW_GOLD, 0.8)
        add_text(
            slide,
            16.82,
            13.62,
            5.15,
            1.40,
            "0.837",
            34,
            color=PAPER,
            font=HEADLINE_FONT,
            bold=True,
            align=PP_ALIGN.CENTER,
            name="EditorialStat2",
        )
        add_text(
            slide,
            16.82,
            14.95,
            5.15,
            1.05,
            "entity F1\nfrom scratch",
            10.3,
            color=UW_GOLD,
            font=BODY_FONT,
            bold=True,
            align=PP_ALIGN.CENTER,
            name="EditorialStat2Label",
        )
        add_text(
            slide,
            16.75,
            16.55,
            5.35,
            2.25,
            "Ahead of mmBERT on topic; slightly behind on entities. The disagreement is part of the finding.",
            11.0,
            color=PAPER,
            font=BODY_FONT,
            align=PP_ALIGN.CENTER,
            name="EditorialStatNote",
        )
        add_figure_frame(
            slide,
            figures,
            "02-tokenizer-gradient.svg",
            1.5,
            20.05,
            13.15,
            8.40,
            label="The mechanism",
            name="EditorialGradient",
        )
        add_panel(
            slide,
            15.08,
            20.05,
            7.42,
            8.40,
            "Why vocabulary fit",
            content["tokenizer"] + "\n\n" + content["causal"],
            fill=GOLD_TINT,
            accent=ORANGE,
            body_size=13.2,
            heading_size=19,
            name="EditorialMechanism",
        )
        col_w = 6.72
        add_panel(
            slide,
            1.5,
            28.90,
            col_w,
            5.76,
            "Text is not the bottleneck",
            content["data"],
            body_size=13.5,
            heading_size=16.5,
            name="EditorialData",
        )
        add_panel(
            slide,
            8.64,
            28.90,
            col_w,
            5.76,
            "Means and variability",
            content["variance"],
            body_size=13.2,
            heading_size=16.5,
            name="EditorialVariance",
        )
        add_panel(
            slide,
            15.78,
            28.90,
            col_w,
            5.76,
            "Scope",
            content["limits"],
            body_size=13.2,
            heading_size=16.5,
            name="EditorialScope",
        )
    else:
        add_figure_frame(
            slide,
            figures,
            "19-the-interface.svg",
            1.5,
            9.20,
            14.4,
            10.35,
            label="One screen to operate the factory",
            name="EditorialInterface",
        )
        add_rect(slide, 16.32, 9.20, 6.18, 10.35, fill=UW_PURPLE, name="EditorialStatsBox")
        for i, (value, label) in enumerate(content["stats"][:3]):
            sy = 9.62 + i * 3.18
            add_text(
                slide,
                16.72,
                sy,
                5.38,
                1.2,
                value,
                31,
                color=PAPER,
                font=HEADLINE_FONT,
                bold=True,
                align=PP_ALIGN.CENTER,
                name=f"EditorialFactoryStat{i}",
            )
            add_text(
                slide,
                16.72,
                sy + 1.2,
                5.38,
                0.82,
                label,
                9.6,
                color=UW_GOLD,
                font=BODY_FONT,
                bold=True,
                align=PP_ALIGN.CENTER,
                name=f"EditorialFactoryStat{i}Label",
            )
            if i < 2:
                add_line(slide, 17.5, sy + 2.55, 21.3, sy + 2.55, UW_GOLD, 0.8)
        add_figure_frame(
            slide,
            figures,
            "14-where-the-speedup-came-from.svg",
            1.5,
            20.05,
            13.15,
            8.40,
            label="Iteration became affordable",
            name="EditorialSpeed",
        )
        add_panel(
            slide,
            15.08,
            20.05,
            7.42,
            8.40,
            "The operating rule",
            content["workflow"] + "\n\n" + content["pipeline"],
            fill=GOLD_TINT,
            accent=ORANGE,
            body_size=13.2,
            heading_size=19,
            name="EditorialWorkflow",
        )
        col_w = 6.72
        add_panel(
            slide,
            1.5,
            28.90,
            col_w,
            5.76,
            "Identity",
            content["records"],
            body_size=13.2,
            heading_size=16.5,
            name="EditorialRecords",
        )
        add_panel(
            slide,
            8.64,
            28.90,
            col_w,
            5.76,
            "Statistical guardrails",
            content["statistics"],
            body_size=13.0,
            heading_size=16.5,
            name="EditorialStats",
        )
        add_panel(
            slide,
            15.78,
            28.90,
            col_w,
            5.76,
            "Failure is evidence",
            "Healthy and doomed runs overlapped at all 11 early checkpoints, so the proposed "
            "detector could not work. The claims gate reports 6 supported, 2 unsupported and "
            "1 underpowered claim—and keeps the failures visible.",
            body_size=13.0,
            heading_size=16.5,
            name="EditorialFailure",
        )
    add_footer(slide, content)
    return prs, figures


BUILDERS = [
    ("design-01-classic", build_classic),
    ("design-02-narrative-spine", build_spine),
    ("design-03-evidence-grid", build_evidence_grid),
    ("design-04-editorial", build_editorial),
]


def save_base_presentation(prs: Presentation, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(path)


def source_aspect(path: Path) -> float:
    if path.suffix.lower() == ".svg":
        import xml.etree.ElementTree as ET

        root = ET.parse(path).getroot()
        view_box = root.attrib.get("viewBox")
        if view_box:
            _, _, width, height = [float(v) for v in view_box.replace(",", " ").split()]
            return width / height
        width = float(root.attrib["width"].replace("pt", "").replace("px", ""))
        height = float(root.attrib["height"].replace("pt", "").replace("px", ""))
        return width / height
    with Image.open(path) as image:
        return image.width / image.height


def contain_box(spec: FigureSpec) -> tuple[float, float, float, float]:
    aspect = source_aspect(spec.path)
    box_aspect = spec.w / spec.h
    if aspect >= box_aspect:
        width = spec.w
        height = width / aspect
        x = spec.x
        y = spec.y + (spec.h - height) / 2
    else:
        height = spec.h
        width = height * aspect
        x = spec.x + (spec.w - width) / 2
        y = spec.y
    return x, y, width, height


def _restore_reopened_picture_aspects(
    presentation, records, sx: float, sy: float
) -> None:
    """Correct the lazy PageSetup transform after reopening the file."""

    for record in records:
        if (
            record["container_kind"] != "slide"
            or not record["shape_name"].startswith("Embedded_")
        ):
            continue
        slide = presentation.Slides(record["container_index"])
        shape = slide.Shapes.Item(record["shape_name"])
        source = ROOT.parent / Path(shape.AlternativeText)
        aspect = source_aspect(source)
        target_left = record["left"] * sx
        target_top = record["top"] * sy
        target_width = record["width"] * sx
        target_height = record["height"] * sy
        if aspect >= target_width / target_height:
            new_width = target_width
            new_height = new_width / aspect
            new_left = target_left
            new_top = target_top + (target_height - new_height) / 2
        else:
            new_height = target_height
            new_width = new_height * aspect
            new_left = target_left + (target_width - new_width) / 2
            new_top = target_top
        try:
            shape.LockAspectRatio = 0
        except Exception:
            pass
        shape.Width = max(new_width, 0.1)
        shape.Height = max(new_height, 0.1)
        shape.Left = new_left
        shape.Top = new_top


def reflow_for_print(presentation, *, landscape: bool):
    """Convert 2:3 template geometry to the project's 3:4 print target.

    PowerPoint reflows shape boxes when PageSetup changes. We let it perform
    that axis-specific layout expansion, then restore picture aspect ratios
    and scale typography for the larger physical sheet.
    """

    if landscape:
        sx, sy = 4 / 3, 1.5
        font_scale = 1.5
        target_width, target_height = 48 * 72, 36 * 72
    else:
        sx, sy = 1.5, 4 / 3
        font_scale = 1.6
        target_width, target_height = 36 * 72, 48 * 72

    master = presentation.SlideMaster
    containers = [("master", 0, master)]
    for layout_index in range(1, master.CustomLayouts.Count + 1):
        containers.append(
            ("layout", layout_index, master.CustomLayouts(layout_index))
        )
    for slide_index in range(1, presentation.Slides.Count + 1):
        containers.append(("slide", slide_index, presentation.Slides(slide_index)))

    records = []
    for container_kind, container_index, container in containers:
        for shape_index in range(1, container.Shapes.Count + 1):
            shape = container.Shapes(shape_index)
            record = {
                "shape": shape,
                "container_kind": container_kind,
                "container_index": container_index,
                "shape_index": shape_index,
                "shape_name": shape.Name,
                "left": float(shape.Left),
                "top": float(shape.Top),
                "width": float(shape.Width),
                "height": float(shape.Height),
            }
            if int(shape.Type) == 13 and shape.Width > 0 and shape.Height > 0:
                record["picture_aspect"] = float(shape.Width) / float(shape.Height)
            try:
                if shape.HasTextFrame and shape.TextFrame2.HasText:
                    text_range = shape.TextFrame2.TextRange
                    if float(text_range.Font.Size) > 0:
                        record["font_size"] = float(text_range.Font.Size)
                    record["margins"] = (
                        float(shape.TextFrame2.MarginLeft),
                        float(shape.TextFrame2.MarginRight),
                        float(shape.TextFrame2.MarginTop),
                        float(shape.TextFrame2.MarginBottom),
                    )
            except Exception:
                pass
            try:
                if shape.Line.Visible:
                    record["line_weight"] = float(shape.Line.Weight)
            except Exception:
                pass
            records.append(record)

    presentation.PageSetup.SlideWidth = target_width
    presentation.PageSetup.SlideHeight = target_height

    for record in records:
        shape = record["shape"]
        target_left = record["left"] * sx
        target_top = record["top"] * sy
        target_box_width = record["width"] * sx
        target_box_height = record["height"] * sy
        if "picture_aspect" in record:
            continue
        else:
            new_left = target_left
            new_top = target_top
            new_width = target_box_width
            new_height = target_box_height
        try:
            shape.LockAspectRatio = 0
        except Exception:
            pass
        shape.Left = new_left
        shape.Top = new_top
        shape.Width = max(new_width, 0.1)
        shape.Height = max(new_height, 0.1)
        if "font_size" in record:
            try:
                shape.TextFrame2.TextRange.Font.Size = (
                    record["font_size"] * font_scale
                )
            except Exception:
                pass
        if "margins" in record:
            try:
                margins = record["margins"]
                shape.TextFrame2.MarginLeft = margins[0] * 4 / 3
                shape.TextFrame2.MarginRight = margins[1] * 4 / 3
                shape.TextFrame2.MarginTop = margins[2] * 4 / 3
                shape.TextFrame2.MarginBottom = margins[3] * 4 / 3
            except Exception:
                pass
        if "line_weight" in record:
            try:
                shape.Line.Weight = record["line_weight"] * 4 / 3
            except Exception:
                pass
    return records, sx, sy


def embed_figures_and_export(
    app,
    pptx_path: Path,
    figures: Iterable[FigureSpec],
    *,
    landscape: bool,
) -> dict:
    presentation = app.Presentations.Open(str(pptx_path.resolve()), WithWindow=False)
    slide = presentation.Slides(1)
    for index, spec in enumerate(figures, start=1):
        x, y, width, height = contain_box(spec)
        shape = slide.Shapes.AddPicture(
            str(spec.path.resolve()),
            False,
            True,
            x * 72,
            y * 72,
            width * 72,
            height * 72,
        )
        shape.Name = f"Embedded_{index:02d}_{spec.name}"
        shape.AlternativeText = str(spec.path.relative_to(ROOT.parent))
    reflow_records, sx, sy = reflow_for_print(
        presentation, landscape=landscape
    )
    # PowerPoint applies its page-size transform lazily on the first save.
    # Reopen before reasserting picture aspect ratios.
    presentation.Save()
    presentation.Close()
    presentation = app.Presentations.Open(
        str(pptx_path.resolve()), WithWindow=False
    )
    _restore_reopened_picture_aspects(presentation, reflow_records, sx, sy)
    presentation.Save()
    slide = presentation.Slides(1)

    pdf_path = pptx_path.with_suffix(".pdf")
    presentation.SaveAs(str(pdf_path.resolve()), 32)
    preview_path = pptx_path.with_name(f"{pptx_path.stem}-preview.png")
    if landscape:
        slide.Export(str(preview_path.resolve()), "PNG", 2400, 1800)
    else:
        slide.Export(str(preview_path.resolve()), "PNG", 1800, 2400)

    overflows = []
    for shape in slide.Shapes:
        try:
            if shape.HasTextFrame and shape.TextFrame2.HasText:
                bound_h = float(shape.TextFrame2.TextRange.BoundHeight)
                available_h = float(shape.Height)
                if bound_h > available_h + 2:
                    overflows.append(
                        {
                            "shape": shape.Name,
                            "bound_height_pt": round(bound_h, 1),
                            "box_height_pt": round(available_h, 1),
                        }
                    )
        except Exception:
            continue
    presentation.Close()
    return {
        "pptx": pptx_path.name,
        "pdf": pdf_path.name,
        "preview": preview_path.name,
        "text_overflows": overflows,
    }


def render_pdf_first_page(pdf_path: Path, png_path: Path) -> None:
    document = fitz.open(pdf_path)
    page = document[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
    pix.save(png_path)
    document.close()


def build_contact_sheet(previews: list[Path], out_path: Path) -> None:
    thumb_w, thumb_h = 560, 840
    margin, label_h = 24, 42
    canvas = Image.new(
        "RGB",
        (2 * thumb_w + 3 * margin, 4 * (thumb_h + label_h) + 5 * margin),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    for index, path in enumerate(previews):
        row, col = divmod(index, 2)
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((thumb_w, thumb_h))
            x = margin + col * (thumb_w + margin)
            y = margin + row * (thumb_h + label_h + margin)
            frame_x = x + (thumb_w - image.width) // 2
            frame_y = y + (thumb_h - image.height) // 2
            canvas.paste(image, (frame_x, frame_y))
            draw.rectangle(
                (frame_x, frame_y, frame_x + image.width, frame_y + image.height),
                outline=(210, 205, 216),
                width=2,
            )
            draw.text((x, y + thumb_h + 8), path.stem.replace("-preview", ""), fill=(50, 0, 110), font=font)
    canvas.save(out_path, quality=95)


def write_readme() -> None:
    text = """# A2-NLP poster concepts

Four visual design systems are provided. Each design has a separate Yoruba findings poster and
a separate Model Factory poster.

| Design | Format | Intent |
|---|---|---|
| 01 Classic | 36 × 48 in portrait | Familiar research-poster hierarchy and strong figure grid |
| 02 Narrative spine | 36 × 48 in portrait | Central evidence chain / workflow with supporting proof |
| 03 Evidence grid | 48 × 36 in landscape | Wide presentation format; fast left-to-right scan |
| 04 Editorial | 36 × 48 in portrait | Fewer, larger statements for distance readability |

Each poster is supplied as an editable `.pptx`, print-oriented `.pdf`, and `.png` preview. Existing
repository figures are embedded as SVG where available so charts remain vector in PowerPoint/PDF.
The supplied 2:3/3:2 templates are reflowed—without stretching figures or logos—to the project's
3:4 physical-board specification.

Typography note: the supplied Encode Sans, Open Sans and Uni Sans files do not collectively cover
all precomposed and combining Yoruba characters. The current English-language poster copy does not
set Yoruba-script examples. If examples are added, use a Yoruba-complete font such as Noto Sans and
preflight `Ẹ/ẹ`, `Ọ/ọ`, `Ṣ/ṣ` and combining tone marks before printing.

The copy follows reports 11–13 and the current claims audit. In particular:

- SIB-200 says **ahead**, not **beats**, because bootstrap intervals overlap by 0.004.
- MasakhaNER rates were not selected under the same held-out protocol.
- The unsupported matched-compute pretraining mean penalty is not presented as a finding.
- The variability result and the matched-compute downstream vocabulary swap are kept distinct.

`build_posters.py` regenerates all files. It requires `python-pptx`, `pywin32`, Pillow and PyMuPDF,
and uses installed Microsoft PowerPoint to embed SVGs and export PDFs/previews.
"""
    (OUTPUTS / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    for old in OUTPUTS.glob("design-*.pptx"):
        old.unlink()
    for old in OUTPUTS.glob("design-*.pdf"):
        old.unlink()
    for old in OUTPUTS.glob("design-*-preview.png"):
        old.unlink()

    jobs: list[tuple[Path, list[FigureSpec], bool]] = []
    for design_name, builder in BUILDERS:
        for kind, content in (("yoruba-findings", YORUBA), ("model-factory", FACTORY)):
            prs, figures = builder(content, "yoruba" if kind == "yoruba-findings" else "factory")
            path = OUTPUTS / f"{design_name}-{kind}.pptx"
            save_base_presentation(prs, path)
            landscape = prs.slide_width > prs.slide_height
            jobs.append((path, figures, landscape))

    app = win32com.client.DispatchEx("PowerPoint.Application")
    app.Visible = True
    reports = []
    try:
        for path, figures, landscape in jobs:
            reports.append(
                embed_figures_and_export(
                    app, path, figures, landscape=landscape
                )
            )
    finally:
        app.Quit()

    preview_paths = [OUTPUTS / report["preview"] for report in reports]
    build_contact_sheet(preview_paths, OUTPUTS / "poster-designs-contact-sheet.png")
    (OUTPUTS / "layout-validation.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8"
    )
    write_readme()

    stray = OUTPUTS / "_template_previews"
    if stray.exists():
        shutil.rmtree(stray)

    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
