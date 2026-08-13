# A2-NLP poster concepts

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
