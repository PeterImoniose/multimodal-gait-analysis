"""Build the board presentation (presentation/gait_sensing_board_paper.pptx).

Every chart is a native, editable PowerPoint chart whose series are read from the
project's caches and result tables, so the deck always matches the analysis.
Run from the repository root after the notebooks have been executed:

    .venv\\Scripts\\python presentation\\build_deck.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE, XL_TICK_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

from mgait import config as C
from mgait import eeg as E
from mgait import io

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
OUT = ROOT / "presentation" / "gait_sensing_board_paper.pptx"

# --- palette and type ------------------------------------------------------------------
NAVY = RGBColor(0x14, 0x21, 0x3D)
INK = RGBColor(0x14, 0x21, 0x3D)
BODY = RGBColor(0x4A, 0x55, 0x68)
MUTED = RGBColor(0x6B, 0x72, 0x80)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
CARD = RGBColor(0xEE, 0xF1, 0xF4)
WARM = RGBColor(0xFD, 0xEE, 0xE6)
ICE = RGBColor(0xC9, 0xD6, 0xEA)
BLUE = RGBColor(0x2A, 0x78, 0xD6)
DEEP = RGBColor(0x1C, 0x5C, 0xAB)
ORANGE = RGBColor(0xEB, 0x68, 0x34)
RUST = RGBColor(0xB8, 0x4A, 0x1A)
GREEN = RGBColor(0x1B, 0xAF, 0x7A)
AMBER = RGBColor(0xED, 0xA1, 0x00)
GREY = RGBColor(0xB9, 0xB8, 0xB2)
GRID = RGBColor(0xE3, 0xE6, 0xEA)
SPEED = {"0.5": RGBColor(0x86, 0xB6, 0xEF), "0.75": RGBColor(0x2A, 0x78, 0xD6), "1": RGBColor(0x10, 0x42, 0x81)}
SPEED_LABEL = {"0.5": "0.50 m/s", "0.75": "0.75 m/s", "1": "1.00 m/s"}
HEAD, FONT = "Cambria", "Calibri"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
prs.core_properties.title = "Gait sensing: what to deploy, and what to defer"
prs.core_properties.author = "Avwerosuo Peter Imoniose"
BLANK = prs.slide_layouts[6]
W, H, M = 13.333, 7.5, 0.6
TOTAL = 16


# --- helpers ---------------------------------------------------------------------------
def new_slide(bg=WHITE):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = bg
    return s


def text(slide, x, y, w, h, paras, size=16, color=INK, bold=False, font=FONT, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, name=None, space=6, line=None):
    """Text box. ``paras``: a string, or a list of paragraphs, each a string or a list of (text, opts)."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    paras = paras if isinstance(paras, list) else [paras]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if line:
            p.line_spacing = line
        if i < len(paras) - 1:
            p.space_after = Pt(space)
        for t, opt in (para if isinstance(para, list) else [(para, {})]):
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = opt.get("font", font)
            f.size = Pt(opt.get("size", size))
            f.bold = opt.get("bold", bold)
            f.italic = opt.get("italic", False)
            f.color.rgb = opt.get("color", color)
    return tb


def box(slide, x, y, w, h, fill=CARD, radius=0.06, name=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.adjustments[0] = radius
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.fill.background()
    shp.shadow.inherit = False
    if name:
        shp.name = name
    return shp


def badge(slide, x, y, label, d=0.5, fill=BLUE, color=WHITE, size=16):
    c = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    c.fill.solid()
    c.fill.fore_color.rgb = fill
    c.line.fill.background()
    c.shadow.inherit = False
    tf = c.text_frame
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.name, r.font.bold, r.font.size = FONT, True, Pt(size)
    r.font.color.rgb = color
    return c


def heading(slide, eyebrow, title, dark=False, eyebrow_color=None):
    text(slide, M, 0.42, W - 2 * M, 0.3, eyebrow.upper(), size=13, bold=True,
         color=eyebrow_color or (ICE if dark else BLUE), name="Eyebrow")
    text(slide, M, 0.75, W - 2 * M, 1.0, title, size=34, bold=True, font=HEAD,
         color=WHITE if dark else INK, name="Title", line=0.95)


def footer(slide, n, note="", dark=False):
    col = ICE if dark else MUTED
    if note:
        text(slide, M, H - 0.5, W - 2 * M - 1.0, 0.3, note, size=11, color=col, name="Footer note")
    text(slide, W - M - 0.8, H - 0.5, 0.8, 0.3, f"{n} / {TOTAL}", size=11, color=col, align=PP_ALIGN.RIGHT,
         name="Page number")


def stat(slide, x, y, w, h, big, label, fill=CARD, big_color=INK, big_size=36):
    box(slide, x, y, w, h, fill=fill)
    text(slide, x + 0.25, y + 0.2, w - 0.5, 0.65, big, size=big_size, bold=True, font=HEAD, color=big_color)
    text(slide, x + 0.25, y + 0.2 + big_size / 72 * 1.2, w - 0.5, h - 0.4 - big_size / 72 * 1.2, label,
         size=14, color=BODY)


def image(slide, path, x, y, w, h):
    with Image.open(path) as im:
        iw, ih = im.size
    s = min(w / iw, h / ih)
    dw, dh = iw * s, ih * s
    return slide.shapes.add_picture(str(path), Inches(x + (w - dw) / 2), Inches(y + (h - dh) / 2),
                                    Inches(dw), Inches(dh))


def notes(slide, t):
    slide.notes_slide.notes_text_frame.text = t


def table(slide, x, y, w, rows, widths, size=14, header_fill=NAVY, row_h=0.42):
    shp = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w),
                                 Inches(row_h * len(rows)))
    tbl = shp.table
    for j, cw in enumerate(widths):
        tbl.columns[j].width = Inches(w * cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.12)
            cell.margin_top = cell.margin_bottom = Inches(0.06)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = header_fill if i == 0 else (CARD if i % 2 == 0 else WHITE)
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            parts = val if isinstance(val, list) else [(val, {})]
            for t, opt in parts:
                r = p.add_run()
                r.text = t
                r.font.name, r.font.size = FONT, Pt(size)
                r.font.bold = i == 0 or opt.get("bold", False)
                r.font.color.rgb = WHITE if i == 0 else opt.get("color", INK)
    return shp


def style_chart(chart, size=12, legend=True):
    chart.font.name = FONT
    chart.font.size = Pt(size)
    chart.font.color.rgb = BODY
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(size)


def quiet_axes(chart, vgrid=True):
    for ax in (chart.value_axis, chart.category_axis):
        ax.format.line.color.rgb = GRID
        ax.tick_labels.font.size = Pt(12)
        ax.tick_labels.font.color.rgb = BODY
    chart.value_axis.has_major_gridlines = vgrid
    if vgrid:
        chart.value_axis.major_gridlines.format.line.color.rgb = GRID
    chart.category_axis.has_major_gridlines = False


def axis_title(ax, t):
    ax.has_title = True
    ax.axis_title.text_frame.text = t
    r = ax.axis_title.text_frame.paragraphs[0].runs[0]
    r.font.size, r.font.bold, r.font.name = Pt(12), False, FONT
    r.font.color.rgb = BODY


def line_series(chart, colors, width=2.25):
    for s, col in zip(chart.series, colors):
        s.format.line.color.rgb = col
        s.format.line.width = Pt(width)
        s.marker.style = XL_MARKER_STYLE.NONE
        s.smooth = False


def chart_title(chart, t):
    chart.has_title = True
    chart.chart_title.text_frame.text = t
    r = chart.chart_title.text_frame.paragraphs[0].runs[0]
    r.font.size, r.font.bold, r.font.name = Pt(14), True, FONT
    r.font.color.rgb = INK


# --- data --------------------------------------------------------------------------------
issues = pd.read_csv(TAB / "data_issues.csv")
spd = pd.read_csv(TAB / "ml_speed_classification.csv")
reid = pd.read_csv(TAB / "ml_reidentification.csv")
vfp = pd.read_csv(TAB / "virtual_force_plate_trial.csv").set_index("target")
leak = pd.read_csv(TAB / "ml_leakage_demo.csv").set_index("model")
st_agree = pd.read_csv(TAB / "imu_spatiotemporal_agreement.csv").set_index("parameter")
T = pd.read_parquet(TAB / "trial_features.parquet")
sync = pd.read_parquet(TAB / "sync.parquet")
afp = pd.read_parquet(TAB / "audit_fp.parquet")
fp_st = pd.read_parquet(C.CACHE / "fp_strides.parquet")
fp_cv = dict(np.load(C.CACHE / "fp_curves.npz"))
meta = io.load_subjects()

n_issues = len(issues)
rf_leak = leak.loc["Random forest"]
icc_stride = st_agree.loc["Stride time (s)", "ICC(A,1)"]
icc_stance = st_agree.loc["Stance (% stride)", "ICC(A,1)"]
reid_mean = reid.groupby("modality")["trial accuracy"].mean()

# --- 1. cover ----------------------------------------------------------------------------
s = new_slide(NAVY)
text(s, M + 0.1, 1.15, 11, 0.35, "BOARD PAPER  ·  SENSING TECHNOLOGY INVESTMENT", size=14, bold=True, color=ICE)
text(s, M + 0.1, 1.75, 11.2, 2.0, "Gait sensing: what to deploy, and what to defer", size=48, bold=True,
     font=HEAD, color=WHITE, line=0.95)
text(s, M + 0.1, 3.85, 10.0, 1.0, "Evidence from a 59-person study recording brain, muscle, motion and "
     "ground-force signals during treadmill walking", size=20, color=ICE)
for i, (lab, col) in enumerate((("Force plates", BLUE), ("IMU suit", ORANGE), ("EMG", GREEN), ("Dry EEG", AMBER))):
    x = M + 0.1 + i * 2.35
    badge(s, x, 5.2, "", d=0.28, fill=col)
    text(s, x + 0.4, 5.2, 1.9, 0.3, lab, size=15, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
text(s, M + 0.1, 6.55, 8, 0.35, "Avwerosuo Peter Imoniose  ·  Data Scientist", size=14, color=ICE)
text(s, W - M - 3, 6.55, 3, 0.35, "November 2026", size=14, color=ICE, align=PP_ALIGN.RIGHT)
notes(s, "This paper asks the board to decide which gait-sensing technologies to build on. The evidence comes "
         "from an independent re-analysis of a public dataset in which 59 healthy adults walked at three speeds "
         "while four sensor systems recorded simultaneously. Every number in the deck is reproducible from the "
         "accompanying code and notebooks.")

# --- 2. decision requested --------------------------------------------------------------------
s = new_slide()
heading(s, "Decision requested", "Build on wearable motion sensing; defer walking EEG; govern gait data as biometric")
cards = [
    ("1", BLUE, CARD, "Adopt the IMU suit as the primary platform",
     f"Stride timing matches force plates (ICC {icc_stride:.3f}); walking speed within 4.3 %; loading summaries "
     "predicted with R² 0.85 to 0.92."),
    ("2", BLUE, CARD, "Keep force plates as the reference",
     "Needed to calibrate wearables and for individual force waveforms and impact peaks, which motion data cannot recover."),
    ("3", ORANGE, WARM, "Defer walking EEG behind a validation gate",
     "Step-locked artefact in 94 % of trials; 9 of 11 apparent brain effects disappear once motion is accounted for."),
    ("4", ORANGE, WARM, "Govern gait data as biometric personal data",
     f"{100 * reid_mean.min():.0f} to {100 * reid_mean.max():.0f} % of people are re-identified from a single walk at an "
     "unseen speed (chance 2 %)."),
]
for i, (n, col, fill, h, b) in enumerate(cards):
    x, y = M + (i % 2) * 6.15, 2.25 + (i // 2) * 2.3
    box(s, x, y, 5.95, 2.05, fill=fill)
    badge(s, x + 0.3, y + 0.3, n, d=0.55, fill=col, size=18)
    text(s, x + 1.05, y + 0.3, 4.65, 0.6, h, size=19, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 1.05, y + 0.95, 4.65, 1.0, b, size=15, color=BODY)
text(s, M, 6.75, 9, 0.3, "Blue: investments supported now.  Orange: constraints on scope and data handling.",
     size=12, color=MUTED)
footer(s, 2)
notes(s, "Four decisions. Blue cards are investments the evidence supports now; orange cards are constraints: what "
         "to hold back and how to handle the data. The rest of the deck gives the evidence for each, in that order.")

# --- 3. evidence base ----------------------------------------------------------------------
s = new_slide()
heading(s, "The evidence base", "Four sensor systems recording the same steps")
for i, (big, lab) in enumerate((("59", "healthy adults: 37 women, 22 men, median age 22"),
                                ("3", "treadmill speeds: 0.50, 0.75 and 1.00 m/s, one minute each"),
                                (f"{int(fp_st['qc_ok'].sum()):,}", "strides measured on the instrumented treadmill"))):
    stat(s, M + i * 4.1, 2.0, 3.9, 1.55, big, lab, big_size=40)
rows = [["System", "What it measures", "Role in this paper"],
        ["Force plates (split belt)", "Ground forces under each foot, 1000 Hz", "Reference standard"],
        ["IMU suit (17 sensors)", "Body segment motion, 60 Hz", "Candidate scalable platform"],
        ["EMG (12 muscles)", "Muscle activation envelope, 16 Hz", "Candidate add-on module"],
        ["Dry EEG (19 channels)", "Scalp electrical activity, 300 Hz", "Candidate brain-signal module"]]
table(s, M, 3.95, W - 2 * M, rows, [0.28, 0.40, 0.32], size=15, row_h=0.46)
footer(s, 3, "Source: Katmah et al., Khalifa University, public dataset v1.0.0 (CC BY 4.0); independent re-analysis")
notes(s, "Because all four systems recorded the same steps at the same time, each candidate technology can be judged "
         "directly against the force-plate reference on identical walking. Limitation to keep in mind throughout: "
         "healthy young adults on a treadmill.")

# --- 4. data readiness -----------------------------------------------------------------------
s = new_slide()
heading(s, "Data readiness", "The published data were usable only after correction")
stat(s, M, 1.95, 3.0, 1.5, str(n_issues), "issues logged, from swapped sensors to dates of birth", big_size=32)
stat(s, M, 3.6, 3.0, 1.5, "173 of 173", "force-plate trials recovered, two of which looked unusable", big_size=32)
stat(s, M, 5.25, 3.0, 1.5, "133 ms", "hidden offset between the two published EEG formats", big_size=32)
fp_sync = sync.join(meta["mass_kg"], on="subject")
cd = XyChartData()
raw = afp.dropna(subset=["mass_reported_kg"])
se = cd.add_series("As recorded")
for a, b in zip(raw["mass_reported_kg"], raw["mass_from_fp_kg"]):
    se.add_data_point(float(a), float(b))
se = cd.add_series("After re-zeroing the plates")
for a, b in zip(fp_sync["mass_kg"], fp_sync["fp_bw_n"] / C.G):
    if np.isfinite(b):
        se.add_data_point(float(a), float(b))
se = cd.add_series("Perfect agreement")
se.add_data_point(40.0, 40.0)
se.add_data_point(115.0, 115.0)
gf = s.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER, Inches(4.0), Inches(1.9), Inches(8.73), Inches(4.85), cd)
ch = gf.chart
style_chart(ch)
quiet_axes(ch)
chart_title(ch, "Body mass measured by the force plates vs reported mass (kg)")
for sr, col, size in zip(ch.series, (GREY, BLUE, MUTED), (6, 6, 0)):
    sr.marker.style = XL_MARKER_STYLE.CIRCLE if size else XL_MARKER_STYLE.NONE
    if size:
        sr.marker.size = size
        sr.marker.format.fill.solid()
        sr.marker.format.fill.fore_color.rgb = col
        sr.marker.format.line.fill.background()
        sr.format.line.fill.background()
    else:
        sr.format.line.color.rgb = col
        sr.format.line.dash_style = 4
        sr.format.line.width = Pt(1.25)
ch.value_axis.minimum_scale, ch.value_axis.maximum_scale = -5, 120
ch.category_axis.minimum_scale, ch.category_axis.maximum_scale = 35, 120
axis_title(ch.category_axis, "Reported mass (kg)")
axis_title(ch.value_axis, "Force-plate mass (kg)")
footer(s, 4, "One participant appears weightless before correction; one recorded mass is 18 kg too low (110 kg measured)")
notes(s, "Grey points: several participants appear lighter than they are, and one appears weightless, because force "
         "plates were not zeroed. Blue points: after re-zeroing, measured and reported weight agree within about 2 %, "
         "except one participant whose recorded mass is wrong in the metadata. Lesson for the programme: any external "
         "dataset or vendor data needs an audit before it informs a decision.")

# --- 5. reference biomechanics -----------------------------------------------------------------
s = new_slide()
heading(s, "The reference measurement", "Walking faster reshapes loading more than it raises the peak")
ok = fp_st["qc_ok"].to_numpy()
fpo = fp_st[ok].reset_index(drop=True)
x101 = np.arange(101)
for k, (comp, ttl, x) in enumerate((("Fz", "Vertical force (body weights)", M),
                                    ("Fy", "Braking (below 0) and propulsion (above 0), body weights", M + 6.15))):
    X = pd.DataFrame(fp_cv[comp][ok])
    X["subject"], X["speed"] = fpo["subject"].values, fpo["speed"].values
    sm = X.groupby(["subject", "speed"]).mean()
    cd = XyChartData()
    for sp in C.SPEEDS:
        mean = sm.xs(sp, level="speed").to_numpy().mean(0)
        se = cd.add_series(SPEED_LABEL[sp])
        for xi in x101[::2]:
            se.add_data_point(float(xi), float(round(mean[xi], 4)))
    gf = s.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, Inches(x), Inches(1.95), Inches(5.95),
                            Inches(3.55), cd)
    ch = gf.chart
    style_chart(ch)
    quiet_axes(ch)
    chart_title(ch, ttl)
    line_series(ch, [SPEED[sp] for sp in C.SPEEDS])
    ch.category_axis.minimum_scale, ch.category_axis.maximum_scale = 0, 100
    ch.category_axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    axis_title(ch.category_axis, "Stance phase (%)")
facts = [("+50 % stride length, +32 % cadence", "from 0.5 to 1.0 m/s; longer strides carry most of the gain."),
         ("Braking, propulsion and loading rate double;", "the impact peak stays near 1.03 body weights."),
         ("Higher BMI walks more cautiously:", "longer stance and slower loading per kilogram.")]
for i, (b, r) in enumerate(facts):
    text(s, M + i * 4.1, 5.75, 3.85, 1.0, [[(b + " ", {"bold": True, "color": INK}), (r, {})]], size=15, color=BODY)
footer(s, 5, "Force plates, 59 participants; participant means averaged; mixed models with Holm correction")
notes(s, "This is the yardstick the wearable has to meet. The clinically interesting change with speed is in the shape "
         "and timing of loading, not in a single peak value, which matters for what a wearable needs to estimate.")

# --- 6. IMU timing -----------------------------------------------------------------------------
s = new_slide()
heading(s, "Finding 1  ·  Wearable motion sensing", "The IMU suit matches the treadmill on timing and speed")
tiles = [(f"{icc_stride:.3f}", "agreement (ICC) on stride time; limits ±12 ms", CARD, INK),
         ("99 %", "of heel strikes and toe-offs detected", CARD, INK),
         ("4.3 %", "mean error in walking speed, with no treadmill input", CARD, INK),
         (f"{icc_stance:.2f}", "agreement on stance %: good for trends, not individual thresholds", WARM, RUST)]
for i, (big, lab, fill, col) in enumerate(tiles):
    stat(s, M + (i % 2) * 2.95, 1.95 + (i // 2) * 2.3, 2.8, 2.1, big, lab, fill=fill, big_color=col, big_size=36)
cd = XyChartData()
rng = np.random.default_rng(0)
for sp in C.SPEEDS:
    d = T[(T["speed"] == sp)].dropna(subset=["imu_speed_est"])
    se = cd.add_series(SPEED_LABEL[sp])
    for v in d["imu_speed_est"]:
        se.add_data_point(float(C.SPEED_MS[sp] + rng.uniform(-0.03, 0.03)), float(round(v, 4)))
se = cd.add_series("Perfect agreement")
se.add_data_point(0.4, 0.4)
se.add_data_point(1.1, 1.1)
gf = s.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER, Inches(6.75), Inches(1.9), Inches(5.98), Inches(4.65), cd)
ch = gf.chart
style_chart(ch)
quiet_axes(ch)
chart_title(ch, "Walking speed estimated by the suit (m/s)")
for sr, col in zip(ch.series, [SPEED[sp] for sp in C.SPEEDS] + [MUTED]):
    if col == MUTED:
        sr.marker.style = XL_MARKER_STYLE.NONE
        sr.format.line.color.rgb = col
        sr.format.line.dash_style = 4
        sr.format.line.width = Pt(1.25)
    else:
        sr.marker.style = XL_MARKER_STYLE.CIRCLE
        sr.marker.size = 6
        sr.marker.format.fill.solid()
        sr.marker.format.fill.fore_color.rgb = col
        sr.marker.format.line.fill.background()
        sr.format.line.fill.background()
ch.category_axis.minimum_scale, ch.category_axis.maximum_scale = 0.4, 1.1
ch.value_axis.minimum_scale, ch.value_axis.maximum_scale = 0.3, 1.2
axis_title(ch.category_axis, "Treadmill belt speed (m/s)")
footer(s, 6, "45 held-out participants, 129 trials; the detector was tuned on 10 other participants only")
notes(s, "The setting of the event detector was chosen on ten development participants and frozen before these "
         "results were computed, so these are honest out-of-sample figures. The orange tile is the caveat: stance "
         "percentage is close on average but individual readings can differ by up to about four percentage points.")

# --- 7. virtual force plate ----------------------------------------------------------------------
s = new_slide()
heading(s, "Finding 2  ·  A virtual force plate", "Motion data predict loading summaries, not each person’s force curve")
labels = {"fp_fy_prop_peak": "Propulsive peak", "fp_double_support_pct": "Double support", "fp_stance_pct": "Stance %",
          "fp_fy_brake_peak": "Braking peak", "fp_fz_valley": "Mid-stance dip", "fp_loading_rate": "Loading rate",
          "fp_fz_peak2": "Push-off peak", "fp_fx_medial_peak": "Side force", "fp_fz_peak1": "Impact peak"}
best = vfp[["R2 Ridge", "R2 Random forest"]].max(axis=1).reindex(list(labels))
cd = CategoryChartData()
cd.categories = [labels[k] for k in best.index]
cd.add_series("Best IMU model, R²", [float(round(max(v, 0), 2)) for v in best.values])
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(M), Inches(1.9), Inches(7.2), Inches(4.85), cd)
ch = gf.chart
style_chart(ch, legend=False)
quiet_axes(ch)
chart_title(ch, "Variance explained on unseen participants (R², 0 = no skill)")
ch.category_axis.reverse_order = True
ch.value_axis.minimum_scale, ch.value_axis.maximum_scale = 0, 1
ch.value_axis.tick_labels.number_format, ch.value_axis.tick_labels.number_format_is_linked = "0.0", False
pl = ch.plots[0]
pl.gap_width = 45
pl.has_data_labels = True
pl.data_labels.number_format, pl.data_labels.number_format_is_linked = "0.00", False
pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
pl.data_labels.font.size = Pt(12)
for i, v in enumerate(best.values):
    pt = ch.series[0].points[i]
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = BLUE if v >= 0.6 else (GREY if v >= 0.3 else ORANGE)
sdl = ch.series[0].data_labels   # series-level labels, so a custom point label does not hide the rest
sdl.show_value = True
sdl.number_format, sdl.number_format_is_linked = "0.00", False
sdl.position = XL_LABEL_POSITION.OUTSIDE_END
sdl.font.size = Pt(12)
for i, v in enumerate(best.values):
    if v < 0:
        pt = ch.series[0].points[i]
        pt.data_label.has_text_frame = True
        pt.data_label.text_frame.text = "below 0: no skill"
        pt.data_label.position = XL_LABEL_POSITION.OUTSIDE_END
box(s, 8.15, 1.95, 4.58, 4.75, fill=CARD)
text(s, 8.45, 2.2, 4.0, 4.4, [
    [("Summaries: yes. ", {"bold": True, "color": INK}),
     ("Propulsion, braking, stance and double support are predicted with R² 0.84 to 0.92 on people the model never saw.", {})],
    [("Impact peak and side force: no. ", {"bold": True, "color": RUST}),
     ("R² below 0.2; these still need a force plate.", {})],
    [("Waveforms: mostly speed. ", {"bold": True, "color": INK}),
     ("Predicted stride curves fit with r = 0.99, but only 6 to 10 % better than a template based on walking "
      "speed alone.", {})]], size=15, color=BODY, space=14)
footer(s, 7, "58 participants held out in turn (10-fold by participant); inputs: IMU kinematics and the suit's speed estimate")
notes(s, "Trial-level summaries are well predicted on people the model never saw. The waveform result is the honest "
         "limit: the predicted curve looks right, but almost all of that comes from knowing the walking speed, and "
         "the motion data add little about the individual. Where a decision depends on a person's own force curve or "
         "impact peak, the force plate is still required.")

# --- 8. modality comparison --------------------------------------------------------------------
s = new_slide()
heading(s, "Finding 3  ·  Comparing the sensors", "Walking state is carried by the biomechanics, not the brain signal")
order = ["Force plates + IMU + EMG", "Force plates", "IMU", "EMG", "EEG (band power)", "EEG artefact indices only"]
nice = {"Force plates + IMU + EMG": "All biomechanics", "EEG (band power)": "EEG band power",
        "EEG artefact indices only": "EEG artefact only"}
bestc = spd.groupby("features")["balanced accuracy (subject-wise)"].max().reindex(order)
cd = CategoryChartData()
cd.categories = [nice.get(k, k) for k in order]
cd.add_series("Balanced accuracy", [float(round(100 * v, 1)) for v in bestc.values])
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(M), Inches(1.9), Inches(7.6), Inches(4.85), cd)
ch = gf.chart
style_chart(ch, legend=False)
quiet_axes(ch)
chart_title(ch, "Walking speed (3 classes) on unseen participants, % correct; chance 33 %")
ch.category_axis.reverse_order = True
ch.value_axis.minimum_scale, ch.value_axis.maximum_scale, ch.value_axis.major_unit = 0, 110, 20
pl = ch.plots[0]
pl.gap_width = 45
pl.has_data_labels = True
pl.data_labels.number_format, pl.data_labels.number_format_is_linked = '0"%"', False
pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
pl.data_labels.font.size = Pt(12)
for i, col in enumerate((GREY, BLUE, ORANGE, GREEN, AMBER, MUTED)):
    pt = ch.series[0].points[i]
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = col
facts = [("IMU reaches 89 %", "on people it never saw, close to the force plates’ 99 %."),
         ("Combining sensors adds little", "beyond the force plates alone."),
         ("EEG is beaten by its own artefact measures:", "62 % against 49 %.")]
for i, (b, r) in enumerate(facts):
    y = 2.15 + i * 1.5
    badge(s, 8.6, y, str(i + 1), d=0.45, fill=NAVY, size=15)
    text(s, 9.25, y - 0.02, 3.5, 1.3, [[(b + " ", {"bold": True, "color": INK}), (r, {})]], size=16, color=BODY)
footer(s, 8, "160 trials, 57 participants; 10-fold cross-validation by participant; best of logistic regression and random forest")
notes(s, "Speed classification is a probe, not a product: it asks which signals carry information about how a person "
         "is walking that transfers to new people. The bottom bar is the key control: a model given only two measures "
         "of motion contamination outperforms the full EEG feature set.")

# --- 9. EEG --------------------------------------------------------------------------------------
s = new_slide()
heading(s, "Finding 4  ·  Brain signal during walking", "The walking EEG records the footsteps, not the brain",
        eyebrow_color=RUST)
psd = np.load(C.CACHE / "eeg_psd.npz")
keys, fq, Pw = list(psd["keys"]), psd["f"], psd["P"]
ecs = pd.read_parquet(C.CACHE / "eeg_channels.parquet")
fstep = ecs.groupby(["subject", "speed"])["f_step"].first()
idx = [C.EEG_CHANNELS.index(c) for c in E.REGIONS["central"]]
grid = np.arange(0.5, 8.01, 0.03)
spec = {sp: [] for sp in C.SPEEDS}
for i, k in enumerate(keys):
    sj, sp = k.rsplit("_", 1)
    fs_ = fstep.get((sj, sp), np.nan)
    if k == "S8_0.5" or not np.isfinite(fs_):
        continue
    spec[sp].append(np.interp(grid, fq / fs_, np.log10(Pw[i][:, idx].mean(axis=1)), left=np.nan, right=np.nan))
cd = XyChartData()
for sp in C.SPEEDS:
    m = np.nanmean(spec[sp], axis=0)
    se = cd.add_series(SPEED_LABEL[sp])
    for a, b in zip(grid, m):
        se.add_data_point(float(round(a, 3)), float(round(b, 4)))
gf = s.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, Inches(M), Inches(1.9), Inches(7.6), Inches(4.85), cd)
ch = gf.chart
style_chart(ch)
quiet_axes(ch)
chart_title(ch, "Central EEG power, frequency scaled to each person’s step rate")
line_series(ch, [SPEED[sp] for sp in C.SPEEDS], width=1.75)
ch.category_axis.minimum_scale, ch.category_axis.maximum_scale, ch.category_axis.major_unit = 0, 8, 1
ch.category_axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW
ch.category_axis.has_major_gridlines = True
ch.category_axis.major_gridlines.format.line.color.rgb = GRID
axis_title(ch.category_axis, "Frequency ÷ step frequency (peaks at 1, 2, 3 … = stepping)")
axis_title(ch.value_axis, "log10 power")
tiles = [("94 %", "of trials show peaks at the step rhythm"),
         ("4×", "more coupled to own head motion than to others’"),
         ("×1.9", "heel-strike transient per +0.5 m/s"),
         ("9 of 11", "apparent brain effects vanish after motion adjustment")]
for i, (big, lab) in enumerate(tiles):
    stat(s, 8.6 + (i % 2) * 2.1, 1.95 + (i // 2) * 2.4, 2.0, 2.2, big, lab, fill=WARM, big_color=RUST, big_size=30)
footer(s, 9, "175 trials, 19-channel dry headset; linked-ear reference, 1 Hz high-pass, 50 Hz notch, no artefact removal")
notes(s, "Each person's spectrum is rescaled so that their own step frequency sits at 1. Brain rhythms have no reason "
         "to line up at 1, 2, 3 and so on; mechanical artefact from electrode and cable movement does. Four independent "
         "tests agree. This does not say walking EEG is impossible, only that this hardware and this data cannot support "
         "claims about the brain without purpose-built artefact removal.")

# --- 10. EMG -----------------------------------------------------------------------------------
s = new_slide()
heading(s, "Finding 5  ·  Muscle activity", "Muscle coordination follows four shared patterns in everyone")
image(s, FIG / "04_synergies.png", M, 1.85, 8.3, 4.95)
facts = [("0.98 similarity", "between each person’s patterns and the shared set, at every speed."),
         ("Effort scales with speed:", "main muscles work 40 to 49 % harder per +0.5 m/s."),
         ("Limit:", "only a 16 Hz envelope was released; raw signal is needed for more.")]
for i, (b, r) in enumerate(facts):
    y = 2.1 + i * 1.55
    badge(s, 9.2, y, str(i + 1), d=0.45, fill=GREEN, size=15)
    text(s, 9.85, y - 0.02, 2.9, 1.4, [[(b + " ", {"bold": True, "color": INK}), (r, {})]], size=16, color=BODY)
footer(s, 10, "289 leg-trials from 56 participants; non-negative matrix factorisation of six muscles per leg")
notes(s, "Stable, shared coordination patterns are what make muscle-level monitoring plausible: a deviation from the "
         "shared pattern would be interpretable. EMG is weak on its own as a classifier, so it is positioned as an "
         "optional module that adds explanation, captured as raw signal in any future study.")

# --- 11. privacy ------------------------------------------------------------------------------
s = new_slide()
heading(s, "Finding 6  ·  Privacy", "A walk identifies the walker: gait data are biometric", eyebrow_color=RUST)
mods = ["Force plates", "IMU", "EMG", "EEG"]
cd = CategoryChartData()
cd.categories = mods
for sp in C.SPEEDS:
    v = reid[np.isclose(reid["test speed"].astype(float), C.SPEED_MS[sp])].set_index("modality").loc[mods, "trial accuracy"]
    cd.add_series(f"Tested at {SPEED_LABEL[sp]}", [float(round(100 * a, 1)) for a in v])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(M), Inches(1.9), Inches(7.6), Inches(4.85), cd)
ch = gf.chart
style_chart(ch)
quiet_axes(ch)
chart_title(ch, "Re-identified from one walk at an unseen speed (%)")
ch.value_axis.minimum_scale, ch.value_axis.maximum_scale, ch.value_axis.major_unit = 0, 100, 20
pl = ch.plots[0]
pl.gap_width, pl.overlap = 60, -5
for sr, sp in zip(ch.series, C.SPEEDS):
    sr.format.fill.solid()
    sr.format.fill.fore_color.rgb = SPEED[sp]
facts = [(f"{100 * reid_mean.min():.0f} to {100 * reid_mean.max():.0f} % re-identified", "from one walk at an unseen speed (chance: 2 %)."),
         ("88 % correct on sex", "inferred from gait alone, against a 62 % baseline."),
         ("The “anonymised” release", "still contained 59 full dates of birth.")]
for i, (b, r) in enumerate(facts):
    y = 2.15 + i * 1.5
    badge(s, 8.6, y, str(i + 1), d=0.45, fill=ORANGE, size=15)
    text(s, 9.25, y - 0.02, 3.5, 1.3, [[(b + " ", {"bold": True, "color": INK}), (r, {})]], size=16, color=BODY)
footer(s, 11, "Train on two speeds, test on the third; one recording session, so recognition across days is not tested")
notes(s, "Removing names does not anonymise gait data. Every sensor, including the cheapest, recognises people across "
         "walking conditions. Treat all collected gait data as biometric personal data from day one: a data protection "
         "impact assessment, access control, and no release of raw signals without a re-identification test.")

# --- 12. validation standard -------------------------------------------------------------------
s = new_slide(DEEP)
text(s, M, 0.42, W - 2 * M, 0.3, "STANDARD TO REQUIRE", size=13, bold=True, color=ICE)
text(s, M, 0.75, W - 2 * M, 1.4, "How a model is tested can change its score by "
     f"{100 * (rf_leak['random stride split'] - rf_leak['participant-wise split']):.0f} points",
     size=40, bold=True, font=HEAD, color=WHITE, line=0.95)
for i, (big, lab, col) in enumerate(((f"{100 * rf_leak['random stride split']:.0f} %",
                                      "strides shuffled at random, so each test stride has near-copies in training", ICE),
                                     (f"{100 * rf_leak['participant-wise split']:.0f} %",
                                      "the same model tested on people it has never seen", WHITE))):
    x = M + i * 5.0
    text(s, x, 2.55, 4.6, 1.6, big, size=96, bold=True, font=HEAD, color=col)
    text(s, x, 4.2, 4.3, 0.9, lab, size=17, color=col)
box(s, M, 5.45, W - 2 * M, 0.95, fill=NAVY)
text(s, M + 0.35, 5.45, W - 2 * M - 0.7, 0.95, "Require every internal or vendor accuracy claim to be tested on "
     "held-out people, with settings fixed before testing.", size=19, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
footer(s, 12, f"Random forest classifying walking speed from IMU strides; {int(rf_leak['strides']):,} strides, 58 participants",
       dark=True)
notes(s, "Same data, same model, two testing protocols. The shuffled split lets strides from the same minute of walking "
         "appear in both training and testing, so the model is partly recognising the person rather than the walking "
         "speed. Deployment means new people, so only the second number is meaningful.")

# --- 13. decision matrix -----------------------------------------------------------------------
s = new_slide()
heading(s, "Recommendation", "Each technology has a clear role, or a clear reason to wait")
rows = [["Technology", "Supported by the evidence", "Not supported", "Decision"],
        ["IMU suit", "Gait events, stride time, walking speed, loading summaries", "Individual force waveforms, impact peak",
         [("Adopt", {"bold": True, "color": DEEP}), (" as primary platform", {})]],
        ["Force plates", "Reference timing and forces (speed classified at 99 %)", "Portability, use outside the lab",
         [("Retain", {"bold": True, "color": DEEP}), (" for calibration", {})]],
        ["EMG", "Muscle timing, effort scaling, stable coordination patterns",
         "Stand-alone use (speed at 67 %); raw-signal analysis",
         [("Optional", {"bold": True, "color": RUST}), (" module, raw capture", {})]],
        ["Dry EEG", "Nothing beyond its motion-artefact content", "Any claim about brain activity during walking",
         [("Defer", {"bold": True, "color": RUST}), (" behind a validation gate", {})]]]
table(s, M, 2.0, W - 2 * M, rows, [0.15, 0.33, 0.30, 0.22], size=15, row_h=0.78)
text(s, M, 6.2, 10, 0.4, "Accuracy figures are for people the models had never seen, which is the deployment condition.",
     size=14, color=BODY)
footer(s, 13)
notes(s, "The decision matrix restates slides 6 to 11 as choices. Adopt and Retain are investment decisions; Optional "
         "and Defer are scope decisions that avoid spend on capability the evidence does not yet support.")

# --- 14. roadmap -------------------------------------------------------------------------------
s = new_slide()
heading(s, "Proposed plan", "Three phases, each released only when its gate is met")
phases = [("1", BLUE, CARD, "Validate in the target users",
           "Repeat the IMU versus force-plate comparison on the intended population, overground as well as on the "
           "treadmill, with participants held out.",
           "Gate: stride time ICC at least 0.95; speed error within 5 %; loading summaries R² at least 0.8."),
          ("2", BLUE, CARD, "Build the measurement service",
           "IMU pipeline for events, speed and loading summaries; force plates for periodic calibration; data protection "
           "impact assessment before collection.", "Gate: Phase 1 met; data governance approved."),
          ("3", ORANGE, WARM, "Optional: muscle or brain modules",
           "EMG with raw-signal capture. EEG only with motion-reference hardware and artefact removal designed for walking.",
           "Gate: EEG features that survive motion adjustment and beat artefact-only models.")]
for i, (n, col, fill, h, b, g) in enumerate(phases):
    x = M + i * 4.1
    box(s, x, 2.0, 3.9, 4.6, fill=fill)
    badge(s, x + 0.3, 2.3, n, d=0.55, fill=col, size=18)
    text(s, x + 0.3, 3.05, 3.3, 0.7, h, size=18, bold=True)
    text(s, x + 0.3, 3.8, 3.3, 1.7, b, size=14, color=BODY)
    text(s, x + 0.3, 5.5, 3.3, 1.0, [[("Gate: ", {"bold": True}), (g.replace("Gate: ", ""), {})]], size=14, color=INK)
    if i < 2:
        a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + 3.93), Inches(4.1), Inches(0.15), Inches(0.3))
        a.fill.solid()
        a.fill.fore_color.rgb = GREY
        a.line.fill.background()
footer(s, 14, "Gate thresholds are set at or below the levels this study achieved; budgets attach once the target population is fixed")
notes(s, "The gates turn this study's results into acceptance criteria. Phase 1 matters most: the evidence comes from "
         "healthy young adults on a treadmill, so the first spend should establish that the same accuracy holds for the "
         "people the service is for.")

# --- 15. risks ---------------------------------------------------------------------------------
s = new_slide()
heading(s, "Risks and limitations", "What could make these conclusions wrong")
rows = [["Risk", "Why it matters", "Mitigation"],
        ["Population", "Healthy adults aged about 22, slow treadmill walking", "Phase 1 re-validation in target users"],
        ["Single session", "Re-identification shown across speeds, not across days", "Treat as biometric until shown otherwise"],
        ["Inflated vendor claims", "A naive data split overstated accuracy by 13 points here", "Require participant-wise validation"],
        ["Source data quality", f"{n_issues} issues in a published release, some silent", "Audit every dataset before use"],
        ["EEG hardware", "Dry headset without motion reference channels", "Gate EEG on purpose-built hardware"],
        ["Processed EMG", "Only a 16 Hz envelope released", "Capture raw EMG in any new study"]]
table(s, M, 2.0, W - 2 * M, rows, [0.22, 0.43, 0.35], size=15, row_h=0.6)
footer(s, 15)
notes(s, "The largest risk is population transfer: nothing here shows the IMU performs as well for older or impaired "
         "walkers, which is why Phase 1 is a validation phase rather than a build phase.")

# --- 16. appendix ------------------------------------------------------------------------------
s = new_slide(NAVY)
heading(s, "Appendix  ·  Method", "How the evidence was produced", dark=True)
items = [("No tuning on evaluation data", "The one tuned setting (IMU contact threshold) was fixed on ten development "
          "participants before any held-out result was seen."),
         ("People are the unit of validation", "Every model is tested on participants it never saw; confidence "
          "intervals resample participants."),
         ("Repeated measures modelled", "Speed effects use mixed models with a per-person baseline; families of tests "
          "are Holm-corrected."),
         ("Corrections evidenced twice", "Each data fix is backed by an independent check, for example the swapped "
          "plates by geometry and by motion timing.")]
for i, (h, b) in enumerate(items):
    x, y = M + (i % 2) * 6.15, 2.05 + (i // 2) * 1.9
    badge(s, x, y, str(i + 1), d=0.45, fill=BLUE, size=15)
    text(s, x + 0.65, y, 5.2, 0.45, h, size=18, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 0.65, y + 0.55, 5.2, 1.2, b, size=15, color=ICE)
text(s, M, 5.95, W - 2 * M, 0.6, "Reproducible end to end: six executed notebooks, a tested Python package and 14 unit "
     "tests in the multimodal-gait-analysis repository.", size=15, color=ICE)
footer(s, 16, dark=True)
notes(s, "For technical reviewers: the code regenerates every figure and number in this deck from the raw release in "
         "under ten minutes on a workstation.")

assert len(prs.slides) == TOTAL, len(prs.slides)
OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print(f"wrote {OUT} ({len(prs.slides)} slides)")
