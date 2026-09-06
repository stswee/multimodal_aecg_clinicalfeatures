#!/usr/bin/env python3
"""Build manuscript-style ECG/text saliency examples from numeric inputs.

This is intentionally ad hoc for the two selected examples:
  - SCD patient 0271
  - PFD patient 0285

Unlike a montage script, this does not paste the existing patient-example PNGs
into a canvas. It redraws the figure from raw ECG, saved context CSVs, token
saliency CSVs, and summary metadata.
"""

from __future__ import annotations

import json
import math
import re
import base64
import html
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent
MUSIC = ROOT.parent / "music"
EXPLAIN_ROOT = MUSIC / "multimodal_nested_4year_v4_detailed" / "explain_multimodal_ecg_text" / "val_fold_1"
RAW_ROOT = MUSIC / "preprocessed_HRV"

FIGSIZE = (14.5, 9.5)
DPI = 300
FS = 200
WINDOW_SEC = 30

FEATURES = [
    ("hr_mean", "Mean\nHR"),
    ("HRV_SDNN", "SDNN"),
    ("HRV_RMSSD", "RMSSD"),
    ("HRV_pNN50", "pNN50"),
    ("longest_rr_pause", "Longest\nRR Pause"),
    ("pvc_burden_pct", "PVC\nBurden (%)"),
]


@dataclass(frozen=True)
class PatientFigure:
    label: str
    pid: str
    endpoint: str
    source_dir: Path


PATIENTS = [
    PatientFigure("SCD_0271_nested_selected", "0271", "SCD", EXPLAIN_ROOT / "SCD" / "selected_fusion" / "0271"),
    PatientFigure("PFD_0285_nested_selected", "0285", "PFD", EXPLAIN_ROOT / "PFD" / "selected_fusion" / "0285"),
]


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 15,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "svg.fonttype": "none",
        }
    )


def raw_ecg(pid: str) -> np.ndarray:
    path = RAW_ROOT / f"{pid}_preprocessed.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, mmap_mode="r")["signal"]


def score_norm(values: np.ndarray):
    values = np.asarray(values, dtype=float)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmin < 0 < vmax:
        return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    if vmax <= 0:
        return Normalize(vmin=vmin, vmax=0.0 if vmin < 0 else 1.0)
    return Normalize(vmin=0.0, vmax=vmax)


def extract_attention_raster(svg_path: Path) -> np.ndarray:
    """Extract the full-duration Matplotlib imshow raster from the SVG."""

    svg = svg_path.read_text(errors="ignore")
    match = re.search(r'<image xlink:href="data:image/png;base64,\s*([^"]+)"', svg)
    if not match:
        raise ValueError(f"No embedded attention raster found in {svg_path}")
    payload = re.sub(r"\s+", "", match.group(1))
    return np.asarray(Image.open(BytesIO(base64.b64decode(payload))).convert("RGB"))


def attention_colorbar_max(svg_path: Path) -> float:
    svg = svg_path.read_text(errors="ignore")
    labels = [float(value) for value in re.findall(r"<!--\s*([0-9]+(?:\.[0-9]+)?)\s*-->", svg)]
    plausible = [value for value in labels if 0.0 <= value <= 0.01]
    return max(plausible) if plausible else 1.0


def draw_panel_a(ax, cax, source_dir: Path) -> None:
    raster = extract_attention_raster(source_dir / "ecg_attention.svg")
    ax.imshow(raster, aspect="auto", extent=[0, 24, 0, 1])
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_xlabel("Time (Hour)")
    ax.set_title("ECG Attention Map", pad=8)
    cbar_max = attention_colorbar_max(source_dir / "ecg_attention.svg")
    sm = mpl.cm.ScalarMappable(cmap="Reds", norm=Normalize(vmin=0.0, vmax=cbar_max))
    sm.set_array([])
    cb = plt.colorbar(sm, cax=cax)
    cb.set_ticks(np.linspace(0.0, cbar_max, 5))
    cb.set_label("Attention", fontsize=9)


def token_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["text_span"]).copy()
    df["char_start"] = df["char_start"].astype(int)
    df["char_end"] = df["char_end"].astype(int)
    return df


def text_and_char_scores(df: pd.DataFrame) -> tuple[str, np.ndarray]:
    max_end = int(df["char_end"].max())
    chars = [" "] * max_end
    scores = np.zeros(max_end, dtype=float)
    counts = np.zeros(max_end, dtype=float)
    for _, row in df.iterrows():
        start = int(row["char_start"])
        end = int(row["char_end"])
        span = str(row["text_span"])
        for offset, char in enumerate(span[: max(0, end - start)]):
            if start + offset < max_end:
                chars[start + offset] = char
        scores[start:end] += float(row["saliency"])
        counts[start:end] += 1
    mask = counts > 0
    scores[mask] /= counts[mask]
    return "".join(chars), scores


def wrap_text(text: str, scores: np.ndarray, max_chars: int = 62) -> list[tuple[str, np.ndarray]]:
    lines: list[tuple[str, np.ndarray]] = []
    pos = 0
    while pos < len(text):
        if text[pos] == "\n":
            lines.append(("", np.array([], dtype=float)))
            pos += 1
            continue
        end = min(len(text), pos + max_chars)
        newline = text.find("\n", pos, end + 1)
        if newline != -1:
            end = newline
        elif end < len(text):
            space = text.rfind(" ", pos, end)
            if space > pos:
                end = space
        line = text[pos:end].rstrip()
        lines.append((line, scores[pos : pos + len(line)]))
        pos = end
        while pos < len(text) and text[pos] == " ":
            pos += 1
        if pos < len(text) and text[pos] == "\n":
            pos += 1
    return lines


def highlight_summary_spans(text: str, summary: dict) -> np.ndarray:
    mask = np.zeros(len(text), dtype=bool)
    search_from = 0
    for span in summary.get("top_text_spans", []):
        phrase = str(span.get("text", "")).strip()
        if not phrase:
            continue
        idx = text.find(phrase, search_from)
        if idx < 0:
            idx = text.find(phrase)
        if idx >= 0:
            mask[idx : idx + len(phrase)] = True
            search_from = idx + len(phrase)
    return mask


def parse_html_segments(html_path: Path, endpoint: str) -> list[tuple[str, float | None]]:
    raw = html_path.read_text(errors="ignore")
    div_match = re.search(r"<div[^>]*>(.*?)</div>", raw, re.S)
    body = div_match.group(1) if div_match else raw
    mark_re = re.compile(
        r"<mark\s+style='background:\s*rgba\(255,\s*214,\s*10,\s*([0-9.]+)\);[^']*'>(.*?)</mark>",
        re.S,
    )
    segments: list[tuple[str, float | None]] = []
    pos = 0
    for match in mark_re.finditer(body):
        pre = re.sub("<[^>]+>", "", body[pos : match.start()])
        if pre:
            segments.append((html.unescape(pre), None))
        marked = re.sub("<[^>]+>", "", match.group(2))
        if marked:
            segments.append((html.unescape(marked), float(match.group(1))))
        pos = match.end()
    tail = re.sub("<[^>]+>", "", body[pos:])
    if tail:
        segments.append((html.unescape(tail), None))

    plain = "".join(text for text, _ in segments)
    start = plain.find(endpoint + "_RISK")
    if start <= 0:
        return segments

    trimmed: list[tuple[str, float | None]] = []
    seen = 0
    for text, alpha in segments:
        end = seen + len(text)
        if end <= start:
            seen = end
            continue
        if seen < start:
            text = text[start - seen :]
        trimmed.append((text, alpha))
        seen = end
    return trimmed


def draw_panel_b(ax, html_path: Path, endpoint: str) -> None:
    segments = parse_html_segments(html_path, endpoint)
    chars: list[tuple[str, float | None]] = []
    for text, alpha in segments:
        chars.extend((char, alpha) for char in text)

    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    max_chars = 61
    lines: list[list[tuple[str, float | None]]] = []
    current: list[tuple[str, float | None]] = []
    for char, alpha in chars:
        if char == "\n":
            lines.append(current)
            current = []
            continue
        if len(current) >= max_chars:
            split = max((idx for idx, (c, _) in enumerate(current) if c == " "), default=len(current) - 1)
            lines.append(current[:split])
            current = current[split + 1 :]
        current.append((char, alpha))
    if current:
        lines.append(current)

    x0 = 0.0
    y = 0.98
    char_w = 0.0147
    line_h = 0.046
    for line in lines:
        if y < 0.02:
            break
        idx = 0
        while idx < len(line):
            alpha = line[idx][1]
            end = idx
            while end < len(line) and line[end][1] == alpha:
                end += 1
            if alpha is not None and any(not c.isspace() for c, _ in line[idx:end]):
                ax.add_patch(
                    Rectangle(
                        (x0 + idx * char_w - 0.0015, y - 0.030),
                        max((end - idx) * char_w, 0.006),
                        0.038,
                        facecolor=(1.0, 0.84, 0.04, min(max(alpha, 0.20), 0.85)),
                        edgecolor="none",
                        zorder=0,
                    )
                )
            idx = end
        ax.text(
            x0,
            y,
            "".join(char for char, _ in line),
            ha="left",
            va="top",
            fontsize=13.2,
            family="DejaVu Sans Mono",
            color="black",
            zorder=1,
        )
        y -= line_h


def draw_focal_trace(ax, raw: np.ndarray, context: pd.DataFrame, score_col: str, cax) -> None:
    cmap = plt.get_cmap("seismic")
    norm = score_norm(context[score_col].to_numpy(dtype=float))
    focal = context.loc[context["is_focal"] == 1].iloc[0]
    start = int(focal["start_idx"])
    snippet = np.asarray(raw[start : start + FS * WINDOW_SEC], dtype=float)
    t = np.arange(snippet.size) / FS
    pts = np.column_stack([t, snippet]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, colors=[cmap(norm(float(focal[score_col])))], linewidths=0.75)
    ax.add_collection(lc)
    ax.set_xlim(0, WINDOW_SEC)
    ymin, ymax = float(np.nanmin(snippet)), float(np.nanmax(snippet))
    pad = 0.05 * (ymax - ymin) if ymax > ymin else 1.0
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Time (Second)")
    ax.set_title("A Single ECG Segment", pad=8)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, cax=cax)
    cb.set_label("Saliency", fontsize=9)


def draw_feature_axis(ax, context: pd.DataFrame, feature: str, label: str, x_minutes: np.ndarray, score_col: str) -> None:
    vals = context[feature].to_numpy(dtype=float)
    cmap = plt.get_cmap("seismic")
    norm = score_norm(context[score_col].to_numpy(dtype=float))
    ax.plot(x_minutes, vals, color="0.65", linewidth=1.0, zorder=1)
    ax.scatter(
        x_minutes,
        vals,
        c=context[score_col].to_numpy(dtype=float),
        cmap=cmap,
        norm=norm,
        edgecolor="black",
        linewidth=0.25,
        s=18,
        zorder=2,
    )
    ax.set_ylabel(label)
    ax.grid(alpha=0.18, linewidth=0.5)


def draw_panel_c(fig, raw: np.ndarray, context: pd.DataFrame) -> None:
    left = 0.60
    width = 0.32
    cbar_left = 0.93
    top = 0.86
    trace_h = 0.135
    small_h = 0.055
    gap = 0.032

    trace_ax = fig.add_axes([left, top - trace_h, width, trace_h])
    cax = fig.add_axes([cbar_left, top - trace_h, 0.012, trace_h])
    draw_focal_trace(trace_ax, raw, context, "saliency", cax)
    trace_ax.set_xlabel("Time (Second)", labelpad=0)

    focal_pos = int(np.where(context["is_focal"].to_numpy(dtype=int) == 1)[0][0])
    x_minutes = (np.arange(len(context)) - focal_pos) * (WINDOW_SEC / 60.0)
    y = top - trace_h - 0.055 - small_h
    mean_hr_top = y + small_h
    for row, (feature, label) in enumerate(FEATURES, start=1):
        ax = fig.add_axes([left, y, width, small_h])
        draw_feature_axis(ax, context, feature, label, x_minutes, "saliency")
        if row < 7:
            ax.tick_params(labelbottom=False)
        y -= small_h + gap
    attention_ax = fig.add_axes([left, y, width, small_h])
    attention_ax.plot(x_minutes, context["attention_weight"], color="0.65", linewidth=1.0, zorder=1)
    attention_ax.scatter(
        x_minutes,
        context["attention_weight"],
        c=context["saliency"],
        cmap="seismic",
        norm=score_norm(context["saliency"].to_numpy(dtype=float)),
        edgecolor="black",
        linewidth=0.25,
        s=18,
        zorder=2,
    )
    attention_ax.set_ylabel("Attention\nWeight")
    attention_ax.set_xlabel("Time Relative to Focal Segment (Min)")
    attention_ax.grid(alpha=0.18, linewidth=0.5)

    bracket_y = top - trace_h - 0.035
    bracket_tick_top = bracket_y + 0.012
    bracket_left = left
    bracket_right = left + width
    arrow_x = left + width / 2
    for x0, x1, y0, y1 in [
        (bracket_left, bracket_right, bracket_y, bracket_y),
        (bracket_left, bracket_left, bracket_y, bracket_tick_top),
        (bracket_right, bracket_right, bracket_y, bracket_tick_top),
    ]:
        fig.add_artist(
            mpl.lines.Line2D(
                [x0, x1],
                [y0, y1],
                transform=fig.transFigure,
                color="0.60",
                linewidth=1.2,
                solid_capstyle="butt",
                clip_on=False,
            )
        )
    fig.add_artist(
        mpl.patches.FancyArrowPatch(
            (arrow_x, bracket_y),
            (arrow_x, mean_hr_top + 0.006),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.2,
            color="0.60",
            shrinkA=0,
            shrinkB=0,
            clip_on=False,
        )
    )


def draw_figure(patient: PatientFigure) -> None:
    summary = json.loads((patient.source_dir / "summary.json").read_text())
    context = pd.read_csv(patient.source_dir / "top_ecg_segments_attention.csv")
    raw = raw_ecg(patient.pid)

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax_a = fig.add_axes([0.035, 0.80, 0.420, 0.11])
    cax_a = fig.add_axes([0.468, 0.80, 0.012, 0.11])
    ax_b = fig.add_axes([0.035, 0.05, 0.54, 0.60])

    draw_panel_a(ax_a, cax_a, patient.source_dir)
    draw_panel_b(ax_b, patient.source_dir / "text_saliency.html", patient.endpoint)
    draw_panel_c(fig, raw, context)

    fig.text(0.015, 0.965, "(a)", fontsize=22, ha="left", va="top")
    fig.text(0.015, 0.715, "(b)", fontsize=22, ha="left", va="top")
    fig.text(0.575, 0.965, "(c)", fontsize=22, ha="left", va="top")
    png_path = OUT_DIR / f"Fig_ECG_{patient.label}_scratch.png"
    svg_path = OUT_DIR / f"Fig_ECG_{patient.label}_scratch.svg"
    fig.savefig(png_path, dpi=DPI)
    fig.savefig(svg_path)
    plt.close(fig)
    print(png_path)
    print(svg_path)


def main() -> None:
    configure_matplotlib()
    for patient in PATIENTS:
        draw_figure(patient)


if __name__ == "__main__":
    main()
