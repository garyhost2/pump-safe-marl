from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.text as mtext


def _drawn_tick_labels(ax):
    out = []
    for axis in (ax.xaxis, ax.yaxis):
        for tick in axis._update_ticks():
            for lab in (tick.label1, tick.label2):
                if lab.get_visible() and lab.get_text().strip():
                    out.append(lab)
    return out


def _all_tick_labels(ax):
    s = set()
    for axis in (ax.xaxis, ax.yaxis):
        for tick in axis.get_major_ticks() + axis.get_minor_ticks():
            s.add(id(tick.label1))
            s.add(id(tick.label2))
    return s


def check_figure(fig, name: str, dpi: float = 300.0, tol_px: float = 0.5) -> Dict:
    fig.set_dpi(dpi)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    tick_ids = set()
    texts: List = []
    for ax in fig.axes:
        tick_ids |= _all_tick_labels(ax)
        texts += _drawn_tick_labels(ax)
    for t in fig.findobj(mtext.Text):
        if id(t) in tick_ids or not t.get_visible() or not t.get_text().strip():
            continue
        texts.append(t)
    boxes = []
    for t in texts:
        bb = t.get_window_extent(renderer=r)
        if bb.width > 0 and bb.height > 0:
            boxes.append((t.get_text().replace("\n", " / "), bb))
    tt = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            w = min(a.x1, b.x1) - max(a.x0, b.x0)
            h = min(a.y1, b.y1) - max(a.y0, b.y0)
            if w > tol_px and h > tol_px:
                tt.append({"a": boxes[i][0], "b": boxes[j][0], "px2": round(w * h, 1)})
    legends = [(f"legend of axes {k}", ax.get_legend().get_window_extent(renderer=r))
               for k, ax in enumerate(fig.axes) if ax.get_legend() is not None]
    tm = []
    for k, ax in enumerate(fig.axes):
        for ln in ax.get_lines():
            mk = ln.get_marker()
            if not ln.get_visible() or mk in (None, "None", "", " ") or len(ln.get_xydata()) == 0:
                continue
            rad = ln.get_markersize() * dpi / 72.0 / 2.0 + ln.get_markeredgewidth() * dpi / 72.0
            for (px, py) in ax.transData.transform(ln.get_xydata()):
                for label, bb in boxes + legends:
                    dx = max(bb.x0 - px, 0.0, px - bb.x1)
                    dy = max(bb.y0 - py, 0.0, py - bb.y1)
                    if (dx * dx + dy * dy) ** 0.5 < rad - tol_px:
                        tm.append({"text": label, "axes": k, "marker_px": [round(px), round(py)]})
    outside = [lab for lab, bb in boxes if bb.x0 < -tol_px or bb.y0 < -tol_px or
               bb.x1 > W + tol_px or bb.y1 > H + tol_px]
    return {"figure": name, "dpi": dpi, "texts_checked": len(boxes),
            "markers_checked": int(sum(len(ln.get_xydata()) for ax in fig.axes for ln in ax.get_lines()
                                       if ln.get_marker() not in (None, "None", "", " "))),
            "text_text_overlaps": tt, "marker_in_text_or_legend": tm, "texts_outside_canvas": outside,
            "clean": not tt and not tm}


def record(result: Dict, path: Path) -> None:
    data = json.loads(path.read_text()) if path.exists() else {}
    data[result["figure"]] = result
    path.write_text(json.dumps(data, indent=1) + "\n")
