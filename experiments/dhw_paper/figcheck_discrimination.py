#!/usr/bin/env python3

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import make_tables as MT
import trace_week as TW
from figcheck import check_figure

OUT = REPO / "results" / "dhw_paper" / "figure_checks.json"


def run_variant(module, func_name, replacements, *args):
    src = inspect.getsource(getattr(module, func_name))
    for a, b in replacements:
        if a not in src:
            raise RuntimeError(f"{func_name}: substitution target not found: {a[:60]!r}")
        src = src.replace(a, b)
    captured = {}
    ns = dict(vars(module))
    ns["_capture"] = lambda fig, name: captured.setdefault("r", check_figure(fig, name))
    exec(compile(src, f"<{func_name} variant>", "exec"), ns)
    ns[func_name](*args)
    plt.close("all")
    return captured["r"]


def main():
    df = MT.pd.concat([d for d in (MT.load_jsonl("main"), MT.load_jsonl("power_limited"),
                                   MT.load_jsonl("base_filter")) if not d.empty],
                      ignore_index=True)
    agg = MT.aggregate(df)
    no_save = [("record(check_figure(fig, path.name), FIGCHECK)", "_capture(fig, path.name)"),
               ('fig.savefig(path, dpi=300, bbox_inches="tight")', "pass")]
    res = {}
    res["known_bad: completion, single-line titles"] = run_variant(
        MT, "fig_completion",
        [('"Time-of-use price:\\nthe price-aware policy places every cycle"',
          '"Time-of-use price: the price-aware policy places every cycle"'),
         ('"Flat price:\\ncompletion depends on the trigger"',
          '"Flat price: completion is the trigger\'s"')] + no_save,
        agg, Path("unused.png"))
    res["known_bad: trade-off, legend lower right of the flat panel"] = run_variant(
        MT, "fig_tradeoff",
        [('axes[0].legend(handles=handles, frameon=False, fontsize=6, loc="lower left")',
          'axes[1].legend(handles=handles, frameon=False, fontsize=6, loc="lower right")')]
        + no_save, agg, Path("unused.png"))
    old_block = '''    axes[0].legend(frameon=False, fontsize=6, loc="center right", ncol=2, bbox_to_anchor=(1.0, 0.62))
    axes[0].text(days[-1], 60.8, "60 °C", fontsize=6, color="#D55E00", ha="right")
    axes[0].text(days[-1], 46, "comfort band", fontsize=6, color="#0072B2", ha="right")'''
    src_main = inspect.getsource(TW.main)
    start = src_main.index("    # direct labels instead of a legend")
    end = src_main.index("    axes[2].bar(")
    trace_repl = [(src_main[start:end], old_block + "\n"),
                  ('np.savetxt(', 'None and np.savetxt('),
                  ('record(check_figure(fig, "fig_trace.png"), out / "figure_checks.json")',
                   '_capture(fig, "fig_trace.png")'),
                  ('fig.savefig(out / "fig_trace.png", dpi=300, bbox_inches="tight")', "pass"),
                  ('print("written", out / "fig_trace.png")', "pass")]
    argv = sys.argv
    sys.argv = [argv[0]]
    try:
        res["known_bad: trace, two-column legend over the band label"] = run_variant(
            TW, "main", trace_repl)
    finally:
        sys.argv = argv
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    current = {k: v for k, v in data.items() if k.endswith(".png")}
    data.update(res)
    data["discrimination"] = {
        "known_bad_flagged": {k: not v["clean"] for k, v in res.items()},
        "current_clean": {k: v["clean"] for k, v in current.items()},
        "passes": all(not v["clean"] for v in res.values()) and all(
            v["clean"] for v in current.values()) and len(current) == 4,
    }
    OUT.write_text(json.dumps(data, indent=1) + "\n")
    for k, v in res.items():
        print(k, "| flagged:", not v["clean"], "| text-text", len(v["text_text_overlaps"]),
              "| marker-in-box", len(v["marker_in_text_or_legend"]))
    print("discrimination passes:", data["discrimination"]["passes"])


if __name__ == "__main__":
    main()
