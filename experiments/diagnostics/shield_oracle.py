from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def rollout(arm_name: str, scenario, config, max_steps: int,
            phase: str = "eval") -> Dict[str, Any]:
    from stems.environment import STEMSEnvironment
    from stems.utils import HistoryBuffer, set_seed
    from experiments.controllers import ARMS, build_controller

    set_seed(0)
    env = STEMSEnvironment(schema=scenario.schema_path(), seed=0,
                           heat_pump=scenario.heat_pump,
                           env_kwargs=scenario.env_kwargs(phase),
                           hvac_control=scenario.hvac_control,
                           allow_missing_obs=scenario.allow_missing_obs)
    controller = build_controller(ARMS[arm_name], env, config)

    names = list(env._selected_obs_names)
    hist = HistoryBuffer(env.num_buildings, env.obs_dim, config.transformer.window_size)
    obs, _ = env.reset()
    hist.prime(obs)

    X: List[np.ndarray] = []
    y: List[int] = []
    mag: List[float] = []
    n, done = 0, False
    while not done:
        actions = controller.select_action(obs, hist.get(), explore=False)
        nominal = getattr(controller, "_last_nominal_actions", None)
        if nominal is None:
            nominal = getattr(controller, "_last_raw_actions", None)
        safe = getattr(controller, "_last_safe_actions", None)
        if nominal is not None and safe is not None:
            delta = np.abs(np.asarray(nominal) - np.asarray(safe))
            for b in range(env.num_buildings):
                X.append(np.asarray(obs[b], dtype=np.float32))
                moved = float(delta[b].max())
                y.append(int(moved > 1e-9))
                mag.append(moved)
        nxt, _, term, trunc, _ = env.step(actions)
        if hasattr(controller, "observe"):
            controller.observe(nxt, env.ev_draw_kwh)
        hist.update(nxt)
        obs = nxt
        n += 1
        done = bool(term or trunc) or n >= max_steps

    if not X:
        raise SystemExit(f"arm {arm_name!r} exposed no pre-shield action, so no "
                         "intervention label could be formed. Only shielded arms "
                         "carry a decision boundary to describe.")
    return {"X": np.asarray(X), "y": np.asarray(y), "magnitude": np.asarray(mag),
            "feature_names": names, "steps": n, "buildings": env.num_buildings}


def _describe(tree, names: List[str]) -> str:
    t = tree.tree_
    lines: List[str] = []

    def walk(node: int, indent: str) -> None:
        if t.children_left[node] == -1:
            v = t.value[node][0]
            lines.append(f"{indent}-> {'INTERVENE' if int(np.argmax(v)) else 'allow'}  "
                         f"(n={int(t.n_node_samples[node])}, purity={v.max()/v.sum():.3f})")
            return
        name, thr = names[t.feature[node]], t.threshold[node]
        lines.append(f"{indent}if {name} <= {thr:.6f}   (n={int(t.n_node_samples[node])})")
        walk(t.children_left[node], indent + "    ")
        lines.append(f"{indent}else")
        walk(t.children_right[node], indent + "    ")

    walk(0, "")
    return "\n".join(lines)


def _degenerate_splits(tree, names: List[str]) -> List[Dict[str, Any]]:
    t = tree.tree_
    out: List[Dict[str, Any]] = []
    for i in range(t.node_count):
        l, r = t.children_left[i], t.children_right[i]
        if l == -1 or t.children_left[l] != -1 or t.children_left[r] != -1:
            continue
        if int(np.argmax(t.value[l][0])) == int(np.argmax(t.value[r][0])):
            out.append({"feature": names[t.feature[i]],
                        "threshold": float(t.threshold[i]),
                        "both_predict": int(np.argmax(t.value[l][0]))})
    return out


def fit_surrogate(data: Dict[str, Any], depth: int, out_dir: str) -> Dict[str, Any]:
    from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
    from sklearn.tree import DecisionTreeClassifier

    X, y, names = data["X"], data["y"], data["feature_names"]
    base_rate = float(y.mean())
    if base_rate in (0.0, 1.0):
        raise SystemExit(f"the shield intervened on {base_rate:.0%} of steps, so there "
                         "is no decision boundary to describe. Pick an arm and a cap "
                         "where the barrier actually does something.")

    per_building = data["buildings"]
    cut = (len(y) // (2 * per_building)) * per_building
    Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]

    tree = DecisionTreeClassifier(max_depth=depth, class_weight="balanced",
                                  random_state=0, min_samples_leaf=50)
    tree.fit(Xtr, ytr)
    pred = tree.predict(Xte)
    prec, rec, f1, _ = precision_recall_fscore_support(yte, pred, average="binary",
                                                       zero_division=0)
    report = {
        "base_rate_intervention": base_rate,
        "base_rate_heldout": float(yte.mean()),
        "n_steps": int(data["steps"]),
        "n_building_steps": int(len(y)),
        "train_building_steps": int(cut),
        "depth": depth,
        "heldout_balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
        "heldout_precision": float(prec),
        "heldout_recall": float(rec),
        "heldout_f1": float(f1),
        "mean_correction_when_intervening": float(data["magnitude"][y == 1].mean()),
        "rules": _describe(tree, names),
    }
    report["degenerate_splits"] = _degenerate_splits(tree, names)
    top = np.argsort(tree.feature_importances_)[::-1][:6]
    report["top_features"] = [{"feature": names[i],
                               "importance": float(tree.feature_importances_[i])}
                              for i in top if tree.feature_importances_[i] > 0]
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "shield_dataset.npz"),
                        X=X, y=y, magnitude=data["magnitude"],
                        feature_names=np.asarray(names, dtype=object))
    with open(os.path.join(out_dir, "shield_oracle.json"), "w") as fh:
        json.dump({k: v for k, v in report.items() if k != "rules"}, fh, indent=2)
    with open(os.path.join(out_dir, "shield_rules.txt"), "w") as fh:
        fh.write(report["rules"])
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="rbc+calibrated")
    ap.add_argument("--season", default="year")
    ap.add_argument("--grid-cap", type=float, default=33.5)
    ap.add_argument("--building-cap", type=float, default=11.5)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--reuse", action="store_true",
                    help="refit from a saved shield_dataset.npz instead of rolling out")
    ap.add_argument("--ladder", default="1,2,3,4,5",
                    help="depths to score for the fidelity ladder")
    ap.add_argument("--out", default="results/shield_oracle")
    args = ap.parse_args()

    from experiments.runner import make_config
    from experiments.scenario import Scenario

    scenario = Scenario(season=args.season, grid_cap_kw=args.grid_cap,
                        building_cap_kw=args.building_cap)
    config = make_config(scenario)
    cached = os.path.join(args.out, "shield_dataset.npz")
    if args.reuse and os.path.exists(cached):
        z = np.load(cached, allow_pickle=True)
        data = {"X": z["X"], "y": z["y"], "magnitude": z["magnitude"],
                "feature_names": list(z["feature_names"]),
                "steps": int(len(z["y"]) // 8), "buildings": 8}
    else:
        data = rollout(args.arm, scenario, config, max_steps=10 ** 9)
    report = fit_surrogate(data, args.depth, args.out)

    ladder = []
    for d in [int(x) for x in args.ladder.split(",") if x.strip()]:
        r = fit_surrogate(dict(data), d, os.path.join(args.out, "_ladder"))
        ladder.append({"depth": d,
                       "heldout_balanced_accuracy": r["heldout_balanced_accuracy"],
                       "heldout_precision": r["heldout_precision"],
                       "heldout_recall": r["heldout_recall"],
                       "degenerate_splits": len(r["degenerate_splits"])})
    report["fidelity_ladder"] = ladder
    with open(os.path.join(args.out, "shield_oracle.json"), "w") as fh:
        json.dump({k: v for k, v in report.items() if k != "rules"}, fh, indent=2)

    print(f"arm {args.arm}  season {args.season}  "
          f"cap {args.grid_cap}/{args.building_cap} kW")
    print(f"  shield intervened on {report['base_rate_intervention']:.1%} of "
          f"{report['n_building_steps']} building-steps "
          f"({report['n_steps']} hours x {data['buildings']} buildings)")
    print(f"  mean correction when it fired: "
          f"{report['mean_correction_when_intervening']:.4f} (action units)")
    print(f"  depth-{args.depth} tree on held-out second half: "
          f"balanced accuracy {report['heldout_balanced_accuracy']:.3f}, "
          f"precision {report['heldout_precision']:.3f}, "
          f"recall {report['heldout_recall']:.3f}")
    print("  features the boundary leans on:")
    for f in report.get("top_features", []):
        print(f"    {f['importance']:.3f}  {f['feature']}")
    deg = report.get("degenerate_splits") or []
    if deg:
        print(f"  {len(deg)} split(s) change no decision (both children agree); "
              f"effective depth is lower than {args.depth}:")
        for g in deg:
            print(f"    {g['feature']} <= {g['threshold']:.6f}")
    print()
    print(report["rules"])
    print("\n  fidelity by depth (held-out second half):")
    for row in ladder:
        print(f"    depth {row['depth']}: balanced accuracy "
              f"{row['heldout_balanced_accuracy']:.4f}  "
              f"precision {row['heldout_precision']:.3f}  "
              f"recall {row['heldout_recall']:.3f}")
    print(f"\nwrote {args.out}/shield_oracle.json and shield_rules.txt")


if __name__ == "__main__":
    main()
