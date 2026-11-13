#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field

import numpy as np

PASS, WARN, FAIL = "pass", "warn", "fail"
_MARK = {PASS: "PASS", WARN: "WARN", FAIL: "FAIL"}

EXOGENOUS = ("outdoor", "weather", "humidity", "solar", "pricing", "price",
             "carbon", "hour", "day_type", "month", "daylight", "diffuse",
             "direct", "occupant_count", "temperature_predicted",
             "dhw_demand", "non_shiftable_load", "power_outage", "set_point")

SCHEDULE_METADATA = ("departure_time", "arrival_time", "required_soc",
                     "battery_capacity", "connected_state", "incoming_state",
                     "priority")


@dataclass
class Check:
    name: str
    verdict: str
    headline: str
    detail: list = field(default_factory=list)
    table: list = field(default_factory=list)
    columns: tuple = ()
    data: dict = field(default_factory=dict)


def _zero_actions(env):
    return [list(np.zeros(s.shape[0], dtype=float)) for s in env.action_space]


def _unwrap(ret):
    if isinstance(ret, tuple) and len(ret) == 2 and isinstance(ret[1], dict):
        return ret[0]
    return ret


def _district_kw(env, steps):
    per_building = np.asarray(
        [np.asarray(b.net_electricity_consumption, dtype=float)[:steps]
         for b in env.buildings])
    if per_building.shape[1] < steps:
        raise RuntimeError(
            f"env exposed {per_building.shape[1]} computed steps, expected {steps}")
    return per_building.sum(axis=0)


def _rollout(make_env, steps, action_fn):
    env = make_env()
    obs = _unwrap(env.reset())
    trace = [obs]
    for t in range(steps):
        obs = _unwrap(env.step(action_fn(env, t))[0])
        trace.append(obs)
    return env, trace, _district_kw(env, steps)


def _flat(trace):
    return np.asarray([np.concatenate([np.asarray(o, dtype=float) for o in step])
                       for step in trace])


def _channel_names(env):
    return [f"b{i}:{n}" for i, b in enumerate(env.buildings)
            for n, on in b.observation_metadata.items() if on]


def _active_actions(b):
    return [k for k, v in b.action_metadata.items() if v]


def check_bindingness(district_kw, cap_kw, steps) -> Check:
    peak = float(district_kw.max())
    if not np.any(district_kw):
        return Check(
            "Constraint bindingness", FAIL,
            f"The district series could not be measured: all {steps} samples are "
            "exactly zero, so no bindingness verdict is possible.",
            ["This is a defect in the audit, not in the dataset. A real no-control "
             "trajectory is never identically zero at every step.",
             "Check that the district series is read at the steps that have been "
             "computed: CityLearn truncates it to `[:time_step + 1]` and `step()` "
             "advances `time_step` after writing the balance, so the final element "
             "is the next step's preallocated zero."])
    if cap_kw is None:
        return Check("Constraint bindingness", WARN,
                     f"No cap supplied. Uncontrolled district peak is {peak:.4g} kW "
                     f"over {steps} steps.",
                     ["Re-run with --cap-kw to test whether your limit can bind.",
                      "Report this peak beside your limit either way."])
    over = int((district_kw >= cap_kw).sum())
    if over == 0:
        return Check(
            "Constraint bindingness", FAIL,
            f"The cap cannot bind: uncontrolled peak {peak:.4g} kW against a "
            f"{cap_kw:.4g} kW cap, reached in 0 of {steps} steps.",
            [f"Headroom {cap_kw - peak:.4g} kW "
             f"({100 * (cap_kw - peak) / cap_kw:.1f}% of the cap).",
             "A violation rate measured here is structurally zero. It is not evidence "
             "that the controller respects the constraint, because no admissible "
             "trajectory could have broken it.",
             f"For the constraint to bind, set the cap below {peak:.4g} kW."])
    frac = 100.0 * over / steps
    return Check("Constraint bindingness", PASS if frac >= 1.0 else WARN,
                 f"The cap can bind: the uncontrolled trajectory reaches {cap_kw:.4g} kW "
                 f"in {over} of {steps} steps ({frac:.2f}%).",
                 [f"Uncontrolled district peak {peak:.4g} kW."]
                 + ([] if frac >= 1.0 else
                    ["Under 1% of steps are binding, so the rate will rest on a few "
                     "events. Report counts, not only rates."]))


def check_responsiveness(make_env, steps, settle) -> Check:
    env0 = make_env()
    env0.reset()
    names = _channel_names(env0)
    uniq = sorted({a for b in env0.buildings for a in _active_actions(b)})

    _, base_trace, _ = _rollout(make_env, steps, lambda e, t: _zero_actions(e))
    base = _flat(base_trace)[settle:]

    moved = dict.fromkeys(names, 0.0)
    per_action = {}
    for a_name in uniq:
        def push(env, t, _a=a_name):
            acts = []
            for b, space in zip(env.buildings, env.action_space):
                v = np.zeros(space.shape[0], dtype=float)
                act = _active_actions(b)
                if _a in act:
                    j = act.index(_a)
                    if float(space.low[j]) < 0.0:
                        amp = 0.5 * min(abs(float(space.low[j])),
                                        float(space.high[j]))
                        v[j] = amp if t % 2 == 0 else -amp
                    else:
                        v[j] = float(space.high[j])
                acts.append(list(v))
            return acts

        _, trace, _ = _rollout(make_env, steps, push)
        pert = _flat(trace)[settle:]
        n = min(len(base), len(pert))
        delta = np.abs(base[:n] - pert[:n]).max(axis=0)
        per_action[a_name] = int((delta > 1e-9).sum())
        for k, nm in enumerate(names):
            if k < len(delta):
                moved[nm] = max(moved[nm], float(delta[k]))

    dead = sorted(n for n, d in moved.items() if d <= 1e-9)
    exogenous = [n for n in dead if any(tok in n.lower() for tok in EXOGENOUS)]
    metadata = [n for n in dead if n not in set(exogenous)
                and any(tok in n.lower() for tok in SCHEDULE_METADATA)]
    suspicious = [n for n in dead
                  if n not in set(exogenous) and n not in set(metadata)]
    tbl = [(a, per_action[a]) for a in uniq]

    if metadata:
        kinds = sorted({n.split(":", 1)[1] if ":" in n else n for n in metadata})
        meta_check = Check(
            "Schedule-metadata channels", WARN,
            f"{len(metadata)} channel(s) describing a scheduled arrival were "
            f"constant over the audited window.",
            ["These describe an entity that arrives rather than a quantity an "
             "action moves, so they are not counted as a responsiveness failure. "
             "They are reported because an experiment that conditions on them is "
             "conditioning on something that did not vary here.",
             f"Kinds: {', '.join(kinds[:8])}"
             + (f" (+{len(kinds) - 8} more)" if len(kinds) > 8 else "") + "."],
            data={"metadata_channels": metadata})
    else:
        meta_check = Check("Schedule-metadata channels", PASS,
                           "No scheduled-arrival channel was constant.")

    if suspicious:
        shown = ", ".join(suspicious[:12])
        more = f" (+{len(suspicious) - 12} more)" if len(suspicious) > 12 else ""
        return Check(
            "Action to observation responsiveness", FAIL,
            f"{len(suspicious)} non-exogenous observation(s) do not respond to any action.",
            ["An observation that never moves cannot inform the policy, regardless of "
             "what the simulator computes internally. If a reward or comfort term reads "
             "it, that term is scoring the dataset rather than the controller.",
             f"Channels that did not move: {shown}{more}.",
             f"Excluded from this verdict: {len(exogenous)} exogenous and "
             f"{len(metadata)} scheduled-arrival channel(s)."],
            table=tbl, columns=("action dimension", "observations moved"),
            data={"dead_channels": suspicious, "exogenous_dead": exogenous,
                  "metadata_dead": metadata}), meta_check
    return Check("Action to observation responsiveness", PASS,
                 "Every non-exogenous observation responds to at least one action.",
                 [f"{len(dead)} channel(s) did not move: {len(exogenous)} exogenous "
                  f"(weather, pricing, calendar) and {len(metadata)} scheduled-arrival, "
                  f"which is expected."],
                 table=tbl, columns=("action dimension", "observations moved"),
                 data={"exogenous_dead": exogenous, "metadata_dead": metadata}), meta_check


def check_actuation(make_env, steps) -> Check:
    def alternate(env, t):
        acts = []
        for b, space in zip(env.buildings, env.action_space):
            v = np.zeros(space.shape[0], dtype=float)
            for j, nm in enumerate(_active_actions(b)):
                if "storage" in nm:
                    v[j] = 0.5 * float(space.high[j]) * (1 if t % 2 == 0 else -1)
            acts.append(list(v))
        return acts

    env, _, _ = _rollout(make_env, steps, alternate)
    rows, dead = [], []
    for i, b in enumerate(env.buildings):
        for dev in ("electrical_storage", "cooling_storage",
                    "heating_storage", "dhw_storage"):
            d = getattr(b, dev, None)
            cap = getattr(d, "capacity", None) if d is not None else None
            if not cap:
                continue
            bal = np.asarray(getattr(d, "energy_balance", np.zeros(1)), dtype=float)
            ev = int((np.abs(bal) > 1e-9).sum())
            rows.append((f"b{i}:{dev}", ev, steps))
            if ev == 0:
                dead.append(f"b{i}:{dev}")

    if not rows:
        return Check("Device actuation", WARN,
                     "No storage device with non-zero capacity found.",
                     ["Nothing to actuate, so a storage constraint record would be vacuous."])
    if dead:
        return Check("Device actuation", FAIL,
                     f"{len(dead)} device(s) with non-zero capacity never moved under an "
                     f"alternating half-power command.",
                     ["A device that never moves has a perfect constraint record by "
                      "construction.",
                      "Devices that did not move: " + ", ".join(dead)],
                     table=rows,
                     columns=("device", "steps with energy flow", "steps commanded"))
    return Check("Device actuation", PASS,
                 "Every device with non-zero capacity responded to the command.",
                 table=rows,
                 columns=("device", "steps with energy flow", "steps commanded"))


def check_resolution(steps, n_buildings, horizon) -> Check:
    per = 1.0 / (steps * n_buildings)
    detail = [f"Over {steps} audited steps and {n_buildings} buildings, one violating "
              f"building-hour is a rate of {per:.3g}."]
    if horizon:
        full = 1.0 / (horizon * n_buildings)
        detail.append(f"Over the full {horizon}-step episode the smallest non-zero rate is "
                      f"{full:.3g}. Any reported rate below that is zero, and any rate "
                      f"within a small multiple of it is a handful of events.")
    detail.append("Report event counts alongside rates. A rate of 9e-05 over 4379 hours is "
                  "one hour, and a one-event difference between two methods is not a "
                  "difference.")
    return Check("Rare-event resolution", WARN,
                 "A rate is only interpretable beside the count it came from.", detail)


def render(dataset, checks, steps) -> str:
    worst = FAIL if any(c.verdict == FAIL for c in checks) else (
        WARN if any(c.verdict == WARN for c in checks) else PASS)
    out = ["# CityLearn validity audit", "",
           f"- **Dataset**: `{dataset}`",
           f"- **Audited steps**: {steps}",
           f"- **Overall**: **{_MARK[worst]}**", "",
           "| check | verdict | finding |", "|---|---|---|"]
    for c in checks:
        out.append(f"| {c.name} | **{_MARK[c.verdict]}** | {c.headline} |")
    out.append("")
    for c in checks:
        out += [f"## {c.name} — {_MARK[c.verdict]}", "", c.headline, ""]
        out += [f"- {d}" for d in c.detail]
        if c.table:
            out += ["", "| " + " | ".join(c.columns) + " |",
                    "|" + "|".join(["---"] * len(c.columns)) + "|"]
            out += ["| " + " | ".join(str(x) for x in r) + " |" for r in c.table]
        out.append("")
    out += ["---", "",
            "A violation rate cannot establish its own validity. Each check above "
            "corresponds to a way a reported zero can arise with no bearing on the "
            "controller: a limit that cannot be reached, an observation the agent never "
            "sees, a device that never moves, or a rate below the resolution of the "
            "experiment."]
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   help="CityLearn dataset name or path to a schema.json")
    p.add_argument("--cap-kw", type=float, default=None,
                   help="District import cap in kW to test for bindingness")
    p.add_argument("--steps", type=int, default=200, help="Steps to audit (default 200)")
    p.add_argument("--settle", type=int, default=30,
                   help="Steps discarded before comparing observations, so models with a "
                        "lookback buffer have warmed up (default 30)")
    p.add_argument("--out", default=None, help="Write the markdown report here")
    p.add_argument("--json", default=None, help="Also write machine-readable results here")
    a = p.parse_args(argv)

    from citylearn.citylearn import CityLearnEnv

    def make_env():
        return CityLearnEnv(a.dataset, central_agent=False, random_seed=0)

    probe = make_env()
    if not all(hasattr(s, "shape") for s in probe.action_space):
        print(f"cannot audit `{a.dataset}`: its action space is "
              f"{type(probe.action_space).__name__}, the flat-entity (tables) "
              f"interface. This audit supports the per-building list of Box "
              f"spaces. Re-run on a dataset that uses that interface, or extend "
              f"_zero_actions/_flat to the tables layout.", file=sys.stderr)
        return 2
    n_buildings, horizon = len(probe.buildings), int(probe.time_steps)
    steps = max(1, min(a.steps, horizon - 1))

    _, _, district = _rollout(make_env, steps, lambda e, t: _zero_actions(e))
    responsiveness, metadata = check_responsiveness(
        make_env, steps, min(a.settle, steps // 2))
    checks = [check_bindingness(district, a.cap_kw, steps),
              responsiveness, metadata,
              check_actuation(make_env, steps),
              check_resolution(steps, n_buildings, horizon)]

    report = render(a.dataset, checks, steps)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(report + "\n")
        print(f"wrote {a.out}")
    else:
        print(report)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"dataset": a.dataset, "steps": steps,
                       "buildings": n_buildings, "horizon": horizon,
                       "district_peak_kw": float(district.max()),
                       "cap_kw": a.cap_kw,
                       "checks": [{"name": c.name, "verdict": c.verdict,
                                   "headline": c.headline, "detail": c.detail,
                                   "table": c.table, "data": c.data}
                                  for c in checks]}, fh, indent=1)
    return 1 if any(c.verdict == FAIL for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
