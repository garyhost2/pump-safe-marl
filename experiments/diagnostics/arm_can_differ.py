#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from experiments.controllers import ARMS, mechanism_switches

NEEDS_CHARGERS = ("forced_penalty", "ev_floor", "ev_request")


def _has_chargers(schema: str) -> bool:
    try:
        data = json.load(open(schema, encoding="utf-8"))
    except OSError:
        return "ev" in schema.lower()
    for b in data.get("buildings", {}).values():
        if b.get("chargers"):
            return True
    return False


def _signature(arm, chargers: bool):
    fields = {}
    for f in dataclasses.fields(arm):
        if f.name == "name":
            continue
        if f.name in NEEDS_CHARGERS and not chargers:
            continue
        if f.name == "mechanism":
            continue
        fields[f.name] = getattr(arm, f.name)
    fields["_resolved_mechanism"] = mechanism_switches(arm)
    return fields


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", required=True)
    ap.add_argument("--baseline", default="rl+calibrated")
    ap.add_argument("arms", nargs="+")
    a = ap.parse_args()

    chargers = _has_chargers(a.schema)
    base = ARMS[a.baseline]
    inert = []
    print(f"schema {a.schema}  (chargers: {chargers})")
    for name in a.arms:
        if name not in ARMS:
            print(f"  {name:<26} UNKNOWN ARM")
            inert.append(name)
            continue
        arm = ARMS[name]
        diffs = {f.name: (getattr(base, f.name), getattr(arm, f.name))
                 for f in dataclasses.fields(arm)
                 if f.name != "name" and getattr(base, f.name) != getattr(arm, f.name)}
        if not diffs:
            print(f"  {name:<26} identical to {a.baseline} by construction")
            inert.append(name)
            continue
        dead = [k for k in diffs if k in NEEDS_CHARGERS and not chargers]
        live = [k for k in diffs if k not in dead]
        if live:
            print(f"  {name:<26} differs in {', '.join(sorted(live))}")
        else:
            print(f"  {name:<26} INERT: only differs in {', '.join(sorted(dead))}, "
                  f"which needs chargers")
            inert.append(name)

    known = [n for n in a.arms if n in ARMS]
    sigs = {n: _signature(ARMS[n], chargers) for n in known}
    dupes = []
    for i, x in enumerate(known):
        for y in known[i + 1:]:
            if sigs[x] == sigs[y]:
                dupes.append((x, y))
    for x, y in dupes:
        print(f"  {x:<26} DUPLICATE of {y} once mechanism='auto' is resolved")

    if inert or dupes:
        if inert:
            print(f"\n{len(inert)} arm(s) cannot differ from {a.baseline} on this "
                  f"schema: {', '.join(inert)}")
        if dupes:
            print(f"{len(dupes)} pair(s) of arms in this grid are the same run: "
                  + "; ".join(f"{x} == {y}" for x, y in dupes))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())