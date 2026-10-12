"""Command-line interface:  python -m recovery2pl <command> --config CONFIG ...

  list-units  [--block B ...] [--population P ...]   print unit ids, one per line
  shard       --index I --of K [--block ...]          print the unit ids of shard I (0-based) of K
  show-unit   --unit ID                               condition, seeds and a data summary (no fit)
  plan        --shards K [--strategy balanced|round_robin] [--unit ID ...] --json FILE
                                                      shard plan for parallel jobs (JSON)
  run         (--unit ID ... | --units-file F | --shard I/K) --out DIR
  collect     --from DIR [DIR ...] --out DIR          merge downloaded unit results (verifies duplicates)
  aggregate   --out DIR                               replicate metrics + summaries + envelopes

`run` and `aggregate` read and write only under --out. Thread caps for numerical libraries should
be set in the environment before the process starts (e.g. OMP_NUM_THREADS=1).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from recovery2pl.config import load_config
from recovery2pl.units import enumerate_units, resolve, shard


def _units(config, args):
    return enumerate_units(config, blocks=args.block or None, populations=args.population or None)


def _selected(config, args):
    if args.unit:
        return resolve(config, args.unit)
    if args.units_file:
        ids = [line.strip() for line in Path(args.units_file).read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.startswith("#")]
        return resolve(config, ids)
    if args.shard:
        i, k = (int(x) for x in args.shard.split("/"))
        return shard(_units(config, args), i, k)
    raise SystemExit("run: give --unit, --units-file or --shard I/K")


def load_results(out_root, study_id: str):
    import pandas as pd
    udir = Path(out_root) / study_id / "units"
    files = sorted(udir.rglob("rep_*.parquet"))
    if not files:
        raise SystemExit(f"no unit results under {udir}")
    items = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    diags = pd.DataFrame([json.loads(f.with_suffix(".json").read_text(encoding="utf-8"))
                          for f in files if f.with_suffix(".json").exists()])
    return items, diags


def cmd_aggregate(config, out_root) -> dict:
    from recovery2pl.metrics import replicate_metrics, summarize, upper_envelope
    sdir = Path(out_root) / config["study_id"]
    items, diags = load_results(out_root, config["study_id"])
    rep = replicate_metrics(items, config["reporting"]["item_error_tolerances"])
    rep.to_csv(sdir / "replicate_metrics.csv", index=False)
    diags.drop(columns=["hyper"], errors="ignore").to_csv(sdir / "fit_diagnostics.csv", index=False)
    written = ["replicate_metrics.csv", "fit_diagnostics.csv"]
    for subset in ("primary", "no_divergent", "strict"):
        summ = summarize(rep, subset)
        summ.to_csv(sdir / f"summary_{subset}.csv", index=False)
        written.append(f"summary_{subset}.csv")
        if subset == "primary":
            for e in config["reporting"]["envelopes"]:
                for m in ("rmse_b", "rmse_a", "rmse_log_a"):
                    env = upper_envelope(summ, f"{m}_mean", e["populations"], e["worlds"])
                    name = f"envelope_{e['name']}_{m}.csv"
                    env.to_csv(sdir / name, index=False)
                    written.append(name)
    return {"units": int(rep["unit_id"].nunique()), "written": written}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="recovery2pl", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["list-units", "shard", "show-unit", "plan", "run", "collect",
                                        "aggregate"])
    ap.add_argument("--shards", type=int)
    ap.add_argument("--strategy", default="balanced", choices=["balanced", "round_robin"])
    ap.add_argument("--json")
    ap.add_argument("--from", dest="sources", nargs="+")
    ap.add_argument("--config", required=True)
    ap.add_argument("--block", action="append")
    ap.add_argument("--population", action="append")
    ap.add_argument("--index", type=int)
    ap.add_argument("--of", type=int)
    ap.add_argument("--unit", action="append")
    ap.add_argument("--units-file")
    ap.add_argument("--shard", help="I/K, e.g. 0/8")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    config = load_config(args.config)

    if args.command == "list-units":
        for u in _units(config, args):
            print(u.unit_id)
    elif args.command == "shard":
        if args.index is None or args.of is None:
            raise SystemExit("shard: give --index and --of")
        for u in shard(_units(config, args), args.index, args.of):
            print(u.unit_id)
    elif args.command == "show-unit":
        from recovery2pl.simulate import simulate_replicate
        for u in resolve(config, args.unit or []):
            sim = simulate_replicate(config, u.condition, u.replicate)
            print(json.dumps({"unit_id": u.unit_id, "block": u.block, "condition": vars(u.condition),
                              "sampler_seed": sim["sampler_seed"], "item_tries": sim["item_tries"],
                              "p_correct": float(sim["Y"].mean()),
                              "theta_mean": float(sim["theta"].mean()),
                              "b_mean": float(sim["b"].mean())}, indent=2))
    elif args.command == "plan":
        from recovery2pl.units import plan
        if not args.shards or not args.json:
            raise SystemExit("plan: give --shards and --json")
        units = resolve(config, args.unit) if args.unit else _units(config, args)
        doc = {"study_id": config["study_id"], "config": args.config, "strategy": args.strategy,
               "n_units": len(units), "shards": plan(units, args.shards, args.strategy)}
        Path(args.json).write_text(json.dumps(doc, indent=1), encoding="utf-8")
        print(json.dumps({"study_id": doc["study_id"], "n_units": doc["n_units"],
                          "shards": [(s["label"], len(s["units"]), s["estimated_cost"])
                                     for s in doc["shards"]]}))
    elif args.command == "collect":
        if not args.sources or not args.out:
            raise SystemExit("collect: give --from and --out")
        from recovery2pl.runner import collect
        print(json.dumps(collect(args.sources, args.out, config["study_id"])))
    elif args.command == "run":
        if not args.out:
            raise SystemExit("run: give --out")
        from recovery2pl.runner import run_units
        res = run_units(config, _selected(config, args), args.out)
        print(json.dumps(res))
        # infrastructure crash -> 1; sampler exception (kept as a gross fit) -> 2; both fail the job
        return 1 if res["crashed"] else (2 if res["sampler_error_units"] else 0)
    elif args.command == "aggregate":
        if not args.out:
            raise SystemExit("aggregate: give --out")
        print(json.dumps(cmd_aggregate(config, args.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
