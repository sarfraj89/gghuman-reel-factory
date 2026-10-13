#!/usr/bin/env python3
"""GGhuman 'Said vs Thought' reel factory.

  python run.py render specs/001_settler_trade.json      # one video
  python run.py batch --specs specs --out out             # render + QC a folder
  python run.py ideate --n 100 --out specs_gen            # LLM writes 100 specs
  python run.py factory --n 100                           # ideate + batch, end to end

  Storyboard formats (doodle skits, cinematic series episodes):
  python run.py render specs_story/ref_doodle_slowroll.json --out out_story
  python run.py story-ideate --format doodle --n 5
  python run.py story-factory --format cinematic --n 3 --series settler_rush
"""
import argparse
import csv
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

def _load_dotenv(path=Path(__file__).resolve().parent / ".env"):
    """Minimal .env loader (KEY=value lines); real environment variables win."""
    import os
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


_load_dotenv()

from pipeline import qc, render, spec as S  # noqa: E402
from pipeline.draw import LayoutError
from pipeline.story import spec as SS  # noqa: E402


def job(spec_path, out_dir):
    """Render + QC one spec. Returns a manifest row; never raises."""
    t0 = time.time()
    row = {"spec": str(spec_path), "id": "", "status": "", "reason": "", "file": "", "seconds": 0}
    try:
        sp = S.load(spec_path)
        story = SS.is_story(sp)
        if story:
            sp = SS.resolve(sp)
        row["id"] = sp.get("id", Path(spec_path).stem)
        errs = SS.validate(sp) if story else S.validate(sp)
        if errs:
            row.update(status="spec_invalid", reason=" | ".join(errs))
            return row
        out = Path(out_dir) / f"{sp['id']}.mp4"
        if story:
            from pipeline.story import render as story_render
            story_render.render(sp, out)
            ok, rep = qc.run_story(out, sp)
            if not ok and all(f.startswith("draft:") for f in rep["fails"]):
                row.update(file=str(out), status="draft", reason=" | ".join(rep["fails"] + rep["warnings"]))
                row["seconds"] = round(time.time() - t0, 1)
                return row
        else:
            render.render(sp, out)
            ok, rep = qc.run(out, sp)
        row.update(file=str(out), status="ok" if ok else "qc_failed",
                   reason=" | ".join(rep["fails"] + [f"warn: {w}" for w in rep["warnings"]]))
    except LayoutError as e:
        row.update(status="layout_error", reason=str(e))
    except Exception as e:
        row.update(status="render_error", reason=f"{e}\n{traceback.format_exc(limit=2)}")
    row["seconds"] = round(time.time() - t0, 1)
    return row


def batch(specs_dir, out_dir, workers):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.csv"
    done = {}
    if manifest.exists():  # resume: skip anything already rendered OK
        with manifest.open() as f:
            done = {r["spec"]: r for r in csv.DictReader(f)}
    specs = sorted(Path(specs_dir).glob("*.json"))
    todo = [p for p in specs if done.get(str(p), {}).get("status") != "ok"]
    print(f"{len(specs)} specs, {len(specs) - len(todo)} already done, rendering {len(todo)} with {workers} workers")
    rows = dict(done)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(job, p, out_dir): p for p in todo}
        for fut in as_completed(futs):
            r = fut.result()
            rows[r["spec"]] = r
            print(f"  [{r['status']:>12}] {r['id']}  {r['seconds']}s  {r['reason'][:120]}")
            _write_manifest(manifest, rows.values())
    stats = {}
    for r in rows.values():
        stats[r["status"]] = stats.get(r["status"], 0) + 1
    print("summary:", json.dumps(stats))
    return list(rows.values())


def _write_manifest(path, rows):
    fields = ["spec", "id", "status", "reason", "file", "seconds"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["spec"]):
            w.writerow({k: r.get(k, "") for k in fields})


def repair_failures(rows, specs_dir):
    """Layout/spec failures go back to the LLM with the exact error once."""
    from pipeline import ideate
    client = ideate._client()
    fixed = 0
    for r in rows:
        if r["status"] in ("layout_error", "spec_invalid"):
            sp = S.load(r["spec"])
            msg = (f"This reel spec failed: {r['reason']}. Return a corrected version with the "
                   f"same id, game and opponent, keeping the idea.\n{json.dumps(sp, ensure_ascii=False)}")
            data, _ = ideate._ask(client, [{"role": "user", "content": msg}])
            if not S.validate(data):
                Path(r["spec"]).write_text(json.dumps(data, ensure_ascii=False, indent=2))
                fixed += 1
    return fixed


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("render"); a.add_argument("spec"); a.add_argument("--out", default="out")
    b = sub.add_parser("batch"); b.add_argument("--specs", default="specs"); b.add_argument("--out", default="out")
    b.add_argument("--workers", type=int, default=2)
    c = sub.add_parser("ideate"); c.add_argument("--n", type=int, default=10); c.add_argument("--out", default="specs_gen")
    c.add_argument("--no-judge", action="store_true")
    d = sub.add_parser("factory"); d.add_argument("--n", type=int, default=100)
    d.add_argument("--specs", default="specs_gen"); d.add_argument("--out", default="out_gen")
    d.add_argument("--workers", type=int, default=4)
    e = sub.add_parser("story-ideate", help="LLM writes new doodle skits / cinematic episodes")
    e.add_argument("--format", choices=["doodle", "cinematic"], required=True)
    e.add_argument("--n", type=int, default=5); e.add_argument("--out", default="specs_story_gen")
    e.add_argument("--series", default="settler_rush"); e.add_argument("--no-judge", action="store_true")
    f = sub.add_parser("story-factory", help="story-ideate + batch render + QC")
    f.add_argument("--format", choices=["doodle", "cinematic"], required=True)
    f.add_argument("--n", type=int, default=10); f.add_argument("--specs", default="specs_story_gen")
    f.add_argument("--out", default="out_story_gen"); f.add_argument("--workers", type=int, default=3)
    f.add_argument("--series", default="settler_rush")
    args = ap.parse_args()

    if args.cmd == "render":
        r = job(args.spec, args.out)
        print(json.dumps(r, indent=2))
        sys.exit(0 if r["status"] == "ok" else 1)
    if args.cmd == "batch":
        batch(args.specs, args.out, args.workers)
    if args.cmd == "ideate":
        from pipeline import ideate
        ideate.generate(args.n, args.out, use_judge=not args.no_judge)
    if args.cmd in ("story-ideate", "story-factory"):
        from pipeline.story import ideate as story_ideate
        story_ideate.generate(args.format, args.n, args.specs if args.cmd == "story-factory" else args.out,
                              series_ref=args.series, use_judge=not getattr(args, "no_judge", False))
        if args.cmd == "story-factory":
            batch(args.specs, args.out, args.workers)
    if args.cmd == "factory":
        from pipeline import ideate
        ideate.generate(args.n, args.specs)
        rows = batch(args.specs, args.out, args.workers)
        if repair_failures(rows, args.specs):
            batch(args.specs, args.out, args.workers)  # resume renders only the repaired ones


if __name__ == "__main__":
    main()
