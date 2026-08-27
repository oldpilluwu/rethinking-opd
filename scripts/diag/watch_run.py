#!/usr/bin/env python3
"""Compact live view of an OPD run, parsed from the training log.

verl prints every metric for every step onto one very long line, which is unreadable
in a terminal. This pulls out the decision-relevant series and prints one row per step,
plus a projected finish time.

    python scripts/diag/watch_run.py                 # latest log in logs/
    python scripts/diag/watch_run.py --follow        # refresh every 60s
    python scripts/diag/watch_run.py --all           # every opd/* metric for the last step
    python scripts/diag/watch_run.py --log logs/opd_20260827_170917.log

Stdlib only, so it runs in any environment that can see the log file.
"""

import argparse
import glob
import os
import re
import time

# (log key, column header, format) -- chosen to answer the questions that matter mid-run,
# not to be exhaustive. Use --all for everything.
COLUMNS = [
    ("timing_s/step", "sec", "{:.0f}"),
    ("opd/overlap_ratio", "overlap", "{:.4f}"),
    ("opd/top1_agreement", "top1", "{:.4f}"),
    ("opd/reverse_kl_sampled_mean", "revKL", "{:.4f}"),
    ("opd/adv_preclip_abs_p999", "adv_p999", "{:.2f}"),
    ("opd/adv_clip_saturation", "clipsat", "{:.5f}"),
    ("opd/entropy_gap_signed", "entgap", "{:+.4f}"),
    ("opd/repetition_rate", "rep", "{:.3f}"),
    ("response_length/mean", "len", "{:.0f}"),
    ("response_length/clip_ratio", "trunc", "{:.3f}"),
    ("critic/true_reward/mean", "acc", "{:.4f}"),
    ("actor/grad_norm", "gnorm", "{:.3f}"),
]

# Matches "key:value" and "key:np.float64(value)" alike.
_PAT = re.compile(r"([A-Za-z0-9_/@.]+):(?:np\.float64\()?(-?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\)?")


def parse(path):
    """Return {step: {metric: float}} for every completed step in the log."""
    steps = {}
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if "training/global_step:" not in line:
                continue
            d = {k: float(v) for k, v in _PAT.findall(line)}
            gs = d.get("training/global_step")
            if gs is not None:
                steps[int(gs)] = d
    return steps


def render(steps, total_steps=None, show_all=False):
    if not steps:
        print("no completed steps yet -- the first can take a few minutes (engine init + compile)")
        return

    order = sorted(steps)

    if show_all:
        last = order[-1]
        print(f"\nall opd/* metrics at step {last}:\n")
        for k in sorted(steps[last]):
            if k.startswith("opd/"):
                print(f"  {k:<48}{steps[last][k]:>14.6f}")
        return

    header = f"{'step':>5}" + "".join(f"{h:>10}" for _, h, _ in COLUMNS)
    print("\n" + header)
    print("-" * len(header))
    for s in order:
        row = f"{s:>5}"
        for key, _, fmt in COLUMNS:
            v = steps[s].get(key)
            row += f"{fmt.format(v):>10}" if v is not None else f"{'-':>10}"
        print(row)

    # Timing / ETA from the last few steps, which reflect steady state better than step 1
    # (engine init and torch.compile inflate it).
    times = [steps[s].get("timing_s/step") for s in order if steps[s].get("timing_s/step")]
    if times:
        recent = times[-5:]
        avg = sum(recent) / len(recent)
        done = order[-1]
        print(f"\n  {len(order)} steps logged | mean of last {len(recent)}: {avg / 60:.1f} min/step")
        if total_steps and done < total_steps:
            left = (total_steps - done) * avg
            eta = time.strftime("%a %H:%M", time.localtime(time.time() + left))
            print(f"  {total_steps - done} steps remaining -> ~{left / 3600:.1f} h, finishing around {eta}")

    # Health flags. Each is a condition worth acting on, not just noticing.
    last = steps[order[-1]]
    warn = []
    if last.get("opd/diagnostics_ok", 1.0) < 1.0:
        warn.append("diagnostics FAILED this step (opd/diagnostics_ok=0)")
    if last.get("response_length/clip_ratio", 0) > 0.4:
        warn.append(f"truncation {last['response_length/clip_ratio']:.0%} -- length inflation")
    if last.get("opd/repetition_rate", 0) > 0.05:
        warn.append(f"repetition {last['opd/repetition_rate']:.0%} -- degeneration")
    if last.get("opd/adv_clip_saturation", 0) > 0.02:
        warn.append(f"clip saturation {last['opd/adv_clip_saturation']:.1%} -- clip is shaping the update")
    if last.get("opd/entropy_gap_signed", 0) < -0.5:
        warn.append("student entropy far below teacher -- over-distillation")
    if warn:
        print("\n  WARNINGS")
        for w in warn:
            print(f"    ! {w}")

    # Movement on the metric that says whether OPD is transferring anything.
    if len(order) > 1:
        f, l = steps[order[0]], steps[order[-1]]
        if "opd/overlap_ratio" in f and "opd/overlap_ratio" in l:
            d = l["opd/overlap_ratio"] - f["opd/overlap_ratio"]
            print(
                f"\n  overlap_ratio {f['opd/overlap_ratio']:.4f} -> {l['opd/overlap_ratio']:.4f} "
                f"({d:+.4f}) across steps {order[0]}-{order[-1]}"
            )
            print("  This is the series that says whether the student is aligning to the teacher.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="default: newest logs/opd_*.log")
    ap.add_argument("--follow", action="store_true", help="refresh until interrupted")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--total-steps", type=int, default=100, help="for the ETA")
    ap.add_argument("--all", action="store_true", help="dump every opd/* metric for the last step")
    args = ap.parse_args()

    path = args.log
    if not path:
        hits = sorted(glob.glob("logs/opd_*.log"), key=os.path.getmtime)
        if not hits:
            raise SystemExit("no logs/opd_*.log found -- run from the repo root")
        path = hits[-1]

    while True:
        if args.follow:
            os.system("cls" if os.name == "nt" else "clear")
        print(f"log: {path}")
        render(parse(path), total_steps=args.total_steps, show_all=args.all)
        if not args.follow:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
