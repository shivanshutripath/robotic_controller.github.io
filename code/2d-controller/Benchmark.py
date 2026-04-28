#!/usr/bin/env python3
"""
benchmark.py

Runs a model comparison benchmark:
  - For each MODEL in a configurable list
  - For each repetition k in [0, R)
  - Run loop_agent.py with up to K iterations
  - Store per-run artifacts (controller.py snapshots, metrics, logs)
  - Produce a combined summary CSV + per-model JSON reports

Usage:
  python benchmark.py --project . --models "gpt-5.2,gpt-4.1,gpt-4o" \
      --R 5 --K 20 --edit-retries 3

  Or via run.sh (recommended).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_cmd(cmd: List[str], cwd: str, timeout: Optional[int] = None) -> tuple[int, str, float]:
    """Run a command, return (returncode, combined_output, elapsed_seconds)."""
    t0 = time.perf_counter()
    try:
        p = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - t0
        output = (p.stdout or "") + "\n" + (p.stderr or "")
        return p.returncode, output.strip(), elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        return -1, f"TIMEOUT after {timeout}s", elapsed
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return -2, f"EXCEPTION: {e}", elapsed


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst)


SUMMARY_FIELDS = [
    "model",
    "rep",              # repetition index k (0-based)
    "converged",        # 1 if all tests passed, else 0
    "converged_iter",   # iteration where it passed (0 = never)
    "final_passed",
    "final_failed",
    "final_errors",
    "final_bad",
    "best_bad",
    "total_wall_s",
    "total_gen_s",
    "total_pytest_s",
    "total_llm_s",
    "n_iters_run",
    "timestamp",
]


def write_summary_row(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if not exists:
            w.writeheader()
        safe = {k: row.get(k, "") for k in SUMMARY_FIELDS}
        w.writerow(safe)
        f.flush()


# ──────────────────────────────────────────────
# Single run: one model, one repetition
# ──────────────────────────────────────────────
def run_single(
    project: Path,
    model: str,
    rep: int,
    K: int,
    edit_retries: int,
    optimizer_model: str,
    output_root: Path,
    gen_cmd_base: str,
    timeout_per_run: Optional[int],
) -> Dict[str, Any]:
    """
    Run loop_agent.py for one (model, rep) pair.
    Returns a summary dict.
    """
    run_label = f"{model}__rep{rep}"
    run_dir = output_root / model / f"rep{rep}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Per-run metrics CSV (loop_agent writes to this)
    metrics_csv = run_dir / "metrics.csv"

    # Per-run log
    log_path = run_dir / "run.log"

    # Backup the original template so loop_agent can mutate it and we restore after
    template_src = project / "controller_template.py"
    template_backup = run_dir / "controller_template_original.py"
    safe_copy(template_src, template_backup)

    # Build the loop_agent command
    cmd = [
        sys.executable, str(project / "loop_agent.py"),
        "--project", str(project),
        "--iters", str(K),
        "--edit-retries", str(edit_retries),
        "--model", optimizer_model,
        "--learner-model", model,
        "--run-id", run_label,
        "--metrics-csv", str(metrics_csv),
        "--gen-cmd", gen_cmd_base,
    ]

    print(f"\n{'='*72}")
    print(f"  MODEL={model}  REP={rep}  (K={K}, edit_retries={edit_retries})")
    print(f"  run_dir={run_dir}")
    print(f"{'='*72}\n")

    t0 = time.perf_counter()
    rc, output, elapsed = run_cmd(cmd, cwd=str(project), timeout=timeout_per_run)
    wall_s = time.perf_counter() - t0

    # Save full log
    log_path.write_text(output, encoding="utf-8")

    # Save final controller.py snapshot
    controller_src = project / "controller.py"
    safe_copy(controller_src, run_dir / "controller_final.py")

    # Save final template state
    safe_copy(template_src, run_dir / "controller_template_final.py")

    # Restore original template for next run
    if template_backup.exists():
        shutil.copy2(template_backup, template_src)

    # Parse the per-run metrics CSV to extract summary
    summary = _parse_run_metrics(metrics_csv, model, rep, wall_s, rc)

    # Clean up per-run report.json etc (copy to run_dir)
    for fname in ["report.json", ".last_failures.json"]:
        safe_copy(project / fname, run_dir / fname)

    return summary


def _parse_run_metrics(
    metrics_csv: Path, model: str, rep: int, wall_s: float, rc: int
) -> Dict[str, Any]:
    """Parse the loop_agent metrics CSV to build a summary row."""
    result: Dict[str, Any] = {
        "model": model,
        "rep": rep,
        "converged": 0,
        "converged_iter": 0,
        "final_passed": 0,
        "final_failed": 0,
        "final_errors": 0,
        "final_bad": 999,
        "best_bad": 999,
        "total_wall_s": round(wall_s, 2),
        "total_gen_s": 0.0,
        "total_pytest_s": 0.0,
        "total_llm_s": 0.0,
        "n_iters_run": 0,
        "timestamp": ts(),
    }

    if not metrics_csv.exists():
        return result

    try:
        rows = []
        with metrics_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)

        if not rows:
            return result

        result["n_iters_run"] = len(rows)

        # Accumulate timings
        total_gen = sum(float(r.get("t_gen_s", 0) or 0) for r in rows)
        total_pytest = sum(float(r.get("t_pytest_s", 0) or 0) for r in rows)
        total_llm = sum(float(r.get("t_llm_s", 0) or 0) for r in rows)
        result["total_gen_s"] = round(total_gen, 2)
        result["total_pytest_s"] = round(total_pytest, 2)
        result["total_llm_s"] = round(total_llm, 2)

        # Final row
        last = rows[-1]
        result["final_passed"] = int(last.get("passed", 0) or 0)
        result["final_failed"] = int(last.get("failed", 0) or 0)
        result["final_errors"] = int(last.get("errors", 0) or 0)
        result["final_bad"] = int(last.get("bad", 0) or 0)

        # Best bad across all iters
        best = min(int(r.get("bad", 999) or 999) for r in rows)
        result["best_bad"] = best

        # Check convergence
        for r in rows:
            if int(r.get("bad", 999) or 999) == 0:
                result["converged"] = 1
                result["converged_iter"] = int(r.get("iter", 0) or 0)
                break

    except Exception as e:
        print(f"[benchmark] Warning: failed to parse {metrics_csv}: {e}")

    return result


# ──────────────────────────────────────────────
# Main benchmark driver
# ──────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Model comparison benchmark")
    ap.add_argument("--project", default=".", help="Project root with loop_agent.py, tests, etc.")
    ap.add_argument("--models", required=True,
                    help="Comma-separated list of models to compare (e.g. 'gpt-5.2,gpt-4.1,gpt-4o')")
    ap.add_argument("--R", type=int, default=5, help="Number of repetitions per model (starting from k=0)")
    ap.add_argument("--K", type=int, default=20, help="Max iterations per run (passed to loop_agent --iters)")
    ap.add_argument("--edit-retries", type=int, default=3, help="Edit retries per iteration")
    ap.add_argument("--optimizer-model", default="gpt-4o",
                    help="Model used by loop_agent for AUTO_REPAIR_RULES optimization")
    ap.add_argument("--output-dir", default="benchmark_results",
                    help="Root directory for all benchmark outputs")
    ap.add_argument("--gen-cmd", default=(
        "python code_agent.py --template controller_template.py "
        "--map ./map_agent_outputs/occupancy.png "
        "--params ./map_agent_outputs/params.json "
        "--robot DDR.png --robotpy robot.py --out controller.py "
        "--model gpt-4.1 --max-output-tokens 9000"
    ), help="Base gen-cmd (--model will be overridden per learner)")
    ap.add_argument("--timeout", type=int, default=None,
                    help="Timeout in seconds per single run (model×rep). Default: no timeout.")
    ap.add_argument("--start-rep", type=int, default=0,
                    help="Starting repetition index (default 0). Useful for resuming.")
    args = ap.parse_args()

    project = Path(args.project).resolve()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    R = args.R
    K = args.K
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summary_csv = output_root / "summary.csv"
    all_results: List[Dict[str, Any]] = []

    print(f"╔{'═'*70}╗")
    print(f"║  BENCHMARK: {len(models)} models × {R} reps × {K} max iters")
    print(f"║  Models: {', '.join(models)}")
    print(f"║  Output: {output_root}")
    print(f"║  Start rep: {args.start_rep}")
    print(f"╚{'═'*70}╝")

    total_runs = len(models) * R
    run_idx = 0

    for model in models:
        for rep in range(args.start_rep, args.start_rep + R):
            run_idx += 1
            print(f"\n>>> Run {run_idx}/{total_runs}: model={model} rep={rep}")

            summary = run_single(
                project=project,
                model=model,
                rep=rep,
                K=K,
                edit_retries=args.edit_retries,
                optimizer_model=args.optimizer_model,
                output_root=output_root,
                gen_cmd_base=args.gen_cmd,
                timeout_per_run=args.timeout,
            )

            all_results.append(summary)
            write_summary_row(summary_csv, summary)

            # Print inline summary
            c = "✅" if summary["converged"] else "❌"
            print(f"  {c} model={model} rep={rep} converged_iter={summary['converged_iter']} "
                  f"best_bad={summary['best_bad']} wall={summary['total_wall_s']}s")

    # ── Final aggregate report ──
    print(f"\n\n{'='*72}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*72}")

    report: Dict[str, Any] = {"models": {}, "timestamp": ts(), "config": {
        "R": R, "K": K, "edit_retries": args.edit_retries,
        "optimizer_model": args.optimizer_model,
        "models": models,
    }}

    for model in models:
        model_runs = [r for r in all_results if r["model"] == model]
        n_converged = sum(1 for r in model_runs if r["converged"])
        conv_iters = [r["converged_iter"] for r in model_runs if r["converged"]]
        best_bads = [r["best_bad"] for r in model_runs]
        wall_times = [r["total_wall_s"] for r in model_runs]

        model_report = {
            "n_runs": len(model_runs),
            "n_converged": n_converged,
            "convergence_rate": round(n_converged / max(1, len(model_runs)), 3),
            "avg_converged_iter": round(sum(conv_iters) / max(1, len(conv_iters)), 2) if conv_iters else None,
            "min_converged_iter": min(conv_iters) if conv_iters else None,
            "max_converged_iter": max(conv_iters) if conv_iters else None,
            "avg_best_bad": round(sum(best_bads) / max(1, len(best_bads)), 2),
            "avg_wall_s": round(sum(wall_times) / max(1, len(wall_times)), 2),
            "per_rep": model_runs,
        }
        report["models"][model] = model_report

        print(f"\n  {model}:")
        print(f"    Convergence: {n_converged}/{len(model_runs)} ({model_report['convergence_rate']*100:.0f}%)")
        if conv_iters:
            print(f"    Converged iter: avg={model_report['avg_converged_iter']} "
                  f"min={model_report['min_converged_iter']} max={model_report['max_converged_iter']}")
        print(f"    Avg best_bad: {model_report['avg_best_bad']}")
        print(f"    Avg wall time: {model_report['avg_wall_s']}s")

    # Save JSON report
    report_path = output_root / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n  Full report: {report_path}")
    print(f"  Summary CSV: {summary_csv}")
    print(f"{'='*72}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())