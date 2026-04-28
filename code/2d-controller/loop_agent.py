#!/usr/bin/env python3
"""
loop_agent.py

Closed-loop agent with *edit-first* strategy:

1) Generate controller.py via code_agent.py (mode=generate)
2) Run pytest + read JSON report
3) If failures exist, attempt to FIX by editing controller.py (mode=edit) up to --edit-retries times
4) Only if edits fail repeatedly, COMPLETELY REPLACE AUTO_REPAIR_RULES in controller_template.py
5) Repeat until tests pass or max iterations reached

Notes:
- AUTO_REPAIR_RULES are completely replaced each iteration (no accumulation).
- The main template text contains all foundational rules; AUTO_REPAIR is for learned fixes only.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from openai import OpenAI

BEGIN = "# === AUTO_REPAIR_RULES_BEGIN ==="
END = "# === AUTO_REPAIR_RULES_END ==="

# =========================
# Metrics schema
# =========================
METRIC_FIELDS = [
    "run_id",
    "optimizer_model",
    "learner_model",
    "iter",
    "passed",
    "failed",
    "errors",
    "bad",
    "best_bad",
    "t_gen_s",
    "t_pytest_s",
    "t_parse_s",
    "t_prompt_s",
    "t_llm_s",
    "t_total_s",
    "merged_rules_chars",
    "failed_tests",
]

def _now_s() -> float:
    return time.perf_counter()

def run(cmd: List[str], cwd: str) -> Tuple[int, str]:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return p.returncode, (p.stdout + ("\n" if p.stdout else "") + p.stderr)

def load_report(report_path: Path) -> Dict[str, Any]:
    if not report_path.exists():
        return {}
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def count_bad(report: Dict[str, Any]) -> Tuple[int, int, int]:
    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    failed = int(summary.get("failed", 0))
    errors = int(summary.get("errors", 0))
    return failed + errors, failed, errors

def summarize_failures(report: Dict[str, Any], max_items: int = 12) -> List[Dict[str, str]]:
    tests = ((report.get("tests") or []) if isinstance(report, dict) else [])
    out: List[Dict[str, str]] = []
    for t in tests:
        if (t.get("outcome") or "") == "passed":
            continue
        nodeid = (t.get("nodeid") or "")
        crash = ((t.get("call") or {}).get("crash") or {})
        msg = (crash.get("message") or "")
        longrepr = (t.get("longrepr") or "")
        out.append({
            "nodeid": nodeid,
            "message": msg[:900],
            "longrepr": longrepr[:1400],
        })
        if len(out) >= max_items:
            break
    return out

def replace_autorules(template_text: str, rules_text: str) -> str:
    if BEGIN not in template_text or END not in template_text:
        raise RuntimeError(f"Template missing markers:\n{BEGIN}\n{END}")
    pre = template_text.split(BEGIN)[0]
    post = template_text.split(END)[1]
    block = f"{BEGIN}\n{rules_text.rstrip()}\n{END}\n"
    return pre + block + post

def extract_autorules_block(template_text: str) -> str:
    if BEGIN not in template_text or END not in template_text:
        return ""
    mid = template_text.split(BEGIN, 1)[1]
    block = mid.split(END, 1)[0]
    return block.strip()

def build_rules_prompt(failures: List[Dict[str, str]], current_rules: str, iteration: int) -> str:
    """
    Build a prompt that asks the LLM to generate COMPLETE replacement rules.
    The LLM sees the current rules and failures, and outputs a complete new ruleset.
    """
    return f"""
You are updating the AUTO_REPAIR_RULES section of a robot controller template.

TASK: Generate a COMPLETE, SELF-CONTAINED set of additional rules that will fix the failing tests.
Your output will COMPLETELY REPLACE the current AUTO_REPAIR_RULES section.

IMPORTANT:
- The main template already contains all foundational rules (imports, structure, collision avoidance, etc.)
- AUTO_REPAIR_RULES is for ADDITIONAL learned fixes specific to test failures
- Output ONLY the rule text (no markers, no code fences, no explanations)
- Use imperative bullet points starting with "-"
- Be specific: include concrete values, exact variable names, precise conditions
- Keep it focused: only rules needed to fix the current failures
- Maximum 1500 characters

ITERATION: {iteration}

CURRENT AUTO_REPAIR_RULES (will be completely replaced):
---
{current_rules[:2000]}
---

FAILING TESTS:
{json.dumps(failures, indent=2)[:5500]}

Output the new AUTO_REPAIR_RULES content:
""".strip()


def deduplicate_rules(rules_text: str) -> str:
    """Remove duplicate or near-duplicate rules."""
    lines = rules_text.strip().split('\n')
    seen_prefixes = set()
    unique_lines = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        prefix = stripped[:50].lower()
        if prefix not in seen_prefixes:
            seen_prefixes.add(prefix)
            unique_lines.append(line)
    
    return '\n'.join(unique_lines)


def _cmd_to_list(cmd_str: str) -> List[str]:
    return shlex.split(cmd_str)

def _infer_out_path(cmd_list: List[str]) -> str:
    if "--out" in cmd_list:
        i = cmd_list.index("--out")
        if i + 1 < len(cmd_list):
            return cmd_list[i + 1]
    return "controller.py"

def _set_flag(cmd: List[str], flag: str, value: str) -> List[str]:
    out = list(cmd)
    if flag in out:
        i = out.index(flag)
        if i + 1 < len(out):
            out[i + 1] = value
        else:
            out.append(value)
    else:
        out += [flag, value]
    return out

def _get_flag(cmd: List[str], flag: str) -> Optional[str]:
    if flag in cmd:
        i = cmd.index(flag)
        if i + 1 < len(cmd):
            return cmd[i + 1]
    return None

def _build_edit_cmd(gen_cmd_list: List[str], failures_path: Path, controller_out: str) -> List[str]:
    cmd = list(gen_cmd_list)
    if "--mode" not in cmd:
        cmd += ["--mode", "edit"]
    if "--controller-in" not in cmd:
        cmd += ["--controller-in", controller_out]
    if "--failures-json" not in cmd:
        cmd += ["--failures-json", str(failures_path)]
    return cmd

def append_metrics_row(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        if not exists:
            w.writeheader()
        safe_row = {k: row.get(k, "") for k in METRIC_FIELDS}
        w.writerow(safe_row)
        f.flush()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--model", default="gpt-4o", help="Optimizer model for AUTO_REPAIR_RULES updates")
    ap.add_argument("--edit-retries", type=int, default=3,
                    help="How many edit attempts to try before updating AUTO_REPAIR_RULES")
    ap.add_argument("--gen-cmd", default=(
        "python code_agent.py --template controller_template.py "
        "--map ./map_agent_outputs/occupancy.png "
        "--params ./map_agent_outputs/params.json "
        "--robot DDR.png --robotpy robot.py --out controller.py "
        "--model gpt-4.1 --max-output-tokens 9000"
    ))
    ap.add_argument("--edit-cmd", default="",
                    help="Optional explicit edit command. If empty, derived from --gen-cmd + edit flags.")
    ap.add_argument("--run-id", default="", help="Unique id for this run")
    ap.add_argument("--metrics-csv", default="metrics_all.csv", help="CSV file for metrics")
    ap.add_argument("--learner-model", default="", help="Model for code_agent.py")
    
    args = ap.parse_args()

    root = Path(args.project).resolve()
    template_path = root / "controller_template.py"
    report_path = root / "report.json"
    failures_path = root / ".last_failures.json"

    if not template_path.exists():
        raise FileNotFoundError(f"Missing {template_path}")

    client = OpenAI()

    best_bad = 10**9
    best_template_text = template_path.read_text(encoding="utf-8")
    original_template_text = best_template_text

    gen_cmd_list = _cmd_to_list(args.gen_cmd)

    if args.learner_model.strip():
        gen_cmd_list = _set_flag(gen_cmd_list, "--model", args.learner_model.strip())

    learner_model = _get_flag(gen_cmd_list, "--model") or ""
    controller_out = _infer_out_path(gen_cmd_list)

    if args.edit_cmd.strip():
        edit_cmd_list = _cmd_to_list(args.edit_cmd)
        if args.learner_model.strip():
            edit_cmd_list = _set_flag(edit_cmd_list, "--model", args.learner_model.strip())
    else:
        edit_cmd_list = _build_edit_cmd(gen_cmd_list, failures_path, controller_out)

    metrics_csv = (root / args.metrics_csv) if not Path(args.metrics_csv).is_absolute() else Path(args.metrics_csv)
    run_id = args.run_id.strip() or f"{learner_model or 'learner'}__opt_{args.model}"

    for it in range(1, args.iters + 1):
        print("=" * 70)
        print(f"[loop] Iteration {it}/{args.iters}")
        iter_t0 = _now_s()

        t_gen_s = 0.0
        t_pytest_s = 0.0
        t_parse_s = 0.0
        t_prompt_s = 0.0
        t_llm_s = 0.0

        failed_tests_str = ""
        merged_rules_chars = 0

        # 1) Generate fresh controller
        t0 = _now_s()
        rc_gen, out = run(gen_cmd_list, cwd=str(root))
        t_gen_s += (_now_s() - t0)
        print(out)

        # 2) Run pytest
        t0 = _now_s()
        rc_py, out = run(["pytest", "-q", "--json-report", f"--json-report-file={report_path.name}"], cwd=str(root))
        t_pytest_s += (_now_s() - t0)
        print(out)

        # Parse report
        t0 = _now_s()
        report = load_report(report_path)
        bad, failed, errors = count_bad(report)
        passed = int((report.get("summary", {}) or {}).get("passed", 0)) if report else 0
        failures = summarize_failures(report)
        t_parse_s += (_now_s() - t0)

        failed_nodeids = [f.get("nodeid", "") for f in failures if f.get("nodeid")]
        failed_tests_str = ";".join(failed_nodeids)

        if rc_gen != 0 and not failed_tests_str:
            failed_tests_str = "GENERATOR_FAILED"
            bad = 1_000_000_000
            best_bad = min(best_bad, bad)

        print(f"[loop] pytest passed={passed} failed={failed} errors={errors} bad={bad}")

        if bad == 0:
            tpl = template_path.read_text(encoding="utf-8")
            merged_rules_chars = len(extract_autorules_block(tpl))
            t_total_s = _now_s() - iter_t0
            append_metrics_row(metrics_csv, {
                "run_id": run_id,
                "optimizer_model": args.model,
                "learner_model": learner_model,
                "iter": it,
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "bad": bad,
                "best_bad": min(best_bad, bad),
                "t_gen_s": round(t_gen_s, 6),
                "t_pytest_s": round(t_pytest_s, 6),
                "t_parse_s": round(t_parse_s, 6),
                "t_prompt_s": round(t_prompt_s, 6),
                "t_llm_s": round(t_llm_s, 6),
                "t_total_s": round(t_total_s, 6),
                "merged_rules_chars": merged_rules_chars,
                "failed_tests": failed_tests_str,
            })
            print("[loop] All tests passed ✅")
            return 0

        # 3) Edit-first attempts
        failures_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")

        for e in range(1, args.edit_retries + 1):
            print("-" * 70)
            print(f"[loop] Edit attempt {e}/{args.edit_retries} (no prompt update)")

            t0 = _now_s()
            rc_edit, out = run(edit_cmd_list, cwd=str(root))
            t_gen_s += (_now_s() - t0)
            print(out)

            t0 = _now_s()
            rc_py, out = run(["pytest", "-q", "--json-report", f"--json-report-file={report_path.name}"], cwd=str(root))
            t_pytest_s += (_now_s() - t0)
            print(out)

            t0 = _now_s()
            report = load_report(report_path)
            bad, failed, errors = count_bad(report)
            passed = int((report.get("summary", {}) or {}).get("passed", 0)) if report else 0
            failures = summarize_failures(report)
            t_parse_s += (_now_s() - t0)

            failed_nodeids = [f.get("nodeid", "") for f in failures if f.get("nodeid")]
            failed_tests_str = ";".join(failed_nodeids)

            print(f"[loop] pytest passed={passed} failed={failed} errors={errors} bad={bad}")

            if bad == 0:
                tpl = template_path.read_text(encoding="utf-8")
                merged_rules_chars = len(extract_autorules_block(tpl))
                t_total_s = _now_s() - iter_t0
                append_metrics_row(metrics_csv, {
                    "run_id": run_id,
                    "optimizer_model": args.model,
                    "learner_model": learner_model,
                    "iter": it,
                    "passed": passed,
                    "failed": failed,
                    "errors": errors,
                    "bad": bad,
                    "best_bad": min(best_bad, bad),
                    "t_gen_s": round(t_gen_s, 6),
                    "t_pytest_s": round(t_pytest_s, 6),
                    "t_parse_s": round(t_parse_s, 6),
                    "t_prompt_s": round(t_prompt_s, 6),
                    "t_llm_s": round(t_llm_s, 6),
                    "t_total_s": round(t_total_s, 6),
                    "merged_rules_chars": merged_rules_chars,
                    "failed_tests": failed_tests_str,
                })
                print("[loop] All tests passed ✅")
                return 0

            failures_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")

        # 4) Update AUTO_REPAIR_RULES - complete replacement
        print("-" * 70)
        print("[loop] Edits failed → updating AUTO_REPAIR_RULES (complete replacement)")

        current_template_text = template_path.read_text(encoding="utf-8")
        current_rules = extract_autorules_block(current_template_text)
        
        if bad < best_bad:
            best_bad = bad
            best_template_text = current_template_text
            print(f"[loop] New best template saved (best_bad={best_bad}).")

        # Build prompt
        t0 = _now_s()
        prompt = build_rules_prompt(failures, current_rules, it)
        t_prompt_s += (_now_s() - t0)

        # LLM generates complete new rules
        t0 = _now_s()
        resp = client.responses.create(
            model=args.model,
            input=prompt,
            max_output_tokens=1200,
        )
        t_llm_s += (_now_s() - t0)

        model_rules = (getattr(resp, "output_text", "") or "").strip()
        
        # Clean up output
        if model_rules.startswith("```"):
            lines = model_rules.split('\n')
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            model_rules = '\n'.join(lines).strip()
        
        # Deduplicate
        model_rules = deduplicate_rules(model_rules)

        # Use model rules directly - no BASE_AUTORULES appending
        # (all foundational rules are in the main template text)
        final_rules = model_rules
        
        # Apply to original template (fresh each time, no accumulation)
        new_template = replace_autorules(original_template_text, final_rules)

        backup = template_path.with_suffix(f".py.bak{it}")
        backup.write_text(current_template_text, encoding="utf-8")
        template_path.write_text(new_template, encoding="utf-8")
        print(f"[loop] Updated AUTO_REPAIR_RULES (backup: {backup.name})")
        print(f"[loop] Rules length: {len(final_rules)} chars")

        merged_rules_chars = len(final_rules)

        t_total_s = _now_s() - iter_t0
        append_metrics_row(metrics_csv, {
            "run_id": run_id,
            "optimizer_model": args.model,
            "learner_model": learner_model,
            "iter": it,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "bad": bad,
            "best_bad": best_bad,
            "t_gen_s": round(t_gen_s, 6),
            "t_pytest_s": round(t_pytest_s, 6),
            "t_parse_s": round(t_parse_s, 6),
            "t_prompt_s": round(t_prompt_s, 6),
            "t_llm_s": round(t_llm_s, 6),
            "t_total_s": round(t_total_s, 6),
            "merged_rules_chars": merged_rules_chars,
            "failed_tests": failed_tests_str,
        })

    print(f"[loop] Reached max iterations; best_bad={best_bad}.")
    return 1

if __name__ == "__main__":
    raise SystemExit(main())